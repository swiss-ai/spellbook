"""
ExperimentChain: sequential training runs with automatic checkpoint passing.

Each Step is a delta of config overrides on top of the chain's base config.
The chain resolves the full config per step, builds a JobSpec, renders an
sbatch script, and submits with --dependency=afterok:<prev_job_id>.

Submission modes
----------------
dependency (default)
    All scripts written upfront, submitted with Slurm dependency chaining.
    Survives disconnects. Slurm cancels downstream steps if one fails.

dry_run
    Scripts are rendered and printed/written but sbatch is never called.
    Activated via ctx.dry_run=True (set by Project.run(dry_run=True)).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from zoo.backends.megatron import MegatronBackend
from zoo.backends.slurm import SlurmBackend
from zoo.config import build_config, load_yaml, merge_configs, _PRESETS_DIR
from zoo.module import Context, JobSpec, Module, ModuleResult, ModuleStatus


# ---------------------------------------------------------------------------
# Step — a single run within a chain
# ---------------------------------------------------------------------------

@dataclass
class Step:
    """
    A single training run inside an ExperimentChain.

    Only carries overrides relative to the chain's base config.
    The chain merges: cluster → model → chain base kwargs → step kwargs.

    Special fields
    --------------
    branch_from : str | None
        Name of another Step in the same chain to branch from instead of
        using the immediately preceding step's checkpoint.
    nodes : int | None
        Override the chain-level node count for this step only.
    run_time : str | None
        Override "HH:MM:SS" wall-clock limit for this step only.
    """
    name: str
    branch_from: str | None = None
    nodes: int | None = None
    run_time: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __init__(self, name: str, branch_from: str | None = None,
                 nodes: int | None = None, run_time: str | None = None,
                 **kwargs: Any):
        self.name = name
        self.branch_from = branch_from
        self.nodes = nodes
        self.run_time = run_time
        self.kwargs = kwargs


# ---------------------------------------------------------------------------
# ExperimentChain
# ---------------------------------------------------------------------------

class ExperimentChain(Module):
    """
    Sequential training run: warmup → main → cooldown etc.

    Checkpoint path is passed automatically between consecutive steps via
    --load / --save Megatron flags. Steps may branch from any earlier step.

    Parameters
    ----------
    name : str
        Identifier used for job names, script paths, and state file.
    cluster : str
        Name of a cluster preset (zoo/presets/clusters/<name>.yaml).
    model : str
        Name of a model preset (zoo/presets/models/<name>.yaml).
    runtime : str | None
        Name of a runtime preset (zoo/presets/runtimes/<name>.yaml).
        Supplies container_edf, megatron_path, megatron_patch_path.
    nodes : int
        Number of nodes (shared by all steps unless overridden per-step).
    run_time : str
        Default wall-clock limit "HH:MM:SS" for all steps.
    scripts_dir : Path
        Where sbatch scripts are written (default: sbatch_scripts/<name>/).
    state_file : Path
        Where chain_state.json is written (default: runs/<name>/chain_state.json).
    steps : list[Step]
        Ordered list of Step objects defining the run sequence.
    depends_on : list[str]
        Module names that must complete before this chain starts.
    **base_kwargs
        Base config overrides shared by all steps (tp, ep, mbs, data_path, …).
    """

    def __init__(
        self,
        name: str,
        cluster: str,
        model: str,
        nodes: int,
        run_time: str,
        steps: list[Step],
        runtime: str | None = None,
        scripts_dir: Path | None = None,
        state_file: Path | None = None,
        depends_on: list[str] | None = None,
        **base_kwargs: Any,
    ):
        super().__init__(name=name, depends_on=depends_on)
        self.cluster = cluster
        self.model = model
        self.runtime = runtime
        self.nodes = nodes
        self.run_time = run_time
        self.steps = steps
        self.base_kwargs = base_kwargs
        self.scripts_dir = scripts_dir or Path("sbatch_scripts") / name
        self.state_file = state_file or Path("runs") / name / "chain_state.json"

    # ------------------------------------------------------------------
    # Module interface
    # ------------------------------------------------------------------

    def run(self, ctx: Context) -> ModuleResult:
        """
        Resolve configs, render scripts, submit (or dry-run) all steps.

        Returns ModuleResult with data keys:
            job_ids     : dict[step_name, job_id_str]
            scripts     : dict[step_name, script_path_str]
            state_file  : path to chain_state.json
        """
        self.scripts_dir.mkdir(parents=True, exist_ok=True)

        # Load cluster preset once
        cluster_cfg = _load_cluster_preset(self.cluster)

        # Per-step tracking
        job_ids: dict[str, str] = {}          # step_name → slurm job id
        checkpoints: dict[str, str] = {}      # step_name → checkpoint save path
        scripts: dict[str, str] = {}          # step_name → script path
        state_entries: list[dict] = []

        # For branching: map step name → the step object
        step_map: dict[str, Step] = {s.name: s for s in self.steps}

        prev_step_name: str | None = None

        print(f"\nChain: {self.name}  ({len(self.steps)} steps)\n")
        _print_step_table(self.steps, self.base_kwargs)

        if not ctx.dry_run:
            answer = input("Submit? [y/N] ").strip().lower()
            if answer != "y":
                print("Aborted.")
                return ModuleResult(
                    module_name=self.name,
                    status=ModuleStatus.SKIPPED,
                    error="User aborted submission",
                )

        for step in self.steps:
            # --- Resolve config for this step ---
            step_cfg = build_config(
                model=self.model,
                runtime=self.runtime,
                **merge_configs(cluster_cfg, self.base_kwargs, step.kwargs),  # type: ignore[arg-type]
            )

            # Determine checkpoint_load from previous step (or branch_from)
            source_step = step.branch_from or prev_step_name
            if source_step and source_step in checkpoints:
                step_cfg.setdefault("MODEL_ARGS", {})
                # Only inject --load if not already set in the step config
                step_cfg["MODEL_ARGS"].setdefault("--load", checkpoints[source_step])

            # Build Megatron args
            training_args = MegatronBackend.to_args(step_cfg.get("MODEL_ARGS", {}))

            # Determine dependency
            dependency: str | None = None
            if source_step and source_step in job_ids:
                dependency = f"afterok:{job_ids[source_step]}"

            # Build JobSpec
            nodes = step.nodes or self.nodes
            run_time = step.run_time or self.run_time
            job_name = f"{self.name}-{step.name}"

            job = JobSpec(
                name=job_name,
                account=step_cfg.get("account", cluster_cfg.get("account", "")),
                partition=step_cfg.get("partition", cluster_cfg.get("partition", "")),
                nodes=nodes,
                gpus_per_node=step_cfg.get("gpus_per_node", cluster_cfg.get("gpus_per_node", 4)),
                run_time=run_time,
                megatron_path=step_cfg.get("megatron_path", ""),
                megatron_patch_path=step_cfg.get("megatron_patch_path", ""),
                training_script=step_cfg.get("training_script", "pretrain_gpt.py"),
                training_args=training_args,
                container_edf=step_cfg.get("container_edf", ""),
                container_mounts=step_cfg.get("container_mounts", cluster_cfg.get("container_mounts", "")),
                env_vars=step_cfg.get("ENV_VARS", {}),
                log_dir=Path(step_cfg.get("log_dir", "slurm_logs")),
                dependency=dependency,
            )

            # Write script
            script_path = self.scripts_dir / f"{step.name}.sh"
            SlurmBackend.write_script(job, script_path)
            scripts[step.name] = str(script_path)
            print(f"  [{step.name}] script → {script_path}")

            # Submit or dry-run
            if ctx.dry_run:
                job_id = f"DRY_{step.name.upper()}"
                print(f"  [{step.name}] dry-run — would submit {script_path}")
            else:
                job_id = SlurmBackend.submit(job, script_path)
                print(f"  [{step.name}] submitted → job {job_id}")

            job_ids[step.name] = job_id

            # Track checkpoint save path for the next step
            save_path = (
                step_cfg.get("MODEL_ARGS", {}).get("--save")
                or step_cfg.get("MODEL_ARGS", {}).get("checkpoint_save")
                or step_cfg.get("checkpoint_save")
            )
            if save_path:
                checkpoints[step.name] = save_path

            state_entries.append({
                "name": step.name,
                "job_id": job_id,
                "status": "PENDING",
                "script": str(script_path),
                "checkpoint": save_path or "",
                "dependency": dependency,
            })

            prev_step_name = step.name

        # Write chain state file
        self._write_state(state_entries)
        print(f"\nState file → {self.state_file}")

        return ModuleResult(
            module_name=self.name,
            status=ModuleStatus.DONE,
            data={
                "job_ids": job_ids,
                "scripts": scripts,
                "state_file": str(self.state_file),
            },
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_state(self, entries: list[dict]) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "chain": self.name,
            "cluster": self.cluster,
            "model": self.model,
            "steps": entries,
        }
        self.state_file.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_cluster_preset(name: str) -> dict:
    path = _PRESETS_DIR / "clusters" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Cluster preset not found: {path}")
    return load_yaml(path)


def _print_step_table(steps: list[Step], base: dict[str, Any]) -> None:
    """Print a summary table of steps showing key overrides."""
    header = f"  {'Step':<20} {'nodes':>5} {'run_time':<10} overrides"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for step in steps:
        overrides = ", ".join(f"{k}={v}" for k, v in step.kwargs.items()) or "(none)"
        nodes_str = str(step.nodes) if step.nodes is not None else "—"
        rt_str = step.run_time or "—"
        branch = f"  [branch: {step.branch_from}]" if step.branch_from else ""
        print(f"  {step.name:<20} {nodes_str:>5} {rt_str:<10} {overrides}{branch}")
    print()
