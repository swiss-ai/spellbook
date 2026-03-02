"""
Sweep: SLURM job array over a list of config variants.

Can be used for benchmark throughput sweeps or hyperparameter searches.
Each variant is a flat dict of config overrides. The sweep generates:

    sbatch_scripts/<name>/
    ├── array.sh        # #SBATCH --array=0-N%max_concurrent
    └── configs.json    # list of fully resolved configs, indexed by task ID

Each entry in configs.json has a ``variant_name`` built automatically from
``name_prefix`` and ``name_keys``, e.g. prefix="G-", keys=["tp","ep"] →
"G-tp2-ep4". The name is injected into MODEL_ARGS as ``wandb_exp_name`` and
``tensorboard_dir`` suffix when those template variables are used.

array.sh reads $SLURM_ARRAY_TASK_ID, pulls the correct arg list from configs.json,
and passes it to torch.distributed.run. All script generation goes through SlurmBackend.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

from zoo.backends.megatron import MegatronBackend
from zoo.backends.slurm import SlurmBackend
from zoo.config import build_config, load_yaml, merge_configs, _PRESETS_DIR
from zoo.module import Context, JobSpec, Module, ModuleResult, ModuleStatus


class Sweep(Module):
    """
    Parallel job array sweep over a list of config variants.

    Parameters
    ----------
    name : str
        Identifier used for job name, script path, and configs.json.
    cluster : str
        Name of a cluster preset (zoo/presets/clusters/<name>.yaml).
    model : str
        Name of a model preset (zoo/presets/models/<name>.yaml).
    runtime : str | None
        Name of a runtime preset (zoo/presets/runtimes/<name>.yaml).
    nodes : int
        Nodes per array task.
    run_time : str
        Wall-clock limit "HH:MM:SS" per task.
    variants : list[dict]
        Each dict is a flat set of config overrides for one array task.
        Keys can be any alias accepted by MegatronBackend (tp, ep, mbs, …)
        or raw --flag-name strings.
    name_keys : list[str] | None
        Keys to include in the auto-generated variant name, in order.
        e.g. ["tp", "ep"] → "tp2-ep4". If None, all variant keys are used.
    name_prefix : str
        Prefix for every variant name, e.g. "G-", "BA-", "TP-".
        Defaults to "".
    max_concurrent : int | None
        Max simultaneously running tasks (#SBATCH --array=0-N%max_concurrent).
        None means no limit.
    scripts_dir : Path | None
        Where to write array.sh and configs.json.
    depends_on : list[str] | None
        Module names that must complete before this sweep starts.
    **base_kwargs
        Base config overrides shared by all variants.
    """

    def __init__(
        self,
        name: str,
        cluster: str,
        model: str,
        nodes: int,
        run_time: str,
        variants: list[dict[str, Any]],
        runtime: str | None = None,
        name_keys: list[str] | None = None,
        name_prefix: str = "",
        max_concurrent: int | None = None,
        scripts_dir: Path | None = None,
        depends_on: list[str] | None = None,
        **base_kwargs: Any,
    ):
        super().__init__(name=name, depends_on=depends_on)
        self.cluster = cluster
        self.model = model
        self.runtime = runtime
        self.nodes = nodes
        self.run_time = run_time
        self.variants = variants
        self.name_keys = name_keys
        self.name_prefix = name_prefix
        self.max_concurrent = max_concurrent
        self.base_kwargs = base_kwargs
        self.scripts_dir = scripts_dir or Path("sbatch_scripts") / name

    # ------------------------------------------------------------------
    # Module interface
    # ------------------------------------------------------------------

    def run(self, ctx: Context) -> ModuleResult:
        """
        Resolve all variant configs, write configs.json + array.sh, submit.

        Returns ModuleResult with data keys:
            job_id      : str (SLURM array job ID, or DRY_<name>)
            array_script: path to array.sh
            configs_file: path to configs.json
            n_variants  : int
        """
        self.scripts_dir.mkdir(parents=True, exist_ok=True)

        cluster_cfg = _load_cluster_preset(self.cluster)

        # Resolve every variant into a list of Megatron arg strings
        resolved_variants: list[dict] = []
        for i, variant in enumerate(self.variants):
            variant_name = _build_variant_name(variant, self.name_prefix, self.name_keys)
            # Inject variant_name so templates like ${variant_name} resolve in MODEL_ARGS
            merged_kwargs = merge_configs(cluster_cfg, self.base_kwargs, variant, {"variant_name": variant_name})  # type: ignore[arg-type]
            cfg = build_config(model=self.model, runtime=self.runtime, **merged_kwargs)
            training_args = MegatronBackend.to_args(cfg.get("MODEL_ARGS", {}))
            resolved_variants.append({
                "task_id": i,
                "variant_name": variant_name,
                "variant": variant,
                "training_args": training_args,
            })

        # Write configs.json
        configs_path = self.scripts_dir / "configs.json"
        configs_path.write_text(json.dumps(resolved_variants, indent=2))

        # Build JobSpec for the array script.
        # training_args is left empty — the script reads it from configs.json at runtime.
        # training_args_shell_expr injects the dynamic lookup into the srun launch line.
        n = len(self.variants)
        array_spec = f"0-{n - 1}"
        if self.max_concurrent is not None:
            array_spec += f"%{self.max_concurrent}"

        # Use first variant's resolved config for headers / paths only.
        # strict=False so that per-variant templates like ${variant_name} don't
        # blow up here — training_args come from resolved_variants, not first_cfg.
        first_cfg = build_config(
            model=self.model,
            runtime=self.runtime,
            strict=False,
            **merge_configs(cluster_cfg, self.base_kwargs, self.variants[0]),  # type: ignore[arg-type]
        )

        # Shell snippet that expands to the training args for this array task.
        # Written as a variable assignment before srun so it stays readable.
        configs_path_str = str(configs_path.resolve())
        training_args_shell_expr = (
            f'$(python3 -c "'
            f'import json,sys; '
            f'cfg=json.load(open(\\"{configs_path_str}\\")); '
            f'args=cfg[int(\\"$SLURM_ARRAY_TASK_ID\\")][\\"training_args\\"]; '
            f'print(\\" \\".join(str(a) for a in args))'
            f'")'
        )

        job = JobSpec(
            name=self.name,
            account=first_cfg.get("account", cluster_cfg.get("account", "")),
            partition=first_cfg.get("partition", cluster_cfg.get("partition", "")),
            nodes=self.nodes,
            gpus_per_node=first_cfg.get("gpus_per_node", cluster_cfg.get("gpus_per_node", 4)),
            run_time=self.run_time,
            megatron_path=first_cfg.get("megatron_path", ""),
            megatron_patch_path=first_cfg.get("megatron_patch_path", ""),
            training_script=first_cfg.get("training_script", "pretrain_gpt.py"),
            training_args=[],
            training_args_shell_expr=training_args_shell_expr,
            container_edf=first_cfg.get("container_edf", ""),
            container_mounts=first_cfg.get("container_mounts", cluster_cfg.get("container_mounts", "")),
            env_vars=first_cfg.get("ENV_VARS", {}),
            log_dir=Path(first_cfg.get("log_dir", "slurm_logs")),
            array_spec=array_spec,
        )

        array_script_path = self.scripts_dir / "array.sh"
        SlurmBackend.write_script(job, array_script_path)

        print(f"\nSweep: {self.name}  ({n} variants)")
        print(f"  array.sh   → {array_script_path}")
        print(f"  configs    → {configs_path}")
        _print_variant_table(resolved_variants, self.name_prefix, self.name_keys)

        if ctx.dry_run:
            job_id = f"DRY_{self.name.upper()}"
            print(f"  dry-run — would submit {array_script_path}")
        else:
            answer = input("Submit? [y/N] ").strip().lower()
            if answer != "y":
                print("Aborted.")
                return ModuleResult(
                    module_name=self.name,
                    status=ModuleStatus.SKIPPED,
                    error="User aborted submission",
                )
            job_id = SlurmBackend.submit(job, array_script_path)
            print(f"  submitted → job array {job_id}")

        return ModuleResult(
            module_name=self.name,
            status=ModuleStatus.DONE,
            data={
                "job_id": job_id,
                "array_script": str(array_script_path),
                "configs_file": str(configs_path),
                "n_variants": n,
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_cluster_preset(name: str) -> dict:
    path = _PRESETS_DIR / "clusters" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Cluster preset not found: {path}")
    return load_yaml(path)


def _build_variant_name(variant: dict[str, Any], prefix: str, keys: list[str] | None) -> str:
    """Build a short human-readable name for a variant.

    If the variant dict contains a ``variant_name`` key, it is used as-is
    (prefix and name_keys are ignored).

    e.g. prefix="G-", keys=["tp","ep"], variant={"tp":2,"ep":4} → "G-tp2-ep4"
    If keys is None, all keys in the variant are used in their insertion order.
    """
    if "variant_name" in variant:
        return variant["variant_name"]
    effective_keys = keys if keys is not None else [k for k in variant if k != "variant_name"]
    parts = [f"{k}{variant[k]}" for k in effective_keys if k in variant]
    return prefix + "-".join(parts)


def variant_grid(**axes: list[Any]) -> list[dict[str, Any]]:
    """
    Expand a grid of axis values into a flat list of variant dicts.

    Example::

        variant_grid(tp=[1, 2, 4], pp=[1, 2], ep=[4])
        # → [
        #     {"tp": 1, "pp": 1, "ep": 4},
        #     {"tp": 1, "pp": 2, "ep": 4},
        #     {"tp": 2, "pp": 1, "ep": 4},
        #     {"tp": 2, "pp": 2, "ep": 4},
        #     {"tp": 4, "pp": 1, "ep": 4},
        #     {"tp": 4, "pp": 2, "ep": 4},
        # ]
    """
    keys = list(axes.keys())
    values = list(axes.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _print_variant_table(
    resolved: list[dict],
    prefix: str,
    name_keys: list[str] | None,
) -> None:
    print(f"\n  {'#':>3}  {'name':<30}  overrides")
    print("  " + "-" * 70)
    for entry in resolved:
        i = entry["task_id"]
        vname = entry["variant_name"]
        overrides = ", ".join(f"{k}={v}" for k, v in entry["variant"].items())
        print(f"  {i:>3}  {vname:<30}  {overrides}")
    print()
