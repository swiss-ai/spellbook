"""
SlurmBackend — renders Jinja2 bash templates and submits via sbatch (or srun).

The template receives the experiment's fully resolved dict, so all logic
(conditionals, derived values) lives in Python, not in env-var bash hacks.

Usage::

    backend = SlurmBackend(
        account="my-account",
        partition="gpu",
        nodes=32,
        gpus_per_node=4,
        run_time="04:00:00",
        extra={"container_edf": "/path/to/container.toml"},
    )

    sweep.render(output_dir="sbatch_scripts")   # write scripts + experiments.csv
    sweep.submit(output_dir="sbatch_scripts")   # render + sbatch

Reservation::

    backend = SlurmBackend(..., reservation="my-reservation")

Running inside an existing allocation (srun mode)::

    backend = SlurmBackend(..., srun_job_id="12345678")
    # Uses srun.sh.j2 template; rendered script is a plain bash script to run
    # inside the allocation with: bash <script>.sh

Launching one Python process per Slurm task instead of torchrun::

    backend = SlurmBackend(..., launch_mode="tasks")
    # Uses slurm_tasks.sh.j2 with ntasks-per-node=gpus_per_node.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from spellbook.backends.slurm_megatron.create_data_config import create_data_prefix
from spellbook.core.experiment import Experiment

_TEMPLATES_DIR = Path(__file__).parent  # slurm_megatron/


def _is_megatron_url(value: str) -> bool:
    """Return whether a Megatron source is a supported Git URL."""
    parsed = urlparse(value)
    return (
        parsed.scheme in {"git", "http", "https", "ssh"}
        and bool(parsed.netloc)
    ) or (value.startswith("git@") and ":" in value)


# Variables always forwarded to srun workers regardless of experiment env_vars.
_SRUN_INFRA_EXPORTS = [
    "LOCAL_RANK",
    "RANK",
    "GPUS_PER_NODE",
    "HOSTNAMES",
    "MASTER_ADDR",
    "MASTER_PORT",
    "WORLD_SIZE",
    "SLURM_JOB_ID",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "CUDA_CACHE_DISABLE",
    "WANDB_API_KEY",
    "WANDB_PROJECT",
    "WANDB_MODE",
    "WANDB_RESUME",
    "WANDB_RUN_ID",
    "HF_TOKEN",
    "HF_HUB_ENABLE_HF_TRANSFER",
    "PYTHONPATH",
    "MEGATRON_PATH",
    "MEGATRON_GIT_BRANCH",
    "MEGATRON_GIT_COMMIT",
    "MEGATRON_GIT_URL",
    "TRITON_HOME",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "SLURM_NETWORK",
]


@dataclass
class SlurmBackend:
    account: str
    partition: str
    gpus_per_node: int
    run_time: str                          # "HH:MM:SS"
    nodes: int | None = None                     # if None, derived from experiment.num_gpus // gpus_per_node
    log_dir: str = "slurm_logs"
    reservation: str = ""                  # adds --reservation to sbatch header + sbatch cmd
    dependency_singleton: bool = True      # adds --dependency=singleton to sbatch header
    no_save: bool = False                  # render and submit without writing the .sh file to disk
    srun_job_id: str = ""                  # when set, use srun.sh.j2 + run inside allocation
    mem_estimator: bool = False            # when True, use mem_estimator.sh.j2 (1 GPU, fake process group)
    launch_mode: str = "torchrun"          # "torchrun" or "tasks"; tasks runs python directly per Slurm task
    auto_requeue: bool = False             # submit next job before srun (sbatch --dependency=singleton $0)
    srun_extra_args: str = ""              # extra flags appended verbatim to every srun call
    env_vars: dict[str, Any] = field(default_factory=dict)  # infrastructure env vars exported by backend
    pythonpath_env_vars: list[str] = field(default_factory=list)  # env var names whose values are prepended to PYTHONPATH
    extra: dict[str, Any] = field(default_factory=dict)

    def _git_metadata(self, path: str) -> dict[str, str]:
        if not path:
            return {"status": "empty"}
        p = Path(path).expanduser()
        if not p.exists() or not p.is_dir():
            return {"status": "missing", "path": str(p)}
        if subprocess.run(
            ["git", "-C", str(p), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
        ).returncode != 0:
            return {"status": "not-git", "path": str(p)}

        def _run(args: list[str]) -> str:
            result = subprocess.run(
                ["git", "-C", str(p), *args],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return "unknown"
            return result.stdout.strip() or "unknown"

        return {
            "status": "ok",
            "path": str(p),
            "branch": _run(["rev-parse", "--abbrev-ref", "HEAD"]),
            "commit": _run(["rev-parse", "HEAD"]),
            "remote": _run(["remote", "get-url", "origin"]),
        }

    def _resolve_path_env_value(self, name: str, d: dict[str, Any]) -> str:
        exp_env = d.get("env_vars") or {}
        if name in exp_env and exp_env[name]:
            return str(exp_env[name])
        return ""

    @staticmethod
    def _shell_quote_value(value: str) -> str:
        """Wrap value in double quotes if it contains shell-special characters."""
        if any(c in value for c in ("|", "(", ")", "*", "&", ";", "<", ">", "`", "!")):
            return f'"{value}"'
        return value

    def _format_training_args_lines(self, training_args: list[Any]) -> list[str]:
        """Keep each --flag and its value on the same rendered line."""
        lines: list[str] = []
        i = 0
        while i < len(training_args):
            token = str(training_args[i])
            if token.startswith("--"):
                if i + 1 < len(training_args):
                    next_token = str(training_args[i + 1])
                    if not next_token.startswith("--"):
                        lines.append(f"{token} {self._shell_quote_value(next_token)}")
                        i += 2
                        continue
                lines.append(token)
                i += 1
                continue
            lines.append(token)
            i += 1
        return lines

    def _inject_git_metadata_env_vars(self, d: dict[str, Any]) -> None:
        env_vars = dict(d.get("env_vars") or {})

        targets: list[tuple[str, str]] = []
        megatron_path = str(d.get("megatron_path") or "")
        if megatron_path:
            targets.append(("MEGATRON_PATH", megatron_path))

        for var in self.pythonpath_env_vars:
            value = self._resolve_path_env_value(var, d)
            if value:
                targets.append((var, value))

        seen: set[tuple[str, str]] = set()
        for label, path in targets:
            key = (label, path)
            if key in seen:
                continue
            seen.add(key)

            base = label[:-5] if label.endswith("_PATH") else label
            meta = self._git_metadata(path)
            env_vars.setdefault(f"{base}_GIT_BRANCH", meta.get("branch", "unknown"))
            env_vars.setdefault(f"{base}_GIT_COMMIT", meta.get("commit", "unknown"))
            env_vars.setdefault(f"{base}_GIT_URL", meta.get("remote", "unknown"))

        d["env_vars"] = env_vars

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _template_name(self) -> str:
        if self.launch_mode not in {"torchrun", "tasks"}:
            raise ValueError(
                f"Unsupported SlurmBackend.launch_mode={self.launch_mode!r}; "
                "expected 'torchrun' or 'tasks'."
            )
        if self.mem_estimator:
            return "mem_estimator.sh.j2"
        if self.srun_job_id:
            return "srun.sh.j2"
        if self.launch_mode == "tasks":
            return "slurm_tasks.sh.j2"
        return "slurm.sh.j2"

    def render(
        self,
        experiment: Experiment,
        generated_data_args_path: str | Path | None = None,
    ) -> str:
        """Render the Jinja2 template for a single experiment. Returns the script string."""
        env = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_DIR)),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
        )
        tmpl = env.get_template(self._template_name())

        d = experiment.to_dict()
        container_path = str(d.get("megatron_container_path") or "").rstrip("/")
        if (
            not container_path.startswith("/")
            or container_path == ""
            or ".." in container_path.split("/")
            or re.fullmatch(r"/[A-Za-z0-9_./-]+", container_path) is None
            or len([part for part in container_path.split("/") if part]) < 2
        ):
            raise ValueError(
                "MegatronExperiment.megatron_container_path must be a shell-safe absolute path with at least two components"
            )
        d["megatron_container_path"] = container_path
        megatron_source = str(d.get("megatron_path") or "")
        if _is_megatron_url(megatron_source):
            d["megatron_url"] = megatron_source
            d["megatron_path"] = ""
            d["megatron_cache_key"] = hashlib.sha256(
                megatron_source.encode()
            ).hexdigest()[:16]
        else:
            d["megatron_url"] = ""
            d["megatron_cache_key"] = ""
        d["megatron_worktree_key"] = hashlib.sha256(
            f"{megatron_source}\0{d.get('megatron_commit') or ''}".encode()
        ).hexdigest()[:16]
        # Explicit data_path wins over data_args_path, which wins over discovery.
        if d.get("data_path") and d.get("data_args_path"):
            d["training_args"] = self._remove_training_arg(
                d.get("training_args") or [], "--data-args-path"
            )
            d["data_args_path"] = ""
        elif (
            not d.get("data_path")
            and not d.get("data_args_path")
            and d.get("base_data_path")
        ):
            paths = [
                path.strip()
                for path in d["base_data_path"].split(",")
                if path.strip()
            ]
            prefixes = create_data_prefix(
                paths, follow_symlinks=bool(d.get("follow_symlinks"))
            )
            if not prefixes:
                warnings.warn(
                    f"[{experiment.name}] base_data_path '{d['base_data_path']}' "
                    "resolved to zero dataset shards — no data path will be configured.",
                    stacklevel=2,
                )
            elif generated_data_args_path is not None:
                manifest_path = Path(generated_data_args_path).resolve()
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text("\n".join(prefixes) + "\n")
                d["data_args_path"] = str(manifest_path)
                d["training_args"] = [
                    *(d.get("training_args") or []),
                    "--data-args-path",
                    str(manifest_path),
                ]
            else:
                # Direct render() calls have no output directory for a companion
                # manifest, so preserve the legacy inline behavior.
                d["data_path"] = " ".join(prefixes)
        elif not d.get("data_path") and not d.get("data_args_path"):
            warnings.warn(
                f"[{experiment.name}] Neither data_path, data_args_path, nor "
                "base_data_path is set — no data path will be configured.",
                stacklevel=2,
            )
        d["training_args_lines"] = self._format_training_args_lines(
            d.get("training_args") or []
        )
        # Backend env vars are infrastructure defaults; experiment env_vars override them.
        d["env_vars"] = {**self.env_vars, **(d.get("env_vars") or {})}
        d["env_vars"].setdefault("MEGATRON_PATH", d.get("megatron_path") or "")
        # Derive nodes from experiment if not explicitly set on the backend
        nodes = self.nodes
        if nodes is None:
            num_gpus = d.get("num_gpus")
            if num_gpus is None:
                raise ValueError(
                    f"Cannot derive nodes for '{experiment.name}': "
                    "set SlurmBackend.nodes or experiment.num_gpus."
                )
            nodes = num_gpus // self.gpus_per_node
        # Resolve mem_estimator_path: default to the bundled copy in slurm_megatron/memory_estimator
        mem_estimator_path = d.get("mem_estimator_path") or str(
            _TEMPLATES_DIR / "memory_estimator"
        )
        env_vars: dict = d.get("env_vars") or {}
        pythonpath_parts = [
            str(env_vars[var])
            for var in self.pythonpath_env_vars
            if env_vars.get(var)
        ]
        srun_export_vars = ",".join(
            dict.fromkeys(_SRUN_INFRA_EXPORTS + list(env_vars.keys()))
        )
        srun_extra_arg_env_vars = {}
        srun_extra_tokens = self.srun_extra_args.split()
        for idx, token in enumerate(srun_extra_tokens):
            if token.startswith("--network="):
                srun_extra_arg_env_vars["SLURM_NETWORK"] = token.split("=", 1)[1]
                break
            if token == "--network" and idx + 1 < len(srun_extra_tokens):
                srun_extra_arg_env_vars["SLURM_NETWORK"] = srun_extra_tokens[idx + 1]
                break

        ctx = {
            **d,
            **self.extra,
            "account": self.account,
            "partition": self.partition,
            "nodes": nodes,
            "gpus_per_node": self.gpus_per_node,
            "total_tasks": nodes * self.gpus_per_node,
            "run_time": self.run_time,
            "log_dir": self.log_dir,
            "job_name": experiment.name,
            "exp_name": experiment.name,
            "wandb_exp_name": d.get("wandb_exp_name", experiment.name),
            "reservation": self.reservation,
            "dependency_singleton": self.dependency_singleton,
            "auto_requeue": self.auto_requeue,
            "srun_job_id": self.srun_job_id,
            "srun_extra_args": self.srun_extra_args,
            "srun_extra_arg_env_vars": srun_extra_arg_env_vars,
            "pythonpath_env_vars": self.pythonpath_env_vars,
            "pythonpath": ":".join(pythonpath_parts),
            "srun_export_vars": srun_export_vars,
            "mem_estimator_path": mem_estimator_path,
        }
        return tmpl.render(ctx)

    @staticmethod
    def _remove_training_arg(args: list[str], flag: str) -> list[str]:
        """Remove a flag and its following value from a flat CLI argument list."""
        try:
            index = args.index(flag)
        except ValueError:
            return args
        return args[:index] + args[index + 2 :]

    def render_all(
        self,
        sweep_name: str,
        experiments: list[Experiment],
        output_dir: str = "sbatch_scripts",
    ) -> list[str]:
        """Render and write all experiments to <output_dir>/<sweep_name>/<exp.name>.sh.

        Also writes experiments.csv alongside the scripts.
        """
        out = Path(output_dir) / sweep_name
        if not self.no_save:
            out.mkdir(parents=True, exist_ok=True)

        paths = []
        rows = []
        for exp in experiments:
            exp_dict = exp.to_dict()
            generated_data_args_path = None
            if (
                not self.no_save
                and exp_dict.get("base_data_path")
                and not exp_dict.get("data_path")
                and not exp_dict.get("data_args_path")
            ):
                generated_data_args_path = out / f"{exp.name}.data_args.txt"
            script = self.render(
                exp, generated_data_args_path=generated_data_args_path
            )
            path = out / f"{exp.name}.sh"
            if self.no_save:
                print(f"# --- {path} (no_save, not written) ---")
                print(script)
            else:
                path.write_text(script)
            paths.append(str(path))
            rows.append(exp.to_dict())

        if not self.no_save:
            # Write CSV — drop training_args list (not useful in tabular form)
            df = pd.DataFrame(rows)
            if "training_args" in df.columns:
                df = df.drop(columns=["training_args"])
            df.to_csv(out / "experiments.csv", index=False)

        return paths

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit_all(
        self,
        sweep_name: str,
        experiments: list[Experiment],
        output_dir: str = "sbatch_scripts",
    ) -> list[str]:
        """Render all experiments, then submit each via sbatch or srun. Returns job IDs."""
        paths = self.render_all(sweep_name, experiments, output_dir=output_dir)
        job_ids = []
        for exp, path in zip(experiments, paths):
            if self.no_save:
                script = self.render(exp)
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".sh", delete=False
                ) as tmp:
                    tmp.write(script)
                    tmp_path = tmp.name
                try:
                    if self.srun_job_id:
                        job_id = self._run_srun_script(tmp_path)
                    else:
                        job_id = self._sbatch(tmp_path)
                finally:
                    os.unlink(tmp_path)
            else:
                if self.srun_job_id:
                    job_id = self._run_srun_script(path)
                else:
                    job_id = self._sbatch(path)
            print(f"  {Path(path).stem}: submitted → job {job_id}")
            job_ids.append(job_id)
        return job_ids

    def _sbatch(self, script_path: str) -> str:
        cmd = ["sbatch"]
        if self.reservation:
            cmd += [f"--reservation={self.reservation}"]
        cmd.append(script_path)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"sbatch failed (exit {result.returncode}):\n{result.stderr.strip()}"
            )
        return result.stdout.strip().split()[-1]

    def _run_srun_script(self, script_path: str) -> str:
        """Execute the rendered srun bash script directly (blocks until done)."""
        subprocess.run(["bash", script_path], check=True)
        return self.srun_job_id
