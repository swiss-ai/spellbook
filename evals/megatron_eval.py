"""
Megatron-LM evaluation runner using lm-evaluation-harness.

Usage — single checkpoint:
    from evals.megatron_eval import MegatronEvalConfig, submit
    cfg = MegatronEvalConfig(...)
    submit(cfg, ckpt_step=3000)

Usage — range of checkpoints:
    from evals.megatron_eval import MegatronEvalConfig, RangeMode, submit_range
    submit_range(cfg, start=500, end=3500, step=500, mode=RangeMode.PARALLEL)
    submit_range(cfg, start=500, end=3500, step=500, mode=RangeMode.SEQUENTIAL)
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from datetime import datetime

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_TEMPLATES_DIR = Path(__file__).parent


class RangeMode(Enum):
    PARALLEL   = "parallel"    # one job per ckpt, all submitted at once
    SEQUENTIAL = "sequential"  # one job per ckpt, dependency=singleton


@dataclasses.dataclass
class MegatronEvalConfig:
    # --- Model ---
    model_name: str
    checkpoint_dir: str              # base dir; ckpt_step is appended at submit time
    tokenizer_model: str

    # --- Megatron ---
    megatron_path: str               # path to Megatron-LM repo
    megatron_commit: str = ""        # if set, a git worktree is pinned to this commit

    # --- Eval ---
    tasks: list[str] = dataclasses.field(default_factory=list)
    batch_size: int = 16
    devices: int = 4                 # total GPUs passed to lm_eval (--devices)
    ep: int = 1
    extra_args: str = ""             # extra flags appended verbatim to lm_eval model_args

    # --- Slurm ---
    account: str = ""
    partition: str = ""
    nodes: int = 1
    gpus_per_node: int = 4
    run_time: str = "03:00:00"
    reservation: str = ""
    log_dir: str = "slurm_logs/eval"

    # --- Container ---
    container_edf: str = ""          # e.g. "apertus2-alps4-temp"
    container_mounts: str = ""       # e.g. "${SCRATCH}:${SCRATCH},${HOME}:${HOME}"

    # --- WandB ---
    wandb_project: str = ""
    wandb_id: str = ""                # run ID; defaults to model_name if empty

    # --- Cache / storage ---
    hf_home: str = ""                # HF_HOME and HF_DATASETS_CACHE; skipped if empty

    # --- lm-eval install ---
    lm_eval_install: str = ""        # pip install URL/path; skipped if empty

    # --- Dataset prefetch ---
    # Mapping from lm-eval task name to load_dataset() call args, e.g.:
    #   {"hellaswag": ["hellaswag"], "arc_easy": ["ai2_arc", "ARC-Easy"]}
    # Rank 0 will prefetch all listed datasets before the eval loop.
    dataset_prefetch: dict[str, list[str]] = dataclasses.field(default_factory=dict)


def _render(cfg: MegatronEvalConfig, ckpt_step: int, dependency_singleton: bool) -> str:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    tmpl = env.get_template("megatron_eval.sh.j2")
    ctx = dataclasses.asdict(cfg)
    ctx["ckpt_step"] = ckpt_step
    ctx["date"] = datetime.now().strftime("%Y-%m-%d")
    ctx["dependency_singleton"] = dependency_singleton
    ctx["tasks_str"] = ",".join(cfg.tasks)
    return tmpl.render(ctx)


def _sbatch(script: str, reservation: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script)
        tmp_path = f.name
    try:
        cmd = ["sbatch"]
        if reservation:
            cmd += [f"--reservation={reservation}"]
        cmd.append(tmp_path)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip().split()[-1]
    finally:
        os.unlink(tmp_path)


def render_watcher_script(
    cfg: MegatronEvalConfig,
    config_path: str,
    interval_hours: float = 1,
    project_dir: Path | None = None,
) -> Path:
    """Render evals/<model>/watcher.sh — the self-scheduling sbatch watcher."""
    if project_dir is None:
        project_dir = Path(__file__).resolve().parent.parent
    output_dir = project_dir / "evals" / cfg.model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / "watcher.sh"

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    tmpl = env.get_template("watcher.sh.j2")
    ctx = {
        "model_name": cfg.model_name,
        "account": cfg.account,
        "partition": cfg.partition,
        "log_dir": cfg.log_dir,
        "reservation": cfg.reservation,
        "checkpoint_dir": cfg.checkpoint_dir,
        "config_path": str(Path(config_path).resolve()),
        "project_dir": str(project_dir),
        "watcher_script_path": str(script_path),
        "interval_minutes": round(interval_hours * 60),
    }
    script_path.write_text(tmpl.render(ctx))
    script_path.chmod(0o755)
    return script_path


def submit(cfg: MegatronEvalConfig, ckpt_step: int, dependency_singleton: bool = False) -> str:
    """Render and submit a single eval job. Returns the Slurm job ID."""
    script = _render(cfg, ckpt_step, dependency_singleton)
    job_id = _sbatch(script, cfg.reservation)
    print(f"  {cfg.model_name} step={ckpt_step}: submitted → job {job_id}")
    return job_id


def submit_range(
    cfg: MegatronEvalConfig,
    start: int,
    end: int,
    step: int,
    mode: RangeMode = RangeMode.PARALLEL,
) -> list[str]:
    """Submit eval jobs for checkpoints in [start, end] (inclusive) with the given step.

    PARALLEL: all jobs submitted at once.
    SEQUENTIAL: each job has dependency=singleton so they queue behind each other.
    """
    job_ids = []
    for ckpt_step in range(start, end + 1, step):
        job_id = submit(
            cfg,
            ckpt_step,
            dependency_singleton=(mode == RangeMode.SEQUENTIAL),
        )
        job_ids.append(job_id)
    return job_ids
