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
import hashlib
import os
import subprocess
import tempfile
from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from evals import flags as lm_eval_flags

_TEMPLATES_DIR = Path(__file__).parent
_LM_EVAL_ARG = {"lm_eval_arg": True}


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
    megatron_path: str               # local path or Git URL for the Megatron-LM repo
    megatron_commit: str = ""        # if set, a git worktree is pinned to this commit

    # --- Eval ---
    tasks: list[str] = dataclasses.field(default_factory=list, metadata=_LM_EVAL_ARG)
    batch_size: int = dataclasses.field(default=16, metadata=_LM_EVAL_ARG)
    cache_requests: str = dataclasses.field(default="", metadata=_LM_EVAL_ARG)
    devices: int = 4                 # total GPUs passed to lm_eval (--devices)
    tp: int = 1
    ep: int = 1
    seq_length: int = 4096           # lm-eval adapter context limit
    micro_batch_size: int = 1        # per-rank batch passed to the lm-eval adapter
    metadata: dict[str, object] = dataclasses.field(default_factory=dict, metadata=_LM_EVAL_ARG)
    extra_args: str = ""             # extra flags appended verbatim to lm_eval model_args
    output_dir: str | None = None     # defaults to <submission directory>/evals
    log_samples: bool = dataclasses.field(default=False, metadata=_LM_EVAL_ARG)
    write_out: bool = dataclasses.field(default=False, metadata=_LM_EVAL_ARG)  # print prompts

    # --- Slurm ---
    account: str = ""
    partition: str = ""
    nodes: int = 1
    gpus_per_node: int = 4
    launch_mode: str = "torchrun"     # "torchrun" or "tasks"
    srun_extra_args: str = ""         # extra flags appended verbatim to srun
    run_time: str = "03:00:00"
    reservation: str = ""
    exclude: str = ""                 # Slurm node list passed to --exclude
    log_dir: str = "slurm_logs/eval"

    # --- Container ---
    container_edf: str = ""          # e.g. "apertus2-alps4-temp"
    container_mounts: str = ""       # e.g. "${SCRATCH}:${SCRATCH},${HOME}:${HOME}"

    # --- WandB ---
    wandb_project: str = ""
    wandb_id: str = ""                # run ID; defaults to model_name if empty

    # --- Cache / storage ---
    hf_home: str = ""                # HF_HOME and HF_DATASETS_CACHE; skipped if empty

    # --- lm-eval install (once per node) ---
    lm_eval_install: str = ""        # pip install URL/path; skipped if empty
    # Use the eval interpreter via python -m pip instead of bare pip.
    lm_eval_install_with_python: bool = False
    install_commands: str = ""       # raw shell commands run before evaluation; skipped if empty

    # --- Dataset prefetch ---
    # Mapping from lm-eval task name to load_dataset() call args, e.g.:
    #   {"hellaswag": ["hellaswag"], "arc_easy": ["ai2_arc", "ARC-Easy"]}
    # Rank 0 will prefetch all listed datasets before the eval loop.
    dataset_prefetch: dict[str, list[str]] = dataclasses.field(default_factory=dict)

    # --- Extra env vars ---
    # Exported inside the srun shell, after the hardcoded defaults.
    env_vars: dict[str, str] = dataclasses.field(default_factory=dict)


def _marked_lm_eval_args(cfg: MegatronEvalConfig) -> dict[str, Any]:
    return {
        field.name: getattr(cfg, field.name)
        for field in dataclasses.fields(cfg)
        if field.metadata.get("lm_eval_arg")
    }


def _model_args(cfg: MegatronEvalConfig, ckpt_step: int) -> str:
    return lm_eval_flags.model_args(
        {
            "load": f"{cfg.checkpoint_dir}/{cfg.model_name}",
            "tokenizer_type": "HuggingFaceTokenizer",
            "tokenizer_model": cfg.tokenizer_model,
            "ckpt_step": ckpt_step,
            "transformer_impl": "transformer_engine",
            "devices": cfg.devices,
            "TP": cfg.tp,
            "EP": cfg.ep,
            "seq_length": cfg.seq_length,
            "micro_batch_size": cfg.micro_batch_size,
            "extra_args": cfg.extra_args,
        }
    )


def _lm_eval_args(
    cfg: MegatronEvalConfig,
    ckpt_step: int,
    output_path: Path,
) -> Mapping[str, Any]:
    return {
        "model": "megatron_lm",
        "model_args": _model_args(cfg, ckpt_step),
        **_marked_lm_eval_args(cfg),
        "output_path": str(output_path),
    }


def _is_megatron_url(value: str) -> bool:
    parsed = urlparse(value)
    return (
        parsed.scheme in {"git", "http", "https", "ssh"}
        and bool(parsed.netloc)
    ) or (value.startswith("git@") and ":" in value)


def _render(cfg: MegatronEvalConfig, ckpt_step: int, dependency_singleton: bool) -> str:
    if cfg.launch_mode not in {"torchrun", "tasks"}:
        raise ValueError(
            f"Unsupported MegatronEvalConfig.launch_mode={cfg.launch_mode!r}; "
            "expected 'torchrun' or 'tasks'."
        )

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    tmpl = env.get_template("megatron_eval.sh.j2")
    ctx = dataclasses.asdict(cfg)
    if _is_megatron_url(cfg.megatron_path):
        ctx["megatron_url"] = cfg.megatron_path
        ctx["megatron_path"] = ""
        ctx["megatron_cache_key"] = hashlib.sha256(
            cfg.megatron_path.encode()
        ).hexdigest()[:16]
    else:
        ctx["megatron_url"] = ""
        ctx["megatron_cache_key"] = ""
    ctx["ckpt_step"] = ckpt_step
    ctx["date"] = datetime.now().strftime("%Y-%m-%d")
    ctx["dependency_singleton"] = dependency_singleton
    ctx["ntasks_per_node"] = cfg.gpus_per_node if cfg.launch_mode == "tasks" else 1
    ctx["total_tasks"] = cfg.nodes * ctx["ntasks_per_node"]
    output_dir = Path(cfg.output_dir).expanduser() if cfg.output_dir else Path.cwd() / "evals"
    output_path = output_dir.resolve() / cfg.model_name / f"step_{ckpt_step}"
    ctx["output_dir"] = str(output_dir.resolve())
    ctx["lm_eval_args_lines"] = lm_eval_flags.to_shell_lines(
        _lm_eval_args(cfg, ckpt_step, output_path)
    )
    wandb_args = {
        "wandb_args": (
            f"project={cfg.wandb_project},"
            f"id={cfg.wandb_id or cfg.model_name},"
            f"step={ckpt_step},resume=allow"
        )
        if cfg.wandb_project
        else None
    }
    ctx["wandb_args_line"] = next(iter(lm_eval_flags.to_shell_lines(wandb_args)), "")
    return tmpl.render(ctx)


def _sbatch(script: str, reservation: str, exclude: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script)
        tmp_path = f.name
    try:
        cmd = ["sbatch"]
        if exclude:
            cmd += [f"--exclude={exclude}"]
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
    job_id = _sbatch(script, cfg.reservation, cfg.exclude)
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
