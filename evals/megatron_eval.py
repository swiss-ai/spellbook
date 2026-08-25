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
import re
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
_RUN_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


class RangeMode(Enum):
    PARALLEL   = "parallel"    # one job per ckpt, all submitted at once
    SEQUENTIAL = "sequential"  # one job per ckpt, dependency=singleton


class AllocationMode(Enum):
    SHARED = "shared"
    SEPARATE = "separate"


@dataclasses.dataclass
class LMEvalRunConfig:
    """One lm-eval invocation inside a shared Megatron evaluation job."""

    name: str
    tasks: list[str]
    lm_eval_args: dict[str, Any] = dataclasses.field(default_factory=dict)
    env_vars: dict[str, str] = dataclasses.field(default_factory=dict)
    wandb_name: str = ""
    wandb_id: str = ""


@dataclasses.dataclass
class MegatronEvalConfig:
    # --- Model ---
    model_name: str
    checkpoint_dir: str              # base dir; ckpt_step is appended at submit time
    tokenizer_model: str

    # --- Megatron ---
    megatron_path: str               # local path or Git URL for the Megatron-LM repo
    megatron_commit: str = ""        # if set, a git worktree is pinned to this commit
    megatron_container_path: str = "/opt/megatron"

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
    model_args_extra: dict[str, Any] = dataclasses.field(default_factory=dict)
    output_dir: str | None = None     # defaults to <submission directory>/evals
    log_samples: bool = dataclasses.field(default=False, metadata=_LM_EVAL_ARG)
    write_out: bool = dataclasses.field(default=False, metadata=_LM_EVAL_ARG)  # print prompts
    eval_runs: list[LMEvalRunConfig] = dataclasses.field(
        default_factory=list, kw_only=True
    )

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
    watcher_dir: str = ""             # generated watcher scripts; defaults to <project>/evals

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
    prefetch_timeout_seconds: int = 1800

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
            **cfg.model_args_extra,
        }
    )


def _lm_eval_args(
    cfg: MegatronEvalConfig,
    ckpt_step: int,
    output_path: Path,
    run: LMEvalRunConfig | None = None,
) -> Mapping[str, Any]:
    return {
        **_marked_lm_eval_args(cfg),
        **(run.lm_eval_args if run is not None else {}),
        "model": "megatron_lm",
        "model_args": _model_args(cfg, ckpt_step),
        "tasks": run.tasks if run is not None else cfg.tasks,
        "output_path": str(output_path),
    }


def _validate_eval_runs(cfg: MegatronEvalConfig) -> None:
    if not cfg.eval_runs:
        if not cfg.tasks:
            raise ValueError("tasks must not be empty when eval_runs is empty")
        return
    names = [run.name for run in cfg.eval_runs]
    if len(names) != len(set(names)):
        raise ValueError("eval run names must be unique")
    for run in cfg.eval_runs:
        if not _RUN_NAME.fullmatch(run.name):
            raise ValueError(
                "eval run names may contain only letters, numbers, dots, underscores, and hyphens"
            )
        if not run.tasks:
            raise ValueError(f"eval run {run.name!r} tasks must not be empty")


def _is_megatron_url(value: str) -> bool:
    parsed = urlparse(value)
    return (
        parsed.scheme in {"git", "http", "https", "ssh"}
        and bool(parsed.netloc)
    ) or (value.startswith("git@") and ":" in value)


def _checkpoint_load_path(cfg: MegatronEvalConfig, reporting_step: int) -> Path:
    """Resolve the checkpoint directory Megatron will actually load."""
    model_root = Path(cfg.checkpoint_dir) / cfg.model_name
    load_step = cfg.model_args_extra.get("ckpt_step", reporting_step)
    if load_step:
        return model_root / f"iter_{int(load_step):07d}"

    tracker = model_root / "latest_checkpointed_iteration.txt"
    if tracker.is_file():
        tracked = tracker.read_text().strip()
        if tracked == "release":
            return model_root / "release"
        return model_root / f"iter_{int(tracked):07d}"

    return model_root / f"iter_{reporting_step:07d}"


def _render_checkpoints(
    cfg: MegatronEvalConfig,
    checkpoints: tuple[tuple[int, int | None], ...],
    dependency_singleton: bool,
) -> str:
    if cfg.launch_mode not in {"torchrun", "tasks"}:
        raise ValueError(
            f"Unsupported MegatronEvalConfig.launch_mode={cfg.launch_mode!r}; "
            "expected 'torchrun' or 'tasks'."
        )
    container_path = cfg.megatron_container_path.rstrip("/")
    if (
        not container_path.startswith("/")
        or container_path == ""
        or ".." in container_path.split("/")
        or re.fullmatch(r"/[A-Za-z0-9_./-]+", container_path) is None
        or len([part for part in container_path.split("/") if part]) < 2
    ):
        raise ValueError(
            "MegatronEvalConfig.megatron_container_path must be a shell-safe absolute path with at least two components"
        )
    if cfg.prefetch_timeout_seconds <= 0:
        raise ValueError("prefetch_timeout_seconds must be greater than zero")
    _validate_eval_runs(cfg)

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    tmpl = env.get_template("megatron_eval.sh.j2")
    ctx = dataclasses.asdict(cfg)
    ctx["megatron_container_path"] = container_path
    if _is_megatron_url(cfg.megatron_path):
        ctx["megatron_url"] = cfg.megatron_path
        ctx["megatron_path"] = ""
        ctx["megatron_cache_key"] = hashlib.sha256(
            cfg.megatron_path.encode()
        ).hexdigest()[:16]
    else:
        ctx["megatron_url"] = ""
        ctx["megatron_cache_key"] = ""
    ctx["megatron_worktree_key"] = hashlib.sha256(
        f"{cfg.megatron_path}\0{cfg.megatron_commit}".encode()
    ).hexdigest()[:16]
    if not checkpoints:
        raise ValueError("checkpoints must not be empty")
    if len({step for step, _ in checkpoints}) != len(checkpoints):
        raise ValueError("checkpoint steps must be unique")
    for step, consumed_tokens in checkpoints:
        if step <= 0:
            raise ValueError("checkpoint steps must be greater than zero")
        if consumed_tokens is not None and consumed_tokens <= 0:
            raise ValueError("consumed_tokens must be greater than zero")
    checkpoint_label = (
        str(checkpoints[0][0])
        if len(checkpoints) == 1
        else f"{checkpoints[0][0]}-{checkpoints[-1][0]}"
    )
    ctx["ckpt_step"] = checkpoint_label

    ctx["date"] = datetime.now().strftime("%Y-%m-%d")
    ctx["dependency_singleton"] = dependency_singleton
    ctx["ntasks_per_node"] = cfg.gpus_per_node if cfg.launch_mode == "tasks" else 1
    ctx["total_tasks"] = cfg.nodes * ctx["ntasks_per_node"]
    output_dir = Path(cfg.output_dir).expanduser() if cfg.output_dir else Path.cwd() / "evals"
    model_output_dir = output_dir.resolve() / cfg.model_name
    ctx["suite_output_dir"] = str(
        model_output_dir / f"step_{checkpoints[0][0]}"
        if len(checkpoints) == 1
        else model_output_dir
    )
    ctx["eval_runs"] = []
    runs: tuple[LMEvalRunConfig | None, ...] = (
        tuple(cfg.eval_runs) if cfg.eval_runs else (None,)
    )
    for ckpt_step, consumed_tokens in checkpoints:
        checkpoint_output = model_output_dir / f"step_{ckpt_step}"
        for run in runs:
            run_output_path = (
                checkpoint_output / run.name if run is not None else checkpoint_output
            )
            base_id = cfg.wandb_id or cfg.model_name
            run_id = (
                base_id
                if run is None
                else run.wandb_id or f"{base_id}-{run.name}"
            )
            wandb_parts = [
                f"project={cfg.wandb_project}",
                f"id={run_id}",
                f"step={ckpt_step}",
            ]
            if run is not None:
                wandb_parts.append(f"name={run.wandb_name or run.name}")
            if consumed_tokens is not None:
                wandb_parts.append(f"consumed_tokens={consumed_tokens}")
            wandb_parts.append("resume=allow")
            wandb_args = {
                "wandb_args": ",".join(wandb_parts) if cfg.wandb_project else None
            }
            ctx["eval_runs"].append(
                {
                    "name": run.name if run is not None else "default",
                    "ckpt_step": ckpt_step,
                    "output_path": str(run_output_path),
                    "checkpoint_path": str(
                        _checkpoint_load_path(cfg, ckpt_step)
                    ),
                    "env_vars": run.env_vars if run is not None else {},
                    "lm_eval_args_lines": lm_eval_flags.to_shell_lines(
                        _lm_eval_args(cfg, ckpt_step, run_output_path, run)
                    ),
                    "wandb_args_line": next(
                        iter(lm_eval_flags.to_shell_lines(wandb_args)), ""
                    ),
                }
            )
    return tmpl.render(ctx)


def _render(
    cfg: MegatronEvalConfig,
    ckpt_step: int,
    dependency_singleton: bool,
    consumed_tokens: int | None = None,
) -> str:
    return _render_checkpoints(
        cfg, ((ckpt_step, consumed_tokens),), dependency_singleton
    )


def _clean_sbatch_env() -> dict[str, str]:
    """Submit jobs without leaking the caller's Python environment."""
    env = os.environ.copy()
    venv = env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    if venv and "PATH" in env:
        venv_bin = str(Path(venv) / "bin")
        env["PATH"] = os.pathsep.join(
            path for path in env["PATH"].split(os.pathsep) if path != venv_bin
        )
    return env


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
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            env=_clean_sbatch_env(),
        )
        return result.stdout.strip().split()[-1]
    finally:
        os.unlink(tmp_path)


def render_watcher_script(
    cfg: MegatronEvalConfig,
    config_path: str,
    interval_hours: float = 1,
    project_dir: Path | None = None,
    watch_checkpoint_dir: str | None = None,
    config_model: str | None = None,
    config_group: str | None = None,
    eval_job_name: str | None = None,
    consumed_tokens_per_step: int | None = None,
    dependency_singleton: bool = False,
    watch_state_dir: str | None = None,
) -> Path:
    """Render evals/<model>/watcher.sh — the self-scheduling sbatch watcher."""
    if interval_hours <= 0:
        raise ValueError("interval_hours must be greater than zero")
    if consumed_tokens_per_step is not None and consumed_tokens_per_step <= 0:
        raise ValueError("consumed_tokens_per_step must be greater than zero")
    if project_dir is None:
        project_dir = Path(__file__).resolve().parent.parent
    watcher_name = (
        cfg.model_name
        if config_group is None
        else f"{cfg.model_name}-{config_group}"
    )
    output_root = (
        Path(cfg.watcher_dir).expanduser()
        if cfg.watcher_dir
        else project_dir / "evals"
    )
    output_dir = output_root / watcher_name
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / "watcher.sh"
    log_dir = Path(cfg.log_dir).expanduser()
    if not log_dir.is_absolute():
        log_dir = project_dir / log_dir
    (log_dir / watcher_name).mkdir(parents=True, exist_ok=True)

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    tmpl = env.get_template("watcher.sh.j2")
    checkpoint_dir = (
        Path(watch_checkpoint_dir).expanduser()
        if watch_checkpoint_dir
        else Path(cfg.checkpoint_dir).expanduser() / cfg.model_name
    )
    state_dir = (
        Path(watch_state_dir).expanduser()
        if watch_state_dir
        else project_dir / "evals" / "state_files"
    )
    stop_file = state_dir / watcher_name / ".watcher_stop"
    ctx = {
        "model_name": cfg.model_name,
        "watcher_name": watcher_name,
        "account": cfg.account,
        "partition": cfg.partition,
        "log_dir": str(log_dir.resolve()),
        "reservation": cfg.reservation,
        "watch_checkpoint_dir": str(checkpoint_dir),
        "stop_file": str(stop_file),
        "state_file": str(state_dir / watcher_name / ".submitted_steps"),
        "config_path": str(Path(config_path).resolve()),
        "config_model": config_model,
        "config_group": config_group,
        "eval_job_name": eval_job_name,
        "consumed_tokens_per_step": consumed_tokens_per_step,
        "dependency_singleton": dependency_singleton,
        "project_dir": str(project_dir),
        "watcher_script_path": str(script_path),
        "interval_minutes": round(interval_hours * 60),
    }
    script_path.write_text(tmpl.render(ctx))
    script_path.chmod(0o755)
    return script_path


def _submit_checkpoints(
    cfg: MegatronEvalConfig,
    checkpoints: tuple[tuple[int, int | None], ...],
    dependency_singleton: bool,
) -> str:
    (Path(cfg.log_dir).expanduser() / cfg.model_name).mkdir(
        parents=True, exist_ok=True
    )
    job_id = _sbatch(
        _render_checkpoints(cfg, checkpoints, dependency_singleton),
        cfg.reservation,
        cfg.exclude,
    )
    steps = ",".join(str(step) for step, _ in checkpoints)
    label = f"step={steps}" if len(checkpoints) == 1 else f"steps={steps}"
    print(f"  {cfg.model_name} {label}: submitted → job {job_id}")
    return job_id


def submit(
    cfg: MegatronEvalConfig,
    ckpt_step: int,
    dependency_singleton: bool = False,
    consumed_tokens: int | None = None,
) -> str:
    """Submit one checkpoint, optionally using consumed tokens as the W&B x-axis."""
    return _submit_checkpoints(
        cfg, ((ckpt_step, consumed_tokens),), dependency_singleton
    )


def submit_evaluations(
    cfg: MegatronEvalConfig,
    ckpt_steps: list[int] | tuple[int, ...],
    checkpoint_mode: AllocationMode = AllocationMode.SEPARATE,
    eval_mode: AllocationMode = AllocationMode.SHARED,
    dependency_singleton: bool = False,
    consumed_tokens: Mapping[int, int] | None = None,
) -> list[str]:
    """Submit the checkpoint/eval-run matrix with independent job grouping."""
    steps = tuple(ckpt_steps)
    if not steps:
        raise ValueError("ckpt_steps must not be empty")
    if len(set(steps)) != len(steps):
        raise ValueError("ckpt_steps must be unique")
    checkpoint_groups = (
        (steps,)
        if checkpoint_mode == AllocationMode.SHARED
        else tuple((step,) for step in steps)
    )
    config_groups = (
        [cfg]
        if eval_mode == AllocationMode.SHARED or not cfg.eval_runs
        else [dataclasses.replace(cfg, eval_runs=[run]) for run in cfg.eval_runs]
    )
    tokens = consumed_tokens or {}
    job_ids: list[str] = []
    for checkpoint_group in checkpoint_groups:
        checkpoints = tuple((step, tokens.get(step)) for step in checkpoint_group)
        for grouped_cfg in config_groups:
            job_ids.append(
                _submit_checkpoints(
                    grouped_cfg, checkpoints, dependency_singleton
                )
            )
    return job_ids


def submit_range(
    cfg: MegatronEvalConfig,
    start: int,
    end: int,
    step: int,
    mode: RangeMode = RangeMode.PARALLEL,
    global_batch_size: int | None = None,
    seq_length: int | None = None,
    checkpoint_mode: AllocationMode = AllocationMode.SEPARATE,
    eval_mode: AllocationMode = AllocationMode.SHARED,
) -> list[str]:
    """Submit eval jobs for checkpoints in [start, end] (inclusive) with the given step.

    PARALLEL: all jobs submitted at once.
    SEQUENTIAL: each job has dependency=singleton so they queue behind each other.
    """
    if step <= 0:
        raise ValueError("step must be greater than zero")
    ckpt_steps = tuple(range(start, end + 1, step))
    consumed_tokens = (
        {
            ckpt_step: global_batch_size * ckpt_step * seq_length
            for ckpt_step in ckpt_steps
        }
        if global_batch_size is not None and seq_length is not None
        else None
    )
    return submit_evaluations(
        cfg,
        ckpt_steps,
        checkpoint_mode=checkpoint_mode,
        eval_mode=eval_mode,
        dependency_singleton=(mode == RangeMode.SEQUENTIAL),
        consumed_tokens=consumed_tokens,
    )
