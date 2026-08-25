"""Render and submit distributed Megatron checkpoint merges."""

from __future__ import annotations

import dataclasses
import hashlib
import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_TEMPLATES_DIR = Path(__file__).parent
_BACKEND_TEMPLATES_DIR = Path(__file__).parents[2] / "spellbook/backends/slurm_megatron"
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_JOB_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclasses.dataclass
class MegatronCheckpointMergeConfig:
    name: str
    checkpoints: list[str]
    output: str
    megatron_path: str

    megatron_commit: str = ""
    megatron_container_path: str = "/opt/megatron"
    checkpoint_steps: list[int] = dataclasses.field(default_factory=list)
    merge_method: str = "mean"
    original_schedule: str = "stable"
    original_end_multiplier: float | None = None
    original_decay_steps: int | None = None
    original_decay_start_step: int | None = None
    target_end_multiplier: float = 1e-4
    backend: str = "gloo"

    account: str = ""
    partition: str = ""
    nodes: int = 1
    workers_per_node: int = 4
    cpus_per_task: int = 18
    gpus_per_node: int = 0
    memory: str = "460000"
    run_time: str = "05:00:00"
    reservation: str = ""
    exclude: str = ""
    log_dir: str = "slurm_logs/merge"
    srun_extra_args: str = ""

    container_edf: str = ""
    container_mounts: str = ""
    env_vars: dict[str, str] = dataclasses.field(default_factory=dict)


def _is_megatron_url(value: str) -> bool:
    parsed = urlparse(value)
    return (
        parsed.scheme in {"git", "http", "https", "ssh"} and bool(parsed.netloc)
    ) or (value.startswith("git@") and ":" in value)


def _validate(cfg: MegatronCheckpointMergeConfig) -> None:
    if not _JOB_NAME.fullmatch(cfg.name):
        raise ValueError(
            "name may contain only letters, numbers, dots, underscores, and hyphens"
        )
    if max(len(cfg.checkpoints), len(cfg.checkpoint_steps)) < 2:
        raise ValueError("checkpoint merging requires at least two checkpoints")
    if not cfg.output or not cfg.megatron_path:
        raise ValueError("output and megatron_path must not be empty")
    container_path = cfg.megatron_container_path.rstrip("/")
    if (
        not container_path.startswith("/")
        or ".." in container_path.split("/")
        or re.fullmatch(r"/[A-Za-z0-9_./-]+", container_path) is None
        or len([part for part in container_path.split("/") if part]) < 2
    ):
        raise ValueError(
            "megatron_container_path must be a shell-safe absolute path with at least two components"
        )
    for field_name in ("nodes", "workers_per_node", "cpus_per_task"):
        if getattr(cfg, field_name) <= 0:
            raise ValueError(f"{field_name} must be greater than zero")
    if cfg.gpus_per_node < 0:
        raise ValueError("gpus_per_node must not be negative")
    if cfg.backend not in {"gloo", "nccl"}:
        raise ValueError("backend must be 'gloo' or 'nccl'")
    if cfg.checkpoint_steps:
        if len(cfg.checkpoints) != 1 and len(cfg.checkpoints) != len(cfg.checkpoint_steps):
            raise ValueError(
                "provide one checkpoint root or one root per checkpoint step"
            )
        if any(
            left >= right
            for left, right in zip(cfg.checkpoint_steps, cfg.checkpoint_steps[1:])
        ):
            raise ValueError("checkpoint_steps must be strictly increasing")
    invalid_env_names = [name for name in cfg.env_vars if not _ENV_NAME.fullmatch(name)]
    if invalid_env_names:
        raise ValueError(f"invalid environment variable names: {invalid_env_names}")


def _merge_args(cfg: MegatronCheckpointMergeConfig) -> list[str]:
    args = [
        "--checkpoints",
        *cfg.checkpoints,
    ]
    if cfg.checkpoint_steps:
        args.extend(["--checkpoint-steps", *map(str, cfg.checkpoint_steps)])
    args.extend(
        [
            "--output",
            cfg.output,
            "--merge-method",
            cfg.merge_method,
            "--original-schedule",
            cfg.original_schedule,
            "--target-end-multiplier",
            str(cfg.target_end_multiplier),
            "--backend",
            cfg.backend,
        ]
    )
    for option, value in (
        ("--original-end-multiplier", cfg.original_end_multiplier),
        ("--original-decay-steps", cfg.original_decay_steps),
        ("--original-decay-start-step", cfg.original_decay_start_step),
    ):
        if value is not None:
            args.extend([option, str(value)])
    return args


def _shell_double_quote(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("$", "\\$")
        .replace("`", "\\`")
    )
    return f'"{escaped}"'


def render(cfg: MegatronCheckpointMergeConfig) -> str:
    """Render a distributed checkpoint merge job."""
    _validate(cfg)
    env = Environment(
        loader=FileSystemLoader([str(_TEMPLATES_DIR), str(_BACKEND_TEMPLATES_DIR)]),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    context = dataclasses.asdict(cfg)
    megatron_url = cfg.megatron_path if _is_megatron_url(cfg.megatron_path) else ""
    context.update(
        {
            "log_dir": str(Path(cfg.log_dir).expanduser().resolve()),
            "total_workers": cfg.nodes * cfg.workers_per_node,
            "megatron_path": (
                "" if megatron_url else str(Path(cfg.megatron_path).expanduser())
            ),
            "megatron_url": megatron_url,
            "megatron_cache_key": hashlib.sha256(
                cfg.megatron_path.encode()
            ).hexdigest()[:16],
            "megatron_worktree_key": hashlib.sha256(
                f"{cfg.megatron_path}\0{cfg.megatron_commit}".encode()
            ).hexdigest()[:16],
            "megatron_container_path": cfg.megatron_container_path.rstrip("/"),
            "merge_args": [_shell_double_quote(arg) for arg in _merge_args(cfg)],
        }
    )
    return env.get_template("checkpoint.sh.j2").render(context)


def _sbatch(script: str) -> str:
    result = subprocess.run(
        ["sbatch"], input=script, capture_output=True, text=True, check=True
    )
    return result.stdout.strip().split()[-1]


def submit(cfg: MegatronCheckpointMergeConfig) -> str:
    """Render and submit a checkpoint merge job."""
    Path(cfg.log_dir).expanduser().mkdir(parents=True, exist_ok=True)
    job_id = _sbatch(render(cfg))
    print(f"  {cfg.name}: submitted → job {job_id}")
    return job_id
