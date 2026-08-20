"""Submit distributed vLLM evaluation runners to Slurm."""

from __future__ import annotations

import dataclasses
import os
import re
import subprocess
import tempfile
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_TEMPLATES_DIR = Path(__file__).parent
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_JOB_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclasses.dataclass
class VLLMEvalConfig:
    """Configuration for an HF-model evaluation driven by a custom vLLM runner."""

    # --- Evaluation ---
    model_name: str
    model: str
    runner: str
    tasks: list[str] = dataclasses.field(default_factory=list)
    tokenizer: str = ""
    batch_size: int = 1
    max_length: int = 8192
    output_dir: str | None = None
    extra_runner_args: list[str] = dataclasses.field(default_factory=list)
    python_executable: str = "/opt/eval-venv/bin/python"

    # --- Slurm ---
    account: str = ""
    partition: str = ""
    nodes: int = 1
    gpus_per_node: int = 4
    launch_mode: str = "torchrun"     # "torchrun" or one process per Slurm "task"
    cpus_per_task: int = 72
    memory: str = "460000"
    run_time: str = "03:00:00"
    reservation: str = ""
    exclude: str = ""
    conversion_job_id: str = ""
    log_dir: str = "slurm_logs/eval"
    master_port: int = 6991
    srun_extra_args: str = ""

    # --- Container ---
    container_edf: str = ""
    container_mounts: str = ""

    # --- Runtime environment ---
    env_vars: dict[str, str] = dataclasses.field(default_factory=dict)


def _quoted_runner_arg(value: str) -> str:
    """Quote one array element embedded in the template's single-quoted shell."""
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("$", "\\$")
        .replace("`", "\\`")
        .replace("'", "'\"'\"'")
    )
    return f'"{escaped}"'


def _runner_args(cfg: VLLMEvalConfig, output_dir: Path) -> list[str]:
    args = [
        "--model",
        cfg.model,
        "--output-dir",
        str(output_dir),
        "--tasks",
        ",".join(cfg.tasks),
        "--batch-size",
        str(cfg.batch_size),
        "--max-length",
        str(cfg.max_length),
    ]
    if cfg.tokenizer:
        args += ["--tokenizer", cfg.tokenizer]
    args += cfg.extra_runner_args
    return args


def _validate(cfg: VLLMEvalConfig) -> None:
    if cfg.launch_mode not in {"torchrun", "tasks"}:
        raise ValueError("launch_mode must be 'torchrun' or 'tasks'")
    if not _JOB_NAME.fullmatch(cfg.model_name):
        raise ValueError(
            "model_name may contain only letters, numbers, dots, underscores, and hyphens"
        )
    for field_name in ("model", "runner", "python_executable"):
        if not getattr(cfg, field_name):
            raise ValueError(f"{field_name} must not be empty")
    if not cfg.tasks:
        raise ValueError("tasks must not be empty")
    for field_name in ("nodes", "gpus_per_node", "cpus_per_task", "batch_size", "max_length"):
        if getattr(cfg, field_name) <= 0:
            raise ValueError(f"{field_name} must be greater than zero")
    if not 1 <= cfg.master_port <= 65535:
        raise ValueError("master_port must be between 1 and 65535")
    invalid_env_names = [name for name in cfg.env_vars if not _ENV_NAME.fullmatch(name)]
    if invalid_env_names:
        raise ValueError(f"invalid environment variable names: {invalid_env_names}")
    if cfg.conversion_job_id and not cfg.conversion_job_id.isdigit():
        raise ValueError("conversion_job_id must be numeric")


def render(cfg: VLLMEvalConfig) -> str:
    """Render a Slurm script for one vLLM evaluation."""
    _validate(cfg)
    output_root = (
        Path(cfg.output_dir).expanduser() if cfg.output_dir else Path.cwd() / "evals"
    )
    output_dir = (output_root / cfg.model_name).resolve()
    log_dir = (Path(cfg.log_dir).expanduser() / cfg.model_name).resolve()

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    template = env.get_template("vllm_eval.sh.j2")
    context = dataclasses.asdict(cfg)
    ntasks_per_node = cfg.gpus_per_node if cfg.launch_mode == "tasks" else 1
    context.update(
        {
            "job_name": f"eval_{cfg.model_name}",
            "ntasks_per_node": ntasks_per_node,
            "total_tasks": cfg.nodes * ntasks_per_node,
            "output_dir": str(output_dir),
            "log_dir": str(log_dir),
            "runner_args_lines": [
                _quoted_runner_arg(arg) for arg in _runner_args(cfg, output_dir)
            ],
        }
    )
    return template.render(context)


def _sbatch(script: str, reservation: str, exclude: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as file:
        file.write(script)
        script_path = file.name
    try:
        command = ["sbatch"]
        if exclude:
            command.append(f"--exclude={exclude}")
        if reservation:
            command.append(f"--reservation={reservation}")
        command.append(script_path)
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout.strip().split()[-1]
    finally:
        os.unlink(script_path)


def submit(cfg: VLLMEvalConfig) -> str:
    """Render and submit one vLLM evaluation job."""
    log_dir = Path(cfg.log_dir).expanduser() / cfg.model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    script = render(cfg)
    job_id = _sbatch(script, cfg.reservation, cfg.exclude)
    print(f"  {cfg.model_name}: submitted → job {job_id}")
    return job_id
