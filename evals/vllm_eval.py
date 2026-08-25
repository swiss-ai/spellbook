"""Submit lm-evaluation-harness evaluations using its vLLM backend."""

from __future__ import annotations

import dataclasses
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from evals import flags as lm_eval_flags

_TEMPLATES_DIR = Path(__file__).parent
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_JOB_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_LM_EVAL_ARG = {"lm_eval_arg": True}


@dataclasses.dataclass
class VLLMEvalConfig:
    # --- Evaluation ---
    model_name: str
    model: str
    tokenizer: str = ""

    # --- Eval ---
    tasks: list[str] = dataclasses.field(default_factory=list, metadata=_LM_EVAL_ARG)
    batch_size: str | int = dataclasses.field(default="auto", metadata=_LM_EVAL_ARG)
    cache_requests: str = dataclasses.field(default="", metadata=_LM_EVAL_ARG)
    num_fewshot: int | None = dataclasses.field(default=None, metadata=_LM_EVAL_ARG)
    limit: float | None = dataclasses.field(default=None, metadata=_LM_EVAL_ARG)
    metadata: dict[str, object] = dataclasses.field(
        default_factory=dict, metadata=_LM_EVAL_ARG
    )
    output_dir: str | None = None
    log_samples: bool = dataclasses.field(default=False, metadata=_LM_EVAL_ARG)
    write_out: bool = dataclasses.field(default=False, metadata=_LM_EVAL_ARG)

    # --- vLLM model arguments ---
    dtype: str = "auto"
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    trust_remote_code: bool = False
    seed: int = 1234
    model_args_extra: dict[str, Any] = dataclasses.field(default_factory=dict)
    lm_eval_args_extra: dict[str, Any] = dataclasses.field(default_factory=dict)

    # --- Slurm ---
    account: str = ""
    partition: str = ""
    nodes: int = 1
    gpus_per_node: int = 4
    cpus_per_task: int = 72
    memory: str = "460000"
    run_time: str = "03:00:00"
    reservation: str = ""
    exclude: str = ""
    conversion_job_id: str = ""
    log_dir: str = "slurm_logs/eval"
    srun_extra_args: str = ""

    # --- Container ---
    container_edf: str = ""
    container_mounts: str = ""

    # --- WandB ---
    wandb_project: str = ""
    wandb_id: str = ""

    # --- Cache / install ---
    hf_home: str = ""
    lm_eval_install: str = ""
    lm_eval_install_with_python: bool = False
    install_commands: str = ""

    # --- Runtime environment ---
    env_vars: dict[str, str] = dataclasses.field(default_factory=dict)


def _marked_lm_eval_args(cfg: VLLMEvalConfig) -> dict[str, Any]:
    return {
        field.name: getattr(cfg, field.name)
        for field in dataclasses.fields(cfg)
        if field.metadata.get("lm_eval_arg")
    }


def _model_args(cfg: VLLMEvalConfig) -> str:
    return lm_eval_flags.model_args(
        {
            "pretrained": cfg.model,
            "tokenizer": cfg.tokenizer,
            "dtype": cfg.dtype,
            "tensor_parallel_size": cfg.tensor_parallel_size,
            "data_parallel_size": cfg.data_parallel_size,
            "gpu_memory_utilization": cfg.gpu_memory_utilization,
            "max_model_len": cfg.max_model_len,
            "trust_remote_code": cfg.trust_remote_code,
            "seed": cfg.seed,
            **cfg.model_args_extra,
        }
    )


def _validate(cfg: VLLMEvalConfig) -> None:
    if cfg.nodes != 1:
        raise ValueError(
            "standard lm-eval vLLM launches currently support one Slurm node"
        )
    if not _JOB_NAME.fullmatch(cfg.model_name):
        raise ValueError(
            "model_name may contain only letters, numbers, dots, underscores, and hyphens"
        )
    if not cfg.model:
        raise ValueError("model must not be empty")
    if not cfg.tasks:
        raise ValueError("tasks must not be empty")
    for field_name in (
        "gpus_per_node",
        "cpus_per_task",
        "tensor_parallel_size",
        "data_parallel_size",
    ):
        if getattr(cfg, field_name) <= 0:
            raise ValueError(f"{field_name} must be greater than zero")
    required_gpus = cfg.tensor_parallel_size * cfg.data_parallel_size
    if required_gpus > cfg.gpus_per_node:
        raise ValueError(
            "tensor_parallel_size * data_parallel_size exceeds gpus_per_node"
        )
    if not 0 < cfg.gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    if cfg.max_model_len is not None and cfg.max_model_len <= 0:
        raise ValueError("max_model_len must be greater than zero")
    invalid_env_names = [name for name in cfg.env_vars if not _ENV_NAME.fullmatch(name)]
    if invalid_env_names:
        raise ValueError(f"invalid environment variable names: {invalid_env_names}")
    if cfg.conversion_job_id and not cfg.conversion_job_id.isdigit():
        raise ValueError("conversion_job_id must be numeric")


def render(cfg: VLLMEvalConfig) -> str:
    """Render a single-node Slurm job using lm-eval's standard vLLM backend."""
    _validate(cfg)
    output_root = (
        Path(cfg.output_dir).expanduser() if cfg.output_dir else Path.cwd() / "evals"
    )
    output_dir = (output_root / cfg.model_name).resolve()
    log_dir = Path(cfg.log_dir).expanduser().resolve()
    lm_eval_args = {
        "model": "vllm",
        "model_args": _model_args(cfg),
        **_marked_lm_eval_args(cfg),
        "output_path": str(output_dir),
        **cfg.lm_eval_args_extra,
    }

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    context = dataclasses.asdict(cfg)
    context.update(
        {
            "job_name": f"eval_{cfg.model_name}",
            "output_dir": str(output_dir),
            "log_dir": str(log_dir),
            "lm_eval_args_lines": lm_eval_flags.to_shell_lines(lm_eval_args),
            "wandb_args_line": next(
                iter(
                    lm_eval_flags.to_shell_lines(
                        {
                            "wandb_args": (
                                f"project={cfg.wandb_project},"
                                f"id={cfg.wandb_id or cfg.model_name},resume=allow"
                                if cfg.wandb_project
                                else None
                            )
                        }
                    )
                ),
                "",
            ),
        }
    )
    return env.get_template("vllm_eval.sh.j2").render(context)


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
    (Path(cfg.log_dir).expanduser() / cfg.model_name).mkdir(
        parents=True, exist_ok=True
    )
    job_id = _sbatch(render(cfg), cfg.reservation, cfg.exclude)
    print(f"  {cfg.model_name}: submitted → job {job_id}")
    return job_id
