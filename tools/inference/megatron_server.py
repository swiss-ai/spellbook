"""Render and submit a Megatron dynamic text-generation HTTP server."""

from __future__ import annotations

import dataclasses
import re
import subprocess
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from spellbook.megatron.flags import to_args

_TEMPLATES_DIR = Path(__file__).parent
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_JOB_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclasses.dataclass
class MegatronServerConfig:
    # Model and Megatron.
    name: str
    checkpoint: str
    ckpt_step: int
    tokenizer_model: str
    megatron_path: str
    megatron_args: dict[str, Any] = dataclasses.field(default_factory=dict)

    # Parallelism and inference engine.
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1
    seq_length: int = 4096
    micro_batch_size: int = 8
    max_requests: int = 8
    max_tokens: int = 32768
    block_size: int = 256
    buffer_size_gb: float = 1
    cuda_graphs: bool = True
    pad_moe_experts_for_cuda_graphs: bool = True

    # HTTP server. Keep host=None for routable per-node coordinator addresses.
    # The HTTP frontend still binds to all interfaces when host is omitted.
    host: str | None = None
    port: int = 5000

    # Slurm.
    account: str = ""
    partition: str = ""
    nodes: int = 1
    gpus_per_node: int = 4
    cpus_per_task: int = 18
    memory: str = "460000"
    run_time: str = "05:00:00"
    reservation: str = ""
    exclude: str = ""
    log_dir: str = "slurm_logs/inference"
    srun_extra_args: str = ""

    # Container and runtime.
    container_edf: str = ""
    container_mounts: str = ""
    install_commands: str = "python -m pip install quart hypercorn"
    env_vars: dict[str, str] = dataclasses.field(default_factory=dict)


def _validate(cfg: MegatronServerConfig) -> None:
    if not _JOB_NAME.fullmatch(cfg.name):
        raise ValueError(
            "name may contain only letters, numbers, dots, underscores, and hyphens"
        )
    for field_name in (
        "ckpt_step",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "expert_parallel_size",
        "seq_length",
        "micro_batch_size",
        "max_requests",
        "max_tokens",
        "block_size",
        "nodes",
        "gpus_per_node",
        "cpus_per_task",
        "port",
    ):
        if getattr(cfg, field_name) <= 0:
            raise ValueError(f"{field_name} must be greater than zero")
    if not cfg.checkpoint:
        raise ValueError("checkpoint must not be empty")
    if not cfg.tokenizer_model:
        raise ValueError("tokenizer_model must not be empty")
    if not cfg.megatron_path:
        raise ValueError("megatron_path must not be empty")
    model_parallel_size = (
        cfg.tensor_parallel_size
        * cfg.pipeline_parallel_size
        * cfg.expert_parallel_size
    )
    total_tasks = cfg.nodes * cfg.gpus_per_node
    if total_tasks % model_parallel_size != 0:
        raise ValueError(
            "nodes * gpus_per_node must be divisible by tensor_parallel_size * "
            "pipeline_parallel_size * expert_parallel_size"
        )
    if cfg.nodes > 1 and cfg.host is not None:
        raise ValueError(
            "host must be None for multi-node launches so each rank advertises "
            "its routable compute-node hostname"
        )
    if cfg.max_tokens < cfg.max_requests:
        raise ValueError("max_tokens must be at least max_requests")
    if not 0 < cfg.port <= 65535:
        raise ValueError("port must be in [1, 65535]")
    invalid_env_names = [name for name in cfg.env_vars if not _ENV_NAME.fullmatch(name)]
    if invalid_env_names:
        raise ValueError(f"invalid environment variable names: {invalid_env_names}")


def _server_args(cfg: MegatronServerConfig) -> list[str]:
    args = [
        "tools/run_dynamic_text_generation_server.py",
        "--load",
        cfg.checkpoint,
        "--ckpt-step",
        str(cfg.ckpt_step),
        "--tokenizer-type",
        "HuggingFaceTokenizer",
        "--tokenizer-model",
        cfg.tokenizer_model,
        "--tensor-model-parallel-size",
        str(cfg.tensor_parallel_size),
        "--pipeline-model-parallel-size",
        str(cfg.pipeline_parallel_size),
        "--expert-model-parallel-size",
        str(cfg.expert_parallel_size),
        "--seq-length",
        str(cfg.seq_length),
        "--micro-batch-size",
        str(cfg.micro_batch_size),
        "--bf16",
        "--use-checkpoint-args",
        "--no-load-optim",
        "--no-load-rng",
        "--exit-on-missing-checkpoint",
        "--auto-detect-ckpt-format",
        "--inference-max-seq-length",
        str(cfg.seq_length),
        "--inference-dynamic-batching",
        "--inference-dynamic-batching-block-size",
        str(cfg.block_size),
        "--inference-dynamic-batching-buffer-size-gb",
        str(cfg.buffer_size_gb),
        "--inference-dynamic-batching-max-requests",
        str(cfg.max_requests),
        "--inference-dynamic-batching-max-tokens",
        str(cfg.max_tokens),
    ]
    if cfg.cuda_graphs:
        args.extend(
            [
                "--cuda-graph-impl",
                "local",
                "--inference-dynamic-batching-num-cuda-graphs",
                "-1",
            ]
        )
    if cfg.cuda_graphs and cfg.pad_moe_experts_for_cuda_graphs:
        args.append("--moe-pad-experts-for-cuda-graph-inference")
    args.extend(to_args(cfg.megatron_args))
    if cfg.host is not None:
        args.extend(["--host", cfg.host])
    args.extend(["--port", str(cfg.port)])
    return args


def _shell_double_quote(value: str) -> str:
    """Quote one value for the inner bash script embedded in single quotes."""
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("$", "\\$")
        .replace("`", "\\`")
    )
    return f'"{escaped}"'


def render(cfg: MegatronServerConfig) -> str:
    """Render a one-Slurm-task-per-GPU server job."""
    _validate(cfg)
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    context = dataclasses.asdict(cfg)
    context.update(
        {
            "log_dir": str(Path(cfg.log_dir).expanduser().resolve()),
            "total_tasks": cfg.nodes * cfg.gpus_per_node,
            "megatron_path_shell": _shell_double_quote(
                str(Path(cfg.megatron_path).expanduser())
            ),
            "server_args": [_shell_double_quote(arg) for arg in _server_args(cfg)],
        }
    )
    return env.get_template("megatron_server.sh.j2").render(context)


def _sbatch(script: str) -> str:
    result = subprocess.run(
        ["sbatch"], input=script, capture_output=True, text=True, check=True
    )
    return result.stdout.strip().split()[-1]


def submit(cfg: MegatronServerConfig) -> str:
    """Render and submit a dynamic server job."""
    (Path(cfg.log_dir).expanduser() / cfg.name).mkdir(parents=True, exist_ok=True)
    job_id = _sbatch(render(cfg))
    print(f"  {cfg.name}: submitted → job {job_id}")
    return job_id
