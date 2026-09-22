"""Render and submit a native multi-node vLLM server."""

from __future__ import annotations

import dataclasses
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from spellbook.megatron.flags import to_args

_TEMPLATES_DIR = Path(__file__).parent
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_JOB_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclasses.dataclass
class VLLMServerConfig:
    name: str
    model: str
    served_model_name: str = ""
    tensor_parallel_size: int = 1
    enable_expert_parallel: bool = False
    all2all_backend: str = ""
    vllm_args: dict[str, Any] = dataclasses.field(default_factory=dict)

    host: str = "0.0.0.0"
    port: int = 8000
    data_parallel_rpc_port: int = 13345

    account: str = ""
    partition: str = ""
    nodes: int = 1
    gpus_per_node: int = 4
    cpus_per_task: int = 288
    memory: str = ""
    run_time: str = "05:00:00"
    reservation: str = ""
    qos: str = ""
    exclude: str = ""
    log_dir: str = "slurm_logs/inference"
    srun_extra_args: str = "--network=disable_rdzv_get"

    container_edf: str = ""
    container_mounts: str = ""
    env_vars: dict[str, str] = dataclasses.field(default_factory=dict)

    @property
    def data_parallel_size_local(self) -> int:
        return self.gpus_per_node // self.tensor_parallel_size

    @property
    def data_parallel_size(self) -> int:
        return self.nodes * self.data_parallel_size_local


def _validate(cfg: VLLMServerConfig) -> None:
    if not _JOB_NAME.fullmatch(cfg.name):
        raise ValueError("name may contain only letters, numbers, dots, underscores, and hyphens")
    if not cfg.model:
        raise ValueError("model must not be empty")
    for field_name in (
        "tensor_parallel_size",
        "nodes",
        "gpus_per_node",
        "cpus_per_task",
        "port",
        "data_parallel_rpc_port",
    ):
        if getattr(cfg, field_name) <= 0:
            raise ValueError(f"{field_name} must be greater than zero")
    if cfg.gpus_per_node % cfg.tensor_parallel_size:
        raise ValueError("gpus_per_node must be divisible by tensor_parallel_size")
    for port in (cfg.port, cfg.data_parallel_rpc_port):
        if port > 65535:
            raise ValueError("ports must be in [1, 65535]")
    invalid = [name for name in cfg.env_vars if not _ENV_NAME.fullmatch(name)]
    if invalid:
        raise ValueError(f"invalid environment variable names: {invalid}")


def _server_args(cfg: VLLMServerConfig) -> list[str]:
    args = [
        cfg.model,
        "--host",
        cfg.host,
        "--tensor-parallel-size",
        str(cfg.tensor_parallel_size),
        "--data-parallel-size",
        str(cfg.data_parallel_size),
        "--data-parallel-size-local",
        str(cfg.data_parallel_size_local),
    ]
    if cfg.served_model_name:
        args.extend(["--served-model-name", cfg.served_model_name])
    if cfg.enable_expert_parallel:
        args.append("--enable-expert-parallel")
    if cfg.all2all_backend:
        args.extend(["--all2all-backend", cfg.all2all_backend])
    args.extend(to_args(cfg.vllm_args))
    return args


def _shell_quote(value: str) -> str:
    return shlex.quote(value)


def _context(cfg: VLLMServerConfig) -> dict[str, Any]:
    context = dataclasses.asdict(cfg)
    context.update(
        {
            "log_dir": str(Path(cfg.log_dir).expanduser().resolve()),
            "data_parallel_size": cfg.data_parallel_size,
            "data_parallel_size_local": cfg.data_parallel_size_local,
            "server_args": [_shell_quote(arg) for arg in _server_args(cfg)],
        }
    )
    return context


def render(cfg: VLLMServerConfig) -> str:
    """Render one vLLM process per node, with local multiprocessing for GPUs."""
    _validate(cfg)
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    return env.get_template("vllm_server.sh.j2").render(_context(cfg))


def _sbatch(script: str, qos: str = "") -> str:
    cmd = ["sbatch"] + ([f"--qos={qos}"] if qos else [])
    result = subprocess.run(cmd, input=script, capture_output=True, text=True, check=True)
    return result.stdout.strip().split()[-1]


def submit(cfg: VLLMServerConfig) -> str:
    """Render and submit a vLLM server job."""
    (Path(cfg.log_dir).expanduser() / cfg.name).mkdir(parents=True, exist_ok=True)
    job_id = _sbatch(render(cfg), cfg.qos)
    print(f"  {cfg.name}: submitted → job {job_id}")
    return job_id
