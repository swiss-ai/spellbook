"""Render and submit disaggregated vLLM prefill/decode serving."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from spellbook.megatron.flags import to_args
from tools.inference.vllm_server import _ENV_NAME, _JOB_NAME, _sbatch, _shell_quote

_TEMPLATES_DIR = Path(__file__).parent


@dataclasses.dataclass
class VLLMPDServerConfig:
    name: str
    model: str
    proxy_script: str
    served_model_name: str = ""
    tensor_parallel_size: int = 1
    prefill_nodes: int = 2
    decode_nodes: int = 2
    gpus_per_node: int = 4
    prefill_backend: str = "deepep_high_throughput"
    decode_backend: str = "deepep_low_latency"
    vllm_args: dict[str, Any] = dataclasses.field(default_factory=dict)

    prefill_port: int = 8100
    decode_port: int = 8200
    proxy_port: int = 9000
    proxy_health_path: str = "/healthcheck"
    prefill_rpc_port: int = 13345
    decode_rpc_port: int = 13346
    nixl_side_channel_port: int = 5600
    health_timeout: int = 900

    uccl_dispatch_config: str = ""
    uccl_combine_config: str = ""

    account: str = ""
    partition: str = ""
    cpus_per_task: int = 288
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
    def total_nodes(self) -> int:
        return self.prefill_nodes + self.decode_nodes

    @property
    def data_parallel_size_local(self) -> int:
        return self.gpus_per_node // self.tensor_parallel_size

    @property
    def prefill_data_parallel_size(self) -> int:
        return self.prefill_nodes * self.data_parallel_size_local

    @property
    def decode_data_parallel_size(self) -> int:
        return self.decode_nodes * self.data_parallel_size_local


def _validate(cfg: VLLMPDServerConfig) -> None:
    if not _JOB_NAME.fullmatch(cfg.name):
        raise ValueError("name may contain only letters, numbers, dots, underscores, and hyphens")
    if not cfg.model:
        raise ValueError("model must not be empty")
    if not cfg.proxy_script:
        raise ValueError("proxy_script must not be empty")
    for field_name in (
        "tensor_parallel_size",
        "prefill_nodes",
        "decode_nodes",
        "gpus_per_node",
        "cpus_per_task",
        "health_timeout",
    ):
        if getattr(cfg, field_name) <= 0:
            raise ValueError(f"{field_name} must be greater than zero")
    if not cfg.proxy_health_path.startswith("/"):
        raise ValueError("proxy_health_path must start with '/'")
    if cfg.gpus_per_node % cfg.tensor_parallel_size:
        raise ValueError("gpus_per_node must be divisible by tensor_parallel_size")
    if cfg.prefill_data_parallel_size != cfg.decode_data_parallel_size:
        raise ValueError("prefill and decode data-parallel sizes must match")
    ports = (
        cfg.prefill_port,
        cfg.decode_port,
        cfg.proxy_port,
        cfg.prefill_rpc_port,
        cfg.decode_rpc_port,
        cfg.nixl_side_channel_port,
    )
    if any(port <= 0 or port > 65535 for port in ports) or len(set(ports)) != len(ports):
        raise ValueError("PD ports must be distinct and in [1, 65535]")
    invalid = [name for name in cfg.env_vars if not _ENV_NAME.fullmatch(name)]
    if invalid:
        raise ValueError(f"invalid environment variable names: {invalid}")


def _common_args(cfg: VLLMPDServerConfig) -> list[str]:
    args = [
        cfg.model,
        "--host",
        "0.0.0.0",
        "--tensor-parallel-size",
        str(cfg.tensor_parallel_size),
        "--data-parallel-size",
        str(cfg.prefill_data_parallel_size),
        "--data-parallel-size-local",
        str(cfg.data_parallel_size_local),
        "--enable-expert-parallel",
    ]
    if cfg.served_model_name:
        args.extend(["--served-model-name", cfg.served_model_name])
    args.extend(to_args(cfg.vllm_args))
    return args


def render(cfg: VLLMPDServerConfig) -> str:
    """Render two explicit native vLLM groups and a routing proxy."""
    _validate(cfg)
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    context = dataclasses.asdict(cfg)
    context.update(
        {
            "total_nodes": cfg.total_nodes,
            "data_parallel_size_local": cfg.data_parallel_size_local,
            "data_parallel_size": cfg.prefill_data_parallel_size,
            "log_dir": str(Path(cfg.log_dir).expanduser().resolve()),
            "proxy_script": str(Path(cfg.proxy_script).expanduser().resolve()),
            "common_args": [_shell_quote(arg) for arg in _common_args(cfg)],
        }
    )
    return env.get_template("vllm_pd_server.sh.j2").render(context)


def submit(cfg: VLLMPDServerConfig) -> str:
    """Render and submit a disaggregated vLLM server job."""
    (Path(cfg.log_dir).expanduser() / cfg.name).mkdir(parents=True, exist_ok=True)
    job_id = _sbatch(render(cfg), cfg.qos)
    print(f"  {cfg.name}: submitted → job {job_id}")
    return job_id
