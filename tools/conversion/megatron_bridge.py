"""Launch Megatron-Bridge checkpoint exports through its native pipeline."""

from __future__ import annotations

import dataclasses
import os
import subprocess
from pathlib import Path

_DTYPES = {"bfloat16", "float16", "float32"}


@dataclasses.dataclass
class BridgeExportConfig:
    bridge_root: str
    hf_model: str
    megatron_path: str
    hf_path: str
    container_image: str

    bridge_commit: str = ""
    megatron_root: str = ""
    megatron_commit: str = ""
    hf_revision: str = ""
    torch_dtype: str = "bfloat16"
    export_weight_dtype: str = ""
    trust_remote_code: bool = False
    overwrite: bool = False
    not_strict: bool = False
    distributed_save: bool = True
    save_every_n_ranks: int = 1

    nodes: int = 1
    gpus_per_node: int = 4
    cpus_per_task: int = 72
    tp: int = 1
    pp: int = 1
    ep: int = 4
    etp: int = 1
    distributed_timeout_minutes: int | None = None

    account: str = ""
    partition: str = ""
    run_time: str = "04:00:00"
    memory: str = "0"
    gres: str = ""
    no_gpu_resource_request: bool = False
    experiment_name: str = ""
    mounts: list[str] = dataclasses.field(default_factory=list)
    env_names: list[str] = dataclasses.field(default_factory=list)
    srun_args: list[str] = dataclasses.field(
        default_factory=lambda: ["--network=disable_rdzv_get", "--mpi=pmix"]
    )
    detach: bool = True
    dry_run: bool = False


def _resolve_checkout(path: str, commit: str, name: str) -> Path:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"{name} checkout does not exist: {root}")
    if commit:
        actual = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        expected = subprocess.run(
            ["git", "-C", str(root), "rev-parse", f"{commit}^{{commit}}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if actual != expected:
            raise ValueError(f"{name} checkout is {actual}, expected {expected}")
    return root


def _validate(cfg: BridgeExportConfig) -> tuple[Path, Path, Path | None]:
    root = _resolve_checkout(cfg.bridge_root, cfg.bridge_commit, "Megatron-Bridge")
    launcher = root / "scripts" / "conversion" / "convert.sh"
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise ValueError(f"missing executable Megatron-Bridge converter: {launcher}")
    if cfg.megatron_commit and not cfg.megatron_root:
        raise ValueError("megatron_commit requires megatron_root")
    megatron_root = None
    if cfg.megatron_root:
        megatron_root = _resolve_checkout(cfg.megatron_root, cfg.megatron_commit, "Megatron-LM")
        if not (megatron_root / "megatron" / "core").is_dir():
            raise ValueError(f"invalid Megatron-LM checkout: {megatron_root}")
    for name in ("hf_model", "megatron_path", "hf_path", "container_image"):
        if not getattr(cfg, name):
            raise ValueError(f"{name} must not be empty")
    checkpoint = Path(cfg.megatron_path).expanduser().resolve()
    if not checkpoint.is_dir():
        raise ValueError(f"Megatron checkpoint does not exist: {checkpoint}")
    has_run_config = (checkpoint / "run_config.yaml").is_file() or any(
        child.is_dir() and child.name.startswith("iter_") and (child / "run_config.yaml").is_file()
        for child in checkpoint.iterdir()
    )
    if not has_run_config:
        raise ValueError(f"Megatron checkpoint has no run_config.yaml: {checkpoint}")
    for name in ("nodes", "gpus_per_node", "cpus_per_task", "tp", "pp", "ep", "etp"):
        if getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be greater than zero")
    world_size = cfg.nodes * cfg.gpus_per_node
    if world_size != cfg.tp * cfg.pp * cfg.ep:
        raise ValueError("nodes * gpus_per_node must equal tp * pp * ep")
    if world_size % (cfg.etp * cfg.ep * cfg.pp):
        raise ValueError("world size must be divisible by etp * ep * pp")
    if cfg.torch_dtype not in _DTYPES:
        raise ValueError(f"unsupported torch_dtype: {cfg.torch_dtype}")
    if cfg.export_weight_dtype and cfg.export_weight_dtype not in _DTYPES:
        raise ValueError(f"unsupported export_weight_dtype: {cfg.export_weight_dtype}")
    if cfg.save_every_n_ranks <= 0:
        raise ValueError("save_every_n_ranks must be greater than zero")
    if not cfg.distributed_save and cfg.save_every_n_ranks != 1:
        raise ValueError("save_every_n_ranks requires distributed_save")
    if cfg.distributed_timeout_minutes is not None and cfg.distributed_timeout_minutes <= 0:
        raise ValueError("distributed_timeout_minutes must be greater than zero")
    if not cfg.account or not cfg.partition:
        raise ValueError("account and partition must not be empty")
    missing_env = [name for name in cfg.env_names if name not in os.environ]
    if missing_env:
        raise ValueError(f"environment variables are not set: {missing_env}")
    return launcher, root, megatron_root


def render_command(cfg: BridgeExportConfig) -> list[str]:
    """Return the native Megatron-Bridge Slurm export command."""
    launcher, bridge_root, megatron_root = _validate(cfg)
    command = [
        str(launcher),
        "export",
        "--executor",
        "slurm",
        "--device",
        "gpu",
        "--nodes",
        str(cfg.nodes),
        "--gpus-per-node",
        str(cfg.gpus_per_node),
        "--account",
        cfg.account,
        "--partition",
        cfg.partition,
        "--time",
        cfg.run_time,
        "--mem",
        cfg.memory,
        "--container-image",
        cfg.container_image,
        "--hf-model",
        cfg.hf_model,
        "--megatron-path",
        str(Path(cfg.megatron_path).expanduser().resolve()),
        "--hf-path",
        str(Path(cfg.hf_path).expanduser().resolve()),
        "--torch-dtype",
        cfg.torch_dtype,
        "--tp",
        str(cfg.tp),
        "--pp",
        str(cfg.pp),
        "--ep",
        str(cfg.ep),
        "--etp",
        str(cfg.etp),
        "--save-every-n-ranks",
        str(cfg.save_every_n_ranks),
        "--distributed-save" if cfg.distributed_save else "--no-distributed-save",
    ]
    for option, value in (
        ("--hf-revision", cfg.hf_revision),
        ("--export-weight-dtype", cfg.export_weight_dtype),
        ("--gres", cfg.gres),
        ("--experiment-name", cfg.experiment_name),
    ):
        if value:
            command.extend([option, value])
    if cfg.distributed_timeout_minutes is not None:
        command.extend(["--distributed-timeout-minutes", str(cfg.distributed_timeout_minutes)])
    for enabled, flag in (
        (cfg.trust_remote_code, "--trust-remote-code"),
        (cfg.overwrite, "--overwrite"),
        (cfg.not_strict, "--not-strict"),
        (cfg.no_gpu_resource_request, "--no-gpu-resource-request"),
        (cfg.detach, "--detach"),
        (cfg.dry_run, "--submission-dry-run"),
    ):
        if enabled:
            command.append(flag)
    for mount in cfg.mounts:
        command.extend(["--mount", mount])
    command.extend(["--mount", f"{bridge_root}:/opt/Megatron-Bridge"])
    if megatron_root is not None:
        command.extend(["--mount", f"{megatron_root}:/opt/Megatron-Bridge/3rdparty/Megatron-LM"])
    for name in cfg.env_names:
        command.extend(["--env", name])
    command.append(f"--srun-arg=--cpus-per-task={cfg.cpus_per_task}")
    for argument in cfg.srun_args:
        command.append(f"--srun-arg={argument}")
    return command


def launch(cfg: BridgeExportConfig) -> subprocess.CompletedProcess[str]:
    """Run Megatron-Bridge's native launcher and return its result."""
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    return subprocess.run(render_command(cfg), env=environment, text=True, check=True)
