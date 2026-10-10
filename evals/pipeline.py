"""Let the existing checkpoint watcher submit a model-owned pipeline script."""

from __future__ import annotations

import dataclasses
import os
import subprocess
from pathlib import Path


@dataclasses.dataclass
class CheckpointPipelineConfig:
    model_name: str
    checkpoint_dir: str
    script: str
    account: str
    partition: str
    log_dir: str = "slurm_logs/eval"
    watcher_dir: str = ""
    reservation: str = ""
    qos: str = ""
    exclude: str = ""


def submit(cfg: CheckpointPipelineConfig, step: int, dependency_singleton: bool = False) -> str:
    """Submit the script with checkpoint arguments; it owns the remaining lifecycle."""
    script = Path(cfg.script).expanduser().resolve()
    if not script.is_file():
        raise FileNotFoundError(script)

    log_dir = Path(cfg.log_dir).expanduser().resolve() / cfg.model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "sbatch",
        "--parsable",
        f"--account={cfg.account}",
        f"--partition={cfg.partition}",
        f"--job-name={os.environ.get('SBATCH_JOB_NAME', f'eval_{cfg.model_name}')}",
        f"--output={log_dir}/pipeline-%j.out",
        f"--error={log_dir}/pipeline-%j.err",
    ]

    for name in ("reservation", "qos", "exclude"):
        if value := getattr(cfg, name):
            command.append(f"--{name}={value}")

    if dependency_singleton:
        command.append("--dependency=singleton")

    command += [
        str(script),
        "--step",
        str(step),
        "--checkpoint-dir",
        str(Path(cfg.checkpoint_dir).expanduser().resolve()),
        "--model",
        cfg.model_name,
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=True)
    identifier = result.stdout.strip().split(";")[0]

    if not identifier.isdigit():
        raise ValueError(f"Invalid sbatch job ID: {result.stdout!r}")
    return identifier
