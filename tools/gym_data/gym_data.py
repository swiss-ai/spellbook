"""Render and submit NeMo Gym environment preparation.

Gym needs two things in place before a NeMo-RL run can use it, and neither belongs
in a training launch:

  * a uv venv per configured server (Gym runs each one as its own FastAPI process),
    plus the NemoGym Ray actor venv NeMo-RL hardcodes;
  * a collated dataset, because ``NemoGym.run_rollouts`` routes every row by
    ``row["agent_ref"]["name"]`` and a hand-built prompt jsonl has no such key.

Both are shared by every experiment against the same Gym checkout, need no GPU, and
download on first use. Prepare once with this tool, then point
``NemoRLExperiment.data_path`` / ``validation_data_path`` at the result.
"""

from __future__ import annotations

import dataclasses
import os
import re
import subprocess
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_TEMPLATES_DIR = Path(__file__).parent
_JOB_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclasses.dataclass
class GymDataConfig:
    name: str
    nemo_rl_path: str
    gym_home: str  # <NeMo-RL>/3rdparty/Gym-workspace/Gym
    # Server configs, relative to the Gym checkout. The transitive set is walked, so
    # naming the entry config is enough.
    config_paths: list[str]

    # Persistent locations; keep these stable so runs reuse the prepared environment.
    venv_dir: str = ""  # NEMO_GYM_VENV_DIR
    nemo_rl_venv_dir: str = ""  # NEMO_RL_VENV_DIR
    uv_cache_dir: str = ""
    scratch_root: str = ""  # writable NEMO_GYM_EXTRA_ROOTS entry
    output_dir: str = ""  # where collate writes

    dataset_name: str = ""  # subdirectory of output_dir; "" skips collation
    build_venvs: bool = True
    # A NeMo-RL config carrying an env.nemo_gym block, handed to
    # examples/nemo_gym/prefetch_venvs.py. Generated from config_paths when empty;
    # point it at a hand-written one (upstream ships
    # examples/nemo_gym/prefetch_*_all_envs.yaml) for aliases, thin client stubs and
    # dummy interpolations.
    prefetch_config: str = ""
    # Pack the built venvs into this squashfs image. A venv is tens of thousands of
    # small files, which Lustre handles badly; mount the image back at venv_dir:
    #   mounts = ["<squashfs>:<venv_dir>:sqsh"]
    squashfs: str = ""
    # Stamped onto every collated row. "binary" scores all-or-nothing across a
    # prompt's constraints, which on a base model is usually a flat-zero reward and
    # therefore zero advantage; "fraction" gives partial credit. "" leaves rows alone.
    grading_mode: str = ""
    # "<server dir>:<requirement>" entries installed into that one server venv once Gym
    # has built it; see prepare.apply_venv_overrides for why they cannot be declared.
    venv_overrides: list[str] = dataclasses.field(default_factory=list)
    # Rows to carve off the tail into validation.jsonl. The instruction_following env
    # declares no validation dataset, and upstream NeMo-RL gym recipes point at
    # externally prepared splits, so this is off unless asked for.
    validation_rows: int = 0

    account: str = ""
    partition: str = ""
    reservation: str = ""
    run_time: str = "02:00:00"
    cpus_per_task: int = 72
    container: str = ""
    container_mounts: str = ""
    log_dir: str = dataclasses.field(
        default_factory=lambda: str(
            Path(
                os.environ.get(
                    "SCRATCH",
                    f"/iopsstor/scratch/cscs/{os.environ.get('USER', 'unknown')}",
                )
            )
            / "tmp/spellbook/nemorl/gym_data/slurm_logs"
        )
    )


def _validate(cfg: GymDataConfig) -> None:
    if not _JOB_NAME.match(cfg.name):
        raise ValueError(f"invalid job name: {cfg.name!r}")
    if not cfg.config_paths:
        raise ValueError("config_paths is required")
    missing = [
        field
        for field in (
            "nemo_rl_path",
            "gym_home",
            "venv_dir",
            "nemo_rl_venv_dir",
            "uv_cache_dir",
            "scratch_root",
            "output_dir",
        )
        if not getattr(cfg, field)
    ]
    if missing:
        raise ValueError(f"missing required paths: {', '.join(missing)}")
    if not Path(cfg.gym_home).expanduser().is_dir():
        raise ValueError(
            f"gym_home not found: {cfg.gym_home}\n"
            f"  git -C {cfg.nemo_rl_path} submodule update --init 3rdparty/Gym-workspace/Gym"
        )
    if cfg.validation_rows and not cfg.dataset_name:
        raise ValueError("validation_rows needs dataset_name (there is nothing to split)")
    if cfg.validation_rows < 0:
        raise ValueError("validation_rows must not be negative")
    malformed = [override for override in cfg.venv_overrides if len(override.split(":", 1)) != 2]
    if malformed:
        raise ValueError(
            "venv_overrides entries must be '<server dir>:<requirement>': " + ", ".join(malformed)
        )


def render(cfg: GymDataConfig) -> str:
    """Render the Gym preparation job."""
    _validate(cfg)
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    context = dataclasses.asdict(cfg)
    context["log_dir"] = str(Path(cfg.log_dir).expanduser().resolve())
    # Absolute: the job runs with --container-workdir set to the NeMo-RL checkout.
    context["prepare_script"] = str(_TEMPLATES_DIR / "prepare.py")
    return env.get_template("gym_data.sh.j2").render(context)


def submit(cfg: GymDataConfig) -> str:
    """Render and submit the Gym preparation job."""
    Path(cfg.log_dir).expanduser().mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["sbatch"], input=render(cfg), capture_output=True, text=True, check=True
    )
    job_id = result.stdout.strip().split()[-1]
    print(f"  {cfg.name}: submitted → job {job_id}")
    return job_id


__all__ = ["GymDataConfig", "render", "submit"]
