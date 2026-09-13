"""Step 1 of the NeMo-RL Gym example: prepare the environment, once.

Builds the Gym server venvs and the NemoGym Ray actor venv through NeMo-RL's own
prefetch path, collates the dataset, and optionally packs the venvs into a squashfs
image. No GPU. Every step is skipped when its output already exists.

    python experiments/nemorl_gym_example/prepare_data.py

Then run experiment.py, which only reads the result.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Running a script by path puts its own directory on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.gym_data import GymDataConfig, submit

SCRATCH = os.environ.get("SCRATCH", "/iopsstor/scratch/cscs/anowak")
NEMO_RL = "/users/anowak/open_source/Nemo-RL"
PREP = f"{SCRATCH}/tmp/spellbook-nemorl-gym"

config = GymDataConfig(
    name="gym-prep-instruction-following",
    nemo_rl_path=NEMO_RL,
    gym_home=f"{NEMO_RL}/3rdparty/Gym-workspace/Gym",
    # The transitive set is walked, so the entry configs are enough. Swap in a
    # hand-written prefetch_config to cover many environments at once -- upstream
    # ships examples/nemo_gym/prefetch_{super,ultra}_all_envs.yaml.
    config_paths=[
        "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml",
        "resources_servers/instruction_following/configs/instruction_following.yaml",
    ],
    venv_dir=f"{PREP}/gym-venvs",
    nemo_rl_venv_dir=f"{PREP}/nemo-rl-venvs",
    uv_cache_dir=f"{PREP}/uv-cache",
    scratch_root=f"{PREP}/gym-root",
    output_dir=f"{PREP}/data",
    dataset_name="instruction_following",
    # instruction_following declares no validation set and defaults to binary
    # grading, which scores a whole GRPO group zero on a base model. Both are
    # off unless asked for; see tools/gym_data/README.md.
    grading_mode="fraction",
    validation_rows=64,
    # Optional. Pack ~120k venv files into one image, then mount it back at
    # venv_dir in experiment.py.
    squashfs=f"{PREP}/gym-venvs.sqsh",
    account="infra01",
    partition="normal",
    reservation="SD-69241-apertus-1-5-0",
    container=f"{SCRATCH}/img/nemorl_alps7_te217_groupgemm_emrgopt_deepgemm_fla52_uccl.toml",
    container_mounts="/iopsstor:/iopsstor,/capstor:/capstor,/users/anowak:/users/anowak",
    log_dir=f"{PREP}/slurm_logs",
)

if __name__ == "__main__":
    submit(config)
