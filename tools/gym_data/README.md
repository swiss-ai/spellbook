# NeMo Gym environment preparation

Prepares a Gym environment once so training runs only have to read it:

- a uv venv per configured server (Gym runs each as its own FastAPI process), plus
  the NemoGym Ray actor venv NeMo-RL hardcodes;
- the collated dataset, because `NemoGym.run_rollouts` routes every row by
  `row["agent_ref"]["name"]` — a hand-built prompt jsonl fails with `KeyError`.

No GPU, one node. Every step is skipped when its output already exists.

```python
from tools.gym_data import GymDataConfig, submit

NEMO_RL = "/users/<you>/open_source/Nemo-RL"
submit(
    GymDataConfig(
        name="gym-prep-instruction-following",
        nemo_rl_path=NEMO_RL,
        gym_home=f"{NEMO_RL}/3rdparty/Gym-workspace/Gym",
        config_paths=[
            "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml",
            "resources_servers/instruction_following/configs/instruction_following.yaml",
        ],
        venv_dir="$SCRATCH/tmp/nemo-gym-venvs",
        nemo_rl_venv_dir="$SCRATCH/tmp/nemo-rl-venvs",
        uv_cache_dir="$SCRATCH/tmp/uv-cache-nemo-gym",
        scratch_root="$SCRATCH/tmp/nemo-gym-root",
        output_dir="$SCRATCH/tmp/nemo-gym-data-collated",
        dataset_name="instruction_following",
        account="infra01",
        partition="normal",
        container="<edf>",
    )
)
```

Then point the experiment at the result:

```python
NemoRLExperiment(
    gym_venv_dir=...,
    nemo_rl_venv_dir=...,
    gym_uv_cache_dir=...,
    gym_scratch_root=...,
    data_path=".../instruction_following/train.jsonl",
    validation_data_path=".../instruction_following/validation.jsonl",
)
```

`SlurmNemoRLBackend` no longer builds venvs or touches datasets; it exports the
paths and fails fast if the actor venv is missing.

## Venvs come from NeMo-RL's own prefetch

`prepare.py` calls `examples/nemo_gym/prefetch_venvs.py`, which replays
`NemoGym._spinup(dry_run=True)`. That is the only way `_inherit_from` aliases, thin
client stubs and `local_vllm_model` servers resolve — they cannot be reproduced by
running `uv venv` per server directory.

It takes a NeMo-RL config with an `env.nemo_gym` block, not Gym server configs. One
is generated from `config_paths`; set `prefetch_config` to use a hand-written one
instead. Upstream ships `examples/nemo_gym/prefetch_{super,ultra}_all_envs.yaml`,
which list ~40 environments with the dummy interpolation values the dry run needs.

## squashfs

Three Gym servers come to ~148k files over 5.6 GB, which is the small-file pattern
Lustre handles worst (https://docs.cscs.ch/guides/storage/). Set `squashfs` to pack
`venv_dir` into one image, then mount it back **at the path it was built at** — the
venvs contain editable `.pth` files holding absolute paths:

```python
SlurmNemoRLBackend(
    container_mounts="...,/iopsstor/.../gym-venvs.sqsh:/iopsstor/.../nemo-gym-venvs:sqsh",
)
```

The `:sqsh` suffix is the CSCS container engine's squashfs mount
(https://docs.cscs.ch/software/container-engine/run/).

## grading_mode and validation_rows

Both default to off, because neither is a Gym concept.

`grading_mode` is stamped onto every row when set. `binary` scores a prompt's
constraints all-or-nothing, which on a base model is usually a flat-zero reward and
therefore zero advantage; `fraction` gives partial credit.

`validation_rows` carves that many rows off the tail into `validation.jsonl`. The
`instruction_following` env declares no validation dataset — `gym dataset collate`
only has `train_preparation` and `example_validation` modes — and upstream NeMo-RL
gym recipes point at externally prepared splits. Ask for it explicitly or not at all.

Both read from `collated.jsonl`, a copy of collate's output taken once, so a re-run
can never re-split an already-split dataset.
