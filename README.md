# spellbook

Declarative experiment runner for Megatron-LM on SLURM.

Spellbook lets you define experiments as Python dataclasses, generate reproducible
job scripts, and submit sweeps without hand-editing shell scripts.

## Agent Guidance

If you are using coding agents in this repository, see [AGENTS.md](AGENTS.md) for
repo-specific instructions and conventions.

## Install

```bash
uv sync
source .venv/bin/activate
```

Optional quality checks:

```bash
uv run ruff check .
uv run ty check
```

## CLI

```bash
python main.py list   experiments/big_moe_speed_ablations/experiment.py
python main.py render experiments/big_moe_speed_ablations/experiment.py
python main.py submit experiments/big_moe_speed_ablations/experiment.py
```

If you prefer explicit uv execution:

```bash
uv run python main.py list experiments/big_moe_speed_ablations/experiment.py
```

Options:
- `--only NAME`: render/submit a single experiment from a sweep.
- `--output-dir DIR`: where generated scripts/CSVs are written (default: `sbatch_scripts`).
- `list --all`: show all experiment fields.
- `list --columns a,b,c`: show selected fields.

### `csv` — export experiments to a CSV file

```bash
# All fields → <experiment_dir>/<sweep_name>.csv
python main.py csv experiments/big_moe_speed_ablations/experiment.py

# Only differing fields → <experiment_dir>/<ClassName>.changed.csv
python main.py csv experiments/big_moe_speed_ablations/experiment.py --changed

# Override output path
python main.py csv experiments/big_moe_speed_ablations/experiment.py --output my.csv
```

Every row includes computed size columns (`total_params_B`, `active_params_B`, `activation_ratio`)
and a `parent_idx` column — the 0-based row index of the experiment that `.change()` was called
on (empty for root/base experiments).

The following fields are always excluded from CSV output: `wandb_project`, `wandb_exp_name`,
`tensorboard_dir`, `save`, `load`.

## `.env` support

Spellbook loads `.env` automatically in `main.py` startup using `python-dotenv`.

Behavior:
- `.env` is optional.
- Existing exported environment variables are not overwritten.
- `.env` is gitignored in this repo.

Example:

```bash
WANDB_API_KEY=...
HF_TOKEN=...
MEGATRON_PATH=/path/to/Megatron-LM
DATA_PATH=/path/to/data
TOKENIZER_MODEL=/path/to/tokenizer.model
CHECKPOINT_SAVE=/path/to/checkpoints
SLURM_ACCOUNT=a139
SLURM_PARTITION=normal
CONTAINER_EDF=apertus2-alps4-temp
CONTAINER_MOUNTS=${SCRATCH}:${SCRATCH},${HOME}:${HOME},/capstor:/capstor,/iopsstor:/iopsstor
```

Recommended starting `.env` for this repository:

```bash
SLURM_ACCOUNT=a139
SLURM_PARTITION=normal
CONTAINER_EDF=apertus2-alps4-temp
CONTAINER_MOUNTS=${SCRATCH}:${SCRATCH},${HOME}:${HOME},/capstor:/capstor,/iopsstor:/iopsstor
MEGATRON_PATH=$HOME/open_source/Megatron-LM-upstream
TOKENIZER_MODEL=swiss-ai/Apertus-70B-2509
```

## Core concepts

- `Experiment`: base dataclass with `change()`, `sweep()`, `diff()`, `to_dict()`.
- `MegatronExperiment`: rich experiment schema plus Megatron CLI flag translation.
- `Sweep`: named collection of experiments rendered/submitted together.
- `SlurmBackend`: renders Jinja templates and submits via `sbatch` or runs via `srun`.

## Minimal experiment file

An experiment module must expose a top-level `sweep` variable.

```python
from spellbook.backends import SlurmBackend
from spellbook.core import Sweep
from spellbook.megatron import MegatronExperiment

BASE = MegatronExperiment(
    name="my-base",
    megatron_path="/path/to/megatron",
    data_path="/path/to/data",
    tokenizer_model="/path/to/tokenizer.model",
    save="/path/to/checkpoints",
    num_layers=24,
    hidden_size=4096,
    ffn_hidden_size=11008,
    num_attention_heads=32,
    seq_length=4096,
    vocab_size=131072,
    tp=2,
    pp=2,
    ep=1,
    num_gpus=16,
    mbs=1,
    gbs=128,
    train_tokens=10_000_000,
    lr=3e-4,
    min_lr=3e-5,
    wandb_project="spellbook-demo",
    install_commands="""
pip install --upgrade --no-deps "nvidia-cutlass-dsl==4.4.2"
pip install --upgrade --no-deps "quack-kernels[cu13]==0.4.1"
""".strip(),
)

backend = SlurmBackend(
    account="a139",
    partition="normal",
    nodes=None,          # auto-derive from experiment.num_gpus / gpus_per_node
    gpus_per_node=4,
    run_time="01:00:00",
    extra={
        "container_edf": "apertus2-alps4-temp",
        "container_mounts": "${SCRATCH}:${SCRATCH},${HOME}:${HOME}",
    },
)

sweep = Sweep(
    name="demo-sweep",
    experiments=[
        BASE,
        BASE.change("my-tp4", tp=4, pp=1),
    ],
    backend=backend,
)
```

## Sweep helpers

Use convenience constructors from `spellbook.core`:

```python
from spellbook.core import sweep_axis, sweep_grid

# one-axis
sweep = sweep_axis(BASE, "tp", [1, 2, 4], backend=backend, name="tp-sweep")

# grid
sweep = sweep_grid(
    BASE,
    {"tp": [1, 2, 4], "ep": [4, 8]},
    backend=backend,
    name="tp-ep-grid",
)
```

## Backend behavior

`SlurmBackend` modes:
- default: render `slurm.sh.j2` and submit with `sbatch`.
- `launch_mode="tasks"`: render `slurm_tasks.sh.j2`, request `ntasks-per-node=gpus_per_node`, and run the training script directly once per Slurm task instead of using `torchrun`.
- `srun_job_id` set: render `srun.sh.j2` and execute script with `bash` in an existing allocation.
- `mem_estimator=True`: render `mem_estimator.sh.j2`.

Important fields:
- `extra`: generic Jinja template context (for example `container_edf`, `container_mounts`).
- In `launch_mode="tasks"`, `extra` may also include `cpus_per_task`, `mem`, `no_requeue`, or raw `sbatch_extra_lines`.
- `srun_extra_args`: extra raw flags inserted into every `srun` command. If this includes `--network=VALUE`, Spellbook also exports `SLURM_NETWORK=VALUE` before `srun` so the step inherits the same network setting.
- `reservation`: added to sbatch header and sbatch invocation.
- `MegatronExperiment.pre_launch_commands`: raw shell commands inserted into `slurm.sh.j2` before the main training `srun`. Use this for setup that needs to run once per job, including a separate one-task setup `srun`.
- `MegatronExperiment.install_commands`: raw shell commands inserted inside `slurm.sh.j2` before data path setup and training launch. Use this for per-experiment package installation.
- `MegatronEvalConfig.install_commands`: raw shell commands run once per node inside the eval `srun` shell before `lm_eval` starts; sibling ranks wait for the local install to finish. Use this for per-eval package installation.

`srun_extra_args` is not the same as `extra`.

## Locking experiments

Once you are happy with a config, call `.lock()` to freeze it:

```python
MY_EXP = BASE.change("MY_EXP", tp=4, lr=3e-4).lock()
```

**First call** (no lock file yet): writes `locks/MY_EXP.lock.yaml` with the full config and a timestamp.

**Subsequent calls** (lock file exists): validates the current config against the file and raises an error if anything differs:

```
RuntimeError: Experiment 'MY_EXP' differs from its lock file (locks/MY_EXP.lock.yaml):
  lr: locked=0.0003  current=0.001
```

**To change a locked experiment**: delete `locks/MY_EXP.lock.yaml`, update the config, and run the experiment file again — a fresh lock is written automatically.

Lock files live at `locks/<name>.lock.yaml` (same level as `sbatch_scripts/`). Commit them to git so drift is caught in code review.

## Output artifacts

`render` and `submit` create:
- one script per experiment at `sbatch_scripts/<sweep_name>/<experiment_name>.sh`

Use `csv` to export a CSV alongside the experiment definition (see the `csv` section above).

## Typical workflow

```bash
# 1) Inspect sweep variants and changed fields
python main.py list experiments/big_moe_speed_ablations/experiment.py

# 2) Render scripts without submitting
python main.py render experiments/big_moe_speed_ablations/experiment.py

# 3) Submit one variant first (sanity check)
python main.py submit experiments/big_moe_speed_ablations/experiment.py --only DEEPSEEK_V3_BASE

# 4) Submit full sweep
python main.py submit experiments/big_moe_speed_ablations/experiment.py
```

Running inside an existing allocation:

- Set `srun_job_id` on `SlurmBackend`.
- Render or submit as usual.
- Spellbook will use the `srun.sh.j2` path and execute scripts with `bash`.

## Megatron flag mapping

Most dataclass fields map automatically:
- `snake_case` field name -> `--kebab-case` Megatron CLI flag.
- Booleans: `True` emits bare flag, `False` omitted.
- Lists emit space-joined values, except `moe_layer_freq`.
- `moe_layer_freq` emits compact bracketed form with no spaces (for example `[1,1,1]`).
- Rendered scripts keep each flag and its value on the same line.
- Aliases: `tp`, `pp`, `ep`, `etp`, `cp`, `vpp`, `mbs`, `gbs` map to canonical Megatron flags.
- Special: `train_tokens` becomes `--train-samples train_tokens // seq_length`.

## Repository layout

```text
spellbook/
  backends/
    slurm_megatron/
      slurm.py
      slurm.sh.j2
      srun.sh.j2
      mem_estimator.sh.j2
  core/
    experiment.py
    sweep.py
  megatron/
    experiment.py
    flags.py
experiments/
  big_moe_speed_ablations/
  big_moe_speed_ablations_example/
main.py
```

## Development

```bash
uv sync
uv run ruff check .
uv run ty check
```
