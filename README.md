# megatron-zoo

Python framework for managing Megatron-LM training experiments on SLURM clusters.
Replaces hand-written sbatch scripts with composable Python experiment files.

## Installation

```bash
uv sync
```

## Concepts

| Concept | Description |
|---|---|
| **Cluster preset** | SLURM account, partition, GPU count, container mounts (`zoo/presets/clusters/`) |
| **Model preset** | Megatron architecture flags for a specific model (`zoo/presets/models/`) |
| **Runtime preset** | Container EDF, Megatron path, mcore version (`zoo/presets/runtimes/`) |
| **Sweep** | SLURM job array over a list of config variants (parallelism search, HP search) |
| **ExperimentChain** | Sequential training steps with automatic checkpoint passing |
| **Project** | Registry that runs modules in dependency order |

Config merge order (lowest → highest priority):
```
runtime preset → model preset → chain/sweep base kwargs → per-step overrides
```

## Quick start

```bash
# dry-run: renders sbatch scripts, prints configs, does not submit
python main.py run experiments/qwen3_33b_ablation.py --dry-run

# submit for real (prompts "Submit? [y/N]" before each module)
python main.py run experiments/qwen3_33b_ablation.py

# list all modules in an experiment file
python main.py list experiments/qwen3_33b_ablation.py

# check status of submitted chains
python main.py status
python main.py status qwen3-33b-lr-schedule
```

## Writing an experiment

```python
from zoo.modules.chain import ExperimentChain, Step
from zoo.modules.sweep import Sweep, variant_grid
from zoo.project import Project

CLUSTER = "clariden"
MODEL   = "qwen3-30b-a3b"
RUNTIME = "ngc-25-11-nemo"

# --- Sweep: try multiple parallelism configs as a SLURM job array ---
sweep = Sweep(
    name="my-tp-sweep",
    cluster=CLUSTER, model=MODEL, runtime=RUNTIME,
    nodes=4, run_time="00:30:00",
    name_prefix="G-", name_keys=["tp", "pp", "ep"],
    # base config shared by all variants
    seq_length=4096, mbs=1, gbs=8, train_iters=20,
    data_path="/path/to/data",
    checkpoint_save="/path/to/ckpt/sweep/${variant_name}",
    tensorboard_dir="/path/to/tb/sweep/${variant_name}",
    wandb_project="my-project",
    wandb_exp_name="${variant_name}",
    # grid: 6 combinations
    variants=variant_grid(tp=[1, 2, 4], pp=[1, 2], ep=[4]),
    max_concurrent=4,
)

# --- Chain: sequential warmup → main → cooldown ---
chain = ExperimentChain(
    name="my-training",
    cluster=CLUSTER, model=MODEL, runtime=RUNTIME,
    nodes=16, run_time="03:50:00",
    depends_on=["my-tp-sweep"],          # runs after sweep completes
    tp=2, pp=1, ep=4,
    seq_length=4096, mbs=2, gbs=256,
    data_path="/path/to/data",
    checkpoint_save="/path/to/ckpt/training",
    tensorboard_dir="/path/to/tb/training",
    wandb_project="my-project",
    steps=[
        Step(name="warmup", wandb_exp_name="my-warmup", lr=1e-4,
             train_tokens=1_000_000_000),
        Step(name="main",   wandb_exp_name="my-main",   lr=3e-4,
             train_tokens=50_000_000_000, override_opt_param_scheduler=True),
        Step(name="cooldown", wandb_exp_name="my-cooldown", lr=1e-5,
             train_tokens=5_000_000_000, override_opt_param_scheduler=True),
    ],
)

project = Project(name="my-project", cluster=CLUSTER)
project.add(sweep)
project.add(chain)
```

## Variant grids

```python
from zoo.modules.sweep import variant_grid

# full grid — all combinations
variant_grid(tp=[1, 2, 4], pp=[1, 2], ep=[4, 8])

# partial grid — concatenate sub-grids
variant_grid(tp=[1, 2, 4], pp=[1], ep=[4, 8]) + \
variant_grid(tp=[1, 2],    pp=[2], ep=[8])

# manual names — set variant_name in the dict to override auto-naming
[
    dict(tp=1, pp=1, ep=4, variant_name="baseline"),
    dict(tp=2, pp=1, ep=4, variant_name="tp2"),
    *variant_grid(tp=[4], pp=[1, 2], ep=[4]),   # auto-named
]
```

## Presets

### Cluster
`zoo/presets/clusters/<name>.yaml` — SLURM account, partition, GPU count, container mounts, ENV_VARS.

### Model
`zoo/presets/models/<name>.yaml` — Megatron architecture flags (MODEL_ARGS).

### Runtime
`zoo/presets/runtimes/<name>.yaml` — container EDF, Megatron path, mcore version.
Copy `template.yaml` to add a new one.

Available runtimes:
- `ngc-25-11-nemo` — NGC 25.11 NeMo container, Megatron-LM mcore 0.15

## Secrets

Create a `.env` file at the project root (gitignored):

```bash
WANDB_API_KEY=...
HF_TOKEN=...
```

These are loaded automatically on startup and forwarded into the SLURM container via `srun --export=`.

## Project layout

```
zoo/
  backends/
    megatron.py      # alias → --flag-name translation
    slurm.py         # JobSpec → sbatch script + submission
  modules/
    chain.py         # ExperimentChain + Step
    sweep.py         # Sweep + variant_grid
  presets/
    clusters/        # clariden.yaml, …
    models/          # qwen3-30b-a3b.yaml, …
    runtimes/        # ngc-25-11-nemo.yaml, template.yaml
  config.py          # YAML loading, merging, template resolution
  module.py          # Module, JobSpec, ModuleResult base classes
  project.py         # Project (dependency-ordered runner)
experiments/         # experiment files (one per project/ablation)
main.py              # CLI entry point
```
