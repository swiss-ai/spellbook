# spellbook

Declarative experiment runner for Megatron-LM on SLURM.

Spellbook lets you define experiments as Python dataclasses, generate reproducible
job scripts, and submit sweeps without hand-editing shell scripts.

## Agent Guidance

If you are using coding agents in this repository, see [AGENTS.md](AGENTS.md) for
repo-specific instructions and conventions.

See [CHANGELOG.md](CHANGELOG.md) for the current main-merge summary and
operational notes.

## Install

```bash
uv sync
source .venv/bin/activate
```

Optional quality checks:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
```

## CLI

```bash
python main.py list   experiments/big_moe_speed_ablations/experiment.py
python main.py render experiments/big_moe_speed_ablations/experiment.py
python main.py submit experiments/big_moe_speed_ablations/experiment.py
python main.py report experiments/big_moe_speed_ablations/experiment.py
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

## Declarative reports

Spellbook report definitions describe data sources, model selection, parameter
metadata, and requested plots without downloading histories or rendering at import
time. A training report normally lives beside its experiment `sweep`:

```python
from spellbook.reporting import (
    ChinchillaScalingLaw,
    EndpointScalingLaw,
    LearningRateBowl,
    LossAlignment,
    Models,
    Report,
    WandbGroup,
)

report = Report(
    name="kda-scaling-ladder-training",
    source=WandbGroup("apertus/scaling-ladder-kda-v2"),
    select=Models(
        names=["1.5b", "5b", "22b"],
        exclude=["router-*"],
    ),
    plots=[
        LearningRateBowl(group_by="gbs", learning_rate="matrix_lr"),
        LossAlignment(
            axes=["tokens", "flops", "lr_cooldown"],
            xlim={"tokens": (20, None)},
            ylim=(None, 3.0),
        ),
        EndpointScalingLaw(x=["total_params", "active_params", "tokens", "flops"]),
        ChinchillaScalingLaw(parameter="active_params"),
    ],
)
```

When the module also defines `sweep`, the report loader selects those experiments
before the WandB history fetch and records their total/active parameter
metadata. Evaluation semantics remain in the evaluation file:

```python
from spellbook.reporting import EvalMacro, Report, TaskHeatmap, WandbGroup

eval_report = Report(
    name="kda-scaling-ladder-evals",
    source=WandbGroup(
        "apertus/apertus2-scaling-ladder-evals",
        group="apertus2-kda-scaling-ladder-evals",
    ),
    plots=[
        EvalMacro(task_groups=EVAL_GROUPS),
        TaskHeatmap(
            task_groups={
                "English": ENGLISH_EVALS,
                "Multilingual": MULTILINGUAL_EVALS,
                "Coding": CODING_EVALS,
            },
            grouped=True,
            task_labels={"custom_benchmark": "Custom Benchmark"},
            namespace_labels={"heldout": "held-out"},
            metric_labels={
                "special::benchmark/score": "Special benchmark label",
            },
        ),
    ],
)
```

Render SVG and PNG figures, CSV histories, and fit metadata with:

```bash
uv run python main.py report experiments/path/experiment.py
uv run python main.py report evals/path/evaluate.py --variable eval_report --output-dir reports
```

Use `--refresh` to ignore cached WandB histories. Use `--plan` to validate and
print the resolved report without contacting WandB.

`Models` supports exact `names`, config-field `where` values or predicates, and
glob-style `exclude` patterns. Training and evaluation reports stay in their
respective definition files.

`WandbGroup.runs` can restrict a report to exact run IDs or names, which is useful
when one logical model trajectory consists of known continuation pieces.
`WandbGroup.run_namespaces` maps labels to run-name globs so evaluation suites that
reuse metric keys remain separate. For example, an eval report can reference both
`0shot::arc_easy/acc_norm` and `10shot::arc_easy/acc_norm`; generated panel titles
show these as `0shot — arc_easy/acc_norm` and `10shot — arc_easy/acc_norm`.

`MegatronExperiment.parameter_counts()` supports standard attention, MLA, and KDA,
including mixed KDA layer patterns and latent-MoE routed inputs/projections. A
specialized experiment can override that method for different parameter accounting.

Reports use one shared Matplotlib style, EMA `0.9` by default, four-decimal endpoint
labels, and winner stars. `MetricCurves.better` accepts `"min"`, `"max"`, or
`"abs_zero"`; the last option is intended for routing-violation metrics where
closer to zero is better. Resumed runs that match one selected model are stitched
on consumed tokens (or optimizer step when tokens are unavailable), with the later
piece winning overlapping points.

Chance-adjusted evaluation metrics and their `chance_baselines` use fractions in
`[0, 1]`; `EvalMacro.value_scale` and `EvalTrajectories.value_scale` default to
`100` for percentage display. Endpoint power laws require at least five models so
the three-parameter fit has more than one residual degree of freedom. The
Chinchilla-style law fits `L(N,D) = E + A N^-alpha + B D^-beta` from sampled
training trajectories, using total or active parameters for `N` and consumed
tokens for `D`. A minimum token cutoff can exclude optimizer warmup, and the
figure and fit JSON warn when either exponent reaches a fit bound.

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
    nodes=None,  # auto-derive from experiment.num_gpus / gpus_per_node
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

## Backends

`SlurmBackend` supports four launch shapes:

| Configuration | Launch |
|---|---|
| default | One `torchrun` launcher per node in a batch job |
| `launch_mode="tasks"` | One Slurm task per GPU, with local NUMA binding |
| `srun_job_id` | Run inside an existing allocation |
| `mem_estimator` or `theoretical_memory` | Run a single-task memory estimate |

Megatron sources may be local checkouts or Git URLs. Remote sources use locked,
commit-specific worktrees under `${SCRATCH}/tmp`; container jobs copy the selected
checkout to `/opt/megatron`. Dataset selection follows `data_path`, then
`data_args_path`, then discovery from `base_data_path`.

Common reliability controls are available across training backends:

- `vetnode=True` checks every GPU before launch and can exclude failed nodes before retrying.
- `kernel_cache=True` reuses Triton and TorchInductor caches; optional warmup jobs populate them.
- `auto_requeue=True` continues long jobs and stops when training completion is detected.

`reservation` and `qos` add `--reservation` and `--qos` to the rendered `#SBATCH`
header and to the `sbatch` call, and are also carried into auto-requeue resubmissions
and memory-estimator `srun` calls. Both default to empty, meaning the flag is omitted.
The same fields exist on `SlurmNemoRLBackend`, the eval configs, and the `tools/`
submitters.

`extra` supplies template values such as container settings. `srun_extra_args` is
reserved for flags passed to `srun`.

### NeMo-RL

`SlurmNemoRLBackend` renders a Hydra recipe and starts one Ray daemon per node. The
driver stays with the head process while Ray places policy workers across the allocation.
Recipes may inherit from `base_config` and apply final `recipe_overrides`.

NeMo Gym environments and datasets are prepared separately with
[`tools/gym_data`](tools/gym_data/README.md); training only mounts and consumes them.
Scheduler and Ray logs default below `$SCRATCH/tmp/spellbook/nemorl`.

Megatron evaluation, checkpoint conversion, and the experimental vLLM path are
documented in [`evals/README.md`](evals/README.md).

### NeMo-RL example

[`experiments/nemorl_gym_example/`](experiments/nemorl_gym_example/README.md) is a worked two-node GRPO run through NeMo Gym: `prepare_data.py` builds the Gym venvs and dataset once, `experiment.py` trains against them.

## Container images

[`containers/`](containers/) holds container definitions laid out as
[`eth-cscs/alps-extended-images`](https://github.com/eth-cscs/alps-extended-images)
application images, so each directory can be copied straight into that repository's
`Alps-Images/apps/`.

- [`containers/nemo-rl`](containers/nemo-rl/README.md): the NeMo-RL image for
  `spellbook.nemorl`. It carries the megachonk kernel pin set (TransformerEngine 2.17,
  UCCL-EP, DeepGEMM, grouped_gemm, Emerging-Optimizers, flash-linear-attention 0.5.2)
  plus the full NeMo-RL runtime stack, so `NemoRLExperiment.overlay_paths` can be empty.
  NeMo-RL, Megatron-Bridge and Megatron-LM are not vendored; bind-mount those checkouts.

The upstream pipeline appends its own Alps revision to the base image named in
`profile.env`, so published tags follow that pipeline rather than this repository.

## Tools

Standalone operational launchers live under [`tools/`](tools/). They are separate from training experiments and evaluation pipelines.

### Dynamic inference HTTP server

[`tools/inference/megatron_server.py`](tools/inference/megatron_server.py) is the Megatron backend for dynamic text generation. It uses one Slurm task per GPU (no `torchrun`), binds each task's CPU and memory to the matching NUMA node, installs Quart and Hypercorn once per node inside the containerized step, and enables dynamic batching and CUDA graphs. Like experiment launches, it accepts a local or Git-backed Megatron checkout, supports commit pinning, and replaces the Megatron copy inside the container.

```python
from tools.inference.megatron_server import MegatronServerConfig, submit

submit(
    MegatronServerConfig(
        name="my-model-step-3000",
        checkpoint="/path/to/checkpoints/my-model",
        ckpt_step=3000,
        tokenizer_model="/path/to/tokenizer",
        megatron_path="/path/to/Megatron-LM",  # Or a Git URL.
        megatron_commit="<commit-or-branch>",
        expert_parallel_size=8,
        nodes=2,
        gpus_per_node=4,
        account="infra01",
        partition="normal",
    )
)
```

Interact with it using curl or the dependency-free Python client:

```bash
export INFERENCE_SERVER_URL="http://${COMPUTE_HOST}:5000"
python -m tools.inference.client health
python -m tools.inference.client interactive
```

See [`tools/inference/examples/megatron_server.py`](tools/inference/examples/megatron_server.py) for a complete placeholder configuration and [`tools/inference/README.md`](tools/inference/README.md) for container, tunnel, chat, and curl examples. The server has no built-in authentication.

### Megatron checkpoint merge

[`tools/merge/checkpoint.py`](tools/merge/checkpoint.py) is the Megatron backend for the distributed `tools/checkpoint/merge.py` added to Megatron-LM-MoE in commit [`38f36a8`](https://github.com/andresnowak/Megatron-LM-MoE/commit/38f36a887d5be7ea3c4932d5a958ffc9b0dc87a8). It supports mean and linear-decay weight-space merges, one checkpoint root with multiple steps or multiple checkpoint paths, and NUMA-bound Slurm workers.

```python
from tools.merge import MegatronCheckpointMergeConfig, submit

submit(
    MegatronCheckpointMergeConfig(
        name="model-steps-1000-2000",
        checkpoints=["/path/to/checkpoints/model"],
        checkpoint_steps=[1000, 2000],
        output="/path/to/checkpoints/model-merged",
        megatron_path="/path/to/Megatron-LM",
        workers_per_node=4,
        account="infra01",
        partition="normal",
    )
)
```

See [`tools/merge/README.md`](tools/merge/README.md) for container and backend options.

### NeMo Gym environment preparation

[`tools/gym_data/`](tools/gym_data/README.md) prepares a Gym environment once — the per-server uv venvs, the NemoGym Ray actor venv, and the collated dataset — on CPU, outside any training job. `SlurmNemoRLBackend` then only points at the result and fails fast if it is missing, so a training run never builds venvs or rewrites datasets. Optional `grading_mode` stamping and `validation_rows` splitting are off by default and always derive from an immutable copy of what `gym dataset collate` produced. Preparation logs default below `$SCRATCH/tmp/spellbook/nemorl/gym_data`.

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
tools/
  inference/
    megatron_server.py
    client.py
  merge/
    checkpoint.py
experiments/
  big_moe_speed_ablations/
  big_moe_speed_ablations_example/
  nemorl_gym_example/
    prepare_data.py
    experiment.py
containers/
  nemo-rl/
    Containerfile
    profile.env
    ci.yaml
    tests/
main.py
```

## Development

```bash
uv sync
uv run ruff format --check .
uv run python -m unittest discover -v
uv run ruff check .
uv run ty check
```

Python formatting uses Ruff with a 100-character line length.

The optional Megatron-dependent memory-estimator subtree is excluded from local
Ruff and ty checks because its imports are supplied only by the runtime Megatron
environment.
