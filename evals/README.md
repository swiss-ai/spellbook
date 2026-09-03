# Evals

Slurm evaluation runners for Megatron-LM and vLLM.

## Files

| File | Purpose |
|------|---------|
| `hf_conversion.py` | hfconverter Stage-2 submission helper |
| `megatron_eval.py` | Megatron dataclass and submission functions |
| `megatron_eval.sh.j2` | Jinja2 template for the Megatron eval sbatch script |
| `vllm_eval.py` | vLLM dataclass, renderer, and submission function |
| `vllm_eval.sh.j2` | Jinja2 template for lm-eval vLLM launches |
| `watcher.py` | One-shot checkpoint checker + `start` command for the self-scheduling watcher |
| `watcher.sh.j2` | Jinja2 template for the self-scheduling watcher sbatch script |

Per-model configs are deployment-specific and should live in the repository that
owns the experiments and checkpoint paths, rather than in Spellbook itself.

## vLLM

> **Status:** The vLLM evaluation path is not correctly implemented end to end
> and is not production-ready. Existing configuration and rendering support
> should not be interpreted as a working or validated evaluator.

```python
from evals.hf_conversion import HFConversionConfig, submit as submit_conversion
from evals.vllm_eval import VLLMEvalConfig, submit as submit_eval

hf_model = "/path/to/hf-checkpoint"
conversion_job = submit_conversion(HFConversionConfig(
    hfconverter_root="https://github.com/swiss-ai/hfconverter.git",
    hfconverter_commit="apertus2/main",
    checkpoint_dir="/path/to/torch_dist/iter_0003000",
    output_dir=hf_model,
    tokenizer_dir="/path/to/tokenizer",
    account="infra01",
    partition="normal",
))

submit_eval(VLLMEvalConfig(
    model_name="apertus-8b-step-3000",
    model=hf_model,
    tokenizer="/path/to/tokenizer",
    tasks=["hellaswag", "arc_easy"],
    batch_size="auto",
    max_model_len=8192,
    tensor_parallel_size=1,
    data_parallel_size=4,
    output_dir="/path/to/results",
    account="infra01",
    partition="normal",
    gpus_per_node=4,
    container_edf="apertus2-vllm",
    container_mounts="${SCRATCH}:${SCRATCH},${HOME}:${HOME}",
    srun_extra_args="--network=disable_rdzv_get",
    conversion_job_id=conversion_job or "",
))
```

Git hfconverter sources use locked caches below `${SCRATCH}/tmp` (or `~/.cache/spellbook/tmp`), completed conversions are reused, and `recreate=True` explicitly replaces partial or completed outputs.

The vLLM job starts one `lm_eval` process with `python -m lm_eval run --model
vllm`; lm-eval and vLLM create their own local workers. The adapter's documented
`tensor_parallel_size` and `data_parallel_size` model arguments are supported.
Their product may not exceed `gpus_per_node`. Pipeline and expert parallelism
are not exposed because the lm-eval adapter does not officially support them.

This launcher intentionally supports one Slurm node. It does not wrap lm-eval
in torchrun or start one independent evaluator per GPU. Use `model_args_extra`
for additional vLLM engine arguments and `lm_eval_args_extra` for additional
lm-eval options. Common Megatron-evaluator features are also available:
`cache_requests`, `metadata`, `log_samples`, `write_out`, `wandb_project`,
`hf_home`, `lm_eval_install`, `install_commands`, and `env_vars`.

The user supplies prebuilt converter and vLLM images; Spellbook only submits
conversion and evaluation jobs, writing results below
`<output_dir>/<model_name>`.

### Verified environment

The hfconverter image and 3B Stage-2 conversion were verified on **2026-08-20** with:

- hfconverter: `2e1e94fd60164ed83b7f1f0251eb30c0ca4c67a5` (`apertus2/main`)
- Megatron-LM-MoE fork used for the image smoke test: `60a7102cfddaa1f366fef9ac944b9d962db9553a`
- image: `/iopsstor/scratch/cscs/anowak/images/apertus2-hf.sqsh`
- test checkpoint: `chonk-3b-parameter-collapse-gbs512-row-fan-in-split-fc1-polar-express/iter_0001907`
- result: conversion and `VERIFY_LOAD=1` completed successfully

Treat later hfconverter or Megatron changes as unverified until the conversion
smoke test is repeated.

## Megatron-LM

Megatron evaluation uses
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

### Submit a single checkpoint

```python
from evals.megatron_eval import MegatronEvalConfig, submit

cfg = MegatronEvalConfig(
    model_name="MY_MODEL",
    checkpoint_dir="/path/to/checkpoints",
    tokenizer_model="swiss-ai/Apertus-70B-2509",
    megatron_path="/path/to/Megatron-LM",
    tasks=["hellaswag", "winogrande", "arc_easy", "arc_challenge"],
    tp=2,
    seq_length=8192,
    micro_batch_size=10,
    cache_requests="true",
    metadata={"training_run": "my-run", "checkpoint_step": 3000},
    output_dir="/path/to/eval-results",
    log_samples=True,
    write_out=True,
    account="a139",
    partition="normal",
    nodes=4,
    gpus_per_node=4,
    cpus_per_task=72,
    exclude="nid007277",
    srun_extra_args="--network=disable_rdzv_get",
    env_vars={
        "LM_HARNESS_CACHE_PATH": "/path/to/lm_eval_requests",
    },
    install_commands="""
pip install --upgrade --no-deps "nvidia-cutlass-dsl==4.4.2"
pip install --upgrade --no-deps "quack-kernels[cu13]==0.4.1"
pip install --no-deps "git+https://github.com/andresnowak/sonic-moe.git@7d931fe0f635f9ccfb2d292e09fdd6e72425dab6"
""".strip(),
)

submit(cfg, ckpt_step=3000)
```

### Run multiple lm-eval configurations in one job

Use `LMEvalRunConfig` when several evaluations should share one Slurm
allocation, container setup, Megatron checkout, package installation, and
dataset prefetch:

```python
from evals.megatron_eval import (
    AllocationMode,
    LMEvalRunConfig,
    MegatronEvalConfig,
    submit_evaluations,
)

cfg = MegatronEvalConfig(
    # Shared model, Megatron, Slurm, and container fields...
    model_name="MY_MODEL",
    checkpoint_dir="/path/to/checkpoints",
    tokenizer_model="/path/to/tokenizer",
    megatron_path="/path/to/Megatron-LM",
    eval_runs=[
        LMEvalRunConfig(
            name="3shot",
            tasks=["mmlu"],
            lm_eval_args={"num_fewshot": 3},
            wandb_name="Core tasks — 3-shot",
            wandb_id="my-model-core-3shot",
        ),
        LMEvalRunConfig(
            name="zero-shot",
            tasks=["hellaswag", "arc_easy"],
            lm_eval_args={"num_fewshot": 0, "batch_size": 4},
            env_vars={"DISABLE_MULTIPROC": "1"},
            wandb_name="Core tasks — zero-shot",
        ),
    ],
)
submit_evaluations(
    cfg,
    ckpt_steps=[1000, 2000, 3000],
    checkpoint_mode=AllocationMode.SHARED,
    eval_mode=AllocationMode.SHARED,
)
```

Checkpoint and run grouping are independent:

| `checkpoint_mode` | `eval_mode` | Jobs |
|---|---|---|
| `SHARED` | `SHARED` | One job for the full checkpoint/run matrix |
| `SEPARATE` | `SHARED` | One job per checkpoint, containing all runs |
| `SHARED` | `SEPARATE` | One job per run, containing all checkpoints |
| `SEPARATE` | `SEPARATE` | One job per checkpoint/run pair |

`lm_eval_args` may override top-level lm-eval options for that invocation; model,
model arguments, tasks, output paths, and WandB arguments remain renderer-owned.
Invocations grouped into one job execute sequentially, and distributed parent
shells meet at a shared-file barrier before starting the next command. Results
are written below
`<output_dir>/<model_name>/step_<step>/<run.name>`. `wandb_name` is the display
name in the WandB UI, while `wandb_id` is the stable resume identifier. An
omitted run ID defaults to `<config.wandb_id-or-model_name>-<run.name>`.
Without `eval_runs`, the existing single-run behavior and output path remain
unchanged.

`MegatronEvalConfig.megatron_path` accepts either a local checkout or a Git URL
such as `https://github.com/NVIDIA/Megatron-LM.git`. A URL is cloned once into a
deterministic cache below `${SCRATCH}/tmp/megatron_repos`; concurrent eval jobs
using the same URL share the cached clone. URL caches are refreshed before each
launch. An unpinned URL follows its remote default branch; `megatron_commit` pins
a commit or resolves a refreshed remote branch. Source-specific worktree keys
prevent collisions across repositories. Without `SCRATCH`, caches use a
user-namespaced directory below `${TMPDIR:-/tmp}`. When `container_edf` is
configured, the resolved checkout is copied into each node's container at
`megatron_container_path`, which defaults to `/opt/megatron`; non-container
launches use the source checkout directly.

Set `MegatronEvalConfig.srun_extra_args` to raw flags that should be appended to
the eval `srun` command. For example,
`srun_extra_args="--network=disable_rdzv_get"` disables Slingshot rendezvous
lookup for the containerized step.

`MegatronEvalConfig.install_commands` is inserted as raw shell inside the eval
`srun` shell before `lm_eval` starts. Package installation runs once per node:
local rank 0 performs it while the other local ranks wait for its result. This
avoids concurrent package-manager writes to the node-shared container
filesystem. Use it for per-eval package installs.

Rendered jobs use `set -e` in both the outer sbatch shell and the nested eval
shell, so setup, installation, and evaluation failures stop the job immediately.
Each eval node also creates its process-specific Triton and TorchInductor cache
directories on local `/tmp` storage before importing the model libraries.

Set `MegatronEvalConfig.lm_eval_install` to an lm-evaluation-harness URL or path
to install it before evaluation. By default the renderer preserves the existing
`pip install` command. Set `lm_eval_install_with_python=True` to emit
`python -m pip install` instead, ensuring the package is installed for the same
`python` interpreter that launches `lm_eval`. Because `install_commands` is raw
shell, use `python -m pip` there as well when interpreter consistency is needed.
This install uses the same once-per-node synchronization as `install_commands`.

Set `MegatronEvalConfig.cache_requests="true"` to pass
`--cache_requests true` to lm-eval, and set `LM_HARNESS_CACHE_PATH` in
`env_vars` to choose the request-cache directory. This cache stores constructed
and tokenized evaluation requests, so it can speed up repeated tasks across
checkpoints. Do not share lm-eval's `--use_cache` across checkpoints: that cache
contains model responses, which are specific to the checkpoint that produced
them.

Set `MegatronEvalConfig.exclude` to a Slurm node list such as `"nid007277"` or
`"nid[007277-007279]"`. It is rendered as `#SBATCH --exclude=...` and passed to
the `sbatch` invocation, so it applies whether the generated script is submitted
through `submit()` or run manually with `sbatch`.

`MegatronEvalConfig.seq_length` is passed directly to the Megatron lm-eval
adapter in `--model_args`. Set it to the desired evaluation context length; a
`--seq-length` flag in `extra_args` only configures Megatron after the adapter's
own maximum length has already been initialized.

Set `MegatronEvalConfig.tp` to the checkpoint's tensor-model-parallel size. It
is passed to the Megatron lm-eval adapter inside `--model_args` as `TP=<value>`,
alongside the existing `EP=<value>` setting. Both default to 1. The adapter
currently requires `tp` to be either 1 or equal to `devices`, and does not
support combining `tp > 1` with `ep > 1`.

`MegatronEvalConfig.micro_batch_size` is passed directly to the Megatron
lm-eval adapter inside `--model_args` as `micro_batch_size=<value>`. It controls
the per-rank request batch used for model forwards and is separate from
lm-eval's top-level `batch_size` option. Do not also pass
`--micro-batch-size` through `extra_args`.

Set `MegatronEvalConfig.metadata` to a JSON-serializable mapping to pass it to
lm-eval as the top-level `--metadata` option. The option is omitted when the
mapping is empty.

Set `MegatronEvalConfig.output_dir` to the base directory for lm-eval's JSON
results. Each job writes below
`<output_dir>/<model_name>/step_<ckpt_step>`. When unset, the base directory is
`<submission directory>/evals`. Configured relative paths are also converted to
absolute paths from the directory where the job is rendered and submitted.
When `dataset_prefetch` is enabled for a multi-node job, `output_dir` must be on
storage shared by every node because it carries the global completion status.
`prefetch_timeout_seconds` bounds both the prefetch and peer wait (default: 1800
seconds), so a missing shared path or failed leader cannot hang indefinitely.

Set `MegatronEvalConfig.log_samples=True` to pass `--log_samples` to lm-eval.
This saves per-example inputs and model responses as
`samples_<task>_<timestamp>.jsonl` alongside the aggregate results beneath the
checkpoint output directory. It is disabled by default because sample logs can
consume substantial storage.

Set `MegatronEvalConfig.write_out=True` to pass `--write_out` to lm-eval and
print the prompts for the first few documents. This is a diagnostic option and
is independent of the per-sample files controlled by `log_samples`.

lm-eval command-line options are generated from the Python configuration field
names (`write_out` becomes `--write_out`) rather than being listed individually
in the Jinja template. Fields marked as lm-eval options in `MegatronEvalConfig`
are therefore rendered consistently for both launch modes.

Set `launch_mode="tasks"` to launch one eval Python process per Slurm task
instead of one node task that starts `torchrun`. In task mode,
`ntasks-per-node` is set to `gpus_per_node`, and each task receives Slurm's
`RANK`, `LOCAL_RANK`, and `WORLD_SIZE` environment. `cpus_per_task` defaults to
72 and is applied to both the batch allocation and its `srun` step; set it to
`None` to omit both options. Only local rank 0 installs packages; its sibling
ranks wait before starting `lm_eval`.

### Submit a range of checkpoints

```python
from evals.megatron_eval import RangeMode, submit_range

submit_range(cfg, start=500, end=3500, step=500, mode=RangeMode.PARALLEL)
```

`PARALLEL` submits separate jobs at once. `SEQUENTIAL` adds
`--dependency=singleton` so separate jobs queue behind one another. By default,
checkpoint allocations are `SEPARATE` and configured eval runs are `SHARED`,
preserving one job per checkpoint. Pass `checkpoint_mode=AllocationMode.SHARED`
to execute the checkpoint range sequentially in one allocation; `eval_mode`
controls run grouping independently.

## Self-scheduling watcher

The watcher is a self-scheduling sbatch job that polls for new checkpoints and submits an eval whenever it finds one not yet submitted. It re-queues itself with `--begin=now+Nhour` so the chain runs indefinitely.

### Start

```bash
uv run python -m evals.watcher start \
    --config evals/configs/my_model.py \
    --interval 1
```

A config may expose either a module-level `cfg`, or a reusable
`build_eval_config(model_name)` factory selected with `--model`:

```bash
uv run python -m evals.watcher start \
    --config /path/to/evaluate_models.py \
    --model exact-model-name \
    --consumed-tokens-per-step 2097152 \
    --interval 1
```

`--consumed-tokens-per-step` is optional. When set, the watcher records
`checkpoint_step * consumed_tokens_per_step` as W&B consumed-token metadata.

This renders `evals/<model_name>/watcher.sh` and submits the first job. Submitted step numbers are recorded in `evals/state_files/<model_name>/.submitted_steps` so re-runs are idempotent.

### Stop

```bash
uv run python -m evals.watcher stop \
  --config /path/to/experiments/evals/configs/my_model.py
```

The stop command writes a persistent stop marker before cancelling jobs with the
watcher's stable Slurm job name. The EXIT trap sees that marker and does not
schedule a successor. Running `start` again removes the marker.

### How it works

```
watcher job runs
  └─ reads latest_checkpointed_iteration.txt
  └─ if step not in .submitted_steps → sbatch eval job, record step
  └─ trap EXIT → sbatch --begin=now+Nhour watcher.sh
```

The `trap EXIT` ensures re-scheduling happens even if the eval submission fails,
unless the persistent stop marker has been created by the `stop` command.

## Adding a new model

1. In your experiments repository, create a config module based on the quick-start
   example above and expose its configuration as a module-level `cfg` variable.
2. Start the watcher from the Spellbook project, passing the config path:

```bash
uv run python -m evals.watcher start \
  --config /path/to/experiments/evals/configs/my_model.py
```
