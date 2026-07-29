# Evals

Megatron-LM evaluation runner using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## Files

| File | Purpose |
|------|---------|
| `megatron_eval.py` | Core dataclass (`MegatronEvalConfig`) and submission functions |
| `megatron_eval.sh.j2` | Jinja2 template for the eval sbatch script |
| `watcher.py` | One-shot checkpoint checker + `start` command for the self-scheduling watcher |
| `watcher.sh.j2` | Jinja2 template for the self-scheduling watcher sbatch script |
| `configs/` | Per-model eval configs |

## Quick start

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
    exclude="nid007277",
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

`MegatronEvalConfig.megatron_path` accepts either a local checkout or a Git URL
such as `https://github.com/NVIDIA/Megatron-LM.git`. A URL is cloned once into a
deterministic cache below `${SCRATCH}/tmp/megatron_repos`; concurrent eval jobs
using the same URL share the cached clone. Set `megatron_commit` to evaluate
against a specific commit from either a local checkout or a cloned URL.

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
`RANK`, `LOCAL_RANK`, and `WORLD_SIZE` environment. Only local rank 0 installs
packages; its sibling ranks wait before starting `lm_eval`.

### Submit a range of checkpoints

```python
from evals.megatron_eval import RangeMode, submit_range

submit_range(cfg, start=500, end=3500, step=500, mode=RangeMode.PARALLEL)
```

`PARALLEL` submits all jobs at once. `SEQUENTIAL` adds `--dependency=singleton` so they queue one after another.

## Self-scheduling watcher

The watcher is a self-scheduling sbatch job that polls for new checkpoints and submits an eval whenever it finds one not yet submitted. It re-queues itself with `--begin=now+Nhour` so the chain runs indefinitely.

### Start

```bash
uv run python -m evals.watcher start \
    --config evals/configs/my_model.py \
    --interval 1
```

This renders `evals/<model_name>/watcher.sh` and submits the first job. Submitted step numbers are recorded in `evals/state_files/<model_name>/.submitted_steps` so re-runs are idempotent.

### Stop

```bash
scancel <job_id>   # printed when you run start
```

Once the running job is cancelled, no future job is scheduled and the chain ends.

### How it works

```
watcher job runs
  └─ reads latest_checkpointed_iteration.txt
  └─ if step not in .submitted_steps → sbatch eval job, record step
  └─ trap EXIT → sbatch --begin=now+Nhour watcher.sh  (always re-schedules)
```

The `trap EXIT` ensures re-scheduling happens even if the eval submission fails.

## Adding a new model

1. Copy an existing config and edit it:

```bash
cp evals/configs/small_100b_cross_doc.py evals/configs/my_model.py
```

2. Update the `cfg` variable in the new file.

3. Start the watcher:

```bash
uv run python -m evals.watcher start --config evals/configs/my_model.py
```
