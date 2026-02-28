# Megatron-Zoo: Python Project Framework — Ideas

## Core Concept

A Python-first framework where a `Project` is the top-level object that owns a collection of
modules. There are two first-class job types:

- **`ExperimentChain`** — sequential runs with checkpoint passing (training)
- **`BenchmarkSweep`** — parallel job array (speed/throughput checks)

Both are modules. The project composes them with other pipeline steps (log parsing, analysis,
checkpoint conversion, etc.).

---

## 1. Project Class

The `Project` is the root container.

```python
project = Project(
    name="qwen3-moe-ablations",
    cluster="clariden",
)
project.add(BenchmarkSweep(name="find-best-tp", ...))
project.add(ExperimentChain(name="qwen3-33b-run", depends_on=["find-best-tp"], ...))
project.add(ParseLogsModule(...))
project.add(AnalysisModule(...))
project.run()
```

Key responsibilities:
- Shared cluster context (loaded from `zoo/presets/clusters/<name>.yaml`)
- Module registry and dependency resolution
- Unified CLI entry point

---

## 2. Module System

Each module has a standard interface:

```python
class Module:
    name: str
    depends_on: list[str]

    def run(self, ctx: Context) -> ModuleResult: ...
    def status(self) -> ModuleStatus: ...    # pending / running / done / failed
```

Modules pass data via a `Context` object that accumulates results as the pipeline progresses.

---

## 3. Configuration: YAML for Static Facts, Python for Decisions

### The split

**YAML presets** hold static facts that rarely change — model architecture and cluster
settings. You add a new model by dropping a YAML file; no Python changes needed.

```yaml
# zoo/presets/models/qwen3-33b.yaml
num_layers: 48
hidden_size: 2048
num_attention_heads: 32
num_query_groups: 4
ffn_hidden_size: 8192
num_experts: 128
moe_ffn_hidden_size: 768
moe_router_topk: 8
seq_length: 4096
tokenizer: Qwen/Qwen3-30B-A3B
extra_flags:
  - --moe-grouped-gemm
  - --swiglu
```

```yaml
# zoo/presets/clusters/clariden.yaml
account: a139
partition: normal
gpus_per_node: 4
container_mounts: "$SCRATCH:$SCRATCH,$HOME:$HOME,/capstor:/capstor"
env:
  CUDA_DEVICE_MAX_CONNECTIONS: "1"
  PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True"
  NCCL_NVLS_ENABLE: "1"
```

**Python** holds experiment decisions: parallelism, training hparams, what to run, in what
order. Any YAML key can be overridden inline — this covers all Megatron flags.

### Two contexts for config resolution

**Standalone `TrainingRun`** — fully self-contained, owns `cluster` and `model`, loads the
YAMLs itself:

```python
run = TrainingRun(
    name="qwen3-33b-ep4-baseline",
    cluster="clariden",       # loads zoo/presets/clusters/clariden.yaml
    model="qwen3-33b",        # loads zoo/presets/models/qwen3-33b.yaml
    tp=1, pp=1, ep=4,
    mbs=1, gbs=512, lr=3e-4,
    train_tokens=50_000_000_000,
    data_path="/capstor/store/.../fineweb-edu",
    checkpoint_save="/cscratch/runs/qwen3-33b-ep4",
)
```

**`Step` inside a chain or sweep** — just a delta record. The parent (`ExperimentChain` or
`BenchmarkSweep`) owns `cluster`, `model`, and the base config. Steps only carry what changes:

```python
Step(name="warmup",   lr=1e-4, train_tokens=1_000_000_000)
Step(name="main",     lr=3e-4, mbs=4, gbs=512, train_tokens=50_000_000_000)
Step(name="cooldown", lr=1e-5, train_tokens=5_000_000_000)
```

### Merge order (lowest → highest priority)

```
cluster YAML  →  model YAML  →  chain/sweep base kwargs  →  per-Step overrides
```

### Validation before submission

- `ep` divides `num_experts` and divides `dp`
- `gbs` is divisible by `mbs × dp`
- exactly one of `train_tokens` / `train_iters` is set
- `pp > 1` implies `num_layers` divisible by `pp`
- checkpoint save path's parent exists and is writable
- all referenced preset names exist as YAML files

---

## 4. ExperimentChain — Sequential Training

```python
chain = ExperimentChain(
    name="qwen3-33b-lr-schedule",
    cluster="clariden",
    model="qwen3-33b",
    # base config — shared by all steps, overridable per-step
    tp=1, pp=1, ep=4,
    mbs=2, gbs=256,
    data_path="/capstor/store/.../fineweb-edu",
    checkpoint_save="/cscratch/runs/qwen3-33b-lr-schedule",
    submission_mode="hybrid",
    runs=[
        Step(name="warmup",   lr=1e-4, train_tokens=1_000_000_000),
        Step(name="main",     lr=3e-4, mbs=4, gbs=512, train_tokens=50_000_000_000),
        Step(name="cooldown", lr=1e-5, train_tokens=5_000_000_000),
    ],
)
```

Checkpoint passing between steps is automatic. Before submission, the framework prints a
diff table so you can verify the resolved config:

```
Chain: qwen3-33b-lr-schedule  (3 steps, mode: hybrid)

Step         lr       mbs   gbs    train_tokens   checkpoint_from
warmup       1e-4     2     256    1.0B           (fresh start)
main         3e-4     4     512    50.0B          ← warmup (latest iter)
cooldown     1e-5     4     512    5.0B           ← main (latest iter)

Scripts will be written to: sbatch_scripts/qwen3-33b-lr-schedule/
Submit? [y/N]
```

### Branching chains

Branch from the same checkpoint into two parallel continuations:

```python
Step(name="high-lr", branch_from="warmup", lr=5e-4, train_tokens=10_000_000_000),
Step(name="low-lr",  branch_from="warmup", lr=5e-5, train_tokens=10_000_000_000),
```

Both get `--dependency=afterok:<warmup_job_id>` and run in parallel.

---

## 5. BenchmarkSweep — Job Arrays

For throughput/speed checks: one `sbatch` submission, N tasks in parallel, each picking its
config by `$SLURM_ARRAY_TASK_ID`. Variants are flat dicts of overrides — any Megatron flag
can vary, not just parallelism.

```python
sweep = BenchmarkSweep(
    name="qwen3-33b-tp-pp-sweep",
    cluster="clariden",
    model="qwen3-33b",
    # base shared by all variants
    mbs=1, gbs=8, train_iters=20,
    variants=[
        dict(tp=1, pp=1, ep=4),
        dict(tp=2, pp=1, ep=4),
        dict(tp=4, pp=1, ep=4),
        dict(tp=1, pp=2, ep=4, recompute_granularity="selective"),
        dict(tp=2, pp=2, ep=4, mbs=2, moe_grouped_gemm=True),
    ],
    max_concurrent=4,   # #SBATCH --array=0-4%4
)
```

### What gets generated

```
sbatch_scripts/qwen3-33b-tp-pp-sweep/
├── array.sh        # #SBATCH --array=0-4%4
└── configs.json    # fully resolved config per task index
```

`array.sh` reads `$SLURM_ARRAY_TASK_ID`, pulls the flags from `configs.json`, passes them
to Megatron.

After all tasks finish: `python main.py analyze qwen3-33b-tp-pp-sweep` scans logs and prints
a ranked TFLOPS table.

---

## 6. Slurm Execution Model

### Submission modes

#### `dependency` — pure Slurm, no daemon

All scripts generated upfront, submitted with `--dependency=afterok` chaining. Slurm handles
sequencing; survives disconnects. Checkpoint path resolved inside the script via bash.
If a step fails, all downstream steps are automatically cancelled.

#### `polling` — Python daemon

Python submits A, polls `squeue`, submits B after A completes. Needs a live tmux session.
Enables smart retry logic and Python-level checkpoint path resolution.

#### `hybrid` (recommended default)

Submit with `--dependency=afterok` (Slurm handles sequencing), plus write `chain_state.json`
so the CLI has visibility at any time:

```json
{
  "chain": "qwen3-33b-lr-schedule",
  "steps": [
    {"name": "warmup",   "job_id": 111, "status": "COMPLETED", "checkpoint": "..."},
    {"name": "main",     "job_id": 222, "status": "RUNNING"},
    {"name": "cooldown", "job_id": 333, "status": "PENDING"}
  ]
}
```

### sbatch scripts are always saved

Written to `sbatch_scripts/<name>/`, never auto-deleted. Full reproducibility, easy manual
resubmit, diff between runs.

### Crash detection

| Signal | How to detect |
|---|---|
| Slurm terminal state | `sacct -j <id> --format=State` → FAILED / OOM / TIMEOUT |
| Log file pattern | scan `.out` for `Traceback`, `CUDA out of memory`, `NCCL error` |
| W&B run status | run state = `crashed` or no heartbeat for N minutes |
| Missing checkpoint | expected checkpoint dir doesn't exist after job ends |

```
Step "warmup" (job 111) FAILED
  Slurm status : FAILED
  Likely cause : CUDA out of memory (found in log line 4821)
  Checkpoint   : NOT FOUND
  Action       : chain halted — fix MBS or recompute settings and resubmit
```

W&B is not used to drive submission decisions — only for linking run URLs in the state file
and pulling final metrics post-chain.

---

## 7. Other Modules

### ParseLogsModule
- Scans `slurm_logs/` for `.out` files
- Extracts TFLOPS, memory stats, param count at a configurable iteration
- Writes `metrics.csv`

### JoinModule
- Merges sweep configs with `metrics.csv` → `ablations_with_metrics.csv`

### AnalysisModule
- Ranks configs by TFLOPS, filters OOM runs
- Rich table or summary CSV; extras: TFLOPS/GB curve, Pareto front

### EvalModule
- Runs lm-evaluation-harness against saved checkpoints
- Combined perf-vs-quality table

### CheckpointConvertModule
- Megatron ↔ HF format conversion
- Accepts source path + target parallelism config

### TokenizationModule
- Wraps `examples/data/megatron_tokenize.py`
- Validates output `.bin`/`.idx` files exist after completion

---

## 8. CLI

```
python main.py list                           # show all modules / chains / sweeps
python main.py run <experiment.py>            # run a project file
python main.py run <experiment.py> --dry-run  # show resolved configs + scripts, don't submit
python main.py status                         # live table: chains, steps, job IDs, W&B links
python main.py status <name>                  # status for one chain or sweep
python main.py logs <name> <step>             # tail the .out file for a step or array task
python main.py cancel <name>                  # scancel all pending/running jobs
python main.py retry <name> <step>            # resubmit a failed step
python main.py analyze <sweep_name>           # ranked TFLOPS table from sweep logs
```

---

## 9. File Layout

```
megatron-zoo/
├── main.py                        # CLI entry point
├── pyproject.toml
├── zoo/
│   ├── project.py                 # Project class
│   ├── module.py                  # Module base class + Context / ModuleResult types
│   ├── config.py                  # YAML loading, config merging, validation
│   ├── slurm.py                   # sbatch generation, submission, sacct queries
│   ├── state.py                   # chain_state.json read/write
│   ├── monitor.py                 # crash detection (log scan, W&B, checkpoint check)
│   ├── presets/
│   │   ├── models/
│   │   │   ├── qwen3-33b.yaml
│   │   │   ├── qwen3-235b.yaml
│   │   │   └── deepseek-v3.yaml
│   │   └── clusters/
│   │       ├── clariden.yaml
│   │       └── alps.yaml
│   └── modules/
│       ├── chain.py               # ExperimentChain + Step
│       ├── sweep.py               # BenchmarkSweep (job arrays)
│       ├── parse_logs.py
│       ├── join.py
│       ├── analysis.py
│       ├── eval.py
│       ├── checkpoint.py
│       └── tokenization.py
├── experiments/
│   └── qwen3_33b_ablation.py      # example experiment definitions
├── sbatch_scripts/                # generated scripts (never auto-deleted)
└── runs/
    └── qwen3_33b/
        ├── chain_state.json
        ├── metrics.csv
        └── ablations_with_metrics.csv
```

---

## 10. Nice-to-Haves

- **Dependency graph visualiser**: `python main.py graph <chain>` → ASCII DAG of steps
- **Module caching**: skip a step if its inputs haven't changed (hash inputs)
- **Notification hooks**: Slack/email when a chain finishes or a step fails
- **Rich progress display**: live status table with job IDs, elapsed time, W&B links
- **Smart retry**: if crash reason is OOM, auto-halve MBS and resubmit
- **sacct report**: pull timing/memory stats from `sacct` into a post-run summary table
