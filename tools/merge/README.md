# Megatron checkpoint merge

Slurm launcher for the Megatron backend's distributed `tools/checkpoint/merge.py`, added to Megatron-LM-MoE in commit [`38f36a8`](https://github.com/andresnowak/Megatron-LM-MoE/commit/38f36a887d5be7ea3c4932d5a958ffc9b0dc87a8). It starts one checkpoint worker per Slurm task, binds its CPU and memory to the matching NUMA node, and defaults to the CPU-compatible `gloo` process-group backend.

```python
from tools.merge import MegatronCheckpointMergeConfig, submit

submit(MegatronCheckpointMergeConfig(
    name="model-steps-1000-2000",
    checkpoints=["/path/to/checkpoints/model"],
    checkpoint_steps=[1000, 2000],
    output="/path/to/checkpoints/model-merged",
    megatron_path="/path/to/Megatron-LM",
    megatron_commit="<commit-containing-tools/checkpoint/merge.py>",
    nodes=1,
    workers_per_node=4,
    account="infra01",
    partition="normal",
    container_edf="apertus2-alps4-temp",
    container_mounts="${SCRATCH}:${SCRATCH},${HOME}:${HOME},/capstor:/capstor,/iopsstor:/iopsstor",
    srun_extra_args="--network=disable_rdzv_get",
))
```

Use `render(config)` to inspect the sbatch script without submitting it. `checkpoints` may contain one root paired with multiple `checkpoint_steps`, multiple roots resolved through their tracker files, or direct iteration/release directories.

`megatron_path` accepts a local checkout or Git URL, and `megatron_commit` pins a commit or branch. Like experiment launches, container jobs remove the existing Megatron copy before copying the selected checkout to `megatron_container_path` (default: `/opt/megatron`).

For weight-space merging with linear decay, set `merge_method="linear-decay"`. The remaining schedule fields and `backend` map directly to Megatron's merge CLI.
