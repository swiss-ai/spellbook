# Checkpoint conversion

Spellbook launches Megatron-Bridge's native `AutoBridge` export pipeline. Model
discovery, configuration mapping, weight mapping, loading, and Hugging Face export
remain owned by Megatron-Bridge.

## hfconverter

Use `HFConverterConfig` inside the model-owned conversion job:

```python
from tools.conversion.hfconverter import HFConverterConfig, launch

config = HFConverterConfig(
    hfconverter_root="/path/to/hfconverter",
    hfconverter_commit="<hfconverter-commit>",
    megatron_root="/path/to/Megatron-LM",
    megatron_commit="<megatron-commit>",
    bridge_root="/path/to/Megatron-Bridge",
    bridge_commit="<bridge-commit>",
    checkpoint="/path/to/native/iter_0008500",
    reference="/path/to/hf-reference",
    output="/path/to/new-hf-export",
    source_cache="/scratch/tmp/conversion-sources",
    options=["--tp", "1", "--pp", "4", "--ep", "8", "--etp", "1"],
)
launch(config)
```

All three commits are required and must exist in the local repositories.
Spellbook creates detached worktrees in the source cache; it does not switch,
reset or clean the original checkouts. A changed cached worktree is rejected.
`render_command(config)` resolves those worktrees and returns the native
`uv run --no-project python convert_checkpoint.py export ...` command.

The caller supplies the container, Slurm resources and distributed rank variables.
Mount the source cache and source repositories in that container. Native converter
options, such as `--pipeline-layout`, `--max-shard-size-gib` and repeated
`--model-override NAME=JSON`, go in `options`. The pipeline script controls staging
and publication of the completed HF export.

## Megatron-Bridge launcher

Copy [`examples/megatron_bridge.py`](examples/megatron_bridge.py) into the
repository that owns your checkpoint paths, then edit its `config`.

```bash
# Print the native Bridge command without running it.
uv run python -m tools.conversion.examples.megatron_bridge render

# Ask NeMo Run to render the complete Slurm submission without calling sbatch.
uv run python -m tools.conversion.examples.megatron_bridge dry-run

# Launch NeMo Run, which submits the distributed conversion with sbatch.
uv run python -m tools.conversion.examples.megatron_bridge submit
```

The launcher selects reproducible local source checkouts:

```python
config = BridgeExportConfig(
    bridge_root="/path/to/Megatron-Bridge",
    bridge_commit="<tested-bridge-commit>",
    # Optional: otherwise Bridge uses its bundled Megatron-LM submodule.
    megatron_root="/path/to/Megatron-LM",
    megatron_commit="<tested-megatron-commit>",
    ...
)
```

`bridge_root` is mounted automatically at `/opt/Megatron-Bridge`. When supplied,
`megatron_root` is mounted over
`/opt/Megatron-Bridge/3rdparty/Megatron-LM`. Both commits are checked before
submission. The launcher does not clone or modify either checkout.

The source checkpoint must contain a valid `run_config.yaml`, and its model must
be registered with `AutoBridge`. The invariant
`nodes * gpus_per_node == tp * pp * ep` is checked before launch.

Use `BridgeExportConfig`, `render_command()`, and `launch()` directly when an
experiment repository needs to compose the launcher in Python. `launch()` runs
`scripts/conversion/convert.sh export --executor slurm --device gpu`; the native
NeMo Run pipeline owns the generated sbatch script and submission.
