# Checkpoint conversion

Spellbook launches Megatron-Bridge's native `AutoBridge` export pipeline. Model
discovery, configuration mapping, weight mapping, loading, and Hugging Face export
remain owned by Megatron-Bridge.

## Launcher

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
