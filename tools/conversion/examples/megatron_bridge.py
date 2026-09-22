"""Example distributed Megatron-Bridge export launcher.

Usage:
    python -m tools.conversion.examples.megatron_bridge render
    python -m tools.conversion.examples.megatron_bridge dry-run
    python -m tools.conversion.examples.megatron_bridge submit

Replace the placeholder checkpoint, model, checkout, container, and Slurm values first.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import shlex

from tools.conversion.megatron_bridge import BridgeExportConfig, launch, render_command


def _mounts() -> list[str]:
    value = os.environ.get("CONTAINER_MOUNTS", "/capstor,/iopsstor,/ritom")
    return [mount for mount in value.split(",") if mount]


config = BridgeExportConfig(
    bridge_root=os.environ.get("BRIDGE_PATH", "/path/to/Megatron-Bridge"),
    bridge_commit=os.environ.get("BRIDGE_COMMIT", ""),
    # Omit these two fields to use Bridge's bundled Megatron-LM submodule.
    megatron_root=os.environ.get("MEGATRON_PATH", ""),
    megatron_commit=os.environ.get("MEGATRON_COMMIT", ""),
    hf_model=os.environ.get("HF_MODEL", "organization/model-or-local-hf-reference"),
    megatron_path=os.environ.get("CHECKPOINT", "/path/to/checkpoint/iter_0001000"),
    hf_path=os.environ.get("HF_OUTPUT", "/path/to/output/model-hf"),
    container_image=os.environ.get("CONTAINER_IMAGE", "/path/to/megatron-bridge.sqsh"),
    nodes=2,
    gpus_per_node=4,
    tp=1,
    pp=1,
    ep=8,
    etp=1,
    export_weight_dtype="bfloat16",
    trust_remote_code=True,
    account=os.environ.get("SLURM_ACCOUNT", "your-account"),
    partition=os.environ.get("SLURM_PARTITION", "preemptable"),
    run_time="05:00:00",
    memory="800000",
    mounts=_mounts(),
    env_names=["HF_TOKEN"] if "HF_TOKEN" in os.environ else [],
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("render", "dry-run", "submit"))
    args = parser.parse_args()
    if args.command == "render":
        print(shlex.join(render_command(config)))
    elif args.command == "dry-run":
        launch(dataclasses.replace(config, dry_run=True, detach=False))
    else:
        launch(config)


if __name__ == "__main__":
    main()
