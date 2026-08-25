"""Example multi-node Megatron dynamic-inference server configuration.

Usage:
    python -m tools.inference.examples.megatron_server render
    python -m tools.inference.examples.megatron_server submit

Replace the placeholder model, tokenizer, Megatron, and Slurm values first.
"""

from __future__ import annotations

import argparse
import os

from tools.inference.megatron_server import MegatronServerConfig, render, submit

config = MegatronServerConfig(
    name="example-model-step-3000",
    checkpoint="/path/to/checkpoints/example-model",
    ckpt_step=3000,
    tokenizer_model="/path/to/tokenizer",
    megatron_path=os.environ.get("MEGATRON_PATH", "/path/to/Megatron-LM"),
    megatron_commit=os.environ.get("MEGATRON_COMMIT", ""),
    # One process per GPU: 2 nodes * 4 GPUs = WORLD_SIZE 8.
    nodes=2,
    gpus_per_node=4,
    tensor_parallel_size=2,
    pipeline_parallel_size=1,
    expert_parallel_size=4,
    seq_length=4096,
    micro_batch_size=8,
    max_requests=8,
    max_tokens=32768,
    # Arbitrary Megatron flags use experiment-style snake_case conversion.
    megatron_args={
        "moe_token_dispatcher_type": "alltoall",
        "moe_router_load_balancing_type": "quantile_balancing",
        "moe_router_quantile_balancing_method": "histogram",
        "attention_output_gate": True,
        "distributed_timeout_minutes": 180,
    },
    account=os.environ.get("SLURM_ACCOUNT", "your-account"),
    partition=os.environ.get("SLURM_PARTITION", "normal"),
    reservation=os.environ.get("SLURM_RESERVATION", ""),
    container_edf=os.environ.get("CONTAINER_EDF", "your-container-environment"),
    container_mounts=os.environ.get(
        "CONTAINER_MOUNTS",
        "${SCRATCH}:${SCRATCH},${HOME}:${HOME},/capstor:/capstor,/iopsstor:/iopsstor",
    ),
    srun_extra_args="--network=disable_rdzv_get",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("render", "submit"))
    args = parser.parse_args()
    if args.command == "render":
        print(render(config), end="")
    else:
        submit(config)


if __name__ == "__main__":
    main()
