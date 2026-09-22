"""Example native multi-node vLLM server."""

import argparse
import os

from tools.inference.vllm_server import VLLMServerConfig, render, submit

config = VLLMServerConfig(
    name="example-vllm",
    model=os.environ.get("MODEL", "/path/to/hf-format-model"),
    served_model_name=os.environ.get("SERVED_MODEL_NAME", "model"),
    nodes=2,
    gpus_per_node=4,
    enable_expert_parallel=True,
    all2all_backend="deepep_low_latency",
    vllm_args={
        "dtype": "bfloat16",
        "load_format": "runai_streamer",
        "model_loader_extra_config": '{"distributed":true,"concurrency":16}',
        "max_model_len": 4096,
    },
    account=os.environ.get("SLURM_ACCOUNT", "your-account"),
    partition=os.environ.get("SLURM_PARTITION", "preemptable"),
    container_edf=os.environ.get("CONTAINER_EDF", "your-vllm-container.toml"),
    container_mounts=os.environ.get(
        "CONTAINER_MOUNTS",
        "${SCRATCH}:${SCRATCH},${HOME}:${HOME},/capstor:/capstor,/iopsstor:/iopsstor",
    ),
    env_vars={"UCCL_EP_TRANSPORT": "cxi"},
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
