"""Example disaggregated vLLM prefill/decode server."""

import argparse
import os

from tools.inference.vllm_pd_server import VLLMPDServerConfig, render, submit

config = VLLMPDServerConfig(
    name="example-vllm-pd",
    model=os.environ.get("MODEL", "/path/to/hf-format-model"),
    proxy_script=os.environ.get("VLLM_PD_PROXY_SCRIPT", "/path/to/disagg_proxy.py"),
    proxy_health_path="/health",
    served_model_name=os.environ.get("SERVED_MODEL_NAME", "model"),
    prefill_nodes=2,
    decode_nodes=2,
    gpus_per_node=4,
    vllm_args={
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "load_format": "runai_streamer",
        "model_loader_extra_config": '{"distributed":true,"concurrency":16}',
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.8,
        "enforce_eager": True,
    },
    # Tuned for EP8 on four-GPU Alps nodes.
    uccl_dispatch_config="24,12,512,32,512",
    uccl_combine_config="24,2,512,24,512",
    account=os.environ.get("SLURM_ACCOUNT", "your-account"),
    partition=os.environ.get("SLURM_PARTITION", "preemptable"),
    container_edf=os.environ.get("CONTAINER_EDF", "your-vllm-container.toml"),
    container_mounts=os.environ.get(
        "CONTAINER_MOUNTS",
        "${SCRATCH}:${SCRATCH},${HOME}:${HOME},/capstor:/capstor,/iopsstor:/iopsstor",
    ),
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
