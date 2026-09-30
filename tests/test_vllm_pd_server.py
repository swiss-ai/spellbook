import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from tools.inference.vllm_pd_server import VLLMPDServerConfig, render, submit


def _config(**changes: Any) -> VLLMPDServerConfig:
    values: dict[str, Any] = {
        "name": "chonk-pd",
        "model": "/models/chonk",
        "proxy_script": "/workspace/vllm/disagg_proxy.py",
        "proxy_health_path": "/health",
        "served_model_name": "chonk",
        "account": "csstaff",
        "partition": "preemptable",
        "container_edf": "/images/vllm.toml",
        "uccl_dispatch_config": "24,12,512,32,512",
        "uccl_combine_config": "24,2,512,24,512",
        "vllm_args": {
            "dtype": "bfloat16",
            "load_format": "runai_streamer",
            "model_loader_extra_config": '{"distributed":true,"concurrency":16}',
            "max_model_len": 4096,
            "enforce_eager": True,
        },
    }
    values.update(changes)
    return VLLMPDServerConfig(**values)


class VLLMPDServerTest(unittest.TestCase):
    def test_renders_validated_uccl_topology(self) -> None:
        script = render(_config(external_data_parallel=False))

        self.assertIn("#SBATCH --nodes=4", script)
        self.assertIn("deepep_high_throughput", script)
        self.assertIn("deepep_low_latency", script)
        self.assertIn("kv_producer", script)
        self.assertIn("kv_consumer", script)
        self.assertIn('"kv_connector":"NixlConnector"', script)
        self.assertIn('"backends":["UCCL"]', script)
        self.assertIn("export UCCL_EP_TRANSPORT=cxi", script)
        self.assertIn("export UCCL_P2P_TRANSPORT=cxi", script)
        self.assertIn("export VLLM_SSM_CONV_STATE_LAYOUT=DS", script)
        self.assertIn('export UCCL_EP_DISPATCH_CONFIG="24,12,512,32,512"', script)
        self.assertIn('export UCCL_EP_COMBINE_CONFIG="24,2,512,24,512"', script)
        self.assertIn("getent ahostsv4", script)
        self.assertIn("--data-parallel-size", script)
        self.assertIn("  8", script)
        self.assertIn("--prefiller-hosts", script)
        self.assertIn("--decoder-hosts", script)
        self.assertIn('9000/health" proxy', script)
        self.assertIn("--model-loader-extra-config", script)
        self.assertIn('{"distributed":true,"concurrency":16}', script)
        self.assertNotIn("torchrun", script)
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_renders_default_external_dp_per_gpu_ranks(self) -> None:
        script = render(_config())

        self.assertIn("#SBATCH --gpus-per-node=4", script)
        self.assertEqual(script.count("srun --overlap --nodes=4 --ntasks=16"), 1)
        self.assertIn("--ntasks-per-node=4 --gpus-per-node=4", script)
        self.assertIn("--input=all", script)
        self.assertIn("if (( procid < 8 )); then", script)
        self.assertIn('role=prefill; rank="$procid"', script)
        self.assertIn('rank="$((procid - 8))"', script)
        self.assertIn('export RANK="$rank" LOCAL_RANK="$SLURM_LOCALID"', script)
        self.assertIn(
            f'export PYTHONPATH="{Path(__file__).resolve().parents[1]}:${{PYTHONPATH:-}}"', script
        )
        self.assertIn('export MASTER_ADDR="$dp_head" MASTER_PORT="$((rpc_port + 1000))"', script)
        self.assertIn(
            'api_port="$prefill_api"; backend="$prefill_backend"; kv_role=kv_producer', script
        )
        self.assertIn(
            'api_port="$decode_api"; backend="$decode_backend"; kv_role=kv_consumer', script
        )
        self.assertIn("SLINGSHOT_VNIS=${SLINGSHOT_VNIS:-}", script)
        self.assertIn("SLINGSHOT_DEVICES=${SLINGSHOT_DEVICES:-}", script)
        self.assertIn("SLINGSHOT_SVC_IDS=${SLINGSHOT_SVC_IDS:-}", script)
        self.assertIn("SLURM_STEP_ID=${SLURM_STEP_ID:-}", script)
        self.assertIn("${role}-${SLURM_JOB_ID}-${rank}-${SLURM_LOCALID}.log", script)
        self.assertIn('[[ -n "${visible_gpus[$SLURM_LOCALID]:-}" ]]', script)
        self.assertNotIn('export CUDA_VISIBLE_DEVICES="${visible_gpus[$SLURM_LOCALID]}"', script)
        self.assertNotIn("--gpus-per-task", script)
        self.assertIn("--cpu-bind=cores --mem-bind=local", script)
        self.assertIn("--data-parallel-rank", script)
        self.assertIn("--data-parallel-size", script)
        self.assertIn("  8", script)
        self.assertIn('"--data-parallel-size-local" "1"', script)
        self.assertIn('"--distributed-executor-backend" "mp"', script)
        self.assertIn('"--worker-cls" "tools.inference.vllm_slurm_worker.SlurmGPUWorker"', script)
        self.assertNotIn('"--distributed-executor-backend" "external_launcher"', script)
        self.assertIn('"--port" "$((api_port + SLURM_LOCALID))"', script)
        self.assertIn("FLASHINFER_WORKSPACE_BASE", script)
        self.assertIn("export LOCAL_WORLD_SIZE=4", script)
        self.assertIn("export VLLM_NIXL_SIDE_CHANNEL_PORT=5600", script)
        self.assertIn("--prefiller-hosts", script)
        self.assertIn('--prefiller-ports "${prefiller_ports[@]}"', script)
        self.assertIn('--decoder-ports "${decoder_ports[@]}"', script)
        self.assertNotIn("join_csv", script)
        self.assertNotIn("--headless", script)
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_tp2_default_keeps_per_node_launcher(self) -> None:
        script = render(_config(tensor_parallel_size=2))
        self.assertIn("#SBATCH --ntasks-per-node=1", script)
        self.assertNotIn('"--distributed-executor-backend" "external_launcher"', script)
        self.assertIn("--data-parallel-size-local", script)

    def test_external_dp_requires_tp1_and_explicit_numa_binding(self) -> None:
        with self.assertRaisesRegex(ValueError, "tensor_parallel_size=1"):
            render(_config(external_data_parallel=True, tensor_parallel_size=2))
        with self.assertRaisesRegex(ValueError, "automatic --numa-bind"):
            render(_config(external_data_parallel=True, vllm_args={"numa_bind": "auto"}))
        with self.assertRaisesRegex(ValueError, "not enough ports"):
            render(_config(external_data_parallel=True, prefill_rpc_port=65000))

    def test_submit_creates_log_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(log_dir=str(Path(tmp) / "logs"))
            with patch("tools.inference.vllm_pd_server._sbatch", return_value="123"):
                self.assertEqual(submit(cfg), "123")
            self.assertTrue((Path(cfg.log_dir) / cfg.name).is_dir())

    def test_rejects_mismatched_groups_and_ports(self) -> None:
        with self.assertRaisesRegex(ValueError, "must match"):
            render(_config(prefill_nodes=1, decode_nodes=2))
        with self.assertRaisesRegex(ValueError, "overlap"):
            render(_config(prefill_port=8200))
        with self.assertRaisesRegex(ValueError, "distinct"):
            render(_config(prefill_port=8200, external_data_parallel=False))
        with self.assertRaisesRegex(ValueError, "proxy_script"):
            render(_config(proxy_script=""))
        with self.assertRaisesRegex(ValueError, "proxy_health_path"):
            render(_config(proxy_health_path="health"))


if __name__ == "__main__":
    unittest.main()
