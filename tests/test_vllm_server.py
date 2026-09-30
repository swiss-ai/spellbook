import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from tools.inference.vllm_server import VLLMServerConfig, render, submit


def _config(**changes: Any) -> VLLMServerConfig:
    values: dict[str, Any] = {
        "name": "chonk-vllm",
        "model": "/models/chonk",
        "served_model_name": "chonk",
        "nodes": 2,
        "gpus_per_node": 4,
        "enable_expert_parallel": True,
        "all2all_backend": "deepep_low_latency",
        "account": "csstaff",
        "partition": "preemptable",
        "container_edf": "/images/vllm.toml",
    }
    values.update(changes)
    return VLLMServerConfig(**values)


class VLLMServerTest(unittest.TestCase):
    def test_renders_native_multi_node_data_parallel_server(self) -> None:
        script = render(
            _config(
                external_data_parallel=False,
                vllm_args={
                    "load_format": "runai_streamer",
                    "trust_remote_code": True,
                    "max_model_len": 4096,
                },
                env_vars={"UCCL_EP_TRANSPORT": "cxi"},
            )
        )

        self.assertIn("#SBATCH --ntasks-per-node=1", script)
        self.assertIn("start_rank=$((SLURM_NODEID * local_dp))", script)
        self.assertIn("--data-parallel-size", script)
        self.assertIn("  8", script)
        self.assertIn("--data-parallel-size-local", script)
        self.assertIn("  4", script)
        self.assertIn('args+=("--data-parallel-start-rank"', script)
        self.assertIn('args+=("--api-server-count" "1"', script)
        self.assertIn('export UCCL_EP_TRANSPORT="cxi"', script)
        self.assertIn("--load-format", script)
        self.assertNotIn("torchrun", script)
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_default_moe_uses_external_data_parallel(self) -> None:
        script = render(_config())
        self.assertIn("#SBATCH --ntasks-per-node=4", script)
        self.assertIn('"--data-parallel-rank" "$rank"', script)
        self.assertNotIn("--data-parallel-size-local", script)

    def test_external_data_parallel_uses_one_slurm_task_per_gpu(self) -> None:
        script = render(_config(external_data_parallel=True))

        self.assertIn("#SBATCH --ntasks-per-node=4", script)
        self.assertIn("#SBATCH --gpus-per-task=1", script)
        self.assertIn("#SBATCH --cpus-per-task=72", script)
        self.assertIn("--ntasks=8", script)
        self.assertIn('export XDG_CACHE_HOME="/tmp/spellbook-vllm-${SLURM_JOB_ID}-${rank}"', script)
        self.assertIn('"--data-parallel-rank" "$rank"', script)
        self.assertIn('"--port" "$((8000 + SLURM_LOCALID))"', script)
        self.assertNotIn("--data-parallel-size-local", script)
        self.assertNotIn("--headless", script)
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_external_data_parallel_requires_single_gpu_moe_ranks(self) -> None:
        with self.assertRaisesRegex(ValueError, "TP=1"):
            render(_config(external_data_parallel=True, tensor_parallel_size=2))
        with self.assertRaisesRegex(ValueError, "expert parallelism"):
            render(_config(external_data_parallel=True, enable_expert_parallel=False))
        with self.assertRaisesRegex(ValueError, "divisible"):
            render(_config(external_data_parallel=True, cpus_per_task=289))
        with self.assertRaisesRegex(ValueError, "explicit NUMA"):
            render(_config(external_data_parallel=True, vllm_args={"numa_bind": True}))

    def test_dense_tp1_keeps_per_node_default(self) -> None:
        script = render(_config(enable_expert_parallel=False))
        self.assertIn("#SBATCH --ntasks-per-node=1", script)
        self.assertIn("--data-parallel-size-local", script)

    def test_tensor_parallelism_reduces_local_data_parallelism(self) -> None:
        script = render(_config(tensor_parallel_size=2))

        self.assertIn("local_dp=2", script)
        self.assertIn("--data-parallel-size", script)
        self.assertIn("  4", script)
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_submit_creates_log_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(log_dir=str(Path(tmp) / "logs"))
            with patch("tools.inference.vllm_server._sbatch", return_value="123"):
                self.assertEqual(submit(cfg), "123")
            self.assertTrue((Path(cfg.log_dir) / cfg.name).is_dir())

    def test_rejects_invalid_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            render(_config(gpus_per_node=4, tensor_parallel_size=3))
        with self.assertRaisesRegex(ValueError, "invalid environment variable"):
            render(_config(env_vars={"BAD-NAME": "value"}))


if __name__ == "__main__":
    unittest.main()
