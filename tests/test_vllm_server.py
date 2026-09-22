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
