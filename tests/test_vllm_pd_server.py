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
        script = render(_config())

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

    def test_submit_creates_log_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(log_dir=str(Path(tmp) / "logs"))
            with patch("tools.inference.vllm_pd_server._sbatch", return_value="123"):
                self.assertEqual(submit(cfg), "123")
            self.assertTrue((Path(cfg.log_dir) / cfg.name).is_dir())

    def test_rejects_mismatched_groups_and_ports(self) -> None:
        with self.assertRaisesRegex(ValueError, "must match"):
            render(_config(prefill_nodes=1, decode_nodes=2))
        with self.assertRaisesRegex(ValueError, "distinct"):
            render(_config(prefill_port=8200))
        with self.assertRaisesRegex(ValueError, "proxy_script"):
            render(_config(proxy_script=""))
        with self.assertRaisesRegex(ValueError, "proxy_health_path"):
            render(_config(proxy_health_path="health"))


if __name__ == "__main__":
    unittest.main()
