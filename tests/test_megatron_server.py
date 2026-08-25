import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from tools.inference.megatron_server import MegatronServerConfig, render, submit


def _config(**changes: Any) -> MegatronServerConfig:
    values: dict[str, Any] = {
        "name": "chonk-step-53396",
        "checkpoint": "/checkpoints/chonk",
        "ckpt_step": 53396,
        "tokenizer_model": "/tokenizers/apertus",
        "megatron_path": "/workspace/Megatron-LM",
        "expert_parallel_size": 4,
        "gpus_per_node": 4,
        "account": "infra01",
        "partition": "normal",
    }
    values.update(changes)
    return MegatronServerConfig(**values)


class DynamicServerTest(unittest.TestCase):
    def test_renders_one_slurm_task_per_gpu_without_torchrun(self) -> None:
        script = render(_config())

        self.assertIn("#SBATCH --ntasks-per-node=4", script)
        self.assertIn("--ntasks=4", script)
        self.assertIn('export RANK="${SLURM_PROCID}"', script)
        self.assertIn('export LOCAL_RANK="${SLURM_LOCALID}"', script)
        self.assertNotIn("torchrun", script)
        self.assertIn("tools/run_dynamic_text_generation_server.py", script)
        self.assertIn(
            'numactl --cpunodebind="${SLURM_LOCALID}" --membind="${SLURM_LOCALID}"',
            script,
        )
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_renders_pinned_megatron_worktree(self) -> None:
        script = render(_config(megatron_commit="abc123"))

        self.assertIn('MEGATRON_REQUESTED_REF="abc123"', script)
        self.assertIn("worktree add --detach", script)
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_renders_megatron_url_checkout(self) -> None:
        script = render(_config(megatron_path="https://example.com/Megatron-LM.git"))

        self.assertIn("Cloning Megatron-LM from https://example.com/Megatron-LM.git", script)
        self.assertIn('MEGATRON_REQUESTED_REF="refs/remotes/origin/HEAD"', script)
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_renders_dynamic_batching_graphs_and_model_args(self) -> None:
        script = render(
            _config(
                megatron_args={
                    "moe_token_dispatcher_type": "alltoall",
                    "window_size": "(512,0)",
                    "attention_output_gate": True,
                    "unused_flag": None,
                },
            )
        )

        for expected in (
            "--inference-dynamic-batching",
            "--inference-dynamic-batching-max-requests",
            "--inference-dynamic-batching-num-cuda-graphs",
            "--moe-pad-experts-for-cuda-graph-inference",
            "--moe-token-dispatcher-type",
            '"(512,0)"',
            "--attention-output-gate",
            "--port",
            "5000",
        ):
            self.assertIn(expected, script)
        self.assertNotIn("--unused-flag", script)
        self.assertNotIn('"--host"', script)

    def test_renders_multiple_nodes_and_total_task_count(self) -> None:
        script = render(_config(nodes=2, expert_parallel_size=8))

        self.assertIn("#SBATCH --nodes=2", script)
        self.assertIn("#SBATCH --ntasks-per-node=4", script)
        self.assertIn("--nodes=2", script)
        self.assertIn("--ntasks=8", script)
        self.assertIn('${SLURM_NTASKS:-8}', script)
        self.assertNotIn('"--host"', script)
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_installs_http_backend_once_inside_container_step(self) -> None:
        script = render(
            _config(
                container_edf="apertus2-alps4-temp",
                container_mounts="/checkpoints:/checkpoints",
                srun_extra_args="--network=disable_rdzv_get",
            )
        )

        self.assertIn('--environment="apertus2-alps4-temp"', script)
        self.assertIn('--container-mounts="/checkpoints:/checkpoints"', script)
        self.assertIn("--network=disable_rdzv_get", script)
        self.assertIn("python -m pip install quart hypercorn", script)
        self.assertIn('if [[ "${SLURM_LOCALID}" == "0" ]]', script)
        self.assertIn('MEGATRON_CONTAINER_PATH="/opt/megatron"', script)
        self.assertIn('rm -rf "${MEGATRON_CONTAINER_PATH}"', script)
        self.assertIn('export MEGATRON_PATH="${MEGATRON_CONTAINER_PATH}"', script)
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_submit_creates_log_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(log_dir=str(Path(tmp) / "logs"))
            with patch("tools.inference.megatron_server._sbatch", return_value="123"):
                self.assertEqual(submit(cfg), "123")
            self.assertTrue((Path(cfg.log_dir) / cfg.name).is_dir())

    def test_rejects_invalid_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            render(_config(expert_parallel_size=3))
        with self.assertRaisesRegex(ValueError, "invalid environment variable"):
            render(_config(env_vars={"BAD-NAME": "value"}))
        with self.assertRaisesRegex(ValueError, "host must be None"):
            render(_config(nodes=2, host="0.0.0.0"))
        with self.assertRaisesRegex(ValueError, "port"):
            render(_config(port=70000))


if __name__ == "__main__":
    unittest.main()
