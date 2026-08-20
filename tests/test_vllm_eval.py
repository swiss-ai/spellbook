import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from evals.vllm_eval import VLLMEvalConfig, render, submit


def _config(**changes: Any) -> VLLMEvalConfig:
    values: dict[str, Any] = {
        "model_name": "apertus-test",
        "model": "/models/apertus",
        "runner": "/workspace/run_vllm_eval.py",
        "tasks": ["hellaswag", "arc_easy"],
        "account": "infra01",
        "partition": "normal",
    }
    values.update(changes)
    return VLLMEvalConfig(**values)


class VLLMEvalTest(unittest.TestCase):
    def test_multi_node_launch_uses_one_slurm_task_per_node(self) -> None:
        script = render(
            _config(
                nodes=2,
                gpus_per_node=4,
                container_edf="apertus2-vllm",
                container_mounts="/models:/models",
                srun_extra_args="--network=disable_rdzv_get",
            )
        )

        self.assertIn("#SBATCH --nodes=2", script)
        self.assertIn("#SBATCH --ntasks-per-node=1", script)
        self.assertIn("  --ntasks=2 \\", script)
        self.assertIn("--nproc-per-node=4", script)
        self.assertIn('--rdzv-endpoint="${RDZV_ENDPOINT}"', script)
        self.assertIn('--environment="apertus2-vllm"', script)
        self.assertIn("--network=disable_rdzv_get", script)
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_single_node_launch_uses_standalone_torchrun(self) -> None:
        script = render(_config(nodes=1, gpus_per_node=4))

        self.assertIn("--standalone", script)
        self.assertIn("--nproc-per-node=4", script)
        self.assertNotIn('--rdzv-endpoint="${RDZV_ENDPOINT}"', script)
        self.assertNotIn("--environment=", script)
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_task_mode_launches_one_runner_per_gpu(self) -> None:
        script = render(_config(nodes=2, gpus_per_node=4, launch_mode="tasks"))

        self.assertIn("#SBATCH --ntasks-per-node=4", script)
        self.assertIn("  --ntasks=8 \\", script)
        self.assertIn('export RANK="${SLURM_PROCID}"', script)
        self.assertIn('export LOCAL_RANK="${SLURM_LOCALID}"', script)
        self.assertIn('export WORLD_SIZE="${SLURM_NTASKS:-8}"', script)
        self.assertIn('numactl --cpunodebind="${SLURM_LOCALID}"', script)
        self.assertNotIn("torch.distributed.run", script)
        self.assertNotIn("--nproc-per-node", script)
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_common_and_extra_runner_arguments_are_rendered_as_array_items(self) -> None:
        script = render(
            _config(
                tokenizer="swiss-ai/Apertus-tokenizer",
                batch_size=8,
                max_length=8192,
                extra_runner_args=["--mode", "native", "--label", "value with spaces"],
            )
        )

        for expected in (
            '"--model"',
            '"/models/apertus"',
            '"--tasks"',
            '"hellaswag,arc_easy"',
            '"--batch-size"',
            '"8"',
            '"--max-length"',
            '"8192"',
            '"--tokenizer"',
            '"swiss-ai/Apertus-tokenizer"',
            '"value with spaces"',
        ):
            self.assertIn(expected, script)

    def test_submit_creates_log_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(log_dir=str(Path(tmp) / "logs"))
            with patch("evals.vllm_eval._sbatch", return_value="123"):
                self.assertEqual(submit(cfg), "123")

            self.assertTrue((Path(cfg.log_dir) / cfg.model_name).is_dir())

    def test_conversion_job_can_be_an_afterok_dependency(self) -> None:
        script = render(_config(conversion_job_id="12345"))

        self.assertIn("#SBATCH --dependency=afterok:12345", script)

    def test_rejects_invalid_parallelism_and_environment_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "launch_mode"):
            render(_config(launch_mode="invalid"))
        with self.assertRaisesRegex(ValueError, "nodes must be greater than zero"):
            render(_config(nodes=0))
        with self.assertRaisesRegex(ValueError, "invalid environment variable"):
            render(_config(env_vars={"BAD-NAME": "value"}))
        with self.assertRaisesRegex(ValueError, "conversion_job_id"):
            render(_config(conversion_job_id="not-a-job-id"))


if __name__ == "__main__":
    unittest.main()
