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
        "tasks": ["hellaswag", "arc_easy"],
        "account": "infra01",
        "partition": "normal",
    }
    values.update(changes)
    return VLLMEvalConfig(**values)


class VLLMEvalTest(unittest.TestCase):
    def test_uses_standard_lm_eval_vllm_backend(self) -> None:
        script = render(_config(tokenizer="/models/tokenizer", max_model_len=8192))

        self.assertIn("python -m lm_eval run \\", script)
        self.assertIn("--model vllm", script)
        self.assertIn(
            "pretrained=/models/apertus,tokenizer=/models/tokenizer,dtype=auto,"
            "tensor_parallel_size=1,data_parallel_size=1,gpu_memory_utilization=0.9,"
            "max_model_len=8192,seed=1234",
            script,
        )
        self.assertIn("--tasks hellaswag,arc_easy", script)
        self.assertIn("--batch_size auto", script)
        self.assertNotIn("torch.distributed.run", script)
        self.assertNotIn("run_vllm_eval.py", script)
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_renders_tensor_and_data_parallelism(self) -> None:
        script = render(
            _config(
                gpus_per_node=4,
                tensor_parallel_size=2,
                data_parallel_size=2,
                model_args_extra={"distributed_executor_backend": "mp"},
            )
        )

        self.assertIn("tensor_parallel_size=2", script)
        self.assertIn("data_parallel_size=2", script)
        self.assertIn("distributed_executor_backend=mp", script)

    def test_renders_common_lm_eval_options(self) -> None:
        script = render(
            _config(
                batch_size=8,
                cache_requests="true",
                num_fewshot=5,
                limit=2,
                metadata={"checkpoint": 1907},
                log_samples=True,
                write_out=True,
                lm_eval_args_extra={"verbosity": "DEBUG"},
            )
        )

        for expected in (
            "--batch_size 8",
            "--cache_requests true",
            "--num_fewshot 5",
            "--limit 2",
            '--metadata "{\\"checkpoint\\": 1907}"',
            "--log_samples",
            "--write_out",
            "--verbosity DEBUG",
        ):
            self.assertIn(expected, script)

    def test_slurm_container_install_cache_and_wandb_options(self) -> None:
        script = render(
            _config(
                container_edf="apertus2-vllm",
                container_mounts="/models:/models",
                srun_extra_args="--network=disable_rdzv_get",
                conversion_job_id="12345",
                hf_home="/cache/hf",
                lm_eval_install="/workspace/lm-evaluation-harness",
                lm_eval_install_with_python=True,
                install_commands="python -m pip install package",
                wandb_project="evals",
                wandb_id="run-1",
                env_vars={"VLLM_USE_V1": "1"},
            )
        )

        self.assertIn("#SBATCH --dependency=afterok:12345", script)
        self.assertIn('--environment="apertus2-vllm"', script)
        self.assertIn('--container-mounts="/models:/models"', script)
        self.assertIn("--network=disable_rdzv_get", script)
        self.assertIn('export HF_HOME="/cache/hf"', script)
        self.assertIn("python -m pip install /workspace/lm-evaluation-harness", script)
        self.assertIn("python -m pip install package", script)
        self.assertIn('export VLLM_USE_V1="1"', script)
        self.assertIn("--wandb_args project=evals,id=run-1,resume=allow", script)
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_submit_creates_log_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(log_dir=str(Path(tmp) / "logs"))
            with patch("evals.vllm_eval._sbatch", return_value="123"):
                self.assertEqual(submit(cfg), "123")

            self.assertTrue((Path(cfg.log_dir) / cfg.model_name).is_dir())

    def test_rejects_unsupported_or_invalid_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "one Slurm node"):
            render(_config(nodes=2))
        with self.assertRaisesRegex(ValueError, "exceeds gpus_per_node"):
            render(_config(gpus_per_node=4, tensor_parallel_size=2, data_parallel_size=4))
        with self.assertRaisesRegex(ValueError, "gpu_memory_utilization"):
            render(_config(gpu_memory_utilization=1.1))
        with self.assertRaisesRegex(ValueError, "invalid environment variable"):
            render(_config(env_vars={"BAD-NAME": "value"}))
        with self.assertRaisesRegex(ValueError, "conversion_job_id"):
            render(_config(conversion_job_id="not-a-job-id"))


if __name__ == "__main__":
    unittest.main()
