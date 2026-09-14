from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml

from spellbook.backends import SlurmNemoRLBackend
from spellbook.nemorl import NemoRLExperiment


def _experiment(**overrides: object) -> NemoRLExperiment:
    values: dict[str, Any] = {
        "name": "nemorl-test",
        "algorithm": "grpo",
        "max_num_steps": 10,
        "num_prompts_per_step": 8,
        "num_generations_per_prompt": 4,
        "hf_config_dir": "/model/config",
        "pretrained_checkpoint_path": "/checkpoints/model",
        "nemo_rl_path": "/src/Nemo-RL",
        "bridge_src_path": "/src/Megatron-Bridge/src",
        "megatron_path": "/src/Megatron-LM",
        "max_total_sequence_length": 4096,
    }
    values.update(overrides)
    return NemoRLExperiment(**values)


def _backend(**overrides: object) -> SlurmNemoRLBackend:
    values: dict[str, Any] = {
        "account": "account",
        "partition": "normal",
        "nodes": 2,
        "gpus_per_node": 4,
    }
    values.update(overrides)
    return SlurmNemoRLBackend(**values)


class NemoRLBackendTest(unittest.TestCase):
    def test_renders_recipe_and_two_node_ray_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            body = _backend(
                vetnode=True,
                srun_extra_args="--network=disable_rdzv_get --mpi=pmix --wait=30",
            ).render(_experiment(), output)
            body_text = body.read_text()
            recipe = yaml.safe_load((output / "nemorl-test.yaml").read_text())
            wrapper = (output / "nemorl-test.sbatch").read_text()

        self.assertEqual(recipe["cluster"], {"gpus_per_node": 4, "num_nodes": 2})
        self.assertEqual(recipe["grpo"]["num_prompts_per_step"], 8)
        self.assertEqual(recipe["policy"]["generation"]["backend"], "megatron")
        self.assertIn("examples/run_grpo.py", body_text)
        self.assertIn('python "examples/run_grpo.py"', body_text)
        self.assertIn("python - <<'PY_WAIT_NODES'", body_text)
        self.assertIn('"$head_node"', wrapper)
        self.assertIn("$((num_nodes - 1))", wrapper)
        self.assertIn('--ntasks-per-node="${SPELLBOOK_VETNODE_TASKS_PER_NODE}"', wrapper)
        self.assertIn('numactl --cpunodebind="${SLURM_LOCALID}"', wrapper)
        self.assertIn(
            'EXPORTS="RAY_HEAD_IP,RAY_ADDRESS,RAY_READY_FILE,RAY_DONE_FILE,RAY_EXPECTED_NODES"',
            wrapper,
        )
        self.assertIn("WANDB_API_KEY,WANDB_ENTITY,WANDB_PROJECT", wrapper)
        self.assertIn("mapfile -t nodes_array", wrapper)
        self.assertIn("--mpi=pmix --wait=30", wrapper)
        subprocess.run(["bash", "-n"], input=wrapper, text=True, check=True)

    def test_default_scheduler_logs_are_outside_checkout(self) -> None:
        backend = _backend()

        self.assertTrue(Path(backend.log_dir).is_absolute())
        self.assertIn("/tmp/spellbook/nemorl/slurm_logs", backend.log_dir)
        self.assertEqual(
            backend.log_dir,
            str(
                Path(
                    os.environ.get(
                        "SCRATCH",
                        f"/iopsstor/scratch/cscs/{os.environ.get('USER', 'unknown')}",
                    )
                )
                / "tmp/spellbook/nemorl/slurm_logs"
            ),
        )

    def test_non_generation_algorithm_omits_generation_settings(self) -> None:
        recipe = _backend().build_recipe(_experiment(algorithm="sft", num_prompts_per_step=0))

        self.assertNotIn("num_prompts_per_step", recipe["sft"])
        self.assertNotIn("generation", recipe["policy"])

    def test_prompt_batch_uses_allocation_data_parallel_size(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(ValueError, "data-parallel size \\(8\\)"),
        ):
            _backend().render_recipe(
                _experiment(
                    num_prompts_per_step=4,
                    expert_model_parallel_size=4,
                ),
                Path(directory),
            )

    def test_model_parallel_size_must_divide_world_size(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(ValueError, "world_size=8"),
        ):
            _backend().render_recipe(_experiment(tensor_model_parallel_size=3), Path(directory))

    def test_auto_requeue_requires_checkpoint_directory(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(ValueError, "checkpoint_dir"),
        ):
            _backend(auto_requeue=True).render_recipe(_experiment(), Path(directory))


if __name__ == "__main__":
    unittest.main()
