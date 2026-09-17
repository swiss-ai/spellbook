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
        self.assertFalse(recipe["policy"]["dtensor_cfg"]["enabled"])
        self.assertTrue(recipe["policy"]["megatron_cfg"]["enabled"])
        self.assertIn("examples/run_grpo.py", body_text)
        self.assertIn('python "examples/run_grpo.py"', body_text)
        self.assertIn("python - <<'PY_WAIT_NODES'", body_text)
        self.assertIn('[[ "${SLURM_PROCID:-0}" == "0" ]]', body_text)
        self.assertIn('while [[ ! -f "$RAY_DONE_FILE" ]]', body_text)
        self.assertIn('--nodes="$num_nodes" --ntasks="$num_nodes" --ntasks-per-node=1', wrapper)
        self.assertIn(f'-u bash "{body}" auto', wrapper)
        self.assertNotIn("$((num_nodes - 1))", wrapper)
        self.assertIn('--ntasks-per-node="${SPELLBOOK_VETNODE_TASKS_PER_NODE}"', wrapper)
        self.assertIn('numactl --cpunodebind="${SLURM_LOCALID}"', wrapper)
        self.assertIn(
            'EXPORTS="RAY_HEAD_IP,RAY_ADDRESS,RAY_READY_FILE,RAY_DONE_FILE,RAY_EXPECTED_NODES"',
            wrapper,
        )
        self.assertIn("WANDB_API_KEY,WANDB_ENTITY,WANDB_PROJECT", wrapper)
        self.assertIn("mapfile -t nodes_array", wrapper)
        self.assertIn("--mpi=pmix --wait=30", wrapper)
        subprocess.run(["bash", "-n"], input=body_text, text=True, check=True)
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


class GymGenerationTest(unittest.TestCase):
    def _mcore(self, **kwargs: object) -> dict:
        backend = SlurmNemoRLBackend(
            account="test", partition="test", nodes=1, gpus_per_node=4, run_time="00:05:00"
        )
        policy = backend.policy_section(_experiment(generation_backend="megatron", **kwargs))
        return policy["generation"]["mcore_generation_config"]

    def test_gym_runs_need_the_async_engine_and_http_server(self) -> None:
        """NeMo-RL asserts on both before it will run against Gym."""
        mcore = self._mcore(gym_config_paths=("responses_api_agents/a/configs/a.yaml",))

        self.assertTrue(mcore["async_engine"])
        self.assertTrue(mcore["expose_http_server"])

    def test_non_gym_runs_leave_both_off(self) -> None:
        mcore = self._mcore()

        self.assertFalse(mcore["async_engine"])
        self.assertFalse(mcore["expose_http_server"])


class PretrainedCheckpointTest(unittest.TestCase):
    def _policy(self, **kwargs: object) -> dict:
        backend = SlurmNemoRLBackend(
            account="test", partition="test", nodes=1, gpus_per_node=4, run_time="00:05:00"
        )
        return backend.policy_section(_experiment(**kwargs))

    def test_absent_when_no_checkpoint_is_configured(self) -> None:
        """NeMo-RL branches on the key's presence, not on its contents.

        An empty block sends it to the Megatron loader, which then resolves '' as a
        checkpoint root instead of importing the HF model in `model_name`.
        """
        self.assertNotIn("pretrained_checkpoint", self._policy(pretrained_checkpoint_path=""))

    def test_present_when_a_checkpoint_is_configured(self) -> None:
        policy = self._policy(
            pretrained_checkpoint_path="/ckpt/iter_0000100",
            pretrained_checkpoint_format="megatron_bridge",
        )

        self.assertEqual(
            policy["pretrained_checkpoint"],
            {"format": "megatron_bridge", "path": "/ckpt/iter_0000100"},
        )


if __name__ == "__main__":
    unittest.main()
