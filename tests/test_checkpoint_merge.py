import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from tools.merge import MegatronCheckpointMergeConfig, render, submit


def _config(**changes: Any) -> MegatronCheckpointMergeConfig:
    values: dict[str, Any] = {
        "name": "chonk-merged",
        "checkpoints": ["/checkpoints/chonk"],
        "checkpoint_steps": [1000, 2000],
        "output": "/checkpoints/chonk-merged",
        "megatron_path": "/workspace/Megatron-LM",
        "account": "infra01",
        "partition": "normal",
    }
    values.update(changes)
    return MegatronCheckpointMergeConfig(**values)


class CheckpointMergeTest(unittest.TestCase):
    def test_renders_generic_distributed_workers(self) -> None:
        script = render(_config(nodes=2, workers_per_node=3))

        self.assertIn("#SBATCH --ntasks-per-node=3", script)
        self.assertIn("--ntasks=6", script)
        self.assertIn('"${MEGATRON_PATH}/tools/checkpoint/merge.py"', script)
        self.assertIn('PYTHONPATH="${MEGATRON_PATH}:${PYTHONPATH:-}"', script)
        self.assertIn(
            'numactl --cpunodebind="${SLURM_LOCALID}" --membind="${SLURM_LOCALID}"',
            script,
        )
        self.assertIn('"--checkpoints" \\\n  "/checkpoints/chonk"', script)
        self.assertIn('"--checkpoint-steps" \\\n  "1000" \\\n  "2000"', script)
        self.assertIn('"--backend" \\\n  "gloo"', script)
        self.assertNotIn("#SBATCH --gpus-per-node", script)
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_renders_pinned_megatron_worktree(self) -> None:
        script = render(_config(megatron_commit="abc123"))

        self.assertIn('MEGATRON_REQUESTED_REF="abc123"', script)
        self.assertIn("worktree add --detach", script)
        self.assertIn("MEGATRON_TARGET_COMMIT", script)
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_renders_megatron_url_checkout(self) -> None:
        script = render(_config(megatron_path="https://example.com/Megatron-LM.git"))

        self.assertIn("Cloning Megatron-LM from https://example.com/Megatron-LM.git", script)
        self.assertIn("git clone", script)
        self.assertIn('MEGATRON_REQUESTED_REF="refs/remotes/origin/HEAD"', script)
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_renders_linear_decay_and_container_options(self) -> None:
        script = render(
            _config(
                merge_method="linear-decay",
                target_end_multiplier=0.2,
                container_edf="apertus2-alps4-temp",
                container_mounts="/checkpoints:/checkpoints",
                srun_extra_args="--network=disable_rdzv_get",
            )
        )

        self.assertIn('"--merge-method" \\\n  "linear-decay"', script)
        self.assertIn('"--target-end-multiplier" \\\n  "0.2"', script)
        self.assertIn('--environment="apertus2-alps4-temp"', script)
        self.assertIn('--container-mounts="/checkpoints:/checkpoints"', script)
        self.assertIn("--network=disable_rdzv_get", script)
        self.assertIn('MEGATRON_CONTAINER_PATH="/opt/megatron"', script)
        self.assertIn('rm -rf "${MEGATRON_CONTAINER_PATH}"', script)
        self.assertIn('export MEGATRON_PATH="${MEGATRON_CONTAINER_PATH}"', script)
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_submit_creates_log_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(log_dir=str(Path(tmp) / "logs"))
            with patch("tools.merge.checkpoint._sbatch", return_value="123"):
                self.assertEqual(submit(cfg), "123")
            self.assertTrue(Path(cfg.log_dir).is_dir())

    def test_rejects_invalid_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            render(_config(checkpoints=["/one"], checkpoint_steps=[]))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            render(_config(checkpoint_steps=[2000, 1000]))


if __name__ == "__main__":
    unittest.main()
