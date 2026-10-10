import argparse
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evals import watcher
from evals.megatron_eval import render_watcher_script
from evals.pipeline import CheckpointPipelineConfig, submit


class EvalPipelineTest(unittest.TestCase):
    def setUp(self):
        scratch = Path(os.environ.get("SCRATCH", "/tmp")) / "tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.script = self.root / "pipeline.sbatch"
        self.script.write_text("#!/bin/bash\nexit 0\n")
        self.cfg = CheckpointPipelineConfig(
            model_name="megachonk",
            checkpoint_dir=str(self.root / "checkpoints"),
            script=str(self.script),
            account="csstaff",
            partition="normal",
            log_dir=str(self.root / "logs"),
            watcher_dir=str(self.root / "watchers"),
        )

    def test_submit_passes_checkpoint_arguments(self):
        result = subprocess.CompletedProcess([], 0, "123;cluster\n", "")
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("evals.pipeline.subprocess.run", return_value=result) as run,
        ):
            self.assertEqual(submit(self.cfg, 8500, dependency_singleton=True), "123")
        command = run.call_args.args[0]
        self.assertIn("--dependency=singleton", command)
        self.assertIn("--account=csstaff", command)
        self.assertEqual(
            command[-7:],
            [
                str(self.script),
                "--step",
                "8500",
                "--checkpoint-dir",
                self.cfg.checkpoint_dir,
                "--model",
                "megachonk",
            ],
        )

    def test_existing_watcher_renders_pipeline_checkpoint_root(self):
        config = self.root / "config.py"
        config.write_text("# config fixture\n")
        path = render_watcher_script(self.cfg, str(config), project_dir=self.root)
        script = path.read_text()
        self.assertIn(
            f'MARKER="{self.cfg.checkpoint_dir}/latest_checkpointed_iteration.txt"', script
        )
        self.assertIn(".submitted_steps", script)
        subprocess.run(["bash", "-n", str(path)], check=True)

    def test_existing_check_dispatches_pipeline_and_records_submission(self):
        checkpoint = Path(self.cfg.checkpoint_dir)
        checkpoint.mkdir()
        (checkpoint / "latest_checkpointed_iteration.txt").write_text("8500")
        args = argparse.Namespace(
            config="fixture",
            model=None,
            group=None,
            eval_job_name=None,
            step=None,
            dependency_singleton=True,
            consumed_tokens_per_step=None,
        )
        with (
            patch.object(
                watcher,
                "_load_config",
                return_value=(self.cfg, str(checkpoint), str(self.root / "state")),
            ),
            patch.object(watcher, "submit_pipeline", return_value="123") as run,
        ):
            watcher.cmd_check(args)
            watcher.cmd_check(args)
        run.assert_called_once_with(self.cfg, 8500, dependency_singleton=True)
        self.assertEqual((self.root / "state/megachonk/.submitted_steps").read_text(), "8500\n")


if __name__ == "__main__":
    unittest.main()
