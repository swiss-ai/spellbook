import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evals.hf_conversion import (
    HFConversionConfig,
    output_is_complete,
    render_submission,
    submit,
)


class HFConversionTest(unittest.TestCase):
    def _config(self, root: Path) -> HFConversionConfig:
        converter = root / "hfconverter"
        cluster = converter / "cluster"
        cluster.mkdir(parents=True)
        convert = cluster / "convert.sh"
        convert.write_text("#!/bin/bash\n")
        convert.chmod(0o755)
        (cluster / "stage2_export.sbatch").touch()

        checkpoint = root / "checkpoints" / "iter_0000001"
        checkpoint.mkdir(parents=True)
        (checkpoint / "common.pt").touch()
        tokenizer = root / "tokenizer"
        tokenizer.mkdir()
        (tokenizer / "tokenizer.json").write_text("{}")

        return HFConversionConfig(
            hfconverter_root=str(converter),
            checkpoint_dir=str(checkpoint),
            output_dir=str(root / "hf" / "step_1"),
            tokenizer_dir=str(tokenizer),
            account="infra01",
            partition="normal",
            reservation="reservation",
            log_dir=str(root / "logs"),
        )

    def _commit_converter(self, cfg: HFConversionConfig, message: str) -> str:
        root = Path(cfg.hfconverter_root)
        if not (root / ".git").exists():
            subprocess.run(["git", "init", "-b", "main", root], check=True)
            subprocess.run(
                ["git", "-C", root, "config", "user.email", "test@example.com"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", root, "config", "user.name", "Test"], check=True
            )
        subprocess.run(["git", "-C", root, "add", "."], check=True)
        subprocess.run(
            ["git", "-C", root, "commit", "--allow-empty", "-m", message],
            check=True,
        )
        return subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    def test_renders_fixed_hfconverter_interface(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            cfg = self._config(Path(raw))
            cfg.max_shard_size = "2GB"
            command, environment = render_submission(cfg)

            self.assertEqual(command[1:3], [cfg.checkpoint_dir, cfg.output_dir])
            self.assertIn("--parsable", command)
            self.assertIn("--account=infra01", command)
            self.assertIn("--partition=normal", command)
            self.assertIn("--reservation=reservation", command)
            self.assertEqual(environment["REPO"], cfg.hfconverter_root)
            self.assertEqual(environment["TOKENIZER_DIR"], cfg.tokenizer_dir)
            self.assertEqual(environment["VERIFY_LOAD"], "1")
            self.assertEqual(
                environment["EXTRA_EXPORT_ARGS"], "--max-shard-size 2GB"
            )

    def test_reuses_completed_output_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            cfg = self._config(Path(raw))
            output = Path(cfg.output_dir)
            output.mkdir(parents=True)
            (output / "conversion_info.json").write_text("{}")

            self.assertTrue(output_is_complete(output))
            self.assertIsNone(submit(cfg))

    def test_partial_output_requires_explicit_recreation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            cfg = self._config(Path(raw))
            output = Path(cfg.output_dir)
            output.mkdir(parents=True)
            (output / ".export_incomplete").touch()

            with self.assertRaisesRegex(ValueError, "recreate=True"):
                submit(cfg)

    def test_recreate_removes_output_and_submits_fresh_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            cfg = self._config(Path(raw))
            cfg.recreate = True
            output = Path(cfg.output_dir)
            output.mkdir(parents=True)
            (output / "stale.safetensors").touch()
            completed = subprocess.CompletedProcess([], 0, stdout="123;cluster\n")

            with patch("evals.hf_conversion.subprocess.run", return_value=completed) as run:
                self.assertEqual(submit(cfg), "123")

            self.assertFalse(output.exists())
            run.assert_called_once()
            self.assertTrue((Path(cfg.log_dir)).is_dir())

    def test_recreate_never_deletes_checkpoint_tree(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            cfg = self._config(Path(raw))
            cfg.output_dir = str(Path(cfg.checkpoint_dir).parent)
            cfg.recreate = True

            with self.assertRaisesRegex(ValueError, "must not contain one another"):
                submit(cfg)
            self.assertTrue(Path(cfg.checkpoint_dir).exists())

    def test_recreate_rejects_symlinked_output(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            cfg = self._config(Path(raw))
            target = Path(raw) / "valuable"
            target.mkdir()
            (target / "weights").touch()
            output = Path(cfg.output_dir)
            output.parent.mkdir()
            output.symlink_to(target, target_is_directory=True)
            cfg.recreate = True

            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                submit(cfg)
            self.assertTrue((target / "weights").exists())

    def test_expected_commit_is_checked(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            cfg = self._config(Path(raw))
            expected = self._commit_converter(cfg, "expected")
            self._commit_converter(cfg, "actual")
            cfg.hfconverter_commit = expected

            with self.assertRaisesRegex(ValueError, f"expected {expected}"):
                render_submission(cfg)

    def test_git_url_cache_locations(self) -> None:
        for custom_cache in (False, True):
            with self.subTest(custom_cache=custom_cache), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                cfg = self._config(root)
                commit = self._commit_converter(cfg, "initial")
                cfg.hfconverter_root = Path(cfg.hfconverter_root).as_uri()
                cfg.hfconverter_commit = commit
                scratch = root / "scratch"
                expected_root = scratch / "tmp"
                if custom_cache:
                    expected_root = root / "custom-cache"
                    cfg.hfconverter_cache_dir = str(expected_root)

                with patch.dict("os.environ", {"SCRATCH": str(scratch)}):
                    command, _ = render_submission(cfg)

                checkout = Path(command[0]).parent.parent
                self.assertTrue(
                    checkout.is_relative_to(expected_root / "hfconverter_worktrees")
                )
                self.assertTrue((expected_root / "hfconverter_repos").is_dir())
                self.assertEqual(
                    subprocess.run(
                        ["git", "-C", checkout, "rev-parse", "HEAD"],
                        capture_output=True,
                        text=True,
                        check=True,
                    ).stdout.strip(),
                    commit,
                )


if __name__ == "__main__":
    unittest.main()
