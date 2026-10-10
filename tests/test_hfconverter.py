import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.conversion.hfconverter import HFConverterConfig, render_command


class HFConverterTest(unittest.TestCase):
    def setUp(self):
        scratch = Path(os.environ.get("SCRATCH", "/tmp")) / "tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.git("init", "-q")
        self.git("config", "user.name", "Test")
        self.git("config", "user.email", "test@example.invalid")
        (self.source / "convert_checkpoint.py").write_text("# pinned converter\n")
        self.git("add", ".")
        self.git("commit", "-qm", "fixture")
        self.commit = self.git("rev-parse", "HEAD").stdout.strip()
        self.cfg = HFConverterConfig(
            hfconverter_root=str(self.source),
            hfconverter_commit=self.commit,
            megatron_root=str(self.source),
            megatron_commit=self.commit,
            bridge_root=str(self.source),
            bridge_commit=self.commit,
            checkpoint=str(self.root / "native/iter_0008500"),
            reference=str(self.root / "reference"),
            output=str(self.root / "output"),
            source_cache=str(self.root / "cache"),
            options=[
                "--tp",
                "1",
                "--pp",
                "4",
                "--ep",
                "8",
                "--pipeline-layout",
                "Et*16|t*15|t*15|t*15L",
            ],
        )

    def git(self, *args):
        return subprocess.run(
            ["git", "-C", str(self.source), *args], capture_output=True, text=True, check=True
        )

    def test_commit_worktree_preserves_dirty_source(self):
        script = self.source / "convert_checkpoint.py"
        script.write_text("# unsaved work\n")
        command = render_command(self.cfg)
        pinned = Path(command[4])
        self.assertEqual(command[:4], ["uv", "run", "--no-project", "python"])
        self.assertEqual(pinned.read_text(), "# pinned converter\n")
        self.assertEqual(script.read_text(), "# unsaved work\n")
        self.assertEqual(command[5], "export")
        self.assertEqual(command[9:-4], self.cfg.options)
        self.assertEqual(render_command(self.cfg), command)

    def test_missing_commit_is_rejected(self):
        self.cfg.hfconverter_commit = ""
        with self.assertRaisesRegex(ValueError, "Git commit is required"):
            render_command(self.cfg)

    def test_changed_cached_checkout_is_not_reset(self):
        command = render_command(self.cfg)
        pinned = Path(command[4])
        pinned.write_text("# changed cache\n")
        with self.assertRaisesRegex(ValueError, "local changes"):
            render_command(self.cfg)
        self.assertEqual(pinned.read_text(), "# changed cache\n")

    def test_untracked_code_in_cached_checkout_is_rejected(self):
        command = render_command(self.cfg)
        extra = Path(command[4]).parent / "untracked_module.py"
        extra.write_text("# extra executable code\n")
        with self.assertRaisesRegex(ValueError, "local changes"):
            render_command(self.cfg)
        self.assertTrue(extra.exists())


if __name__ == "__main__":
    unittest.main()
