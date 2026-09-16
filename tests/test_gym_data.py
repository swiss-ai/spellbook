from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.gym_data import GymDataConfig, render
from tools.gym_data.prepare import head_deps


class GymDataTest(unittest.TestCase):
    def test_head_deps_falls_back_when_openai_is_absent(self) -> None:
        def version(name: str) -> str:
            if name == "openai":
                raise __import__("importlib").metadata.PackageNotFoundError(name)
            return "2.55.1"

        with patch("tools.gym_data.prepare.md.version", side_effect=version):
            self.assertEqual(head_deps(), ["ray[default]==2.55.1", "openai==2.7.2"])

    def test_default_scheduler_logs_are_outside_checkout(self) -> None:
        cfg = GymDataConfig(
            name="gym-data-test",
            nemo_rl_path="/src/Nemo-RL",
            gym_home="/src/Gym",
            config_paths=["resources_servers/test/config.yaml"],
        )

        self.assertTrue(Path(cfg.log_dir).is_absolute())
        self.assertEqual(
            cfg.log_dir,
            str(
                Path(
                    os.environ.get(
                        "SCRATCH",
                        f"/iopsstor/scratch/cscs/{os.environ.get('USER', 'unknown')}",
                    )
                )
                / "tmp/spellbook/nemorl/gym_data/slurm_logs"
            ),
        )

    def test_squashfs_is_built_outside_the_container(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gym_home = root / "Gym"
            gym_home.mkdir()
            script = render(
                GymDataConfig(
                    name="gym-data-test",
                    nemo_rl_path="/src/Nemo-RL",
                    gym_home=str(gym_home),
                    config_paths=["resources_servers/test/config.yaml"],
                    venv_dir=str(root / "venvs"),
                    nemo_rl_venv_dir=str(root / "actor-venvs"),
                    uv_cache_dir=str(root / "uv-cache"),
                    scratch_root=str(root / "gym-root"),
                    output_dir=str(root / "data"),
                    squashfs=str(root / "venvs.sqsh"),
                    container="image.toml",
                )
            )

        srun_end = script.index("    ;")
        self.assertNotIn("--squashfs", script[:srun_end])
        self.assertGreater(script.index("mksquashfs"), srun_end)
        self.assertIn("set -euxo pipefail", script)
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)


if __name__ == "__main__":
    unittest.main()
