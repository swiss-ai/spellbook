import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from tools.conversion.megatron_bridge import BridgeExportConfig, launch, render_command


class MegatronBridgeConversionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "bridge"
        launcher = self.root / "scripts/conversion/convert.sh"
        launcher.parent.mkdir(parents=True)
        launcher.write_text("#!/bin/bash\n")
        launcher.chmod(0o755)
        self.megatron_root = Path(self.tmp.name) / "Megatron-LM"
        (self.megatron_root / "megatron/core").mkdir(parents=True)
        self.checkpoint = Path(self.tmp.name) / "checkpoint/iter_0001000"
        self.checkpoint.mkdir(parents=True)
        (self.checkpoint / "run_config.yaml").write_text("model: {}\n")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def config(self, **changes: Any) -> BridgeExportConfig:
        values: dict[str, Any] = {
            "bridge_root": str(self.root),
            "hf_model": "/models/reference",
            "megatron_path": str(self.checkpoint),
            "hf_path": "/exports/model-hf",
            "container_image": "/images/megatron-bridge.sqsh",
            "nodes": 8,
            "gpus_per_node": 4,
            "ep": 32,
            "account": "csstaff",
            "partition": "preemptable",
            "mounts": ["/checkpoints", "/exports", "/models"],
            "env_names": [],
            "export_weight_dtype": "bfloat16",
        }
        values.update(changes)
        return BridgeExportConfig(**values)

    def test_renders_native_distributed_export_pipeline(self) -> None:
        command = render_command(self.config())

        self.assertEqual(command[1:5], ["export", "--executor", "slurm", "--device"])
        self.assertIn("gpu", command)
        self.assertIn("--nodes", command)
        self.assertIn("8", command)
        self.assertIn("--ep", command)
        self.assertIn("32", command)
        self.assertIn("--distributed-save", command)
        self.assertIn("--export-weight-dtype", command)
        self.assertIn("bfloat16", command)
        self.assertIn("--detach", command)
        bridge_mount = f"{self.root.resolve()}:/opt/Megatron-Bridge"
        self.assertEqual(command[command.index(bridge_mount) - 1], "--mount")
        self.assertIn("--srun-arg=--cpus-per-task=72", command)
        self.assertIn("--srun-arg=--network=disable_rdzv_get", command)
        self.assertIn("--srun-arg=--mpi=pmix", command)

    def test_mounts_selected_megatron_checkout(self) -> None:
        command = render_command(self.config(megatron_root=str(self.megatron_root)))

        mount = f"{self.megatron_root.resolve()}:/opt/Megatron-Bridge/3rdparty/Megatron-LM"
        self.assertEqual(command[command.index(mount) - 1], "--mount")

    def test_megatron_commit_requires_checkout(self) -> None:
        with self.assertRaisesRegex(ValueError, "megatron_commit requires megatron_root"):
            render_command(self.config(megatron_commit="deadbeef"))

    def test_launch_uses_isolated_bridge_environment(self) -> None:
        cfg = self.config()
        with (
            patch.dict("os.environ", {"VIRTUAL_ENV": "/spellbook/.venv"}),
            patch("tools.conversion.megatron_bridge.subprocess.run") as run,
        ):
            launch(cfg)

        self.assertEqual(run.call_args.args[0], render_command(cfg))
        self.assertNotIn("VIRTUAL_ENV", run.call_args.kwargs["env"])
        self.assertTrue(run.call_args.kwargs["text"])
        self.assertTrue(run.call_args.kwargs["check"])

    def test_requires_bridge_run_config(self) -> None:
        (self.checkpoint / "run_config.yaml").unlink()
        with self.assertRaisesRegex(ValueError, "run_config.yaml"):
            render_command(self.config())

    def test_rejects_invalid_parallelism(self) -> None:
        with self.assertRaisesRegex(ValueError, "nodes.*tp"):
            render_command(self.config(ep=16))
        with self.assertRaisesRegex(ValueError, "save_every_n_ranks"):
            render_command(self.config(distributed_save=False, save_every_n_ranks=2))


if __name__ == "__main__":
    unittest.main()
