import hashlib
import subprocess
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Any

from spellbook.backends import SlurmBackend
from spellbook.backends.slurm_megatron.create_data_config import create_data_prefix
from spellbook.megatron import MegatronExperiment


def _backend(**kwargs: Any) -> SlurmBackend:
    return SlurmBackend(
        account="test",
        partition="test",
        nodes=1,
        gpus_per_node=1,
        run_time="00:05:00",
        **kwargs,
    )


def _container_backend(**kwargs: Any) -> SlurmBackend:
    return SlurmBackend(
        account="test",
        partition="test",
        nodes=1,
        gpus_per_node=1,
        run_time="00:05:00",
        extra={"container_edf": "test-container"},
        **kwargs,
    )


def _experiment(**kwargs: Any) -> MegatronExperiment:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MegatronExperiment(name="data-test", **kwargs)


class DataPathTest(unittest.TestCase):
    def test_megatron_is_copied_inside_every_backend_launch_mode(self) -> None:
        experiment = _experiment(
            megatron_path="/source/Megatron-LM", data_path="/data/prefix"
        )
        backends = [
            _container_backend(),
            _container_backend(launch_mode="tasks"),
            _container_backend(srun_job_id="123"),
            _container_backend(mem_estimator=True),
        ]

        for backend in backends:
            with self.subTest(template=backend._template_name()):
                script = backend.render(experiment)
                copy_position = script.index(
                    "cp -R --no-preserve=all"
                )
                self.assertGreater(copy_position, script.index("srun \\"))
                self.assertIn(
                    'export MEGATRON_PATH="${MEGATRON_CONTAINER_PATH}"', script
                )

    def test_megatron_is_not_copied_without_a_container(self) -> None:
        experiment = _experiment(
            megatron_path="/source/Megatron-LM", data_path="/data/prefix"
        )

        script = _backend().render(experiment)

        self.assertNotIn("cp -R --no-preserve=all", script)
        self.assertNotIn('MEGATRON_CONTAINER_PATH="/opt/megatron"', script)
        self.assertIn('export MEGATRON_PATH="${MEGATRON_SOURCE_PATH}"', script)

    def test_git_url_uses_source_specific_worktree_cache(self) -> None:
        url = "https://example.com/megatron.git"
        experiment = _experiment(
            megatron_path=url,
            megatron_commit="abc123",
            data_path="/data/prefix",
        )

        script = _container_backend().render(experiment)
        worktree_key = hashlib.sha256(f"{url}\0abc123".encode()).hexdigest()[:16]

        self.assertIn('fetch origin --tags --prune', script)
        self.assertIn(f"megatron_worktrees/{worktree_key}", script)

    def test_task_mode_serializes_install_commands_per_node(self) -> None:
        experiment = _experiment(
            megatron_path="/source/Megatron-LM",
            data_path="/data/prefix",
            install_commands="echo install",
        )

        script = _container_backend(launch_mode="tasks").render(experiment)

        self.assertIn("spellbook-training-install-", script)
        self.assertIn('if [[ "${SLURM_LOCALID:-0}" == "0" ]]', script)
        self.assertIn("Package installation failed", script)

    def test_auto_requeue_can_cancel_successor_on_completion_regex(self) -> None:
        experiment = _experiment(
            megatron_path="/source/Megatron-LM", data_path="/data/prefix"
        )
        regex = r"training (finished|model's complete)$"

        for launch_mode in ("torchrun", "tasks"):
            with self.subTest(launch_mode=launch_mode):
                script = _backend(
                    launch_mode=launch_mode,
                    auto_requeue=True,
                    auto_requeue_stop_regex=regex,
                ).render(experiment)

                self.assertIn("sbatch --parsable --dependency=singleton", script)
                self.assertIn(
                    'AUTO_REQUEUE_JOB_ID="${AUTO_REQUEUE_SUBMISSION%%;*}"', script
                )
                self.assertIn('scontrol show job "$SLURM_JOB_ID" -o', script)
                self.assertIn("grep -Eq --", script)
                self.assertIn('scancel "${AUTO_REQUEUE_JOB_ID}"', script)
                self.assertIn('exit "${SRUN_EXIT_CODE}"', script)
                subprocess.run(
                    ["bash", "-n"],
                    input=script,
                    text=True,
                    check=True,
                )

    def test_auto_requeue_uses_megatron_completion_marker_by_default(self) -> None:
        experiment = _experiment(
            megatron_path="/source/Megatron-LM", data_path="/data/prefix"
        )

        script = _backend(auto_requeue=True).render(experiment)

        self.assertIn(r"grep -Eq -- '\[after training is done\] datetime:'", script)
        self.assertIn('scancel "${AUTO_REQUEUE_JOB_ID}"', script)

    def test_auto_requeue_completion_check_requires_both_settings(self) -> None:
        experiment = _experiment(
            megatron_path="/source/Megatron-LM", data_path="/data/prefix"
        )

        disabled_regex = _backend(
            auto_requeue=True, auto_requeue_stop_regex=""
        ).render(experiment)
        without_requeue = _backend(
            auto_requeue_stop_regex="training finished"
        ).render(experiment)

        self.assertNotIn("grep -Eq --", disabled_regex)
        self.assertNotIn("grep -Eq --", without_requeue)
        self.assertNotIn("scancel", disabled_regex)
        self.assertNotIn("scancel", without_requeue)

    def test_discovery_can_follow_symlinked_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shards = root / "shards"
            shards.mkdir()
            (shards / "part.bin").touch()
            linked_root = root / "dataset"
            linked_root.mkdir()
            (linked_root / "linked-shards").symlink_to(shards, target_is_directory=True)

            self.assertEqual(create_data_prefix([str(linked_root)]), [])
            self.assertEqual(
                create_data_prefix([str(linked_root)], follow_symlinks=True),
                [str(linked_root / "linked-shards" / "part")],
            )

    def test_existing_data_args_path_is_passed_without_warning(self) -> None:
        experiment = _experiment(data_args_path="/data/manifest.txt")

        with warnings.catch_warnings(record=True) as caught:
            script = _backend().render(experiment)

        self.assertIn("--data-args-path /data/manifest.txt", script)
        self.assertFalse(
            any("no data path" in str(warning.message).lower() for warning in caught)
        )

    def test_saved_render_generates_manifest_for_base_data_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shards = root / "shards"
            shards.mkdir()
            (shards / "b.bin").touch()
            (shards / "a.bin").touch()
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "linked").symlink_to(shards, target_is_directory=True)
            output_dir = root / "rendered"
            experiment = _experiment(
                base_data_path=str(dataset),
                follow_symlinks=True,
            )

            paths = _backend().render_all(
                "test-sweep", [experiment], output_dir=str(output_dir)
            )

            script_path = Path(paths[0])
            manifest_path = script_path.with_suffix(".data_args.txt")
            self.assertEqual(
                manifest_path.read_text().splitlines(),
                [
                    str(dataset / "linked" / "a"),
                    str(dataset / "linked" / "b"),
                ],
            )
            script = script_path.read_text()
            self.assertIn(f"--data-args-path {manifest_path.resolve()}", script)
            self.assertIn('DATA_PATH=""', script)

    def test_explicit_data_path_takes_precedence(self) -> None:
        experiment = _experiment(
            data_path="/data/prefix",
            data_args_path="/data/manifest.txt",
        )

        script = _backend().render(experiment)

        self.assertIn("--data-path ${DATA_PATH}", script)
        self.assertNotIn("--data-args-path", script)


if __name__ == "__main__":
    unittest.main()
