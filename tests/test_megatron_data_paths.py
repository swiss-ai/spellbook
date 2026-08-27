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
            _container_backend(theoretical_memory=True),
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

    def test_theoretical_memory_uses_megatron_report_tool(self) -> None:
        experiment = _experiment(
            megatron_path="/users/anowak/open_source/Megatron-LM-MoE",
            data_path="/data/prefix",
        )

        script = _backend(theoretical_memory=True).render(experiment)

        self.assertIn(
            'python "${MEGATRON_PATH}/tools/report_theoretical_memory.py"', script
        )
        self.assertNotIn("estimate_013.py", script)
        self.assertNotIn("--fake-process-group", script)
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_memory_estimator_modes_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            _backend(mem_estimator=True, theoretical_memory=True)._template_name()

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

    def test_kernel_cache_defaults_to_full_identity(self) -> None:
        experiment = _experiment(
            megatron_path="/source/Megatron-LM",
            data_path="/data/prefix",
        )

        script = _container_backend(
            launch_mode="tasks",
            kernel_cache=True,
        ).render(experiment)

        self.assertIn(
            'SPELLBOOK_KERNEL_CACHE_ROOT="${SCRATCH:-/iopsstor/scratch/cscs/$USER}/tmp/spellbook/kernel-cache"',
            script,
        )
        self.assertIn('container=${SPELLBOOK_KERNEL_CACHE_CONTAINER}', script)
        self.assertIn('megatron=${MEGATRON_GIT_COMMIT:-unknown}', script)
        self.assertIn('printf "%s\\n" "experiment=', script)
        self.assertIn('SPELLBOOK_LOCAL_CACHE_ROOT="/tmp/spellbook-kernel-cache/', script)
        self.assertIn('export TRITON_CACHE_DIR="${TRITON_HOME}/cache"', script)
        self.assertIn('export TORCHINDUCTOR_CACHE_DIR=', script)
        self.assertIn('touch "${SPELLBOOK_KERNEL_CACHE_PATH}/.complete"', script)
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_kernel_cache_key_and_warmup_are_configurable(self) -> None:
        experiment = _experiment(
            megatron_path="/source/Megatron-LM",
            data_path="/data/prefix",
            gbs=8,
            extra_args=["--exit-interval", "100"],
        )

        script = _container_backend(
            kernel_cache=True,
            kernel_cache_key_fields=("container", "megatron"),
            kernel_cache_key_values={"fla": "0.5.2"},
            kernel_cache_warmup_steps=6,
        ).render(experiment)

        self.assertIn('megatron=${MEGATRON_GIT_COMMIT:-unknown}', script)
        self.assertIn("--exit-interval 6", script)
        self.assertIn("--train-samples 48", script)
        self.assertNotIn("--exit-interval 100", script)
        self.assertIn("Kernel cache already exists; skipping warmup", script)
        self.assertGreater(
            script.index('trap "spellbook_kernel_cache_save'),
            script.index("Kernel cache already exists; skipping warmup"),
        )
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_kernel_cache_rejects_invalid_configuration(self) -> None:
        experiment = _experiment(data_path="/data/prefix")

        with self.assertRaisesRegex(ValueError, "requires kernel_cache=True"):
            _backend(kernel_cache_warmup_steps=6).render(experiment)
        with self.assertRaisesRegex(ValueError, "Unsupported kernel_cache_key_fields"):
            _backend(
                kernel_cache=True,
                kernel_cache_key_fields=("python",),
            ).render(experiment)
        with self.assertRaisesRegex(ValueError, "cannot use auto_requeue"):
            _backend(
                kernel_cache=True,
                kernel_cache_warmup_steps=6,
                auto_requeue=True,
            ).render(experiment)

    def test_task_mode_can_profile_selected_launcher_ranks_with_custom_nsys_args(self) -> None:
        experiment = _experiment(
            megatron_path="/source/Megatron-LM",
            data_path="/data/prefix",
            profile=True,
            profile_ranks=[64],
            nsys_launcher_ranks=[64],
            nsys_profile_args=[
                "-s none",
                "--trace=nvtx,cudnn,cublas,cuda",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop",
                "--force-overwrite=true",
            ],
            nsys_tmpdir="${SLURM_TMPDIR:-/tmp}",
            nsys_output="/reports/${SLURM_JOB_ID}",
        )

        script = _container_backend(launch_mode="tasks").render(experiment)

        self.assertIn('NSYS_LAUNCHER_RANKS=" 64 "', script)
        self.assertIn(
            'if [[ "${NSYS_LAUNCHER_RANKS}" == *" ${SLURM_PROCID} "* ]]',
            script,
        )
        self.assertIn('export NSYS_TMPDIR="${SLURM_TMPDIR:-/tmp}"', script)
        self.assertIn("--trace=nvtx,cudnn,cublas,cuda", script)
        self.assertNotIn("--cuda-graph-trace=node", script)
        self.assertIn("--profile-ranks 64", script)
        self.assertIn("/reports/${SLURM_JOB_ID}/data-test-${SLURM_PROCID}", script)
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_task_mode_profiles_every_launcher_rank_by_default(self) -> None:
        experiment = _experiment(
            megatron_path="/source/Megatron-LM",
            data_path="/data/prefix",
            profile=True,
        )

        script = _container_backend(launch_mode="tasks").render(experiment)

        self.assertNotIn("NSYS_LAUNCHER_RANKS", script)
        self.assertIn("--cuda-graph-trace=node", script)
        self.assertIn('PROFILE_CMD="nsys profile', script)

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
