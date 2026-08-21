import argparse
import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evals.megatron_eval import (
    AllocationMode,
    LMEvalRunConfig,
    MegatronEvalConfig,
    _clean_sbatch_env,
    _is_megatron_url,
    _render,
    render_watcher_script,
    submit,
    submit_evaluations,
)
from evals.watcher import _consumed_tokens, _load_config, cmd_start, cmd_stop


def _config(megatron_path: str, megatron_commit: str = "") -> MegatronEvalConfig:
    return MegatronEvalConfig(
        model_name="model",
        checkpoint_dir="/checkpoints",
        tokenizer_model="tokenizer",
        megatron_path=megatron_path,
        megatron_commit=megatron_commit,
        tasks=["hellaswag"],
    )


class MegatronPathTest(unittest.TestCase):
    def test_recognizes_git_urls(self) -> None:
        self.assertTrue(_is_megatron_url("https://github.com/NVIDIA/Megatron-LM.git"))
        self.assertTrue(_is_megatron_url("git@github.com:NVIDIA/Megatron-LM.git"))
        self.assertFalse(_is_megatron_url("/path/to/Megatron-LM"))

    def test_local_path_is_used_without_clone_or_container_copy(self) -> None:
        script = _render(_config("/path/to/Megatron-LM"), 10, False)

        self.assertIn('MEGATRON_SOURCE_PATH="/path/to/Megatron-LM"', script)
        self.assertNotIn("git clone", script)
        self.assertNotIn("cp -R --no-preserve=all", script)
        self.assertNotIn('MEGATRON_CONTAINER_PATH="/opt/megatron"', script)
        self.assertIn('export MEGATRON_PATH="${MEGATRON_SOURCE_PATH}"', script)

    def test_container_launch_copies_megatron_after_srun(self) -> None:
        cfg = _config("/path/to/Megatron-LM")
        cfg.container_edf = "container"
        script = _render(cfg, 10, False)

        copy_position = script.index("cp -R --no-preserve=all")
        self.assertGreater(copy_position, script.index("srun \\"))
        self.assertIn('MEGATRON_CONTAINER_PATH="/opt/megatron"', script)
        self.assertIn('export MEGATRON_PATH="${MEGATRON_CONTAINER_PATH}"', script)

    def test_container_copy_path_is_configurable_and_validated(self) -> None:
        cfg = _config("/path/to/Megatron-LM")
        cfg.container_edf = "container"
        cfg.megatron_container_path = "/workspace/megatron"
        script = _render(cfg, 10, False)

        self.assertIn('MEGATRON_CONTAINER_PATH="/workspace/megatron"', script)
        self.assertIn(
            "Megatron source and container destination must not contain one another",
            script,
        )

        for unsafe_path in ["/../", "/opt", "/opt/$(touch bad)"]:
            with self.subTest(unsafe_path=unsafe_path):
                cfg.megatron_container_path = unsafe_path
                with self.assertRaises(ValueError):
                    _render(cfg, 10, False)

    def test_url_cache_fetches_missing_commit_and_uses_isolated_worktree(self) -> None:
        url = "https://github.com/NVIDIA/Megatron-LM.git"
        script = _render(_config(url, "abc123"), 10, False)
        worktree_key = hashlib.sha256(f"{url}\0abc123".encode()).hexdigest()[:16]

        self.assertIn(f'git clone "{url}" "${{MEGATRON_REPO_PATH}}"', script)
        self.assertIn('fetch origin --tags --prune', script)
        self.assertIn(
            'MEGATRON_CACHE_ROOT="${SCRATCH:-${TMPDIR:-/tmp}/spellbook-${USER:-$(id -u)}}/tmp"',
            script,
        )
        self.assertIn(
            f'MEGATRON_WORKTREE_PATH="${{MEGATRON_CACHE_ROOT}}/megatron_worktrees/{worktree_key}"',
            script,
        )
        self.assertIn(
            'git -C "${MEGATRON_SOURCE_PATH}" worktree add --detach', script
        )
        self.assertNotIn(f'git -C "{url}"', script)

    def test_unpinned_url_refreshes_and_uses_remote_head_worktree(self) -> None:
        script = _render(_config("https://example.com/repo.git"), 10, False)

        self.assertIn("fetch origin --tags --prune", script)
        self.assertIn('MEGATRON_REQUESTED_REF="refs/remotes/origin/HEAD"', script)
        self.assertIn("worktree add --detach", script)

    def test_worktree_cache_key_includes_source_repository(self) -> None:
        first = _render(_config("https://example.com/first.git", "abc123"), 10, False)
        second = _render(_config("https://example.com/second.git", "abc123"), 10, False)

        first_key = hashlib.sha256(
            "https://example.com/first.git\0abc123".encode()
        ).hexdigest()[:16]
        second_key = hashlib.sha256(
            "https://example.com/second.git\0abc123".encode()
        ).hexdigest()[:16]
        self.assertIn(f"megatron_worktrees/{first_key}", first)
        self.assertIn(f"megatron_worktrees/{second_key}", second)
        self.assertNotEqual(first_key, second_key)

    def test_srun_extra_args_are_rendered(self) -> None:
        cfg = _config("/path/to/Megatron-LM")
        cfg.srun_extra_args = "--network=disable_rdzv_get --cpu-bind=none"

        script = _render(cfg, 10, False)

        self.assertIn(
            "  --network=disable_rdzv_get --cpu-bind=none \\\n  -u bash -lc '",
            script,
        )

    def test_sbatch_does_not_inherit_python_environment(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "PATH": "/project/.venv/bin:/usr/bin:/bin",
                "VIRTUAL_ENV": "/project/.venv",
                "PYTHONPATH": "/project/python",
                "PYTHONHOME": "/project/python-home",
                "WANDB_API_KEY": "keep-me",
            },
            clear=True,
        ):
            env = _clean_sbatch_env()

        self.assertEqual(env["PATH"], "/usr/bin:/bin")
        self.assertEqual(env["WANDB_API_KEY"], "keep-me")
        self.assertNotIn("VIRTUAL_ENV", env)
        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("PYTHONHOME", env)

    def test_dataset_prefetch_has_global_completion_barrier(self) -> None:
        cfg = _config("/path/to/Megatron-LM")
        cfg.dataset_prefetch = {"task": ["dataset"]}
        script = _render(cfg, 10, False)

        status = script.index("PREFETCH_STATUS=")
        wait = script.index('while [[ ! -f "${PREFETCH_STATUS}" ]]')
        offline = script.index("export HF_HUB_OFFLINE=1")
        self.assertLess(status, wait)
        self.assertLess(wait, offline)
        self.assertIn("Dataset prefetch failed with exit code", script)
        self.assertIn("timeout --signal=TERM 1800s python", script)
        self.assertNotIn("timeout --signal=TERM 1800s python3", script)
        self.assertIn("Timed out waiting for dataset prefetch status", script)

    def test_missing_dotenv_file_is_allowed(self) -> None:
        script = _render(_config("/path/to/Megatron-LM"), 10, False)

        self.assertIn('if [[ -f ".env" ]]; then', script)

    def test_multiple_lm_eval_runs_share_one_job(self) -> None:
        cfg = _config("/path/to/Megatron-LM")
        cfg.launch_mode = "tasks"
        cfg.nodes = 2
        cfg.gpus_per_node = 4
        cfg.wandb_project = "evals"
        cfg.wandb_id = "checkpoint"
        cfg.eval_runs = [
            LMEvalRunConfig(
                name="3shot",
                tasks=["mmlu"],
                lm_eval_args={"num_fewshot": 3},
                wandb_name="Core 3-shot",
                wandb_id="core-3shot",
            ),
            LMEvalRunConfig(
                name="zero-shot",
                tasks=["hellaswag", "arc_easy"],
                lm_eval_args={"num_fewshot": 0, "batch_size": 4},
                env_vars={"DISABLE_MULTIPROC": "1"},
                wandb_name="Core zero-shot",
            ),
        ]

        script = _render(cfg, 10, False)

        self.assertEqual(script.count("python -m lm_eval"), 2)
        self.assertIn("--tasks mmlu", script)
        self.assertIn("--num_fewshot 3", script)
        self.assertIn("--tasks hellaswag,arc_easy", script)
        self.assertIn("--num_fewshot 0", script)
        self.assertIn("--batch_size 4", script)
        self.assertIn("step_10/3shot", script)
        self.assertIn("step_10/zero-shot", script)
        self.assertIn('export DISABLE_MULTIPROC="1"', script)
        self.assertIn("id=core-3shot", script)
        self.assertIn("name=Core 3-shot", script)
        self.assertIn("id=checkpoint-zero-shot", script)
        self.assertIn("name=Core zero-shot", script)
        self.assertIn("suite_barrier 0", script)
        self.assertIn("suite_barrier 1", script)
        self.assertIn("< 8", script)
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_checkpoint_and_eval_run_allocation_modes_are_independent(self) -> None:
        cfg = _config("/path/to/Megatron-LM")
        cfg.wandb_project = "evals"
        cfg.eval_runs = [
            LMEvalRunConfig(name="first", tasks=["a"]),
            LMEvalRunConfig(name="second", tasks=["b"]),
        ]
        with patch(
            "evals.megatron_eval._sbatch", side_effect=["1", "2", "3", "4"]
        ) as sbatch:
            job_ids = submit_evaluations(
                cfg,
                [10, 20],
                checkpoint_mode=AllocationMode.SEPARATE,
                eval_mode=AllocationMode.SEPARATE,
            )

        self.assertEqual(job_ids, ["1", "2", "3", "4"])
        self.assertEqual(sbatch.call_count, 4)
        for call in sbatch.call_args_list:
            self.assertEqual(call.args[0].count("-m lm_eval"), 1)

        with patch("evals.megatron_eval._sbatch", return_value="5") as sbatch:
            job_ids = submit_evaluations(
                cfg,
                [10, 20],
                checkpoint_mode=AllocationMode.SHARED,
                eval_mode=AllocationMode.SHARED,
                consumed_tokens={10: 1000, 20: 2000},
            )

        self.assertEqual(job_ids, ["5"])
        script = sbatch.call_args.args[0]
        self.assertEqual(script.count("-m lm_eval"), 4)
        self.assertIn('export CKPT_STEP="10"', script)
        self.assertIn('export CKPT_STEP="20"', script)
        self.assertIn("step_10/first", script)
        self.assertIn("step_20/second", script)
        self.assertIn("consumed_tokens=1000", script)
        self.assertIn("consumed_tokens=2000", script)
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_eval_run_names_are_unique_and_renderer_arguments_win(self) -> None:
        cfg = _config("/path/to/Megatron-LM")
        cfg.eval_runs = [
            LMEvalRunConfig(name="same", tasks=["a"]),
            LMEvalRunConfig(name="same", tasks=["b"]),
        ]
        with self.assertRaisesRegex(ValueError, "unique"):
            _render(cfg, 10, False)

        cfg.eval_runs = [
            LMEvalRunConfig(
                name="run",
                tasks=["a"],
                lm_eval_args={
                    "model": "hf",
                    "model_args": "pretrained=wrong",
                    "tasks": ["wrong"],
                    "output_path": "/tmp/wrong",
                },
            )
        ]
        script = _render(cfg, 10, False)
        self.assertIn("--model megatron_lm", script)
        self.assertIn("--tasks a", script)
        self.assertNotIn("pretrained=wrong", script)
        self.assertNotIn("/tmp/wrong", script)

    def test_submit_creates_slurm_log_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config("/path/to/Megatron-LM")
            cfg.log_dir = str(Path(tmp) / "logs")
            with patch("evals.megatron_eval._sbatch", return_value="123"):
                self.assertEqual(submit(cfg, 10), "123")

            self.assertTrue((Path(cfg.log_dir) / cfg.model_name).is_dir())

    def test_watcher_uses_shared_state_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = _config("/path/to/Megatron-LM")
            watch_dir = root / "custom-checkpoints"
            state_dir = root / "custom-state"
            script_path = render_watcher_script(
                cfg,
                config_path=str(root / "config.py"),
                project_dir=root,
                watch_checkpoint_dir=str(watch_dir),
                config_model="selected",
                consumed_tokens_per_step=2048,
                watch_state_dir=str(state_dir),
            )

            script = script_path.read_text()
            self.assertIn(
                f'MARKER="{watch_dir}/latest_checkpointed_iteration.txt"', script
            )
            self.assertIn(
                str(
                    state_dir
                    / cfg.model_name
                    / ".submitted_steps"
                ),
                script,
            )
            self.assertTrue((root / cfg.log_dir / cfg.model_name).is_dir())
            self.assertIn("Stop requested — watcher will not be rescheduled", script)
            self.assertIn('--model "selected"', script)
            self.assertIn("--consumed-tokens-per-step 2048", script)

    def test_watcher_start_passes_checkpoint_override(self) -> None:
        cfg = _config("/path/to/Megatron-LM")
        args = argparse.Namespace(
            config="config.py",
            interval=0.5,
            model=None,
            consumed_tokens_per_step=None,
        )
        completed = subprocess.CompletedProcess(
            ["sbatch"], returncode=0, stdout="Submitted batch job 123\n", stderr=""
        )
        with (
            patch(
                "evals.watcher._load_config",
                return_value=(cfg, "/override", "/state"),
            ),
            patch("evals.watcher._watcher_stop_file") as stop_file,
            patch("evals.watcher.render_watcher_script", return_value=Path("watcher.sh")) as render,
            patch("evals.watcher.subprocess.run", return_value=completed),
        ):
            stop_file.return_value.unlink.return_value = None
            cmd_start(args)

        self.assertEqual(render.call_args.kwargs["watch_checkpoint_dir"], "/override")

    def test_watcher_stop_marks_chain_before_cancelling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config("/path/to/Megatron-LM")
            stop_file = Path(tmp) / ".watcher_stop"
            args = argparse.Namespace(config="config.py", model=None)
            completed = subprocess.CompletedProcess(
                ["scancel"], returncode=0, stdout="", stderr=""
            )
            with (
                patch("evals.watcher._load_config", return_value=(cfg, None, None)),
                patch("evals.watcher._watcher_stop_file", return_value=stop_file),
                patch("evals.watcher.subprocess.run", return_value=completed) as cancel,
            ):
                cmd_stop(args)

            self.assertTrue(stop_file.is_file())
            cancel.assert_called_once_with(
                ["scancel", f"--name=watcher_{cfg.model_name}"],
                capture_output=True,
                text=True,
                check=False,
            )

    def test_external_config_can_import_a_sibling_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sibling_values.py").write_text('MODEL_NAME = "external"\n')
            config_path = root / "config.py"
            config_path.write_text(
                "from sibling_values import MODEL_NAME\n"
                "from evals.megatron_eval import MegatronEvalConfig\n"
                "cfg = MegatronEvalConfig(\n"
                "    model_name=MODEL_NAME, checkpoint_dir='/c', \n"
                "    tokenizer_model='tok', megatron_path='/m',\n"
                ")\n"
            )

            cfg, _, _ = _load_config(str(config_path))

            self.assertEqual(cfg.model_name, "external")

    def test_watcher_computes_consumed_tokens(self) -> None:
        self.assertEqual(_consumed_tokens(1907, 512 * 4096), 3_999_268_864)
        self.assertIsNone(_consumed_tokens(1907, None))
        with self.assertRaises(ValueError):
            _consumed_tokens(1907, 0)

    def test_external_config_can_build_selected_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.py"
            config_path.write_text(
                "from evals.megatron_eval import MegatronEvalConfig\n"
                "def build_eval_config(model_name):\n"
                "    return MegatronEvalConfig(\n"
                "        model_name=model_name, checkpoint_dir='/c',\n"
                "        tokenizer_model='tok', megatron_path='/m',\n"
                "    )\n"
            )

            cfg, _, _ = _load_config(str(config_path), "selected")

            self.assertEqual(cfg.model_name, "selected")


if __name__ == "__main__":
    unittest.main()
