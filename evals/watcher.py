"""
Eval watcher — two modes of operation:

  One-shot check (called by the self-scheduling sbatch):
      python evals/watcher.py --config evals/configs/small_100b.py [--step N]

    Without --step: reads latest_checkpointed_iteration.txt and submits if unseen.
    With --step N:  submits step N unconditionally (state-file check already done by bash).

  Start the self-scheduling sbatch chain:
      python evals/watcher.py start --config evals/configs/small_100b.py [--interval 1]

    Renders evals/<model>/watcher.sh and submits it; each run re-submits itself
    with --begin=now+<interval>hour so the chain continues indefinitely.

The config file must define a module-level variable `cfg` of type MegatronEvalConfig.
It may also define `watch_checkpoint_dir` (str) to override cfg.checkpoint_dir/<model_name>
as the directory containing latest_checkpointed_iteration.txt.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

from evals.megatron_eval import MegatronEvalConfig, render_watcher_script, submit


def _load_config(config_path: str) -> tuple[MegatronEvalConfig, str | None]:
    path = Path(config_path).resolve()
    spec = importlib.util.spec_from_file_location("_eval_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = getattr(mod, "cfg", None)
    if not isinstance(cfg, MegatronEvalConfig):
        print(f"ERROR: {config_path} must define a module-level `cfg: MegatronEvalConfig`")
        sys.exit(1)
    return cfg, getattr(mod, "watch_checkpoint_dir", None)


def _latest_step(checkpoint_dir: Path) -> int | None:
    marker = checkpoint_dir / "latest_checkpointed_iteration.txt"
    if not marker.exists():
        return None
    text = marker.read_text().strip()
    if not text.isdigit():
        return None
    return int(text)


def _state_file(cfg: MegatronEvalConfig) -> Path:
    return Path("evals") / "state_files" / cfg.model_name / ".submitted_steps"


def _submitted_steps(state: Path) -> set[int]:
    if not state.exists():
        return set()
    return {int(l.strip()) for l in state.read_text().splitlines() if l.strip().isdigit()}


def _record_step(state: Path, step: int) -> None:
    state.parent.mkdir(parents=True, exist_ok=True)
    with state.open("a") as f:
        f.write(f"{step}\n")


def cmd_check(args: argparse.Namespace) -> None:
    cfg, watch_dir_override = _load_config(args.config)

    if args.step is not None:
        # Called from watcher.sh with the step already validated by bash.
        submit(cfg, args.step)
        return

    ckpt_dir = Path(watch_dir_override) if watch_dir_override else Path(cfg.checkpoint_dir) / cfg.model_name
    latest = _latest_step(ckpt_dir)
    if latest is None:
        print(f"No completed checkpoint found in {ckpt_dir} — nothing to do.")
        return

    state = _state_file(cfg)
    submitted = _submitted_steps(state)

    if latest in submitted:
        print(f"Step {latest} already submitted — nothing to do.")
        return

    print(f"New checkpoint detected: step {latest}. Submitting eval...")
    submit(cfg, latest)
    _record_step(state, latest)


def cmd_start(args: argparse.Namespace) -> None:
    cfg, _ = _load_config(args.config)
    project_dir = Path(__file__).resolve().parent.parent
    script_path = render_watcher_script(
        cfg,
        config_path=args.config,
        interval_hours=args.interval,
        project_dir=project_dir,
    )
    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    job_id = result.stdout.strip().split()[-1]
    print(f"Watcher started: job {job_id}")
    print(f"Script saved at: {script_path}")
    print(f"Re-runs every {round(args.interval * 60)}min. Cancel with: scancel {job_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval checkpoint watcher")
    sub = parser.add_subparsers(dest="command")

    # Default mode (no subcommand): one-shot check, called by the sbatch watcher
    parser.add_argument("--config", help="Path to eval config .py file")
    parser.add_argument("--step", type=int, default=None, help="Submit this specific step (skips state-file check)")

    # start: render watcher.sh and submit the chain
    p_start = sub.add_parser("start", help="Render watcher.sh and submit the self-scheduling sbatch chain")
    p_start.add_argument("--config", required=True, help="Path to eval config .py file")
    p_start.add_argument("--interval", type=float, default=1, help="Hours between watcher runs, accepts decimals e.g. 0.5 (default: 1)")

    args = parser.parse_args()

    if args.command == "start":
        cmd_start(args)
    else:
        if not args.config:
            parser.error("--config is required")
        cmd_check(args)


if __name__ == "__main__":
    main()
