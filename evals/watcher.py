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

The config file must define either a module-level `cfg: MegatronEvalConfig` or a
`build_eval_config(model_name)` factory used with `--model`. It may also define
`watch_checkpoint_dir` (str) to override cfg.checkpoint_dir/<model_name> as the
directory containing latest_checkpointed_iteration.txt.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

from evals.megatron_eval import MegatronEvalConfig, render_watcher_script, submit


def _load_config(
    config_path: str,
    model_name: str | None = None,
) -> tuple[MegatronEvalConfig, str | None, str | None]:
    path = Path(config_path).resolve()
    spec = importlib.util.spec_from_file_location("_eval_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load eval config from {path}")
    mod = importlib.util.module_from_spec(spec)
    config_dir = str(path.parent)
    sys.path.insert(0, config_dir)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(config_dir)
    cfg = getattr(mod, "cfg", None)
    if cfg is None and model_name is not None:
        builder = getattr(mod, "build_eval_config", None)
        if callable(builder):
            cfg = builder(model_name)
    if not isinstance(cfg, MegatronEvalConfig):
        print(
            f"ERROR: {config_path} must define `cfg: MegatronEvalConfig` or "
            "`build_eval_config(model_name)` used with --model"
        )
        sys.exit(1)
    return (
        cfg,
        getattr(mod, "watch_checkpoint_dir", None),
        getattr(mod, "watch_state_dir", None),
    )


def _latest_step(checkpoint_dir: Path) -> int | None:
    marker = checkpoint_dir / "latest_checkpointed_iteration.txt"
    if not marker.exists():
        return None
    text = marker.read_text().strip()
    if not text.isdigit():
        return None
    return int(text)


def _state_file(
    cfg: MegatronEvalConfig,
    state_dir: str | None = None,
) -> Path:
    root = Path(state_dir) if state_dir else Path("evals") / "state_files"
    return root / cfg.model_name / ".submitted_steps"


def _submitted_steps(state: Path) -> set[int]:
    if not state.exists():
        return set()
    return {
        int(line.strip())
        for line in state.read_text().splitlines()
        if line.strip().isdigit()
    }


def _record_step(state: Path, step: int) -> None:
    state.parent.mkdir(parents=True, exist_ok=True)
    with state.open("a") as f:
        f.write(f"{step}\n")


def _consumed_tokens(step: int, tokens_per_step: int | None) -> int | None:
    if tokens_per_step is None:
        return None
    if tokens_per_step <= 0:
        raise ValueError("consumed_tokens_per_step must be greater than zero")
    return step * tokens_per_step


def cmd_check(args: argparse.Namespace) -> None:
    cfg, watch_dir_override, state_dir = _load_config(args.config, args.model)

    if args.step is not None:
        # Called from watcher.sh with the step already validated by bash.
        submit(
            cfg,
            args.step,
            consumed_tokens=_consumed_tokens(
                args.step, args.consumed_tokens_per_step
            ),
        )
        return

    ckpt_dir = Path(watch_dir_override) if watch_dir_override else Path(cfg.checkpoint_dir) / cfg.model_name
    latest = _latest_step(ckpt_dir)
    if latest is None:
        print(f"No completed checkpoint found in {ckpt_dir} — nothing to do.")
        return

    state = _state_file(cfg, state_dir)
    submitted = _submitted_steps(state)

    if latest in submitted:
        print(f"Step {latest} already submitted — nothing to do.")
        return

    print(f"New checkpoint detected: step {latest}. Submitting eval...")
    submit(
        cfg,
        latest,
        consumed_tokens=_consumed_tokens(latest, args.consumed_tokens_per_step),
    )
    _record_step(state, latest)


def _watcher_stop_file(
    cfg: MegatronEvalConfig,
    project_dir: Path,
    state_dir: str | None = None,
) -> Path:
    root = Path(state_dir) if state_dir else project_dir / "evals" / "state_files"
    return root / cfg.model_name / ".watcher_stop"


def cmd_start(args: argparse.Namespace) -> None:
    cfg, watch_dir_override, state_dir = _load_config(args.config, args.model)
    project_dir = Path(__file__).resolve().parent.parent
    stop_file = _watcher_stop_file(cfg, project_dir, state_dir)
    stop_file.unlink(missing_ok=True)
    script_path = render_watcher_script(
        cfg,
        config_path=args.config,
        interval_hours=args.interval,
        project_dir=project_dir,
        watch_checkpoint_dir=watch_dir_override,
        config_model=args.model,
        consumed_tokens_per_step=args.consumed_tokens_per_step,
        watch_state_dir=state_dir,
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
    print(f"Re-runs every {round(args.interval * 60)}min.")
    stop_command = f"uv run python -m evals.watcher stop --config {args.config}"
    if args.model:
        stop_command += f" --model {args.model}"
    print(f"Stop with: {stop_command}")


def cmd_stop(args: argparse.Namespace) -> None:
    cfg, _, state_dir = _load_config(args.config, args.model)
    project_dir = Path(__file__).resolve().parent.parent
    stop_file = _watcher_stop_file(cfg, project_dir, state_dir)
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    stop_file.touch()
    result = subprocess.run(
        ["scancel", f"--name=watcher_{cfg.model_name}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(result.stderr.strip(), file=sys.stderr)
    print(f"Watcher stop requested for {cfg.model_name}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval checkpoint watcher")
    sub = parser.add_subparsers(dest="command")

    # Default mode (no subcommand): one-shot check, called by the sbatch watcher
    parser.add_argument("--config", help="Path to eval config .py file")
    parser.add_argument("--step", type=int, default=None, help="Submit this specific step (skips state-file check)")
    parser.add_argument("--model", help="Model passed to build_eval_config(model_name)")
    parser.add_argument(
        "--consumed-tokens-per-step",
        type=int,
        help="Multiply checkpoint steps by this value for W&B consumed-token metadata",
    )

    # start: render watcher.sh and submit the chain
    p_start = sub.add_parser(
        "start", help="Render watcher.sh and submit the self-scheduling sbatch chain"
    )
    p_start.add_argument("--config", required=True, help="Path to eval config .py file")
    p_start.add_argument("--model", help="Model passed to build_eval_config(model_name)")
    p_start.add_argument(
        "--consumed-tokens-per-step",
        type=int,
        help="Multiply checkpoint steps by this value for W&B consumed-token metadata",
    )
    p_start.add_argument(
        "--interval",
        type=float,
        default=1,
        help="Hours between watcher runs, accepts decimals e.g. 0.5 (default: 1)",
    )

    p_stop = sub.add_parser("stop", help="Stop a self-scheduling watcher chain")
    p_stop.add_argument("--config", required=True, help="Path to eval config .py file")
    p_stop.add_argument("--model", help="Model passed to build_eval_config(model_name)")

    args = parser.parse_args()

    if args.command == "start":
        cmd_start(args)
    elif args.command == "stop":
        cmd_stop(args)
    else:
        if not args.config:
            parser.error("--config is required")
        cmd_check(args)


if __name__ == "__main__":
    main()
