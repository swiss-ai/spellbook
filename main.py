"""
megatron-zoo CLI

Usage:
    python main.py run <experiment.py> [--dry-run]
    python main.py status [<name>]
    python main.py list <experiment.py>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    project = _load_project(args.experiment)
    results = project.run(dry_run=args.dry_run)

    print("\n--- Results ---")
    for name, result in results.items():
        status = result.status.value
        if result.error:
            print(f"  {name}: {status} ({result.error})")
        else:
            print(f"  {name}: {status}")
            for k, v in result.data.items():
                print(f"    {k}: {v}")


def cmd_list(args: argparse.Namespace) -> None:
    project = _load_project(args.experiment)
    print(f"Project: {project.name}")
    for name in project.list_modules():
        module = project._modules[name]
        deps = f"  [depends: {', '.join(module.depends_on)}]" if module.depends_on else ""
        print(f"  {module.__class__.__name__:<20} {name}{deps}")


def cmd_status(args: argparse.Namespace) -> None:
    """Read chain_state.json files under runs/ and print a status table."""
    runs_dir = Path("runs")
    if not runs_dir.exists():
        print("No runs/ directory found.")
        return

    state_files = list(runs_dir.glob("*/chain_state.json"))
    if not state_files:
        print("No chain state files found under runs/.")
        return

    for state_file in sorted(state_files):
        state = json.loads(state_file.read_text())
        chain_name = state.get("chain", state_file.parent.name)

        if args.name and chain_name != args.name:
            continue

        print(f"\nChain: {chain_name}")
        print(f"  {'Step':<20} {'Job ID':<12} {'Status':<12} Checkpoint")
        print("  " + "-" * 60)
        for step in state.get("steps", []):
            ckpt = step.get("checkpoint") or "—"
            print(
                f"  {step['name']:<20} {step.get('job_id', '?'):<12} "
                f"{step.get('status', '?'):<12} {ckpt}"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_project(experiment_path: str):
    """
    Import an experiment Python file and return its `project` variable.

    The file must define a module-level variable named `project` that is
    a zoo.project.Project instance.
    """
    path = Path(experiment_path).resolve()
    if not path.exists():
        print(f"Error: experiment file not found: {path}", file=sys.stderr)
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("_experiment", path)
    if spec is None or spec.loader is None:
        print(f"Error: cannot load {path}", file=sys.stderr)
        sys.exit(1)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    if not hasattr(module, "project"):
        print(
            f"Error: {path} does not define a top-level 'project' variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    return module.project


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="megatron-zoo",
        description="Manage Megatron-LM training experiments on SLURM clusters.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Run an experiment file")
    p_run.add_argument("experiment", help="Path to experiment .py file")
    p_run.add_argument(
        "--dry-run", action="store_true",
        help="Render scripts and print configs without submitting to SLURM",
    )

    # list
    p_list = sub.add_parser("list", help="List modules in an experiment file")
    p_list.add_argument("experiment", help="Path to experiment .py file")

    # status
    p_status = sub.add_parser("status", help="Show status of submitted chains")
    p_status.add_argument("name", nargs="?", default=None, help="Filter to a specific chain name")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
