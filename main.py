"""
spellbook CLI

Usage:
    python main.py render  <experiments/foo/experiment.py>  [--only NAME] [--output-dir DIR]
    python main.py submit  <experiments/foo/experiment.py>  [--only NAME] [--output-dir DIR]
    python main.py list    <experiments/foo/experiment.py>  [--all | --columns COL,COL,...]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from dotenv import load_dotenv


# Fields shown by default in the list table (in addition to changed fields).
# Edit this list to suit the most commonly inspected dimensions.
_DEFAULT_COLUMNS: list[str] = [
    "tp", "pp", "ep", "etp", "cp",
    "mbs", "gbs", "num_gpus",
    "bf16", "fp8_format",
    "moe_token_dispatcher_type",
    "overlap_moe_expert_parallel_comm",
]


def _load_sweep(path: str, only: str | None = None):
    """Import an experiment file and return its top-level `sweep` variable.

    If `only` is given, filters the sweep to the single experiment with that name.
    """
    p = Path(path).resolve()
    if not p.exists():
        print(f"Error: file not found: {p}", file=sys.stderr)
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("_experiment", p)
    if spec is None or spec.loader is None:
        print(f"Error: cannot load {p}", file=sys.stderr)
        sys.exit(1)

    mod = importlib.util.module_from_spec(spec)
    sys.modules["_experiment"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    if not hasattr(mod, "sweep"):
        print(f"Error: {p} must define a top-level 'sweep' variable.", file=sys.stderr)
        sys.exit(1)

    sweep = mod.sweep

    if only:
        matches = [e for e in sweep.experiments if e.name == only]
        if not matches:
            names = [e.name for e in sweep.experiments]
            print(f"Error: no experiment named '{only}'. Available: {', '.join(names)}", file=sys.stderr)
            sys.exit(1)
        sweep.experiments = matches

    return sweep


def cmd_render(args: argparse.Namespace) -> None:
    sweep = _load_sweep(args.experiment, only=args.only)
    paths = sweep.render(output_dir=args.output_dir)
    print(f"Rendered {len(paths)} script(s) to {args.output_dir}/{sweep.name}/")
    for p in paths:
        print(f"  {p}")


def cmd_submit(args: argparse.Namespace) -> None:
    sweep = _load_sweep(args.experiment, only=args.only)
    job_ids = sweep.submit(output_dir=args.output_dir)
    print(f"\nSubmitted {len(job_ids)} job(s).")


def cmd_list(args: argparse.Namespace) -> None:
    sweep = _load_sweep(args.experiment)
    exps = sweep.experiments
    if not exps:
        print(f"Sweep '{sweep.name}': (empty)")
        return

    changed = sweep.changed_fields()

    if args.all:
        # Every field in the dataclass, alphabetically (excluding name which is first)
        extra_cols = sorted(k for k in exps[0].to_dict() if k != "name" and k != "training_args")
    elif args.columns:
        extra_cols = [c.strip() for c in args.columns.split(",")]
    else:
        # Default: changed fields + the standard set (deduplicated, preserving changed-first order)
        seen: set[str] = set(changed)
        extra_cols = list(changed)
        for c in _DEFAULT_COLUMNS:
            if c not in seen:
                seen.add(c)
                extra_cols.append(c)

    # Only keep cols that actually exist in the experiment dict
    sample = exps[0].to_dict()
    cols = ["name"] + [c for c in extra_cols if c in sample]

    # Mark which columns differ (for the header indicator)
    changed_set = set(changed)

    # Column width: max of header length and any value, capped reasonably
    col_w: dict[str, int] = {}
    for c in cols:
        vals = [str(e.to_dict().get(c, "")) for e in exps]
        col_w[c] = max(len(c) + (2 if c in changed_set else 0), max(len(v) for v in vals)) + 2

    # Header: changed columns get a * marker
    header_parts = []
    for c in cols:
        label = f"*{c}" if c in changed_set else c
        header_parts.append(f"{label:<{col_w[c]}}")
    header = "".join(header_parts)

    print(f"\nSweep: {sweep.name}  ({len(exps)} experiments)")
    print(f"  (* = differs across experiments)\n")
    print(header)
    print("-" * len(header))
    for exp in exps:
        d = exp.to_dict()
        row = "".join(f"{str(d.get(c, '')):<{col_w[c]}}" for c in cols)
        print(row)
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spellbook")
    sub = parser.add_subparsers(dest="command", required=True)

    p_render = sub.add_parser("render", help="Render sbatch scripts without submitting")
    p_render.add_argument("experiment", help="Path to experiment .py file")
    p_render.add_argument("--only", default=None, metavar="NAME", help="Render only this experiment name")
    p_render.add_argument("--output-dir", default="sbatch_scripts")

    p_submit = sub.add_parser("submit", help="Render and submit all experiments")
    p_submit.add_argument("experiment", help="Path to experiment .py file")
    p_submit.add_argument("--only", default=None, metavar="NAME", help="Submit only this experiment name")
    p_submit.add_argument("--output-dir", default="sbatch_scripts")

    p_list = sub.add_parser("list", help="Print a summary table of all experiments")
    p_list.add_argument("experiment", help="Path to experiment .py file")
    col_group = p_list.add_mutually_exclusive_group()
    col_group.add_argument("--all", action="store_true", help="Show all fields")
    col_group.add_argument(
        "--columns", default=None, metavar="COL,COL,...",
        help="Comma-separated list of fields to show (always includes changed fields)",
    )

    return parser


def main() -> None:
    # Load .env from the current working tree if present; do not override exported env.
    load_dotenv(override=False)
    args = build_parser().parse_args()
    {"render": cmd_render, "submit": cmd_submit, "list": cmd_list}[args.command](args)


if __name__ == "__main__":
    main()
