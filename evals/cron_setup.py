"""
Install or remove a crontab entry that runs watcher.py on a schedule.

Usage:
    python evals/cron_setup.py install --config evals/configs/small_100b.py [--interval 5]
    python evals/cron_setup.py remove  --config evals/configs/small_100b.py
    python evals/cron_setup.py list

The crontab entry is tagged with a comment so it can be identified and removed cleanly.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_TAG_PREFIX = "# spellbook-eval-watcher:"


def _tag(config_path: str) -> str:
    return f"{_TAG_PREFIX} {Path(config_path).resolve()}"


def _current_crontab() -> str:
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if result.returncode != 0:
        return ""
    return result.stdout


def _set_crontab(content: str) -> None:
    proc = subprocess.run(["crontab", "-"], input=content, text=True, check=True)


def _cron_entry(interval: int, config_path: str, project_dir: Path) -> str:
    python = "uv run python"
    watcher = project_dir / "evals" / "watcher.py"
    log = project_dir / "evals" / ".watcher.log"
    config = Path(config_path).resolve()
    tag = _tag(config_path)
    cron_expr = f"*/{interval} * * * *"
    cmd = f"cd {project_dir} && {python} {watcher} --config {config} >> {log} 2>&1"
    return f"{tag}\n{cron_expr} {cmd}\n"


def cmd_install(args: argparse.Namespace, project_dir: Path) -> None:
    config_path = str(Path(args.config).resolve())
    tag = _tag(config_path)
    current = _current_crontab()

    if tag in current:
        print(f"Watcher for {config_path} is already installed.")
        return

    entry = _cron_entry(args.interval, config_path, project_dir)
    new_crontab = current.rstrip("\n") + ("\n" if current else "") + entry
    _set_crontab(new_crontab)
    print(f"Installed: every {args.interval} min → {config_path}")


def cmd_remove(args: argparse.Namespace) -> None:
    config_path = str(Path(args.config).resolve())
    tag = _tag(config_path)
    current = _current_crontab()

    if tag not in current:
        print(f"No watcher found for {config_path}.")
        return

    lines = current.splitlines(keepends=True)
    filtered = []
    skip_next = False
    for line in lines:
        if line.strip() == tag:
            skip_next = True
            continue
        if skip_next:
            skip_next = False
            continue
        filtered.append(line)

    _set_crontab("".join(filtered))
    print(f"Removed watcher for {config_path}.")


def cmd_list() -> None:
    current = _current_crontab()
    entries = [l for l in current.splitlines() if l.startswith(_TAG_PREFIX)]
    if not entries:
        print("No eval watchers installed.")
        return
    print("Installed eval watchers:")
    for e in entries:
        print(f"  {e[len(_TAG_PREFIX):].strip()}")


def main() -> None:
    project_dir = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_install = sub.add_parser("install", help="Install a crontab watcher")
    p_install.add_argument("--config", required=True, help="Path to eval config .py file")
    p_install.add_argument("--interval", type=int, default=5, help="Check interval in minutes (default: 5)")

    p_remove = sub.add_parser("remove", help="Remove a crontab watcher")
    p_remove.add_argument("--config", required=True, help="Path to eval config .py file")

    sub.add_parser("list", help="List installed watchers")

    args = parser.parse_args()

    if args.command == "install":
        cmd_install(args, project_dir)
    elif args.command == "remove":
        cmd_remove(args)
    elif args.command == "list":
        cmd_list()


if __name__ == "__main__":
    main()
