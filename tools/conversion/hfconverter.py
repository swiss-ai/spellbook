"""Run hfconverter from pinned checkouts inside the caller's conversion job."""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import os
import subprocess
from pathlib import Path


@dataclasses.dataclass
class HFConverterConfig:
    hfconverter_root: str
    hfconverter_commit: str
    megatron_root: str
    megatron_commit: str
    bridge_root: str
    bridge_commit: str
    checkpoint: str
    reference: str
    output: str
    source_cache: str = ""
    options: list[str] = dataclasses.field(default_factory=list)


def _checkout(source: str, commit: str, cache: Path) -> Path:
    root = Path(source).expanduser().resolve()
    if not commit:
        raise ValueError("A Git commit is required for each conversion checkout")
    target = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", f"{commit}^{{commit}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    key = hashlib.sha256(f"{root}\0{target}".encode()).hexdigest()[:16]
    worktree = cache / key
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / f"{key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not worktree.exists():
            subprocess.run(
                ["git", "-C", str(root), "worktree", "add", "--detach", str(worktree), target],
                check=True,
            )
        actual = subprocess.run(
            ["git", "-C", str(worktree), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if actual != target:
            raise ValueError(f"Cached checkout has the wrong commit: {worktree}")
        changes = subprocess.run(
            ["git", "-C", str(worktree), "status", "--porcelain", "--untracked-files=all"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if changes:
            raise ValueError(f"Cached checkout has local changes: {worktree}")
    return worktree


def render_command(cfg: HFConverterConfig) -> list[str]:
    """Resolve detached worktrees, then return the native export command."""
    cache = (
        Path(cfg.source_cache).expanduser()
        if cfg.source_cache
        else (Path(os.environ.get("SCRATCH", "/tmp")) / "tmp/hfconverter-sources")
    )
    cache = cache.resolve()
    converter = _checkout(cfg.hfconverter_root, cfg.hfconverter_commit, cache)
    megatron = _checkout(cfg.megatron_root, cfg.megatron_commit, cache)
    bridge = _checkout(cfg.bridge_root, cfg.bridge_commit, cache)
    script = converter / "convert_checkpoint.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    return [
        "uv",
        "run",
        "--no-project",
        "python",
        str(script),
        "export",
        str(Path(cfg.checkpoint).expanduser().resolve()),
        str(Path(cfg.reference).expanduser().resolve()),
        str(Path(cfg.output).expanduser().resolve()),
        *cfg.options,
        "--megatron-root",
        str(megatron),
        "--bridge-root",
        str(bridge),
    ]


def launch(cfg: HFConverterConfig) -> subprocess.CompletedProcess[str]:
    """Execute in the existing allocation; the caller supplies ranks and resources."""
    return subprocess.run(render_command(cfg), text=True, check=True)
