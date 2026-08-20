"""Submit hfconverter Stage-2 exports for Megatron torch_dist checkpoints."""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

_JOB_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclasses.dataclass
class HFConversionConfig:
    # Local checkout or Git URL containing cluster/convert.sh.
    hfconverter_root: str
    checkpoint_dir: str
    output_dir: str
    tokenizer_dir: str

    hfconverter_commit: str = ""
    # Parent of hfconverter_repos/ and hfconverter_worktrees/. Defaults below
    # $SCRATCH/tmp, matching the Megatron checkout cache layout.
    hfconverter_cache_dir: str = ""
    verify_load: bool = True
    max_shard_size: str = "5GB"
    recreate: bool = False

    account: str = ""
    partition: str = ""
    run_time: str = "03:00:00"
    memory: str = "460000"
    reservation: str = ""
    exclude: str = ""
    log_dir: str = "slurm_logs/conversion"


def output_is_complete(output_dir: str | Path) -> bool:
    """Return whether Stage 2 left its completion certificate."""
    output = Path(output_dir).expanduser()
    return (
        output.is_dir()
        and (output / "conversion_info.json").is_file()
        and not (output / ".export_incomplete").exists()
    )


def _is_git_url(value: str) -> bool:
    parsed = urlparse(value)
    return (
        parsed.scheme in {"file", "git", "http", "https", "ssh"}
        and (bool(parsed.netloc) or parsed.scheme == "file")
    ) or (value.startswith("git@") and ":" in value)


def _cache_root(cfg: HFConversionConfig) -> Path:
    if cfg.hfconverter_cache_dir:
        return Path(cfg.hfconverter_cache_dir).expanduser().resolve()
    if scratch := os.environ.get("SCRATCH"):
        return Path(scratch).expanduser().resolve() / "tmp"
    # The checkout must remain visible after this process submits the Slurm job;
    # unlike node-local TMPDIR, HOME is normally mounted by the converter EDF.
    return Path.home() / ".cache" / "spellbook" / "tmp"


def _resolve_ref(root: Path, requested: str) -> str:
    candidates = [requested]
    if requested and not requested.startswith("refs/"):
        candidates.insert(0, f"refs/remotes/origin/{requested}")
    for candidate in candidates:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", f"{candidate}^{{commit}}"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    raise ValueError(f"cannot resolve hfconverter ref {requested!r} in {root}")


def _prepare_hfconverter(cfg: HFConversionConfig) -> Path:
    if not _is_git_url(cfg.hfconverter_root):
        return Path(cfg.hfconverter_root).expanduser().resolve()

    cache_root = _cache_root(cfg)
    source_key = hashlib.sha256(cfg.hfconverter_root.encode()).hexdigest()[:16]
    repository = cache_root / "hfconverter_repos" / source_key
    worktree_key = hashlib.sha256(
        f"{cfg.hfconverter_root}\0{cfg.hfconverter_commit}".encode()
    ).hexdigest()[:16]
    worktree = cache_root / "hfconverter_worktrees" / worktree_key
    repository.parent.mkdir(parents=True, exist_ok=True)
    worktree.parent.mkdir(parents=True, exist_ok=True)

    lock_path = repository.with_suffix(".lock")
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if repository.exists() and not (repository / ".git").is_dir():
            raise ValueError(f"hfconverter cache is not a Git repository: {repository}")
        if not repository.exists():
            subprocess.run(
                ["git", "clone", cfg.hfconverter_root, str(repository)], check=True
            )
        subprocess.run(
            ["git", "-C", str(repository), "fetch", "origin", "--tags", "--prune"],
            check=True,
        )
        target = _resolve_ref(
            repository,
            cfg.hfconverter_commit or "refs/remotes/origin/HEAD",
        )

        if worktree.exists():
            actual = subprocess.run(
                ["git", "-C", str(worktree), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
            )
            if actual.returncode != 0 or actual.stdout.strip() != target:
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(repository),
                        "worktree",
                        "remove",
                        "--force",
                        str(worktree),
                    ],
                    check=False,
                )
                if worktree.exists():
                    shutil.rmtree(worktree)
        if not worktree.exists():
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "worktree",
                    "add",
                    "--detach",
                    str(worktree),
                    target,
                ],
                check=True,
            )
    return worktree


def _validate(
    cfg: HFConversionConfig, root: Path
) -> tuple[Path, Path, Path, Path]:
    checkpoint = Path(cfg.checkpoint_dir).expanduser().resolve()
    # Keep the final component unresolved so recreation can reject symlinks.
    output = Path(cfg.output_dir).expanduser().absolute()
    tokenizer = Path(cfg.tokenizer_dir).expanduser().resolve()
    convert = root / "cluster" / "convert.sh"
    stage2 = root / "cluster" / "stage2_export.sbatch"
    if not convert.is_file() or not os.access(convert, os.X_OK):
        raise ValueError(f"missing executable hfconverter wrapper: {convert}")
    if not stage2.is_file():
        raise ValueError(f"missing hfconverter Stage-2 script: {stage2}")
    if not (checkpoint / "common.pt").is_file():
        raise ValueError(f"checkpoint is not a torch_dist iteration: {checkpoint}")
    if not (tokenizer / "tokenizer.json").is_file():
        raise ValueError(f"tokenizer directory has no tokenizer.json: {tokenizer}")
    if re.fullmatch(r"[1-9][0-9]*(?:KB|MB|GB|TB)", cfg.max_shard_size) is None:
        raise ValueError("max_shard_size must look like '5GB'")
    if output.is_symlink():
        raise ValueError(f"output directory must not be a symlink: {output}")
    resolved_output = output.resolve()
    if (
        resolved_output == checkpoint
        or resolved_output.is_relative_to(checkpoint)
        or checkpoint.is_relative_to(resolved_output)
    ):
        raise ValueError("output and checkpoint directories must not contain one another")
    if output.exists() and not output.is_dir():
        raise ValueError(f"output path exists and is not a directory: {output}")
    if cfg.hfconverter_commit and not _is_git_url(cfg.hfconverter_root):
        actual = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        expected = _resolve_ref(root, cfg.hfconverter_commit)
        if actual != expected:
            raise ValueError(
                f"hfconverter checkout is {actual}, expected {expected}"
            )
    return root, checkpoint, output, tokenizer


def render_submission(cfg: HFConversionConfig) -> tuple[list[str], dict[str, str]]:
    """Prepare hfconverter and render its command and environment."""
    root, checkpoint, output, tokenizer = _validate(
        cfg, _prepare_hfconverter(cfg)
    )
    name = _JOB_NAME.sub("-", f"hf_{output.name}").strip("-") or "hf_conversion"
    log_dir = Path(cfg.log_dir).expanduser().resolve()
    command = [
        str(root / "cluster" / "convert.sh"),
        str(checkpoint),
        str(output),
        "--parsable",
        f"--job-name={name}",
        f"--chdir={root}",
        f"--output={log_dir}/%x-%j.log",
        f"--error={log_dir}/%x-%j.log",
    ]
    for option, value in (
        ("account", cfg.account),
        ("partition", cfg.partition),
        ("time", cfg.run_time),
        ("mem", cfg.memory),
        ("reservation", cfg.reservation),
        ("exclude", cfg.exclude),
    ):
        if value:
            command.append(f"--{option}={value}")

    environment = os.environ.copy()
    environment.update(
        REPO=str(root),
        TOKENIZER_DIR=str(tokenizer),
        VERIFY_LOAD="1" if cfg.verify_load else "0",
        EXTRA_EXPORT_ARGS=f"--max-shard-size {cfg.max_shard_size}",
    )
    environment.pop("TORCH_FORCE_WEIGHTS_ONLY_LOAD", None)
    environment.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
    return command, environment


def submit(cfg: HFConversionConfig) -> str | None:
    """Submit Stage 2, or return ``None`` when a completed output is reused."""
    command, environment = render_submission(cfg)
    output = Path(cfg.output_dir).expanduser().absolute()
    if output_is_complete(output) and not cfg.recreate:
        print(f"  reusing converted checkpoint: {output}")
        return None

    if output.exists() and any(output.iterdir()):
        if not cfg.recreate:
            raise ValueError(
                f"incomplete conversion output already exists: {output}; "
                "set recreate=True to replace it"
            )
        shutil.rmtree(output)
    elif cfg.recreate and output.exists():
        output.rmdir()

    Path(cfg.log_dir).expanduser().mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        command,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"hfconverter returned an invalid job id: {result.stdout!r}")
    print(f"  conversion submitted → job {job_id}")
    return job_id
