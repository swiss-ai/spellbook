"""Prepare a NeMo Gym environment: server venvs, the actor venv, and the dataset.

Runs inside the container, invoked by the job tools/gym_data renders. Every step is
skipped when its output already exists, so re-running is cheap and safe.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def _run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print(f"[gym] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def head_deps() -> list[str]:
    """Pins Gym applies to each child venv.

    The official NeMo RL v0.7 parent has Ray but intentionally omits OpenAI.
    Gym supports OpenAI through 2.7.2, so use that release when no parent pin is
    available instead of requiring the policy environment to install it.
    """
    try:
        openai_version = md.version("openai")
    except md.PackageNotFoundError:
        openai_version = "2.7.2"
    return [f"ray[default]=={md.version('ray')}", f"openai=={openai_version}"]


def actor_venv_ready(venv: Path) -> bool:
    """An interrupted uv build may leave bin/python behind but no dependencies."""
    python = venv / "bin" / "python"
    if not python.is_file():
        return False
    return (
        subprocess.run(
            [str(python), "-c", "import nemo_gym, openai, ray, torch"],
            env=os.environ,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def build_actor_venv(gym_home: Path, nemo_rl_venv_dir: Path) -> None:
    """Pre-create the NemoGym Ray actor venv so the prefetch does not `uv sync`.

    ray_actor_environment_registry.py pins that actor to PY_EXECUTABLES.NEMO_GYM and,
    unlike the vllm/mcore entries, does not switch it off with
    NEMO_RL_PY_EXECUTABLES_SYSTEM, so prefetch_venvs.py calls
    create_local_venv_on_each_node -> `uv sync --directory <NeMo-RL>`. That fails in a
    checkout whose vendored submodules are not initialised:

        Failed to generate package metadata for `nemo-automodel @ editable+3rdparty/...`

    venvs.py:_env_builder returns early when this venv already exists, so creating it
    here skips the sync. A .pth file exposes the official parent environment's
    torch/transformers; nemo_rl, bridge and megatron come from PYTHONPATH.
    """
    venv = nemo_rl_venv_dir / "nemo_rl.environments.nemo_gym.NemoGym"
    if actor_venv_ready(venv):
        print(f"[gym] actor venv present: {venv}")
        return

    if venv.exists():
        print(f"[gym] removing incomplete actor venv: {venv}")
        shutil.rmtree(venv)

    print("[gym] building NemoGym actor venv")
    _run(
        [
            "uv",
            "venv",
            "--seed",
            "--allow-existing",
            "--python",
            sys.executable,
            str(venv),
        ]
    )
    parent_site = subprocess.check_output(
        [sys.executable, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        text=True,
    ).strip()
    actor_site = subprocess.check_output(
        [
            str(venv / "bin" / "python"),
            "-c",
            "import sysconfig; print(sysconfig.get_path('purelib'))",
        ],
        text=True,
    ).strip()
    Path(actor_site, "nemo_rl_parent.pth").write_text(f"{parent_site}\n")
    _run(
        ["uv", "pip", "install", "-e", ".", *head_deps()],
        cwd=gym_home,
        env={**os.environ, "VIRTUAL_ENV": str(venv)},
    )
    if not actor_venv_ready(venv):
        raise RuntimeError(f"actor venv failed import smoke test: {venv}")


def write_prefetch_config(args: argparse.Namespace, path: Path) -> Path:
    """Minimal NeMo-RL config carrying just the env.nemo_gym block prefetch reads.

    examples/nemo_gym/prefetch_venvs.py takes NeMo-RL configs, not Gym server configs:
    it reads config["env"]["nemo_gym"] and replays NemoGym._spinup(dry_run=True).
    Pass --prefetch-config instead to use a hand-written one (upstream ships
    examples/nemo_gym/prefetch_*_all_envs.yaml), which is how aliases, thin client
    stubs and dummy interpolations get expressed.
    """
    path.write_text(
        yaml.safe_dump(
            {
                "env": {
                    "nemo_gym": {
                        "skip_venv_if_present": True,
                        "port_range_low": 5000,
                        "port_range_high": 5999,
                        "config_paths": list(args.config_path),
                    }
                }
            },
            sort_keys=False,
        )
    )
    return path


def prefetch_venvs(nemo_rl_path: Path, config: Path) -> None:
    """Build every venv through Gym's own spinup rather than reimplementing it."""
    script = nemo_rl_path / "examples" / "nemo_gym" / "prefetch_venvs.py"
    if not script.is_file():
        sys.exit(f"prefetch script not found: {script}")
    _run([sys.executable, str(script), str(config)], cwd=nemo_rl_path)


def collate(args: argparse.Namespace, out: Path) -> None:
    """Run `gym dataset collate`.

    NemoGym.run_rollouts routes each row by row["agent_ref"]["name"], so rows have to
    come from collate rather than a hand-built prompt jsonl.
    """
    if (out / "train.jsonl").is_file() and (out / "train.jsonl").stat().st_size:
        print(f"[gym] data present: {out / 'train.jsonl'}")
        return

    scratch_root = Path(args.scratch_root)
    (scratch_root / "resources_servers" / args.dataset_name / "data").mkdir(
        parents=True, exist_ok=True
    )
    gym_bin = (
        Path(args.venv_dir) / "resources_servers" / args.dataset_name / ".venv" / "bin" / "gym"
    )
    print(f"[gym] collating -> {out}")
    # cwd is the scratch root so the raw download and the *_prepare.jsonl / metrics
    # written beside the dataset stay off $HOME.
    _run(
        [
            str(gym_bin),
            "dataset",
            "collate",
            f"+config_paths=[{','.join(args.config_path)}]",
            f"+output_dirpath={out}",
            "+mode=train_preparation",
            "+should_download=true",
        ],
        cwd=scratch_root,
    )


def postprocess(out: Path, grading_mode: str, validation_rows: int) -> None:
    """Stamp grading_mode and carve off a validation split.

    Always derives from collated.jsonl, a copy of collate's output taken once, so a
    re-run can never re-split an already-split dataset.
    """
    collated = out / "collated.jsonl"
    if not collated.exists():
        shutil.copyfile(out / "train.jsonl", collated)

    rows = [json.loads(line) for line in collated.open() if line.strip()]
    if grading_mode:
        for row in rows:
            row["grading_mode"] = grading_mode

    train, val = (
        (rows[:-validation_rows], rows[-validation_rows:]) if validation_rows else (rows, [])
    )
    parts = [("train", train)] + ([("validation", val)] if validation_rows else [])
    for name, part in parts:
        target = out / f"{name}.jsonl"
        with target.open("w") as handle:
            for row in part:
                handle.write(json.dumps(row) + "\n")
        print(f"[gym] wrote {len(part)} rows to {target}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gym-home", required=True)
    p.add_argument("--nemo-rl-path", required=True)
    p.add_argument("--config-path", action="append", default=[], required=True)
    p.add_argument("--venv-dir", required=True)
    p.add_argument("--nemo-rl-venv-dir", required=True)
    p.add_argument("--uv-cache-dir", required=True)
    p.add_argument("--scratch-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset-name", default="")
    p.add_argument("--grading-mode", default="")
    p.add_argument("--validation-rows", type=int, default=0)
    p.add_argument("--no-build-venvs", action="store_true")
    p.add_argument(
        "--prefetch-config",
        default="",
        help="NeMo-RL config with an env.nemo_gym block; generated from --config-path when omitted",
    )
    args = p.parse_args(argv)

    gym_home = Path(args.gym_home)
    if not (gym_home / "nemo_gym").is_dir():
        sys.exit(f"Gym submodule not initialised at {gym_home}")

    for path in (
        args.uv_cache_dir,
        args.venv_dir,
        args.nemo_rl_venv_dir,
        args.scratch_root,
    ):
        Path(path).mkdir(parents=True, exist_ok=True)

    os.environ.update(
        {
            "PYTHONNOUSERSITE": "1",
            "GYM_HOME": args.gym_home,
            "NEMO_GYM_VENV_DIR": args.venv_dir,
            "NEMO_RL_VENV_DIR": args.nemo_rl_venv_dir,
            "UV_CACHE_DIR": args.uv_cache_dir,
            "NRL_CONTAINER": "1",  # makes NeMo-RL forward UV_CACHE_DIR to Gym
            "NEMO_GYM_SCRATCH_ROOT": args.scratch_root,
            # Earliest root wins (nemo_gym/__init__.py) and a dataset's jsonl_fpath is
            # relative, so the writable scratch root must come first or Gym downloads
            # into the checkout.
            "NEMO_GYM_EXTRA_ROOTS": os.pathsep.join([args.scratch_root, args.gym_home]),
            "PYTHONPATH": os.pathsep.join(
                [args.nemo_rl_path, args.gym_home, os.environ.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep),
        }
    )

    if not args.no_build_venvs:
        # Before the prefetch: it would otherwise try to `uv sync` the checkout.
        build_actor_venv(gym_home, Path(args.nemo_rl_venv_dir))
        config = (
            Path(args.prefetch_config)
            if args.prefetch_config
            else write_prefetch_config(args, Path(args.venv_dir).parent / "spellbook-prefetch.yaml")
        )
        prefetch_venvs(Path(args.nemo_rl_path), config)

    if args.dataset_name:
        out = Path(args.output_dir) / args.dataset_name
        out.mkdir(parents=True, exist_ok=True)
        collate(args, out)
        if args.grading_mode or args.validation_rows:
            postprocess(out, args.grading_mode, args.validation_rows)

    print("[gym] preparation complete")
    print(f"  NEMO_GYM_VENV_DIR = {args.venv_dir}")
    print(f"  NEMO_RL_VENV_DIR  = {args.nemo_rl_venv_dir}")
    if args.dataset_name:
        print(f"  data              = {Path(args.output_dir) / args.dataset_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
