"""
SlurmBackend: JobSpec → sbatch script string + optional submission.

Mirrors the sbatch heredoc in sbatch_benchmarking.sh exactly:
  - #SBATCH headers from JobSpec fields
  - export block from env_vars
  - srun with Pyxis/Enroot flags + torch.distributed.run launcher

The srun --export list is built automatically from every variable exported
in the script body, so nothing is hardcoded or forgotten.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from zoo.module import JobSpec

# Variables set in the script body that must be forwarded into the srun task.
# This list is exactly what the script exports before the srun call, so
# the --export= flag stays in sync with the script automatically.
_SCRIPT_VARS = [
    # Megatron paths
    "MEGATRON_PATH",
    "MEGATRON_PATCH_PATH",
    "PYTHONPATH",
    # Distributed
    "LOCAL_RANK",
    "RANK",
    "GPUS_PER_NODE",
    "HOSTNAMES",
    "MASTER_ADDR",
    "MASTER_PORT",
    "WORLD_SIZE",
    "SLURM_JOB_ID",
    # PyTorch / NCCL
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "CUDA_CACHE_DISABLE",
    # Optional passthrough (set or unset in the environment)
    "WANDB_MODE",
    "WANDB_API_KEY",
    "WANDB_PROJECT",
    "WANDB_RUN_ID",
    "WANDB_RESUME",
    "HF_TOKEN",
    "HF_HUB_ENABLE_HF_TRANSFER",
    "PYTHONBUFFERED",
    "ACTIVATE_VENV",
    "NCCL_DEBUG",
    "NCCL_DEBUG_SUBSYS",
    "NCCL_NVLS_ENABLE",
    "NVTE_FWD_LAYERNORM_SM_MARGIN",
    "NVTE_BWD_LAYERNORM_SM_MARGIN",
    "NVTE_NVTX_ENABLED",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
    "TRITON_HOME",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "PYTORCH_CUDA_ALLOC_CONF",
]


class SlurmBackend:
    """
    Stateless translator: JobSpec → sbatch script string or submitted job ID.
    """

    @staticmethod
    def render_script(job: JobSpec) -> str:
        """
        Render a complete sbatch script string from a JobSpec.

        Structure:
          1. #SBATCH headers
          2. MEGATRON_PATH / PYTHONPATH exports
          3. ENV_VARS exports (from job.env_vars)
          4. Distributed env var setup (MASTER_ADDR etc.)
          5. srun with Pyxis flags + torch.distributed.run launcher

        The srun --export list is generated from _SCRIPT_VARS + job.env_vars keys,
        so it always matches what the script actually exports.
        """
        training_script = (
            f"{job.megatron_path}/{job.training_script}"
            if not job.training_script.startswith("/")
            else job.training_script
        )
        if job.training_args_shell_expr:
            training_args_str = job.training_args_shell_expr
        else:
            training_args_str = " \\\n      ".join(job.training_args)

        # PYTHONPATH: patch path first (if set), then megatron path
        pythonpath_parts = [p for p in [job.megatron_patch_path, job.megatron_path] if p]
        pythonpath = ":".join(pythonpath_parts)

        # --- srun --export list: script vars + any extra env_vars keys ---
        all_export_vars = list(_SCRIPT_VARS) + [
            k for k in job.env_vars if k not in _SCRIPT_VARS
        ]
        srun_export = ",".join(all_export_vars)

        # --- env_vars export block ---
        env_export_lines = "\n".join(
            f'export {k}="{v}"' for k, v in job.env_vars.items()
        )

        sbatch_headers = _render_sbatch_headers(job)
        srun_container_flags = _render_srun_container_flags(job)

        script = f"""\
#!/bin/bash
{sbatch_headers}

set -x
ulimit -c 0

# --- Megatron paths ---
export MEGATRON_PATH="{job.megatron_path}"
export MEGATRON_PATCH_PATH="{job.megatron_patch_path}"
export PYTHONPATH="{pythonpath}:${{PYTHONPATH:-}}"

# --- ENV_VARS from config ---
{env_export_lines}

# --- Distributed setup ---
export LOCAL_RANK=${{SLURM_LOCALID}}
export RANK="${{SLURM_PROCID}}"
export GPUS_PER_NODE="${{SLURM_GPUS_ON_NODE}}"
export HOSTNAMES="$(scontrol show hostnames "$SLURM_JOB_NODELIST")"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT=6800
export WORLD_SIZE="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_CACHE_DISABLE=1

# --- Launch ---
srun \\
  --export={srun_export} \\
  --mpi=pmix -l \\
{srun_container_flags}  -u bash -lc '
    python -m torch.distributed.run \\
      --nproc_per_node "$SLURM_GPUS_ON_NODE" \\
      --nnodes "{job.nodes}" \\
      --node_rank "$SLURM_PROCID" \\
      --master_addr "$MASTER_ADDR" \\
      --master_port "$MASTER_PORT" \\
      {training_script} \\
      {training_args_str}
  '
"""
        return script

    @staticmethod
    def write_script(job: JobSpec, path: Path) -> Path:
        """Render and write the sbatch script to disk. Returns the path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        script = SlurmBackend.render_script(job)
        path.write_text(script)
        return path

    @staticmethod
    def submit(job: JobSpec, script_path: Path) -> str:
        """
        Submit a pre-written sbatch script and return the SLURM job ID.

        Args:
            job: JobSpec (used for dependency flag if set).
            script_path: Path to the rendered sbatch script on disk.

        Returns:
            The SLURM job ID as a string (e.g. "12345678").
        """
        cmd = ["sbatch"]
        if job.dependency:
            cmd += [f"--dependency={job.dependency}"]
        cmd.append(str(script_path))

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # sbatch stdout: "Submitted batch job 12345678"
        return result.stdout.strip().split()[-1]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _render_sbatch_headers(job: JobSpec) -> str:
    lines = [
        f"#SBATCH --nodes={job.nodes}",
        f"#SBATCH --account={job.account}",
        f"#SBATCH --partition={job.partition}",
        "#SBATCH --ntasks-per-node=1",
        f"#SBATCH --time={job.run_time}",
        f"#SBATCH --job-name={job.name}",
        f"#SBATCH --output={job.log_dir}/{job.name}-%A_%a.out" if job.array_spec else f"#SBATCH --output={job.log_dir}/{job.name}-%j.out",
        f"#SBATCH --error={job.log_dir}/{job.name}-%A_%a.err" if job.array_spec else f"#SBATCH --error={job.log_dir}/{job.name}-%j.err",
        "#SBATCH --exclusive",
        f"#SBATCH --gres=gpu:{job.gpus_per_node}",
    ]
    if job.array_spec:
        lines.append(f"#SBATCH --array={job.array_spec}")
    if job.dependency:
        lines.append(f"#SBATCH --dependency={job.dependency}")
    return "\n".join(lines)


def _render_srun_container_flags(job: JobSpec) -> str:
    lines = []
    if job.container_edf:
        lines.append(f'  --environment="{job.container_edf}" \\')
    if job.container_mounts:
        lines.append(f'  --container-mounts="{job.container_mounts}" \\')
    if job.megatron_path:
        lines.append(f'  --container-workdir="{job.megatron_path}" \\')
    lines.append("  --no-container-mount-home \\")
    return "\n".join(lines) + "\n"
