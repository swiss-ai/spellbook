from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# JobSpec — handoff between MegatronBackend and SlurmBackend
# ---------------------------------------------------------------------------

@dataclass
class JobSpec:
    """
    The handoff point between MegatronBackend and SlurmBackend.

    MegatronBackend populates training_args.
    SlurmBackend consumes everything to produce an sbatch script.
    """
    # --- Job identity ---
    name: str
    account: str
    partition: str
    nodes: int
    gpus_per_node: int
    run_time: str                       # "HH:MM:SS"

    # --- Megatron ---
    megatron_path: str                  # abs path, used as container workdir + PYTHONPATH
    megatron_patch_path: str = ""       # prepended to PYTHONPATH before megatron_path
    training_script: str = "pretrain_gpt.py"   # relative to megatron_path
    training_args: list[str] = field(default_factory=list)  # output of MegatronBackend.to_args()

    # --- Container (Pyxis/Enroot) ---
    container_edf: str = ""             # abs path to .toml EDF file
    container_mounts: str = ""          # "$SCRATCH:$SCRATCH,..."

    # --- Environment ---
    env_vars: dict[str, str] = field(default_factory=dict)  # from ENV_VARS config section

    # --- Output ---
    log_dir: Path = Path("slurm_logs")

    # --- Array jobs ---
    # Set to e.g. "0-4%4" to emit #SBATCH --array=... in the script.
    # When set, training_args is ignored and training_args_shell_expr is used instead.
    array_spec: str | None = None
    # Shell expression that expands to the training args at runtime (for array jobs).
    # e.g. "$(python3 -c "...")"
    training_args_shell_expr: str = ""

    # --- Chain sequencing ---
    dependency: str | None = None       # e.g. "afterok:12345"


# ---------------------------------------------------------------------------
# Module status
# ---------------------------------------------------------------------------

class ModuleStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# ModuleResult — what a module passes downstream
# ---------------------------------------------------------------------------

@dataclass
class ModuleResult:
    """Returned by Module.run(). Accumulated in Context."""
    module_name: str
    status: ModuleStatus
    # Arbitrary key-value data for downstream modules (job IDs, paths, metrics…)
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


# ---------------------------------------------------------------------------
# Context — accumulates results as the pipeline progresses
# ---------------------------------------------------------------------------

class Context:
    """
    Shared state passed through a pipeline run.

    Modules write results via ctx.record(result) and read upstream
    results via ctx[module_name] or ctx.get(module_name).
    """

    def __init__(self, project_name: str, dry_run: bool = False):
        self.project_name = project_name
        self.dry_run = dry_run
        self._results: dict[str, ModuleResult] = {}

    def record(self, result: ModuleResult) -> None:
        self._results[result.module_name] = result

    def get(self, module_name: str) -> ModuleResult | None:
        return self._results.get(module_name)

    def __getitem__(self, module_name: str) -> ModuleResult:
        if module_name not in self._results:
            raise KeyError(f"No result for module '{module_name}' in context")
        return self._results[module_name]

    def __contains__(self, module_name: str) -> bool:
        return module_name in self._results

    def all_results(self) -> dict[str, ModuleResult]:
        return dict(self._results)


# ---------------------------------------------------------------------------
# Module base class
# ---------------------------------------------------------------------------

class Module:
    """
    Base class for all pipeline modules (ExperimentChain, Sweep, …).

    Subclasses implement run() and optionally override status().
    depends_on is a list of module names that must complete before this one runs.
    """

    def __init__(self, name: str, depends_on: list[str] | None = None):
        self.name = name
        self.depends_on: list[str] = depends_on or []

    def run(self, ctx: Context) -> ModuleResult:
        raise NotImplementedError(f"{self.__class__.__name__}.run() not implemented")

    def status(self) -> ModuleStatus:
        return ModuleStatus.PENDING

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
