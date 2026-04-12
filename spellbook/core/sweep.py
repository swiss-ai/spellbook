"""
Sweep — a named collection of experiment variants.

A Sweep is the unit that gets submitted: it holds a list of Experiment
instances and knows which backend to use for rendering/submission.

For a single-axis sweep use sweep_axis()::

    sweep = sweep_axis(base, "tp", [1, 2, 4, 8], backend=slurm)

For multi-axis grids use sweep_grid()::

    sweep = sweep_grid(base, {"tp": [1, 2, 4], "ep": [4, 8]}, backend=slurm)

Or build the list by hand::

    sweep = Sweep(
        name="custom",
        experiments=[
            base.change("v1", tp=2, ep=8),
            base.change("v2", tp=4, ep=4),
        ],
        backend=slurm,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

from spellbook.core.experiment import Experiment


@dataclass
class Sweep:
    name: str
    experiments: list[Experiment]
    backend: Any  # SlurmBackend or any future backend

    def render(self, output_dir: str = "sbatch_scripts") -> list[str]:
        """Render all experiments via the backend. Returns list of rendered script paths."""
        return self.backend.render_all(self.name, self.experiments, output_dir=output_dir)

    def submit(self, output_dir: str = "sbatch_scripts") -> list[str]:
        """Render + submit all experiments. Returns list of job IDs."""
        return self.backend.submit_all(self.name, self.experiments, output_dir=output_dir)

    def changed_fields(self) -> list[str]:
        """Return sorted list of fields that differ across experiments."""
        if len(self.experiments) < 2:
            return []
        base = self.experiments[0]
        found: set[str] = set()
        for exp in self.experiments[1:]:
            found.update(base.diff(exp).keys())
        return sorted(found)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def sweep_axis(base: Experiment, field: str, values: list[Any], *, backend: Any, name: str | None = None) -> Sweep:
    """Single-axis sweep over `field` values."""
    sweep_name = name or f"{base.name}-{field}"
    return Sweep(
        name=sweep_name,
        experiments=base.sweep(field, values),
        backend=backend,
    )


def sweep_grid(base: Experiment, axes: dict[str, list[Any]], *, backend: Any, name: str | None = None) -> Sweep:
    """
    Multi-axis grid sweep.

    axes={"tp": [1, 2, 4], "ep": [4, 8]} produces tp*ep variants,
    each named <base.name>-tp<v>-ep<v>.
    """
    keys = list(axes.keys())
    sweep_name = name or f"{base.name}-{'_'.join(keys)}"
    variants = [
        base.change(
            "-".join([base.name] + [f"{k}{v}" for k, v in zip(keys, combo)]),
            **dict(zip(keys, combo)),
        )
        for combo in product(*axes.values())
    ]
    return Sweep(name=sweep_name, experiments=variants, backend=backend)
