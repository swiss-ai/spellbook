"""
Experiment — base dataclass for a single run configuration.

Subclass this for each framework (MegatronExperiment, etc.).
The only required field is `name`. All other fields are framework-specific.

Usage::

    @dataclass
    class MyExp(Experiment):
        tp: int = 1
        pp: int = 1

    base = MyExp(name="base", tp=2, pp=4)
    variant = base.change("tp4", tp=4)          # only tp changes
    variants = base.sweep("tp", [1, 2, 4, 8])   # one-axis sweep
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass
class Experiment:
    name: str

    def change(self, name: str, **kwargs: Any) -> Experiment:
        """Return a new instance with name and given fields overridden."""
        return dataclasses.replace(self, name=name, **kwargs)

    def sweep(self, field: str, values: list[Any]) -> list[Experiment]:
        """One-axis sweep: return one variant per value, named <name>-<field><value>."""
        return [self.change(f"{self.name}-{field}{v}", **{field: v}) for v in values]

    def diff(self, other: Experiment) -> dict[str, tuple[Any, Any]]:
        """Return fields that differ between self and other (excluding name)."""
        result = {}
        for f in dataclasses.fields(self):
            if f.name == "name":
                continue
            a, b = getattr(self, f.name), getattr(other, f.name)
            if a != b:
                result[f.name] = (a, b)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Flat dict of all fields. Used by backends for template rendering."""
        return dataclasses.asdict(self)
