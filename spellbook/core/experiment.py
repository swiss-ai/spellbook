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
from pathlib import Path
from typing import Any, ClassVar


@dataclasses.dataclass
class Experiment:
    name: str

    # Fields excluded from lock comparison. Override in subclasses to add
    # dynamic/derived fields that change on every instantiation.
    _lock_exclude: ClassVar[frozenset[str]] = frozenset()

    def change(self, name: str, **kwargs: Any) -> Experiment:
        """Return a new instance with name and given fields overridden."""
        child = dataclasses.replace(self, name=name, **kwargs)
        object.__setattr__(child, "_parent_name", self.name)
        return child

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

    def lock(self, locks_dir: Path = Path("locks")) -> Experiment:
        """Lock this experiment's config. Returns self for chaining.

        - No lock file yet: writes locks/<name>.lock.yaml.
        - Lock file exists: validates current config against it; raises on drift.

        To change a locked experiment: delete its .lock.yaml, update the config,
        and re-lock by running the experiment file again.
        """
        import yaml
        from datetime import datetime

        locks_dir.mkdir(parents=True, exist_ok=True)
        path = locks_dir / f"{self.name}.lock.yaml"
        current = {k: v for k, v in self.to_dict().items() if k not in self._lock_exclude}

        if not path.exists():
            data = {
                "name": self.name,
                "locked_at": datetime.now().isoformat(timespec="seconds"),
                "fields": current,
            }
            path.write_text(yaml.dump(data, default_flow_style=None, sort_keys=True))
            print(f"[lock] {path} written.")
        else:
            locked = yaml.safe_load(path.read_text()).get("fields", {})
            diffs = {
                k: (locked.get(k), current.get(k))
                for k in set(locked) | set(current)
                if locked.get(k) != current.get(k)
            }
            if diffs:
                lines = "\n".join(
                    f"  {k}: locked={v[0]!r}  current={v[1]!r}"
                    for k, v in sorted(diffs.items())
                )
                raise RuntimeError(
                    f"Experiment '{self.name}' differs from its lock file ({path}):\n{lines}\n\n"
                    f"To unlock: delete {path}, update the config, and call .lock() again."
                )

        object.__setattr__(self, "_locked", True)
        return self

