"""Load report specifications from ordinary Python definition files."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from spellbook.reporting.specs import CombinedReport, Report


def load_report(path: str | Path, variable: str | None = None) -> Report | CombinedReport:
    definition = Path(path).resolve()
    if not definition.exists():
        raise FileNotFoundError(definition)

    module_name = f"_spellbook_report_{abs(hash(definition))}"
    spec = importlib.util.spec_from_file_location(module_name, definition)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load report definition from {definition}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if variable:
        report = getattr(module, variable, None)
        if not isinstance(report, Report | CombinedReport):
            raise ValueError(f"{definition} does not define report variable {variable!r}")
    else:
        candidates = [
            value for value in vars(module).values()
            if isinstance(value, Report | CombinedReport)
        ]
        unique = list({id(value): value for value in candidates}.values())
        if len(unique) != 1:
            raise ValueError(
                f"{definition} must define exactly one report object, found {len(unique)}; "
                "use --variable when the module intentionally defines more"
            )
        report = unique[0]

    sweep = getattr(module, "sweep", None)
    if isinstance(report, Report) and sweep is not None and not report.models:
        report = report.with_experiments(sweep.experiments)
    return report
