"""Declarative report specifications.

Report definitions intentionally contain no WandB or plotting side effects. They
describe what to fetch and render; the reporting backend can resolve that plan
later without changing experiment or evaluation files.
"""

from __future__ import annotations

import dataclasses
import fnmatch
from collections.abc import Mapping, Sequence
from typing import Any

AxisRange = tuple[float | None, float | None]


@dataclasses.dataclass(frozen=True)
class WandbGroup:
    """A WandB project, optionally narrowed to one run group."""

    project: str
    group: str | None = None
    runs: Sequence[str] = ()
    run_namespaces: Mapping[str, Sequence[str]] = dataclasses.field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.project.count("/") != 1:
            raise ValueError("WandbGroup.project must be '<entity>/<project>'")


@dataclasses.dataclass(frozen=True)
class Models:
    """Select model/run IDs before histories are downloaded.

    ``names`` are exact stable IDs, ``where`` matches config fields, and
    ``exclude`` applies shell-style globs after the positive filters.
    """

    names: Sequence[str] = ()
    where: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    exclude: Sequence[str] = ()

    def matches(self, model: Any) -> bool:
        values = _model_values(model)
        name = str(values.get("name", ""))
        if self.names and name not in self.names:
            return False
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in self.exclude):
            return False
        return all(
            _matches_predicate(values.get(field), expected)
            for field, expected in self.where.items()
        )

    def select(self, models: Sequence[Any]) -> list[Any]:
        selected = [model for model in models if self.matches(model)]
        if self.names:
            found = {str(_model_values(model).get("name", "")) for model in selected}
            missing = [name for name in self.names if name not in found]
            if missing:
                raise ValueError(
                    f"Selected model IDs were not found: {', '.join(missing)}"
                )
        return selected


def _model_values(model: Any) -> Mapping[str, Any]:
    if isinstance(model, Mapping):
        return model
    if isinstance(model, ModelMetadata):
        return {
            **model.config,
            "name": model.name,
            "total_params": model.total_params,
            "active_params": model.active_params,
            "parameter_source": model.parameter_source,
        }
    if hasattr(model, "to_dict"):
        return model.to_dict()
    if dataclasses.is_dataclass(model) and not isinstance(model, type):
        return {
            field.name: getattr(model, field.name)
            for field in dataclasses.fields(model)
        }
    return vars(model)


def _matches_predicate(value: Any, expected: Any) -> bool:
    if callable(expected):
        return bool(expected(value))
    return value == expected


@dataclasses.dataclass(frozen=True)
class ModelMetadata:
    """Stable model identity and parameter-accounting provenance."""

    name: str
    total_params: float | None = None
    active_params: float | None = None
    config: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    parameter_source: str | None = None

    @classmethod
    def from_experiment(cls, experiment: Any) -> ModelMetadata:
        values = dict(_model_values(experiment))
        total_params = active_params = None
        source = None
        if hasattr(experiment, "parameter_counts"):
            counts = experiment.parameter_counts()
            source = f"{type(experiment).__module__}.{type(experiment).__qualname__}.parameter_counts"
        else:
            counts = None
        if counts is not None:
            total_params = float(counts["total_B"]) * 1e9
            active_params = float(counts["active_B"]) * 1e9
        return cls(
            name=str(values.pop("name")),
            total_params=total_params,
            active_params=active_params,
            config=values,
            parameter_source=source,
        )


@dataclasses.dataclass(frozen=True)
class LossAlignment:
    axes: Sequence[str] = ("tokens",)
    ema: float | None = 0.9
    metric: str = "lm loss"
    title: str | None = None
    xlim: Mapping[str, AxisRange] = dataclasses.field(default_factory=dict)
    ylim: AxisRange | None = None


@dataclasses.dataclass(frozen=True)
class LearningRateBowl:
    learning_rate: str = "matrix_lr"
    group_by: str | None = None
    metric: str = "lm loss"
    ema: float | None = 0.9
    log_x: bool = True
    xlim: AxisRange | None = None
    ylim: AxisRange | None = None
    title: str | None = None


@dataclasses.dataclass(frozen=True)
class EndpointScalingLaw:
    x: Sequence[str] = ("total_params", "active_params", "tokens", "flops")
    metric: str = "lm loss"
    ema: float | None = 0.9
    title: str | None = None


@dataclasses.dataclass(frozen=True)
class ChinchillaScalingLaw:
    parameter: str = "active_params"
    metric: str = "lm loss"
    ema: float | None = 0.9
    min_tokens: float = 0.0
    samples_per_model: int = 64
    title: str | None = None


@dataclasses.dataclass(frozen=True)
class EvalMacro:
    task_groups: Mapping[str, Sequence[str]] = dataclasses.field(default_factory=dict)
    chance_baselines: Mapping[str, float] = dataclasses.field(default_factory=dict)
    chance_adjusted: bool = False
    value_scale: float = 100.0
    title: str | None = None


@dataclasses.dataclass(frozen=True)
class EvalTrajectories:
    task_groups: Mapping[str, Sequence[str]] = dataclasses.field(default_factory=dict)
    axes: Sequence[str] = ("tokens", "flops")
    chance_baselines: Mapping[str, float] = dataclasses.field(default_factory=dict)
    chance_adjusted: bool = False
    value_scale: float = 100.0
    title: str | None = None


@dataclasses.dataclass(frozen=True)
class TaskHeatmap:
    task_groups: Mapping[str, Sequence[str]] = dataclasses.field(default_factory=dict)
    grouped: bool = False
    value_scale: float = 100.0
    decimals: int = 4
    higher_is_better: bool = True
    task_labels: Mapping[str, str] = dataclasses.field(default_factory=dict)
    namespace_labels: Mapping[str, str] = dataclasses.field(default_factory=dict)
    metric_labels: Mapping[str, str] = dataclasses.field(default_factory=dict)
    title: str | None = None


@dataclasses.dataclass(frozen=True)
class MetricCurves:
    metrics: Sequence[str] = ()
    x: str = "tokens"
    ema: float | None = 0.9
    better: Mapping[str, str] = dataclasses.field(default_factory=dict)
    y_limits: Mapping[str, tuple[float, float]] = dataclasses.field(
        default_factory=dict
    )
    title: str | None = None


PlotSpec = (
    LossAlignment
    | LearningRateBowl
    | EndpointScalingLaw
    | ChinchillaScalingLaw
    | EvalMacro
    | EvalTrajectories
    | TaskHeatmap
    | MetricCurves
)


@dataclasses.dataclass(frozen=True)
class Report:
    """A report owned by one experiment or evaluation definition."""

    name: str
    source: WandbGroup
    plots: Sequence[PlotSpec]
    select: Models = dataclasses.field(default_factory=Models)
    models: Sequence[ModelMetadata] = ()

    def with_experiments(self, experiments: Sequence[Any]) -> Report:
        selected = self.select.select(experiments)
        return dataclasses.replace(
            self,
            models=tuple(ModelMetadata.from_experiment(exp) for exp in selected),
        )

    def selected_models(self) -> list[ModelMetadata]:
        if not self.models:
            return []
        return self.select.select(self.models)

    def to_dict(self) -> dict[str, Any]:
        selected = self.selected_models()
        return {
            "name": self.name,
            "source": dataclasses.asdict(self.source),
            "select": _jsonable(dataclasses.asdict(self.select)),
            "models": [_jsonable(dataclasses.asdict(model)) for model in selected],
            "plots": [
                {"kind": type(plot).__name__, **_jsonable(dataclasses.asdict(plot))}
                for plot in self.plots
            ],
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if callable(value):
        return getattr(value, "__qualname__", repr(value))
    return value
