"""Declarative report specifications.

Report definitions intentionally contain no WandB or plotting side effects. They
describe what to fetch and render; the reporting backend can resolve that plan
later without changing experiment or evaluation files.
"""

from __future__ import annotations

import dataclasses
import fnmatch
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar


@dataclasses.dataclass(frozen=True)
class WandbGroup:
    """A WandB project, optionally narrowed to one run group."""

    project: str
    group: str | None = None

    def __post_init__(self) -> None:
        if self.project.count("/") != 1:
            raise ValueError("WandbGroup.project must be '<entity>/<project>'")

    @property
    def entity(self) -> str:
        return self.project.split("/", 1)[0]

    @property
    def project_name(self) -> str:
        return self.project.split("/", 1)[1]


@dataclasses.dataclass(frozen=True)
class Between:
    """Inclusive numeric range used in model-selection predicates."""

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if self.minimum > self.maximum:
            raise ValueError("Between.minimum cannot exceed maximum")

    def matches(self, value: Any) -> bool:
        return isinstance(value, int | float) and self.minimum <= value <= self.maximum


Predicate = Between | Callable[[Any], bool] | Any
ParameterCounter = Callable[[Any], Mapping[str, float]]


def _callable_name(value: Callable[..., Any]) -> str:
    module = getattr(value, "__module__", type(value).__module__)
    name = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}.{name}"


@dataclasses.dataclass(frozen=True)
class Models:
    """Select model/run IDs before histories are downloaded.

    ``names`` are exact stable IDs, ``where`` matches config fields, and
    ``exclude`` applies shell-style globs after the positive filters.
    """

    names: Sequence[str] = ()
    where: Mapping[str, Predicate] = dataclasses.field(default_factory=dict)
    exclude: Sequence[str] = ()

    def matches(self, model: Any) -> bool:
        values = _model_values(model)
        name = str(values.get("name", ""))
        if self.names and name not in self.names:
            return False
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in self.exclude):
            return False
        return all(_matches_predicate(values.get(field), expected) for field, expected in self.where.items())

    def select(self, models: Sequence[Any]) -> list[Any]:
        selected = [model for model in models if self.matches(model)]
        if self.names:
            found = {str(_model_values(model).get("name", "")) for model in selected}
            missing = [name for name in self.names if name not in found]
            if missing:
                raise ValueError(f"Selected model IDs were not found: {', '.join(missing)}")
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
        return {field.name: getattr(model, field.name) for field in dataclasses.fields(model)}
    return vars(model)


def _matches_predicate(value: Any, expected: Predicate) -> bool:
    if isinstance(expected, Between):
        return expected.matches(value)
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
    def from_experiment(
        cls,
        experiment: Any,
        parameter_counter: ParameterCounter | None = None,
    ) -> ModelMetadata:
        values = dict(_model_values(experiment))
        total_params = active_params = None
        source = None
        if parameter_counter is not None:
            counts = parameter_counter(experiment)
            source = _callable_name(parameter_counter)
        elif hasattr(experiment, "parameter_counts"):
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
class Plot:
    """Base class for declarative plot requests."""

    kind: ClassVar[str] = "plot"


@dataclasses.dataclass(frozen=True)
class LossAlignment(Plot):
    kind: ClassVar[str] = "loss_alignment"
    axes: Sequence[str] = ("tokens",)
    ema: float | None = 0.9


@dataclasses.dataclass(frozen=True)
class EndpointScalingLaw(Plot):
    kind: ClassVar[str] = "endpoint_scaling_law"
    x: Sequence[str] = ("total_params", "active_params", "flops")


@dataclasses.dataclass(frozen=True)
class EvalMacro(Plot):
    kind: ClassVar[str] = "eval_macro"
    task_groups: Mapping[str, Sequence[str]] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class EvalTrajectories(Plot):
    kind: ClassVar[str] = "eval_trajectories"
    task_groups: Mapping[str, Sequence[str]] = dataclasses.field(default_factory=dict)
    axes: Sequence[str] = ("tokens", "flops")


@dataclasses.dataclass(frozen=True)
class TaskHeatmap(Plot):
    kind: ClassVar[str] = "task_heatmap"
    task_groups: Mapping[str, Sequence[str]] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class MetricCurves(Plot):
    kind: ClassVar[str] = "metric_curves"
    metrics: Sequence[str] = ()
    x: str = "tokens"
    ema: float | None = 0.9


@dataclasses.dataclass(frozen=True)
class Report:
    """A report owned by one experiment or evaluation definition."""

    name: str
    source: WandbGroup
    plots: Sequence[Plot]
    select: Models = dataclasses.field(default_factory=Models)
    models: Sequence[ModelMetadata] = ()
    parameter_counter: ParameterCounter | None = dataclasses.field(
        default=None,
        repr=False,
        compare=False,
    )

    def with_experiments(self, experiments: Sequence[Any]) -> Report:
        selected = self.select.select(experiments)
        return dataclasses.replace(
            self,
            models=tuple(
                ModelMetadata.from_experiment(exp, self.parameter_counter)
                for exp in selected
            ),
        )

    def selected_models(self) -> list[ModelMetadata]:
        return self.select.select(self.models)

    def to_dict(self) -> dict[str, Any]:
        selected = self.selected_models()
        return {
            "name": self.name,
            "source": dataclasses.asdict(self.source),
            "select": _jsonable(dataclasses.asdict(self.select)),
            "parameter_counter": (
                _callable_name(self.parameter_counter)
                if self.parameter_counter is not None else None
            ),
            "models": [_jsonable(dataclasses.asdict(model)) for model in selected],
            "plots": [
                {"kind": plot.kind, **_jsonable(dataclasses.asdict(plot))}
                for plot in self.plots
            ],
        }


@dataclasses.dataclass(frozen=True)
class EvalReport(Report):
    """Semantic alias for a report declared beside evaluation definitions."""


@dataclasses.dataclass(frozen=True)
class CombinedReport:
    """Compose training and evaluation reports under one shared selector."""

    name: str
    training: Report
    evaluations: EvalReport
    select: Models = dataclasses.field(default_factory=Models)

    def to_dict(self) -> dict[str, Any]:
        training = dataclasses.replace(self.training, select=self.select)
        evaluations = dataclasses.replace(self.evaluations, select=self.select)
        train_models = training.selected_models()
        eval_models = evaluations.selected_models()
        if train_models and eval_models:
            train_names = {model.name for model in train_models}
            eval_names = {model.name for model in eval_models}
            if train_names != eval_names:
                missing_evals = sorted(train_names - eval_names)
                missing_training = sorted(eval_names - train_names)
                raise ValueError(
                    "Combined report model mismatch: "
                    f"missing evaluations={missing_evals}, missing training={missing_training}"
                )
        return {
            "name": self.name,
            "training": training.to_dict(),
            "evaluations": evaluations.to_dict(),
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if callable(value):
        return getattr(value, "__qualname__", repr(value))
    return value
