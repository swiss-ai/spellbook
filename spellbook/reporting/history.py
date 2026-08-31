"""WandB history loading, caching, model matching, and continuation stitching."""

from __future__ import annotations

import dataclasses
import fnmatch
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from spellbook.reporting.specs import (
    ChinchillaScalingLaw,
    EndpointScalingLaw,
    EvalMacro,
    EvalTrajectories,
    LearningRateBowl,
    LossAlignment,
    MetricCurves,
    ModelMetadata,
    Report,
    TaskHeatmap,
)

AXIS_KEYS: dict[str, tuple[str, ...]] = {
    "step": ("_step", "iteration", "step", "OptimizerStep"),
    "tokens": ("consumed-tokens", "consumed_tokens", "tokens", "ConsumedTokens"),
    "flops": ("flops", "cumulative-flops", "cumulative_flops"),
    "lr": ("learning-rate", "learning_rate", "lr"),
}


@dataclasses.dataclass
class RunSeries:
    model: str
    run_ids: list[str]
    run_names: list[str]
    config: dict[str, Any]
    frame: pd.DataFrame


def required_history_keys(report: Report) -> list[str]:
    keys = {key for aliases in AXIS_KEYS.values() for key in aliases}
    for plot in report.plots:
        if isinstance(
            plot,
            (
                LossAlignment,
                LearningRateBowl,
                EndpointScalingLaw,
                ChinchillaScalingLaw,
            ),
        ):
            keys.add(plot.metric)
        elif isinstance(plot, MetricCurves):
            keys.update(plot.metrics)
        elif isinstance(plot, (EvalMacro, EvalTrajectories, TaskHeatmap)):
            for metrics in plot.task_groups.values():
                keys.update(_wandb_metric_key(metric) for metric in metrics)
    return sorted(keys)


def resolve_column(frame: pd.DataFrame, requested: str) -> str | None:
    if requested in frame.columns:
        return requested
    for candidate in AXIS_KEYS.get(requested, (requested,)):
        if candidate in frame.columns:
            return candidate
    normalized = requested.lower().replace("_", "-")
    for column in frame.columns:
        if str(column).lower().replace("_", "-") == normalized:
            return str(column)
    return None


def ema(values: Sequence[float], beta: float | None) -> list[float]:
    clean = [float(value) for value in values]
    if beta is None or not clean:
        return clean
    if not 0.0 <= beta < 1.0:
        raise ValueError("EMA beta must satisfy 0 <= beta < 1")
    result = [clean[0]]
    for value in clean[1:]:
        result.append(beta * result[-1] + (1.0 - beta) * value)
    return result


def numeric_series(frame: pd.DataFrame, key: str) -> pd.Series:
    column = resolve_column(frame, key)
    if column is None:
        return pd.Series(index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def fetch_report_series(
    report: Report,
    cache_dir: Path,
    *,
    refresh: bool = False,
) -> list[RunSeries]:
    import wandb

    cache_dir.mkdir(parents=True, exist_ok=True)
    filters = {"group": report.source.group} if report.source.group else None
    api = wandb.Api(timeout=120)
    runs = list(api.runs(report.source.project, filters=filters))
    assignments: dict[str, list[tuple[Any, str | None]]] = {}
    metadata = report.selected_models()

    for run in runs:
        if report.source.runs and not any(
            candidate in report.source.runs for candidate in (run.id, run.name)
        ):
            continue
        namespace = _run_namespace(run, report.source.run_namespaces)
        if report.source.run_namespaces and namespace is None:
            continue
        model = _match_model(run, report, metadata)
        if model is not None:
            assignments.setdefault(model, []).append((run, namespace))

    if metadata:
        missing = [model.name for model in metadata if model.name not in assignments]
        if missing:
            raise ValueError(
                f"No WandB runs matched selected models: {', '.join(missing)}"
            )

    keys = required_history_keys(report)
    result = []
    for model, pieces in assignments.items():
        pieces.sort(key=lambda piece: str(getattr(piece[0], "created_at", "")))
        frames = []
        for piece_order, (run, namespace) in enumerate(pieces):
            frame = _load_run_history(run, keys, cache_dir, refresh=refresh)
            if frame.empty:
                continue
            if namespace:
                frame = _namespace_metrics(frame, namespace)
            frame["_spellbook_piece"] = piece_order
            frames.append(frame)
        if not frames:
            continue
        stitched = stitch_frames(
            frames,
            checkpoint_identity=bool(report.source.run_namespaces),
        )
        result.append(
            RunSeries(
                model=model,
                run_ids=[run.id for run, _ in pieces],
                run_names=[run.name for run, _ in pieces],
                config=dict(pieces[-1][0].config or {}),
                frame=stitched,
            )
        )
    model_order = [model.name for model in metadata] or list(report.select.names)
    order = {model: index for index, model in enumerate(model_order)}
    return sorted(result, key=lambda series: order.get(series.model, len(order)))


def stitch_frames(
    frames: Sequence[pd.DataFrame], *, checkpoint_identity: bool = False
) -> pd.DataFrame:
    frame = pd.concat(frames, ignore_index=True, sort=False)
    frame["_spellbook_row"] = range(len(frame))
    for aliases in AXIS_KEYS.values():
        present = [column for column in aliases if column in frame]
        if present:
            frame[aliases[0]] = frame[present].bfill(axis=1).iloc[:, 0]
    token_column = resolve_column(frame, "tokens")
    step_column = _checkpoint_column(frame)
    identity = step_column if checkpoint_identity else token_column or step_column
    if identity:
        frame[identity] = pd.to_numeric(frame[identity], errors="coerce")
        frame = frame.sort_values(
            [identity, "_spellbook_piece", "_spellbook_row"],
            kind="stable",
        )
        frame = frame.groupby(identity, as_index=False, sort=True).last()
    return frame.drop(columns="_spellbook_row", errors="ignore").reset_index(drop=True)


def _load_run_history(
    run, keys: Sequence[str], cache_dir: Path, *, refresh: bool
) -> pd.DataFrame:
    path = cache_dir / f"{run.id}.json"
    if path.exists() and not refresh:
        payload = json.loads(path.read_text())
        if payload.get("rows") and set(keys).issubset(payload.get("keys", [])):
            return pd.DataFrame(payload["rows"])

    available = set((run.summary or {}).keys())
    axis_keys = {key for aliases in AXIS_KEYS.values() for key in aliases}
    # Axis values are often logged in history without being copied into the
    # final W&B summary. Always request them so token/FLOP plots remain usable.
    requested = [key for key in keys if key in available or key in axis_keys]
    rows = []
    for row in run.scan_history(keys=requested, page_size=10_000):
        cleaned = {str(key): value for key, value in row.items() if _json_scalar(value)}
        if cleaned:
            rows.append(cleaned)
    if not rows:
        rows = [
            {str(key): value for key, value in row.items() if _json_scalar(value)}
            for row in run.history(samples=100_000, keys=requested, pandas=False)
        ]
        rows = [row for row in rows if row]
    path.write_text(
        json.dumps(
            {
                "run_id": run.id,
                "run_name": run.name,
                "keys": list(keys),
                "rows": rows,
            },
            allow_nan=False,
        )
    )
    frame = pd.DataFrame(rows)
    checkpoint = _checkpoint_column(frame)
    if checkpoint:
        frame = frame.drop_duplicates(checkpoint, keep="last")
    return frame.reset_index(drop=True)


def _checkpoint_column(frame: pd.DataFrame) -> str | None:
    for column in ("OptimizerStep", "iteration", "step", "_step"):
        if column in frame:
            return column
    return None


def _run_namespace(run, namespaces: Mapping[str, Sequence[str]]) -> str | None:
    matches = [
        namespace
        for namespace, patterns in namespaces.items()
        if any(
            fnmatch.fnmatchcase(str(value), pattern)
            for pattern in patterns
            for value in (run.id, run.name)
        )
    ]
    if len(matches) > 1:
        raise ValueError(f"WandB run {run.id} matches multiple run namespaces")
    return matches[0] if matches else None


def _wandb_metric_key(metric: str) -> str:
    return metric.partition("::")[2] or metric


def _namespace_metrics(frame: pd.DataFrame, namespace: str) -> pd.DataFrame:
    axis_keys = {key for aliases in AXIS_KEYS.values() for key in aliases}
    return frame.rename(
        columns={
            column: f"{namespace}::{column}"
            for column in frame.columns
            if column not in axis_keys
        }
    )


def _json_scalar(value: Any) -> bool:
    if value is None or isinstance(value, str | bool | int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def _match_model(run, report: Report, metadata: Sequence[ModelMetadata]) -> str | None:
    values = {**dict(run.config or {}), "name": run.name, "id": run.id}
    run_filter = dataclasses.replace(report.select, names=())
    if not run_filter.matches(values):
        return None
    if not metadata:
        if report.select.names:
            matches = []
            for name in report.select.names:
                if score := _alias_score(name, run):
                    matches.append((score, name))
            if not matches:
                return None
            matches.sort(reverse=True)
            if len(matches) > 1 and matches[0][0] == matches[1][0]:
                raise ValueError(
                    f"WandB run {run.id} ambiguously matches multiple models"
                )
            return matches[0][1]
        return run.name

    matches = []
    for model in metadata:
        aliases = [model.name]
        for key in ("wandb_exp_name", "wandb_id"):
            value = model.config.get(key)
            if value:
                aliases.append(str(value))
        score = max((_alias_score(alias, run) for alias in aliases), default=0)
        if score:
            matches.append((score, model.name))
    if not matches:
        return None
    matches.sort(reverse=True)
    if len(matches) > 1 and matches[0][0] == matches[1][0]:
        raise ValueError(f"WandB run {run.id} ambiguously matches multiple models")
    return matches[0][1]


def _alias_score(alias: str, run) -> int:
    alias_lower = alias.lower()
    if alias_lower == str(run.id).lower():
        return 10_000 + len(alias)
    if alias_lower == str(run.name).lower():
        return 9_000 + len(alias)
    pattern = rf"(?<![a-z0-9.]){re.escape(alias_lower)}(?![a-z0-9])"
    return len(alias) if re.search(pattern, str(run.name).lower()) else 0


def endpoint(frame: pd.DataFrame, metric: str, smoothing: float | None) -> float | None:
    values = numeric_series(frame, metric).dropna().tolist()
    if not values:
        return None
    return ema(values, smoothing)[-1]


def export_histories(series: Iterable[RunSeries], path: Path) -> None:
    frames = []
    for item in series:
        frame = item.frame.copy()
        frame.insert(0, "model", item.model)
        frames.append(frame)
    if frames:
        pd.concat(frames, ignore_index=True, sort=False).to_csv(path, index=False)
