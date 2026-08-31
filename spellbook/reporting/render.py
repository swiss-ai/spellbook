"""Matplotlib renderer for declarative Spellbook reports."""

from __future__ import annotations

import dataclasses
import json
import math
import re
import textwrap
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter, ScalarFormatter
from scipy.optimize import curve_fit

from spellbook.reporting.history import (
    RunSeries,
    ema,
    endpoint,
    export_histories,
    fetch_report_series,
    numeric_series,
)
from spellbook.reporting.specs import (
    ChinchillaScalingLaw,
    EndpointScalingLaw,
    EvalMacro,
    EvalTrajectories,
    LearningRateBowl,
    LossAlignment,
    MetricCurves,
    Report,
    TaskHeatmap,
)
from spellbook.reporting.style import (
    HEATMAP_CMAP,
    apply_style,
    finish_axis,
    model_colors,
)

TASK_LABELS = {
    "agieval_lsat_ar": "AGIEval LSAT-AR",
    "arc_challenge": "ARC Challenge",
    "arc_easy": "ARC Easy",
    "boolq": "BoolQ",
    "commonsense_qa": "CommonsenseQA",
    "copa": "COPA",
    "gsm8k": "GSM8K",
    "hellaswag": "HellaSwag",
    "humaneval": "HumanEval",
    "lambada_openai": "LAMBADA OpenAI",
    "mbpp": "MBPP",
    "mmlu": "MMLU",
    "openbookqa": "OpenBookQA",
    "pawsx": "PAWS-X",
    "piqa": "PIQA",
    "winogrande": "WinoGrande",
    "xcopa": "XCOPA",
    "xnli": "XNLI",
    "xwinograd": "XWinograd",
}


@dataclasses.dataclass(frozen=True)
class Artifact:
    title: str
    stem: str


def render_report(
    report: Report,
    output_dir: str | Path,
    *,
    refresh: bool = False,
) -> Path:
    root = Path(output_dir) / _slug(report.name)
    root.mkdir(parents=True, exist_ok=True)
    series = fetch_report_series(report, root / ".cache", refresh=refresh)
    if not series:
        raise ValueError("No non-empty WandB histories matched this report")

    apply_style()
    artifacts = render_plots(report, series, root)
    export_histories(series, root / "histories.csv")
    manifest = report.to_dict()
    manifest["runs"] = [
        {
            "model": item.model,
            "run_ids": item.run_ids,
            "run_names": item.run_names,
            "rows": len(item.frame),
        }
        for item in series
    ]
    manifest["artifacts"] = [dataclasses.asdict(artifact) for artifact in artifacts]
    (root / "report.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )
    return root


def render_plots(
    report: Report, series: Sequence[RunSeries], root: Path
) -> list[Artifact]:
    artifacts = []
    for index, plot in enumerate(report.plots, start=1):
        stem = f"{index:02d}-{_slug(type(plot).__name__)}"
        if isinstance(plot, LossAlignment):
            artifact = _loss_alignment(plot, series, root, stem)
        elif isinstance(plot, LearningRateBowl):
            artifact = _learning_rate_bowl(plot, series, root, stem)
        elif isinstance(plot, MetricCurves):
            artifact = _metric_curves(plot, series, root, stem)
        elif isinstance(plot, EndpointScalingLaw):
            artifact = _endpoint_laws(plot, report, series, root, stem)
        elif isinstance(plot, ChinchillaScalingLaw):
            artifact = _chinchilla_law(plot, report, series, root, stem)
        elif isinstance(plot, EvalMacro):
            artifact = _eval_macro(plot, series, root, stem)
        elif isinstance(plot, EvalTrajectories):
            artifact = _eval_trajectories(plot, series, root, stem)
        elif isinstance(plot, TaskHeatmap):
            artifact = _task_heatmap(plot, series, root, stem)
        else:
            raise TypeError(f"Unsupported plot specification: {type(plot).__name__}")
        artifacts.append(artifact)
    return artifacts


def _loss_alignment(
    plot: LossAlignment,
    series: Sequence[RunSeries],
    root: Path,
    stem: str,
) -> Artifact:
    axes_names = list(plot.axes)
    fig, axes = plt.subplots(
        1,
        len(axes_names),
        figsize=(5.1 * len(axes_names), 4.8),
        sharey=True,
        squeeze=False,
    )
    palette = model_colors([item.model for item in series])
    endpoints: list[list[tuple[float, float, str, str]]] = [[] for _ in axes_names]
    for item in series:
        loss = numeric_series(item.frame, plot.metric)
        valid = loss.notna()
        frame = item.frame.loc[valid].reset_index(drop=True)
        loss_values = np.asarray(ema(loss.loc[valid].tolist(), plot.ema))
        if not len(loss_values):
            continue
        for column, axis_name in enumerate(axes_names):
            x, y = _aligned_values(frame, loss_values, axis_name)
            if not len(x):
                continue
            axis = axes[0, column]
            axis.plot(x, y, color=palette[item.model], linewidth=2.0, label=item.model)
            axis.scatter(
                x[-1],
                y[-1],
                marker="*",
                s=95,
                color=palette[item.model],
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
            )
            endpoints[column].append(
                (float(x[-1]), float(y[-1]), item.model, palette[item.model])
            )

    titles = {
        "tokens": "Same consumed tokens",
        "flops": "Same cumulative compute",
        "lr_cooldown": "Same normalized LR during cooldown",
        "step": "Same optimizer step",
    }
    xlabels = {
        "tokens": "consumed tokens (B)",
        "flops": "log10(cumulative FLOPs)",
        "lr_cooldown": "current LR / peak LR (1 to 0)",
        "step": "optimizer step",
    }
    for column, axis_name in enumerate(axes_names):
        axis = axes[0, column]
        axis.set_title(titles.get(axis_name, axis_name))
        axis.set_xlabel(xlabels.get(axis_name, axis_name))
        if axis_name in plot.xlim:
            axis.set_xlim(*plot.xlim[axis_name])
        if axis_name == "lr_cooldown":
            axis.invert_xaxis()
        if plot.ylim:
            axis.set_ylim(*plot.ylim)
        finish_axis(axis)
    axes[0, 0].set_ylabel(
        f"{plot.metric} (EMA {plot.ema})" if plot.ema else plot.metric
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=min(6, len(labels)),
    )
    title = plot.title or "Loss alignment"
    fig.suptitle(title, fontsize=15, y=0.99)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.79, bottom=0.15, wspace=0.12)
    fig.canvas.draw()
    for column, points in enumerate(endpoints):
        _annotate_spread_endpoints(axes[0, column], points, fig.dpi)
    _save(fig, root, stem)
    return Artifact(title, stem)


def _learning_rate_bowl(
    plot: LearningRateBowl,
    series: Sequence[RunSeries],
    root: Path,
    stem: str,
) -> Artifact:
    grouped: dict[str, list[tuple[float, float, float, str]]] = {}
    for item in series:
        learning_rate = item.config.get(plot.learning_rate)
        smoothed = endpoint(item.frame, plot.metric, plot.ema)
        raw = endpoint(item.frame, plot.metric, None)
        if learning_rate is None or smoothed is None or raw is None:
            continue
        group = str(item.config.get(plot.group_by, "all")) if plot.group_by else "all"
        grouped.setdefault(group, []).append(
            (float(learning_rate), smoothed, raw, item.model)
        )
    if not grouped:
        raise ValueError(
            f"No runs contain {plot.learning_rate!r} and {plot.metric!r} for the LR bowl"
        )

    columns = min(4, len(grouped))
    rows = math.ceil(len(grouped) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.4 * columns, 4.2 * rows),
        sharey=True,
        squeeze=False,
    )
    colors = model_colors(list(grouped))
    for index, (group, points) in enumerate(sorted(grouped.items())):
        axis = axes.flat[index]
        points.sort(key=lambda point: point[0])
        learning_rates = np.asarray([point[0] for point in points])
        smoothed = np.asarray([point[1] for point in points])
        raw = np.asarray([point[2] for point in points])
        color = colors[group]
        axis.plot(
            learning_rates,
            smoothed,
            color=color,
            linewidth=2.0,
            marker="o",
            label=f"final EMA {plot.ema}" if plot.ema else "final value",
        )
        axis.scatter(
            learning_rates,
            raw,
            marker="D",
            s=48,
            facecolor="none",
            edgecolor=color,
            linewidth=1.2,
            label="raw endpoint",
        )
        winner = int(np.nanargmin(smoothed))
        axis.scatter(
            learning_rates[winner],
            smoothed[winner],
            marker="*",
            s=155,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
            label="best final value",
        )
        for learning_rate, value in zip(learning_rates, smoothed, strict=True):
            axis.annotate(
                f"{value:.4f}",
                (learning_rate, value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        axis.set_title(f"{plot.group_by} {group}" if plot.group_by else "all runs")
        axis.set_xlabel(plot.learning_rate)
        if plot.log_x:
            axis.set_xscale("log", base=2)
            axis.set_xticks(np.unique(learning_rates))
            axis.xaxis.set_major_formatter(ScalarFormatter())
        if plot.xlim:
            axis.set_xlim(*plot.xlim)
        if plot.ylim:
            axis.set_ylim(*plot.ylim)
        else:
            axis.margins(y=0.14)
        finish_axis(axis)
    for index in range(len(grouped), rows * columns):
        axes.flat[index].set_visible(False)
    axes.flat[0].set_ylabel(
        f"{plot.metric} (EMA {plot.ema})" if plot.ema else plot.metric
    )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=len(labels),
    )
    title = plot.title or "Learning-rate bowls"
    fig.suptitle(title, fontsize=15, y=0.995)
    fig.subplots_adjust(
        left=0.07, right=0.99, top=0.76, bottom=0.13, hspace=0.40, wspace=0.14
    )
    _save(fig, root, stem)
    return Artifact(title, stem)


def _metric_curves(
    plot: MetricCurves,
    series: Sequence[RunSeries],
    root: Path,
    stem: str,
) -> Artifact:
    metrics = list(plot.metrics)
    columns = min(3, max(1, len(metrics)))
    rows = math.ceil(len(metrics) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(5.2 * columns, 4.0 * rows), squeeze=False
    )
    palette = model_colors([item.model for item in series])

    for index, metric in enumerate(metrics):
        axis = axes.flat[index]
        endpoints = {}
        for item in series:
            x = numeric_series(item.frame, plot.x)
            y = numeric_series(item.frame, metric)
            valid = x.notna() & y.notna()
            if plot.x == "flops":
                valid &= x > 0
            x_values = _scale_axis(x.loc[valid].to_numpy(), plot.x)
            y_values = np.asarray(ema(y.loc[valid].tolist(), plot.ema))
            if not len(x_values):
                continue
            axis.plot(
                x_values,
                y_values,
                color=palette[item.model],
                linewidth=1.9,
                label=item.model,
            )
            endpoints[item.model] = (x_values[-1], y_values[-1])
        winner = _winner(endpoints, plot.better.get(metric, "min"))
        if winner:
            x_value, y_value = endpoints[winner]
            axis.scatter(
                x_value,
                y_value,
                marker="*",
                s=125,
                color=palette[winner],
                edgecolor="white",
                linewidth=0.8,
                zorder=5,
            )
            axis.annotate(
                f"best: {winner} ({y_value:.4f})",
                (x_value, y_value),
                xytext=(-5, 8),
                textcoords="offset points",
                ha="right",
                fontsize=8.5,
                color=palette[winner],
                fontweight="bold",
            )
        if metric in plot.y_limits:
            axis.set_ylim(*plot.y_limits[metric])
        comparison = {
            "min": "lower is better",
            "max": "higher is better",
            "abs_zero": "closer to 0 is better",
        }.get(plot.better.get(metric, "min"), "")
        axis.set_title(f"{metric}\n{comparison}", fontsize=10)
        axis.set_xlabel(_axis_label(plot.x))
        finish_axis(axis)

    for index in range(len(metrics), rows * columns):
        axes.flat[index].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=min(6, len(labels)),
    )
    title = plot.title or "Training metrics"
    fig.suptitle(title, fontsize=15, y=0.995)
    fig.subplots_adjust(
        left=0.07, right=0.99, top=0.78, bottom=0.10, hspace=0.40, wspace=0.20
    )
    _save(fig, root, stem)
    return Artifact(title, stem)


def _endpoint_laws(
    plot: EndpointScalingLaw,
    report: Report,
    series: Sequence[RunSeries],
    root: Path,
    stem: str,
) -> Artifact:
    metadata = {model.name: model for model in report.selected_models()}
    endpoints = {
        item.model: endpoint(item.frame, plot.metric, plot.ema) for item in series
    }
    x_specs = []
    for key in plot.x:
        values = []
        models = []
        for item in series:
            value = _law_x(key, item, metadata.get(item.model))
            loss = endpoints[item.model]
            if value and loss is not None:
                models.append(item.model)
                values.append((value, loss))
        if len(values) >= 5:
            x_specs.append((key, models, values))
    if not x_specs:
        raise ValueError(
            "Endpoint scaling laws require at least five points with x metadata"
        )

    fig, axes = plt.subplots(
        2,
        len(x_specs),
        figsize=(5.7 * len(x_specs), 7.4),
        gridspec_kw={"height_ratios": [3.0, 1.15]},
        squeeze=False,
    )
    palette = model_colors([item.model for item in series])
    fit_payload = {}
    for column, (key, models, values) in enumerate(x_specs):
        x = np.asarray([value[0] for value in values], dtype=float)
        y = np.asarray([value[1] for value in values], dtype=float)
        if np.allclose(y, y[0]):
            raise ValueError(
                f"Cannot fit {key}: every endpoint has the same {plot.metric}"
            )
        floor_upper = max(1e-8, float(y.min()) - 1e-6)
        params, covariance = curve_fit(
            _power_law,
            x,
            y,
            p0=(floor_upper * 0.9, 1.0, 0.4),
            bounds=([0.0, 0.0, 0.001], [floor_upper, np.inf, 5.0]),
            maxfev=100_000,
        )
        prediction = _power_law(x, *params)
        residual = y - prediction
        r2 = 1.0 - float(np.sum(residual**2) / np.sum((y - y.mean()) ** 2))
        grid = np.geomspace(x.min() * 0.92, x.max() * 1.08, 400)
        top, bottom = axes[0, column], axes[1, column]
        top.plot(grid, _power_law(grid, *params), color="#202428", linewidth=2.2)
        for index, model in enumerate(models):
            top.scatter(
                x[index],
                y[index],
                s=62,
                color=palette[model],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            label = {
                "total_params": f"{model} - {x[index]:.3f}B total",
                "active_params": f"{model} - {x[index]:.3f}B active",
                "tokens": f"{model} - {x[index]:.1f}B tokens",
            }.get(key, model)
            top.annotate(
                label,
                (x[index], y[index]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8.5,
            )
            bottom.scatter(
                x[index],
                residual[index],
                s=48,
                color=palette[model],
                edgecolor="white",
                linewidth=0.8,
            )
        top.set_xscale("log")
        top.set_title(_law_title(key))
        top.set_ylabel(f"final {plot.metric} (EMA {plot.ema})" if column == 0 else "")
        top.text(
            0.04,
            0.06,
            f"E={params[0]:.4f}  |  A={params[1]:.4f}  |  α={params[2]:.4f}\n"
            f"R²={r2:.5f}",
            transform=top.transAxes,
            fontsize=9,
            bbox={
                "boxstyle": "square,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#C6C9CB",
                "alpha": 0.92,
            },
        )
        bottom.axhline(0.0, color="#555B60", linewidth=1.1)
        bottom.set_xscale("log")
        bottom.set_xlabel(_law_label(key))
        bottom.set_ylabel("observed - fitted" if column == 0 else "")
        for axis in (top, bottom):
            axis.xaxis.set_major_formatter(ScalarFormatter())
            axis.xaxis.set_minor_formatter(NullFormatter())
            finish_axis(axis)
        limit = max(0.006, float(np.max(np.abs(residual))) * 1.35)
        bottom.set_ylim(-limit, limit)
        fit_payload[key] = {
            "floor": float(params[0]),
            "scale": float(params[1]),
            "exponent": float(params[2]),
            "r2": r2,
            "standard_errors": np.sqrt(np.diag(covariance)).tolist(),
        }
    title = plot.title or "Final-checkpoint one-dimensional power laws"
    title = f"{title}\nL(x) = E + A x^(-α); final {plot.metric}, EMA {plot.ema}"
    fig.suptitle(title, fontsize=15, y=0.985)
    fig.subplots_adjust(
        left=0.075, right=0.985, top=0.85, bottom=0.10, hspace=0.10, wspace=0.16
    )
    _save(fig, root, stem)
    (root / f"{stem}.json").write_text(json.dumps(fit_payload, indent=2) + "\n")
    return Artifact(title, stem)


def _chinchilla_law(
    plot: ChinchillaScalingLaw,
    report: Report,
    series: Sequence[RunSeries],
    root: Path,
    stem: str,
) -> Artifact:
    if plot.parameter not in {"total_params", "active_params"}:
        raise ValueError(
            "ChinchillaScalingLaw.parameter must be 'total_params' or 'active_params'"
        )
    if plot.samples_per_model < 2:
        raise ValueError("samples_per_model must be at least 2")

    metadata = {model.name: model for model in report.selected_models()}
    rows: list[tuple[str, float, float, float]] = []
    for item in series:
        parameter = _law_x(plot.parameter, item, metadata.get(item.model))
        if parameter is None:
            continue
        tokens = numeric_series(item.frame, "tokens") / 1e9
        losses = numeric_series(item.frame, plot.metric)
        valid = tokens.notna() & (tokens >= plot.min_tokens) & losses.notna()
        if not valid.any():
            continue
        points = pd.DataFrame(
            {
                "tokens": tokens.loc[valid].to_numpy(),
                "loss": ema(losses.loc[valid].tolist(), plot.ema),
            }
        ).sort_values("tokens")
        if len(points) > plot.samples_per_model:
            indices = np.linspace(0, len(points) - 1, plot.samples_per_model, dtype=int)
            points = points.iloc[np.unique(indices)]
        rows.extend(
            (item.model, parameter, float(token), float(loss))
            for token, loss in points.itertuples(index=False, name=None)
        )

    if len({row[1] for row in rows}) < 3 or len(rows) <= 5:
        raise ValueError(
            "Chinchilla scaling laws require at least three model sizes and six "
            "valid trajectory points"
        )

    parameters = np.asarray([row[1] for row in rows], dtype=float)
    tokens = np.asarray([row[2] for row in rows], dtype=float)
    losses = np.asarray([row[3] for row in rows], dtype=float)
    floor_upper = max(1e-8, float(losses.min()) - 1e-6)
    amplitude = max(0.01, float(np.ptp(losses)))
    fit, covariance = curve_fit(
        _chinchilla,
        (parameters, tokens),
        losses,
        p0=(floor_upper * 0.9, amplitude / 2, 0.3, amplitude / 2, 0.3),
        bounds=(
            [0.0, 0.0, 0.001, 0.0, 0.001],
            [floor_upper, np.inf, 5.0, np.inf, 5.0],
        ),
        maxfev=200_000,
    )
    predicted = _chinchilla((parameters, tokens), *fit)
    residual = losses - predicted
    denominator = float(np.sum((losses - losses.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / denominator
    warnings = []
    if fit[2] <= 0.0011:
        warnings.append(
            "alpha reached its lower bound; the model-size term is not identified"
        )
    elif fit[2] >= 4.999:
        warnings.append(
            "alpha reached its upper bound; the model-size term is not identified"
        )
    if fit[4] <= 0.0011:
        warnings.append(
            "beta reached its lower bound; the token term is not identified"
        )
    elif fit[4] >= 4.999:
        warnings.append(
            "beta reached its upper bound; the token term is not identified"
        )

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4))
    palette = model_colors([item.model for item in series])
    for model in dict.fromkeys(row[0] for row in rows):
        selected = np.asarray([row[0] == model for row in rows])
        model_tokens = tokens[selected]
        model_losses = losses[selected]
        model_parameter = parameters[selected][0]
        order = np.argsort(model_tokens)
        grid = np.geomspace(model_tokens.min(), model_tokens.max(), 200)
        axes[0].scatter(
            model_tokens,
            model_losses,
            s=18,
            alpha=0.45,
            color=palette[model],
        )
        axes[0].plot(
            grid,
            _chinchilla((np.full_like(grid, model_parameter), grid), *fit),
            color=palette[model],
            linewidth=2.0,
            label=model,
        )
        axes[1].scatter(
            predicted[selected][order],
            model_losses[order],
            s=27,
            alpha=0.7,
            color=palette[model],
            label=model,
        )

    axes[0].set_xscale("log")
    axes[0].set_xlabel("consumed tokens D (B)")
    axes[0].set_ylabel(f"{plot.metric} (EMA {plot.ema})")
    axes[0].set_title("Observed points and fitted curves")
    low = min(float(losses.min()), float(predicted.min()))
    high = max(float(losses.max()), float(predicted.max()))
    axes[1].plot([low, high], [low, high], color="#202428", linewidth=1.4)
    axes[1].set_xlabel("fitted loss")
    axes[1].set_ylabel("observed loss")
    axes[1].set_title("Observed versus fitted")
    annotation = (
        f"E={fit[0]:.4f}\nA={fit[1]:.4f}, α={fit[2]:.4f}\n"
        f"B={fit[3]:.4f}, β={fit[4]:.4f}\nR²={r2:.5f}"
    )
    if warnings:
        annotation += "\n\nWARNING: " + "\n".join(warnings)
    axes[1].text(
        0.96,
        0.04,
        annotation,
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={
            "boxstyle": "square,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#C6C9CB",
            "alpha": 0.92,
        },
    )
    for axis in axes:
        finish_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=min(6, len(labels)),
    )
    parameter_label = _law_label(plot.parameter)
    title = plot.title or "Chinchilla-style trajectory scaling law"
    title = (
        f"{title}\nL(N,D) = E + A N^(-α) + B D^(-β); "
        f"N={parameter_label}, D=consumed tokens (B)"
    )
    if plot.min_tokens:
        title += f"; fit uses D >= {plot.min_tokens:g}B"
    fig.suptitle(title, fontsize=12.5, y=0.99)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.78, bottom=0.12, wspace=0.18)
    _save(fig, root, stem)
    payload = {
        "formula": "L(N,D) = E + A * N^(-alpha) + B * D^(-beta)",
        "parameter": plot.parameter,
        "parameter_unit": "billions",
        "data": "consumed_tokens",
        "data_unit": "billions",
        "metric": plot.metric,
        "ema": plot.ema,
        "min_tokens_b": plot.min_tokens,
        "samples_per_model": plot.samples_per_model,
        "points": len(rows),
        "floor": float(fit[0]),
        "parameter_scale": float(fit[1]),
        "parameter_exponent": float(fit[2]),
        "data_scale": float(fit[3]),
        "data_exponent": float(fit[4]),
        "r2": r2,
        "warnings": warnings,
        "standard_errors": np.sqrt(np.diag(covariance)).tolist(),
    }
    (root / f"{stem}.json").write_text(json.dumps(payload, indent=2) + "\n")
    return Artifact(title, stem)


def _eval_macro(
    plot: EvalMacro, series: Sequence[RunSeries], root: Path, stem: str
) -> Artifact:
    groups = list(plot.task_groups)
    columns = min(3, max(1, len(groups)))
    rows = math.ceil(len(groups) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.2 * columns, 3.2 * rows + 0.9),
        squeeze=False,
        sharey=True,
    )
    palette = model_colors([item.model for item in series])
    for index, group in enumerate(groups):
        axis = axes.flat[index]
        values = [
            _macro_endpoint(
                item.frame,
                plot.task_groups[group],
                plot.chance_baselines,
                plot.chance_adjusted,
            )
            for item in series
        ]
        values = [value * plot.value_scale for value in values]
        x = np.arange(len(series))
        axis.plot(x, values, color="#767D82", linewidth=1.4, zorder=1)
        for model_index, (item, value) in enumerate(zip(series, values, strict=True)):
            axis.scatter(
                model_index,
                value,
                s=65,
                color=palette[item.model],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            axis.annotate(
                f"{value:.4f}",
                (model_index, value),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        finite = np.flatnonzero(np.isfinite(values))
        if len(finite):
            winner = int(finite[np.argmax(np.asarray(values)[finite])])
            axis.scatter(
                winner,
                values[winner],
                marker="*",
                s=145,
                color=palette[series[winner].model],
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
            )
        axis.set_title(
            f"{group}\n{_metric_summary(plot.task_groups[group])}", fontsize=8.5
        )
        axis.set_xticks(x, [item.model for item in series], rotation=30, ha="right")
        finish_axis(axis)
    for index in range(len(groups), rows * columns):
        axes.flat[index].set_visible(False)
    title = plot.title or "Endpoint evaluation macros"
    axes.flat[0].set_ylabel(_eval_label(plot.chance_adjusted, plot.value_scale))
    fig.suptitle(title, fontsize=15, y=0.985)
    fig.subplots_adjust(
        left=0.07,
        right=0.99,
        top=0.89,
        bottom=0.11,
        hspace=0.42,
        wspace=0.16,
    )
    _save(fig, root, stem)
    return Artifact(title, stem)


def _eval_trajectories(
    plot: EvalTrajectories, series: Sequence[RunSeries], root: Path, stem: str
) -> Artifact:
    groups = list(plot.task_groups)
    axes_names = list(plot.axes)
    fig, axes = plt.subplots(
        len(groups),
        len(axes_names),
        figsize=(6.4 * len(axes_names), 4.6 * len(groups) + 1.7),
        sharey=True,
        squeeze=False,
    )
    palette = model_colors([item.model for item in series])
    for row, group in enumerate(groups):
        for item in series:
            macro = _macro_trajectory(
                item.frame,
                plot.task_groups[group],
                plot.chance_baselines,
                plot.chance_adjusted,
            )
            for column, axis_name in enumerate(axes_names):
                x = numeric_series(macro, axis_name)
                y = pd.to_numeric(macro["macro"], errors="coerce") * plot.value_scale
                valid = x.notna() & y.notna()
                if axis_name == "flops":
                    valid &= x > 0
                axes[row, column].plot(
                    _scale_axis(x.loc[valid].to_numpy(), axis_name),
                    y.loc[valid],
                    color=palette[item.model],
                    linewidth=2.0,
                    marker="o",
                    markersize=3.6,
                    label=item.model,
                )
                if valid.any():
                    final_x = _scale_axis(x.loc[valid].to_numpy()[-1:], axis_name)[0]
                    final_y = float(y.loc[valid].iloc[-1])
                    axes[row, column].annotate(
                        f"{item.model} {final_y:.4f}",
                        (final_x, final_y),
                        xytext=(-4, 7),
                        textcoords="offset points",
                        ha="right",
                        color=palette[item.model],
                        fontsize=8,
                        fontweight="bold",
                    )
        for column, axis_name in enumerate(axes_names):
            axis = axes[row, column]
            axis.set_title(
                f"{group} by {_axis_label(axis_name)}\n"
                f"{_metric_summary(plot.task_groups[group])}",
                fontsize=9,
            )
            axis.set_xlabel(_axis_label(axis_name))
            finish_axis(axis)
        axes[row, 0].set_ylabel(_eval_label(plot.chance_adjusted, plot.value_scale))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=min(6, len(labels)),
        frameon=False,
        handlelength=2.2,
    )
    title = plot.title or "Evaluation trajectories"
    fig.suptitle(title, fontsize=16, y=0.99)
    fig.text(
        0.5,
        0.018,
        "Each point combines only suites evaluated at that checkpoint. Higher is better; "
        "four-decimal labels show final values.",
        ha="center",
        fontsize=8.5,
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.985,
        top=0.80,
        bottom=0.08,
        hspace=0.32,
        wspace=0.10,
    )
    _save(fig, root, stem)
    return Artifact(title, stem)


def _task_heatmap(
    plot: TaskHeatmap, series: Sequence[RunSeries], root: Path, stem: str
) -> Artifact:
    tasks = [
        (group, metric)
        for group, metrics in plot.task_groups.items()
        for metric in metrics
    ]
    values = np.asarray(
        [
            [
                value
                if (value := endpoint(item.frame, metric, None)) is not None
                else np.nan
                for item in series
            ]
            for _, metric in tasks
        ],
        dtype=float,
    )
    scaled = values * plot.value_scale
    normalized = np.zeros_like(values)
    for row in range(len(tasks)):
        finite = np.isfinite(values[row])
        if not finite.any():
            normalized[row] = np.nan
            continue
        minimum, maximum = np.nanmin(values[row]), np.nanmax(values[row])
        normalized[row] = (
            (values[row] - minimum) / (maximum - minimum) if maximum > minimum else 0.5
        )
    title = plot.title or "Raw endpoint results per task"
    table = pd.DataFrame(
        scaled,
        index=[_task_label(group, metric) for group, metric in tasks],
        columns=[item.model for item in series],
    )
    table.to_csv(root / f"{stem}.csv")
    if plot.grouped:
        _grouped_task_heatmap(
            plot, series, values, scaled, normalized, title, root, stem
        )
        return Artifact(title, stem)

    fig_height = max(4.5, 0.42 * len(tasks) + 1.8)
    fig, axis = plt.subplots(figsize=(1.35 * len(series) + 4.7, fig_height))
    image = axis.imshow(normalized, cmap=HEATMAP_CMAP, aspect="auto", vmin=0, vmax=1)
    _annotate_heatmap(axis, values, scaled, normalized, plot)
    axis.set_xticks(range(len(series)), [item.model for item in series])
    axis.set_yticks(
        range(len(tasks)),
        [_task_label(group, metric) for group, metric in tasks],
    )
    axis.tick_params(length=0)
    axis.grid(False)
    axis.set_title(title, fontsize=15, pad=14)
    fig.colorbar(
        image, ax=axis, fraction=0.025, pad=0.02, label="within-task relative score"
    )
    fig.subplots_adjust(left=0.34, right=0.96, top=0.93, bottom=0.08)
    _save(fig, root, stem)
    return Artifact(title, stem)


def _grouped_task_heatmap(
    plot: TaskHeatmap,
    series: Sequence[RunSeries],
    values: np.ndarray,
    scaled: np.ndarray,
    normalized: np.ndarray,
    title: str,
    root: Path,
    stem: str,
) -> None:
    groups = [
        (group, metrics) for group, metrics in plot.task_groups.items() if metrics
    ]
    row_counts = [len(metrics) for _, metrics in groups]
    fig_height = max(4.0, 0.30 * sum(row_counts) + 0.30 * len(groups) + 1.2)
    fig, axes = plt.subplots(
        len(groups),
        1,
        figsize=(1.25 * len(series) + 4.2, fig_height),
        gridspec_kw={"height_ratios": row_counts},
        squeeze=False,
    )
    offset = 0
    for axis, (group, metrics) in zip(axes[:, 0], groups, strict=True):
        rows = slice(offset, offset + len(metrics))
        axis.imshow(
            normalized[rows],
            cmap=HEATMAP_CMAP,
            aspect="auto",
            vmin=0,
            vmax=1,
        )
        _annotate_heatmap(
            axis,
            values[rows],
            scaled[rows],
            normalized[rows],
            plot,
        )
        axis.set_xticks(range(len(series)), [item.model for item in series])
        axis.xaxis.tick_top()
        axis.tick_params(axis="x", length=0, pad=4)
        axis.set_yticks(
            range(len(metrics)),
            [_heatmap_metric_label(metric, plot) for metric in metrics],
        )
        axis.tick_params(axis="y", length=0, pad=7)
        axis.set_title(group, loc="left", fontsize=12, fontweight="bold", pad=22)
        axis.set_xticks(np.arange(-0.5, len(series), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(metrics), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.4)
        axis.grid(which="major", visible=False)
        axis.tick_params(which="minor", bottom=False, left=False)
        offset += len(metrics)

    fig.suptitle(title, fontsize=15, y=0.995)
    fig.text(
        0.5,
        0.012,
        "Cell labels are raw percentages. Color is normalized independently within "
        "each task row; darker means better within that row. Bold marks the row winner.",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(
        left=0.31,
        right=0.985,
        top=0.90,
        bottom=0.045,
        hspace=0.32,
    )
    _save(fig, root, stem)


def _annotate_heatmap(
    axis,
    values: np.ndarray,
    scaled: np.ndarray,
    normalized: np.ndarray,
    plot: TaskHeatmap,
) -> None:
    for row, row_values in enumerate(values):
        winner = _heatmap_winner(row_values, plot.higher_is_better)
        for column, value in enumerate(scaled[row]):
            axis.text(
                column,
                row,
                f"{value:.{plot.decimals}f}" if np.isfinite(value) else "—",
                ha="center",
                va="center",
                color="white" if normalized[row, column] > 0.58 else "#202428",
                fontsize=8.5,
                fontweight="bold" if column == winner else "normal",
            )


def _heatmap_winner(values: np.ndarray, higher_is_better: bool) -> int:
    if not np.isfinite(values).any():
        return -1
    return int(np.nanargmax(values) if higher_is_better else np.nanargmin(values))


def _aligned_values(
    frame: pd.DataFrame, loss: np.ndarray, axis_name: str
) -> tuple[np.ndarray, np.ndarray]:
    if axis_name == "lr_cooldown":
        lr = numeric_series(frame, "lr").to_numpy(dtype=float)
        finite = np.isfinite(lr)
        if not finite.any():
            return np.asarray([]), np.asarray([])
        peak = np.nanmax(lr)
        peak_index = int(np.where(lr == peak)[0][-1])
        return lr[peak_index:] / peak, loss[peak_index:]
    x = numeric_series(frame, axis_name)
    valid = x.notna()
    if axis_name == "flops":
        valid &= x > 0
    return _scale_axis(x.loc[valid].to_numpy(), axis_name), loss[valid.to_numpy()]


def _scale_axis(values: np.ndarray, name: str) -> np.ndarray:
    values = values.astype(float)
    if name == "tokens":
        return values / 1e9
    if name == "flops":
        return np.log10(values)
    return values


def _axis_label(name: str) -> str:
    return {
        "tokens": "consumed tokens (B)",
        "flops": "log10(cumulative FLOPs)",
        "step": "optimizer step",
    }.get(name, name)


def _winner(endpoints: Mapping[str, tuple[float, float]], better: str) -> str | None:
    if not endpoints:
        return None
    if better == "max":
        return max(endpoints, key=lambda model: endpoints[model][1])
    if better == "abs_zero":
        return min(endpoints, key=lambda model: abs(endpoints[model][1]))
    if better != "min":
        raise ValueError(f"Unknown winner direction: {better}")
    return min(endpoints, key=lambda model: endpoints[model][1])


def _law_x(key: str, item: RunSeries, metadata) -> float | None:
    if key == "tokens":
        values = numeric_series(item.frame, "tokens").dropna()
        values = values[values > 0]
        return float(values.iloc[-1]) / 1e9 if not values.empty else None
    if key == "flops":
        values = numeric_series(item.frame, "flops").dropna()
        values = values[values > 0]
        return float(values.iloc[-1]) / 1e21 if not values.empty else None
    if metadata is None:
        return None
    if key == "total_params":
        return metadata.total_params / 1e9 if metadata.total_params else None
    if key == "active_params":
        return metadata.active_params / 1e9 if metadata.active_params else None
    value = metadata.config.get(key)
    return float(value) if value is not None else None


def _power_law(x, floor, scale, exponent):
    return floor + scale * np.power(x, -exponent)


def _chinchilla(inputs, floor, parameter_scale, alpha, data_scale, beta):
    parameters, tokens = inputs
    return (
        floor
        + parameter_scale * np.power(parameters, -alpha)
        + data_scale * np.power(tokens, -beta)
    )


def _law_title(key: str) -> str:
    return {
        "total_params": "Total-parameter endpoint law",
        "active_params": "Active-parameter endpoint law",
        "tokens": "Consumed-token endpoint law",
        "flops": "Compute endpoint law",
    }.get(key, f"{key} endpoint law")


def _law_label(key: str) -> str:
    return {
        "total_params": "total parameters (B)",
        "active_params": "active parameters (B)",
        "tokens": "consumed tokens (B)",
        "flops": "cumulative training FLOPs (1e21)",
    }.get(key, key)


def _macro_endpoint(
    frame: pd.DataFrame,
    metrics: Sequence[str],
    baselines: Mapping[str, float],
    adjusted: bool,
) -> float:
    values = []
    for metric in metrics:
        value = endpoint(frame, metric, None)
        if value is not None:
            values.append(_adjust(value, baselines.get(metric)) if adjusted else value)
    return float(np.mean(values)) if values else float("nan")


def _macro_trajectory(
    frame: pd.DataFrame,
    metrics: Sequence[str],
    baselines: Mapping[str, float],
    adjusted: bool,
) -> pd.DataFrame:
    result = frame.copy()
    columns = []
    for metric in metrics:
        values = numeric_series(result, metric).ffill()
        if adjusted:
            values = values.map(
                partial(_adjust, baseline=baselines.get(metric)),
                na_action="ignore",
            )
        name = f"_macro_{len(columns)}"
        result[name] = values
        columns.append(name)
    result["macro"] = result[columns].mean(axis=1, skipna=False)
    return result


def _adjust(value: float, baseline: float | None) -> float:
    if baseline is None:
        return value
    if not 0.0 <= baseline < 1.0 or not 0.0 <= value <= 1.0:
        raise ValueError(
            "Chance-adjusted metrics and baselines must use the [0, 1] scale"
        )
    return (value - baseline) / (1.0 - baseline)


def _eval_label(adjusted: bool, scale: float) -> str:
    suffix = " (%)" if scale == 100.0 else ""
    return ("chance-adjusted score" if adjusted else "raw score") + suffix


def _metric_list(metrics: Sequence[str], group: str | None = None) -> str:
    return textwrap.fill(
        ", ".join(_metric_label(metric, group) for metric in metrics), width=58
    )


def _metric_summary(metrics: Sequence[str]) -> str:
    """Compact benchmark names for plot subtitles; full keys remain in report JSON."""
    aliases = {
        "arc_challenge": "ARC Challenge",
        "arc_easy": "ARC Easy",
        "boolq": "BoolQ",
        "commonsense_qa": "CSQA",
        "copa": "COPA",
        "gsm8k": "GSM8K",
        "hellaswag": "HellaSwag",
        "humaneval": "HumanEval",
        "lambada_openai": "LAMBADA",
        "mbpp": "MBPP",
        "mmlu": "MMLU",
        "openbookqa": "OBQA",
        "pawsx": "PAWS-X",
        "piqa": "PIQA",
        "winogrande": "Winogrande",
        "xcopa": "XCOPA",
        "xnli": "XNLI",
    }
    names = []
    for metric in metrics:
        key = metric.partition("::")[2] or metric
        task = key.split("/", 1)[0].casefold().replace("-", "_")
        name = aliases.get(task, task.replace("_", " ").title())
        if name not in names:
            names.append(name)
    return textwrap.fill(
        f"{len(metrics)} metrics: " + ", ".join(names),
        width=72,
    )


def _annotate_spread_endpoints(axis, points, dpi: float) -> None:
    if not points:
        return
    display = sorted(
        (
            axis.transData.transform((x, y))[1],
            x,
            y,
            model,
            color,
        )
        for x, y, model, color in points
    )
    lower, upper = axis.bbox.ymin + 8, axis.bbox.ymax - 8
    targets = [value[0] for value in display]
    gap = 13.0
    for index in range(1, len(targets)):
        targets[index] = max(targets[index], targets[index - 1] + gap)
    overflow = targets[-1] - upper
    if overflow > 0:
        targets = [target - overflow for target in targets]
    if targets[0] < lower:
        shift = lower - targets[0]
        targets = [target + shift for target in targets]
    midpoint = sum(axis.get_xlim()) / 2
    for target, (original, x, y, model, color) in zip(targets, display, strict=True):
        right_side = x > midpoint
        axis.annotate(
            f"{model} {y:.4f}",
            (x, y),
            xytext=(-5 if right_side else 5, (target - original) * 72.0 / dpi),
            textcoords="offset points",
            ha="right" if right_side else "left",
            va="center",
            fontsize=8,
            color=color,
        )


def _metric_label(metric: str, group: str | None = None) -> str:
    namespace, separator, key = metric.partition("::")
    if not separator:
        return metric
    if group and namespace.casefold() == group.casefold():
        return key
    return f"{namespace} — {key}"


def _task_label(group: str, metric: str) -> str:
    return f"{group} — {_metric_label(metric, group)}"


def _heatmap_metric_label(metric: str, plot: TaskHeatmap) -> str:
    if metric in plot.metric_labels:
        return plot.metric_labels[metric]
    namespace, separator, key = metric.partition("::")
    if not separator:
        return metric
    task, _, score = key.partition("/")
    task_name = plot.task_labels.get(
        task, TASK_LABELS.get(task, task.replace("_", " ").title())
    )
    if namespace in plot.namespace_labels:
        detail = plot.namespace_labels[namespace]
    else:
        shot = re.fullmatch(r"(\d+)shot", namespace)
        if shot:
            detail = f"{shot.group(1)}-shot"
        elif namespace == "multilingual":
            detail = "0-shot"
        elif namespace == "math":
            detail = "5-shot"
        elif task == "humaneval":
            detail = score.split(",", 1)[0]
        elif task == "mbpp":
            detail = score.replace("pass_at_", "pass@")
        else:
            detail = namespace.replace("_", " ")
    return f"{task_name} - {detail}"


def _max_metric_lines(task_groups: Mapping[str, Sequence[str]]) -> int:
    return max(
        (
            _metric_list(metrics, group).count("\n") + 1
            for group, metrics in task_groups.items()
        ),
        default=1,
    )


def _save(fig, root: Path, stem: str) -> None:
    fig.savefig(root / f"{stem}.svg", format="svg", bbox_inches="tight")
    fig.savefig(root / f"{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _slug(value: str) -> str:
    value = re.sub(r"(?<!^)(?=[A-Z])", "-", value)
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
