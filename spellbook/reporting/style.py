"""Shared visual language for Spellbook reports."""

from __future__ import annotations

from collections.abc import Sequence

from matplotlib.colors import LinearSegmentedColormap

MODEL_COLORS: tuple[str, ...] = (
    "#4C78A8",
    "#E39C37",
    "#B24735",
    "#A05195",
    "#5B8E7D",
    "#2F3E46",
    "#76B7B2",
    "#F28E2B",
)

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "spellbook_scores",
    ("#ECEDE9", "#E39C37", "#4C78A8", "#2F3E46"),
)


def model_colors(models: Sequence[str]) -> dict[str, str]:
    return {
        model: MODEL_COLORS[index % len(MODEL_COLORS)]
        for index, model in enumerate(models)
    }


def apply_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.facecolor": "#F2F2EF",
            "figure.facecolor": "white",
            "axes.edgecolor": "#8A8F93",
            "axes.grid": True,
            "grid.color": "white",
            "grid.linewidth": 1.0,
            "axes.axisbelow": True,
            "legend.framealpha": 0.92,
            "svg.fonttype": "none",
        }
    )


def finish_axis(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
