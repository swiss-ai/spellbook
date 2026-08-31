import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from matplotlib.axes import Axes

from spellbook.reporting.history import (
    RunSeries,
    _load_run_history,
    _match_model,
    _namespace_metrics,
    required_history_keys,
    stitch_frames,
)
from spellbook.reporting.render import _adjust, _macro_trajectory, render_plots
from spellbook.reporting.specs import (
    ChinchillaScalingLaw,
    EndpointScalingLaw,
    EvalMacro,
    EvalTrajectories,
    LearningRateBowl,
    LossAlignment,
    MetricCurves,
    ModelMetadata,
    Models,
    Report,
    TaskHeatmap,
    WandbGroup,
)
from spellbook.reporting.style import apply_style


def _series(model: str, offset: float) -> RunSeries:
    return RunSeries(
        model=model,
        run_ids=[model],
        run_names=[model],
        config={"matrix_lr": 0.01 if model == "small" else 0.02},
        frame=pd.DataFrame(
            {
                "_step": [0, 1, 2, 3],
                "consumed-tokens": [1e9, 2e9, 3e9, 4e9],
                "flops": [1e18, 2e18, 3e18, 4e18],
                "learning-rate": [1e-3, 1e-3, 5e-4, 0.0],
                "lm loss": [3.0, 2.6, 2.3, 2.1 + offset],
                "global max violation": [0.8, 0.4, 0.2, offset],
                "arc": [0.0, 0.1, 0.2, 0.3 + offset],
                "0shot::arc_challenge/acc": [0.0, 0.1, 0.2, 0.3 + offset],
            }
        ),
    )


class ReportingRenderTest(unittest.TestCase):
    def test_loss_alignment_applies_open_y_and_per_axis_x_ranges(self) -> None:
        report = Report(
            "ranges",
            WandbGroup("entity/project"),
            [
                LossAlignment(
                    axes=["tokens", "flops"],
                    xlim={"tokens": (2.0, 3.5)},
                    ylim=(None, 2.5),
                )
            ],
        )
        x_calls = []
        y_calls = []
        set_xlim = Axes.set_xlim
        set_ylim = Axes.set_ylim

        def record_xlim(axis, *args, **kwargs):
            x_calls.append(args)
            return set_xlim(axis, *args, **kwargs)

        def record_ylim(axis, *args, **kwargs):
            y_calls.append(args)
            return set_ylim(axis, *args, **kwargs)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(Axes, "set_xlim", record_xlim),
            patch.object(Axes, "set_ylim", record_ylim),
        ):
            apply_style()
            render_plots(report, [_series("small", 0.0)], Path(directory))

        self.assertIn((2.0, 3.5), x_calls)
        self.assertGreaterEqual(y_calls.count((None, 2.5)), 2)

    def test_common_figures_render_svg_and_png(self) -> None:
        report = Report(
            name="example",
            source=WandbGroup("entity/project"),
            models=[ModelMetadata("small"), ModelMetadata("large")],
            plots=[
                LossAlignment(
                    axes=["tokens", "flops", "lr_cooldown"],
                    xlim={"tokens": (1.0, 4.0)},
                    ylim=(None, 3.0),
                ),
                LearningRateBowl(),
                MetricCurves(
                    metrics=["global max violation"],
                    better={"global max violation": "abs_zero"},
                ),
                EvalMacro(task_groups={"reasoning": ["arc"]}),
                EvalTrajectories(task_groups={"reasoning": ["arc"]}),
                TaskHeatmap(
                    task_groups={
                        "reasoning": ["0shot::arc_challenge/acc"],
                        "empty": ["missing"],
                    },
                    grouped=True,
                    task_labels={"arc_challenge": "ARC-C"},
                ),
            ],
        )
        series = [_series("small", 0.08), _series("large", -0.02)]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            apply_style()
            artifacts = render_plots(report, series, root)

            self.assertEqual(len(artifacts), 6)
            for artifact in artifacts:
                self.assertTrue((root / f"{artifact.stem}.svg").exists())
                self.assertTrue((root / f"{artifact.stem}.png").exists())
            eval_svg = (root / "04-eval-macro.svg").read_text()
            self.assertIn("reasoning", eval_svg)
            self.assertIn("1 metrics: Arc", eval_svg)
            trajectory_svg = (root / "05-eval-trajectories.svg").read_text()
            self.assertIn("reasoning by consumed tokens (B)", trajectory_svg)
            self.assertIn("1 metrics: Arc", trajectory_svg)
            heatmap = pd.read_csv(root / "06-task-heatmap.csv", index_col=0)
            self.assertAlmostEqual(
                heatmap.loc["reasoning — 0shot — arc_challenge/acc", "small"],
                38.0,
            )
            heatmap_svg = (root / "06-task-heatmap.svg").read_text()
            self.assertIn("reasoning", heatmap_svg)
            self.assertIn("empty", heatmap_svg)
            self.assertIn("ARC-C - 0-shot", heatmap_svg)
            self.assertGreaterEqual(heatmap_svg.count("small"), 2)

    def test_continuation_overrides_overlapping_history(self) -> None:
        first = pd.DataFrame(
            {
                "consumed-tokens": [1, 2, 3],
                "lm loss": [3.0, 2.8, 2.7],
                "arc": [0.1, 0.2, 0.3],
                "_spellbook_piece": 0,
            }
        )
        resumed = pd.DataFrame(
            {
                "consumed_tokens": [3, 4],
                "lm loss": [2.6, 2.5],
                "boolq": [0.6, 0.7],
                "_spellbook_piece": 1,
            }
        )

        stitched = stitch_frames([first, resumed])

        self.assertEqual(stitched["consumed-tokens"].tolist(), [1, 2, 3, 4])
        self.assertEqual(stitched["lm loss"].tolist(), [3.0, 2.8, 2.6, 2.5])
        self.assertAlmostEqual(stitched.loc[2, "arc"], 0.3)
        self.assertAlmostEqual(stitched.loc[2, "boolq"], 0.6)

    def test_eval_suites_merge_on_checkpoint_and_keep_newest_coordinates(self) -> None:
        older = pd.DataFrame(
            {
                "OptimizerStep": [2_000],
                "consumed-tokens": [6.3e9],
                "0shot::arc/acc": [0.4],
                "_spellbook_piece": [0],
            }
        )
        newer = pd.DataFrame(
            {
                "OptimizerStep": [2_000],
                "consumed-tokens": [4.2e9],
                "10shot::arc/acc": [0.5],
                "_spellbook_piece": [1],
            }
        )

        stitched = stitch_frames(
            [older, newer],
            checkpoint_identity=True,
        )

        self.assertEqual(len(stitched), 1)
        self.assertEqual(stitched.loc[0, "consumed-tokens"], 4.2e9)
        self.assertEqual(stitched.loc[0, "0shot::arc/acc"], 0.4)
        self.assertEqual(stitched.loc[0, "10shot::arc/acc"], 0.5)

    def test_eval_model_alias_does_not_match_inside_decimal_name(self) -> None:
        report = Report(
            "evals",
            WandbGroup("entity/evals"),
            [],
            select=Models(names=["5b"]),
        )
        run = SimpleNamespace(
            id="run-id",
            name="scaling-ladder-kda-1.5b-0shot",
            config={},
        )

        self.assertIsNone(_match_model(run, report, []))

    def test_run_filters_apply_when_parameter_metadata_is_present(self) -> None:
        report = Report(
            "training",
            WandbGroup("entity/project"),
            [],
            select=Models(exclude=["*router*"]),
            models=[ModelMetadata("1.5b", active_params=0.4e9)],
        )
        run = SimpleNamespace(
            id="run-id",
            name="scaling-ladder-1.5b-router-ablation",
            config={},
        )

        self.assertIsNone(_match_model(run, report, report.selected_models()))

    def test_eval_run_namespaces_keep_shot_metrics_separate(self) -> None:
        report = Report(
            "evals",
            WandbGroup(
                "entity/evals",
                run_namespaces={
                    "0shot": ["*-0shot"],
                    "10shot": ["*-10shot"],
                },
            ),
            [
                EvalMacro(
                    task_groups={
                        "reasoning": [
                            "0shot::arc_easy/acc_norm",
                            "10shot::arc_easy/acc_norm",
                        ]
                    }
                )
            ],
        )

        keys = required_history_keys(report)
        self.assertIn("arc_easy/acc_norm", keys)
        self.assertNotIn("0shot::arc_easy/acc_norm", keys)
        frame = _namespace_metrics(
            pd.DataFrame({"ConsumedTokens": [1], "arc_easy/acc_norm": [0.5]}),
            "0shot",
        )
        self.assertIn("0shot::arc_easy/acc_norm", frame)
        self.assertIn("ConsumedTokens", frame)

    def test_history_fallback_uses_only_keys_present_in_run(self) -> None:
        class Run:
            def __init__(self):
                self.id = "run-id"
                self.name = "run"
                self.summary = {"_step": 2, "lm loss": 2.5}

            def scan_history(self, *, keys, page_size):
                self.scanned_keys = keys
                return []

            def history(self, *, samples, keys, pandas):
                return [{"_step": 2, "lm loss": 2.5}]

        run = Run()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            (cache / "run-id.json").write_text(
                json.dumps(
                    {
                        "keys": ["_step", "consumed_tokens", "lm loss"],
                        "rows": [],
                    }
                )
            )
            frame = _load_run_history(
                run,
                ["_step", "consumed_tokens", "lm loss"],
                cache,
                refresh=False,
            )

        self.assertEqual(
            run.scanned_keys,
            ["_step", "consumed_tokens", "lm loss"],
        )
        self.assertEqual(frame["lm loss"].tolist(), [2.5])

    def test_chance_adjustment_requires_fraction_units(self) -> None:
        self.assertAlmostEqual(_adjust(0.5, 0.25), 1.0 / 3.0)
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            _adjust(50.0, 0.25)

    def test_chance_adjusted_trajectory_preserves_leading_missing_values(self) -> None:
        trajectory = _macro_trajectory(
            pd.DataFrame({"task/acc": [float("nan"), 0.5]}),
            ["task/acc"],
            {"task/acc": 0.25},
            True,
        )

        self.assertTrue(pd.isna(trajectory.loc[0, "macro"]))
        self.assertAlmostEqual(trajectory.loc[1, "macro"], 1.0 / 3.0)

    def test_endpoint_and_chinchilla_scaling_laws_render(self) -> None:
        models = []
        series = []
        for index, active_billions in enumerate([0.4, 0.6, 0.8, 1.1, 1.5, 2.0]):
            name = f"model-{index}"
            total_billions = active_billions * 8
            final_tokens = 40.0 + 30.0 * index
            tokens = pd.Series([final_tokens * point / 16 for point in range(1, 17)])
            loss = 1.2 + 0.35 * active_billions**-0.45 + 0.5 * tokens**-0.3
            models.append(
                ModelMetadata(
                    name,
                    total_params=total_billions * 1e9,
                    active_params=active_billions * 1e9,
                )
            )
            series.append(
                RunSeries(
                    name,
                    [name],
                    [name],
                    {},
                    pd.DataFrame(
                        {
                            "consumed-tokens": tokens * 1e9,
                            "flops": tokens * active_billions * 6e18,
                            "lm loss": loss,
                        }
                    ),
                )
            )
        report = Report(
            "laws",
            WandbGroup("entity/project"),
            [
                EndpointScalingLaw(
                    x=["total_params", "active_params", "tokens", "flops"]
                ),
                ChinchillaScalingLaw(samples_per_model=16),
            ],
            models=models,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            apply_style()
            artifacts = render_plots(report, series, root)

            self.assertEqual(len(artifacts), 2)
            endpoint_fit = json.loads(
                (root / "01-endpoint-scaling-law.json").read_text()
            )
            self.assertIn("tokens", endpoint_fit)
            chinchilla_fit = json.loads(
                (root / "02-chinchilla-scaling-law.json").read_text()
            )
            self.assertEqual(chinchilla_fit["parameter"], "active_params")
            self.assertEqual(chinchilla_fit["points"], 96)
            self.assertGreater(chinchilla_fit["r2"], 0.99)


if __name__ == "__main__":
    unittest.main()
