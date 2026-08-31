import dataclasses
import tempfile
import textwrap
import unittest
import warnings
from pathlib import Path

from spellbook.megatron import MegatronExperiment
from spellbook.reporting import (
    Between,
    CombinedReport,
    EvalReport,
    LossAlignment,
    ModelMetadata,
    Models,
    Report,
    WandbGroup,
    load_report,
)


def _experiment(name: str, hidden_size: int = 768) -> MegatronExperiment:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MegatronExperiment(
            name=name,
            hidden_size=hidden_size,
            ffn_hidden_size=1920,
            num_layers=10,
            num_attention_heads=12,
            vocab_size=131072,
            moe_layer_freq=[0] + [1] * 9,
            num_experts=128,
            moe_router_topk=4,
            moe_ffn_hidden_size=448,
        )


class ReportingTest(unittest.TestCase):
    def test_model_selection_supports_names_ranges_and_exclusion(self) -> None:
        models = [
            {"name": "1.5b", "total_params": 1.5e9, "mode": "flat"},
            {"name": "5b", "total_params": 5e9, "mode": "row_fan_in"},
            {"name": "router-5b", "total_params": 5e9, "mode": "row_fan_in"},
        ]
        selector = Models(
            where={"total_params": Between(1e9, 6e9)},
            exclude=["router-*"],
        )
        self.assertEqual([model["name"] for model in selector.select(models)], ["1.5b", "5b"])

    def test_missing_explicit_model_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "22b"):
            Models(names=["1.5b", "22b"]).select([{"name": "1.5b"}])

    def test_report_infers_selected_parameter_metadata_from_sweep(self) -> None:
        report = Report(
            name="training",
            source=WandbGroup("entity/project"),
            select=Models(names=["large"]),
            plots=[LossAlignment(axes=["tokens", "flops"])],
        ).with_experiments([_experiment("small"), _experiment("large", 1024)])

        payload = report.to_dict()
        self.assertEqual([model["name"] for model in payload["models"]], ["large"])
        self.assertGreater(payload["models"][0]["total_params"], 0)
        self.assertIn("parameter_counts", payload["models"][0]["parameter_source"])

    def test_explicit_parameter_counter_is_recorded(self) -> None:
        def counter(_experiment):
            return {"total_B": 2.0, "active_B": 0.5, "ratio": 0.25}

        report = Report(
            name="custom",
            source=WandbGroup("entity/project"),
            plots=[],
            parameter_counter=counter,
        ).with_experiments([_experiment("model")])
        payload = report.to_dict()
        self.assertEqual(payload["models"][0]["total_params"], 2e9)
        self.assertTrue(payload["parameter_counter"].endswith("counter"))

    def test_combined_report_rejects_mismatched_models(self) -> None:
        training = Report(
            "training", WandbGroup("entity/train"), [],
            models=[ModelMetadata("1.5b")],
        )
        evaluations = EvalReport(
            "evals", WandbGroup("entity/evals"), [],
            models=[ModelMetadata("3b")],
        )
        with self.assertRaisesRegex(ValueError, "model mismatch"):
            CombinedReport("combined", training, evaluations).to_dict()

    def test_loader_attaches_sweep_experiments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            definition = Path(directory) / "definition.py"
            definition.write_text(textwrap.dedent("""
                from spellbook.core import Sweep
                from spellbook.megatron import MegatronExperiment
                from spellbook.reporting import LossAlignment, Models, Report, WandbGroup

                experiments = [
                    MegatronExperiment(
                        name="small", hidden_size=64, ffn_hidden_size=128,
                        num_layers=2, num_attention_heads=4, vocab_size=256,
                        moe_layer_freq=[0, 0],
                    ),
                    MegatronExperiment(
                        name="large", hidden_size=128, ffn_hidden_size=256,
                        num_layers=2, num_attention_heads=4, vocab_size=256,
                        moe_layer_freq=[0, 0],
                    ),
                ]
                sweep = Sweep("models", experiments, backend=None)
                report = Report(
                    "training",
                    WandbGroup("entity/project"),
                    [LossAlignment()],
                    select=Models(names=["large"]),
                )
            """))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                report = load_report(definition)
            self.assertIsInstance(report, Report)
            assert isinstance(report, Report)
            self.assertEqual([model.name for model in report.models], ["large"])


@dataclasses.dataclass
class KDAExperiment(MegatronExperiment):
    experimental_attention_variant: str = "kda"
    linear_key_head_dim: int = 16
    linear_value_head_dim: int = 16
    linear_num_key_heads: int = 4
    linear_num_value_heads: int = 4
    linear_attention_full_rank_output_gate: bool = False
    linear_conv_kernel_dim: int = 4
    linear_attention_freq: list[int] = dataclasses.field(
        default_factory=lambda: [1] * 9 + [0]
    )
    moe_latent_size: int | None = 384


class KDAParameterCountTest(unittest.TestCase):
    def test_kda_and_latent_experts_are_counted(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kda = KDAExperiment(
                name="kda",
                hidden_size=768,
                ffn_hidden_size=1920,
                num_layers=10,
                num_attention_heads=12,
                num_query_groups=12,
                vocab_size=131072,
                moe_layer_freq=[0] + [1] * 9,
                num_experts=256,
                moe_router_topk=8,
                moe_ffn_hidden_size=448,
            )
            standard = dataclasses.replace(kda, experimental_attention_variant="")

        self.assertNotEqual(kda.parameter_counts(), standard.parameter_counts())
        self.assertGreater(kda.parameter_counts()["total_B"], kda.parameter_counts()["active_B"])
