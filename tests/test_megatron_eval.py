from __future__ import annotations

import unittest

from evals.megatron_eval import MegatronEvalConfig, _render


class MegatronEvalRenderTests(unittest.TestCase):
    def _config(
        self,
        *,
        lm_eval_install_with_python: bool = False,
    ) -> MegatronEvalConfig:
        return MegatronEvalConfig(
            model_name="test-model",
            checkpoint_dir="/checkpoints",
            tokenizer_model="test-tokenizer",
            megatron_path="/megatron",
            lm_eval_install="git+https://example.com/lm-evaluation-harness.git",
            lm_eval_install_with_python=lm_eval_install_with_python,
        )

    def test_lm_eval_install_uses_bare_pip_by_default(self) -> None:
        script = _render(self._config(), ckpt_step=10, dependency_singleton=False)

        self.assertIn(
            "pip install git+https://example.com/lm-evaluation-harness.git "
            "--no-build-isolation",
            script,
        )
        self.assertNotIn("python -m pip install", script)

    def test_lm_eval_install_can_use_eval_python(self) -> None:
        script = _render(
            self._config(lm_eval_install_with_python=True),
            ckpt_step=10,
            dependency_singleton=False,
        )

        self.assertIn(
            "python -m pip install "
            "git+https://example.com/lm-evaluation-harness.git --no-build-isolation",
            script,
        )


if __name__ == "__main__":
    unittest.main()
