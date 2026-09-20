"""
NemoRLExperiment — one NeMo-RL RL run (GRPO/SFT) on a native Megatron checkpoint.

Mirrors MegatronExperiment, but the unit of work is a NeMo-RL Hydra recipe plus a
Ray lifecycle rather than a Megatron training script.

Field defaults encode what was validated against the Apertus2 KDA scaling-ladder
checkpoints on 2026-09-11/12 (see
agents-scratchpad/debugging-development/apertus2-nemorl-native-megatron-20260911T142606Z/HANDOFF.md):

* weights load from a native Megatron DCP via ``pretrained_checkpoint.format=megatron_lm``;
  only ``config.json`` + ``configuration_apertus2.py`` are needed on the HF side,
  no weights and no ``modeling_*.py`` (AutoBridge dispatches on ``architectures``).
* rollouts use the Megatron generation backend, so no vLLM and no weight refit.
* base (non-instruction-tuned) checkpoints need the PASSTHROUGH chat template;
  chat markers push them into meta-commentary.
* ``top_p`` must be < 1.0 — ``1.0`` disables nucleus truncation and admits tail garbage.
* md_decoupling requires ``use_layer_wise_distributed_optimizer`` and must NOT use
  the standard distributed optimizer or ``overlap_param_gather``.
* resuming a trained checkpoint requires ``hypersphere_preserve_init`` so the row
  gains absorb weight magnitude instead of the weights being reprojected.
"""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar

from spellbook.core.experiment import Experiment

# Concatenate message contents, no control tokens. Correct for base checkpoints.
PASSTHROUGH_CHAT_TEMPLATE = "{% for message in messages %}{{ message['content'] }}{% endfor %}"


@dataclasses.dataclass
class NemoRLExperiment(Experiment):
    """A single NeMo-RL run."""

    _lock_exclude: ClassVar[frozenset[str]] = frozenset({"wandb_exp_name"})

    # ---- repos / runtime ----
    nemo_rl_path: str = ""
    bridge_src_path: str = ""  # must point at <Megatron-Bridge>/src, not the repo root
    megatron_path: str = ""
    # Isolated `pip install --target` overlays prepended to PYTHONPATH, for a
    # container that lacks a dependency. Leave empty with the nemo-rl image
    # (containers/nemo-rl), which already carries ray and flashinfer — the latter
    # is mandatory, since NeMo-RL hardcodes sampling_backend="flashinfer".
    overlay_paths: tuple[str, ...] = ()
    # NeMo-RL otherwise builds a uv venv per worker; use the container interpreter.
    py_executables_system: bool = True
    # Raw shell run inside the container before the Ray head starts, e.g. to try a
    # package the image does not carry yet. Mirrors MegatronExperiment.
    install_commands: str = ""

    # ---- algorithm ----
    # Selects the entrypoint (examples/run_<algorithm>.py) and the top-level recipe
    # key. The backend only emits the rollout counts and policy.generation for
    # algorithms that generate (SlurmNemoRLBackend.GENERATION_ALGORITHMS), so
    # sft/dpo/rm render correctly too; anything specific to those blocks
    # (preference_loss_weight, val_batches, ...) comes from base_config or
    # recipe_overrides. On-policy distillation rides on grpo with an extra
    # top-level `on_policy_distillation` block.
    algorithm: str = "grpo"
    # Overrides the derived entrypoint. Relative paths resolve against nemo_rl_path.
    # NeMo-RL ships a dozen run_*.py variants (single_controller, sliding_puzzle,
    # vlm_*, xtoken_*) that the <algorithm> convention cannot name.
    entrypoint: str = ""
    # Recipe this run inherits from, rendered as NeMo-RL's `defaults` key: spellbook
    # emits only the delta and the parent supplies the rest. Leave empty to render a
    # standalone recipe, in which case everything NeMo-RL requires must come from the
    # fields below plus recipe_overrides.
    base_config: str = ""
    max_num_steps: int = 0
    num_prompts_per_step: int = 0  # must be a multiple of data-parallel size
    num_generations_per_prompt: int = 0

    # ---- model ----
    hf_config_dir: str = ""  # config.json + configuration_*.py only
    tokenizer_dir: str = ""
    pretrained_checkpoint_path: str = ""  # native Megatron DCP root
    pretrained_checkpoint_format: str = "megatron_lm"
    chat_template: str = ""  # PASSTHROUGH_CHAT_TEMPLATE for base checkpoints

    # ---- parallelism ----
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    context_parallel_size: int = 1

    # ---- generation ----
    generation_backend: str = "megatron"  # megatron => no vLLM, no refit, no param mapping
    temperature: float = 1.0
    top_p: float = 0.9  # never 1.0: that disables nucleus truncation
    top_k: int = 0  # 0 = disabled; 1 = greedy
    max_new_tokens: int = 0
    stop_strings: tuple[str, ...] = ()
    cuda_graph_impl: str = "none"
    num_cuda_graphs: int = 0

    # ---- optimizer ----
    optimizer: str = "adam"  # adam | md_decoupling | muon | dist_muon
    lr: float = 0.0
    min_lr: float = 0.0
    weight_decay: float = 0.0
    # Merged last over the fields above and passed through verbatim to Megatron's
    # OptimizerConfig, so no spellbook change is needed to reach a new knob. The
    # optimizer's own settings belong in the recipe this run inherits from.
    megatron_optimizer: dict[str, Any] = dataclasses.field(default_factory=dict)

    # ---- scheduler ----
    lr_warmup_iters: int = 0
    lr_warmup_init: float = 0.0
    lr_decay_iters: int = 0
    lr_decay_style: str = "linear"

    # ---- data ----
    dataset_name: str = ""
    data_path: str = ""  # for ResponseDataset
    validation_data_path: str = ""
    prompt_file: str = ""
    env_name: str = "math"
    max_input_seq_length: int = 0
    max_total_sequence_length: int = 0

    # ---- NeMo Gym ----
    # With gym_config_paths set, rollouts run through NeMo Gym: each configured server
    # is a FastAPI process in its own uv venv, reaching the policy over the endpoint
    # the Megatron backend exposes. Paths are relative to the Gym checkout.
    gym_home: str = ""  # <NeMo-RL>/3rdparty/Gym-workspace/Gym
    gym_config_paths: tuple[str, ...] = ()
    gym_venv_dir: str = ""  # persistent <type>/<name>/.venv root
    gym_uv_cache_dir: str = ""
    gym_port_range: tuple[int, int] = (5000, 5999)
    # Writable root placed ahead of the Gym checkout in NEMO_GYM_EXTRA_ROOTS, so the
    # datasets Gym downloads land here instead of inside the checkout.
    gym_scratch_root: str = ""
    # NeMo-RL's should_log_nemo_gym_responses: True means Gym owns response logging
    # and NeMo-RL SKIPS its per-step train_data_step*.jsonl dump, which leaves no
    # generations on disk when wandb is off.
    gym_skip_response_dump: bool = False
    gym_mask_flagged_samples: bool = True
    # Where NeMo-RL looks for per-actor venvs (NEMO_RL_VENV_DIR). Kept off the repo
    # so the venv tools/gym_data builds survives between runs.
    nemo_rl_venv_dir: str = ""

    # ---- checkpointing ----
    # Empty disables checkpointing; the requeue chain reads step_* from here to decide
    # whether the run is finished.
    checkpoint_dir: str = ""
    save_period: int = 50
    keep_top_k: int = 3

    # ---- logging ----
    log_dir: str = ""
    wandb_enabled: bool = False
    wandb_exp_name: str = ""
    tensorboard_enabled: bool = True

    # ---- escape hatch ----
    # Deep-merged into the rendered recipe last. Anything the fields above do not
    # model goes here rather than into the backend, so a new NeMo-RL or Megatron
    # knob never requires a spellbook change.
    recipe_overrides: dict[str, Any] = dataclasses.field(default_factory=dict)

    env_vars: dict[str, Any] = dataclasses.field(default_factory=dict)

    def validate(self) -> None:
        """Fail fast on combinations known to be broken."""
        if self.top_p >= 1.0 and self.top_k == 0:
            raise ValueError(
                "top_p=1.0 with top_k=0 disables all truncation and samples the full "
                "vocabulary tail; use top_p<1.0 (e.g. 0.9) or top_k=1 for greedy."
            )
        if self.generation_backend not in ("megatron", "vllm"):
            raise ValueError(
                f"unknown generation_backend {self.generation_backend!r}; "
                "use 'megatron' (no refit) or 'vllm'."
            )
        if self.gym_config_paths:
            if not self.gym_home:
                raise ValueError("gym_config_paths needs gym_home (the Gym checkout root).")
            missing = [
                name
                for name in (
                    "gym_venv_dir",
                    "gym_uv_cache_dir",
                    "gym_scratch_root",
                    "nemo_rl_venv_dir",
                )
                if not getattr(self, name)
            ]
            if missing:
                raise ValueError(f"gym_config_paths needs {', '.join(missing)}.")
        if (
            self.generation_backend == "megatron"
            and self.pretrained_checkpoint_format == "megatron_lm"
            and not self.hf_config_dir
        ):
            raise ValueError(
                "megatron_lm format still needs hf_config_dir: AutoBridge dispatches "
                "on config.json 'architectures'."
            )


__all__ = ["PASSTHROUGH_CHAT_TEMPLATE", "NemoRLExperiment"]
