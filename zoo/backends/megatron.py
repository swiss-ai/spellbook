"""
MegatronBackend: translates a resolved config dict → list of Megatron CLI arg strings.

Two input paths:
  1. Short aliases (tp, ep, mbs, …) or full snake_case aliases → mapped via _ALIAS_MAP
  2. Raw Megatron flags (--tensor-model-parallel-size, …) → passed through directly

If a Megatron flag name changes in a new version, pass it directly via model_args
or extra_flags without touching the alias table.
"""

from __future__ import annotations

from typing import Any


def _flag_to_alias(flag: str) -> str:
    """Convert --some-flag-name → some_flag_name (used to build _ALIAS_MAP)."""
    return flag.lstrip("-").replace("-", "_")


# ---------------------------------------------------------------------------
# All known Megatron flags, extracted from megatron/training/arguments.py.
# The alias is the snake_case name (--flag-name → flag_name).
# Short convenience aliases are added below for the most common params.
# ---------------------------------------------------------------------------

_ALL_FLAGS: list[str] = [
    "--account-for-embedding-in-pipeline-split",
    "--account-for-loss-in-pipeline-split",
    "--accumulate-allreduce-grads-in-fp32",
    "--activation-func-clamp-value",
    "--adam-beta1",
    "--adam-beta2",
    "--adam-eps",
    "--add-qkv-bias",
    "--adlr-autoresume",
    "--adlr-autoresume-interval",
    "--allow-ambiguous-pad-tokens",
    "--apply-layernorm-1p",
    "--apply-query-key-layer-scaling",
    "--apply-residual-connection-post-layernorm",
    "--app-tag-run-name",
    "--app-tag-run-version",
    "--async-save",
    "--attention-backend",
    "--attention-dropout",
    "--attention-softmax-in-fp32",
    "--auto-detect-ckpt-format",
    "--batch-invariant-mode",
    "--batch-size",
    "--bf16",
    "--biencoder-projection-dim",
    "--biencoder-shared-query-context-model",
    "--cache-mla-latents",
    "--calculate-per-token-loss",
    "--check-for-large-grads",
    "--check-for-spiky-loss",
    "--checkpoint-activations",
    "--ckpt-assume-constant-structure",
    "--ckpt-convert-format",
    "--ckpt-convert-save",
    "--ckpt-format",
    "--ckpt-fully-parallel-load",
    "--ckpt-fully-parallel-save",
    "--ckpt-step",
    "--clip-grad",
    "--config-logger-dir",
    "--context-parallel-size",
    "--cp-comm-type",
    "--cpu-offloading-num-layers",
    "--cross-entropy-fusion-impl",
    "--cross-entropy-loss-fusion",
    "--cuda-graph-impl",
    "--cuda-graph-scope",
    "--data-cache-path",
    "--dataloader-defer-npy-index-mmap",
    "--dataloader-fast-cache-load",
    "--dataloader-type",
    "--data-parallel-random-init",
    "--data-parallel-sharding-strategy",
    "--data-path",
    "--ddp-average-in-collective",
    "--ddp-bucket-size",
    "--ddp-num-buckets",
    "--ddp-pad-buckets-for-high-nccl-busbw",
    "--ddp-reduce-scatter-with-fp32-accumulation",
    "--decoder-first-pipeline-num-layers",
    "--decoder-last-pipeline-num-layers",
    "--decoder-num-layers",
    "--decoder-seq-length",
    "--decoupled-lr",
    "--decoupled-min-lr",
    "--defer-embedding-wgrad-compute",
    "--delay-wgrad-compute",
    "--deterministic-mode",
    "--disable-bf16-reduced-precision-matmul",
    "--disable-bias-linear",
    "--disable-gloo-process-groups",
    "--disable-mamba-mem-eff-path",
    "--disable-straggler-on-startup",
    "--disable-tp-comm-bulk-dgrad",
    "--disable-tp-comm-bulk-wgrad",
    "--disable-tp-comm-overlap-ag",
    "--disable-tp-comm-overlap-rs",
    "--disable-tp-comm-split-ag",
    "--disable-tp-comm-split-rs",
    "--dist-ckpt-format",
    "--dist-ckpt-optim-fully-reshardable",
    "--dist-ckpt-strictness",
    "--distrib-optim-fully-reshardable-mem-efficient",
    "--distributed-backend",
    "--distributed-timeout-minutes",
    "--distributed-timeout-seconds-after-init",
    "--distribute-saved-activations",
    "--embedding-init-method-std",
    "--embedding-path",
    "--enable-cuda-graph",
    "--enable-experimental",
    "--enable-ft-package",
    "--enable-full-sharding-in-hsdp",
    "--encoder-num-layers",
    "--encoder-seq-length",
    "--end-weight-decay",
    "--eod-mask-loss",
    "--exp-avg-dtype",
    "--exp-avg-sq-dtype",
    "--expert-model-parallel-size",
    "--expert-tensor-parallel-size",
    "--ffn-hidden-size",
    "--finetune",
    "--first-last-layers-bf16",
    "--fp16",
    "--fp16-lm-cross-entropy",
    "--fp32-residual-connection",
    "--fp4-format",
    "--fp4-param-gather",
    "--fp4-recipe",
    "--fp8-amax-compute-algo",
    "--fp8-amax-history-len",
    "--fp8-format",
    "--fp8-interval",
    "--fp8-margin",
    "--fp8-param-gather",
    "--fp8-recipe",
    "--fsdp-double-buffer",
    "--fsdp-manual-registration",
    "--grad-reduce-in-bf16",
    "--group-query-attention",
    "--hidden-dropout",
    "--hidden-size",
    "--hierarchical-context-parallel-sizes",
    "--init-method-std",
    "--init-method-xavier-uniform",
    "--init-model-with-meta-device",
    "--inprocess-restart",
    "--is-hybrid-model",
    "--keep-fp8-transpose-cache",
    "--kv-channels",
    "--kv-lora-rank",
    "--lazy-mpu-init",
    "--load",
    "--load-main-params-from-ckpt",
    "--local-rank",
    "--log-energy",
    "--logging-level",
    "--log-interval",
    "--log-max-attention-logit",
    "--log-memory-to-tensorboard",
    "--log-num-zeros-in-grad",
    "--log-params-norm",
    "--log-progress",
    "--log-straggler",
    "--log-throughput",
    "--log-timers-to-tensorboard",
    "--log-validation-ppl-to-tensorboard",
    "--log-world-size-to-tensorboard",
    "--loss-scale",
    "--loss-scale-window",
    "--lr",
    "--lr-decay-iters",
    "--lr-decay-samples",
    "--lr-decay-style",
    "--lr-warmup-fraction",
    "--lr-warmup-init",
    "--lr-warmup-iters",
    "--lr-warmup-samples",
    "--lr-wsd-decay-iters",
    "--lr-wsd-decay-samples",
    "--lr-wsd-decay-style",
    "--main-grads-dtype",
    "--main-params-dtype",
    "--make-vocab-size-divisible-by",
    "--manual-gc",
    "--manual-gc-interval",
    "--max-position-embeddings",
    "--merge-file",
    "--microbatch-group-size-per-virtual-pipeline-stage",
    "--min-loss-scale",
    "--min-lr",
    "--mock-data",
    "--model-parallel-size",
    "--moe-apply-probs-on-input",
    "--moe-aux-loss-coeff",
    "--moe-deepep-num-sms",
    "--moe-enable-deepep",
    "--moe-expert-capacity-factor",
    "--moe-extended-tp",
    "--moe-ffn-hidden-size",
    "--moe-flex-dispatcher-backend",
    "--moe-grouped-gemm",
    "--moe-hybridep-num-sms",
    "--moe-input-jitter-eps",
    "--moe-layer-freq",
    "--moe-layer-recompute",
    "--moe-pad-expert-input-to-capacity",
    "--moe-per-layer-logging",
    "--moe-permute-fusion",
    "--moe-router-bias-update-rate",
    "--moe-router-dtype",
    "--moe-router-enable-expert-bias",
    "--moe-router-force-load-balancing",
    "--moe-router-fused-impl",
    "--moe-router-fusion",
    "--moe-router-group-topk",
    "--moe-router-load-balancing-type",
    "--moe-router-num-groups",
    "--moe-router-padding-for-fp8",
    "--moe-router-pre-softmax",
    "--moe-router-score-function",
    "--moe-router-topk",
    "--moe-router-topk-scaling-factor",
    "--moe-shared-expert-intermediate-size",
    "--moe-shared-expert-overlap",
    "--moe-token-dispatcher-type",
    "--moe-token-drop-policy",
    "--moe-use-legacy-grouped-gemm",
    "--moe-use-upcycling",
    "--moe-z-loss-coeff",
    "--multi-latent-attention",
    "--no-align-grad-reduce",
    "--no-align-param-gather",
    "--no-async-tensor-model-parallel-allreduce",
    "--no-barrier-with-level-1-timing",
    "--no-bias-dropout-fusion",
    "--no-bias-gelu-fusion",
    "--no-bias-swiglu-fusion",
    "--no-check-for-nan-in-loss-and-grad",
    "--no-ckpt-fully-parallel-save",
    "--no-clone-scatter-output-in-embedding",
    "--no-create-attention-mask-in-dataloader",
    "--no-data-sharding",
    "--no-fp8-wgrad",
    "--no-gradient-accumulation-fusion",
    "--no-gradient-reduce-div-fusion",
    "--no-initialization",
    "--no-load-optim",
    "--no-load-rng",
    "--no-log-loss-scale-to-tensorboard",
    "--no-masked-softmax-fusion",
    "--no-mmap-bin-files",
    "--non-persistent-ckpt-type",
    "--non-persistent-global-ckpt-dir",
    "--non-persistent-local-ckpt-algo",
    "--non-persistent-local-ckpt-dir",
    "--non-persistent-save-interval",
    "--no-overlap-p2p-communication",
    "--no-persist-layer-norm",
    "--no-pin-cpu-grads",
    "--no-pin-cpu-params",
    "--no-position-embedding",
    "--normalization",
    "--norm-epsilon",
    "--no-rope-freq",
    "--no-rope-fusion",
    "--no-save-optim",
    "--no-save-rng",
    "--no-scatter-gather-tensors-in-pipeline",
    "--no-use-tokenizer-model-from-checkpoint-args",
    "--num-attention-heads",
    "--num-distributed-optimizer-instances",
    "--num-experts",
    "--num-layers",
    "--num-layers-at-end-in-bf16",
    "--num-layers-at-start-in-bf16",
    "--num-layers-per-virtual-pipeline-stage",
    "--num-query-groups",
    "--num-virtual-stages-per-pipeline-rank",
    "--num-workers",
    "--openai-gelu",
    "--optimizer",
    "--optimizer-cpu-offload",
    "--optimizer-offload-fraction",
    "--overlap-cpu-optimizer-d2h-h2d",
    "--overlap-grad-reduce",
    "--overlap-moe-expert-parallel-comm",
    "--overlap-p2p-communication-warmup-flush",
    "--overlap-param-gather",
    "--overlap-param-gather-with-optimizer-step",
    "--padded-vocab-size",
    "--pipeline-model-parallel-comm-backend",
    "--pipeline-model-parallel-layout",
    "--pipeline-model-parallel-size",
    "--position-embedding-type",
    "--profile",
    "--profile-ranks",
    "--profile-step-end",
    "--profile-step-start",
    "--qk-layernorm",
    "--q-lora-rank",
    "--recompute-activations",
    "--recompute-granularity",
    "--recompute-method",
    "--recompute-modules",
    "--recompute-num-layers",
    "--reset-attention-mask",
    "--reset-position-ids",
    "--rope-scaling-factor",
    "--rope-type",
    "--rotary-base",
    "--rotary-interleaved",
    "--rotary-percent",
    "--rotary-scaling-factor",
    "--rotary-seq-len-interpolation-factor",
    "--save",
    "--save-interval",
    "--save-retain-interval",
    "--seed",
    "--seq-length",
    "--sequence-parallel",
    "--sgd-momentum",
    "--spec",
    "--split",
    "--squared-relu",
    "--start-weight-decay",
    "--swiglu",
    "--tensorboard-dir",
    "--tensorboard-log-interval",
    "--tensorboard-queue-size",
    "--tensor-model-parallel-size",
    "--tokenizer-hf-include-special-tokens",
    "--tokenizer-hf-use-fast",
    "--tokenizer-model",
    "--tokenizer-type",
    "--tp-comm-bootstrap-backend",
    "--tp-comm-overlap",
    "--tp-comm-overlap-cfg",
    "--tp-comm-overlap-rs-dgrad",
    "--train-data-path",
    "--train-iters",
    "--train-samples",
    "--transformer-impl",
    "--untie-embeddings-and-output-weights",
    "--use-checkpoint-args",
    "--use-checkpoint-opt_param-scheduler",
    "--use-cpu-initialization",
    "--use-dist-ckpt",
    "--use-distributed-optimizer",
    "--use-flash-attn",
    "--use-legacy-models",
    "--use-mcore-models",
    "--use-megatron-fsdp",
    "--use-nccl-ub",
    "--use-precision-aware-optimizer",
    "--use-pytorch-profiler",
    "--use-ring-exchange-p2p",
    "--use-rope-scaling",
    "--use-rotary-position-embeddings",
    "--use-sharp",
    "--use-tp-pp-dp-mapping",
    "--vocab-extra-ids",
    "--vocab-file",
    "--vocab-size",
    "--wandb-entity",
    "--wandb-exp-name",
    "--wandb-project",
    "--wandb-save-dir",
    "--weight-decay",
    "--weight-decay-incr-style",
    "--wgrad-deferral-limit",
    "--window-size",
    "--yaml-cfg",
    "--override-opt_param-scheduler",
]

# Build alias map: snake_case_name → --flag-name for every known flag
_ALIAS_MAP: dict[str, str] = {_flag_to_alias(f): f for f in _ALL_FLAGS}

# Short convenience aliases that differ from the snake_case form
_ALIAS_MAP.update({
    "tp":               "--tensor-model-parallel-size",
    "pp":               "--pipeline-model-parallel-size",
    "ep":               "--expert-model-parallel-size",
    "cp":               "--context-parallel-size",
    "etp":              "--expert-tensor-parallel-size",
    "vpp":              "--num-layers-per-virtual-pipeline-stage",
    "mbs":              "--micro-batch-size",
    "gbs":              "--global-batch-size",
    "checkpoint_load":  "--load",
    "checkpoint_save":  "--save",
    "moe_token_dispatcher": "--moe-token-dispatcher-type",
})

# train_tokens is a computed alias: --train-samples = train_tokens // seq_length
_COMPUTED = {"train_tokens"}


class MegatronBackend:
    """
    Stateless translator: config dict → list[str] of Megatron CLI args.

    Accepts three kinds of inputs (all can be mixed freely):

    1. Snake_case aliases matching Megatron flag names::

           MegatronBackend.to_args(model_args, num_layers=48, hidden_size=2048)

    2. Short convenience aliases::

           MegatronBackend.to_args(model_args, tp=4, ep=8, mbs=2)

    3. Raw Megatron flags in model_args dict (direct passthrough)::

           {"--tensor-model-parallel-size": 4, "--num-layers": 48}

    4. Raw flag passthrough via ``extra_flags`` key::

           {"extra_flags": ["--recompute-granularity", "full"]}

    Priority (highest → lowest): kwargs > model_args aliases > model_args raw flags
    """

    @staticmethod
    def to_args(model_args: dict[str, Any], **kwargs: Any) -> list[str]:
        """
        Build a flat list of Megatron CLI arg strings.

        Args:
            model_args: MODEL_ARGS section from a resolved config dict.
                        Keys can be snake_case aliases, short aliases,
                        raw ``--flag-name`` strings, or ``extra_flags``.
            **kwargs:   Alias overrides (highest priority).

        Returns:
            List of strings ready to be joined and passed to pretrain_gpt.py.
            ``True``  → bare flag (``--swiglu``)
            ``False`` → flag omitted entirely
        """
        # kwargs override model_args
        effective: dict[str, Any] = dict(model_args)
        effective.update(kwargs)

        # seq_length for train_tokens computation
        seq_length = _coerce_int(
            effective.get("seq_length") or effective.get("--seq-length")
        )

        args: list[str] = []
        emitted: set[str] = set()

        # --- 1. Aliases (snake_case + short forms) ---
        for alias, flag in _ALIAS_MAP.items():
            if alias in effective:
                _emit(args, emitted, flag, effective[alias])

        # --- 2. Computed: train_tokens → --train-samples ---
        if "train_tokens" in effective:
            if seq_length is None:
                raise ValueError(
                    "train_tokens requires seq_length (or --seq-length) to be set "
                    "for --train-samples = train_tokens // seq_length"
                )
            train_samples = int(effective["train_tokens"]) // seq_length
            _emit(args, emitted, "--train-samples", train_samples)

        # --- 3. Raw --flag-name keys: direct passthrough ---
        for key, val in model_args.items():
            if not key.startswith("--"):
                continue
            if key in emitted:
                continue  # already handled by alias
            _emit(args, emitted, key, val)

        # --- 4. extra_flags: raw string passthrough, no transformation ---
        extra = effective.get("extra_flags", [])
        if isinstance(extra, list):
            args.extend(str(f) for f in extra)
        elif isinstance(extra, str):
            args.append(extra)

        return args

    @staticmethod
    def flag_for_alias(alias: str) -> str | None:
        """Return the Megatron flag name for a short alias, or None."""
        return _ALIAS_MAP.get(alias)

    @staticmethod
    def alias_map() -> dict[str, str]:
        """Return a copy of the full alias → flag mapping table."""
        return dict(_ALIAS_MAP)

    @staticmethod
    def known_flags() -> list[str]:
        """Return all known Megatron flag names."""
        return list(_ALL_FLAGS)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _emit(args: list[str], emitted: set[str], flag: str, val: Any) -> None:
    """Append flag+value to args, respecting bool semantics."""
    if val is False or val == "false":
        return  # omit flag entirely
    emitted.add(flag)
    if val is True or val == "true":
        args.append(flag)  # bare flag, no value
    else:
        args.extend([flag, str(val)])


def _coerce_int(val: Any) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None
