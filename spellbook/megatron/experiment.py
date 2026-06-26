"""
MegatronExperiment — Experiment subclass for Megatron-LM training runs.

Defines all common fields (architecture, parallelism, training schedule, MoE,
precision, recompute, data, logging) and overrides to_dict() to inject a
`training_args` list that the Jinja2 template passes directly to pretrain_gpt.py.

Subclass this for a specific experiment type and add extra fields::

    @dataclass
    class BigMoEExperiment(MegatronExperiment):
        force_router_balancing: bool = False
        main_grads_dtype: str = "fp32"

        def __post_init__(self):
            super().__post_init__()
            if self.overlap_moe_expert_parallel_comm:
                self.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] = 32
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any, ClassVar

from spellbook.core.experiment import Experiment
from spellbook.megatron import flags as megatron_flags


@dataclasses.dataclass
class MegatronExperiment(Experiment):
    # wandb_exp_name contains a timestamp and training_args is derived — both change on every import.
    _lock_exclude: ClassVar[frozenset[str]] = frozenset({"wandb_exp_name", "training_args"})

    # --- Megatron path ---
    megatron_path: str = ""
    megatron_commit: str = ""  # if set, a git worktree is created at this commit and used instead
    training_script: str = "pretrain_gpt.py"
    pre_launch_commands: str = ""  # raw shell commands rendered before the main srun launch
    install_commands: str = ""  # raw shell commands rendered before training, e.g. pip install ...

    # --- Environment variables injected into the sbatch script ---
    env_vars: dict[str, Any] = dataclasses.field(default_factory=dict)

    # --- Architecture ---
    num_layers: int = 0
    hidden_size: int = 0
    ffn_hidden_size: int = 0
    num_attention_heads: int = 0
    num_query_groups: int = 0
    kv_channels: int | None = None
    max_position_embeddings: int | None = None
    seq_length: int = 0
    vocab_size: int = 0
    make_vocab_size_divisible_by: int = 128
    swiglu: bool = True
    normalization: str = "RMSNorm"
    norm_epsilon: float = 1e-5
    position_embedding_type: str = "rope"
    rotary_base: int = 10_000
    rotary_percent: float | None = None
    rotary_seq_len_interpolation_factor: float | None = None
    group_query_attention: bool = True
    qk_layernorm: bool = False
    multi_latent_attention: bool = False
    q_lora_rank: int | None = None
    kv_lora_rank: int | None = None
    qk_head_dim: int | None = None
    qk_pos_emb_head_dim: int | None = None
    v_head_dim: int | None = None
    rotary_scaling_factor: float | None = None # yarn (to use use MLA or rope_type="yarn")
    rope_scaling_factor: float | None = None # llama3-style RoPE scaling (by default rope_type used is "rope" so llama 3)
    rope_type: str | None = None  # "rope" (default, in megatron when using None), "yarn"
    mscale: float | None = None
    mscale_all_dim: float | None = None
    untie_embeddings_and_output_weights: bool = True
    disable_bias_linear: bool = True
    transformer_impl: str = "transformer_engine"
    use_mcore_models: bool = True
    use_flash_attn: bool = False
    attention_dropout: float | None = None
    hidden_dropout: float | None = None
    mla_down_proj_fusion: bool = False

    # --- MoE ---
    num_experts: int | None = None
    moe_router_topk: int = 1
    moe_ffn_hidden_size: int | None = None
    moe_shared_expert_intermediate_size: int | None = None
    moe_token_dispatcher_type: str = "alltoall"
    moe_router_load_balancing_type: str = "aux_loss"
    moe_aux_loss_coeff: float | None = None
    moe_z_loss_coeff: float | None = None
    moe_router_score_function: str = "softmax"
    moe_router_topk_scaling_factor: float = 1.0
    moe_router_num_groups: int | None = None
    moe_router_group_topk: int | None = None
    moe_router_enable_expert_bias: bool = False
    moe_router_force_load_balancing: bool = False
    moe_permute_fusion: bool = False
    moe_router_dtype: str = ""
    moe_router_fusion: bool = False
    moe_layer_freq: list[int] = dataclasses.field(default_factory=list)
    moe_grouped_gemm: bool = True
    overlap_moe_expert_parallel_comm: bool = False
    moe_router_bias_update_rate: float | None = None

    # --- Parallelism ---
    tp: int = 1
    pp: int = 1
    ep: int = 1
    etp: int = 1
    cp: int = 1
    vpp: int | None = None
    pipeline_model_parallel_layout: str = ""
    num_gpus: int | None = None   # drives dp derivation; not passed to Megatron
    dp: int = 1                   # derived; not passed to Megatron directly
    sequence_parallel: bool = True
    tp_comm_overlap: bool = False

    # --- Batch / schedule ---
    mbs: int = 1
    gbs: int = 0
    train_tokens: int = 0         # converted to --train-samples = train_tokens // seq_length
    lr: float = 0.0
    min_lr: float = 0.0
    lr_decay_style: str = "cosine"
    lr_decay_samples: int | None = None
    lr_warmup_samples: int | None = None
    lr_warmup_iters: int = 0
    lr_wsd_decay_iters: int = 0
    weight_decay: float = 0.1
    clip_grad: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    seed: int = 42
    init_method_std: float = 0.02
    optimizer: str = "adam"
    use_distributed_optimizer: bool = True
    overlap_param_gather: bool = True
    overlap_grad_reduce: bool = True

    # --- Precision ---
    bf16: bool = True
    fp8_format: str | None = None
    fp8_recipe: str | None = None
    fp8_amax_history_len: int | None = None
    fp8_amax_compute_algo: str | None = None
    fp8_param_gather: bool = False

    # --- Recompute ---
    recompute_granularity: str | None = None
    recompute_method: str | None = None
    recompute_num_layers: int | None = None
    recompute_modules: list[str] = dataclasses.field(default_factory=list)

    # --- Checkpointing ---
    save: str = ""
    load: str = ""
    save_interval: int = 1000
    ckpt_format: str = "torch_dist"
    use_dist_ckpt: bool = True
    ckpt_fully_parallel_save: bool = True
    ckpt_fully_parallel_load: bool = True
    no_load_optim: bool = False
    no_load_rng: bool = False

    # --- Data ---
    # Set base_data_path to auto-discover all .bin shards (equal weight 1.0 each).
    # Set data_path directly to override with an explicit weighted string, e.g.:
    #   "1.0 /path/a 2.0 /path/b"
    # If both are set, data_path takes precedence.
    base_data_path: str = ""   # comma-separated dirs; shards found via create_data_config.py
    data_path: str = ""        # explicit weighted data-path string (passed as-is to Megatron)
    tokenizer_type: str = "HuggingFaceTokenizer"
    tokenizer_model: str = ""
    split: str = "100,0,0"
    no_mmap_bin_files: bool = False
    num_workers: int = 4
    dataloader_type: str = "single"  # "single" (default), "cyclic" or external
    no_create_attention_mask_in_dataloader: bool = False
    reset_attention_mask: bool = False
    reset_position_ids: bool = False
    data_cache_path: str = ""

    # --- Logging ---
    log_interval: int = 1
    tensorboard_dir: str = ""
    wandb_project: str = ""
    wandb_exp_name: str = ""
    wandb_id: str = ""  # if set, resumes the run via WANDB_RUN_ID + WANDB_RESUME=must
    log_throughput: bool = True
    log_params_norm: bool = True
    log_num_zeros_in_grad: bool = True
    log_validation_ppl_to_tensorboard: bool = False
    log_memory_to_tensorboard: bool = True
    log_timers_to_tensorboard: bool = True
    calculate_per_token_loss: bool = False
    eval_iters: int | None = None
    eval_interval: int | None = None

    # --- Checkpointing (resume) ---
    ckpt_step: int | None = None
    override_opt_param_schedule: bool = False

    # --- Misc ---
    async_save: bool = False
    use_persistent_ckpt_worker: bool = False
    delay_wgrad_compute: bool = False
    no_check_for_nan_in_loss_and_grad: bool = False
    auto_detect_ckpt_format: bool = False
    dist_ckpt_strictness: str = ""
    distributed_timeout_minutes: int | None = None
    exit_duration_in_mins: int | None = None
    manual_gc: bool = False
    manual_gc_interval: int | None = None
    cross_entropy_loss_fusion: bool = False
    cross_entropy_fusion_impl: str = ""
    enable_experimental: bool = False
    cuda_graph_impl: str = ""
    cuda_graph_scope: list[str] = dataclasses.field(default_factory=list)
    te_rng_tracker: bool = False


    # --- Extra passthrough args ---
    extra_args: list[str] = dataclasses.field(default_factory=list)

    # --- torchrun flags ---
    torchrun_standalone: bool = False  # passes --standalone to torchrun; useful for single-node runs

    # --- nsys profiling ---
    # When profile=True the template wraps the python launch with:
    #   nsys profile ... -o <nsys_output>/<name>-$SLURM_PROCID
    profile: bool = False
    nsys_output: str = "nsys"              # directory for .nsys-rep files
    profile_types: str = "cuda,nvtx"       # -t argument
    profile_step_start: int = 5
    profile_step_end: int = 7
    profile_ranks: list[int] = dataclasses.field(default_factory=lambda: [0])
    pytorch_nsys_profile: str = "none"
    python_sampling: bool = False
    nic_metrics: str = "none"

    # --- debugpy remote debugging ---
    # When debug=True the template replaces `python -m` with
    #   python -m debugpy --listen 0.0.0.0:<debug_port> --wait-for-client -m
    # Combine with profile=True to profile a paused process after attaching.
    debug: bool = False
    debug_port: int = 5678

    def resume_from(self, name: str, ckpt_step: int, **kwargs: Any) -> Experiment:
        """Return a variant that resumes from a checkpoint step.

        Sets ckpt_step and override_opt_param_schedule=True by default.
        Pass override_opt_param_schedule=False to suppress it.
        Any other field can be overridden via kwargs (e.g. save, lr_warmup_iters).
        """
        kwargs.setdefault("override_opt_param_schedule", True)

        if "save" not in kwargs and self.save:
            warnings.warn(
                f"resume_from({name!r}): 'save' is unchanged from the source experiment "
                f"({self.save!r}). The resumed run will overwrite the original checkpoint. "
                "Pass save=... to use a different path.",
                stacklevel=2,
            )

        _SCHEDULE_FIELDS = {
            "lr_warmup_iters", "lr_warmup_samples",
            "lr_decay_samples", "lr_wsd_decay_iters",
            "lr_decay_style", "train_tokens",
        }
        if kwargs.get("override_opt_param_schedule", True) and not (_SCHEDULE_FIELDS & kwargs.keys()):
            warnings.warn(
                f"resume_from({name!r}): --override-opt-param-schedule is set but no "
                "schedule fields were changed (lr_warmup_iters, lr_wsd_decay_iters, "
                "train_tokens, ...). Pass the relevant fields to update the schedule.",
                stacklevel=2,
            )

        return self.change(name, ckpt_step=ckpt_step, **kwargs)

    def __post_init__(self) -> None:
        self.env_vars = dict(self.env_vars)
        if self.num_gpus is not None:
            self.dp = self.num_gpus // (self.tp * self.pp * self.cp)

        if self.wandb_id:
            self.env_vars["WANDB_RUN_ID"] = self.wandb_id
            self.env_vars["WANDB_RESUME"] = "must"

        if not self.save:
            warnings.warn(
                f"{self.name!r}: 'save' is empty — checkpoints will not be written.",
                stacklevel=3,
            )
        if not self.load:
            warnings.warn(
                f"{self.name!r}: 'load' is empty — training will start from scratch.",
                stacklevel=3,
            )

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def total_gpus(self) -> int:
        if self.num_gpus is not None:
            return self.num_gpus
        return self.dp * self.tp * self.pp * self.cp

    @property
    def nnodes(self) -> int:
        return self.total_gpus // 4  # assumes 4 GPUs/node; override if needed

    # ------------------------------------------------------------------
    # Parameter count (Qwen3 MoE architecture, QK LayerNorm assumed when qk_layernorm=True)
    # ------------------------------------------------------------------

    def _param_counts(self) -> dict[str, float]:
        h = self.hidden_size

        embedding_params = self.vocab_size * h
        output_params = self.vocab_size * h if self.untie_embeddings_and_output_weights else 0

        if self.multi_latent_attention:
            # MLA projections: W_DQ (h->q_lora_rank), W_UQ (q_lora_rank->nheads*qk_head_dim),
            # W_DKV (h->kv_lora_rank), W_UK (kv_lora_rank->nheads*qk_head_dim),
            # W_UV (kv_lora_rank->nheads*v_head_dim), W_O (nheads*v_head_dim->h),
            # plus RoPE query projection W_QR (q_lora_rank->nheads*qk_pos_emb_head_dim)
            nh = self.num_attention_heads
            q_lora = self.q_lora_rank or 0
            kv_lora = self.kv_lora_rank or 0
            qk_dim = self.qk_head_dim or 0
            qk_rope = self.qk_pos_emb_head_dim or 0
            v_dim = self.v_head_dim or 0
            attn_params = (
                h * q_lora                    # W_DQ
                + q_lora * nh * qk_dim        # W_UQ (nope part)
                + q_lora * nh * qk_rope       # W_QR (rope part)
                + h * kv_lora                 # W_DKV
                + kv_lora * nh * qk_dim       # W_UK
                + kv_lora * nh * v_dim        # W_UV
                + nh * v_dim * h              # W_O
            )
        else:
            head_dim = self.kv_channels or (h // self.num_attention_heads)
            kv_h = (self.num_query_groups or self.num_attention_heads) * head_dim
            attn_params = h * h + h * kv_h + h * kv_h + h * h  # q+k+v+o

        qk_ln_params = 2 * h if self.qk_layernorm else 0
        layernorm_params = 2 * h

        ffn_multiplier = 3 if self.swiglu else 2  # SwiGLU: gate+up+down; standard: up+down
        dense_ffn_params = ffn_multiplier * h * self.ffn_hidden_size

        num_experts = self.num_experts or 0
        moe_ffn = self.moe_ffn_hidden_size or 0
        shared_intermediate = self.moe_shared_expert_intermediate_size or 0

        router_params = h * num_experts
        expert_params = ffn_multiplier * h * moe_ffn
        shared_expert_params = ffn_multiplier * h * shared_intermediate

        total_params = embedding_params + output_params
        active_params = embedding_params + output_params

        for is_moe in self.moe_layer_freq:
            base = attn_params + qk_ln_params + layernorm_params
            if is_moe:
                layer_total = base + router_params + num_experts * expert_params + shared_expert_params
                layer_active = base + router_params + self.moe_router_topk * expert_params + shared_expert_params
            else:
                layer_total = base + dense_ffn_params
                layer_active = layer_total
            total_params += layer_total
            active_params += layer_active

        return {
            "total_B": round(total_params / 1e9, 3),
            "active_B": round(active_params / 1e9, 3),
            "ratio": round(active_params / total_params, 4) if total_params else 0.0,
        }

    # ------------------------------------------------------------------
    # to_dict: produce the context dict the Jinja2 template receives
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["training_args"] = megatron_flags.to_args(d) + list(self.extra_args)
        return d
