"""
Big MoE speed ablation — spellbook version.

Mirrors big-moe-speed-ablations from Megatron-MoE-ModelZoo.
Each variant is defined by calling BASE.change(...) with only what differs.

Usage:
    python main.py list    experiments/big_moe_speed_ablations/experiment.py
    python main.py render  experiments/big_moe_speed_ablations/experiment.py
    python main.py submit  experiments/big_moe_speed_ablations/experiment.py
"""

from __future__ import annotations

import dataclasses
import os

from spellbook.backends import SlurmBackend
from spellbook.core import Sweep
from spellbook.megatron import MegatronExperiment


# ---------------------------------------------------------------------------
# Extended experiment dataclass — fields specific to this ablation
# ---------------------------------------------------------------------------

_L61_FREQ = [0, 0, 0] + [1] * 58


@dataclasses.dataclass
class BigMoEExperiment(MegatronExperiment):
    gradient_clipping: float = 1.0
    force_router_balancing: bool = False
    main_grads_dtype: str = "fp32"
    delay_wgrad_compute: bool = False
    ddp_bucket_size: int | None = None
    ddp_pad_buckets: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self.env_vars.setdefault("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")
        self.env_vars.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        self.env_vars.setdefault("NCCL_NVLS_ENABLE", "0")
        self.env_vars.setdefault("TORCH_NCCL_AVOID_RECORD_STREAMS", "0")
        self.env_vars.setdefault("NCCL_GRAPH_REGISTER", "0")
        self.env_vars.setdefault("TORCH_NCCL_HIGH_PRIORITY", "1")
        self.env_vars.setdefault("NVTE_FUSED_ATTN", "1")
        self.env_vars.setdefault("NVTE_NORM_FWD_USE_CUDNN", "1")
        self.env_vars.setdefault("NVTE_NORM_BWD_USE_CUDNN", "0")
        self.env_vars.setdefault("NVTE_USE_CUTLASS_GROUPED_GEMM", "0")
        self.env_vars.setdefault("NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK", "1")
        self.env_vars.setdefault("TOKENIZERS_PARALLELISM", "False")
        self.env_vars.setdefault("NCCL_DEBUG", "WARN")
        if self.overlap_moe_expert_parallel_comm:
            self.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] = 32
            self.env_vars["NVTE_FWD_LAYERNORM_SM_MARGIN"] = 24
            self.env_vars["NVTE_BWD_LAYERNORM_SM_MARGIN"] = 24
        else:
            self.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] = 1
            self.env_vars["NVTE_FWD_LAYERNORM_SM_MARGIN"] = 0
            self.env_vars["NVTE_BWD_LAYERNORM_SM_MARGIN"] = 0


# ---------------------------------------------------------------------------
# Base experiment
# ---------------------------------------------------------------------------

BASE = BigMoEExperiment(
    name="DEEPSEEK_V3_BASE",
    # Architecture
    num_layers=61,
    hidden_size=7168,
    ffn_hidden_size=18432,
    num_attention_heads=128,
    num_query_groups=128,
    vocab_size=131072,
    make_vocab_size_divisible_by=128,
    seq_length=4096,
    moe_layer_freq=_L61_FREQ,
    # MoE
    num_experts=256,
    moe_router_topk=8,
    moe_ffn_hidden_size=2048,
    moe_shared_expert_intermediate_size=2048,
    moe_router_score_function="sigmoid",
    moe_router_topk_scaling_factor=2.5,
    moe_router_num_groups=16,
    moe_router_group_topk=2,
    moe_router_load_balancing_type="seq_aux_loss",
    moe_aux_loss_coeff=0.0001,
    moe_router_force_load_balancing=True,
    moe_token_dispatcher_type="alltoall",
    moe_grouped_gemm=True,
    overlap_moe_expert_parallel_comm=True,
    # Parallelism
    tp=2, pp=8, ep=64, etp=1, num_gpus=1024,
    pipeline_model_parallel_layout="Et|(tt|)*30L",
    sequence_parallel=True,
    # Batch / schedule
    mbs=1, gbs=8192,
    train_tokens=1_048_576_000,
    lr=3.9e-6, min_lr=3.9e-7,
    lr_decay_style="WSD",
    lr_warmup_iters=0,
    lr_wsd_decay_iters=0,
    adam_beta1=0.9, adam_beta2=0.95, adam_eps=1e-8,
    init_method_std=0.0059,
    weight_decay=0.1,
    clip_grad=1.0,
    seed=28,
    # Precision
    bf16=False,
    fp8_format="hybrid",
    fp8_recipe="tensorwise",
    fp8_amax_history_len=16,
    fp8_amax_compute_algo="max",
    # Recompute
    recompute_granularity="selective",
    recompute_modules=["mla_up_proj", "mlp"],
    # Optimizer
    optimizer="adam",
    use_distributed_optimizer=True,
    overlap_param_gather=True,
    overlap_grad_reduce=True,
    delay_wgrad_compute=True,
    # Misc
    calculate_per_token_loss=True,
    log_throughput=True,
    # Paths (fill in per cluster)
    megatron_path=os.environ.get("MEGATRON_PATH", "/path/to/megatron"),
    data_path=os.environ.get("DATA_PATH", ""),
    tokenizer_model=os.environ.get("TOKENIZER_MODEL", ""),
    save=os.environ.get("CHECKPOINT_SAVE", ""),
    wandb_project="big-moe-speed-ablations",
)

# ---------------------------------------------------------------------------
# Variants — only specify what changes vs BASE
# ---------------------------------------------------------------------------

AYUSH = BASE.change(
    "DEEPSEEK_V3_AYUSH",
    ffn_hidden_size=16384,
    lr=3e-4, min_lr=3e-5,
    init_method_std=0.008944,
    mbs=2,
    moe_aux_loss_coeff=1e-2,
    moe_router_load_balancing_type="aux_loss",
    moe_router_topk_scaling_factor=1.0,
    moe_router_num_groups=8,
    moe_router_group_topk=4,
    overlap_moe_expert_parallel_comm=False,
    delay_wgrad_compute=False,
    tp=4, pp=16, ep=8, num_gpus=512,
    moe_token_dispatcher_type="allgather",
    main_grads_dtype="bf16",
)

AYUSH_OVERLAP = AYUSH.change(
    "DEEPSEEK_V3_AYUSH_OVERLAP",
    overlap_moe_expert_parallel_comm=True,
    delay_wgrad_compute=True,
)

AYUSH_CUDA_GRAPH = AYUSH.change(
    "DEEPSEEK_V3_AYUSH_CUDA_GRAPH",
    transformer_impl="transformer_engine",
)

# ---------------------------------------------------------------------------
# Backend — cluster/container config lives here, not in the experiment
# ---------------------------------------------------------------------------

backend = SlurmBackend(
    account=os.environ.get("SLURM_ACCOUNT", "your-account"),
    partition=os.environ.get("SLURM_PARTITION", "gpu"),
    gpus_per_node=4,
    run_time="04:00:00",
    log_dir="slurm_logs/big-moe-speed-ablations",
    extra={
        "container_edf": os.environ.get("CONTAINER_EDF", ""),
        "container_mounts": os.environ.get("CONTAINER_MOUNTS", ""),
    },
    env_vars={
        "EMERGING_OPTIMIZERS_PATH": os.environ.get(
            "EMERGING_OPTIMIZERS_PATH", "/path/to/Emerging-Optimizers"
        ),
    },
    pythonpath_env_vars=["EMERGING_OPTIMIZERS_PATH"],
)

# ---------------------------------------------------------------------------
# Sweep — the object main.py operates on
# ---------------------------------------------------------------------------

sweep = Sweep(
    name="big-moe-speed-ablations",
    experiments=[BASE, AYUSH, AYUSH_OVERLAP, AYUSH_CUDA_GRAPH],
    backend=backend,
)
