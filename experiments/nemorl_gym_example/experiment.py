"""NeMo-RL GRPO through NeMo Gym, on two nodes.

Apertus2 KDA 5B against the instruction_following environment. Validated on
2026-09-14 (job 3394272: two Ray nodes, 10 steps, rewards 0.1250-0.3021, rc=0).

Run prepare_data.py first; this file only reads what it produced.

    python main.py list   experiments/nemorl_gym_example/experiment.py
    python main.py render experiments/nemorl_gym_example/experiment.py
    python main.py submit experiments/nemorl_gym_example/experiment.py

The recipe is rendered as a DELTA over base_config, which NeMo-RL merges underneath
it (`defaults`). Everything not set here -- MoE dispatcher, DDP overlap, loss_fn,
micro-batch sizes, advantage clipping -- comes from that parent. Leave base_config
empty to render a standalone recipe instead, in which case this file must supply
everything NeMo-RL requires.
"""

from __future__ import annotations

import os

from spellbook.backends import SlurmNemoRLBackend
from spellbook.core import Sweep
from spellbook.nemorl import PASSTHROUGH_CHAT_TEMPLATE, NemoRLExperiment

SCRATCH = os.environ.get("SCRATCH", "/iopsstor/scratch/cscs/anowak")
NEMO_RL = "/users/anowak/open_source/Nemo-RL"
PREP = f"{SCRATCH}/tmp/spellbook-nemorl-gym"  # must match prepare_data.py
RUN = f"{SCRATCH}/tmp/spellbook-nemorl-gym-run"
CHECKPOINTS = "/capstor/store/cscs/swissai/infra01/apertus_2/kda_scaling_ladder"
# config.json + configuration_apertus2.py only, produced by the session that first
# ran this checkpoint. No weights and no modeling_*.py: AutoBridge dispatches on
# config.json's `architectures`.
PRIOR = (
    "/users/anowak/agents-scratchpad/debugging-development/"
    "apertus2-nemorl-native-megatron-20260911T142606Z"
)

experiment = NemoRLExperiment(
    name="kda5b-gym-if-2node",
    # The maintained NeMo-RL recipe this run inherits from.
    base_config=f"{NEMO_RL}/examples/configs/recipes/llm/grpo-apertus2-kda5b-gym-instruction-following-long.yaml",
    # ---- repos: bind-mounted, not vendored in the image ----
    nemo_rl_path=NEMO_RL,
    bridge_src_path="/users/anowak/open_source/SwissAI-Megatron-Bridge/src",  # .../src, not the repo root
    megatron_path="/users/anowak/open_source/Megatron-LM-MoE",
    overlay_paths=(),  # the nemo-rl image carries ray, flashinfer and the rest
    # ---- algorithm ----
    algorithm="grpo",  # entrypoint and recipe key; set `entrypoint` for run_*.py variants
    max_num_steps=10,
    # The eval sampler's data parallelism is world_size / (TP x PP x CP) = 8 here;
    # expert parallelism does not reduce it, and Megatron defaults
    # eval_global_batch_size to global_batch_size. So the batch must divide by 8.
    num_prompts_per_step=8,
    num_generations_per_prompt=8,
    # ---- model ----
    # Weights load from the native Megatron DCP; the HF dir supplies only the
    # config.json that AutoBridge dispatches on -- no weights, no modeling_*.py.
    hf_config_dir=f"{PRIOR}/apertus2-kda-5b-hf-config",
    tokenizer_dir=f"{PRIOR}/apertus2-kda-5b-hf-config",
    pretrained_checkpoint_path=(
        f"{CHECKPOINTS}/kda-5b-moe-256e-latent-kda31-nope-muonmd-lr3.21e-3-"
        "mlr1.60e-2-latmoe-e256-top8-kda31-nope-kda-5b-moe-256e-latent-kda31-nope"
    ),
    pretrained_checkpoint_format="megatron_lm",
    # Base checkpoint, never trained on chat markers: they push it into
    # meta-commentary and degenerate repetition.
    chat_template=PASSTHROUGH_CHAT_TEMPLATE,
    # ---- parallelism ----
    expert_model_parallel_size=4,
    # ---- generation (Megatron backend: no vLLM, so no weight refit) ----
    generation_backend="megatron",
    temperature=1.0,
    top_p=0.9,  # never 1.0: that disables nucleus truncation and admits tail garbage
    top_k=0,
    max_new_tokens=512,
    max_total_sequence_length=4096,
    max_input_seq_length=3584,
    # ---- optimizer ----
    # md_decoupling's own settings (hypersphere_*, muon_*, the layer-wise optimizer)
    # live in base_config; megatron_optimizer is passed through to Megatron's
    # OptimizerConfig verbatim, so only this run's values go here.
    optimizer="md_decoupling",
    # Two orders of magnitude below the 3B: this checkpoint is further along and a
    # 2e-3 restart moves the weights far more than the gains can absorb.
    lr=2.0e-5,
    min_lr=2.0e-6,
    weight_decay=0.0,
    megatron_optimizer={"matrix_lr": 8.0e-4, "gains_lr": 2.0e-5},
    lr_decay_style="linear",
    lr_decay_iters=550,
    lr_warmup_iters=50,
    lr_warmup_init=1.0e-6,
    # ---- NeMo Gym: prepared by prepare_data.py ----
    gym_home=f"{NEMO_RL}/3rdparty/Gym-workspace/Gym",
    gym_config_paths=(
        "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml",
        "resources_servers/instruction_following/configs/instruction_following.yaml",
    ),
    gym_venv_dir=f"{PREP}/gym-venvs",
    nemo_rl_venv_dir=f"{PREP}/nemo-rl-venvs",
    gym_uv_cache_dir=f"{PREP}/uv-cache",
    gym_scratch_root=f"{PREP}/gym-root",
    data_path=f"{PREP}/data/instruction_following/train.jsonl",
    validation_data_path=f"{PREP}/data/instruction_following/validation.jsonl",
    # ---- logging ----
    log_dir=f"{RUN}/nemo-rl-logs",
    # Override the parent recipe's checkpoint directory even with saving disabled:
    # NeMo-RL otherwise auto-resumes any existing step_* it finds there.
    checkpoint_dir=f"{RUN}/checkpoints",
    wandb_enabled=False,
    tensorboard_enabled=True,
    # ---- anything the backend does not model ----
    recipe_overrides={
        # The Gym runner rejects a non-null max_val_samples.
        "grpo": {
            "max_val_samples": None,
            "val_period": 0,
            "val_at_start": False,
            "val_at_end": False,
        },
        "checkpointing": {"enabled": False},
        "policy": {"train_global_batch_size": 8},
    },
)

backend = SlurmNemoRLBackend(
    account="infra01",
    partition="normal",
    reservation="SD-69241-apertus-1-5-0",
    nodes=2,
    gpus_per_node=4,
    cpus_per_task=288,
    run_time="02:00:00",
    container=f"{SCRATCH}/img/nemorl_alps7_te217_groupgemm_emrgopt_deepgemm_fla52_uccl.toml",
    # Mount the venv squashfs at exactly gym_venv_dir: the venvs hold editable .pth
    # files with absolute paths. Drop this entry to read the loose venv directory.
    container_mounts=(
        "/iopsstor:/iopsstor,/capstor:/capstor,/users/anowak:/users/anowak,"
        f"{PREP}/gym-venvs.sqsh:{PREP}/gym-venvs:sqsh"
    ),
    log_dir=f"{RUN}/slurm_logs",
    ray_tmpdir="/tmp/ray-spellbook-nemorl",
    # Requeue under the same name until max_num_steps is reached. Needs a
    # checkpoint_dir to count step_* against, so it is off here.
    auto_requeue=False,
    # Validate every node before the Ray cluster starts. Reports per node, so a
    # failure names the bad node, and the min_bandwidth checks also catch a node
    # that works but is slow. Failing nodes are excluded and the job resubmits.
    vetnode=True,
)

sweep = Sweep(name="kda5b-gym-if-2node", experiments=[experiment], backend=backend)
