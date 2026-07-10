from evals.megatron_eval import MegatronEvalConfig, RangeMode, submit_range

cfg = MegatronEvalConfig(
    model_name="BASE_FINAL",
    checkpoint_dir="/capstor/scratch/cscs/anowak/runs/checkpoints_2/apertus_moe_final/",
    tokenizer_model="swiss-ai/Apertus-70B-2509",

    megatron_path="/users/anowak/open_source/Megatron-LM",
    megatron_commit="441f49035dcbdccb8584759bb84245908b476dbb", # this one has sonicmoe, and the old one that the training model is using is this one "596f5f00cc76fdffc68dd9a5d60e3dd5a146e6d7",

    tasks=["hellaswag", "winogrande", "arc_easy", "arc_challenge", "piqa", "mmlu"], # Maybe remove MMLU
    batch_size=128, # NOTE: logical global batch size, but still need to pass megatron microbatch to control how many it does in one forward pass. So have to control DP * MBS <= 128
    devices=32,
    ep=1,
    seq_length=8192,
    # extra_args="--no-rope-fusion --moe-router-dtype fp32 --disable-bias-linear --apply-layernorm-1p --moe-use-sonicmoe --moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepep",
    extra_args="--no-rope-fusion --moe-router-dtype fp32 --disable-bias-linear --apply-layernorm-1p --distributed-timeout-minutes 80 --micro-batch-size 2",

    account="a139",
    partition="normal",
    nodes=8,
    gpus_per_node=4,
    run_time="07:00:00",
    reservation="SD-69241-apertus-1-5-0",

    container_edf="apertus2-alps4-temp",
    container_mounts="${SCRATCH}:${SCRATCH},${HOME}:${HOME},/capstor:/capstor,/iopsstor:/iopsstor",

    wandb_project="apertus-moe-final-evals",
    wandb_id="BASE_FINAL",
    hf_home="/iopsstor/scratch/cscs/anowak/hf_cache",
    # env_vars={
    #     "HF_HUB_OFFLINE": "1",
    #     "HF_DATASETS_OFFLINE": "1",
    #     "TRANSFORMERS_OFFLINE": "1",
    # },
    lm_eval_install="git+https://github.com/andresnowak/lm-evaluation-harness@0aab9dc0335996df37b6b5cc3137fa4f6c766ed9",
    # lm_eval_install="git+https://github.com/andresnowak/lm-evaluation-harness@03db3a0c5891ab76b6e1d872baf88f51d14c87c9",
    install_commands="""
pip install --upgrade --no-deps "nvidia-cutlass-dsl==4.4.2"
pip install --upgrade --no-deps "quack-kernels[cu13]==0.4.1"
pip install --no-deps "git+https://github.com/andresnowak/sonic-moe.git@7d931fe0f635f9ccfb2d292e09fdd6e72425dab6"
""".strip(),
    dataset_prefetch={
        "hellaswag":     ["Rowan/hellaswag"],
        "winogrande":    ["allenai/winogrande", "winogrande_xl"],
        "arc_easy":      ["allenai/ai2_arc", "ARC-Easy"],
        "arc_challenge": ["allenai/ai2_arc", "ARC-Challenge"],
        "piqa":          ["baber/piqa"],
        "mmlu":          ["cais/mmlu", "all"],
    },
)

if __name__ == "__main__":
    submit_range(cfg, start=34000, end=88000, step=2000, mode=RangeMode.SEQUENTIAL)
    # submit_range(cfg, start=34000, end=34000, step=2000, mode=RangeMode.SEQUENTIAL)
