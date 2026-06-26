"""
Megatron-LM flag translation: Python field names → --cli-flags.

The vast majority of flags follow a simple rule:
    snake_case_name → --snake-case-name  (replace _ with -, prefix --)

Short convenience aliases for the most common parallelism params are also
supported: tp, pp, ep, etp, cp, vpp, mbs, gbs.

to_args() takes a flat dict of experiment fields and returns a list of
strings ready to pass to pretrain_gpt.py.

Boolean handling:
  True  → bare flag  (--swiglu)
  False / None → omitted entirely

Special:
  train_tokens → --train-samples = train_tokens // seq_length
    Lists        → space-joined value string (except moe_layer_freq)
    moe_layer_freq → bracketed CSV list string (e.g. [1,1,1])
"""

from __future__ import annotations

from typing import Any


# Short aliases that don't follow the snake→kebab rule
_ALIASES: dict[str, str] = {
    "tp":   "--tensor-model-parallel-size",
    "pp":   "--pipeline-model-parallel-size",
    "ep":   "--expert-model-parallel-size",
    "etp":  "--expert-tensor-parallel-size",
    "cp":   "--context-parallel-size",
    "vpp":  "--num-layers-per-virtual-pipeline-stage",
    "mbs":  "--micro-batch-size",
    "gbs":  "--global-batch-size",
}

# Fields that are Python-only metadata — never forwarded to Megatron
_SKIP = frozenset({
    "name",
    "env_vars",
    "training_args",
    "train_tokens",
    # Infra fields handled by the backend, not Megatron CLI
    "megatron_path",
    "megatron_commit",
    "training_script",
    "pre_launch_commands",
    "install_commands",
    # Spellbook-level parallelism helpers (num_gpus drives dp, not a Megatron flag)
    "num_gpus",
    "dp",
    "edp",
    # Data path resolution — handled by the template via create_data_config.py
    "base_data_path",
    "data_path",
    # nsys fields — handled by the template, not Megatron (except profile_step_start/end/ranks
    # which are also passed as --profile-step-start etc. but via the template block)
    "nsys_output",
    "profile_types",
    "pytorch_nsys_profile",
    "python_sampling",
    "nic_metrics",
    # debugpy fields — handled by the template
    "debug",
    "debug_port",
    # W&B resume — injected as env vars (WANDB_RUN_ID, WANDB_RESUME), not a Megatron flag
    "wandb_id",
    "torchrun_standalone",
    "extra_args",
})


def _to_flag(field_name: str) -> str:
    """snake_case_field → --kebab-case-flag"""
    return "--" + field_name.replace("_", "-")


def to_args(fields: dict[str, Any]) -> list[str]:
    """
    Translate experiment fields into a flat list of Megatron CLI arg strings.

    Lookup order for each field:
      1. Short alias (_ALIASES)
      2. Mechanical snake→kebab conversion

    Unknown / _SKIP fields are silently ignored.
    """
    emitted: set[str] = set()
    args: list[str] = []

    def emit(flag: str, val: Any) -> None:
        if flag in emitted:
            return
        if val is None or val is False or val == "" or val == []:
            return
        emitted.add(flag)
        if val is True:
            args.append(flag)
        elif isinstance(val, list):
            if flag == "--moe-layer-freq":
                args.extend([flag, "[" + ",".join(str(x) for x in val) + "]"])
            else:
                args.extend([flag, " ".join(str(x) for x in val)])
        else:
            args.extend([flag, str(val)])

    for field_name, val in fields.items():
        if field_name in _SKIP:
            continue
        flag = _ALIASES.get(field_name) or _to_flag(field_name)
        emit(flag, val)

    # Special: train_tokens → --train-samples = train_tokens // seq_length
    if "train_tokens" in fields and fields["train_tokens"]:
        seq = fields.get("seq_length") or fields.get("seq_len")
        if seq:
            emit("--train-samples", int(fields["train_tokens"]) // int(seq))

    return args
