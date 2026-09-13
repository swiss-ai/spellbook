# NeMo-RL GRPO through NeMo Gym — worked example

Apertus2 KDA 5B against the `instruction_following` environment on two nodes.
Validated 2026-09-14 (job 3394272): vetnode passed all eight GPU ranks, both Ray
nodes registered, the squashfs-mounted Gym services launched, and all 10 training
steps completed with rewards from 0.1250 to 0.3021 (`rc=0`).

Two steps, because environment preparation is shared across runs, needs no GPU, and
downloads on first use.

```bash
# 1. once per Gym checkout: server venvs, the NemoGym actor venv, the dataset
python experiments/nemorl_gym_example/prepare_data.py

# 2. per run
python main.py render experiments/nemorl_gym_example/experiment.py
python main.py submit experiments/nemorl_gym_example/experiment.py
```

Step 2 fails fast if step 1 has not run.

## What the backend does

Two `srun` steps, not one per GPU: Slurm starts a Ray daemon per node and Ray spawns
the eight `MegatronPolicyWorker` actors itself.

- **head** — `ray start --head`, waits for every worker to register, then runs the
  driver *in the same step*. The driver cannot be its own step: Ray's CoreWorker
  opens the raylet's Unix socket under the head's `/tmp`, which is per-container on
  Alps.
- **worker** — `ray start --address=<head>` on the remaining nodes.

`ray.sub` does the same thing but keeps the driver separate, using pyxis
`--container-name` to share one container. The CSCS container engine has no
equivalent.

## Recipe composition

`base_config` is rendered as NeMo-RL's `defaults`, so the file above is a **delta**.
The merged config that actually runs pulls `moe_permute_fusion`, `apply_rope_fusion`,
`overlap_param_gather`, `loss_fn`, micro-batch sizes and advantage clipping from the
parent recipe — 86 rendered nodes over ~295 inherited.

Every field the backend *does* render overrides the parent unconditionally, which is
why the generation settings are all listed explicitly: leaving `max_new_tokens` at
its default `0` would silently override the parent's `128`.

Set `base_config=""` to render a standalone recipe instead; then this file must
supply everything NeMo-RL requires.

Anything the backend does not model goes in `recipe_overrides`, deep-merged last.

## Things that will bite

- **Batch size and data parallelism.** The eval sampler's DP is
  `world_size / (TP × PP × CP)` = 8 here — expert parallelism does not reduce it —
  and Megatron defaults `eval_global_batch_size` to `global_batch_size`. A batch of 4
  fails with `eval_global_batch_size (4) must be divisible by ... (1 * 8 = 8)`.
- **`top_p`** must stay below 1.0. At 1.0 with `top_k=0` nothing truncates and the
  whole vocabulary tail is sampleable.
- **The chat template.** This base checkpoint was never trained on chat markers;
  `PASSTHROUGH_CHAT_TEMPLATE` concatenates message contents with no control tokens.
- **The squashfs mount target must equal `gym_venv_dir`.** The venvs contain
  editable `.pth` files holding absolute paths. Drop that mount entry to read the
  loose directory instead.
