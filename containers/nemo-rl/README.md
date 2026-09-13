# `nemo-rl` container image

An Alps application image for NeMo-RL runs driven by `spellbook.nemorl`: a Megatron
policy with the Megatron generation backend (no vLLM, so no weight refit and no
`param_mapping`), plus the MoE/FP8 kernel stack the Apertus2 KDA checkpoints need.

The pin set matches
`pytorch2512_alps6_te217_groupgemm0518_emrgopt04_deepgemm_fla52_cconv17_uccl`
("megachonk") and adds the NeMo-RL runtime stack on top, so **jobs need no
`PYTHONPATH` overlays**. `NemoRLExperiment.overlay_paths` can be left empty.

## Publishing to `eth-cscs/alps-extended-images`

This directory is laid out exactly as an upstream app. To contribute it:

```bash
ALPS=~/open_source/alps-extended-images
cp -a containers/nemo-rl "$ALPS/Alps-Images/apps/nemo-rl"
```

Then add `nemo-rl` to the app matrix and the `publish-gate.needs` entries in
`ci-pipelines/build-alps-extended-images.yaml` — publishing depends on that gate,
not on stage order alone. `ngc_base_refs` appends the pipeline-wide `ALPS_REV` to
`CUDA_BASE_IMAGE`, so the published image is `nemo-rl-cuda:alps7-dev` today.

Note the build context is the repository root, not this directory: the
`COPY Alps-Images/...` lines only resolve from there.

```bash
cd "$ALPS"
podman build -f Alps-Images/apps/nemo-rl/Containerfile \
  --build-arg BASE_IMAGE=jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/ngc-pytorch:25.12-py3-alps7-dev \
  -t nemo-rl:local .
```

## What the image contains

| Component | Pin | Why |
|---|---|---|
| UCCL-EP | `swiss-ai/uccl@fa4c325c` | `uccl.ep` + the `deep_ep` wrapper Megatron's flex dispatcher imports, so jobs skip the per-job srun build |
| TransformerEngine | `v2.17` | CUDA graph support; this is megachonk's TE, one minor above the `te212` sibling image |
| DeepGEMM | `FFGGSSJJ/DeepGEMM@559d79fb` | FP8 grouped GEMM |
| grouped_gemm | `FFGGSSJJ/grouped_gemm@45118e54` | MoE GEMM with gradient-accumulation fusion |
| nvidia-resiliency-ext | `0.6.0` | the version Megatron-LM pins; older ones break async checkpoint save |
| Emerging-Optimizers | `FFGGSSJJ@cc1385ee` | decoupled Muon (`md_decoupling`) |
| flash-linear-attention | `v0.5.2` | KDA kernels |
| ray | `2.56.1` | NeMo-RL worker runtime; not in the base image |
| flashinfer-python | `0.6.18.post1` | **required**, not optional: NeMo-RL hardcodes `sampling_backend="flashinfer"` in its Megatron worker, so `InferenceConfig.__post_init__` raises `ImportError` without it |
| openai | `2.7.2` | nemo-gym requires `<=2.7.2` and pins each child server venv to the *parent* version, so a newer parent makes every Gym venv unresolvable |
| transformers / megatron-energon / hydra-core / math-verify / mlflow / tensordict / swanlab | pinned | NeMo-RL import-time dependencies |
| uv | `0.11.23` | installs everything in this image, and is what NeMo Gym shells out to at run time to build its per-server venvs |

Installs go through `uv pip install --system`, not pip: it resolves the whole
dependency set at once rather than package by package. Versions that affect the
validated kernel and NeMo-RL stack are pinned inline. The image intentionally
does not use `--exclude-newer`: JFrog metadata for some required build packages
does not include upload dates, causing uv to exclude those packages entirely.
uv is bootstrapped with the repository's `pip_install` helper because nothing
else exists at that point, and `UV_DEFAULT_INDEX` points at the same CSCS JFrog
mirror so Gym venvs built at run time resolve through it too.

Every install is additionally run against `/opt/alps/base-pins.txt`, a constraints
file generated from the selected base image's own
`torch`/`torchvision`/`triton`/`numpy` versions. A transitive dependency therefore
cannot drag in a PyPI torch wheel and shadow that NGC stack, and a final build
check verifies that those versions remained unchanged during installation.

## What the image does *not* contain

NeMo-RL, Megatron-Bridge and Megatron-LM are **not** vendored. They are actively
developed forks, so bind-mount them and point the experiment at them:

```python
NemoRLExperiment(
    nemo_rl_path="/users/<you>/open_source/Nemo-RL",
    bridge_src_path="/users/<you>/open_source/SwissAI-Megatron-Bridge/src",  # .../src, not the repo root
    megatron_path="/users/<you>/open_source/Megatron-LM-MoE",
    overlay_paths=(),          # nothing to overlay: the image has it all
)
```

NeMo Gym's per-server venvs are also built at run time rather than baked in, since
they are derived from the Gym checkout's own `requirements.txt`.

## Adding packages without rebuilding

`NemoRLExperiment.install_commands` runs arbitrary shell before the Ray head
starts, matching `install_commands` in the Megatron and eval backends. Use it to
try a package before committing it to this Containerfile:

```python
NemoRLExperiment(install_commands="pip install --no-deps some-package==1.2.3")
```
