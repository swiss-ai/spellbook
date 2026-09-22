# `vllm-apertus2` container image

This Alps CUDA application image builds the tested Apertus2 vLLM 0.28 source commit
`andresnowak/vllm@929e3cfbed769a2fe44239fe568befa8700351d4` from
[swiss-ai/vllm#23](https://github.com/swiss-ai/vllm/pull/23) for GH200. It also
builds upstream UCCL EP and P2P plus the matching NIXL 1.3.2 UCCL plugin. Switch
`VLLM_REPO` back to `swiss-ai/vllm` only after that PR is merged and the pin includes it.

The pinned Apertus2 vLLM source vendors its required Flash Linear Attention
implementation under `vllm.third_party.flash_linear_attention`. The image verifies
that vendored KDA module directly; it does not install the unrelated external
`flash-linear-attention` package.

The communication paths are separate:

- vLLM MoE expert parallelism uses the `deep_ep` compatibility wrapper backed by
  `uccl.ep` with `UCCL_EP_TRANSPORT=cxi`.
- Disaggregated prefill/decode uses `NixlConnector`, the NIXL `UCCL` plugin,
  `libuccl_p2p.so`, and `UCCL_P2P_TRANSPORT=cxi`.
- NCCL remains on the Alps AWS Libfabric plugin. The image does not replace it with
  UCCL's verbs-oriented NCCL plugin.

## Source and package pins

| Component | Pin |
|---|---|
| Apertus2 vLLM fork | `andresnowak/vllm@929e3cfbed769a2fe44239fe568befa8700351d4` ([PR #23](https://github.com/swiss-ai/vllm/pull/23)) |
| UCCL fork | `andresnowak/uccl@0c166355310a04b2344ca0c5c0df818b04d26af0` (`ep-runtime-config-overrides`) |
| NIXL source/plugin | `v1.3.2` / `de8115ca97d3f8fb63a4988e9b4d4a038b2e0f72` |
| NIXL Python package | `1.3.2`, matching vLLM's `requirements/kv_connectors.txt` |
| Run:ai model streamer | `0.15.7` |
| Transformers | `5.17.0` |
| CUDA architecture | `sm_90` |

The vLLM build keeps the base image's Torch, CUDA, and NVIDIA runtime libraries. It
uses vLLM's `use_existing_torch.py`, builds all native extensions from the pinned
source, and installs the exact CUTLASS DSL 4.6.2 and quack-kernels 0.6.4 pair required
by that source tree. The final image runs `pip check`, checks native-library linkage,
and requires the NIXL UCCL plugin to exist.

## Build structure and cache

The Containerfile uses a shared `uv-base` plus separate `vllm-builder`,
`uccl-builder`, `nixl-builder`, and `runtime` stages. Package installation uses
pinned `uv` with the base interpreter and constraints that preserve the base
Torch stack. Wheel builds use `uv build`; only UCCL's upstream-owned build script
uses its own build frontend. Each native project is compiled once, and only
wheels and runtime artifacts enter the final image.

On Alps, use the [`eth-cscs/local-registry`](https://github.com/eth-cscs/local-registry)
wrapper exactly as documented:

```bash
. ~/.local/share/local-registry/env-registry
registry up "$SCRATCH/tmp/local-registry/registry"
podman-cached --network=host \
  --build-arg "BASE_IMAGE=<base-image>" \
  -f Alps-Images/apps/vllm-apertus2/Containerfile \
  -t vllm-apertus2:0.28 .
```

Do not add a separate `--cache-from`, `--cache-to`, output format, timestamp, or
cache TTL. `podman-cached` owns those settings and uses `$LOCAL_REGISTRY/cache`.

## Publishing to `eth-cscs/alps-extended-images`

This directory follows that repository's application-image layout, but Spellbook is
the source of truth for now. To stage a build in a separate checkout:

```bash
ALPS=~/developer/alps-extended-images
rsync -a --delete --exclude README.md \
  containers/vllm-apertus2/ "$ALPS/Alps-Images/apps/vllm-apertus2/"
```

Do not commit the staged copy unless this image is deliberately proposed upstream.
The upstream build context is the repository root because the Containerfile copies
`Alps-Images/common/package-helpers.sh` and its app-local tests.

The profile uses `pytorch-cuda:26.07-py3`, matching the vLLM 0.29 image recipe used
as the build reference. The Apertus2 vLLM source and all communication components
are rebuilt rather than overlaid as Python files on the 0.29 image.

## Runtime

Use [`vllm-apertus2.toml`](vllm-apertus2.toml). Its image path is:

```text
${SCRATCH}/img/vllm028_apertus2_uccl.sqsh
```

Persistent caches use Ritom. The existing GLM-5.3 weights remain in the Iopstor HF
cache. Per-rank compiler caches use node-local `/tmp`; jobs do not write caches or
logs under `$HOME`.

The image first needs to pass `tests/import-smoke.sh` on one GPU node. The pinned
UCCL branch adds five-integer `UCCL_EP_DISPATCH_CONFIG` and
`UCCL_EP_COMBINE_CONFIG` overrides without changing UCCL defaults. Switch
`UCCL_REPO` back to `uccl-project/uccl` only after the corresponding upstream PR
is merged and the pin includes it.

The complete Chonk topology has been validated on four GH200 nodes: EP8 DeepEP high-throughput prefill, EP8 DeepEP low-latency decode, and NIXL UCCL KV transfer over CXI. Use `tools.inference.vllm_pd_server` for that explicit topology.
