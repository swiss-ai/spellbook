# Changelog

Notable changes to Spellbook are documented here.

## Unreleased

### Added

- A declarative experiment-reporting API for selecting WandB runs, caching and
  stitching continuation histories, and rendering consistent training curves,
  evaluation macros and trajectories, grouped task heatmaps, learning-rate
  bowls, and endpoint or Chinchilla-style scaling-law fits.
- Megatron parameter counting for standard attention, MLA, and KDA
  architectures, including mixed KDA layer patterns and latent-MoE projections.
- Task-mode Megatron launches can disable Spellbook's explicit `numactl` wrapper
  with `SlurmBackend(numa_bind=False)`, leaving CPU affinity to Slurm.
- Slurm launches can use a non-login command shell with
  `SlurmBackend(login_shell=False)` when all environment setup is explicit.
- An optional, self-contained pre-launch torch/NCCL node-health gate for Megatron batch experiments. It localizes failed nodes, maintains a shared exclusion list, and retries on a fresh allocation before training starts. The design is adapted from Guanshujie’s Megatron Slurm launcher.

- A `tools` package for standalone operational launchers, containing the
  multi-node Megatron dynamic-inference HTTP server and a distributed Slurm
  launcher for Megatron checkpoint merging (e.g. checkpoint averaging)
- Configurable checkpoint/evaluation grouping for Megatron jobs, allowing each
  axis to use shared or separate allocations while retaining per-run outputs,
  environments, and WandB names and IDs.
- A single-node Slurm launcher for lm-evaluation-harness's standard vLLM
  backend, with tensor and data parallel options, common lm-eval settings,
  install hooks, caching, and WandB integration.
- A fixed-interface hfconverter Stage-2 submission helper with local or Git URL
  sources, locked refreshed checkout caches, optional commit pinning,
  completed-output reuse, explicit recreation, and vLLM job dependencies.
- Persistent, configuration-keyed Triton and TorchInductor caches with
  node-local staging, distributed cache merging, and warmup-only runs.

### Changed

- Megatron evaluation launches now use the shared IOPStor readiness barrier after container copy and package installation, with a ten-minute default timeout.


- Slurm Megatron task launches now use a shared IOPStor readiness barrier after Megatron copying and package installation, so every rank waits for all ranks before starting Python; the default timeout is 10 minutes.


- Megatron evaluation prefetch now builds lm-eval task-derived dataset caches on
  rank zero before releasing distributed ranks, including custom task paths.

### Fixed

- Batch-launched Megatron training, evaluation, inference, and checkpoint-merge
  steps now inherit their task layout from the matching `#SBATCH` allocation
  instead of redundantly overriding `--nodes`, `--ntasks`, or
  `--ntasks-per-node` in `srun`. This avoids incorrect worker creation on Slurm
  configurations where job-step task overrides conflict with the allocation.
- Cache-hit warmups no longer enter distributed cache-save cleanup.
- Megatron inference servers bind to each node's routable IP instead of assuming
  its hostname names a local network interface.

### Known limitations

- The vLLM evaluation path is not correctly implemented end to end and is not
  production-ready.

## 0.2.0 - 2026-08-06

### Added

- A packaged Megatron-LM evaluation runner built around lm-evaluation-harness,
  including single-checkpoint and range submission, a self-scheduling checkpoint
  watcher, result upload support, and deployment documentation.
- Evaluation options for task-based Slurm launches, tensor and expert
  parallelism, sequence and micro-batch sizes, metadata, output directories,
  sample logging, request caching, excluded nodes, custom `srun` flags,
  environment variables, and per-node install hooks.
- A one-process-per-Slurm-task backend launch mode with local NUMA binding.
- Training pre-launch and install command hooks, backend environment defaults,
  `srun` network propagation, and configurable CUDA graph scopes.
- Local or Git URL Megatron sources for training and evaluation. URL caches are
  refreshed under a lock, branch refs resolve against the remote, and
  source-specific worktrees prevent cross-repository collisions.
- Configurable container-local Megatron copies, defaulting to `/opt/megatron`,
  with non-container launches retaining the original source checkout.
- Manifest-backed dataset discovery with deterministic shard ordering, optional
  symlink traversal, existing manifest support, and explicit data-path
  precedence.
- Unit coverage for data discovery, manifests, launch rendering, Git cache and
  worktree behavior, container safety, watcher lifecycle, and distributed setup.

### Changed

- Evaluation and backend Jinja templates are included in built distributions.
- Triton and TorchInductor caches remain process-specific on container-local
  `/tmp` storage.
- Package installation is serialized once per node where tasks share a
  container filesystem.
- Dataset prefetch now publishes a bounded shared completion status before ranks
  enter Hugging Face offline mode.
- Deployment-specific eval configurations and generated runtime artifacts now
  live in the experiments repository rather than the Spellbook package.
- Local Ruff and ty checks exclude the optional Megatron-dependent memory
  estimator subtree, whose dependencies are supplied by its runtime environment.

### Fixed

- Prevented concurrent WandB initialization from multiple eval nodes.
- Made evaluation setup, installation, prefetch, and execution failures stop
  jobs instead of silently continuing or leaving peers waiting indefinitely.
- Fixed lm-eval metadata and install-status shell quoting.
- Created Slurm log directories before submission.
- Made `.env` optional for evaluation jobs.
- Corrected watcher checkpoint overrides, shared state placement, and reliable
  stopping of self-rescheduling watcher chains.
- Avoided destructive `/opt/megatron` copies during non-container launches and
  rejected unsafe or nested copy destinations.
- Refreshed cached Git sources so unpinned GitHub URLs do not remain stale as new
  commits are added.

### Operational notes

- Git URL sources perform a remote fetch before launch. An unpinned URL follows
  the remote default branch; set `megatron_commit` for reproducible pinned runs.
- Multi-node `dataset_prefetch` requires `output_dir` on storage shared by every
  node. `prefetch_timeout_seconds` bounds both download and peer-wait time.
- Per-model eval configs should be kept in the deployment/experiments repository
  and passed to Spellbook by path.
