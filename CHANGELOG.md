# Changelog

Notable changes to Spellbook are documented here.

## Unreleased

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
