"""Use Slurm's per-task local rank for vLLM's native-MP GPU selection."""

import os

from vllm.v1.worker.gpu_worker import Worker as GPUWorker


class SlurmGPUWorker(GPUWorker):
    """GPU worker that maps each Slurm task to its allocated local GPU."""

    def init_device(self) -> None:
        self.parallel_config.data_parallel_rank_local = int(os.environ["SLURM_LOCALID"])
        super().init_device()
