"""
SlurmNemoRLBackend — render/submit NeMo-RL RL runs.

Two artifacts per experiment:
  1. a Hydra recipe YAML (inherits from a maintained NeMo-RL recipe),
  2. a launch script (Ray head + dependency overlays + `run_<algorithm>.py`).

Same surface as SlurmBackend: ``render``, ``render_all``, ``submit_all``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any, ClassVar

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from spellbook.nemorl.experiment import NemoRLExperiment

_TEMPLATE_DIR = Path(__file__).parent
_SHARED_TEMPLATE_DIR = Path(__file__).resolve().parent.parent


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` in place."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


# Forwarded to every srun step. The list is explicit rather than --export=ALL:
# ALL hands the submitting shell's environment to the container, so a $HOME/.local/bin
# entry on PATH shadows the image's interpreter and an inherited PYTHONPATH shadows
# its packages. Matches _SRUN_INFRA_EXPORTS in slurm_megatron.
_SRUN_INFRA_EXPORTS = [
    "SLURM_JOB_ID",
    "SLURM_NETWORK",
    "RAY_HEAD_IP",
    "RAY_ADDRESS",
    "RAY_READY_FILE",
    "RAY_DONE_FILE",
    "RAY_EXPECTED_NODES",
    "WANDB_API_KEY",
    "WANDB_ENTITY",
    "WANDB_PROJECT",
    "WANDB_MODE",
    "WANDB_RESUME",
    "WANDB_RUN_ID",
    "HF_TOKEN",
    "HF_HUB_ENABLE_HF_TRANSFER",
    "PYTHONNOUSERSITE",
]


@dataclasses.dataclass
class SlurmNemoRLBackend:
    # Algorithms that roll out, and therefore carry policy.generation and the
    # per-step rollout counts. sft and dpo train on a fixed dataset and have
    # neither (examples/configs/{sft,dpo}.yaml). Extend in a subclass if a new
    # NeMo-RL algorithm generates.
    GENERATION_ALGORITHMS: ClassVar[frozenset[str]] = frozenset({"grpo", "ppo", "distillation"})

    account: str
    partition: str
    gpus_per_node: int
    run_time: str = "01:00:00"
    nodes: int = 1
    cpus_per_task: int = 72
    reservation: str = ""
    container: str = ""
    container_mounts: str = ""
    srun_job_id: str = ""  # run inside an existing allocation instead of sbatch
    # Keep scheduler output away from the source checkout by default. This is
    # evaluated when the backend is constructed so $SCRATCH remains user-specific.
    log_dir: str = dataclasses.field(
        default_factory=lambda: str(
            Path(
                os.environ.get(
                    "SCRATCH",
                    f"/iopsstor/scratch/cscs/{os.environ.get('USER', 'unknown')}",
                )
            )
            / "tmp/spellbook/nemorl/slurm_logs"
        )
    )
    hf_home: str = "${SCRATCH:-/iopsstor/scratch/cscs/$USER}/hf_cache"
    ray_port: int = 1200
    ray_tmpdir: str = "/tmp/ray-spellbook-nemorl"
    srun_extra_args: str = "--network=disable_rdzv_get --mpi=pmix"
    # Requeue the job under the same name until max_num_steps is reached, so a run
    # longer than run_time continues from its last checkpoint. Shares the megatron
    # backend's mechanism (spellbook/backends/auto_requeue.sh.j2); NeMo-RL has no
    # completion marker to grep, so it stops on the checkpoint step count instead.
    auto_requeue: bool = False
    # CSCS node validation before the Ray cluster starts
    # (https://docs.cscs.ch/running/vetnode/).
    vetnode: bool = False
    vetnode_config: str = ""
    vetnode_install: str = "vetnode"
    vetnode_skip_install: bool = False
    vetnode_verbose: bool = False
    vetnode_numa_bind: bool = True
    vetnode_exclude: bool = True
    vetnode_max_excluded_nodes: int = 256
    env_vars: dict[str, Any] = dataclasses.field(default_factory=dict)

    # ---------- recipe ----------
    # One method per recipe section so a subclass can replace a single section
    # without copying the whole builder.
    def generates(self, exp: NemoRLExperiment) -> bool:
        return exp.algorithm in self.GENERATION_ALGORITHMS

    def validate(self, exp: NemoRLExperiment) -> None:
        """Validate settings that depend on both experiment and allocation."""
        exp.validate()
        model_parallel_size = (
            exp.tensor_model_parallel_size
            * exp.pipeline_model_parallel_size
            * exp.context_parallel_size
        )
        world_size = self.nodes * self.gpus_per_node
        if world_size % model_parallel_size:
            raise ValueError(
                f"world_size={world_size} must be divisible by TP×PP×CP ({model_parallel_size})."
            )
        data_parallel_size = world_size // model_parallel_size
        if (
            self.generates(exp)
            and exp.num_prompts_per_step
            and exp.num_prompts_per_step % data_parallel_size
        ):
            raise ValueError(
                f"num_prompts_per_step={exp.num_prompts_per_step} must be a multiple "
                f"of the data-parallel size ({data_parallel_size})."
            )
        if self.auto_requeue and not exp.checkpoint_dir:
            raise ValueError("auto_requeue requires checkpoint_dir")

    def algorithm_section(self, exp: NemoRLExperiment) -> dict[str, Any]:
        section: dict[str, Any] = {"max_num_steps": exp.max_num_steps}
        if self.generates(exp):
            section["num_prompts_per_step"] = exp.num_prompts_per_step
            section["num_generations_per_prompt"] = exp.num_generations_per_prompt
        return section

    def checkpointing_section(self, exp: NemoRLExperiment) -> dict[str, Any]:
        if not exp.checkpoint_dir:
            return {"enabled": False}
        return {
            "enabled": True,
            "checkpoint_dir": exp.checkpoint_dir,
            "save_period": exp.save_period,
            "keep_top_k": exp.keep_top_k,
        }

    def optimizer_section(self, exp: NemoRLExperiment) -> dict[str, Any]:
        """Passed through verbatim to Megatron's OptimizerConfig by NeMo-RL."""
        return {
            "optimizer": exp.optimizer,
            "lr": exp.lr,
            "min_lr": exp.min_lr,
            "weight_decay": exp.weight_decay,
            **exp.megatron_optimizer,
        }

    def scheduler_section(self, exp: NemoRLExperiment) -> dict[str, Any]:
        return {
            "lr_decay_style": exp.lr_decay_style,
            "lr_decay_iters": exp.lr_decay_iters,
            "lr_warmup_iters": exp.lr_warmup_iters,
            "lr_warmup_init": exp.lr_warmup_init,
            "start_weight_decay": exp.weight_decay,
            "end_weight_decay": exp.weight_decay,
        }

    def megatron_section(self, exp: NemoRLExperiment) -> dict[str, Any]:
        return {
            "enabled": True,
            "tensor_model_parallel_size": exp.tensor_model_parallel_size,
            "pipeline_model_parallel_size": exp.pipeline_model_parallel_size,
            "expert_model_parallel_size": exp.expert_model_parallel_size,
            "expert_tensor_parallel_size": exp.expert_tensor_parallel_size,
            "context_parallel_size": exp.context_parallel_size,
            "optimizer": self.optimizer_section(exp),
            "scheduler": self.scheduler_section(exp),
        }

    def generation_section(self, exp: NemoRLExperiment) -> dict[str, Any]:
        return {
            "backend": exp.generation_backend,
            "temperature": exp.temperature,
            "top_p": exp.top_p,
            "top_k": exp.top_k,
            "max_new_tokens": exp.max_new_tokens,
            "stop_strings": list(exp.stop_strings) or None,
            "mcore_generation_config": {
                "cuda_graph_impl": exp.cuda_graph_impl,
                "num_cuda_graphs": exp.num_cuda_graphs,
                "max_model_len": exp.max_total_sequence_length,
                # Gym rollouts reach the policy over this HTTP endpoint.
                "expose_http_server": bool(exp.gym_config_paths),
            },
        }

    def policy_section(self, exp: NemoRLExperiment) -> dict[str, Any]:
        policy: dict[str, Any] = {
            "model_name": exp.hf_config_dir,
            "tokenizer": {
                "name": exp.tokenizer_dir or exp.hf_config_dir,
                "chat_template": exp.chat_template or None,
            },
            "max_total_sequence_length": exp.max_total_sequence_length,
            # NeMo-RL base recipes may enable DTensor. This backend always uses
            # Megatron, and LMPolicy rejects configurations with both enabled.
            "dtensor_cfg": {"enabled": False},
            "megatron_cfg": self.megatron_section(exp),
        }
        # NeMo-RL branches on the key's presence: a `pretrained_checkpoint` block sends
        # it to the Megatron loader, so emitting an empty one makes it resolve `''` as
        # a checkpoint root instead of importing the HF model named by `model_name`.
        if exp.pretrained_checkpoint_path:
            policy["pretrained_checkpoint"] = {
                "format": exp.pretrained_checkpoint_format,
                "path": exp.pretrained_checkpoint_path,
            }
        if self.generates(exp):
            policy["generation"] = self.generation_section(exp)
        return policy

    def data_section(self, exp: NemoRLExperiment) -> dict[str, Any]:
        if exp.gym_config_paths:
            # Gym owns the dataset format and the reward, so the dataset and env are
            # replaced wholesale. dataset_name must be repeated on train/validation:
            # data.default is only a fallback, so the base recipe would otherwise win.
            return {
                "max_input_seq_length": exp.max_input_seq_length,
                "train": {"dataset_name": "NemoGymDataset", "data_path": exp.data_path},
                "validation": (
                    {
                        "dataset_name": "NemoGymDataset",
                        "data_path": exp.validation_data_path,
                    }
                    if exp.validation_data_path
                    else None
                ),
                "default": {
                    "dataset_name": "NemoGymDataset",
                    "env_name": "nemo_gym",
                    "processor": "nemo_gym_data_processor",
                },
            }
        data: dict[str, Any] = {
            "max_input_seq_length": exp.max_input_seq_length,
            "train": {"dataset_name": exp.dataset_name},
            "default": {"env_name": exp.env_name},
        }
        if exp.data_path:
            data["train"]["data_path"] = exp.data_path
        if exp.prompt_file:
            data["default"]["prompt_file"] = exp.prompt_file
        return data

    def env_section(self, exp: NemoRLExperiment) -> dict[str, Any]:
        return {
            "should_use_nemo_gym": True,
            "should_log_nemo_gym_responses": exp.gym_skip_response_dump,
            "should_mask_flagged_samples": exp.gym_mask_flagged_samples,
            "nemo_gym": {
                "port_range_low": exp.gym_port_range[0],
                "port_range_high": exp.gym_port_range[1],
                # tools/gym_data prebuilds these; do not resolve deps every run.
                "skip_venv_if_present": True,
                "config_paths": list(exp.gym_config_paths),
            },
        }

    def logger_section(self, exp: NemoRLExperiment) -> dict[str, Any]:
        return {
            "log_dir": exp.log_dir,
            "wandb_enabled": exp.wandb_enabled,
            "tensorboard_enabled": exp.tensorboard_enabled,
        }

    def cluster_section(self, exp: NemoRLExperiment) -> dict[str, Any]:
        return {"gpus_per_node": self.gpus_per_node, "num_nodes": self.nodes}

    def build_recipe(self, exp: NemoRLExperiment) -> dict[str, Any]:
        """Assemble the recipe from its sections, then apply the experiment's overrides."""
        recipe: dict[str, Any] = {
            exp.algorithm: self.algorithm_section(exp),
            "checkpointing": self.checkpointing_section(exp),
            "policy": self.policy_section(exp),
            "data": self.data_section(exp),
            "logger": self.logger_section(exp),
            "cluster": self.cluster_section(exp),
        }
        if exp.gym_config_paths:
            recipe["env"] = self.env_section(exp)
        if exp.base_config:
            # NeMo-RL pops `defaults` and merges the parent underneath this file
            # (utils/config.py). Without it the recipe must be complete on its own.
            recipe = {"defaults": exp.base_config, **recipe}
        return _deep_merge(recipe, exp.recipe_overrides)

    def render_recipe(self, exp: NemoRLExperiment, output_dir: Path) -> Path:
        """Write the Hydra recipe for one experiment and return its path."""
        self.validate(exp)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{exp.name}.yaml"
        path.write_text(yaml.safe_dump(self.build_recipe(exp), sort_keys=False, width=10**6))
        return path

    def _entrypoint(self, exp: NemoRLExperiment) -> str:
        """The script the driver runs, relative to nemo_rl_path unless absolute."""
        if exp.entrypoint:
            return exp.entrypoint
        if exp.gym_config_paths:
            # NOT examples/run_<algo>.py: that builds task_to_env keyed by the
            # DATASET's task_name, while the Gym rollout path looks up the fixed key
            # "nemo_gym" (rollouts.py) -> KeyError. The Gym runner binds
            # task_to_env = {"nemo_gym": ...}.
            return f"examples/nemo_gym/run_{exp.algorithm}_nemo_gym.py"
        return f"examples/run_{exp.algorithm}.py"

    # ---------- launch script ----------
    def render(self, exp: NemoRLExperiment, output_dir: Path) -> Path:
        # srun runs these with --container-workdir set to the NeMo-RL checkout, so a
        # relative path resolves against the wrong directory inside the container.
        output_dir = Path(output_dir).resolve()
        recipe_path = self.render_recipe(exp, output_dir)
        env = Environment(
            loader=FileSystemLoader([str(_TEMPLATE_DIR), str(_SHARED_TEMPLATE_DIR)]),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
        )
        body = env.get_template("nemorl.sh.j2").render(
            overlay_paths=list(exp.overlay_paths),
            overlay_bin_paths=[f"{p}/bin" for p in exp.overlay_paths if (Path(p) / "bin").is_dir()],
            nemo_rl_path=exp.nemo_rl_path,
            bridge_src_path=exp.bridge_src_path,
            megatron_path=exp.megatron_path,
            py_executables_system=exp.py_executables_system,
            install_commands=exp.install_commands,
            hf_home=self.hf_home,
            env_vars={**self.env_vars, **exp.env_vars},
            ray_tmpdir=self.ray_tmpdir,
            ray_port=self.ray_port,
            gpus_per_node=self.gpus_per_node,
            cpus_per_task=self.cpus_per_task,
            entrypoint=self._entrypoint(exp),
            recipe_path=str(recipe_path),
            gym_config_paths=list(exp.gym_config_paths),
            gym_home=exp.gym_home,
            gym_scratch_root=exp.gym_scratch_root,
            gym_venv_dir=exp.gym_venv_dir,
            gym_uv_cache_dir=exp.gym_uv_cache_dir,
            nemo_rl_venv_dir=exp.nemo_rl_venv_dir,
        )
        path = output_dir / f"{exp.name}.sh"
        path.write_text(body)
        path.chmod(0o755)

        # The body must run inside the container, so the Slurm/container boundary lives
        # in a wrapper that srun's into it -- matching slurm_megatron/slurm_tasks.sh.j2.
        log_dir = Path(self.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        wrapper = env.get_template("nemorl_sbatch.sh.j2").render(
            body_path=str(path),
            job_name=exp.name,
            account=self.account,
            partition=self.partition,
            reservation=self.reservation,
            nodes=self.nodes,
            cpus_per_task=self.cpus_per_task,
            gpus_per_node=self.gpus_per_node,
            run_time=self.run_time,
            log_dir=str(log_dir),
            ray_port=self.ray_port,
            srun_extra_args=self.srun_extra_args,
            srun_export_vars=",".join(
                dict.fromkeys(_SRUN_INFRA_EXPORTS + list({**self.env_vars, **exp.env_vars}))
            ),
            vetnode=self.vetnode,
            vetnode_config=self.vetnode_config or str(_SHARED_TEMPLATE_DIR / "vetnode-config.yaml"),
            vetnode_install=self.vetnode_install,
            vetnode_skip_install=self.vetnode_skip_install,
            vetnode_verbose=self.vetnode_verbose,
            vetnode_numa_bind=self.vetnode_numa_bind,
            vetnode_exclude=self.vetnode_exclude,
            vetnode_max_excluded_nodes=self.vetnode_max_excluded_nodes,
            vetnode_key=hashlib.sha256(exp.name.encode()).hexdigest()[:16],
            container_edf=self.container,
            container=self.container,
            container_mounts=self.container_mounts,
            nemo_rl_path=exp.nemo_rl_path,
            auto_requeue=self.auto_requeue,
            auto_requeue_stop_mode="checkpoint_steps",
            auto_requeue_checkpoint_dir=exp.checkpoint_dir,
            auto_requeue_target_steps=exp.max_num_steps,
        )
        wrapper_path = output_dir / f"{exp.name}.sbatch"
        wrapper_path.write_text(wrapper)
        wrapper_path.chmod(0o755)
        return path

    def render_all(
        self,
        name: str,
        experiments: list[NemoRLExperiment],
        output_dir: str = "rendered",
    ) -> list[str]:
        out = Path(output_dir) / name
        return [str(self.render(e, out)) for e in experiments]

    def submit_all(
        self,
        name: str,
        experiments: list[NemoRLExperiment],
        output_dir: str = "rendered",
    ) -> list[str]:
        results = []
        for script in self.render_all(name, experiments, output_dir=output_dir):
            results.append(self._launch(script))
        return results

    def _launch(self, script: str) -> str:
        if self.srun_job_id:
            # Run the body directly inside an existing allocation. The container and
            # mounts go here because there is no wrapper in this mode.
            cmd = [
                "srun",
                "--jobid",
                self.srun_job_id,
                "--overlap",
                "--nodes=1",
                "--ntasks=1",
                f"--cpus-per-task={self.cpus_per_task}",
                f"--gres=gpu:{self.gpus_per_node}",
                *self.srun_extra_args.split(),
                *(["--environment", self.container] if self.container else []),
                *([f"--container-mounts={self.container_mounts}"] if self.container_mounts else []),
                "bash",
                script,
                "standalone",
            ]
        else:
            cmd = ["sbatch", "--parsable", str(Path(script).with_suffix(".sbatch"))]
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


__all__ = ["SlurmNemoRLBackend"]
