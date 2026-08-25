# Inference

Slurm launcher for Megatron-LM's dynamic text-generation HTTP server. Model- and checkpoint-specific configs should live in the repository that owns those artifacts. The launcher is backend-specific while `tools.inference.client` targets the common HTTP API, leaving room for a future vLLM launcher without changing the interactive client.

The launcher uses one Slurm task per GPU, not `torchrun`. Every task starts one Megatron process using `SLURM_PROCID`, `SLURM_LOCALID`, and `SLURM_NTASKS`. Quart and Hypercorn are installed once per node inside the containerized `srun` step before the server starts.

A complete placeholder configuration is available at [`examples/megatron_server.py`](examples/megatron_server.py). Replace its paths and Slurm settings, then inspect or submit it with:

```bash
python -m tools.inference.examples.megatron_server render
python -m tools.inference.examples.megatron_server submit
```

Minimal configuration shape:

```python
from tools.inference.megatron_server import MegatronServerConfig, submit

submit(MegatronServerConfig(
    name="my-model-step-3000",
    checkpoint="/path/to/checkpoint/root",
    ckpt_step=3000,
    tokenizer_model="/path/to/tokenizer",
    megatron_path="/path/to/Megatron-LM",
    tensor_parallel_size=1,
    expert_parallel_size=8,
    nodes=2,
    gpus_per_node=4,
    megatron_args={
        "moe_token_dispatcher_type": "alltoall",
        "moe_router_load_balancing_type": "quantile_balancing",
        "moe_router_quantile_balancing_method": "histogram",
        "attention_output_gate": True,
    },
    account="infra01",
    partition="normal",
    container_edf="apertus2-alps4-temp",
    container_mounts="${SCRATCH}:${SCRATCH},${HOME}:${HOME},/capstor:/capstor,/iopsstor:/iopsstor",
    srun_extra_args="--network=disable_rdzv_get",
))
```

Use `render(cfg)` to inspect the sbatch script without submitting it. `nodes * gpus_per_node` determines the total Slurm task/world count; configure tensor, pipeline, and expert parallelism so the model is actually sharded across that world. Keep `host=None` (the default) for multi-node jobs: Megatron then advertises each compute node's routable hostname while its rank-0 HTTP frontend still binds to all interfaces.

`megatron_args` accepts new Megatron options without changes to Spellbook: keys use the same `snake_case` to `--kebab-case` conversion as experiment fields, `True` emits a bare flag, and `None`/`False` omit it.

The job log prints its compute hostname. A small standard-library client is included, so no Python dependencies or hand-written curl payloads are needed:

```bash
export INFERENCE_SERVER_URL="http://${COMPUTE_HOST}:5000"
python -m tools.inference.client health
python -m tools.inference.client interactive --max-tokens 128

# Or preserve message history when the tokenizer has a chat template:
python -m tools.inference.client interactive --chat --system "You are a helpful assistant."

# One-shot requests remain available:
python -m tools.inference.client completion "The capital of Switzerland is" --max-tokens 32
python -m tools.inference.client chat "What is the capital of Switzerland?" --max-tokens 32
```

The same requests can be sent with curl from a cluster login node:

```bash
curl "http://${COMPUTE_HOST}:5000/v1/health"
curl "http://${COMPUTE_HOST}:5000/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of Switzerland is","max_tokens":32,"temperature":0}'
```

For laptop access, tunnel through a login node:

```bash
ssh -L 5000:${COMPUTE_HOST}:5000 USER@LOGIN_HOST
curl http://localhost:5000/v1/health
```

The API has `GET /v1/health`, `POST /v1/completions`, and `POST /v1/chat/completions`. It is only approximately OpenAI-compatible and has no built-in authentication. Do not expose it beyond trusted cluster networks.
