# AGENTS.md

Guidance for coding agents working in this repository.

## Scope

These instructions apply to the whole repository.

## Project basics

- Python tooling is managed with uv.
- Primary workflows:
  - list: python main.py list experiments/<name>/experiment.py
  - render: python main.py render experiments/<name>/experiment.py
  - submit: python main.py submit experiments/<name>/experiment.py
- Type and lint checks:
  - uv run ty check
  - uv run ruff check .

## Code and config conventions

- Keep experiment definitions declarative in Python dataclasses.
- Prefer adding reusable behavior in spellbook/megatron or spellbook/backends over one-off script edits.
- Do not hand-edit generated files in sbatch_scripts/; regenerate with render.
- Keep environment defaults in experiment or backend env_vars using explicit setdefault patterns.
- Keep backend concerns in SlurmBackend (container options, srun args, backend env vars).

## Megatron args behavior

- Most fields map snake_case to --kebab-case.
- Keep flag/value pairs together on the same rendered line.
- Keep moe-layer-freq in compact list form without spaces, for example: [1,1,1].

## Documentation updates

When behavior changes, update README.md in the same change so user docs stay accurate.
