"""
Config layer: YAML loading, dict merging, template resolution.

Merge order (lowest → highest priority):
    runtime preset YAML  (container EDF, megatron_path, mcore version)
      → model preset YAML  (architecture flags, MODEL_ARGS)
        → chain/sweep base kwargs
          → per-Step overrides
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on conflict."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def merge_configs(*layers: dict) -> dict:
    """Merge an ordered sequence of config dicts. Later layers win."""
    result: dict = {}
    for layer in layers:
        result = _deep_merge(result, layer)
    return result


def load_yaml(path: Path | str) -> dict:
    """Load a YAML file and return as a dict."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


# ---------------------------------------------------------------------------
# Template resolution
# ---------------------------------------------------------------------------

def resolve_templates(cfg: dict, context: dict[str, Any], strict: bool = True) -> dict:
    """
    Walk every string value in cfg and expand ${VAR} markers using context.

    - strict=True (default): unknown variables raise KeyError.
    - strict=False: unknown variables are left as the original ${VAR} string.
    - Non-string values (int, float, bool, None) pass through unchanged.
    - Nested dicts are resolved recursively.
    """
    return _resolve_node(cfg, context, strict)


def _resolve_node(node: Any, ctx: dict[str, Any], strict: bool = True) -> Any:
    if isinstance(node, dict):
        return {k: _resolve_node(v, ctx, strict) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_node(item, ctx, strict) for item in node]
    if isinstance(node, str):
        return _resolve_str(node, ctx, strict)
    return node


_TMPL_RE = re.compile(r"\$\{(\w+)\}")


def _resolve_str(s: str, ctx: dict[str, Any], strict: bool = True) -> Any:
    """
    Expand ${VAR} in s.  If the entire string is a single ${VAR} token,
    return the raw Python value (preserves int/bool/etc. from context).
    """
    tokens = _TMPL_RE.findall(s)
    if not tokens:
        return s

    # Single bare token like "${VAR}" → return the raw value (preserves type)
    if s.strip() == f"${{{tokens[0]}}}" and len(tokens) == 1:
        key = tokens[0]
        if key not in ctx:
            if strict:
                raise KeyError(f"Template variable '${{{key}}}' not found in context")
            return s  # leave as-is
        return ctx[key]

    # Mixed string like "TP${TP}PP${PP}" → always a string
    def _sub(m: re.Match) -> str:
        key = m.group(1)
        if key not in ctx:
            if strict:
                raise KeyError(f"Template variable '${{{key}}}' not found in context")
            return m.group(0)  # leave as-is
        return str(ctx[key])

    return _TMPL_RE.sub(_sub, s)


# ---------------------------------------------------------------------------
# Preset loading
# ---------------------------------------------------------------------------

_PRESETS_DIR = Path(__file__).parent / "presets"


def load_runtime_preset(name: str) -> dict:
    """Load zoo/presets/runtimes/<name>.yaml"""
    path = _PRESETS_DIR / "runtimes" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Runtime preset not found: {path}")
    return load_yaml(path)


def load_model_preset(name: str) -> dict:
    """Load zoo/presets/models/<name>.yaml"""
    path = _PRESETS_DIR / "models" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Model preset not found: {path}")
    return load_yaml(path)


# ---------------------------------------------------------------------------
# High-level: build a fully resolved config for a job
# ---------------------------------------------------------------------------

def build_config(
    model: str,
    runtime: str | None = None,
    strict: bool = True,
    **kwargs: Any,
) -> dict:
    """
    Produce a fully resolved config dict.

    Merge order (lowest → highest priority):
      1. Runtime preset YAML  (container_edf, megatron_path, mcore version)
      2. Model preset YAML    (MODEL_ARGS architecture flags)
      3. User kwargs          (tp, ep, mbs, data_path, …)

    Template markers (``${var}``) in MODEL_ARGS / ENV_VARS are resolved
    using the merged flat scalar values — no shell envsubst needed.

    Returns:
        {
            "MODEL_ARGS":      {...},   # consumed by MegatronBackend
            "ENV_VARS":        {...},   # consumed by SlurmBackend
            "EXPERIMENT_META": {...},   # consumed by chain/sweep
            # plus all flat scalars (tp, ep, megatron_path, …) at top level
        }
    """
    layers: list[dict] = []

    if runtime is not None:
        layers.append(load_runtime_preset(runtime))

    layers.append(load_model_preset(model))
    layers.append(kwargs)

    merged = merge_configs(*layers)

    # Flat scalar context for template resolution
    flat_ctx: dict[str, Any] = {k: v for k, v in merged.items() if not isinstance(v, dict)}
    flat_ctx.update(kwargs)  # kwargs always win

    # Promote flat kwargs into MODEL_ARGS so MegatronBackend.to_args() sees them.
    # This means any alias (tp, ep, override_opt_param_scheduler, …) passed as a
    # kwarg to build_config() overrides the model preset value in MODEL_ARGS.
    flat_kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, dict)}
    merged_model_args = _deep_merge(merged.get("MODEL_ARGS", {}), flat_kwargs)

    # Resolve templates in each namespace
    result: dict = dict(merged)  # preserve flat scalars at top level
    result["MODEL_ARGS"] = resolve_templates(merged_model_args, flat_ctx, strict=strict)
    for namespace in ("ENV_VARS", "EXPERIMENT_META"):
        ns_data = merged.get(namespace, {})
        result[namespace] = resolve_templates(ns_data, flat_ctx, strict=strict)

    return result
