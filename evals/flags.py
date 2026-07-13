"""Translate Python lm-eval option names into command-line arguments."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any


_SHELL_SAFE = re.compile(r"^[A-Za-z0-9_@%+=:,./-]+$")


def model_args(fields: Mapping[str, Any]) -> str:
    """Render lm-eval model arguments as a comma-separated key/value string."""
    return ",".join(
        f"{name}={value}"
        for name, value in fields.items()
        if value is not None and value is not False and value != ""
    )


def to_args(fields: Mapping[str, Any]) -> list[str]:
    """Translate ``field_name: value`` entries into lm-eval CLI tokens.

    lm-eval uses underscores in its long option names, so Python dictionary keys
    map directly to flags: ``write_out`` becomes ``--write_out``. True booleans
    become bare flags, false/empty values are omitted, lists become comma-separated
    values, and dictionaries become JSON values.
    """
    args: list[str] = []
    for field_name, value in fields.items():
        if value is None or value is False or value == "" or value == [] or value == {}:
            continue

        flag = f"--{field_name}"
        args.append(flag)
        if value is True:
            continue
        if isinstance(value, list):
            value = ",".join(str(item) for item in value)
        elif isinstance(value, dict):
            value = json.dumps(value, sort_keys=True)
        args.append(str(value))
    return args


def to_shell_lines(fields: Mapping[str, Any]) -> list[str]:
    """Return one shell line per flag/value pair for the nested eval shell."""
    args = to_args(fields)
    lines: list[str] = []
    index = 0
    while index < len(args):
        flag = args[index]
        if index + 1 == len(args) or args[index + 1].startswith("--"):
            lines.append(flag)
            index += 1
            continue
        lines.append(f"{flag} {_quote_value(args[index + 1])}")
        index += 2
    return [_escape_outer_single_quotes(line) for line in lines]


def _quote_value(value: str) -> str:
    if _SHELL_SAFE.fullmatch(value):
        return value
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("$", "\\$")
        .replace("`", "\\`")
    )
    return f'"{escaped}"'


def _escape_outer_single_quotes(value: str) -> str:
    # The generated command is embedded in bash -lc '...'. Preserve apostrophes
    # in option values without terminating that outer single-quoted string.
    return value.replace("'", "'\"'\"'")
