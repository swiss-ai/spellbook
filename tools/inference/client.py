"""Tiny standard-library client for an OpenAI-style inference server."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from typing import Any


def request(base_url: str, route: str, payload: dict[str, Any] | None = None) -> Any:
    """Send one request and return the decoded JSON response."""
    url = f"{base_url.rstrip('/')}{route}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(req) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise SystemExit(f"HTTP {error.code}: {detail}") from error
    return json.loads(body) if body else None


def _text(result: Any) -> str:
    """Extract assistant/completion text, falling back to formatted JSON."""
    try:
        choice = result["choices"][0]
        return choice.get("message", {}).get("content") or choice["text"]
    except (KeyError, IndexError, TypeError):
        return json.dumps(result, indent=2, ensure_ascii=False)


def interactive(
    base_url: str,
    *,
    chat: bool,
    system: str | None,
    max_tokens: int,
    temperature: float,
) -> None:
    """Run a persistent prompt loop; chat mode preserves conversation history."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    print("Enter /quit to exit or /clear to reset chat history.")
    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not prompt:
            continue
        if prompt == "/quit":
            return
        if prompt == "/clear":
            messages = messages[:1] if system else []
            print("History cleared.")
            continue
        if chat:
            messages.append({"role": "user", "content": prompt})
            result = request(
                base_url,
                "/v1/chat/completions",
                {
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            answer = _text(result)
            messages.append({"role": "assistant", "content": answer})
        else:
            result = request(
                base_url,
                "/v1/completions",
                {
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            answer = _text(result)
        print(answer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default=os.environ.get("INFERENCE_SERVER_URL", "http://localhost:5000")
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("health")

    completion = subparsers.add_parser("completion")
    completion.add_argument("prompt")
    completion.add_argument("--max-tokens", type=int, default=32)
    completion.add_argument("--temperature", type=float, default=0)
    completion.add_argument("--top-p", type=float)

    chat = subparsers.add_parser("chat")
    chat.add_argument("message")
    chat.add_argument("--system")
    chat.add_argument("--max-tokens", type=int, default=32)
    chat.add_argument("--temperature", type=float, default=0)
    chat.add_argument("--top-p", type=float)

    repl = subparsers.add_parser("interactive", aliases=["repl"])
    repl.add_argument("--chat", action="store_true", help="preserve chat history")
    repl.add_argument("--system")
    repl.add_argument("--max-tokens", type=int, default=128)
    repl.add_argument("--temperature", type=float, default=0)

    args = parser.parse_args()
    if args.command in ("interactive", "repl"):
        interactive(
            args.url,
            chat=args.chat,
            system=args.system,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        return
    if args.command == "health":
        result = request(args.url, "/v1/health")
    elif args.command == "completion":
        payload = {
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
        if args.top_p is not None:
            payload["top_p"] = args.top_p
        result = request(args.url, "/v1/completions", payload)
    else:
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.message})
        payload = {
            "messages": messages,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
        if args.top_p is not None:
            payload["top_p"] = args.top_p
        result = request(args.url, "/v1/chat/completions", payload)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
