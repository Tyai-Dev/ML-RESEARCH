"""Blackboard's assistant: a streaming Claude chat with board context.

The browser keeps the conversation history and sends it whole on each turn
(the API is stateless); optionally it attaches the current board — notebook
cells and tex — which lands after the cached system prompt so the stable
prefix keeps its prompt-cache entry.

Credentials resolve the standard way (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,
or an `ant auth login` profile); nothing is stored here.
"""

import os
from typing import Iterator

import anthropic

MODEL = os.environ.get("BLACKBOARD_MODEL", "claude-opus-5")

_SYSTEM = """\
You are the assistant inside Blackboard, a personal research studio: a \
three-pane local app with the user's files, an executable Python notebook \
(persistent kernel, numpy and matplotlib preloaded as np/plt), and a LaTeX \
"theory board" compiled with pdflatex (article class; amsmath/amssymb/amsthm/\
graphicx/geometry available).

You help write and improve Python code, LaTeX, and research notes. The user \
can insert any fenced code block from your replies directly into the notebook \
with one click, so put runnable code in ```python blocks and LaTeX in \
```latex blocks, one logical unit per block.

Keep responses focused, brief, and concise. Lead with the answer or the code; \
explain after, only as much as the change needs. When the current board \
content is attached, ground your answers in it — refer to the user's actual \
variables, cells, and tex sections rather than inventing parallel examples.\
"""


class ChatError(Exception):
    """Raised for failures the UI should show as a friendly message."""


def stream_chat(messages: list[dict], context: str | None = None) -> Iterator[str]:
    """Yield response text chunks; raises ChatError before the first chunk
    for auth/config problems so the server can return a proper status."""
    system: list[dict] = [
        {"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}
    ]
    if context:
        system.append({"type": "text", "text": f"Current board:\n\n{context}"})

    client = anthropic.Anthropic()
    try:
        manager = client.messages.stream(
            model=MODEL,
            max_tokens=64000,
            system=system,
            messages=messages,
        )
        stream = manager.__enter__()  # request is sent here; auth errors raise now
    except (anthropic.AuthenticationError, TypeError) as exc:
        # TypeError: the SDK found no credential source at all (raised at
        # request-build time); AuthenticationError: a key was found but rejected.
        raise ChatError(
            "No Claude API credentials. Set ANTHROPIC_API_KEY (or run "
            "`ant auth login`) in the environment that launches blackboard."
        ) from exc
    except anthropic.APIConnectionError as exc:
        raise ChatError("Could not reach the Claude API — check your network.") from exc
    except anthropic.APIStatusError as exc:
        raise ChatError(f"Claude API error ({exc.status_code}): {exc.message}") from exc

    def generate() -> Iterator[str]:
        try:
            for text in stream.text_stream:
                yield text
            final = stream.get_final_message()
            if final.stop_reason == "refusal":
                yield "\n\n*[the assistant declined this request]*"
            elif final.stop_reason == "max_tokens":
                yield "\n\n*[response hit the length limit]*"
        finally:
            manager.__exit__(None, None, None)

    return generate()
