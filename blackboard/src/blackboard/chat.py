"""Blackboard's assistant: streaming chat over Claude, OpenAI, or Gemini.

The browser keeps the conversation history and sends it whole on each turn;
optionally it attaches the current board (notebook cells + tex) as context.
The provider is chosen per message from the UI.

Keys come from the environment — Blackboard loads a `.env` file at startup
(see server.main), so put them there:

    ANTHROPIC_API_KEY=...   -> provider "claude"   (platform.claude.com)
    OPENAI_API_KEY=...      -> provider "openai"   (platform.openai.com)
    GEMINI_API_KEY=...      -> provider "gemini"   (aistudio.google.com)
"""

import os
from typing import Iterator

MODELS = {
    "claude": os.environ.get("BLACKBOARD_CLAUDE_MODEL", "claude-opus-5"),
    "openai": os.environ.get("BLACKBOARD_OPENAI_MODEL", "gpt-5"),
    "gemini": os.environ.get("BLACKBOARD_GEMINI_MODEL", "gemini-2.5-pro"),
}

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

_KEY_HINTS = {
    "claude": "Set ANTHROPIC_API_KEY in the .env file (create a key at platform.claude.com -> Settings -> API keys).",
    "openai": "Set OPENAI_API_KEY in the .env file (create a key at platform.openai.com/api-keys).",
    "gemini": "Set GEMINI_API_KEY in the .env file (create a key at aistudio.google.com/apikey).",
}


class ChatError(Exception):
    """Raised for failures the UI should show as a friendly message."""


def _no_key(provider: str) -> ChatError:
    return ChatError(f"No {provider} API credentials. {_KEY_HINTS[provider]}")


def _system_text(context: str | None) -> str:
    return _SYSTEM + (f"\n\nCurrent board:\n\n{context}" if context else "")


# ---------------------------------------------------------------- claude

def _stream_claude(messages: list[dict], context: str | None) -> Iterator[str]:
    import anthropic

    # static prompt first + cache breakpoint, volatile board context after
    system: list[dict] = [
        {"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}
    ]
    if context:
        system.append({"type": "text", "text": f"Current board:\n\n{context}"})

    client = anthropic.Anthropic()
    try:
        manager = client.messages.stream(
            model=MODELS["claude"], max_tokens=64000, system=system, messages=messages
        )
        stream = manager.__enter__()  # request sent here; auth errors raise now
    except (anthropic.AuthenticationError, TypeError) as exc:
        raise _no_key("claude") from exc
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


# ---------------------------------------------------------------- openai

def _stream_openai(messages: list[dict], context: str | None) -> Iterator[str]:
    import openai

    if not os.environ.get("OPENAI_API_KEY"):
        raise _no_key("openai")
    client = openai.OpenAI()
    try:
        stream = client.chat.completions.create(
            model=MODELS["openai"],
            messages=[{"role": "system", "content": _system_text(context)}, *messages],
            stream=True,
        )
    except openai.AuthenticationError as exc:
        raise _no_key("openai") from exc
    except openai.APIConnectionError as exc:
        raise ChatError("Could not reach the OpenAI API — check your network.") from exc
    except openai.APIStatusError as exc:
        raise ChatError(f"OpenAI API error ({exc.status_code}): {exc.message}") from exc

    def generate() -> Iterator[str]:
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return generate()


# ---------------------------------------------------------------- gemini

def _stream_gemini(messages: list[dict], context: str | None) -> Iterator[str]:
    from google import genai
    from google.genai import errors as genai_errors

    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise _no_key("gemini")
    client = genai.Client()
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user",
         "parts": [{"text": m["content"]}]}
        for m in messages
    ]
    try:
        stream = client.models.generate_content_stream(
            model=MODELS["gemini"],
            contents=contents,
            config={"system_instruction": _system_text(context)},
        )
        first = next(stream, None)  # force the request so auth errors raise here
    except genai_errors.APIError as exc:
        if exc.code in (401, 403):
            raise _no_key("gemini") from exc
        raise ChatError(f"Gemini API error ({exc.code}): {exc.message}") from exc

    def generate() -> Iterator[str]:
        try:
            if first is not None and first.text:
                yield first.text
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except genai_errors.APIError as exc:
            yield f"\n\n*[Gemini stream error: {exc.message}]*"

    return generate()


_PROVIDERS = {
    "claude": _stream_claude,
    "openai": _stream_openai,
    "gemini": _stream_gemini,
}


def stream_chat(
    messages: list[dict], context: str | None = None, provider: str = "claude"
) -> Iterator[str]:
    """Yield response text chunks; raises ChatError before the first chunk
    for auth/config problems so the server can return a proper status."""
    if provider not in _PROVIDERS:
        raise ChatError(f"unknown provider {provider!r}; use one of {sorted(_PROVIDERS)}")
    return _PROVIDERS[provider](messages, context)
