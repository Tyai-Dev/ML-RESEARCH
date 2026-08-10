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

_TOOLS_ADDENDUM = """

You can act on the workspace directly with tools: read_file, write_file, \
add_notebook_cell, edit_notebook_cell. When the user asks you to add cells, \
change the notebook, or write/extend a tex file, USE THE TOOLS to apply the \
change to the actual files — the UI reloads them automatically, so never \
print the content into the chat instead of applying it. The attached board \
labels notebook cells 'cell N'; tool indices are 0-based, so that is index \
N-1. write_file overwrites whole files: read or reuse the attached content \
and write the complete result. Use plain code blocks only when the user \
wants to discuss code without applying it.\
"""

_MAX_TOOL_ROUNDS = 8


def _stream_claude(
    messages: list[dict], context: str | None, root=None
) -> Iterator[str]:
    import anthropic

    from blackboard import tools as bb_tools

    # static prompt first + cache breakpoint, volatile board context after
    system: list[dict] = [
        {
            "type": "text",
            "text": _SYSTEM + (_TOOLS_ADDENDUM if root else ""),
            "cache_control": {"type": "ephemeral"},
        }
    ]
    if context:
        system.append({"type": "text", "text": f"Current board:\n\n{context}"})

    convo = [dict(m) for m in messages]
    request = dict(
        model=MODELS["claude"],
        max_tokens=64000,
        system=system,
        tools=bb_tools.DEFINITIONS if root else [],
    )

    client = anthropic.Anthropic()
    try:
        manager = client.messages.stream(messages=convo, **request)
        stream = manager.__enter__()  # request sent here; auth errors raise now
    except (anthropic.AuthenticationError, TypeError) as exc:
        raise _no_key("claude") from exc
    except anthropic.APIConnectionError as exc:
        raise ChatError("Could not reach the Claude API — check your network.") from exc
    except anthropic.APIStatusError as exc:
        raise ChatError(f"Claude API error ({exc.status_code}): {exc.message}") from exc

    def generate() -> Iterator[str]:
        nonlocal manager, stream
        for _ in range(_MAX_TOOL_ROUNDS):
            try:
                for text in stream.text_stream:
                    yield text
                final = stream.get_final_message()
            finally:
                manager.__exit__(None, None, None)

            if final.stop_reason != "tool_use":
                if final.stop_reason == "refusal":
                    yield "\n\n*[the assistant declined this request]*"
                elif final.stop_reason == "max_tokens":
                    yield "\n\n*[response hit the length limit]*"
                return

            convo.append({"role": "assistant", "content": final.content})
            results = []
            for block in final.content:
                if block.type != "tool_use":
                    continue
                out, is_error = bb_tools.execute(root, block.name, block.input)
                yield f"\n⚙ {block.name} → {out}\n"
                result = {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": out,
                }
                if is_error:
                    result["is_error"] = True
                results.append(result)
            convo.append({"role": "user", "content": results})

            try:
                manager = client.messages.stream(messages=convo, **request)
                stream = manager.__enter__()
            except Exception as exc:  # mid-stream errors surface as text
                yield f"\n\n*[assistant error: {exc}]*"
                return
        yield "\n\n*[stopped after too many tool rounds]*"

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


def stream_chat(
    messages: list[dict],
    context: str | None = None,
    provider: str = "claude",
    root=None,
) -> Iterator[str]:
    """Yield response text chunks; raises ChatError before the first chunk
    for auth/config problems so the server can return a proper status.

    ``root`` (a workspace Path) enables file-editing tools — currently on
    the claude provider; openai/gemini are chat-only.
    """
    if provider == "claude":
        return _stream_claude(messages, context, root=root)
    if provider == "openai":
        return _stream_openai(messages, context)
    if provider == "gemini":
        return _stream_gemini(messages, context)
    raise ChatError(
        f"unknown provider {provider!r}; use one of ['claude', 'gemini', 'openai']"
    )
