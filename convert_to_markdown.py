#!/usr/bin/env python3
"""Convert captured coding-agent <-> LLM wire payloads into readable Markdown.

The input files in this directory are UTF-8 text with a misleading ``.bin``
extension. Each one is either a JSON request body or a text/event-stream (SSE)
response, produced by one of three clients talking to MiMo endpoints:

    Claude Code -> Anthropic Messages API     (POST /v1/messages)
    Codex CLI   -> OpenAI Responses API       (POST /v1/responses)
    OpenCode    -> OpenAI Chat Completions    (POST /v1/chat/completions)

Direction is decided from the *content*, never from the filename: at least one
capture in this dataset is named backwards (see README / CLAUDE.md).

Usage:
    python3 convert_to_markdown.py               # write md/
    python3 convert_to_markdown.py --dry-run     # classify only, no writes
    python3 convert_to_markdown.py --verify      # assert no payload text is lost
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

ANTHROPIC = "Anthropic Messages API"
RESPONSES = "OpenAI Responses API"
CHAT = "OpenAI Chat Completions API"
UNKNOWN = "Unknown"

SSE = "sse"
JSON_KIND = "json"
TEXT_KIND = "text"

REQUEST = "request"
RESPONSE = "response"

#: Prose longer than this goes inside a <details> block (content is never cut).
COLLAPSE_OVER = 400
#: JSON schemas longer than this go inside a <details> block as well.
JSON_COLLAPSE_OVER = 3000

#: Extensions scanned for captures. Deliberately excludes .md so that CLAUDE.md
#: and other documentation are never mistaken for a captured payload.
SOURCE_SUFFIXES = {".bin", ".json", ".txt", ".log", ".sse"}

#: Responses-API delta events whose payload is text accumulated per output item.
RESPONSES_TEXT_DELTAS = {
    "response.output_text.delta",
    "response.reasoning_text.delta",
    "response.reasoning_summary_text.delta",
}
#: Responses-API delta events whose payload is (possibly partial) JSON arguments.
RESPONSES_ARG_DELTAS = {
    "response.function_call_arguments.delta",
    "response.custom_tool_call_input.delta",
}

PROTOCOL_SHORT = {
    ANTHROPIC: "Anthropic Messages",
    RESPONSES: "OpenAI Responses",
    CHAT: "OpenAI Chat Completions",
}

# --------------------------------------------------------------------------
# Small Markdown helpers
# --------------------------------------------------------------------------


def fence(text: str, lang: str = "text") -> str:
    """Fenced code block whose fence is longer than any backtick run inside.

    Tool descriptions and system prompts in this dataset contain ``` fences of
    their own; a fixed 3-backtick fence would let them escape the block.
    """
    text = "" if text is None else str(text)
    if text and not text.endswith("\n"):
        text += "\n"
    longest = max([0] + [len(r) for r in re.findall(r"`+", text)])
    bar = "`" * max(3, longest + 1)
    return f"{bar}{lang}\n{text}{bar}\n"


def details(summary: str, body: str) -> str:
    """Collapsible block. Renders as normal Markdown on GitHub if spaced right."""
    body = body.rstrip("\n")
    return f"<details>\n<summary>{summary}</summary>\n\n{body}\n\n</details>\n"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(v: Any) -> str:
        s = "" if v is None else str(v)
        return s.replace("|", "\\|").replace("\n", "<br>").replace("\r", "")

    out = [
        "| " + " | ".join(cell(h) for h in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        out.append("| " + " | ".join(cell(c) for c in row) + " |")
    return "\n".join(out) + "\n"


def inline(v: Any, limit: int = 100) -> str:
    """Compact single-line rendering of any JSON value.

    Pipes are deliberately *not* escaped here: every value produced by this
    function is passed through ``table()``, which does the cell escaping.
    """
    if isinstance(v, (dict, list)):
        s = json.dumps(v, ensure_ascii=False, sort_keys=False)
    elif v is None:
        s = "`null`"
    elif isinstance(v, bool):
        s = "`true`" if v else "`false`"
    elif isinstance(v, (int, float)):
        s = f"`{v}`"
    else:
        s = str(v).replace("\n", " ").replace("\r", " ").replace("`", "'")
    return s if len(s) <= limit else f"{s[:limit]}… (+{len(s) - limit:,} chars)"


def preview(text: str, limit: int = 90) -> str:
    """Lossy one-line teaser for timeline tables (the Raw sections are exact)."""
    s = (text or "").replace("\r", "").replace("\n", "\\n").replace("`", "'")
    s = " ".join(s.split())
    return s if len(s) <= limit else s[:limit] + "…"


def chars(n: int) -> str:
    return f"{n:,}"


def iso(ts: Any) -> str:
    try:
        return _dt.datetime.fromtimestamp(float(ts), _dt.timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%SZ"
        )
    except Exception:
        return ""


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024 or unit == "MB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024.0
    return f"{n:,.1f} MB"


def parse_json_string(s: Any) -> Any | None:
    """Some fields (metadata.user_id, client_metadata["…"]) are JSON in a string."""
    if not isinstance(s, str):
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def text_block(text: str, lang: str = "text", label: str = "text") -> str:
    """Prose block: full content, collapsed only when long."""
    text = "" if text is None else str(text).replace("\r\n", "\n")
    body = fence(text, lang)
    if len(text) > COLLAPSE_OVER:
        first_line = text.strip().split("\n", 1)[0][:70]
        return details(f"{label} — {chars(len(text))} chars — {first_line}…", body)
    return body


def json_block(obj: Any, label: str | None = None, lang: str = "json") -> str:
    """JSON block: full content, collapsed only when very long."""
    body = fence(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False), lang)
    if label and len(body) > JSON_COLLAPSE_OVER:
        return details(f"{label} — {chars(len(body))} chars", body)
    return body


def usage_table(usage: dict) -> str:
    rows: list[list[Any]] = []
    for k, v in usage.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                rows.append([f"{k}.{k2}", f"`{v2:,}`" if isinstance(v2, int) else inline(v2)])
        elif isinstance(v, int):
            rows.append([k, f"`{v:,}`"])
        else:
            rows.append([k, inline(v)])
    return table(["usage field", "value"], rows)


def unknown_type_fallback(block: dict) -> str:
    return f"> Unrecognised block type `{block.get('type')}` — raw JSON:\n\n" + fence(
        json.dumps(block, ensure_ascii=False, indent=2)
    )


# --------------------------------------------------------------------------
# SSE parsing
# --------------------------------------------------------------------------


@dataclass
class SSEEvent:
    index: int
    event: str | None  # from an "event:" line, may be absent
    data: str  # raw data payload (joined)
    parsed: Any  # json.loads(data) when possible
    raw: str  # the original text of the frame

    @property
    def is_done(self) -> bool:
        return self.data.strip() == "[DONE]"

    @property
    def name(self) -> str:
        if self.event:
            return self.event
        if isinstance(self.parsed, dict):
            return self.parsed.get("object") or self.parsed.get("type") or "(data)"
        if self.is_done:
            return "[DONE]"
        return "(raw)" if self.data else "(empty)"

    @property
    def body(self) -> dict:
        return self.parsed if isinstance(self.parsed, dict) else {}


def parse_sse(text: str) -> list[SSEEvent]:
    """Parse SSE, tolerating every framing seen in this dataset.

    Handles ``event: X`` / ``event:X``, ``data: {...}`` / ``data:{...}``,
    multi-line ``data:`` joins, comment lines, and the ``[DONE]`` sentinel.
    """
    events: list[SSEEvent] = []
    cur_event: str | None = None
    data_lines: list[str] = []
    raw_lines: list[str] = []

    def flush() -> None:
        nonlocal cur_event, data_lines, raw_lines
        if cur_event is None and not data_lines:
            raw_lines = []
            return
        data = "\n".join(data_lines)
        parsed: Any = None
        if data and data.strip() != "[DONE]":
            try:
                parsed = json.loads(data)
            except Exception:
                parsed = None
        events.append(
            SSEEvent(len(events), cur_event, data, parsed, "\n".join(raw_lines))
        )
        cur_event = None
        data_lines = []
        raw_lines = []

    for line in text.split("\n"):
        line = line.rstrip("\r")
        if not line.strip():
            flush()
            continue
        if line.startswith(":"):  # comment / keep-alive
            continue
        raw_lines.append(line)
        field, sep, value = line.partition(":")
        if sep and value.startswith(" "):
            value = value[1:]
        if field == "event":
            cur_event = value.strip() or None
        elif field == "data":
            data_lines.append(value)
        # "id" and "retry" fields carry no payload we render
    flush()
    return events


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


def detect_kind(text: str) -> str:
    head = text.lstrip()
    if head.startswith("{") or head.startswith("["):
        return JSON_KIND
    for line in text.split("\n")[:50]:
        s = line.strip()
        if s.startswith("event:") or s.startswith("data:"):
            return SSE
    return TEXT_KIND


def detect_protocol_request(obj: Any) -> str:
    if not isinstance(obj, dict):
        return UNKNOWN
    if {"instructions", "input"} <= set(obj):
        return RESPONSES
    tools = obj.get("tools") or []
    t0 = tools[0] if tools and isinstance(tools[0], dict) else {}
    if isinstance(obj.get("system"), list) or "input_schema" in t0:
        return ANTHROPIC
    if "function" in t0:
        return CHAT
    msgs = obj.get("messages")
    if isinstance(msgs, list) and any(
        isinstance(m, dict) and m.get("role") == "system" for m in msgs
    ):
        return CHAT
    if "system" in obj:
        return ANTHROPIC
    if isinstance(msgs, list):
        return CHAT
    return UNKNOWN


def detect_protocol_response(events: list[SSEEvent], obj: Any = None) -> str:
    names = {e.name for e in events}
    if names & {"message_start", "content_block_start", "content_block_delta", "message_stop"}:
        return ANTHROPIC
    if names & {"response.created", "response.in_progress", "response.completed"}:
        return RESPONSES
    for e in events:
        if isinstance(e.parsed, dict):
            if e.parsed.get("object") == "chat.completion.chunk":
                return CHAT
            if e.parsed.get("object") == "response":
                return RESPONSES
            if e.parsed.get("object") == "chat.completion":
                return CHAT
            if e.parsed.get("type") in ("message_start", "message"):
                return ANTHROPIC
    if isinstance(obj, dict):
        if obj.get("object") == "chat.completion" or "choices" in obj:
            return CHAT
        if obj.get("object") == "response":
            return RESPONSES
        if obj.get("type") == "message":
            return ANTHROPIC
    return UNKNOWN


def detect_direction(kind: str, obj: Any) -> str:
    if kind == SSE:
        return RESPONSE
    if kind == JSON_KIND and isinstance(obj, dict):
        if obj.get("object") in ("chat.completion", "response") or "choices" in obj:
            return RESPONSE
        if "output" in obj and "usage" in obj and "input" not in obj:
            return RESPONSE
        return REQUEST
    return "unknown"


FNAME_RE = re.compile(
    r"^(?P<agent>[A-Za-z][A-Za-z0-9]*)[-_](?P<mode>.+?)[-_](?P<dir>Request|Response)$",
    re.IGNORECASE,
)


def parse_filename(path: Path) -> tuple[str, str, str | None]:
    m = FNAME_RE.match(path.stem)
    if not m:
        return path.stem, "", None
    return (
        m.group("agent"),
        m.group("mode"),
        "request" if m.group("dir").lower() == "request" else "response",
    )


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


@dataclass
class Capture:
    path: Path
    text: str
    agent: str
    mode: str
    direction_hint: str | None
    kind: str
    protocol: str
    direction: str
    payload: Any  # dict for JSON, list[SSEEvent] for SSE

    @property
    def mismatch(self) -> bool:
        return self.direction_hint is not None and self.direction_hint != self.direction

    @property
    def size(self) -> int:
        return len(self.text.encode("utf-8"))


def classify(path: Path) -> Capture:
    text = path.read_text(encoding="utf-8", errors="replace")
    agent, mode, hint = parse_filename(path)
    kind = detect_kind(text)
    payload: Any = None
    obj: Any = None
    if kind == JSON_KIND:
        try:
            obj = json.loads(text)
            payload = obj
        except Exception:
            kind = TEXT_KIND
    elif kind == SSE:
        payload = parse_sse(text)

    if kind == SSE:
        direction = RESPONSE
        protocol = detect_protocol_response(payload)
    else:
        direction = detect_direction(kind, obj)
        protocol = (
            detect_protocol_request(obj)
            if direction == REQUEST
            else detect_protocol_response([], obj)
        )
    return Capture(path, text, agent, mode, hint, kind, protocol, direction, payload)


# --------------------------------------------------------------------------
# Request renderers
# --------------------------------------------------------------------------

REQUEST_SKIP = {"messages", "system", "tools", "input", "instructions"}

ANTHROPIC_PARAM_ORDER = [
    "model",
    "max_tokens",
    "stream",
    "temperature",
    "top_p",
    "top_k",
    "stop_sequences",
    "tool_choice",
    "thinking",
    "output_config",
    "context_management",
    "service_tier",
]

RESPONSES_PARAM_ORDER = [
    "model",
    "tool_choice",
    "parallel_tool_calls",
    "reasoning",
    "store",
    "stream",
    "include",
    "prompt_cache_key",
    "max_output_tokens",
    "truncation",
]

CHAT_PARAM_ORDER = [
    "model",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "stop",
    "tool_choice",
    "parallel_tool_calls",
    "stream",
    "stream_options",
    "response_format",
]


def render_params(obj: dict, order: list[str], headers: list[str] | None = None) -> str:
    headers = headers or ["field", "value"]
    rows: list[list[Any]] = []
    seen: set[str] = set()
    for k in order:
        if k in obj:
            rows.append([f"`{k}`", inline(obj[k])])
            seen.add(k)
    for k, v in obj.items():
        if k in seen or k in REQUEST_SKIP:
            continue
        rows.append([f"`{k}`", inline(v)])
    return table(headers, rows)


def render_anthropic_request(obj: dict) -> list[str]:
    out = ["## 1. Request — Anthropic Messages API\n"]
    out.append("### 1.1 Parameters\n")
    out.append(render_params(obj, ANTHROPIC_PARAM_ORDER))

    meta = obj.get("metadata")
    if isinstance(meta, dict) and meta:
        out.append("\n### 1.2 Metadata\n")
        uid = meta.get("user_id")
        parsed_uid = parse_json_string(uid)
        if parsed_uid is not None:
            rows = [[f"`{k}`", f"`{v}`"] for k, v in parsed_uid.items()]
            out.append(
                "`metadata.user_id` is a JSON string; expanded below "
                f"({chars(len(str(uid)))} chars raw).\n\n"
            )
            out.append(table(["user_id field", "value"], rows))
        for k, v in meta.items():
            if k != "user_id":
                out.append(f"- `{k}`: {inline(v)}\n")
        out.append(
            "\n"
            + details(
                f"raw metadata — {chars(len(json.dumps(meta, ensure_ascii=False)))} chars",
                fence(json.dumps(meta, ensure_ascii=False, indent=2)),
            )
        )

    system = obj.get("system")
    if isinstance(system, list):
        out.append(f"\n### 1.3 System prompt ({len(system)} blocks)\n")
        rows = []
        for i, b in enumerate(system):
            cc = b.get("cache_control")
            rows.append(
                [
                    f"`[{i}]`",
                    f"`{b.get('type')}`",
                    chars(len(b.get("text", ""))),
                    inline(cc) if cc else "—",
                ]
            )
        out.append(table(["#", "type", "text chars", "cache_control"], rows))
        out.append("")
        for i, b in enumerate(system):
            out.append(f"**system[{i}]** — `{b.get('type')}`\n")
            out.append(text_block(b.get("text", ""), label=f"system[{i}]"))
            extra = {k: v for k, v in b.items() if k not in ("type", "text")}
            if extra:
                out.append(f"\nblock fields: {inline(extra)}\n")
            out.append("")
    elif isinstance(system, str):
        out.append("\n### 1.3 System prompt\n")
        out.append(text_block(system, label="system"))

    messages = obj.get("messages")
    if isinstance(messages, list):
        out.append(f"\n### 1.4 Messages ({len(messages)})\n")
        for i, m in enumerate(messages):
            out.extend(render_anthropic_message(m, i, level=4))

    tools = obj.get("tools")
    if isinstance(tools, list) and tools:
        out.append(f"\n### 1.5 Tools ({len(tools)})\n")
        out.append(render_anthropic_tools(tools))
    return out


def render_anthropic_message(m: dict, i: int, level: int = 4) -> list[str]:
    out: list[str] = []
    h = "#" * level
    role = m.get("role")
    out.append(f"{h} [{i}] role: `{role}`\n")
    extra = {k: v for k, v in m.items() if k not in ("role", "content")}
    if extra:
        out.append(f"other fields: {inline(extra)}\n\n")
    content = m.get("content")
    if isinstance(content, str):
        out.append(text_block(content, label=f"messages[{i}].content"))
        out.append("")
        return out
    if not isinstance(content, list):
        out.append(json_block(content))
        out.append("")
        return out
    for j, b in enumerate(content):
        if not isinstance(b, dict):
            out.append(f"**content[{j}]** (raw): {inline(b)}\n")
            continue
        btype = b.get("type")
        out.append(f"**content[{j}]** — `{btype}`\n")
        if btype == "text":
            out.append(text_block(b.get("text", ""), label=f"content[{j}].text"))
            for k, v in b.items():
                if k not in ("type", "text"):
                    out.append(f"- `{k}`: {inline(v)}\n")
        elif btype in ("thinking", "redacted_thinking"):
            out.append(
                text_block(b.get("thinking", b.get("data", "")), label=f"content[{j}].thinking")
            )
            if b.get("signature"):
                sig = b["signature"]
                out.append(
                    "\n"
                    + details(
                        f"signature — {chars(len(sig))} chars",
                        fence(sig),
                    )
                )
        elif btype == "tool_use":
            out.append(
                table(
                    ["field", "value"],
                    [
                        ["`id`", inline(b.get("id"))],
                        ["`name`", inline(b.get("name"))],
                        ["`input`", f"see below"],
                    ],
                )
            )
            out.append(
                "\n**input**\n\n" + json_block(b.get("input"), label="tool input")
            )
        elif btype == "tool_result":
            out.append(
                table(
                    ["field", "value"],
                    [
                        ["`tool_use_id`", inline(b.get("tool_use_id"))],
                        ["`is_error`", inline(b.get("is_error"))],
                    ],
                )
            )
            inner = b.get("content")
            out.append("\n**content**\n\n")
            if isinstance(inner, str):
                out.append(text_block(inner, label=f"content[{j}].content"))
            elif isinstance(inner, list):
                for k, sub in enumerate(inner):
                    if isinstance(sub, dict) and sub.get("type") == "text":
                        out.append(text_block(sub.get("text", ""), label=f"content[{j}][{k}]"))
                    else:
                        out.append(json_block(sub))
            else:
                out.append(json_block(inner))
        elif btype == "image":
            src = b.get("source") or {}
            out.append(
                table(
                    ["field", "value"],
                    [
                        ["`source.type`", inline(src.get("type"))],
                        ["`source.media_type`", inline(src.get("media_type"))],
                        [
                            "`source.data`",
                            f"base64, {chars(len(src.get('data', '')))} chars (omitted)",
                        ],
                    ],
                )
            )
        elif btype == "document":
            src = b.get("source") or {}
            rows = []
            for k, v in src.items():
                if k in ("data", "text"):
                    rows.append([f"`source.{k}`", f"{chars(len(str(v)))} chars — see below"])
                else:
                    rows.append([f"`source.{k}`", inline(v)])
            for k in ("title", "context", "citations"):
                if b.get(k) is not None:
                    rows.append([f"`{k}`", inline(b[k])])
            out.append(table(["field", "value"], rows))
            if isinstance(src.get("text"), str):
                out.append("\n**source.text**\n\n" + text_block(src["text"], label="document text"))
        elif btype == "server_tool_use":
            out.append(
                table(
                    ["field", "value"],
                    [["`id`", inline(b.get("id"))], ["`name`", inline(b.get("name"))]],
                )
            )
            out.append("\n**input**\n\n" + json_block(b.get("input")))
        elif isinstance(btype, str) and btype.endswith("_tool_result"):
            out.append(
                table(
                    ["field", "value"],
                    [["`tool_use_id`", inline(b.get("tool_use_id"))]],
                )
            )
            content = b.get("content")
            out.append("\n**content**\n\n")
            if isinstance(content, str):
                out.append(text_block(content, label="tool result"))
            elif isinstance(content, list):
                for k, sub in enumerate(content):
                    if isinstance(sub, dict) and not sub.get("type", "").endswith("_tool_result"):
                        out.append(
                            f"**content[{k}]** — `{sub.get('type')}`\n\n"
                            + table(["field", "value"], [[f"`{kk}`", inline(vv)] for kk, vv in sub.items()])
                            + "\n"
                        )
                    else:
                        out.append(json_block(sub))
            else:
                out.append(json_block(content))
        else:
            out.append(unknown_type_fallback(b))
        out.append("")
    return out


def render_anthropic_tools(tools: list[dict]) -> str:
    out: list[str] = []
    rows = []
    for i, t in enumerate(tools):
        schema = t.get("input_schema") or {}
        props = schema.get("properties") or {}
        rows.append(
            [
                f"`[{i}]`",
                f"`{t.get('name')}`",
                chars(len(t.get("description", ""))),
                len(props),
                ", ".join(schema.get("required") or []) or "—",
                ", ".join(k for k in t if k not in ("name", "description", "input_schema")) or "—",
            ]
        )
    out.append(table(["#", "name", "desc chars", "#params", "required", "other keys"], rows))
    out.append("")
    for i, t in enumerate(tools):
        out.append(f"#### [{i}] `{t.get('name')}`\n")
        desc = t.get("description", "")
        out.append(f"- description: {chars(len(desc))} chars\n")
        for k, v in t.items():
            if k not in ("name", "description", "input_schema"):
                out.append(f"- `{k}`: {inline(v)}\n")
        out.append("")
        out.append(details("description", fence(desc)))
        out.append("")
        out.append("<details>\n<summary>input_schema</summary>\n\n")
        out.append(json_block(t.get("input_schema")))
        out.append("\n</details>\n")
    return "\n".join(out)


def render_responses_request(obj: dict) -> list[str]:
    out = ["## 1. Request — OpenAI Responses API\n"]
    out.append("### 1.1 Parameters\n")
    out.append(render_params(obj, RESPONSES_PARAM_ORDER))

    instr = obj.get("instructions")
    if instr is not None:
        out.append("\n### 1.2 Instructions\n")
        out.append(text_block(instr, label="instructions"))

    meta = obj.get("client_metadata")
    if isinstance(meta, dict) and meta:
        out.append("\n### 1.3 Client metadata\n")
        rows = []
        nested: list[tuple[str, Any]] = []
        for k, v in meta.items():
            parsed = parse_json_string(v)
            if parsed is not None:
                rows.append([f"`{k}`", f"JSON string, {chars(len(str(v)))} chars → expanded"])
                nested.append((k, parsed))
            else:
                rows.append([f"`{k}`", f"`{v}`"])
        out.append(table(["field", "value"], rows))
        for k, parsed in nested:
            out.append(f"\n**`{k}` decoded**\n\n")
            if isinstance(parsed, dict):
                out.append(
                    table(
                        ["field", "value"],
                        [[f"`{k2}`", inline(v2)] for k2, v2 in parsed.items()],
                    )
                )
            else:
                out.append(json_block(parsed))
            raw = meta[k]
            out.append(
                "\n"
                + details(
                    f"`{k}` — raw JSON string, {chars(len(str(raw)))} chars",
                    fence(raw, "json"),
                )
            )

    inp = obj.get("input")
    if isinstance(inp, list):
        out.append(f"\n### 1.4 Input ({len(inp)} items)\n")
        for i, item in enumerate(inp):
            out.extend(render_responses_input_item(item, i))

    tools = obj.get("tools")
    if isinstance(tools, list) and tools:
        out.append(f"\n### 1.5 Tools ({len(tools)})\n")
        out.append(render_responses_tools(tools))
    return out


def render_responses_input_item(item: Any, i: int) -> list[str]:
    out: list[str] = []
    if not isinstance(item, dict):
        out.append(f"#### [{i}] (raw)\n\n{inline(item)}\n")
        return out
    itype = item.get("type")
    role = item.get("role")
    head = f"#### [{i}] type: `{itype}`"
    if role:
        head += f", role: `{role}`"
    if item.get("id"):
        head += f", id: `{item['id']}`"
    out.append(head + "\n")

    if item.get("name") and itype in ("function_call", "custom_tool_call"):
        out.append(
            table(
                ["field", "value"],
                [
                    ["`name`", inline(item.get("name"))],
                    ["`call_id`", inline(item.get("call_id"))],
                    ["`status`", inline(item.get("status"))],
                ],
            )
        )
        args = item.get("arguments")
        if args is not None:
            parsed = parse_json_string(args)
            out.append("\n**arguments**\n\n")
            out.append(
                json_block(parsed)
                if parsed is not None
                else fence(args if isinstance(args, str) else json.dumps(args))
            )
        out.append("")
        return out

    if itype in ("function_call_output", "custom_tool_call_output"):
        out.append(f"- `call_id`: {inline(item.get('call_id'))}\n")
        output = item.get("output")
        out.append("\n**output**\n\n")
        if isinstance(output, str):
            out.append(text_block(output, label="output"))
        else:
            out.append(json_block(output))
        out.append("")
        return out

    if itype == "reasoning":
        out.append(f"- `summary`: {inline(item.get('summary'))}\n")
        for j, part in enumerate(item.get("content") or []):
            if isinstance(part, dict) and part.get("type") == "reasoning_text":
                out.append(f"\n**content[{j}]** — `reasoning_text`\n\n")
                out.append(text_block(part.get("text", ""), label=f"content[{j}]"))
        encrypted = item.get("encrypted_content")
        if encrypted:
            out.append(
                "\n"
                + details(
                    f"encrypted_content — {chars(len(encrypted))} chars",
                    fence(encrypted, "text"),
                )
            )
        out.append("")
        return out

    content = item.get("content")
    if isinstance(content, str):
        out.append(text_block(content, label=f"input[{i}].content"))
        out.append("")
        return out
    if isinstance(content, list):
        for j, part in enumerate(content):
            if not isinstance(part, dict):
                out.append(f"**content[{j}]** (raw): {inline(part)}\n")
                continue
            ptype = part.get("type")
            out.append(f"**content[{j}]** — `{ptype}`\n")
            if ptype in ("input_text", "output_text", "text", "reasoning_text", "summary_text"):
                out.append(text_block(part.get("text", ""), label=f"content[{j}]"))
            elif ptype in ("input_image", "input_file"):
                rows = [
                    [f"`{k}`", f"<{chars(len(str(v)))} chars>" if k in ("image_url", "file_data") else inline(v)]
                    for k, v in part.items()
                    if k != "type"
                ]
                out.append(table(["field", "value"], rows))
            else:
                out.append(unknown_type_fallback(part))
            out.append("")
        return out

    out.append(json_block(item))
    out.append("")
    return out


def render_responses_tools(tools: list[dict], level: int = 4) -> str:
    out: list[str] = []
    rows = []
    for i, t in enumerate(tools):
        ttype = t.get("type")
        nested = t.get("tools") or []
        params = t.get("parameters") or {}
        props = params.get("properties") or {}
        rows.append(
            [
                f"`[{i}]`",
                f"`{t.get('name')}`",
                f"`{ttype}`",
                chars(len(t.get("description", ""))),
                len(nested) if nested else (len(props) or "—"),
                ", ".join(params.get("required") or []) or "—",
                inline(t.get("strict")) if "strict" in t else "—",
            ]
        )
    out.append(
        table(
            ["#", "name", "type", "desc chars", "#params/#subtools", "required", "strict"],
            rows,
        )
    )
    out.append("")
    h = "#" * level
    for i, t in enumerate(tools):
        out.append(f"{h} [{i}] `{t.get('name')}` — `{t.get('type')}`\n")
        out.append(f"- description: {chars(len(t.get('description', '')))} chars\n")
        for k, v in t.items():
            if k not in ("name", "description", "parameters", "tools"):
                out.append(f"- `{k}`: {inline(v)}\n")
        out.append("")
        out.append(details("description", fence(t.get("description", ""))))
        out.append("")
        if t.get("parameters") is not None:
            out.append("<details>\n<summary>parameters</summary>\n\n")
            out.append(json_block(t["parameters"]))
            out.append("\n</details>\n")
        nested = t.get("tools")
        if isinstance(nested, list) and nested:
            out.append(
                f"\n**Nested tools ({len(nested)})** — this tool type `{t.get('type')}` "
                "carries a sub-tool namespace.\n\n"
            )
            out.append(render_responses_tools(nested, level=level + 1))
    return "\n".join(out)


def render_chat_request(obj: dict) -> list[str]:
    out = ["## 1. Request — OpenAI Chat Completions API\n"]
    out.append("### 1.1 Parameters\n")
    out.append(render_params(obj, CHAT_PARAM_ORDER))

    messages = obj.get("messages")
    if isinstance(messages, list):
        out.append(f"\n### 1.2 Messages ({len(messages)})\n")
        for i, m in enumerate(messages):
            out.extend(render_chat_message(m, i))

    tools = obj.get("tools")
    if isinstance(tools, list) and tools:
        out.append(f"\n### 1.3 Tools ({len(tools)})\n")
        out.append(render_chat_tools(tools))
    return out


def render_chat_message(m: Any, i: int) -> list[str]:
    out: list[str] = []
    if not isinstance(m, dict):
        out.append(f"#### [{i}] (raw)\n\n{inline(m)}\n")
        return out
    head = f"#### [{i}] role: `{m.get('role')}`"
    if m.get("tool_call_id"):
        head += f", tool_call_id: `{m['tool_call_id']}`"
    if m.get("name"):
        head += f", name: `{m['name']}`"
    out.append(head + "\n")
    content = m.get("content")
    if isinstance(content, str):
        out.append(text_block(content, label=f"messages[{i}].content"))
        out.append("")
    elif isinstance(content, list):
        for j, part in enumerate(content):
            if isinstance(part, dict) and part.get("type") == "text":
                out.append(f"**content[{j}]** — `text`\n")
                out.append(text_block(part.get("text", ""), label=f"content[{j}]"))
            else:
                out.append(f"**content[{j}]**\n\n{json_block(part)}")
            out.append("")
    elif content is not None:
        out.append(f"`content`: {inline(content)}\n\n")

    for j, tc in enumerate(m.get("tool_calls") or []):
        fn = tc.get("function") or {}
        out.append(f"**tool_calls[{j}]**\n\n")
        out.append(
            table(
                ["field", "value"],
                [
                    ["`id`", inline(tc.get("id"))],
                    ["`type`", inline(tc.get("type"))],
                    ["`function.name`", inline(fn.get("name"))],
                ],
            )
        )
        args = fn.get("arguments")
        parsed = parse_json_string(args)
        out.append("\n**function.arguments**\n\n")
        out.append(
            json_block(parsed)
            if parsed is not None
            else fence(args if isinstance(args, str) else json.dumps(args))
        )
        out.append("")
    for k, v in m.items():
        if k not in ("role", "content", "tool_calls", "tool_call_id", "name"):
            out.append(f"- `{k}`: {inline(v)}\n")
    return out


def render_chat_tools(tools: list[dict]) -> str:
    out: list[str] = []
    rows = []
    for i, t in enumerate(tools):
        fn = t.get("function") or {}
        params = fn.get("parameters") or {}
        props = params.get("properties") or {}
        rows.append(
            [
                f"`[{i}]`",
                f"`{fn.get('name')}`",
                f"`{t.get('type')}`",
                chars(len(fn.get("description", ""))),
                len(props),
                ", ".join(params.get("required") or []) or "—",
                ", ".join(k for k in fn if k not in ("name", "description", "parameters")) or "—",
            ]
        )
    out.append(
        table(["#", "function.name", "type", "desc chars", "#params", "required", "other keys"], rows)
    )
    out.append("")
    for i, t in enumerate(tools):
        fn = t.get("function") or {}
        out.append(f"#### [{i}] `{fn.get('name')}`\n")
        out.append(f"- wrapper type: `{t.get('type')}`\n")
        out.append(f"- description: {chars(len(fn.get('description', '')))} chars\n")
        out.append("")
        out.append(details("description", fence(fn.get("description", ""))))
        out.append("")
        out.append("<details>\n<summary>parameters</summary>\n\n")
        out.append(json_block(fn.get("parameters")))
        out.append("\n</details>\n")
    return "\n".join(out)


# --------------------------------------------------------------------------
# Response renderers
# --------------------------------------------------------------------------


def timeline_table(rows: list[list[Any]], headers: list[str]) -> str:
    return table(headers, rows)


def render_anthropic_response(events: list[SSEEvent]) -> list[str]:
    out = ["## 2. Response — Anthropic Messages API (SSE stream)\n"]
    message: dict = {}
    deltas_final: dict = {}
    blocks: dict[int, dict] = {}
    delta_counts: dict[str, int] = {}
    timeline: list[list[Any]] = []

    for e in events:
        b = e.body
        t = b.get("type") or e.event or ""
        if t == "message_start":
            message = b.get("message") or {}
        elif t == "content_block_start":
            idx = b.get("index")
            cb = b.get("content_block") or {}
            # Seed with any inline content: some servers put the opening text
            # (or a full input) in content_block_start rather than in deltas.
            blocks[idx] = {
                "type": cb.get("type"),
                "text": cb.get("text") or cb.get("thinking") or "",
                "signature": cb.get("signature") or "",
                "json": json.dumps(cb["input"]) if cb.get("input") is not None else "",
            }
            timeline.append([e.index, f"`{t}`", idx, f"`{cb.get('type')}`", 0, ""])
            continue
        elif t == "content_block_delta":
            idx = b.get("index")
            d = b.get("delta") or {}
            dt = d.get("type", "?")
            delta_counts[dt] = delta_counts.get(dt, 0) + 1
            blk = blocks.setdefault(idx, {"type": None, "text": "", "signature": "", "json": ""})
            payload = ""
            if dt == "text_delta":
                payload = d.get("text", "")
                blk["text"] += payload
            elif dt == "thinking_delta":
                payload = d.get("thinking", "")
                blk["text"] += payload
            elif dt == "signature_delta":
                payload = d.get("signature", "")
                blk["signature"] += payload
            elif dt == "input_json_delta":
                payload = d.get("partial_json", "")
                blk["json"] += payload
            else:
                payload = json.dumps(d, ensure_ascii=False)
                blk["text"] += payload
            timeline.append(
                [e.index, f"`{t}`", idx, f"`{dt}`", len(payload), preview(payload)]
            )
            continue
        elif t == "message_delta":
            deltas_final = b
            timeline.append([e.index, f"`{t}`", "—", "`stop_reason`", 0, inline(b.get("delta"))])
            continue
        timeline.append([e.index, f"`{t}`", b.get("index", "—"), "—", 0, ""])

    out.append("### 2.1 Stream summary\n")
    usage_start = message.get("usage") or {}
    usage_end = deltas_final.get("usage") or {}
    rows = [
        ["message id", f"`{message.get('id')}`"],
        ["model", f"`{message.get('model')}`"],
        ["role", f"`{message.get('role')}`"],
        ["events", chars(len(events))],
        ["content blocks", len(blocks)],
        [
            "delta frames",
            ", ".join(f"`{k}`×{v}" for k, v in sorted(delta_counts.items())) or "—",
        ],
        ["stop_reason", f"`{(deltas_final.get('delta') or {}).get('stop_reason')}`"],
        ["usage (final)", inline(usage_end or usage_start)],
    ]
    out.append(table(["field", "value"], rows))

    out.append("\n### 2.2 Event timeline\n")
    out.append(
        timeline_table(
            timeline, ["#", "event", "block idx", "delta / type", "chars", "preview"]
        )
    )
    out.append(
        "\n> `preview` is a lossy teaser (backticks shown as `'`, pipes escaped). "
        "The rebuilt message and the raw stream below are exact.\n"
    )

    out.append("\n### 2.3 Rebuilt assistant message\n")
    out.append(
        "Deltas reassembled in arrival order — this is what the client effectively received.\n\n"
    )
    if usage_start:
        out.append("**usage at message_start**\n\n" + usage_table(usage_start) + "\n")
    for idx in sorted(blocks):
        blk = blocks[idx]
        out.append(f"#### block[{idx}] — `{blk['type']}`\n")
        if blk["json"]:
            parsed = parse_json_string(blk["json"])
            out.append(f"- streamed `input_json_delta` total: {chars(len(blk['json']))} chars\n")
            out.append("\n**assembled input**\n\n")
            out.append(
                json_block(parsed)
                if parsed is not None
                else fence(blk["json"], "json") + "\n> (fragment did not parse as JSON)\n"
            )
        if blk["text"]:
            out.append(text_block(blk["text"], label=f"block[{idx}] text"))
        if blk["signature"]:
            out.append(
                details(
                    f"signature — {chars(len(blk['signature']))} chars",
                    fence(blk["signature"]),
                )
            )
        out.append("")

    if usage_end:
        out.append("**final usage (message_delta)**\n\n" + usage_table(usage_end) + "\n")

    out.append("\n### 2.4 Raw event stream\n")
    out.append(render_raw_sse(events))
    return out


def render_responses_response(events: list[SSEEvent]) -> list[str]:
    out = ["## 2. Response — OpenAI Responses API (SSE stream)\n"]
    response_meta: dict = {}
    final_response: dict = {}
    items: dict[int, dict] = {}
    acc: dict[int, dict] = {}
    timeline: list[list[Any]] = []

    for e in events:
        b = e.body
        t = b.get("type") or e.event or ""
        if t in ("response.created", "response.in_progress"):
            response_meta = b.get("response") or {}
            timeline.append([e.index, f"`{t}`", "—", "—", 0, f"status={response_meta.get('status')}"])
            continue
        if t == "response.output_item.added":
            idx = b.get("output_index")
            item = b.get("item") or {}
            items[idx] = item
            acc.setdefault(idx, {"text": "", "args": ""})
            timeline.append(
                [e.index, f"`{t}`", idx, f"`{item.get('type')}`", 0, f"id={item.get('id')}"]
            )
            continue
        if t == "response.output_item.done":
            idx = b.get("output_index")
            items[idx] = b.get("item") or items.get(idx, {})
            timeline.append(
                [e.index, f"`{t}`", idx, f"`{(items[idx] or {}).get('type')}`", 0, "final item"]
            )
            continue
        if t == "response.content_part.added":
            idx = b.get("output_index")
            part = b.get("part") or {}
            timeline.append([e.index, f"`{t}`", idx, f"`{part.get('type')}`", 0, ""])
            continue
        if t.endswith(".delta"):
            idx = b.get("output_index")
            d = b.get("delta", "")
            slot = acc.setdefault(idx, {"text": "", "args": ""})
            if isinstance(d, str) and t in RESPONSES_TEXT_DELTAS:
                slot["text"] += d
            elif isinstance(d, str) and t in RESPONSES_ARG_DELTAS:
                slot["args"] += d
            timeline.append(
                [e.index, f"`{t}`", idx, "`delta`", len(d) if isinstance(d, str) else 0, preview(d if isinstance(d, str) else "")]
            )
            continue
        if t == "response.completed":
            final_response = b.get("response") or {}
            timeline.append(
                [
                    e.index,
                    f"`{t}`",
                    "—",
                    "—",
                    0,
                    f"status={final_response.get('status')}, usage={inline((final_response.get('usage') or {}).get('total_tokens'))}",
                ]
            )
            continue
        timeline.append([e.index, f"`{t}`", b.get("output_index", "—"), "—", 0, ""])

    out.append("### 2.1 Stream summary\n")
    usage = final_response.get("usage") or {}
    out.append(
        table(
            ["field", "value"],
            [
                ["response id", f"`{response_meta.get('id') or final_response.get('id')}`"],
                ["model", f"`{response_meta.get('model') or final_response.get('model')}`"],
                ["created_at", f"{iso(response_meta.get('created_at'))} ({response_meta.get('created_at')})"],
                ["object", f"`{response_meta.get('object')}`"],
                ["final status", f"`{final_response.get('status')}`"],
                ["error", inline(final_response.get("error"))],
                ["incomplete_details", inline(final_response.get("incomplete_details"))],
                ["output items", len(final_response.get("output") or items)],
                ["events", chars(len(events))],
            ],
        )
    )
    if usage:
        out.append("\n**usage**\n\n" + usage_table(usage) + "\n")

    out.append("\n### 2.2 Event timeline\n")
    out.append(timeline_table(timeline, ["#", "event", "output idx", "item / delta", "chars", "preview"]))
    out.append(
        "\n> `preview` is a lossy teaser; the rebuilt items and the raw stream below are exact.\n"
    )

    out.append("\n### 2.3 Rebuilt output items\n")
    final_items = final_response.get("output") or []
    order = sorted(set(list(items) + list(acc) + list(range(len(final_items)))))
    for idx in order:
        item = items.get(idx)
        if item is None:
            item = final_items[idx] if idx < len(final_items) else {}
        itype = (item or {}).get("type")
        out.append(f"#### output[{idx}] — `{itype}`\n")
        if itype == "reasoning":
            for j, part in enumerate(item.get("content") or []):
                if isinstance(part, dict) and part.get("type") == "reasoning_text":
                    out.append(text_block(part.get("text", ""), label=f"reasoning content[{j}]"))
            streamed = acc.get(idx, {}).get("text", "")
            if streamed and not (item.get("content")):
                out.append(text_block(streamed, label=f"reasoning (from deltas)"))
            if item.get("summary"):
                out.append(f"\n- `summary`: {inline(item['summary'])}\n")
        elif itype in ("function_call", "custom_tool_call"):
            out.append(
                table(
                    ["field", "value"],
                    [
                        ["`id`", inline(item.get("id"))],
                        ["`call_id`", inline(item.get("call_id"))],
                        ["`name`", inline(item.get("name"))],
                        ["`status`", inline(item.get("status"))],
                    ],
                )
            )
            args = item.get("arguments")
            streamed = acc.get(idx, {}).get("args", "")
            parsed = parse_json_string(args)
            out.append(
                f"\n- streamed argument fragments: {chars(len(streamed))} chars\n"
                f"- final `arguments`: {chars(len(args or ''))} chars\n\n"
            )
            out.append("**arguments (parsed)**\n\n")
            out.append(
                json_block(parsed)
                if parsed is not None
                else fence(args if isinstance(args, str) else json.dumps(args), "json")
            )
        elif itype == "message":
            for j, part in enumerate(item.get("content") or []):
                if isinstance(part, dict):
                    out.append(f"**content[{j}]** — `{part.get('type')}`\n")
                    out.append(text_block(part.get("text", ""), label=f"content[{j}]"))
        else:
            out.append(json_block(item))
        out.append("")

    out.append("\n### 2.4 Raw event stream\n")
    out.append(render_raw_sse(events))
    return out


def render_chat_response(events: list[SSEEvent]) -> list[str]:
    out = ["## 2. Response — OpenAI Chat Completions API (SSE stream)\n"]
    content = ""
    reasoning = ""
    tool_calls: dict[int, dict] = {}
    finish_reason = None
    usage: dict = {}
    meta: dict = {}
    timeline: list[list[Any]] = []

    for e in events:
        if e.is_done:
            timeline.append([e.index, "`[DONE]`", "—", "—", 0, "stream terminator"])
            continue
        b = e.body
        if b and not meta:
            meta = {k: b.get(k) for k in ("id", "model", "created", "object")}
        if b.get("usage"):
            usage = b["usage"]
        for ch in b.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content"):
                content += d["content"]
            if d.get("reasoning_content"):
                reasoning += d["reasoning_content"]
            for tc in d.get("tool_calls") or []:
                i = tc.get("index", 0)
                slot = tool_calls.setdefault(i, {"id": "", "type": "", "name": "", "arguments": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                if tc.get("type"):
                    slot["type"] = tc["type"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
            if ch.get("finish_reason"):
                finish_reason = ch["finish_reason"]
            keys = ",".join(k for k in d if d.get(k) not in (None, "")) or "—"
            n = len(d.get("content") or "") + len(d.get("reasoning_content") or "")
            n += sum(len((t.get("function") or {}).get("arguments") or "") for t in (d.get("tool_calls") or []))
            timeline.append(
                [
                    e.index,
                    f"`{ch.get('index', 0)}`",
                    f"`{keys}`",
                    f"`{ch.get('finish_reason')}`" if ch.get("finish_reason") else "—",
                    n,
                    preview(d.get("content") or d.get("reasoning_content") or ""),
                ]
            )

    out.append("### 2.1 Stream summary\n")
    out.append(
        table(
            ["field", "value"],
            [
                ["chunk id", f"`{meta.get('id')}`"],
                ["model", f"`{meta.get('model')}`"],
                ["object", f"`{meta.get('object')}`"],
                ["created", f"{iso(meta.get('created'))} ({meta.get('created')})"],
                ["data frames", chars(len(events))],
                ["finish_reason", f"`{finish_reason}`"],
                ["reasoning_content", f"{chars(len(reasoning))} chars" if reasoning else "—"],
                ["content", f"{chars(len(content))} chars" if content else "—"],
                ["tool_calls", len(tool_calls) or "—"],
            ],
        )
    )
    if usage:
        out.append("\n**usage**\n\n" + usage_table(usage) + "\n")

    out.append("\n### 2.2 Chunk timeline\n")
    out.append(timeline_table(timeline, ["#", "choice idx", "delta keys", "finish_reason", "chars", "preview"]))
    out.append("\n> `preview` is a lossy teaser; the rebuilt output and raw chunks below are exact.\n")

    out.append("\n### 2.3 Rebuilt assistant message\n")
    if reasoning:
        out.append("**reasoning_content**\n\n" + text_block(reasoning, label="reasoning_content"))
    if content:
        out.append("\n**content**\n\n" + text_block(content, label="content"))
    for i in sorted(tool_calls):
        slot = tool_calls[i]
        out.append(f"\n**tool_calls[{i}]**\n\n")
        out.append(
            table(
                ["field", "value"],
                [
                    ["`id`", inline(slot["id"])],
                    ["`type`", inline(slot["type"])],
                    ["`function.name`", inline(slot["name"])],
                ],
            )
        )
        parsed = parse_json_string(slot["arguments"])
        out.append("\n**function.arguments (assembled)**\n\n")
        out.append(
            json_block(parsed)
            if parsed is not None
            else fence(slot["arguments"], "json") + "\n> (fragment did not parse as JSON)\n"
        )

    out.append("\n### 2.4 Raw chunk stream\n")
    out.append(render_raw_sse(events))
    return out


def render_raw_sse(events: list[SSEEvent]) -> str:
    out: list[str] = []
    lines = []
    for e in events:
        if e.is_done:
            lines.append(f"- `#{e.index}` `[DONE]`")
        else:
            lines.append(f"- `#{e.index}` `{e.name}` — {chars(len(e.raw))} chars")
    out.append(
        details(
            f"all {len(events)} frames (index list)",
            "\n".join(lines),
        )
    )
    out.append("")
    for e in events:
        label = f"#{e.index} — {e.name}"
        body = e.raw if e.is_done else fence(e.data, "json")
        out.append(details(label, body))
        out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------
# Fallback renderers
# --------------------------------------------------------------------------


def render_unknown_request(obj: Any, protocol: str) -> list[str]:
    out = [f"## 1. Request — {protocol}\n"]
    out.append(
        "> This payload did not match any known protocol. It is reproduced verbatim below "
        "so nothing is lost.\n\n"
    )
    out.append(json_block(obj))
    out.append("")
    return out


def render_unknown_response(events: list[SSEEvent]) -> list[str]:
    out = ["## 2. Response — unrecognised SSE stream\n"]
    out.append(f"- frames: {chars(len(events))}\n\n")
    out.append(render_raw_sse(events))
    return out


# --------------------------------------------------------------------------
# Document assembly
# --------------------------------------------------------------------------


@dataclass
class Document:
    key: tuple[str, str]
    filename: str
    content: str
    captures: list[Capture]


def render_document(agent: str, mode: str, req: Capture | None, resp: Capture | None) -> str:
    title = f"{agent}-{mode}"
    out: list[str] = [f"# {title}\n"]
    out.append(
        f"Captured interaction between **{agent}** (mode: `{mode}`) and a MiMo endpoint.\n"
    )
    out.append(
        "Generated by `convert_to_markdown.py`. Every user, system, and tool payload is "
        "reproduced in full; long prose is wrapped in `<details>` blocks so the document "
        "stays scannable. Line endings inside payload text are normalised to LF. "
        "The `preview` columns in timeline tables are the only lossy part.\n"
    )

    rows = []
    mismatches = []
    for cap in (req, resp):
        if cap is None:
            continue
        hint = cap.direction_hint or "—"
        rows.append(
            [
                f"`{cap.path.name}`",
                human_bytes(cap.size),
                f"`{cap.direction}`",
                f"`{cap.protocol}`",
                f"`{cap.kind}`",
                f"`{hint}`",
                "⚠️ MISMATCH" if cap.mismatch else "ok",
            ]
        )
        if cap.mismatch:
            mismatches.append(cap)
    out.append("\n### Source files\n\n")
    out.append(
        table(
            ["file", "size", "content is", "protocol", "kind", "filename says", "check"],
            rows,
        )
    )
    for cap in mismatches:
        out.append(
            f"\n> ⚠️ **Filename/direction mismatch**: `{cap.path.name}` is named "
            f"`{cap.direction_hint}` but its content is a **{cap.direction}** "
            f"(`{cap.kind}`, `{cap.protocol}`). The filename was ignored; content won.\n"
        )
    if not mismatches:
        out.append("\n> Filenames agree with detected content direction in both files.\n")

    out.append("")
    if req is not None:
        if req.protocol == ANTHROPIC and isinstance(req.payload, dict):
            out.extend(render_anthropic_request(req.payload))
        elif req.protocol == RESPONSES and isinstance(req.payload, dict):
            out.extend(render_responses_request(req.payload))
        elif req.protocol == CHAT and isinstance(req.payload, dict):
            out.extend(render_chat_request(req.payload))
        else:
            out.extend(render_unknown_request(req.payload if req.payload is not None else req.text, req.protocol))
    else:
        out.append("## 1. Request\n\n> No request capture found for this pair.\n")

    out.append("\n---\n")
    if resp is not None and resp.kind == SSE:
        events = resp.payload
        if resp.protocol == ANTHROPIC:
            out.extend(render_anthropic_response(events))
        elif resp.protocol == RESPONSES:
            out.extend(render_responses_response(events))
        elif resp.protocol == CHAT:
            out.extend(render_chat_response(events))
        else:
            out.extend(render_unknown_response(events))
    elif resp is not None:
        out.append(f"## 2. Response — {resp.protocol} (non-streaming JSON)\n")
        out.append(json_block(resp.payload))
    else:
        out.append("## 2. Response\n\n> No response capture found for this pair.\n")
    return "\n".join(out).rstrip() + "\n"


def build_documents(captures: list[Capture]) -> tuple[list[Document], list[str]]:
    groups: dict[tuple[str, str], dict[str, Capture]] = {}
    for cap in captures:
        groups.setdefault((cap.agent, cap.mode), {})[cap.direction] = cap

    docs: list[Document] = []
    warnings: list[str] = []
    for (agent, mode), pair in sorted(groups.items()):
        req = pair.get(REQUEST)
        resp = pair.get(RESPONSE)
        if req is None:
            warnings.append(f"{agent}-{mode}: no request capture")
        if resp is None:
            warnings.append(f"{agent}-{mode}: no response capture")
        for cap in pair.values():
            if cap.mismatch:
                warnings.append(
                    f"{cap.path.name}: named '{cap.direction_hint}' but content is a "
                    f"{cap.direction} ({cap.kind}, {cap.protocol})"
                )
        content = render_document(agent, mode, req, resp)
        doc = Document(
            key=(agent, mode),
            filename=f"{agent}-{mode}.md",
            content=content,
            captures=[c for c in (req, resp) if c is not None],
        )
        docs.append(doc)
    return docs, warnings


def build_index(docs: list[Document], warnings: list[str]) -> str:
    out = ["# Captured agent ↔ LLM interactions — Markdown index\n", "\n"]
    out.append(
        "Each document below pairs one captured **request** with the **response** that "
        "followed it. All three clients were pointed at Xiaomi **MiMo** endpoints "
        "(`mimo-v2.5`, `mimo-v2.5-pro`) through three different wire protocols.\n"
    )
    rows = []
    for doc in docs:
        req = next((c for c in doc.captures if c.direction == REQUEST), None)
        resp = next((c for c in doc.captures if c.direction == RESPONSE), None)
        rows.append(
            [
                f"[{doc.filename}]({doc.filename})",
                f"`{PROTOCOL_SHORT.get(req.protocol, req.protocol) if req else '—'}`",
                human_bytes(req.size) if req else "—",
                f"`{PROTOCOL_SHORT.get(resp.protocol, resp.protocol) if resp else '—'}`",
                human_bytes(resp.size) if resp else "—",
                "⚠️" if any(c.mismatch for c in doc.captures) else "ok",
            ]
        )
    out.append("\n")
    out.append(
        table(
            ["document", "request protocol", "req size", "response protocol", "resp size", "check"],
            rows,
        )
    )
    out.append("\n## Protocol coverage\n\n")
    out.append(
        table(
            ["agent", "wire protocol", "direction", "system prompt lives in", "tool object shape"],
            [
                [
                    "Claude Code",
                    "Anthropic Messages API",
                    "POST /v1/messages",
                    "`system[]` blocks (+ a `role:\"system\"` message)",
                    "`{name, description, input_schema}`",
                ],
                [
                    "Codex CLI",
                    "OpenAI Responses API",
                    "POST /v1/responses",
                    "`instructions` + first `input[]` item with `role:\"developer\"`",
                    "`{type, name, strict, parameters}`, incl. nested `type:\"namespace\"`",
                ],
                [
                    "OpenCode",
                    "OpenAI Chat Completions API",
                    "POST /v1/chat/completions",
                    "`messages[0]` with `role:\"system\"` (a plain string)",
                    "`{type:\"function\", function:{name, description, parameters}}`",
                ],
            ],
        )
    )
    if warnings:
        out.append("\n## Warnings\n\n")
        for w in warnings:
            out.append(f"- {w}\n")
    out.append(
        "\n---\n\nRegenerate with `python3 convert_to_markdown.py`; "
        "audit completeness with `--verify`.\n"
    )
    return "".join(out)


# --------------------------------------------------------------------------
# Fidelity verification
# --------------------------------------------------------------------------


def normalize(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return " ".join(s.split())


def collect_strings(node: Any, sink: set[str], min_len: int = 40) -> None:
    if isinstance(node, str):
        if len(normalize(node)) >= min_len:
            sink.add(node)
    elif isinstance(node, dict):
        for v in node.values():
            collect_strings(v, sink, min_len)
    elif isinstance(node, list):
        for v in node:
            collect_strings(v, sink, min_len)
    elif isinstance(node, SSEEvent):
        if node.parsed is not None:
            collect_strings(node.parsed, sink, min_len)
        else:
            collect_strings(node.data, sink, min_len)


def verify(docs: list[Document]) -> tuple[bool, list[str]]:
    report: list[str] = []
    ok = True
    for doc in docs:
        haystack = normalize(doc.content)
        missing: list[str] = []
        checked = 0
        for cap in doc.captures:
            strings: set[str] = set()
            collect_strings(cap.payload if cap.payload is not None else cap.text, strings)
            for s in strings:
                checked += 1
                if normalize(s) in haystack:
                    continue
                # It may only be present in JSON-escaped form (inside a JSON dump).
                escaped = json.dumps(s, ensure_ascii=False)[1:-1]
                if normalize(escaped) in haystack:
                    continue
                missing.append(s[:80])
        status = "PASS" if not missing else f"FAIL ({len(missing)} missing)"
        if missing:
            ok = False
        report.append(
            f"  {doc.filename:<22} {status:<22} {checked} strings checked "
            f"across {len(doc.captures)} payload(s)"
        )
        for m in missing[:5]:
            report.append(f"        missing: {m!r}")
    return ok, report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def discover(input_dir: Path) -> list[Path]:
    files = [
        p
        for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in SOURCE_SUFFIXES
    ]
    return files


def print_classification(captures: list[Capture]) -> None:
    rows = []
    for cap in captures:
        rows.append(
            [
                cap.path.name,
                cap.agent,
                f"`{cap.mode}`",
                f"`{cap.direction}`",
                cap.direction_hint or "—",
                "MISMATCH" if cap.mismatch else "ok",
                f"`{cap.kind}`",
                cap.protocol,
                human_bytes(cap.size),
            ]
        )
    print(table(["file", "agent", "mode", "content is", "filename says", "check", "kind", "protocol", "size"], rows))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", default=None, help="directory holding the captures (default: script dir)")
    ap.add_argument("--out", default=None, help="output directory (default: <input>/md)")
    ap.add_argument("--dry-run", action="store_true", help="classify and report, write nothing")
    ap.add_argument("--verify", action="store_true", help="check no payload text was dropped")
    args = ap.parse_args(argv)

    input_dir = Path(args.input).resolve() if args.input else Path(__file__).resolve().parent
    out_dir = Path(args.out).resolve() if args.out else input_dir / "md"

    paths = discover(input_dir)
    if not paths:
        print(f"no capture files found in {input_dir}", file=sys.stderr)
        return 2

    captures = [classify(p) for p in paths]
    print(f"Scanned {len(paths)} files in {input_dir}\n")
    print_classification(captures)

    docs, warnings = build_documents(captures)
    print(
        f"\n{len(captures)} files -> {len(docs)} request/response pair(s): "
        + ", ".join(d.filename for d in docs)
    )
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")

    if args.verify:
        print("\nFidelity check:")
        ok, report = verify(docs)
        print("\n".join(report))
        if not ok:
            print("\nFAILED: some payload text is missing from the generated Markdown.")
            return 1
        print("\nOK: every long string in every payload is present in its Markdown document.")
        return 0

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        (out_dir / doc.filename).write_text(doc.content, encoding="utf-8")
    (out_dir / "README.md").write_text(build_index(docs, warnings), encoding="utf-8")
    print(f"\nWrote {len(docs)} document(s) + README.md to {out_dir}")

    print("\nFidelity check:")
    ok, report = verify(docs)
    print("\n".join(report))
    if not ok:
        print("\nWARNING: some payload text is missing from the generated Markdown.")
        return 1
    print("\nOK: every long string in every payload is present in its Markdown document.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
