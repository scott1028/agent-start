# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared command options, panels, request fields and cross-agent helpers."""

import os
import re
from pathlib import Path
from typing import Literal, NamedTuple, NoReturn, Optional

import click
import typer
from typer.core import TyperCommand

from agent_switch import providers


_SUBAGENT_DESCRIPTION = (
    "Local coding subagent running on a local model for debugging, implementation, and "
    "codebase research. Use when the user asks to spawn a local agent."
)


_SUBAGENT_INSTRUCTIONS = (
    "You are a local coding subagent running on a local model. Complete the assigned task directly, "
    "use the available tools when useful, verify your work, and return a concise result to the "
    "parent agent."
)


# OpenCode sends min(limit.output, this) as max_tokens unless the env var below raises it.
_OPENCODE_OUTPUT_TOKEN_MAX = 32_000


_PASSTHROUGH = {"allow_extra_args": True, "ignore_unknown_options": True}


class _PassthroughCommand(TyperCommand):
    """Preserve the option separator when forwarding arguments to an agent."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        raw_args = list(args)
        try:
            separator = raw_args.index("--")
        except ValueError:
            return super().parse_args(ctx, args)
        trailing_count = len(raw_args) - separator - 1
        remaining = super().parse_args(ctx, args)
        insert_at = max(0, len(remaining) - trailing_count)
        if insert_at >= len(remaining) or remaining[insert_at] != "--":
            remaining.insert(insert_at, "--")
            ctx.args = remaining
        return remaining


# Shared by every agent command; only the config/env/command differ. Help is grouped into rich panels so `--help` reads as Model / Server / Session instead of one long unaligned list.
_PANEL_MODEL = "Model"


_PANEL_SERVER = "Server"


_PANEL_SAMPLING = "Sampling (in the agent's requests, not on the server; ignored if unsent)"


_PANEL_SESSION = "Agent session"


_MODEL_OPTION = typer.Option(
    None,
    "--model",
    "-m",
    rich_help_panel = _PANEL_MODEL,
    help = "Model for the agent, or a bare `org/name(:variant)` positional. "
    "Defaults to the one loaded on the server.",
)


_CONTEXT_OPTION = typer.Option(
    0,
    "--max-seq-length",
    "--context-length",
    rich_help_panel = _PANEL_MODEL,
    help = "Context length in tokens for the load (0 = model default).",
)


_URL_OPTION = typer.Option(
    None,
    "--url",
    rich_help_panel = _PANEL_SERVER,
    help = "Model server URL, e.g. http://127.0.0.1:11434. Default: the one Ollama, LM Studio, "
    "llama-server or vLLM server answering on its usual local port.",
)


_PROVIDER_OPTION = typer.Option(
    None,
    "--provider",
    rich_help_panel = _PANEL_SERVER,
    help = "Server type, when detection should be skipped. Without --url, its usual local port.",
)


# --provider picks the model-server adapter in providers/, not the coding agent, so these server
# names are intentional.
ProviderName = Literal["ollama", "lmstudio", "llamacpp", "vllm", "openai"]


_MODEL_LOAD_OPTION = typer.Option(
    True,
    "--model-load/--no-model-load",
    rich_help_panel = _PANEL_SERVER,
    help = "--no-model-load: never load, reload or unload a model on the server; "
    "--model must already be loaded there.",
)


_REASONING_OPTION = typer.Option(
    None,
    "--reasoning",
    rich_help_panel = _PANEL_SERVER,
    help = (
        "Reasoning mode for this agent session. Defaults to auto so the model's chat "
        "template decides; use 'on' or 'off' to override it."
    ),
)


_REASONING_EFFORT_OPTION = typer.Option(
    None,
    "--reasoning-effort",
    rich_help_panel = _PANEL_SERVER,
    help = (
        "Reasoning effort for this agent session, e.g. 'medium'. The "
        "levels are the model's own, so pass one its chat template accepts. Default: "
        "unset, which keeps the template's level."
    ),
)


# Sampling overrides ride in the agent's own config, so only this session uses them; one the agent cannot send is ignored. Default unset means the model's recommended sampling is used.
_TEMPERATURE_OPTION = typer.Option(
    None,
    "--temperature",
    min = 0.0,
    max = 2.0,
    rich_help_panel = _PANEL_SAMPLING,
    help = "Pin the sampling temperature. Default: unset (per-model recommendation).",
)


_TOP_P_OPTION = typer.Option(
    None,
    "--top-p",
    min = 0.0,
    max = 1.0,
    rich_help_panel = _PANEL_SAMPLING,
    help = "Pin top-p (nucleus) sampling. Default: unset (per-model recommendation).",
)


_TOP_K_OPTION = typer.Option(
    None,
    "--top-k",
    min = -1,
    max = 100,
    rich_help_panel = _PANEL_SAMPLING,
    help = "Pin top-k sampling. Default: unset (per-model recommendation).",
)


_MIN_P_OPTION = typer.Option(
    None,
    "--min-p",
    min = 0.0,
    max = 1.0,
    rich_help_panel = _PANEL_SAMPLING,
    help = "Pin min-p sampling threshold. Default: unset (per-model recommendation).",
)


_REPETITION_PENALTY_OPTION = typer.Option(
    None,
    "--repetition-penalty",
    min = 1.0,
    max = 2.0,
    rich_help_panel = _PANEL_SAMPLING,
    help = "Pin the repetition penalty. Default: unset (per-model recommendation).",
)


_PRESENCE_PENALTY_OPTION = typer.Option(
    None,
    "--presence-penalty",
    min = 0.0,
    max = 2.0,
    rich_help_panel = _PANEL_SAMPLING,
    help = "Pin the presence penalty. Default: unset (per-model recommendation).",
)


_MAX_TOKENS_OPTION = typer.Option(
    None,
    "--max-tokens",
    min = 1,
    rich_help_panel = _PANEL_SAMPLING,
    help = (
        "Most tokens the agent may generate in one response. Default: a quarter of the "
        "context window, up to 32,000. Capped at half the window so the conversation "
        "keeps room."
    ),
)


# Agent-session knobs.
_KEY_OPTION = typer.Option(
    None,
    "--api-key",
    envvar = "AGENT_SWITCH_API_KEY",
    rich_help_panel = _PANEL_SESSION,
    help = "API key for the model server (or AGENT_SWITCH_API_KEY); it is remembered per "
    "server for next time.",
)


_HEADER_OPTION = typer.Option(
    None,
    "--header",
    metavar = "NAME=VALUE",
    rich_help_panel = _PANEL_SESSION,
    help = "Add an HTTP header to every request sent to the model server; repeat the flag "
    "for more. An Authorization header here replaces the built-in Bearer <api-key>.",
)


_LAUNCH_OPTION = typer.Option(
    True,
    "--launch/--no-launch",
    rich_help_panel = _PANEL_SESSION,
    help = "--no-launch prints the env and command instead (remote shells, WSL).",
)


# One normalized "run tools without prompting" switch. Each agent spells this differently and it is easy to forget which is which, so accept every spelling and route to the agent's own mechanism in _yolo_command_flags / the config writers.
_YOLO_OPTION = typer.Option(
    False,
    "--yolo",
    "--dangerously-skip-permissions",
    "--dangerously-bypass-approvals-and-sandbox",
    rich_help_panel = _PANEL_SESSION,
    help = "Auto-approve all tool actions for this session; routed to the agent's own "
    "flag/config. Any of the three spellings works for any agent.",
)


_PERSIST_OPTION = typer.Option(
    False,
    "--persist/--no-persist",
    rich_help_panel = _PANEL_SESSION,
    help = (
        "Keep this agent's agent-switch session dir so you can resume it later. "
        "codex/pi/dsh have their whole home relocated into a session dir "
        "that is a throwaway temp dir (wiped on exit) by default; with --persist it "
        "lives under the agent-switch agents dir and survives, so their own resume can reopen "
        "it. claude and opencode keep sessions in your own stores (~/.claude, "
        "~/.local/share/opencode), so they already resume regardless. To reopen a "
        "session, pass the agent's own resume command through, e.g. "
        "`agent-switch codex --persist resume` or `claude --resume <id>`; those flow to "
        "the agent unchanged."
    ),
)


_AS_SUBAGENT_OPTION = typer.Option(
    False,
    "--as-subagent",
    rich_help_panel = _PANEL_SESSION,
    help = "Keep the coding agent's current model and add the local model as a subagent.",
)


_MCP_OPTION = typer.Option(
    None,
    "--mcp",
    metavar = "NAME",
    rich_help_panel = _PANEL_SESSION,
    help = "Mount this MCP server from the agent-switch registry (mcp.json in its home) for "
    "this session only; repeat the flag for more. They replace the agent's own MCP servers.",
)


_MCP_ALL_OPTION = typer.Option(
    False,
    "--mcp-all",
    rich_help_panel = _PANEL_SESSION,
    help = "Mount every MCP server in the agent-switch registry for this session only.",
)


_COMPACT_AT_OPTION = typer.Option(
    None,
    "--compact-at",
    min = 0.5,
    max = 0.95,
    rich_help_panel = _PANEL_SESSION,
    help = (
        "Fraction of the context window that triggers the agent's auto-compaction: "
        "0.85 starts it once 85% of the context is used. Applies to Claude Code, Codex, "
        "OpenCode, Pi and DeepSeek Harness's headless profile (its web profile ignores it, "
        "with a warning); Claude Code scales its own effective window, so there the ratio can "
        "only pull its built-in trigger earlier. Unset keeps each agent's current behavior."
    ),
)


_HEADER_NAME_TOKEN = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")  # RFC 7230 tchar


_HEADER_VALUE_TEXT = re.compile(r"[\t\x20-\x7e]*")  # RFC 7230 field VCHAR + SP + HTAB


def parse_headers(values: Optional[list]) -> dict:
    """Validate and merge repeated --header NAME=VALUE pairs; a later name wins, case-insensitively."""
    headers = {}
    for value in values or []:
        name, separator, header_value = value.partition("=")
        if not separator:
            _fail(f"--header needs NAME=VALUE, got {value!r}.")
        if not _HEADER_NAME_TOKEN.fullmatch(name):
            _fail(f"--header name {name!r} has characters an HTTP header name cannot carry.")
        if "\r" in header_value or "\n" in header_value:
            _fail(f"--header value for {name!r} cannot carry a line break.")
        if not _HEADER_VALUE_TEXT.fullmatch(header_value):
            _fail(f"--header value for {name!r} has characters an HTTP header value cannot carry.")
        headers = {k: v for k, v in headers.items() if k.lower() != name.lower()}
        headers[name] = header_value
    return headers


# Per-agent CLI flag for "run tools without prompting". OpenCode (native --auto is command-scoped, handled below) is absent from this prefix map.
_YOLO_COMMAND_FLAGS = {
    "claude": ["--dangerously-skip-permissions"],
    "codex": ["--dangerously-bypass-approvals-and-sandbox"],
    # Pi never prompts per tool call; its only approval gate is project trust, so -a (trust project resources) is the closest "do not ask me" equivalent.
    "pi": ["--approve"],
}


def _yolo_command_flags(agent: str, yolo: bool) -> list:
    # .get so a config-based agent (or a typo) yields no flag instead of a KeyError.
    return _YOLO_COMMAND_FLAGS.get(agent, []) if yolo else []


class LoadOptions(NamedTuple):
    """Model-load knobs used when --model triggers a load on the server."""

    max_seq_length: int = 0
    # --no-model-load: never load, reload or unload a model on the server.
    allow_load: bool = True


class ServerOptions(NamedTuple):
    """Start flags: carried fields ride in the agent's requests."""

    reasoning: Optional[Literal["on", "off", "auto"]] = None
    reasoning_effort: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    # Fields the agent's own config sends with each request, kept off the server.
    carried: frozenset = frozenset()
    # The server the body goes to; others name some fields differently or drop them. None keeps
    # the body untranslated.
    provider: Optional[str] = None

    def sent_by_agent(self) -> frozenset:
        # A request cannot ask for the template default back, so auto stays a server setting.
        unsent = {"reasoning"} if self.reasoning == "auto" else set()
        if self.reasoning_effort not in _REASONING_EFFORTS:
            unsent.add("reasoning_effort")
        return self.carried - unsent

    def request_body(self) -> dict:
        sent = self.sent_by_agent()
        body = {
            name: getattr(self, name)
            for name in _SAMPLING_FIELDS
            if name in sent and getattr(self, name) is not None
        }
        if "reasoning" in sent and self.reasoning in ("on", "off"):
            body["enable_thinking"] = self.reasoning == "on"
        if "reasoning_effort" in sent:
            body["reasoning_effort"] = self.reasoning_effort
        return providers.request_body(self.provider, body)[0] if self.provider else body


_SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
)


_REASONING_FIELDS = frozenset({"reasoning", "reasoning_effort"})


_ALL_REQUEST_FIELDS = frozenset(_SAMPLING_FIELDS) | _REASONING_FIELDS


_REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "max", "xhigh")


def _split_repo_variant(model: str) -> tuple:
    """Split ``org/name:QUANT`` into ``("org/name", "QUANT")``, the ``:variant`` shorthand llama.cpp and Ollama accept. Local paths, Windows drive letters and ids without a ``:`` pass through unchanged."""
    s = (model or "").strip()
    if not s or s.startswith(("/", "./", "../", "~")) or s == ".":
        return s, None
    if len(s) >= 2 and s[1] == ":" and s[0].isalpha():  # Windows drive, e.g. C:\models\x
        return s, None
    if ":" not in s:
        return s, None
    repo, _, variant = s.rpartition(":")
    if not repo or not variant or "/" in variant:
        return s, None
    return repo, variant


def _looks_like_model(token: str) -> bool:
    """True for a bare `org/name(:variant)` hub id that is not a flag or a local path. Reuses `_is_hub_model_id`, so a relative dir like `owner/repo` that actually exists is left for the agent instead of being taken as a model; a non-existent `org/name` is treated as a hub id."""
    if not token or token.startswith("-") or " " in token:
        return False
    repo, _ = _split_repo_variant(token)
    return _is_hub_model_id(repo)


def _consume_positional_model(model: Optional[str], args: list) -> tuple:
    """Route a leading `org/name` positional to --model when --model was not given. Only the FIRST token is considered so an option value like `--profile owner/repo` is never stolen, and only when --model is absent so an explicit --model always wins. Returns (model, remaining_args) with the consumed token removed from the passthrough."""
    args = list(args)
    if model or not args or not _looks_like_model(args[0]):
        return model, args
    return args[0], args[1:]


def _fail(message: str) -> NoReturn:
    typer.echo(message, err = True)
    raise typer.Exit(code = 1)


_HF_REPO_ID_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _is_hub_model_id(value: object) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if "\\" in text:
        return False
    if text.startswith(("/", "./", "../", "~")):
        return False
    if len(text) >= 2 and text[1] == ":" and text[0].isalpha():
        return False
    # A hub id is exactly "namespace/name" over a restricted charset. Anything with extra path segments (a relative path such as models/Llama/Foo.gguf) is not a hub id. The existence probe below leaves a local directory of that shape to the agent.
    parts = text.split("/")
    if len(parts) != 2:
        return False
    if any(part in ("", ".", "..") or not _HF_REPO_ID_SEGMENT_RE.match(part) for part in parts):
        return False
    try:
        if Path(os.path.expanduser(text)).exists():
            return False
    except OSError:
        return False
    return True


def _check_compact_at(compact_at: Optional[float], model: dict) -> None:
    """--compact-at scales off the server-reported window; without one there is nothing to scale."""
    if compact_at is not None and not (model.get("context_length") or model.get("max_context_length")):
        typer.echo(
            "Warning: the server did not report the model's context length, so --compact-at is ignored.",
            err = True,
        )


def opencode_output_limit(window: int, max_tokens: Optional[int] = None) -> int:
    if max_tokens:
        return max(1, min(int(max_tokens), window // 2))
    return max(1, min(window // 4, _OPENCODE_OUTPUT_TOKEN_MAX))


def _agent_output_limit(window: int, max_tokens: Optional[int]) -> int:
    output = opencode_output_limit(window, max_tokens)
    if max_tokens and output < max_tokens:
        typer.echo(
            f"Warning: --max-tokens {max_tokens} leaves too little of the {window:,}-token "
            f"context for the conversation; using {output:,}.",
            err = True,
        )
    return output


def _get_compaction_reserve(window: int, ratio: float) -> int:
    return max(1, int(window * (1 - ratio)))
