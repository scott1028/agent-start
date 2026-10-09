# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`agent-switch <agent>` — launch a coding agent against a running model server."""

import base64
import contextlib
import errno
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Literal, NamedTuple, NoReturn, Optional

import click
import typer
from typer.core import TyperCommand

from agent_switch._coding_agents import (
    deepseek_harness_executables_on_path,
    is_deepseek_harness_executable,
)
from agent_switch import providers
from agent_switch.providers.types import ProviderError, Target
from agent_switch.providers.utils import get_has_custom_authorization

start_app = typer.Typer(
    help = "Start a coding agent against a local model server: Ollama, LM Studio, "
    "llama-server, vLLM or any OpenAI-compatible server.",
    no_args_is_help = True,
    context_settings = {"help_option_names": ["-h", "--help"]},
)

_CODEX_PROFILE = "agent_switch"
_CODEX_ENV_KEY = "AGENT_SWITCH_AUTH_TOKEN"
# Codex treats an SSE stream with no bytes for this long as lost, cancels it and reconnects. Its default is 300000 (5 minutes), measured against the WHOLE quiet period, and llama-server sends nothing at all while it processes the prompt. A local CPU host chews through a prompt at low tens of tokens a second and Codex's own preamble is several thousand tokens before the user has typed anything: 16.1 tok/s measured on a 2-core box means ~460s of silence for a ~7300-token first turn, so the default trips before the first token exists. The reconnect is worse than the wait, because llama-server hands the retry a different parallel slot whose KV cache shares no prefix, so each attempt restarts prompt processing from zero and the five retries can never converge. Observed as a request completing in exactly 300056ms with `Reconnecting... 1/5` and no turn ever finishing. 20 minutes here, sized to be longer than a slow local first turn rather than to any server-side budget: nothing here bounds generation, and a genuinely dead stream is still caught, just later.
_CODEX_STREAM_IDLE_TIMEOUT_MS = 1_200_000
_DSH_PROVIDER = "agent-switch"
_DSH_ENV_KEY = "AGENT_SWITCH_API_KEY"
_DSH_PATCH_FILE = "agent-switch.patch.yml"
_DSH_PACKAGE = "@deepseek-ai/dsh"
# dsh picks its sandbox+approval preset from DSH_PERMISSION_MODE via ??, so omitting it would inherit a danger-full-access exported in the parent shell, and "" is not unset to ??. Pin the mode in both directions instead of only setting it for --yolo.
_DSH_SAFE_PERMISSION_MODE = "workspace-write"
_DSH_YOLO_PERMISSION_MODE = "danger-full-access"
_PI_PROVIDER = "agent-switch"
_SUBAGENT_NAME = "local"
_SUBAGENT_DESCRIPTION = (
    "Local coding subagent running on a local model for debugging, implementation, and "
    "codebase research. Use when the user asks to spawn a local agent."
)
_SUBAGENT_INSTRUCTIONS = (
    "You are a local coding subagent running on a local model. Complete the assigned task directly, "
    "use the available tools when useful, verify your work, and return a concise result to the "
    "parent agent."
)
_SUBAGENT_PLAN_DESCRIPTION = (
    "Read-only local coding subagent running on a local model for planning and codebase research. "
    "Use this local agent when Claude is in plan mode."
)
_SUBAGENT_PLAN_INSTRUCTIONS = (
    "You are a read-only local coding subagent running on a local model. Investigate the assigned "
    "task with read-only tools, produce a concrete plan or answer, and return a concise result "
    "to the parent agent. Do not modify files."
)
_CLAUDE_SUBAGENT_MCP_MODULE = "agent_switch.claude_subagent_mcp"
_CLAUDE_SUBAGENT_SETTINGS_ENV = "AGENT_SWITCH_CLAUDE_SUBAGENT_SETTINGS"
_CLAUDE_SUBAGENT_TOOL = "mcp__plugin_local-agent_local__local_agent"
_CLAUDE_SUBAGENT_PLAN_TOOL = "mcp__plugin_local-agent_local__local_plan_agent"
_CODEX_SUBAGENT_MCP_MODULE = "agent_switch.codex_subagent_mcp"
_CODEX_SUBAGENT_MCP_SERVER = "local_agent"
_CODEX_SUBAGENT_MCP_TOOL = "spawn_local_agent"
_CODEX_SUBAGENT_CONFIG_ENV = "AGENT_SWITCH_CODEX_SUBAGENT_CONFIG"
_CODEX_PARENT_OVERLAY_MANIFEST = ".agent-switch-parent-overlay.json"
_CODEX_EPHEMERAL_STALE_SECONDS = 24 * 60 * 60
_CODEX_EPHEMERAL_HEARTBEAT_SECONDS = 60
_CODEX_SUBAGENT_TOOL_DESCRIPTION = (
    f"{_SUBAGENT_DESCRIPTION} Use this tool instead of the built-in spawn_agent tool for those "
    "requests. Other subagent requests may use the built-in tools normally."
)
_CODEX_SUBAGENT_ROUTING_INSTRUCTIONS = (
    "When the user asks to spawn a local agent, you must call the "
    "spawn_local_agent MCP tool once with the complete task. Do not answer, simulate the "
    "result, call wait, or use a built-in subagent before calling the tool. Use built-in "
    "subagents for other delegation requests."
)
_PI_SUBAGENT_EXTENSION = Path(__file__).parent / "pi_subagent.ts"
_PI_USER_RESOURCE_DIRS = ("extensions", "skills", "prompts", "themes", "npm", "git")
_PI_USER_RESOURCE_SETTINGS = ("packages", "extensions", "skills", "prompts", "themes")
_PI_USER_VERBATIM_SETTINGS = ("npmCommand",)
_PI_USER_RESOURCES_MANIFEST = ".agent-switch-user-resources.json"
# OpenCode selects a model by "<providerID>/<modelID>". Use a dedicated id to avoid colliding with a user's providers; provider filters are set in the launch-time overlay.
_OPENCODE_PROVIDER = "agent-switch"
# OpenCode sends min(limit.output, this) as max_tokens unless the env var below raises it.
_OPENCODE_OUTPUT_TOKEN_MAX = 32_000
_OPENCODE_OUTPUT_TOKEN_MAX_ENV = "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"
_PROVIDER_HEADER = f"[model_providers.{_CODEX_PROFILE}]"
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


# provider routing overrides ANTHROPIC_BASE_URL and would bypass the local server (#9864).
_CLAUDE_ENV_UNSET = (
    "ANTHROPIC_API_KEY",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_UNIX_SOCKET",
    "CLAUDE_CODE_USE_FOUNDRY",
    "ANTHROPIC_FOUNDRY_BASE_URL",
    "ANTHROPIC_FOUNDRY_RESOURCE",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_ANTHROPIC_AWS",
    "CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD",
    "CLAUDE_CODE_USE_MANTLE",
)
_CODEX_ENV_UNSET = ("OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN")

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


# Subcommands that reject --auto (OpenCode exposes it only on the default TUI and `run`), so `opencode serve --auto` is never emitted. Includes console/generate, hidden from `opencode --help` but still registered. Unknown first positionals are TUI paths and get --auto.
_OPENCODE_NON_AUTO_SUBCOMMANDS = frozenset(
    "completion acp mcp attach debug providers auth agent upgrade uninstall serve web "
    "models stats export import github pr session plugin plug db console generate".split()
)
_OPENCODE_V2_SUBCOMMANDS = frozenset(
    "acp api debug console auth mcp plugin models export import mini run service pair serve".split()
)
_OPENCODE_V2_STANDALONE_SUBCOMMANDS = frozenset("api models export import mini run".split())
_OPENCODE_GLOBAL_BOOLEAN_OPTIONS = frozenset(
    "-h --help -v --version --print-logs --pure --mdns --standalone --wizard".split()
)
_OPENCODE_GLOBAL_VALUE_OPTIONS = frozenset(
    "--log-level --port --hostname --mdns-domain --cors --server --completions --cpu-profile".split()
)
_OPENCODE_NATIVE_AUTO_MIN_VERSION = (1, 17, 12)


def _opencode_command() -> tuple[str, bool]:
    resolved_v2 = _which_with_install_dirs("opencode2")
    if resolved_v2:
        return resolved_v2, True
    return "opencode", False


def _opencode_supports_native_auto(command: str = "opencode") -> bool:
    if Path(command).stem.lower() == "opencode2":
        return True
    executable = _which_with_install_dirs(command)
    if executable is None:
        # No local binary: a --no-launch recipe may run elsewhere, and _run installs the current release on launch, so either way assume native --auto is available.
        return True
    try:
        output = subprocess.check_output(
            [executable, "--version"],
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 10,
            stderr = subprocess.DEVNULL,
            env = _probe_env(),
        )
    except Exception:
        return False
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", output)
    return bool(match) and tuple(int(part) for part in match.groups()) >= (
        _OPENCODE_NATIVE_AUTO_MIN_VERSION
    )


def _opencode_subcommand(args: list[str]) -> tuple[Optional[str], Optional[int]]:
    """Return an explicit OpenCode subcommand and its index after global options."""
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--":
            return None, None
        if arg in _OPENCODE_GLOBAL_BOOLEAN_OPTIONS:
            index += 1
            continue
        if arg in _OPENCODE_GLOBAL_VALUE_OPTIONS:
            index += 2
            continue
        if any(arg.startswith(f"{option}=") for option in _OPENCODE_GLOBAL_VALUE_OPTIONS):
            index += 1
            continue
        # A non-global option (--session) is a TUI flag; stop before its value is mistaken for a subcommand.
        if arg.startswith("-"):
            return None, None
        return arg, index
    return None, None


def _opencode_native_auto_args(
    args: list[str],
    yolo: bool,
    *,
    v2: bool = False,
) -> tuple[list[str], bool]:
    """Add OpenCode's native --auto when the selected command supports it."""
    routed = list(args)
    if not yolo:
        return routed, False
    subcommand, _ = _opencode_subcommand(routed)
    if v2 and subcommand in _OPENCODE_V2_SUBCOMMANDS and subcommand != "run":
        return routed, False
    if not v2 and subcommand in _OPENCODE_NON_AUTO_SUBCOMMANDS:
        return routed, False
    separator = routed.index("--") if "--" in routed else len(routed)
    # --mini's runMini TUI forces auto=false and never forwards --auto, so appending it is useless; fall back to the config permission block so --yolo still auto-approves.
    if any(arg == "--mini" or arg.startswith("--mini=") for arg in routed[:separator]):
        return routed, False
    if "--auto" not in routed[:separator]:
        routed.insert(separator, "--auto")
    return routed, True


def _opencode_v2_standalone_args(args: list[str]) -> list[str]:
    """Keep session-only config on the V2 server that consumes it."""
    routed = list(args)
    separator = routed.index("--") if "--" in routed else len(routed)
    head = routed[:separator]
    if (
        "--standalone" in head
        or "--server" in head
        or any(arg.startswith("--server=") for arg in head)
    ):
        return routed
    subcommand, subcommand_index = _opencode_subcommand(routed)
    if (
        subcommand in _OPENCODE_V2_SUBCOMMANDS
        and subcommand not in _OPENCODE_V2_STANDALONE_SUBCOMMANDS
    ):
        return routed
    insert_at = (
        subcommand_index + 1 if subcommand in _OPENCODE_V2_STANDALONE_SUBCOMMANDS else separator
    )
    routed.insert(insert_at, "--standalone")
    return routed


def _npm_install_hint(package: str, *, ignore_scripts: bool = False) -> str:
    parts = ["npm", "install", "-g"]
    if os.name != "nt":
        # No home (bare container UID): fall back to npm's own prefix instead of failing.
        try:
            parts.extend(("--prefix", str(Path.home() / ".local")))
        except (RuntimeError, OSError):
            pass
    if ignore_scripts:
        parts.append("--ignore-scripts")
    parts.append(package)
    if os.name == "nt":
        return " ".join(_powershell_quote(part) for part in parts)
    return shlex.join(parts)


_DSH_LAUNCHER_ARGS = frozenset(
    "--profile --patch --dump-config --dump-default-config -V --version plugin web".split()
)


# Launcher invocations that boot no profile, so they take no --patch overlay.
_DSH_NO_PROFILE_ARGS = frozenset("-V --version plugin".split())


def _dsh_command(args: list[str], patch: Optional[str] = None) -> list[str]:
    head = args[0] if args else ""
    if head in _DSH_LAUNCHER_ARGS or head.startswith(("--profile=", "--patch=")):
        command = ["dsh", *args]
    else:
        command = ["dsh", "web", *args]
    if patch is not None and command[1] not in _DSH_NO_PROFILE_ARGS:
        # `dsh <name>` only expands to `--profile <name>` when the name comes first, so the
        # overlay goes after a bare profile name and ahead of a leading launcher option.
        at = 1 if command[1].startswith("-") else 2
        command[at:at] = ["--patch", patch]
    return command


# Launcher options that take a value, as `--name value` or `--name=value`.
_DSH_VALUE_ARGS = frozenset("--profile --patch --from-default-profile".split())
# Launcher options that print the profile tree and exit instead of booting it.
_DSH_DUMP_ARGS = frozenset("--dump-config --dump-default-config".split())


def _get_dsh_boot_profile(command: list[str]) -> Optional[str]:
    """The profile a `_dsh_command` argv boots, or None when it boots none (plugin, version, dump).

    Like the launcher, read only its own leading options: they end at `--` or at the first
    other argument, and everything from there on belongs to the app. The `web` alias takes
    only --patch and the dumps.
    """
    args = command[1:]
    if args[:1] == ["web"]:
        profile, value_args, exit_args, args = "web", {"--patch"}, _DSH_DUMP_ARGS, args[1:]
    else:
        profile, value_args, exit_args = None, _DSH_VALUE_ARGS, _DSH_DUMP_ARGS | {"-V", "--version"}
    while args:
        name, equals, value = args[0].partition("=")
        if name in exit_args:
            return None
        if name not in value_args:
            break
        if not equals:
            value = args[1] if len(args) > 1 else None
            args = args[1:]
        args = args[1:]
        if name == "--profile":
            profile = value
    return profile


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
_CODEX_REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")


def _codex_reasoning_effort(
    reasoning: Optional[str], reasoning_effort: Optional[str]
) -> Optional[str]:
    if reasoning == "off":
        return "none"
    if reasoning_effort in _CODEX_REASONING_EFFORTS:
        return reasoning_effort
    return None


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


def _reject_as_subagent(agent: str, args: list) -> None:
    # Reject early, or the flag reaches the agent binary after the server loaded the model.
    if any(arg == "--as-subagent" or arg.startswith("--as-subagent=") for arg in args):
        _fail(f"--as-subagent is not supported for {agent}.")


def _agent_switch_home() -> Path:
    configured = os.environ.get("AGENT_SWITCH_HOME")
    return Path(configured) if configured else Path.home() / ".agent-switch"


def _provider_key_cache_path() -> Path:
    """API keys given with --api-key, remembered per server."""
    return _agent_switch_home() / "api_keys.json"


def _read_cache(cache: Path) -> dict:
    try:
        data = json.loads(cache.read_text(encoding = "utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _saved_keys(servers: object, base: str) -> list:
    # Tolerate a corrupt file: anything but {"saved": [str, ...]} for this base reads as no keys.
    entry = servers.get(base) if isinstance(servers, dict) else None
    saved = entry.get("saved") if isinstance(entry, dict) else None
    return [k for k in saved if isinstance(k, str)] if isinstance(saved, list) else []


def _cached_keys(cache: Path, base: str) -> list:
    # Keys are scoped per server, so a key given for one base is never sent to another.
    return _saved_keys(_read_cache(cache).get("servers"), base)


def _write_private_json(path: Path, data: dict) -> None:
    # O_CREAT with 0o600 so a file holding an API key is never world-readable, even briefly (existing files keep whatever perms the user set).
    path.parent.mkdir(parents = True, exist_ok = True, mode = 0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps(data, indent = 2) + "\n")


def _write_private_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents = True, exist_ok = True, mode = 0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding = "utf-8") as handle:
        handle.write(text)


def _read_yaml_object(path: Path) -> Optional[dict]:
    import yaml

    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding = "utf-8"))
    except (yaml.YAMLError, OSError):
        return None
    if data is None:
        return {}
    return data if isinstance(data, dict) else None


def _read_json_object(path: Path) -> Optional[dict]:
    # {} when missing, None when it cannot be parsed as an object, so the caller leaves a user-managed file untouched rather than clobbering it.
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding = "utf-8"))
    except (ValueError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _subdict(parent: dict, key: str) -> dict:
    child = parent.get(key)
    if not isinstance(child, dict):
        child = parent[key] = {}
    return child


def _remember_key(cache: Path, base: str, key: str) -> None:
    data = _read_cache(cache)
    servers = data.get("servers")
    if not isinstance(servers, dict):
        servers = data["servers"] = {}
    new_entry = {"saved": ([key] + [k for k in _saved_keys(servers, base) if k != key])[:8]}
    if servers.get(base) == new_entry:
        return
    servers[base] = new_entry
    try:
        _write_private_json(cache, data)
    except OSError:
        pass  # worst case the next launch needs --api-key again


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


_DYNAMIC_SECTIONS_FLAG = "--exclude-dynamic-system-prompt-sections"


def _claude_settings_overlay(model_id: str, local_env: Optional[dict] = None) -> str:
    # Command-tier pins beat user/project settings, which Claude applies after the process env.
    settings_env = {name: "" for name in _CLAUDE_ENV_UNSET}
    settings_env.update(local_env or {})
    settings_env.update(
        {
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            "CLAUDE_CODE_SUBAGENT_MODEL": "inherit",
        }
    )
    return json.dumps(
        {
            "env": settings_env,
            "availableModels": [model_id],
        }
    )


def _write_claude_settings(path: Path, model_id: str, local_env: dict) -> Path:
    overlay = _claude_settings_overlay(model_id, local_env)
    digest = hashlib.sha256(overlay.encode("utf-8")).hexdigest()[:16]
    settings = path / f"settings-{digest}.json"
    _write_private_text(settings, overlay)
    return settings


def _claude_version() -> Optional[tuple]:
    # None means no local `claude` (a --no-launch printout for another machine; assume a current build). An unparseable version is treated as too old for the new flags.
    executable = _which_with_install_dirs("claude")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 10,
            env = _probe_env(),
        )
        # Pull the X.Y.Z out of the output rather than assuming it is the first token: claude prints it first today ("2.1.98 (Claude Code)"), but a format change should not silently drop the optimization flags. No match falls through to "too old", same as an unparseable version.
        match = re.search(r"(\d+)\.(\d+)\.(\d+)", result.stdout)
        return tuple(int(part) for part in match.groups()) if match else (0,)
    except Exception:
        return (0,)


def _claude_flags(model_id: str, settings: Optional[str] = None) -> list:
    # KV-cache-preserving flags: move per-session context out of the system prompt and pass the session overlay. claude < 2.1.98 rejects the dynamic-sections flag but already supports --settings; no local binary means a printout for another machine, so assume a current build.
    version = _claude_version()
    settings_flags = ["--settings", settings or _claude_settings_overlay(model_id)]
    if version is not None and version < (2, 1, 98):
        return settings_flags
    return [_DYNAMIC_SECTIONS_FLAG, *settings_flags]


def _claude_local_command(model_id: str, settings: str, yolo: bool, passthrough: list) -> list:
    local_args = [
        "--model",
        model_id,
        *_claude_flags(model_id, settings),
        *_yolo_command_flags("claude", yolo),
    ]
    forwarded = list(passthrough)
    separator = forwarded.index("--") if "--" in forwarded else len(forwarded)
    before_separator = forwarded[:separator]
    forwarded_settings = []
    remaining = []
    index = 0
    while index < len(before_separator):
        arg = before_separator[index]
        if arg == "--settings" and index + 1 < len(before_separator):
            forwarded_settings.extend(before_separator[index : index + 2])
            index += 2
            continue
        if arg.startswith("--settings="):
            forwarded_settings.append(arg)
        else:
            remaining.append(arg)
        index += 1
    return [
        "claude",
        *forwarded_settings,
        *local_args,
        *remaining,
        *forwarded[separator:],
    ]


def _check_compact_at(compact_at: Optional[float], model: dict) -> None:
    """--compact-at scales off the server-reported window; without one there is nothing to scale."""
    if compact_at is not None and not (model.get("context_length") or model.get("max_context_length")):
        typer.echo(
            "Warning: the server did not report the model's context length, so --compact-at is ignored.",
            err = True,
        )


def _claude_local_env(
    base: str,
    key: str,
    entry: dict,
    extra_body: Optional[dict] = None,
    headers: Optional[dict] = None,
    compact_at: Optional[float] = None,
) -> dict:
    """Build the local endpoint, cache, display, and compaction environment."""
    model_id = entry["id"]
    env = {"ANTHROPIC_BASE_URL": base}
    # claude outranks ANTHROPIC_CUSTOM_HEADERS with ANTHROPIC_AUTH_TOKEN for Authorization (verified on 2.1.291),
    # so a custom Authorization only wins when the token is pinned empty; the settings overlay applies it
    # after the process env and user settings, which is where an inherited or user-set token would come back.
    env["ANTHROPIC_AUTH_TOKEN"] = "" if get_has_custom_authorization(headers or {}) else key
    env.update({
        "ANTHROPIC_MODEL": model_id,
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
        # Per-tool countdown reminders change the system prefix on local models.
        "CLAUDE_CODE_TOTAL_TOKENS_REMINDER": "off",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "CLAUDE_CODE_NO_FLICKER": "1",
    })
    if headers:
        env["ANTHROPIC_CUSTOM_HEADERS"] = "\n".join(f"{name}: {value}" for name, value in headers.items())
    window = entry.get("context_length") or entry.get("max_context_length")
    if window:
        # claude assumes 200k for a model id it does not recognize, and clamps AUTO_COMPACT_WINDOW to [100k, that]. MAX_CONTEXT_TOKENS sets the window itself.
        env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(int(window))
        env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(int(window))
        env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] = (
            str(int(round(compact_at * 100))) if compact_at is not None else "90"
        )
    if extra_body:
        env["CLAUDE_CODE_EXTRA_BODY"] = json.dumps(extra_body)
    return env


def _codex_provider_table(base: str, headers: Optional[dict] = None) -> str:
    table = (
        f"{_PROVIDER_HEADER}\n"
        'name = "agent-switch"\n'
        f"base_url = {json.dumps(base + '/v1')}\n"
    )
    # A custom Authorization header replaces the Bearer <env_key> auth; keeping both would send two.
    if not get_has_custom_authorization(headers or {}):
        table += f'env_key = "{_CODEX_ENV_KEY}"\n'
    table += (
        'wire_api = "responses"\n'
        "requires_openai_auth = false\n"
        f"stream_idle_timeout_ms = {_CODEX_STREAM_IDLE_TIMEOUT_MS}\n"
    )
    if headers:
        pairs = ", ".join(f"{json.dumps(name)} = {json.dumps(value)}" for name, value in headers.items())
        table += f"http_headers = {{ {pairs} }}\n"
    return table


_CODEX_PROVIDER_TABLES = (_PROVIDER_HEADER, _PROVIDER_HEADER[:-1] + ".")


def _merge_codex_config(existing: str, base: str, headers: Optional[dict] = None) -> str:
    chunks = re.split(r"(?m)^(?=\[)", existing)  # preamble, then one chunk per table
    if not re.search(r"(?m)^\s*oss_provider\s*=", chunks[0]):
        if chunks[0] and not chunks[0].endswith("\n"):
            chunks[0] += "\n"
        chunks[0] += f'oss_provider = "{_CODEX_PROFILE}"\n'
    text = "".join(c for c in chunks if not c.startswith(_CODEX_PROVIDER_TABLES))
    if not text.endswith("\n"):
        text += "\n"
    if not text.endswith("\n\n"):
        text += "\n"
    return text + _codex_provider_table(base, headers)


# Keep custom-model behavior aligned with Codex's own unknown-model fallback. This Apache-2.0 prompt is copied from openai/codex rust-v0.144.0 models-manager/prompt.md.
_CODEX_FALLBACK_PROMPT = Path(__file__).parent / "codex_fallback_prompt.md"
_CODEX_MODEL_CATALOG_MIN_VERSION = (0, 110, 0)
_CODEX_PATCH_LINE_ENDINGS_MIN_VERSION = (0, 148, 0)
# Older Codex sends no reasoning for a model without reasoning summaries, and older Pi has no samplingParams.
_CODEX_REASONING_REQUEST_MIN_VERSION = (0, 145, 0)
_PI_SAMPLING_PARAMS_MIN_VERSION = (0, 84, 0)


def _codex_executable_version(executable: str) -> Optional[tuple[int, int, int]]:
    try:
        output = subprocess.check_output(
            [executable, "--version"],
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 10,
            stderr = subprocess.DEVNULL,
            env = _probe_env(),
        )
    except Exception:
        return None
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", output)
    return tuple(int(part) for part in match.groups()) if match else None


def _codex_supports_model_catalog() -> bool:
    executable = _which_with_install_dirs("codex")
    if executable is None:
        # A --no-launch recipe may be copied to another machine; assume a current Codex.
        return True
    version = _codex_executable_version(executable)
    return version is not None and version >= _CODEX_MODEL_CATALOG_MIN_VERSION


def _agent_version_at_least(command: str, minimum: tuple) -> bool:
    executable = _which_with_install_dirs(command)
    if executable is None:
        # Only --no-launch gets here without the agent; its recipe may run elsewhere.
        return True
    version = _codex_executable_version(executable)
    return version is not None and version >= minimum


def _codex_supports_patch_line_endings() -> bool:
    executable = _which_with_install_dirs("codex")
    if executable is None:
        # A normal launch installs Codex after this check; no-launch may run elsewhere.
        return True
    version = _codex_executable_version(executable)
    return version is not None and version >= _CODEX_PATCH_LINE_ENDINGS_MIN_VERSION


def _codex_model_catalog(model: dict) -> dict:
    """Return conservative metadata for a local model unknown to Codex's built-in catalog."""
    model_id = model["id"]
    window = model.get("context_length") or model.get("max_context_length")
    entry = {
        "slug": model_id,
        "display_name": model_id,
        "description": "Model served by a local server",
        "supported_reasoning_levels": [],
        "shell_type": "default",
        "visibility": "none",
        "supported_in_api": True,
        "priority": 99,
        "availability_nux": None,
        "upgrade": None,
        "base_instructions": _CODEX_FALLBACK_PROMPT.read_text(encoding = "utf-8"),
        "supports_reasoning_summaries": False,
        "supports_reasoning_summary_parameter": False,
        "support_verbosity": False,
        "default_verbosity": None,
        "apply_patch_tool_type": "freeform",
        "truncation_policy": {"mode": "bytes", "limit": 10_000},
        "supports_parallel_tool_calls": False,
        "experimental_supported_tools": [],
    }
    if window:
        entry["context_window"] = int(window)
        entry["max_context_window"] = int(window)
    return {"models": [entry]}


def write_codex_config(
    base: str,
    model: dict,
    home: Path,
    reasoning_effort: Optional[str] = None,
    headers: Optional[dict] = None,
    compact_at: Optional[float] = None,
) -> None:
    home.mkdir(parents = True, exist_ok = True)

    config = home / "config.toml"
    existing = config.read_text(encoding = "utf-8") if config.exists() else ""
    merged = _merge_codex_config(existing, base, headers)
    if merged != existing:
        config.write_text(merged, encoding = "utf-8")
        # http_headers can carry a secret Authorization, so keep the file owner-only like the JSON configs.
        config.chmod(0o600)
        typer.echo(f"Updated {config}")

    # oss_provider here too: codex --oss picks the provider from it, and the profile layer must beat a user-set value ("ollama") in config.toml.
    profile_text = (
        f'oss_provider = "{_CODEX_PROFILE}"\n'
        f'model_provider = "{_CODEX_PROFILE}"\n'
        f"model = {json.dumps(model['id'])}\n"
    )
    if _codex_supports_patch_line_endings():
        profile_text += (
            "suppress_unstable_features_warning = true\n"
            "features.apply_patch_preserve_line_endings = true\n"
        )
    if _codex_supports_model_catalog() and _CODEX_FALLBACK_PROMPT.is_file():
        catalog = home / "model-catalog.json"
        catalog_text = json.dumps(_codex_model_catalog(model), indent = 2) + "\n"
        if not catalog.exists() or catalog.read_text(encoding = "utf-8") != catalog_text:
            catalog.write_text(catalog_text, encoding = "utf-8")
            typer.echo(f"Updated {catalog}")
        # Resolve relative to the profile file. This also survives WSL launching a Windows Codex binary, where a Linux absolute path inside TOML would not be usable.
        profile_text += f"model_catalog_json = {json.dumps(catalog.name)}\n"

    window = model.get("context_length") or model.get("max_context_length")
    if window:
        profile_text += f"model_context_window = {int(window)}\n"
        if compact_at is not None:
            profile_text += f"model_auto_compact_token_limit = {int(int(window) * compact_at)}\n"
    if reasoning_effort:
        profile_text += f"model_reasoning_effort = {json.dumps(reasoning_effort)}\n"
    profile = home / f"{_CODEX_PROFILE}.config.toml"
    if not profile.exists() or profile.read_text(encoding = "utf-8") != profile_text:
        profile.write_text(profile_text, encoding = "utf-8")
        typer.echo(f"Updated {profile}")


def write_codex_subagent_bridge(
    base: str,
    key: str,
    model: dict,
    home: Path,
    *,
    yolo: bool,
    reasoning_effort: Optional[str] = None,
    headers: Optional[dict] = None,
    compact_at: Optional[float] = None,
) -> Path:
    """Write private config for an explicit local Codex child launched through MCP."""
    child_home = home / "child"
    write_codex_config(base, model, child_home, reasoning_effort, headers, compact_at)
    path = home / "subagent.json"
    _write_private_json(
        path,
        {
            "api_key": key,
            "codex_home": str(child_home),
            "bypass_permissions": yolo,
        },
    )
    return path


def _wsl_windows_user_profile(executable: str) -> Path:
    """Return the Windows user profile as a path accessible from WSL."""
    profile = os.environ.get("USERPROFILE", "").strip()
    if not profile:
        try:
            profile = subprocess.check_output(
                ["cmd.exe", "/d", "/c", "echo %USERPROFILE%"],
                text = True,
                encoding = "utf-8",
                # The path is the value: a corrupted home is worse than a loud failure.
                errors = "strict",
                stderr = subprocess.DEVNULL,
                cwd = str(Path(executable).parent),
            ).strip()
        except UnicodeDecodeError as exc:
            _fail(
                f"Could not read the Windows user profile for Codex ({exc}); "
                "set USERPROFILE in the WSL environment, for example through WSLENV."
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            _fail(f"Could not find the Windows user profile for Codex: {exc}")
    if not profile or profile == "%USERPROFILE%":
        _fail("Could not find the Windows user profile for Codex.")
    if profile.startswith("/"):
        return Path(profile)
    try:
        translated = subprocess.check_output(
            ["wslpath", "-u", profile],
            text = True,
            encoding = "utf-8",
            errors = "replace",
            stderr = subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        _fail(f"Could not translate Windows user profile {profile}: {exc}")
    if not translated:
        _fail(f"Could not translate Windows user profile {profile}.")
    return Path(translated)


def _codex_source_home(*, ignore_configured: bool = False) -> Path:
    configured = None if ignore_configured else os.environ.get("CODEX_HOME")
    if configured:
        if _wsl_windows_executable(["codex"]) and _looks_like_path(configured):
            if not configured.startswith("/"):
                try:
                    configured = subprocess.check_output(
                        ["wslpath", "-u", configured],
                        text = True,
                        encoding = "utf-8",
                        errors = "replace",
                        stderr = subprocess.DEVNULL,
                    ).strip()
                except (OSError, subprocess.CalledProcessError) as exc:
                    _fail(f"Could not translate Windows CODEX_HOME {configured}: {exc}")
                if not configured:
                    _fail("Could not translate Windows CODEX_HOME.")
        return Path(configured).expanduser()
    executable = _wsl_windows_executable(["codex"])
    if executable:
        return _wsl_windows_user_profile(executable) / ".codex"
    return Path.home() / ".codex"


def _is_junction(path: Path) -> bool:
    # Path.is_junction() was added in Python 3.12.
    if hasattr(path, "is_junction"):
        return path.is_junction()
    try:
        return (
            getattr(os.lstat(path), "st_reparse_tag", None) == 0xA0000003
        )  # IO_REPARSE_TAG_MOUNT_POINT
    except OSError:
        return False


def _is_directory_link(path: Path) -> bool:
    # lstat reads the link, so FILE_ATTRIBUTE_DIRECTORY answers even when dangling.
    try:
        return bool(getattr(os.lstat(path), "st_file_attributes", 0) & 0x10)
    except OSError:
        return False


def _remove_overlay_entry(path: Path) -> None:
    if _is_junction(path):
        path.rmdir()
    elif os.name == "nt" and path.is_symlink() and _is_directory_link(path):
        # DeleteFileW, which unlink maps to, refuses a directory entry; rmdir drops the link.
        path.rmdir()
    elif path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _create_directory_junction(source: Path, target: Path) -> bool:
    if os.name != "nt":
        return False
    try:
        result = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(target), str(source)],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 30,
            check = False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def write_codex_parent_overlay(overlay: Path) -> Path:
    """Add local-agent routing without replacing the cloud parent's configuration."""
    overlay.mkdir(parents = True, exist_ok = True, mode = 0o700)

    manifest_path = overlay / _CODEX_PARENT_OVERLAY_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding = "utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        manifest = None
    source_home = _codex_source_home()
    overlay_key = str(overlay.resolve(strict = False))
    source_key = str(source_home.resolve(strict = False))
    if source_key == overlay_key:
        previous_source = manifest.get("source_home") if isinstance(manifest, dict) else None
        if isinstance(previous_source, str) and previous_source:
            candidate = Path(previous_source).expanduser()
            if str(candidate.resolve(strict = False)) != overlay_key:
                source_home = candidate
            else:
                source_home = _codex_source_home(ignore_configured = True)
        else:
            source_home = _codex_source_home(ignore_configured = True)
        source_key = str(source_home.resolve(strict = False))
    same_source = isinstance(manifest, dict) and manifest.get("source_home") == source_key
    if same_source:
        managed_entries = manifest.get("entries", [])
        if not isinstance(managed_entries, list):
            managed_entries = []
        for name in managed_entries:
            if isinstance(name, str) and name not in {"", ".", ".."} and Path(name).name == name:
                _remove_overlay_entry(overlay / name)
    else:
        # A reused overlay must never mix credentials, config, or plugins from two different Codex homes. Legacy overlays have no manifest, so rebuild them once.
        for target in list(overlay.iterdir()):
            _remove_overlay_entry(target)

    # Keep the user's auth, config, plugins, agents, skills, rules and session state visible. Symlinks make this an overlay rather than a stale copy; if Windows denies them, directory junctions keep large runtime state shared without a bulk copy, and the configuration surfaces and sessions are copied only when both link forms are unavailable.
    fallback_dirs = {"agents", "skills", "rules", "plugins", "marketplaces", "sessions"}
    entries = []
    if source_home.is_dir():
        for source in source_home.iterdir():
            if source.name in {
                "AGENTS.md",
                "AGENTS.override.md",
                _CODEX_PARENT_OVERLAY_MANIFEST,
            }:
                continue
            target = overlay / source.name
            _remove_overlay_entry(target)
            try:
                target.symlink_to(source, target_is_directory = source.is_dir())
                entries.append(source.name)
            except OSError:
                if source.is_file():
                    shutil.copy2(source, target)
                    entries.append(source.name)
                elif source.is_dir():
                    if _create_directory_junction(source, target):
                        entries.append(source.name)
                    elif source.name in fallback_dirs:
                        shutil.copytree(source, target)
                        entries.append(source.name)

    _write_private_json(
        manifest_path,
        {"source_home": source_key, "entries": sorted(entries)},
    )

    inherited = ""
    instruction_name = "AGENTS.md"
    for candidate in (source_home / "AGENTS.override.md", source_home / "AGENTS.md"):
        try:
            text = candidate.read_text(encoding = "utf-8")
        except FileNotFoundError:
            continue
        except OSError as exc:
            _fail(f"Could not preserve Codex instructions from {candidate}: {exc}")
        if text.strip():
            inherited = text.rstrip()
            instruction_name = candidate.name
            break

    other_name = "AGENTS.md" if instruction_name == "AGENTS.override.md" else "AGENTS.override.md"
    other = overlay / other_name
    if other.is_file() or other.is_symlink():
        other.unlink()
    routing = _CODEX_SUBAGENT_ROUTING_INSTRUCTIONS
    combined = f"{inherited}\n\n{routing}\n" if inherited else f"{routing}\n"
    _write_private_text(overlay / instruction_name, combined)
    return overlay


def _agent_config_path(path: Path, command: list) -> str:
    """Translate a generated config path when a Windows agent runs through WSL."""
    return _wsl_windows_path(path) if _wsl_windows_executable(command) else str(path)


def _opencode_subagent_inline_config(
    path: Path,
    permission: dict,
    command: str = "opencode",
    v2: bool = False,
) -> dict:
    """Keep the local provider visible without hiding the parent's allowed providers."""
    inline: dict = {}
    inherited = os.environ.get("OPENCODE_CONFIG_CONTENT")
    if inherited:
        try:
            parsed = json.loads(inherited)
        except ValueError:
            _fail("OPENCODE_CONFIG_CONTENT is not valid JSON.")
        if not isinstance(parsed, dict):
            _fail("OPENCODE_CONFIG_CONTENT must contain a JSON object.")
        inline.update(parsed)

    def merge_provider_filters(effective_config: dict) -> None:
        enabled = effective_config.get("enabled_providers")
        if isinstance(enabled, list):
            inherited_enabled = inline.get("enabled_providers")
            if not isinstance(inherited_enabled, list):
                inherited_enabled = []
            providers = [
                provider
                for provider in [*inherited_enabled, *enabled]
                if provider != _OPENCODE_PROVIDER
            ]
            inline["enabled_providers"] = list(dict.fromkeys([*providers, _OPENCODE_PROVIDER]))
        disabled = effective_config.get("disabled_providers")
        if isinstance(disabled, list) and _OPENCODE_PROVIDER in disabled:
            inline["disabled_providers"] = [
                provider for provider in disabled if provider != _OPENCODE_PROVIDER
            ]

    # Keep an inherited inline allowlist usable by the local provider. V2 turns these filters into policies where global/project rules still intentionally outrank this content; the message at launch makes that boundary explicit.
    merge_provider_filters(inline)
    effective = inline

    executable = None if v2 else _which_with_install_dirs(command)
    if v2:
        legacy_depth = inline.pop("subagent_depth", None)
        if (
            isinstance(legacy_depth, int)
            and not isinstance(legacy_depth, bool)
            and legacy_depth > 0
        ):
            experimental = inline.get("experimental")
            if not isinstance(experimental, dict):
                experimental = {}
            experimental.setdefault("subagent_depth", legacy_depth)
            inline["experimental"] = experimental
    elif executable is None:
        typer.echo(
            f"Warning: OpenCode is not installed, so provider filters could not be checked. "
            f"The target configuration must allow '{_OPENCODE_PROVIDER}'.",
            err = True,
        )
    else:
        env = _probe_env(OPENCODE_CONFIG = _agent_config_path(path, [command]))
        try:
            resolved = subprocess.run(
                [executable, "debug", "config"],
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 15,
                env = env,
            )
        except Exception as exc:
            _fail(f"Could not inspect OpenCode provider filters: {exc}")
        if resolved.returncode != 0:
            detail = resolved.stderr.strip() or resolved.stdout.strip()
            _fail(f"Could not inspect OpenCode provider filters: {detail or 'unknown error'}")
        try:
            effective = json.loads(resolved.stdout)
        except ValueError:
            _fail("Could not inspect OpenCode provider filters: invalid JSON response.")
        if not isinstance(effective, dict):
            _fail("Could not inspect OpenCode provider filters: expected a JSON object.")

        merge_provider_filters(effective)

    if not v2:
        depth = effective.get("subagent_depth")
        inline["subagent_depth"] = (
            depth if isinstance(depth, int) and not isinstance(depth, bool) and depth > 0 else 1
        )
    if permission:
        inline["permission"] = permission
    return inline


def _b64_path(path: Path) -> str:
    """Path as base64, so it can cross a shell without being expanded."""
    return base64.b64encode(str(path).encode("utf-8")).decode("ascii")


_CLAUDE_PLAN_GATE_SCRIPT = '''\
"""Deny the editing agent while the parent session is in plan mode."""
import json, sys

try:
    mode = (json.load(sys.stdin) or {}).get("permission_mode")
except Exception:
    sys.exit(0)  # fail open: a hook error must never block the parent session
if mode == "plan":
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": (
            "Plan mode is active. Call the read-only local plan agent "
            "(local_plan_agent) instead of local_agent."
        ),
    }}))
sys.exit(0)
'''


def write_claude_subagent_plugin(path: Path, server_env: dict) -> Path:
    """Write a session plugin that exposes the local Claude child through MCP."""
    plugin = path / "local-agent"
    command = sys.executable
    args = ["-m", _CLAUDE_SUBAGENT_MCP_MODULE]
    mcp_env = dict(server_env)
    base = server_env.get("AGENT_SWITCH_CLAUDE_SUBAGENT_BASE_URL")
    key = server_env.get("AGENT_SWITCH_CLAUDE_SUBAGENT_API_KEY")
    model_id = server_env.get("AGENT_SWITCH_CLAUDE_SUBAGENT_MODEL")
    if base and key and model_id:
        entry = {
            "id": model_id,
            "context_length": int(
                server_env.get("AGENT_SWITCH_CLAUDE_SUBAGENT_CONTEXT_WINDOW", "0") or 0
            ),
        }
        headers = json.loads(server_env.get("AGENT_SWITCH_CLAUDE_SUBAGENT_HEADERS") or "{}")
        compact_at_raw = server_env.get("AGENT_SWITCH_CLAUDE_SUBAGENT_COMPACT_AT")
        compact_at = float(compact_at_raw) if compact_at_raw else None
        local_env = _claude_local_env(base, key, entry, headers = headers, compact_at = compact_at)
        if "CLAUDE_CODE_EXTRA_BODY" in server_env:
            local_env["CLAUDE_CODE_EXTRA_BODY"] = server_env["CLAUDE_CODE_EXTRA_BODY"]
        settings = _write_claude_settings(plugin, model_id, local_env)
        mcp_env[_CLAUDE_SUBAGENT_SETTINGS_ENV] = str(settings)
    if _wsl_windows_executable(["claude"]):
        command = "wsl.exe"
        args = [
            "-d",
            os.environ["WSL_DISTRO_NAME"],
            "--",
            sys.executable,
            "-m",
            _CLAUDE_SUBAGENT_MCP_MODULE,
        ]
        mcp_env["WSLENV"] = _merge_wslenv(
            os.environ.get("WSLENV", ""),
            (
                *_wsl_bridge_names(server_env, ()),
                *([_CLAUDE_SUBAGENT_SETTINGS_ENV] if base and key and model_id else []),
            ),
        )
    _write_private_json(
        plugin / ".claude-plugin" / "plugin.json",
        {
            "name": "local-agent",
            "version": "1.0.0",
            "description": _SUBAGENT_DESCRIPTION,
            "author": {"name": "agent-switch"},
        },
    )
    _write_private_json(
        plugin / ".mcp.json",
        {
            "mcpServers": {
                "local": {
                    "type": "stdio",
                    "command": command,
                    "args": args,
                    "env": mcp_env,
                }
            }
        },
    )
    # Claude already refuses the editing tool in plan mode, since it advertises readOnlyHint false. This PreToolUse hook replaces that dead end with a reason naming the read-only tool to call instead. Skipped under the WSL bridge, where the gate is a Linux path but the hook would run beside the Windows claude.
    gate = plugin / "hooks" / "plan_gate.py"
    if command == "wsl.exe":
        # A persisted plugin dir may still hold a gate from an earlier non-WSL run.
        for stale in (gate, plugin / "hooks" / "hooks.json"):
            stale.unlink(missing_ok = True)
    else:
        _write_private_text(gate, _CLAUDE_PLAN_GATE_SCRIPT)
        _write_private_json(
            plugin / "hooks" / "hooks.json",
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": _CLAUDE_SUBAGENT_TOOL,
                            "hooks": [
                                {
                                    "type": "command",
                                    # Run through runpy rather than handing the path to the interpreter: a missing gate is then an ordinary traceback (exit 1, fails open) instead of exit 2, which Claude treats as a blocking error and would deny the tool in every mode. The path is base64'd because this string goes through a shell: a temp root holding $(..) or a backtick expands under sh, %VAR% under cmd, and the gate then silently fails open. base64's alphabet has no metacharacter in either.
                                    "command": (
                                        f'"{sys.executable}" -c '
                                        f'"import base64,runpy; runpy.run_path('
                                        f"base64.b64decode('{_b64_path(gate)}').decode())\""
                                    ),
                                    # A hook with no timeout stalls the parent for as long as it hangs; measured unbounded past 400s.
                                    "timeout": 10,
                                }
                            ],
                        }
                    ]
                }
            },
        )
    skill = plugin / "skills" / "local-agent" / "SKILL.md"
    skill.parent.mkdir(parents = True, exist_ok = True, mode = 0o700)
    skill.write_text(
        "---\n"
        "description: Delegate a task to the local agent running on a local model. Use when "
        "the user asks to spawn a local agent.\n"
        "---\n\n"
        "Call the local agent tool once with the complete task. In plan mode, call "
        "the read-only local plan agent instead. Return its result to the user without "
        "claiming that the cloud parent completed the local work.\n",
        encoding = "utf-8",
    )
    return plugin


def _codex_subagent_flags(path: Path) -> list[str]:
    command = sys.executable
    package_root = str(Path(__file__).resolve().parents[1])
    bootstrap = (
        f"import sys;sys.path.insert(0,{json.dumps(package_root)});"
        f"from {_CODEX_SUBAGENT_MCP_MODULE} import main;main()"
    )
    args = ["-c", bootstrap, str(path)]
    if _wsl_windows_executable(["codex"]):
        command = "wsl.exe"
        args = [
            "-d",
            os.environ["WSL_DISTRO_NAME"],
            "--",
            sys.executable,
            "-c",
            bootstrap,
            str(path),
        ]
    server = (
        "{ "
        f"command = {json.dumps(command)}, "
        f"args = {json.dumps(args)}, "
        f"required = true, enabled_tools = [{json.dumps(_CODEX_SUBAGENT_MCP_TOOL)}], "
        'default_tools_approval_mode = "approve", '
        "startup_timeout_sec = 15, tool_timeout_sec = 3600 }"
    )
    return ["-c", f"mcp_servers.{_CODEX_SUBAGENT_MCP_SERVER}={server}"]


def _wsl_windows_executable(command: list) -> Optional[str]:
    if os.name == "nt" or not os.environ.get("WSL_DISTRO_NAME"):
        return None
    executable = shutil.which(command[0])
    if executable and executable.startswith("/mnt/"):
        return executable
    return None


def _wsl_windows_path(path: Path) -> str:
    try:
        translated = subprocess.check_output(
            ["wslpath", "-w", str(path)], text = True, encoding = "utf-8", errors = "replace"
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        _fail(f"Could not translate WSL path {path}: {exc}")
    if not translated:
        _fail(f"Could not translate WSL path {path}")
    return translated


def _looks_like_path(value: str) -> bool:
    # A var only wants the WSLENV /p flag if its value is a filesystem path: an absolute POSIX path (/...), a UNC path (\\\\...), or a drive-qualified Windows path (C:...). Scalar knobs such as a numeric context window must pass through untranslated, so they get no flag.
    return bool(value) and (value.startswith(("/", "\\")) or (len(value) >= 2 and value[1] == ":"))


def _wsl_bridge_names(env: dict, unset_env: tuple) -> tuple:
    # Build the WSLENV share list for a Windows shim reached from WSL. Path-valued vars get /p so WSLENV translates them to the Windows path the /mnt shim can actually open; a cleared var carries no value to translate.
    names = [name + ("/p" if _looks_like_path(value) else "") for name, value in env.items()]
    names.extend(unset_env)
    return tuple(dict.fromkeys(names))


def _merge_wslenv(current: str, names: tuple) -> str:
    # Index WSLENV entries by bare var name, preserving first-seen order. The vars we bridge are applied last so our entry wins: a user's pre-existing unflagged "HOME" is upgraded to "HOME/p" rather than left as-is, since WSLENV ignores a duplicate name and a bare entry would leave the path untranslated for a Windows shim.
    ordered = []
    by_name = {}
    for entry in (*current.split(":"), *names):
        if not entry:
            continue
        base = entry.split("/", 1)[0]
        if base not in by_name:
            ordered.append(base)
        by_name[base] = entry
    return ":".join(by_name[base] for base in ordered)


def _powershell_quote(arg: str) -> str:
    # PowerShell reads single-quoted strings literally (an embedded ' is doubled), so JSON args such as `--settings {"env":...}` survive intact. list2cmdline's backslash-escaped double quotes are cmd.exe syntax and PowerShell mis-parses them.
    if arg and re.fullmatch(r"[A-Za-z0-9_./:=+-]+", arg):
        return arg
    return "'" + arg.replace("'", "''") + "'"


def _print_env(
    env: dict,
    command: list,
    unset_env: tuple = (),
    wsl_env_bridge: tuple = (),
) -> None:
    if os.name == "nt":
        for name in unset_env:
            typer.echo(f"Remove-Item Env:{name} -ErrorAction SilentlyContinue")
        for name, value in env.items():
            # PowerShell: ` is the escape char, and $ triggers expansion inside "".
            escaped = value.replace("`", "``").replace('"', '`"').replace("$", "`$")
            typer.echo(f'$env:{name} = "{escaped}"')
        typer.echo(" ".join(_powershell_quote(arg) for arg in command))
        return
    for name in unset_env:
        typer.echo(f"export {name}=" if wsl_env_bridge else f"unset {name}")
    for name, value in env.items():
        typer.echo(f"export {name}={shlex.quote(value)}")
    if wsl_env_bridge:
        typer.echo(
            f"export WSLENV={shlex.quote(_merge_wslenv(os.environ.get('WSLENV', ''), wsl_env_bridge))}"
        )
    # The final line is a SELF-CONTAINED one-liner (inline env, VAR=... cmd) rather than a bare command. People copy just the last line, and a bare `codex`/`claude` would then run against their real ~/.codex or Anthropic credentials with zero isolation, for example inheriting a pre-existing damaged ~/.codex state DB and blaming the recipe. Inline assignments scope every var, and empty-string the conflicting ones, to this single invocation, so a partial copy behaves the same as pasting the whole block.
    inline = [f"{name}=" for name in unset_env]
    inline += [f"{name}={shlex.quote(value)}" for name, value in env.items()]
    if wsl_env_bridge:
        inline.append(
            f"WSLENV={shlex.quote(_merge_wslenv(os.environ.get('WSLENV', ''), wsl_env_bridge))}"
        )
    typer.echo(" ".join((*inline, shlex.join(command))))


def _refresh_windows_path() -> None:
    # Merge Windows registry PATH hives after the current process PATH so a freshly installed agent is visible without changing existing precedence.
    if os.name != "nt":
        return
    try:
        import winreg
    except Exception:
        return

    entries = []
    seen = set()

    def add_path(value: str) -> bool:
        added = False
        for entry in str(value).split(os.pathsep):
            entry = entry.strip()
            if not entry:
                continue
            key = os.path.normcase(entry).casefold()
            if key in seen:
                continue
            seen.add(key)
            entries.append(entry)
            added = True
        return added

    add_path(os.environ.get("PATH", ""))
    added_registry = False
    hives = (
        (winreg.HKEY_CURRENT_USER, "Environment"),
        (
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        ),
    )
    for root, sub in hives:
        try:
            with winreg.OpenKey(root, sub) as key:
                value, _ = winreg.QueryValueEx(key, "Path")
        except OSError:
            continue
        if value:
            added_registry = add_path(os.path.expandvars(str(value))) or added_registry
    if added_registry:
        os.environ["PATH"] = os.pathsep.join(entries)


def _augment_path_with_install_dirs() -> None:
    # Add known install dirs to PATH so a freshly installed agent resolves without a new shell. User dirs are appended, so existing tools keep precedence; a missing home (bare container UID) just means there are none.
    try:
        home = Path.home()
    except (RuntimeError, OSError):
        home = None
    candidates = [home / ".local" / "bin", home / ".opencode" / "bin"] if home is not None else []
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            candidates.append(Path(appdata) / "npm")
    current = os.environ.get("PATH")
    if current is None:
        # PATH unset: shutil.which() and exec*p* fall back to os.defpath (/bin:/usr/bin), so keep that default instead of collapsing to just the install dirs, which would hide a system-installed agent and strip the launched child's normal PATH. An explicitly empty PATH is left as-is: like shutil.which, it means "search nothing", not os.defpath.
        current = os.defpath
    seen = {os.path.normcase(entry) for entry in current.split(os.pathsep) if entry}
    additions = [
        str(directory)
        for directory in candidates
        if directory.is_dir() and os.path.normcase(str(directory)) not in seen
    ]
    if additions:
        parts = [part for part in (current, *additions) if part]
        os.environ["PATH"] = os.pathsep.join(parts)


def _probe_env(**extra: str) -> dict:
    """Environment for probes that RUN a resolved shim. _which_with_install_dirs restores PATH before returning, so a Node-backed shim whose node sits in an install dir would not find it when executed."""
    original = os.environ.get("PATH")
    _augment_path_with_install_dirs()
    env = os.environ.copy()
    if original is None:
        os.environ.pop("PATH", None)
    else:
        os.environ["PATH"] = original
    env.update(extra)
    return env


def _prefer_windows_cmd_sibling(executable: Optional[str]) -> Optional[str]:
    """Prefer the sibling .cmd when Windows resolved an extensionless npm/pnpm shim. cmd-shim writes ``to``, ``to.cmd`` and ``to.ps1``, and shutil.which can return the extensionless POSIX shim, which CreateProcess rejects with WinError 193. Measured on windows-latest: 3.12.0 probes the bare name before PATHEXT (gh-109590) and 3.12.1 onwards do not, and a PATHEXT holding "." reaches the same place on any version. Substituted only when the file opens with a shebang, so a real PE keeps priority over a stale wrapper beside it; matched on not-a-Windows-suffix so a dotted bin name is caught too."""
    if executable is None or os.name != "nt":
        return executable
    if Path(executable).suffix.lower() in {".exe", ".com", ".cmd", ".bat", ".ps1"}:
        return executable
    with contextlib.suppress(OSError):
        with open(executable, "rb") as resolved_file:
            if resolved_file.read(2) == b"#!":
                # .CMD only matters on case-sensitive volumes; no writer emits .bat.
                for extension in (".cmd", ".CMD"):
                    sibling = Path(executable + extension)
                    if sibling.is_file():
                        return str(sibling)
    return executable


def _which_with_install_dirs(name: str) -> Optional[str]:
    # shutil.which(name), but searching the known agent install dirs too, so a version probe resolves the same binary _launch() will (it augments PATH before it runs). Without this an agent present only in ~/.local/bin / %APPDATA%
    # pm is missed, wrongly assumed current, and launched with flags an older build rejects. PATH is restored afterward: only _launch() should persist the augmentation for the child process.
    original = os.environ.get("PATH")
    _augment_path_with_install_dirs()
    try:
        # Callers spawn this result directly, so the shim rescue is needed here too, not only in _resolved_launch_command.
        return _prefer_windows_cmd_sibling(shutil.which(name))
    finally:
        if original is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = original


def _which_deepseek_harness_with_install_dirs() -> Optional[str]:
    """Find the first valid DeepSeek Harness even when another ``dsh`` shadows it."""
    original = os.environ.get("PATH")
    _augment_path_with_install_dirs()
    try:
        for executable in deepseek_harness_executables_on_path():
            executable = _prefer_windows_cmd_sibling(executable)
            if executable is not None and is_deepseek_harness_executable(executable):
                return executable
        return None
    finally:
        if original is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = original


def _install_source(install_hint: str) -> Optional[str]:
    """The first http(s) URL an install hint fetches, or None (e.g. an npm install)."""
    match = re.search(r"https?://[^\s'\")]+", install_hint)
    return match.group(0) if match else None


def _pinned_raw_github_commit(source: str) -> Optional[str]:
    """Return the immutable full commit in a raw GitHub URL, if present."""
    match = re.match(
        r"^https://raw\.githubusercontent\.com/[^/]+/[^/]+/([0-9a-f]{40})/",
        source,
        flags = re.IGNORECASE,
    )
    return match.group(1).lower() if match else None


def _npm_executable() -> Optional[str]:
    executable = _prefer_windows_cmd_sibling(shutil.which("npm"))
    if executable and not _wsl_windows_executable([executable]):
        return executable
    if executable:
        # WSL inherits the Windows PATH, so the rejected shim may shadow a native npm.
        for directory in os.get_exec_path():
            candidate = _prefer_windows_cmd_sibling(shutil.which("npm", path = directory))
            if candidate and not _wsl_windows_executable([candidate]):
                return candidate
    return None


def _install_command(install_hint: str) -> tuple[list[str], Optional[dict]]:
    if not re.match(r"^\s*npm(?:\s|$)", install_hint):
        if os.name == "nt":
            return (
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    install_hint,
                ],
                None,
            )
        return ["/bin/sh", "-c", install_hint], None

    npm = _npm_executable()
    if npm is None:
        _fail(
            "npm is required to install this agent, but no native system npm was found. "
            "Install Node.js with npm, then re-run."
        )
    args = shlex.split(install_hint)
    env = dict(os.environ)
    # dirname, not Path().parent: Path picks its flavour from os.name, which the tests override. Empty means npm is a bare name; prepending "" would put the cwd on PATH.
    npm_dir = os.path.dirname(npm)
    current_path = env.get("PATH", "")
    if npm_dir:
        env["PATH"] = os.pathsep.join([npm_dir, current_path]) if current_path else npm_dir
    if os.name == "nt":
        command = "& " + " ".join(_powershell_quote(arg) for arg in [npm, *args[1:]])
        return (
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                command,
            ],
            env,
        )
    return [npm, *args[1:]], env


def _install_agent(name: str, install_hint: str) -> Optional[str]:
    # Missing agent under --launch: offer to run its documented install command, then re-resolve it on PATH. Consent-based, and a non-interactive stdin cannot answer the prompt, so both the no-TTY and declined cases return None and let the caller print the hint and exit.
    if not sys.stdin.isatty():
        return None
    typer.echo(f"`{name}` is not installed.")
    # Make the supply-chain risk explicit before the prompt: these are the vendors' own installers (curl | bash, irm | iex, npm), run with the user's privileges, and nothing checks a signature or hash on the fetched content. Naming the source turns a blind "yes" into informed consent.
    source = _install_source(install_hint)
    if source:
        pinned_commit = _pinned_raw_github_commit(source)
        if pinned_commit:
            warning = (
                "Security warning: This will download and execute a third-party script "
                f"from {source} with your privileges. agent-switch pins this content to "
                f"immutable upstream commit {pinned_commit}, but does not independently "
                "verify or sandbox it. Continue only if you trust this source and commit."
            )
        else:
            warning = (
                "Security warning: This will download and execute an unverified third-party "
                f"script from {source} with your privileges. agent-switch does not pin or verify "
                "the downloaded content. Continue only if you trust this source."
            )
    else:
        warning = (
            f"This will RUN `{install_hint}` with your privileges; "
            "there is no signature or hash check."
        )
    typer.secho(warning, fg = "yellow", err = True)
    if not typer.confirm(f"Install `{name}` now with `{install_hint}`?", default = False):
        return None
    install_command, install_env = _install_command(install_hint)
    try:
        result = subprocess.run(install_command, env = install_env)
    except OSError as exc:
        _fail(
            f"Could not run the install command: {exc}. "
            f"Run it yourself, then re-run: {install_hint}"
        )
    if result.returncode != 0:
        message = f"Install command failed. Run it yourself, then re-run: {install_hint}"
        if os.name == "nt":
            # A hand-run retry can still hit the policy; point at the one-time per-user fix.
            message += (
                "\nIf it fails because running scripts is disabled (PSSecurityException), "
                "allow local scripts for your user, then retry:\n"
                "  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned"
            )
        _fail(message)
    # Resolve the freshly installed agent without a shell restart: pull registry PATH (Windows) plus well-known install dirs the installer may not have added to PATH.
    _refresh_windows_path()
    _augment_path_with_install_dirs()
    executable = shutil.which(name)
    if executable is None:
        _fail(
            f"`{name}` installed but isn't on PATH yet. Open a new shell (or add it to "
            f"PATH), then re-run. Install command: {install_hint}"
        )
    return executable


def _resolve_or_install_agent(name: str, install_hint: str, resolver) -> str:
    executable = resolver(name)
    invalid_executable = None
    if executable is not None:
        if name != "dsh" or is_deepseek_harness_executable(executable):
            return executable
        invalid_executable = executable
        executable = _which_deepseek_harness_with_install_dirs()
        if executable is not None:
            return executable

    executable = _install_agent(name, install_hint)
    if executable is not None:
        if name != "dsh" or is_deepseek_harness_executable(executable):
            return executable
        invalid_executable = executable
    if name == "dsh":
        executable = _which_deepseek_harness_with_install_dirs()
        if executable is not None:
            return executable

    if invalid_executable is not None:
        _fail(
            f"`{invalid_executable}` is not DeepSeek Harness. Install DeepSeek Harness "
            f"with: {install_hint}"
        )
    _fail(f"`{name}` not found on PATH. Install it with: {install_hint}")


def _require_agent_for_launch(name: str, install_hint: str, launch: bool) -> Optional[str]:
    if not launch:
        return None
    return _resolve_or_install_agent(name, install_hint, _which_with_install_dirs)


def _wsl_shim_env(command: list, env: dict, unset_env: tuple) -> tuple[dict, tuple]:
    if not _wsl_windows_executable(command):
        return env, ()
    wsl_env_bridge = _wsl_bridge_names(env, unset_env)
    if not wsl_env_bridge:
        return env, ()
    # Bridge PWD via WSLENV (PWD/p) so the Windows shim finds its project root from the live cwd, not a stale inherited Linux PWD. Do not freeze env["PWD"]: a --no-launch recipe must translate the live PWD when run, not when generated; _launch overrides it.
    return env, tuple(dict.fromkeys((*wsl_env_bridge, "PWD/p")))


_NPM_CMD_SHIM_HEAD = (
    "@ECHO off\n"
    "GOTO start\n"
    ":find_dp0\n"
    "SET dp0=%~dp0\n"
    "EXIT /b\n"
    ":start\n"
    "SETLOCAL\n"
    "CALL :find_dp0\n"
)
_NPM_NODE_CMD_SHIM_PREFIX = (
    re.escape(_NPM_CMD_SHIM_HEAD)
    + r"(?P<environment>(?:@SET [^=\r\n]+=[^\r\n]+\n)*)"
    + re.escape(
        '\nIF EXIST "%dp0%\\node.exe" (\n'
        + '  SET "_prog=%dp0%\\node.exe"\n'
        + ") ELSE (\n"
        + '  SET "_prog=node"\n'
    )
)
_NPM_NODE_CMD_SHIM_SUFFIX = (
    r"(?P<node_args>[^\r\n]*?)[ \t]+" + r'"%dp0%\\(?P<target>[^"\r\n]+)"[ \t]+%\*'
)
_NPM_NODE_CMD_SHIMS = (
    re.compile(
        _NPM_NODE_CMD_SHIM_PREFIX
        + re.escape(
            "  SET PATHEXT=%PATHEXT:;.JS;=;%\n"
            + ")\n\n"
            + 'endLocal & goto #_undefined_# 2>NUL || title %COMSPEC% & "%_prog%"'
        )
        + _NPM_NODE_CMD_SHIM_SUFFIX,
        re.IGNORECASE,
    ),
    re.compile(
        _NPM_NODE_CMD_SHIM_PREFIX
        + re.escape(
            ")\n\n"
            + "endLocal & goto #_undefined_# 2>NUL || title %COMSPEC% & "
            + "set PATHEXT=%PATHEXT:;.JS;=;% & "
            + '"%_prog%"'
        )
        + _NPM_NODE_CMD_SHIM_SUFFIX,
        re.IGNORECASE,
    ),
)
_NPM_NATIVE_CMD_SHIM = re.compile(
    re.escape(_NPM_CMD_SHIM_HEAD) + r'"%dp0%\\(?P<target>[^"\r\n]+)"[ \t]+%\*',
    re.IGNORECASE,
)
_NPM_NODE_SHEBANG = re.compile(
    r"^#!\s*(?:/usr/bin/env\s+(?:-S\s+)?((?:[^ \t=]+=[^ \t=]+\s+)*))?([^ \t]+)(.*)$"
)
_NPM_SHEBANG_DOLLAR = re.compile(r"\$\{?([^$@#?\- \t{}:]+)\}?")


def _npm_batch_environment(declarations: str) -> str:
    lines = []
    for declaration in declarations.split():
        name, separator, value = declaration.partition("=")
        name = name.strip()
        value = value.strip()
        if separator and name and value:
            value = _NPM_SHEBANG_DOLLAR.sub(lambda match: f"%{match.group(1)}%", value)
            lines.append(f"@SET {name}={value}\n")
    return "".join(lines)


def _windows_expand_environment(value: str, environment: dict) -> str:
    folded = {name.casefold(): item for name, item in environment.items()}
    return re.sub(
        r"%([^%\r\n]+)%",
        lambda match: folded.get(match.group(1).casefold(), ""),
        value,
    )


def _npm_node_shim_metadata(target: Path, match, environment: dict) -> Optional[tuple]:
    environment_block = match.group("environment") or ""
    node_args_text = (match.group("node_args") or "").strip()
    known_node_suffix = target.suffix.lower() in {".js", ".cjs", ".mjs"}
    if not environment_block and not node_args_text and known_node_suffix:
        return [], {}

    first_line = target.read_text(encoding = "utf-8").splitlines()[0]
    shebang = _NPM_NODE_SHEBANG.fullmatch(first_line)
    if shebang is None or Path(shebang.group(2)).name.casefold() not in {"node", "node.exe"}:
        return None
    declarations = shebang.group(1) or ""
    if _npm_batch_environment(declarations).casefold() != environment_block.casefold():
        return None
    if (shebang.group(3) or "").strip() != node_args_text:
        return None
    try:
        node_args = shlex.split(node_args_text) if node_args_text else []
    except ValueError:
        return None

    updates = {}
    expanded_environment = dict(environment)
    for line in environment_block.splitlines():
        name, value = line.removeprefix("@SET ").split("=", 1)
        expanded = _windows_expand_environment(value, expanded_environment)
        expanded_environment[name] = expanded
        updates[name] = expanded
    return node_args, updates


def _apply_windows_environment(environment: dict, updates: dict) -> None:
    for name, value in updates.items():
        existing = next((key for key in environment if key.casefold() == name.casefold()), None)
        if existing is not None and existing != name:
            del environment[existing]
        environment[name] = value


def _resolved_launch_command(
    executable: str,
    arguments: list,
    environment: Optional[dict] = None,
) -> list:
    """Return an argv that preserves arguments through standard Windows npm shims."""
    # _launch resolves with raw shutil.which, so rescue here too; the sibling then enters the parser below.
    executable = _prefer_windows_cmd_sibling(executable)
    if os.name == "nt" and Path(executable).suffix.lower() in {".cmd", ".bat"}:
        # cmd.exe treats CR/LF inside `%*` as command separators, and Windows PowerShell's native-command bridge also rewrites embedded quotes. Match complete cmd-shim templates so custom wrappers keep their setup behavior.
        with contextlib.suppress(OSError, UnicodeError, IndexError):
            shim = Path(executable)
            contents = shim.read_text(encoding = "utf-8").replace("\r\n", "\n").strip()
            for pattern in _NPM_NODE_CMD_SHIMS:
                match = pattern.fullmatch(contents)
                if match is None:
                    continue
                relative = Path(*re.split(r"[\\/]+", match.group("target")))
                target = (shim.parent / relative).resolve()
                if not target.is_file() or not any(
                    part.casefold() == "node_modules" for part in target.parts
                ):
                    continue
                metadata = _npm_node_shim_metadata(target, match, environment or os.environ)
                if metadata is None:
                    continue
                node_args, environment_updates = metadata
                bundled_node = shim.parent / "node.exe"
                node = str(bundled_node) if bundled_node.is_file() else shutil.which("node.exe")
                if node:
                    if environment is not None:
                        _apply_windows_environment(environment, environment_updates)
                    return [node, *node_args, str(target), *arguments]

            match = _NPM_NATIVE_CMD_SHIM.fullmatch(contents)
            if match is not None:
                relative = Path(*re.split(r"[\\/]+", match.group("target")))
                target = (shim.parent / relative).resolve()
                if (
                    target.is_file()
                    and any(part.casefold() == "node_modules" for part in target.parts)
                    and target.suffix.lower() in {".exe", ".com"}
                ):
                    return [str(target), *arguments]
    return [executable, *arguments]


def _launch(
    command: list,
    env: dict,
    install_hint: str,
    unset_env: tuple = (),
) -> int:
    # Resolve well-known install dirs (~/.local/bin) first, so an already-installed agent not yet on PATH is found instead of prompting a needless reinstall.
    _augment_path_with_install_dirs()
    executable = _resolve_or_install_agent(command[0], install_hint, shutil.which)
    env, wsl_env_bridge = _wsl_shim_env(command, env, unset_env)
    child_env = dict(os.environ)
    if wsl_env_bridge:
        # Override stale inherited PWD with the real cwd so the shim resolves the project root.
        env = {**env, "PWD": os.getcwd()}
        child_env["WSLENV"] = _merge_wslenv(child_env.get("WSLENV", ""), wsl_env_bridge)
        for name in unset_env:
            child_env[name] = ""
    else:
        for name in unset_env:
            child_env.pop(name, None)
    child_env.update(env)
    if os.name != "nt" and not wsl_env_bridge:
        # Keep POSIX child processes from seeing a stale inherited PWD when subprocess cwd was changed by the caller. Some Node CLIs use PWD for project-root discovery instead of process.cwd().
        child_env["PWD"] = os.getcwd()
    # Ctrl+C cancels a turn inside the agent; do not let it kill this wrapper. A no-op handler, not SIG_IGN: exec preserves an ignored signal but resets a caught one.
    previous = signal.signal(signal.SIGINT, lambda *_: None)
    try:
        launch_command = _resolved_launch_command(executable, command[1:], child_env)
        code = subprocess.run(launch_command, env = child_env).returncode
    finally:
        signal.signal(signal.SIGINT, previous)
    # Negative returncode means killed by signal N; shells expect 128+N.
    return code if code >= 0 else 128 - code


# The server this invocation talks to, for the status lines _run prints. Set by _resolve_target and _connect.
_active_target: Optional[Target] = None
_REQUEST_FLAGS = {
    **{name: "--" + name.replace("_", "-") for name in _SAMPLING_FIELDS},
    "enable_thinking": "--reasoning",
    "reasoning": "--reasoning",
    "reasoning_effort": "--reasoning-effort",
}


def _resolve_target(
    url: Optional[str],
    provider: Optional[str],
    api_key: Optional[str] = None,
    headers: Optional[dict] = None,
) -> Target:
    """The server to use: --url/--provider, else the one server answering on a usual local port."""
    global _active_target
    if url and not api_key:
        api_key = next(iter(_cached_keys(_provider_key_cache_path(), providers.root_url(url))), None)
    try:
        target = providers.resolve_target(url, provider, api_key, headers)
    except ProviderError as exc:
        _fail(str(exc))
    if target is None:
        # The scan never sends the --header pairs; only the server it picks gets them.
        found = providers.scan_local_servers()
        if not found:
            _fail(
                "No model server found on the usual Ollama, LM Studio, llama-server or vLLM port. "
                "Start one, or pass its address with --url (or its type with --provider)."
            )
        if len(found) > 1:
            listed = "\n".join(f"  {providers.label(t.name)} at {t.base}" for t in found)
            _fail(f"Found several model servers:\n{listed}\nPick one with --url (or --provider).")
        target = found[0]
        if headers:
            target = target._replace(headers = headers)
    _active_target = target
    return target


def _warn_unsent_pins(server_options: ServerOptions) -> None:
    """Sampling/reasoning pins this agent cannot carry in its own requests are ignored."""
    sent = server_options.sent_by_agent()
    unsent = [
        _REQUEST_FLAGS[name]
        for name in (*_SAMPLING_FIELDS, "reasoning", "reasoning_effort")
        if getattr(server_options, name) is not None
        and name not in sent
        and not (name == "reasoning" and server_options.reasoning == "auto")
    ]
    if unsent:
        typer.echo(f"Warning: this agent can't send {', '.join(unsent)} itself, so it is ignored.", err = True)


def _connect(
    api_key: Optional[str],
    model: Optional[str],
    load: LoadOptions = LoadOptions(),
    *,
    server_options: ServerOptions = ServerOptions(),
    target: Target,
    needs: tuple = (),
) -> tuple:
    global _active_target
    _active_target = target
    label = providers.label(target.name)
    _warn_unsent_pins(server_options)
    _, dropped = providers.request_body(target.name, server_options._replace(provider = None).request_body())
    for name in dropped:
        typer.echo(f"Warning: {label} ignores {_REQUEST_FLAGS[name]}, so it is left out.", err = True)
    cache = _provider_key_cache_path()
    key = api_key or next(iter(_cached_keys(cache, target.base)), None)
    try:
        base, key, entry = providers.connect(
            target, key, model, load.max_seq_length or None, needs, allow_load = load.allow_load
        )
    except ProviderError as exc:
        _fail(str(exc))
    if api_key:
        _remember_key(cache, base, api_key)
    return base, key, entry


def _run(
    base: str,
    entry: dict,
    env: dict,
    command: list,
    *,
    launch: bool,
    install_hint: str,
    unset_env: tuple = (),
    clear_screen: bool = False,
) -> None:
    # Some agents (Pi) render inline from wherever the cursor sits: their first paint assumes a clean screen rather than clearing or entering the alternate screen themselves. Hand them one so the session does not start mid-scroll under our connection output. click.clear() is cross-platform and a no-op when stdout is not a terminal, so transcripts and --no-launch recipes stay intact.
    if launch and clear_screen:
        click.clear()
    typer.echo(f"{providers.label(_active_target.name)} ready at {base} · model {entry['id']}")
    if not launch:
        env, wsl_env_bridge = _wsl_shim_env(command, env, unset_env)
        _print_env(
            env,
            command,
            unset_env = unset_env,
            wsl_env_bridge = wsl_env_bridge,
        )
        return
    code = _launch(
        command,
        env,
        install_hint = install_hint,
        unset_env = unset_env,
    )
    if code:
        # A failed session must not end silently behind the agent's own output.
        typer.echo(f"The agent exited with code {code}.")
    raise typer.Exit(code = code)


def _agents_config_root() -> Path:
    return _agent_switch_home() / "agents"


@contextlib.contextmanager
def _temporary_agent_config(prefix: str):
    # Nothing else prunes the agents tree, so reuse the locked session helper: the next launch reclaims homes left by a killed wrapper, and the lock spares live sessions.
    temp_root = _agents_config_root() / ".tmp"
    with contextlib.ExitStack() as stack:
        try:
            temp_root.mkdir(parents = True, exist_ok = True, mode = 0o700)
            path = stack.enter_context(_short_ephemeral_session(temp_root, prefix))
        except OSError:
            # The agent-switch home may be absent or unwritable. Fall back to the system temp dir, as before: no reclamation there, but the OS prunes it.
            path = Path(tempfile.mkdtemp(prefix = prefix))
            stack.callback(shutil.rmtree, path, ignore_errors = True)
        yield path


# codex-subagent nests CODEX_HOME under <home>/parent, so it needs the short root too.
_CODEX_SHORT_HOME_AGENTS = ("codex", "codex-subagent")


def _ephemeral_session_parent(agent: str) -> Optional[Path]:
    """Return a non-system-temp parent when an agent needs one."""
    if os.name != "nt" or agent not in _CODEX_SHORT_HOME_AGENTS:
        return None
    # Codex creates a deeply nested curated-plugin checkout below CODEX_HOME. A normal %TEMP%\\agent-switch-codex-* home can exceed legacy Windows path limits during startup, and Codex also refuses to create its PATH helpers below the system temp directory. Keep the throwaway home short but still private to the current user; _session_config removes it on exit.
    root = _agent_switch_home() / ".tmp"
    root.mkdir(parents = True, exist_ok = True, mode = 0o700)
    return root


def _ephemeral_session_prefix(agent: str, parent: Optional[Path]) -> str:
    """Return the platform-specific prefix for an ephemeral agent home."""
    if agent in _CODEX_SHORT_HOME_AGENTS and parent is not None:
        return "a-codex-"
    return f"agent-switch-{agent}-"


@contextlib.contextmanager
def _locked_file(path: Path, blocking: bool = True):
    """Yield whether an advisory lock was acquired for the first byte of path."""
    handle = path.open("a+b")
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    acquired = True
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    if not blocking:
                        break
                    # LK_LOCK gives up after roughly ten seconds. Poll LK_NBLCK instead so a large stale plugin checkout cannot make a concurrent launch fail just because cleanup takes longer.
                    time.sleep(0.05)
        else:
            import fcntl
            mode = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            try:
                fcntl.flock(handle.fileno(), mode)
                acquired = True
            except BlockingIOError:
                acquired = False
        yield acquired
    finally:
        if acquired:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _reclaim_stale_ephemeral_sessions(parent: Path, prefix: str) -> None:
    """Remove abandoned session homes while preserving locked live sessions."""
    for path in parent.glob(f"{prefix}*"):
        if not path.is_dir():
            continue
        active_lock = path / ".active.lock"
        try:
            modified = active_lock.stat().st_mtime if active_lock.exists() else path.stat().st_mtime
        except FileNotFoundError:
            continue
        # The wrapper owns the advisory lock, not the Codex child. If only the wrapper is killed, its child may still be using CODEX_HOME; give that process a full day to finish before treating the unlocked home as stale.
        if time.time() - modified < _CODEX_EPHEMERAL_STALE_SECONDS:
            continue
        try:
            with _locked_file(active_lock, blocking = False) as stale:
                pass
        except FileNotFoundError:
            # A normally exiting session may have removed itself after the glob.
            continue
        if stale:
            shutil.rmtree(path, ignore_errors = True)


def _refresh_ephemeral_session_marker(path: Path, stop: threading.Event) -> None:
    """Keep the stale grace period relative to wrapper death, not session start."""
    while not stop.wait(_CODEX_EPHEMERAL_HEARTBEAT_SECONDS):
        with contextlib.suppress(OSError):
            os.utime(path, None)


@contextlib.contextmanager
def _short_ephemeral_session(parent: Path, prefix: str = "a-codex-"):
    """Create a session home whose lock makes crash cleanup concurrency-safe."""
    path = None
    active_lock = contextlib.ExitStack()
    heartbeat_stop = None
    heartbeat = None
    try:
        with _locked_file(parent / ".cleanup.lock") as cleanup_lock:
            if not cleanup_lock:  # The blocking acquisition should always succeed.
                raise RuntimeError(f"Could not lock ephemeral session root: {parent}")
            _reclaim_stale_ephemeral_sessions(parent, prefix)
            path = Path(tempfile.mkdtemp(prefix = prefix, dir = parent))
            locked = active_lock.enter_context(_locked_file(path / ".active.lock"))
            if not locked:
                raise RuntimeError(f"Could not lock ephemeral session home: {path}")
            heartbeat_stop = threading.Event()
            heartbeat = threading.Thread(
                target = _refresh_ephemeral_session_marker,
                args = (path / ".active.lock", heartbeat_stop),
                name = "agent-switch-home-heartbeat",
                daemon = True,
            )
            heartbeat.start()
        yield path
    finally:
        if heartbeat_stop is not None:
            heartbeat_stop.set()
        if heartbeat is not None:
            heartbeat.join(timeout = 1)
        try:
            with _locked_file(parent / ".cleanup.lock") as cleanup_lock:
                if not cleanup_lock:  # The blocking acquisition should always succeed.
                    raise RuntimeError(f"Could not lock ephemeral session root: {parent}")
                # Release the live marker only after deletion is serialized with startup scavenging, so no scanner can race this rmtree.
                active_lock.close()
                if path is not None:
                    shutil.rmtree(path, ignore_errors = True)
        finally:
            active_lock.close()


@contextlib.contextmanager
def _session_config(
    agent: str,
    launch: bool,
    persist: bool = False,
):
    """Yield a private directory for an agent's session config (never the user's own). launch (the default) uses an ephemeral temp dir removed after the agent process exits, so nothing persists; no-launch uses a stable agent-switch dir, since the printed recipe is run later on this machine; persist (from --persist) uses that same stable dir even for a launch, so the agent's session survives the exit and can be resumed. Either way the user's real ~/.<agent> config is left untouched."""
    if launch and not persist:
        # Windows codex keeps #7519's short home (MAX_PATH); everyone else uses the agent-switch root.
        parent = _ephemeral_session_parent(agent)
        prefix = _ephemeral_session_prefix(agent, parent)
        if parent is not None:
            with _short_ephemeral_session(parent, prefix) as path:
                yield path
        else:
            with _temporary_agent_config(prefix) as path:
                yield path
    else:
        # Never wipe this dir: a previously printed recipe may still be running an agent whose sessions and state live here, and every config writer merges idempotently into an existing home anyway. Writers must also reset any state a previous run's flags left behind (--yolo especially), since files here outlive the invocation that wrote them.
        path = _agents_config_root() / agent
        path.mkdir(parents = True, exist_ok = True, mode = 0o700)
        yield path


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


def opencode_compaction_reserved(window: int, output: int, ratio: Optional[float] = None) -> int:
    if ratio is not None:
        # An explicit ratio is honored exactly; the default's output cap and 8192 floor
        # would block the earlier compaction the flag is asked for.
        return _get_compaction_reserve(window, ratio)
    return max(1, min(output, max(window // 10, 8192)))


def _opencode_output_env(model: dict, max_tokens: Optional[int]) -> dict:
    """Lift OpenCode's output ceiling when --max-tokens exceeds it; re-emit an inherited one so a --no-launch recipe keeps it."""
    window = model.get("context_length") or model.get("max_context_length")
    if not max_tokens:
        return {}
    if not window:
        typer.echo(
            "Warning: the server did not report the model's context length, so --max-tokens is ignored.",
            err = True,
        )
        return {}
    output = _agent_output_limit(int(window), max_tokens)
    raw = os.environ.get(_OPENCODE_OUTPUT_TOKEN_MAX_ENV, "")
    inherited = int(raw) if raw.isdigit() and int(raw) > 0 else None
    ceiling = inherited or _OPENCODE_OUTPUT_TOKEN_MAX
    if output <= ceiling and inherited is None:
        return {}
    return {_OPENCODE_OUTPUT_TOKEN_MAX_ENV: str(max(output, ceiling))}


def _opencode_provider(
    base: str,
    key: str,
    model: dict,
    max_tokens: Optional[int] = None,
    request_body: Optional[dict] = None,
    headers: Optional[dict] = None,
) -> dict:
    model_entry = {"name": model["id"]}
    window = model.get("context_length") or model.get("max_context_length")
    if window:
        window = int(window)
        output = opencode_output_limit(window, max_tokens)
        # Without a limit OpenCode assumes context 0 and never compacts. Without input it compacts at context - output and ignores compaction.reserved.
        model_entry["limit"] = {"context": window, "input": window, "output": output}
    provider_options = {"baseURL": f"{base}/v1"}
    # A custom Authorization header replaces the SDK's Bearer <apiKey>; keeping both would send two.
    if not get_has_custom_authorization(headers or {}):
        provider_options["apiKey"] = key
    if headers:
        provider_options["headers"] = dict(headers)
    if request_body:
        # OpenCode 1.x sends model options, reading the effort only as reasoningEffort; 2.x sends only the provider body.
        model_entry["options"] = {
            "reasoningEffort" if name == "reasoning_effort" else name: value
            for name, value in request_body.items()
        }
        if "temperature" in request_body:
            model_entry["temperature"] = True
        provider_options["body"] = request_body
    return {
        "npm": "@ai-sdk/openai-compatible",
        "name": "agent-switch",
        "options": provider_options,
        "models": {model["id"]: model_entry},
    }


def write_opencode_config(
    base: str,
    key: str,
    model: dict,
    path: Path,
    yolo: bool = False,
    as_subagent: bool = False,
    max_tokens: Optional[int] = None,
    request_body: Optional[dict] = None,
    headers: Optional[dict] = None,
    compact_at: Optional[float] = None,
) -> dict:
    config = _read_json_object(path)
    if config is None:
        typer.echo(
            f"Warning: couldn't parse {path} — add an '{_OPENCODE_PROVIDER}' provider "
            "there yourself, or move the file aside and re-run.",
            err = True,
        )
        return {}
    before = json.dumps(config, sort_keys = True)
    config.setdefault("$schema", "https://opencode.ai/config.json")
    window = model.get("context_length") or model.get("max_context_length")
    reserved = None
    if window:
        window = int(window)
        reserved = opencode_compaction_reserved(
            window, opencode_output_limit(window, max_tokens), compact_at
        )
    # Keep the provider definition in this private session file. The launch path adjusts effective provider filters in the higher-priority inline overlay.
    _subdict(config, "provider")[_OPENCODE_PROVIDER] = _opencode_provider(
        base, key, model, max_tokens, request_body, headers
    )
    # Normal mode pins this as the session model. Subagent mode leaves the user's main/small models alone and exposes the local model to @local and /models.
    opencode_model = f"{_OPENCODE_PROVIDER}/{model['id']}"
    if as_subagent:
        for field in ("model", "small_model"):
            if str(config.get(field) or "").startswith(f"{_OPENCODE_PROVIDER}/"):
                config.pop(field, None)
        # Drop a managed compaction block, current or legacy value.
        managed = {reserved, max(1, window // 10)} if window else set()
        compaction = config.get("compaction")
        if (
            isinstance(compaction, dict)
            and compaction.keys() == {"auto", "reserved"}
            and compaction["auto"] is True
            and compaction["reserved"] in managed
        ):
            config.pop("compaction", None)
        _subdict(config, "agent")[_SUBAGENT_NAME] = {
            "description": _SUBAGENT_DESCRIPTION,
            "mode": "subagent",
            "model": opencode_model,
            "prompt": _SUBAGENT_INSTRUCTIONS,
        }
    else:
        config["model"] = opencode_model
        agents = config.get("agent")
        if isinstance(agents, dict):
            agents.pop(_SUBAGENT_NAME, None)
            if not agents:
                config.pop("agent", None)
    if window and not as_subagent:
        # The fixed 20k-token default buffer over-compacts, or never settles, on a small local context.
        compaction = _subdict(config, "compaction")
        compaction["auto"] = True
        compaction["reserved"] = reserved
    tools = ("edit", "bash", "webfetch", *(("task",) if as_subagent else ()))
    if yolo:
        # Fallback for commands without native --auto and for the append-safe bare --no-launch command, where the subcommand is not known yet. Rides inline (OPENCODE_CONFIG_CONTENT) so it wins over a project config. TUI and `run` launches use --auto and call here with yolo=False, letting OpenCode preserve explicit deny rules.
        session_permission = {t: "allow" for t in tools}
        session_permission["external_directory"] = {"*": "allow"}
        config["permission"] = dict(session_permission)
    else:
        # Undo only what --yolo wrote: our yolo sets an explicit per-tool "allow" for these three tools, so flip exactly those explicit allows back to "ask". A "deny"/"ask", a granular object, a string, or a "*" catch-all is the user's own rule and is left untouched. We do NOT carry a permission inline for a non-yolo session: since OPENCODE_CONFIG_CONTENT outranks the project opencode.json we cannot read, any value forced there would override the user's project rules, weakening a project deny or auto-approving through a granular object's permissive default. Clearing our own persisted yolo state is the fix.
        session_permission: dict = {}
        permission = config.get("permission")
        if isinstance(permission, dict):
            for tool in tools:
                if permission.get(tool) == "allow":
                    permission[tool] = "ask"
            if permission.get("external_directory") == {"*": "allow"}:
                permission["external_directory"] = {"*": "ask"}
    if json.dumps(config, sort_keys = True) != before:
        _write_private_json(path, config)
        typer.echo(f"Updated {path}")
    return session_permission


def _pi_header_values(headers: dict) -> dict:
    """Pi resolves $NAME and !command in header values; doubling $ keeps literal values intact,
    and a leading ! is prefixed with $ so pi's $! escape makes it literal instead of a shell command."""
    escaped = {}
    for name, value in headers.items():
        value = value.replace("$", "$$")
        escaped[name] = f"${value}" if value.startswith("!") else value
    return escaped


def write_pi_config(
    base: str,
    key: str,
    model: dict,
    path: Path,
    *,
    max_tokens: Optional[int] = None,
    request_body: Optional[dict] = None,
    headers: Optional[dict] = None,
) -> None:
    config = _read_json_object(path)
    if config is None:
        typer.echo(
            f"Warning: couldn't parse {path} — add an '{_PI_PROVIDER}' provider there "
            "yourself, or move the file aside and re-run.",
            err = True,
        )
        return
    before = json.dumps(config, sort_keys = True)
    # Pi reads custom providers from ~/.pi/agent/models.json (HOME-relocated for the session). The server is a generic OpenAI-compatible /v1 endpoint, and the key lives in the config rather than the env, matching opencode.
    provider_model = {"id": model["id"]}
    window = model.get("context_length") or model.get("max_context_length")
    if window:
        window = int(window)
        # An unspecified model defaults to contextWindow 128000 / maxTokens 16384, far larger than a small local context, so Pi compacts too late and overflows the server. Pin the real window and a sane output cap, mirroring OpenCode.
        provider_model["contextWindow"] = window
        provider_model["maxTokens"] = _agent_output_limit(window, max_tokens)
    elif max_tokens:
        provider_model["maxTokens"] = max_tokens
    if request_body:
        provider_model["samplingParams"] = request_body
    provider_entry = {
        "api": "openai-completions",
        "baseUrl": f"{base}/v1",
        # pi refuses the prompt without an apiKey ("No API key found for ..."); its OpenAI SDK merges
        # the custom Authorization over the Bearer later (case-insensitively), so one header wins.
        "apiKey": key,
        **({"headers": _pi_header_values(headers)} if headers else {}),
        "models": [provider_model],
    }
    _subdict(config, "providers")[_PI_PROVIDER] = provider_entry
    if json.dumps(config, sort_keys = True) != before:
        _write_private_json(path, config)
        typer.echo(f"Updated {path}")


def write_pi_compaction(agent_dir: Path, model: dict, compact_at: Optional[float]) -> None:
    """Scale Pi's auto-compaction trigger to --compact-at in the session settings."""
    path = agent_dir / "settings.json"
    settings = _read_json_object(path)
    if settings is None:
        return  # write_pi_user_resources already warned
    window = model.get("context_length") or model.get("max_context_length")
    before = json.dumps(settings, sort_keys = True)
    if compact_at is not None and window:
        # Pi compacts once the context exceeds contextWindow - reserveTokens.
        settings["compaction"] = {
            "enabled": True,
            "reserveTokens": _get_compaction_reserve(int(window), compact_at),
        }
    else:
        # Undo a --compact-at block an earlier run left in a persisted session; compaction
        # is not a key this session inherits from the user, so this exact shape is ours.
        compaction = settings.get("compaction")
        if (
            isinstance(compaction, dict)
            and compaction.keys() == {"enabled", "reserveTokens"}
            and compaction["enabled"] is True
            and isinstance(compaction["reserveTokens"], int)
        ):
            settings.pop("compaction", None)
    if json.dumps(settings, sort_keys = True) != before:
        _write_private_json(path, settings)


def _link_user_dir(source: Path, target: Path) -> bool:
    """Expose source at target. True once target resolves to source."""
    # Refresh links, but preserve real session directories.
    if target.is_symlink() or _is_junction(target):
        _remove_overlay_entry(target)
    if target.exists() or not source.is_dir():
        return False
    target.parent.mkdir(parents = True, exist_ok = True, mode = 0o700)
    try:
        target.symlink_to(source, target_is_directory = True)
    except OSError:
        if not _create_directory_junction(source, target):
            typer.echo(f"Warning: couldn't link {source} into the Pi session.", err = True)
            return False
    return True


def _pi_local_entry(
    entry: str,
    source: Path,
    home: Path,
    linked: frozenset,
    agents_skills = None,
) -> str:
    """Re-anchor a user path from the original Pi agent directory."""
    value = entry.strip()
    if not value or value == "." or value.startswith("file:"):
        # Nothing to anchor: "" and "." would name the whole agent directory.
        return entry
    if value == "~" or value.startswith(("~/", "~" + os.sep)):
        target = os.path.join(home, value[2:])
    else:
        # Pi stores local packages relative to its agent directory.
        target = os.path.join(source, value)
    target = os.path.normpath(target)
    if agents_skills is not None:
        # Pi reads ~/.agents/skills through HOME, which moved, so a rule naming the
        # user's copy must follow it or it stops matching.
        user_root, session_root = agents_skills
        try:
            inside = os.path.relpath(target, user_root)
        except ValueError:  # on another Windows drive
            inside = os.pardir
        if inside == os.curdir:
            return session_root
        if inside != os.pardir and not inside.startswith(os.pardir + os.sep):
            return os.path.join(session_root, inside)
    try:
        relative = os.path.relpath(target, source)
    except ValueError:  # on another Windows drive
        return target
    # Session-relative only where the link landed: a real session directory blocks
    # the link, and the entry would then point into it instead of at the user's.
    if relative.split(os.sep)[0] in linked:
        return relative
    return target


def _pi_settings_entries(
    key: str,
    entries,
    source: Path,
    home: Path,
    linked: frozenset,
    agents_skills = None,
) -> list:
    if not isinstance(entries, list):
        return []
    result = []
    for entry in entries:
        if key == "packages":
            spec = entry.get("source") if isinstance(entry, dict) else entry
            # All other package sources are local paths.
            if isinstance(spec, str) and not spec.strip().startswith(
                ("npm:", "git:", "github:", "http:", "https:", "ssh:")
            ):
                spec = _pi_local_entry(spec, source, home, linked, agents_skills)
                entry = {**entry, "source": spec} if isinstance(entry, dict) else spec
        elif isinstance(entry, str):
            prefix = entry[:1] if entry.startswith(("!", "+", "-")) else ""
            pattern = entry[len(prefix) :]
            if not prefix and "*" not in entry and "?" not in entry:
                entry = _pi_local_entry(entry, source, home, linked, agents_skills)
            elif not pattern.strip().startswith("~"):  # Pi does not expand ~ in patterns
                # Pi matches patterns against paths relative to the agent directory, which moved.
                # Keep the original too: it still matches basenames and linked directories.
                anchored = prefix + _pi_local_entry(
                    pattern,
                    source,
                    home,
                    linked,
                    agents_skills,
                )
                if anchored != entry:
                    result.append(entry)
                    entry = anchored
        result.append(entry)
    return result


def _clear_pi_user_resources(agent_dir: Path, home: Path) -> None:
    """Undo what an earlier launch linked and copied, leaving session state alone."""
    targets = [agent_dir / name for name in _PI_USER_RESOURCE_DIRS]
    targets.append(home / ".agents" / "skills")
    for target in targets:
        if target.is_symlink() or _is_junction(target):
            _remove_overlay_entry(target)
    manifest_path = agent_dir / _PI_USER_RESOURCES_MANIFEST
    previous = _read_json_object(manifest_path)
    if not previous:
        return
    settings_path = agent_dir / "settings.json"
    settings = _read_json_object(settings_path)
    if settings is None:
        return
    before = json.dumps(settings, sort_keys = True)
    for key, copied in previous.items():
        own = settings.get(key)
        if key in _PI_USER_VERBATIM_SETTINGS:
            # An argument vector, not entries: subtracting drops whatever the two share.
            if own == copied:
                settings.pop(key, None)
        elif isinstance(copied, list) and isinstance(own, list):
            rest = [item for item in own if item not in copied]
            if rest:
                settings[key] = rest
            else:
                settings.pop(key, None)
        elif own == copied:
            settings.pop(key, None)
    if json.dumps(settings, sort_keys = True) != before:
        _write_private_json(settings_path, settings)
    manifest_path.unlink(missing_ok = True)


def write_pi_user_resources(agent_dir: Path, home: Path) -> None:
    """Expose selected user Pi resources inside an isolated session."""
    if _wsl_windows_executable(["pi"]):
        # Windows Pi cannot reliably follow WSL links into mounted drives, and a session
        # an earlier Linux pi prepared still holds them, so drop those before returning.
        _clear_pi_user_resources(agent_dir, home)
        return
    user_home = Path.home()
    configured = os.environ.get("PI_CODING_AGENT_DIR")
    configured = configured.strip() if configured else ""
    # Pi resolves a relative override from the launch directory.
    source = (
        Path(os.path.abspath(os.path.expanduser(configured)))
        if configured
        else user_home / ".pi" / "agent"
    )
    if source.resolve(strict = False) == agent_dir.resolve(strict = False):
        # Do not treat this session as its own resource source.
        source = user_home / ".pi" / "agent"
    if configured and not source.is_dir():
        # Otherwise this looks exactly like the bug this function exists to fix.
        typer.echo(
            f"Warning: PI_CODING_AGENT_DIR points at {source}, which is not a directory; "
            "no Pi extensions or packages will load in this session.",
            err = True,
        )
    linked = frozenset(
        name for name in _PI_USER_RESOURCE_DIRS if _link_user_dir(source / name, agent_dir / name)
    )
    # HOME is relocated, so link Pi's other global skill directory too.
    user_skills = user_home / ".agents" / "skills"
    session_skills = home / ".agents" / "skills"
    agents_skills = (
        (str(user_skills), str(session_skills))
        if _link_user_dir(user_skills, session_skills)
        else None
    )

    user_settings_path = source / "settings.json"
    user_settings = _read_json_object(user_settings_path)
    if user_settings is None:
        typer.echo(
            f"Warning: couldn't parse {user_settings_path}; "
            "Pi packages listed there won't load in this session.",
            err = True,
        )
        user_settings = {}
    settings_path = agent_dir / "settings.json"
    settings = _read_json_object(settings_path)
    if settings is None:
        typer.echo(
            f"Warning: couldn't parse {settings_path}; your Pi packages won't load in this session.",
            err = True,
        )
        return
    manifest_path = agent_dir / _PI_USER_RESOURCES_MANIFEST
    previous = _read_json_object(manifest_path)
    if previous is None:
        # Provenance is lost, so entries the user has since removed cannot be reconciled.
        typer.echo(
            f"Warning: couldn't parse {manifest_path}; Pi resources copied by an earlier "
            "launch stay in this session even if you removed them since.",
            err = True,
        )
        previous = {}
    before = json.dumps(settings, sort_keys = True)
    copied = {}
    for key in _PI_USER_RESOURCE_SETTINGS:
        entries = _pi_settings_entries(
            key,
            user_settings.get(key),
            source,
            user_home,
            linked,
            agents_skills,
        )
        # Refresh copied entries while preserving settings added inside the session.
        stale = previous.get(key) if isinstance(previous.get(key), list) else []
        own = settings.get(key)
        if own is not None and not isinstance(own, list):
            # Pi types these as arrays; leave a shape we do not understand alone.
            typer.echo(
                f"Warning: {settings_path} has a non-list {key!r}; "
                "leaving it as is, so your Pi entries for it won't load in this session.",
                err = True,
            )
            continue
        own = [item for item in own or [] if item not in stale and item not in entries]
        if entries or own:
            # Pi de-dupes packages by identity keeping the FIRST, so session entries
            # lead. Patterns apply in order instead, so those stay user-first.
            settings[key] = own + entries if key == "packages" else entries + own
        else:
            settings.pop(key, None)
        if entries:
            copied[key] = entries
    for key in _PI_USER_VERBATIM_SETTINGS:
        # Pi runs every package lookup and install through npmCommand, so inheriting
        # the package list without it falls back to an npm that cannot find them.
        value = user_settings.get(key)
        own = settings.get(key)
        if own is not None and own != previous.get(key):
            continue  # changed inside the session, so the session owns it now
        if isinstance(value, list) and value and all(isinstance(arg, str) for arg in value):
            settings[key] = value
            copied[key] = value
        else:
            settings.pop(key, None)
    if json.dumps(settings, sort_keys = True) != before:
        _write_private_json(settings_path, settings)
    if copied != previous:
        if copied:
            _write_private_json(manifest_path, copied)
        else:
            manifest_path.unlink(missing_ok = True)


def write_pi_subagent_config(
    base: str,
    key: str,
    model: dict,
    path: Path,
    approve: bool = False,
    max_tokens: Optional[int] = None,
    request_body: Optional[dict] = None,
    headers: Optional[dict] = None,
) -> None:
    """Write private bootstrap data for the bundled Pi extension."""
    window = model.get("context_length") or model.get("max_context_length")
    window = int(window) if window else 32768
    _write_private_json(
        path,
        {
            "baseUrl": f"{base}/v1",
            "apiKey": key,
            "model": model["id"],
            "contextWindow": window,
            "maxTokens": _agent_output_limit(window, max_tokens),
            "approve": approve,
            **({"headers": _pi_header_values(headers)} if headers else {}),
            **({"samplingParams": request_body} if request_body else {}),
        },
    )


def write_dsh_patch(
    base: str,
    model: dict,
    path: Path,
    request_body: Optional[dict] = None,
    *,
    headers: Optional[dict] = None,
    max_tokens: Optional[int] = None,
    compact_at: Optional[float] = None,
) -> None:
    """Write the dsh loader patch that points the booted profile at the model server.

    dsh 0.1.7 dropped `settings.yaml`: it now imports a leftover one into the profile only
    after the first boot has settled, so that boot still runs on the DeepSeek default. A
    `--patch` overlay is read at boot on every dsh release this supports, and the file is
    agent-switch's own, so it is rewritten whole rather than merged.
    """
    import yaml

    model_entry = {"id": model["id"]}
    window = model.get("context_length") or model.get("max_context_length")
    if window:
        window = int(window)
        model_entry["contextWindow"] = window
        model_entry["maxTokens"] = _agent_output_limit(window, max_tokens)
    elif max_tokens:
        model_entry["maxTokens"] = max_tokens
    compat = {"supportsDeveloperRole": False, "maxTokensField": "max_tokens"}
    if request_body:
        # dsh sends template kwargs only for a model that declares reasoning levels.
        model_entry["reasoningEfforts"] = {
            "off": None,
            "low": "low",
            "medium": "medium",
            "high": "high",
        }
        compat["thinkingFormat"] = "chat-template"
        compat["chatTemplateKwargs"] = request_body
    entries = [
        {
            "id": "llm-pi-ai",
            "name": "@deepseek-ai/dsh-llm-pi-ai",
            "config": {
                "providers": {
                    _DSH_PROVIDER: {
                        "displayName": "agent-switch",
                        "api": "openai-completions",
                        "baseURL": f"{base}/v1",
                        "apiKeyEnv": _DSH_ENV_KEY,
                        # Values go out verbatim: unlike Pi, dsh resolves no $NAME or !command.
                        **({"headers": headers} if headers else {}),
                        # pi-ai reads an unknown base URL as OpenAI itself.
                        "compat": compat,
                        "models": [model_entry],
                    }
                }
            },
        },
        {
            "id": "agent-default-model",
            "name": "@deepseek-ai/dsh-agent-default-model",
            "config": {"provider": _DSH_PROVIDER, "model": model["id"]},
        },
    ]
    if compact_at is not None and window:
        # dsh compacts once the context passes thresholdRatio of the model's window.
        entries.append(
            {
                "id": "compaction-basic",
                "name": "@deepseek-ai/dsh-compaction-basic",
                "config": {"thresholdRatio": compact_at},
            }
        )
    text = yaml.safe_dump(entries, sort_keys = False)
    if not path.exists() or path.read_text(encoding = "utf-8") != text:
        # Private: a --header value may be a token.
        _write_private_text(path, text)
        typer.echo(f"Updated {path}")


@start_app.command("claude", cls = _PassthroughCommand, context_settings = _PASSTHROUGH)
def claude(
    ctx: typer.Context,
    model: Optional[str] = _MODEL_OPTION,
    api_key: Optional[str] = _KEY_OPTION,
    header: Optional[list[str]] = _HEADER_OPTION,
    launch: bool = _LAUNCH_OPTION,
    max_seq_length: int = _CONTEXT_OPTION,
    reasoning: Optional[Literal["on", "off", "auto"]] = _REASONING_OPTION,
    reasoning_effort: Optional[str] = _REASONING_EFFORT_OPTION,
    temperature: Optional[float] = _TEMPERATURE_OPTION,
    top_p: Optional[float] = _TOP_P_OPTION,
    top_k: Optional[int] = _TOP_K_OPTION,
    min_p: Optional[float] = _MIN_P_OPTION,
    repetition_penalty: Optional[float] = _REPETITION_PENALTY_OPTION,
    presence_penalty: Optional[float] = _PRESENCE_PENALTY_OPTION,
    compact_at: Optional[float] = _COMPACT_AT_OPTION,
    model_load: bool = _MODEL_LOAD_OPTION,
    url: Optional[str] = _URL_OPTION,
    provider: Optional[ProviderName] = _PROVIDER_OPTION,
    yolo: bool = _YOLO_OPTION,
    persist: bool = _PERSIST_OPTION,
    as_subagent: bool = _AS_SUBAGENT_OPTION,
):
    """Point Claude Code at a local model server and start it."""
    # Route a leading `org/name` positional to --model; forward the rest to the agent.
    model, ctx.args[:] = _consume_positional_model(model, ctx.args)
    headers = parse_headers(header)
    target = _resolve_target(url, provider, api_key, headers)
    install_hint = (
        "irm https://claude.ai/install.ps1 | iex"
        if os.name == "nt"
        else "curl -fsSL https://claude.ai/install.sh | bash"
    )
    _require_agent_for_launch("claude", install_hint, launch)
    server_options = ServerOptions(
        reasoning = reasoning,
        reasoning_effort = reasoning_effort,
        temperature = temperature,
        top_p = top_p,
        top_k = top_k,
        min_p = min_p,
        repetition_penalty = repetition_penalty,
        presence_penalty = presence_penalty,
        carried = _ALL_REQUEST_FIELDS,
        provider = target.name,
    )
    base, key, entry = _connect(
        api_key,
        model,
        LoadOptions(max_seq_length, model_load),
        server_options = server_options,
        target = target,
        needs = ("/v1/messages",),
    )
    model_id = entry["id"]
    _check_compact_at(compact_at, entry)
    if as_subagent:
        window = entry.get("context_length") or entry.get("max_context_length")
        server_env = {
            "AGENT_SWITCH_CLAUDE_SUBAGENT_BASE_URL": base,
            "AGENT_SWITCH_CLAUDE_SUBAGENT_API_KEY": key,
            "AGENT_SWITCH_CLAUDE_SUBAGENT_MODEL": model_id,
            "AGENT_SWITCH_CLAUDE_SUBAGENT_BYPASS_PERMISSIONS": "1" if yolo else "0",
        }
        if window:
            server_env["AGENT_SWITCH_CLAUDE_SUBAGENT_CONTEXT_WINDOW"] = str(int(window))
        if compact_at is not None:
            server_env["AGENT_SWITCH_CLAUDE_SUBAGENT_COMPACT_AT"] = str(compact_at)
        if headers:
            server_env["AGENT_SWITCH_CLAUDE_SUBAGENT_HEADERS"] = json.dumps(headers)
        if server_options.request_body():
            server_env["CLAUDE_CODE_EXTRA_BODY"] = json.dumps(server_options.request_body())
        with _session_config("claude-subagent", launch, persist = persist) as config:
            plugin = write_claude_subagent_plugin(config, server_env)
            command = [
                "claude",
                "--plugin-dir",
                _agent_config_path(plugin, ["claude"]),
                # Before ctx.args: a forwarded `--` would turn later flags positional. The `=` form is used because --allowedTools is variadic, so a detached value swallows the first forwarded positional.
                f"--allowedTools={_CLAUDE_SUBAGENT_TOOL},{_CLAUDE_SUBAGENT_PLAN_TOOL}",
                *_yolo_command_flags("claude", yolo),
                *ctx.args,
            ]
            typer.echo(
                "A local agent is available. Ask Claude to spawn a local agent."
            )
            _run(
                base,
                entry,
                {},
                command,
                launch = launch,
                install_hint = install_hint,
            )
        return

    env = _claude_local_env(base, key, entry, server_options.request_body(), headers, compact_at)
    # Claude Code auto-compacts against its native context window; the local env above supplies the loaded model's real window and a 90% threshold instead. --yolo (or its aliases) maps to Claude's own --dangerously-skip-permissions. IS_SANDBOX is left unset on purpose: Claude refuses bypass mode as root unless a sandbox is detected, and we do not want to falsely claim one on the user's host. claude keeps its history in ~/.claude/projects, which --settings/env never relocate, so a session already survives exit; resume it with `claude --continue` or `--resume <id>` passed through.
    with _session_config("claude", launch, persist = persist) as config:
        settings = _write_claude_settings(config, model_id, env)
        command = _claude_local_command(
            model_id,
            _agent_config_path(settings, ["claude"]),
            yolo,
            ctx.args,
        )
        _run(
            base,
            entry,
            env,
            command,
            launch = launch,
            install_hint = install_hint,
            unset_env = _CLAUDE_ENV_UNSET,
        )


@start_app.command("codex", cls = _PassthroughCommand, context_settings = _PASSTHROUGH)
def codex(
    ctx: typer.Context,
    model: Optional[str] = _MODEL_OPTION,
    api_key: Optional[str] = _KEY_OPTION,
    header: Optional[list[str]] = _HEADER_OPTION,
    launch: bool = _LAUNCH_OPTION,
    max_seq_length: int = _CONTEXT_OPTION,
    reasoning: Optional[Literal["on", "off", "auto"]] = _REASONING_OPTION,
    reasoning_effort: Optional[str] = _REASONING_EFFORT_OPTION,
    temperature: Optional[float] = _TEMPERATURE_OPTION,
    top_p: Optional[float] = _TOP_P_OPTION,
    top_k: Optional[int] = _TOP_K_OPTION,
    min_p: Optional[float] = _MIN_P_OPTION,
    repetition_penalty: Optional[float] = _REPETITION_PENALTY_OPTION,
    presence_penalty: Optional[float] = _PRESENCE_PENALTY_OPTION,
    compact_at: Optional[float] = _COMPACT_AT_OPTION,
    model_load: bool = _MODEL_LOAD_OPTION,
    url: Optional[str] = _URL_OPTION,
    provider: Optional[ProviderName] = _PROVIDER_OPTION,
    yolo: bool = _YOLO_OPTION,
    persist: bool = _PERSIST_OPTION,
    as_subagent: bool = _AS_SUBAGENT_OPTION,
):
    """Point OpenAI Codex at a local model server and start it."""
    # Route a leading `org/name` positional to --model; forward the rest to the agent.
    model, ctx.args[:] = _consume_positional_model(model, ctx.args)
    headers = parse_headers(header)
    target = _resolve_target(url, provider, api_key, headers)
    install_hint = _npm_install_hint("@openai/codex")
    _require_agent_for_launch("codex", install_hint, launch)
    codex_effort = _codex_reasoning_effort(reasoning, reasoning_effort)
    if codex_effort and not _agent_version_at_least("codex", _CODEX_REASONING_REQUEST_MIN_VERSION):
        codex_effort = None
    server_options = ServerOptions(
        reasoning = reasoning,
        reasoning_effort = reasoning_effort,
        temperature = temperature,
        top_p = top_p,
        top_k = top_k,
        min_p = min_p,
        repetition_penalty = repetition_penalty,
        presence_penalty = presence_penalty,
        carried = _REASONING_FIELDS if codex_effort else frozenset(),
        provider = target.name,
    )
    base, key, entry = _connect(
        api_key,
        model,
        LoadOptions(max_seq_length, model_load),
        server_options = server_options,
        target = target,
        needs = ("/v1/responses",),
    )
    _check_compact_at(compact_at, entry)
    if as_subagent:
        with _session_config("codex-subagent", launch, persist = persist) as home:
            bridge_config = write_codex_subagent_bridge(
                base,
                key,
                entry,
                home,
                yolo = yolo,
                reasoning_effort = codex_effort,
                headers = headers,
                compact_at = compact_at,
            )
            parent_home = write_codex_parent_overlay(home / "parent")
            command = [
                "codex",
                *_codex_subagent_flags(bridge_config),
                *_yolo_command_flags("codex", yolo),
                *ctx.args,
            ]
            typer.echo(
                "A local agent is available. Ask Codex to spawn a local agent."
            )
            _run(
                base,
                entry,
                {"CODEX_HOME": str(parent_home)},
                command,
                launch = launch,
                install_hint = install_hint,
            )
        return
    command = [
        "codex",
        "--oss",
        "--profile",
        _CODEX_PROFILE,
        *_yolo_command_flags("codex", yolo),
        *ctx.args,
    ]
    with _session_config("codex", launch, persist = persist) as home:
        write_codex_config(base, entry, home, codex_effort, headers, compact_at)
        env = {_CODEX_ENV_KEY: key, "CODEX_HOME": str(home)}
        _run(
            base,
            entry,
            env,
            command,
            launch = launch,
            install_hint = install_hint,
            unset_env = _CODEX_ENV_UNSET,
        )


@start_app.command("opencode", cls = _PassthroughCommand, context_settings = _PASSTHROUGH)
def opencode(
    ctx: typer.Context,
    model: Optional[str] = _MODEL_OPTION,
    api_key: Optional[str] = _KEY_OPTION,
    header: Optional[list[str]] = _HEADER_OPTION,
    launch: bool = _LAUNCH_OPTION,
    max_seq_length: int = _CONTEXT_OPTION,
    reasoning: Optional[Literal["on", "off", "auto"]] = _REASONING_OPTION,
    reasoning_effort: Optional[str] = _REASONING_EFFORT_OPTION,
    temperature: Optional[float] = _TEMPERATURE_OPTION,
    top_p: Optional[float] = _TOP_P_OPTION,
    top_k: Optional[int] = _TOP_K_OPTION,
    min_p: Optional[float] = _MIN_P_OPTION,
    repetition_penalty: Optional[float] = _REPETITION_PENALTY_OPTION,
    presence_penalty: Optional[float] = _PRESENCE_PENALTY_OPTION,
    max_tokens: Optional[int] = _MAX_TOKENS_OPTION,
    compact_at: Optional[float] = _COMPACT_AT_OPTION,
    model_load: bool = _MODEL_LOAD_OPTION,
    url: Optional[str] = _URL_OPTION,
    provider: Optional[ProviderName] = _PROVIDER_OPTION,
    yolo: bool = _YOLO_OPTION,
    persist: bool = _PERSIST_OPTION,
    as_subagent: bool = _AS_SUBAGENT_OPTION,
):
    """Point OpenCode at a local model server and start it."""
    # Route a leading `org/name` positional to --model; forward the rest to the agent.
    model, ctx.args[:] = _consume_positional_model(model, ctx.args)
    headers = parse_headers(header)
    target = _resolve_target(url, provider, api_key, headers)
    command_name, opencode_v2 = _opencode_command()
    install_hint = _npm_install_hint("@opencode-ai/cli@beta" if opencode_v2 else "opencode-ai")
    _require_agent_for_launch(command_name, install_hint, launch)
    server_options = ServerOptions(
        reasoning = reasoning,
        reasoning_effort = reasoning_effort,
        temperature = temperature,
        top_p = top_p,
        top_k = top_k,
        min_p = min_p,
        repetition_penalty = repetition_penalty,
        presence_penalty = presence_penalty,
        carried = _ALL_REQUEST_FIELDS,
        provider = target.name,
    )
    base, key, entry = _connect(
        api_key,
        model,
        LoadOptions(max_seq_length, model_load),
        server_options = server_options,
        target = target,
    )
    _check_compact_at(compact_at, entry)
    if opencode_v2:
        typer.echo(
            f"OpenCode V2 provider policies must allow '{_OPENCODE_PROVIDER}'.",
            err = True,
        )
    if as_subagent:
        if compact_at is not None:
            typer.echo(
                "Warning: --compact-at does not apply with --as-subagent for OpenCode; ignoring it.",
                err = True,
            )
        # Stay append-safe for a bare no-launch recipe: a later `run <prompt>` would make `opencode --auto run ...` parse as the TUI, so keep yolo in the inline fallback.
        route_native_auto = (
            yolo and _opencode_supports_native_auto(command_name) and (launch or bool(ctx.args))
        )
        opencode_args = list(ctx.args)
        if opencode_v2:
            opencode_args = _opencode_v2_standalone_args(opencode_args)
        opencode_args, native_auto = _opencode_native_auto_args(
            opencode_args, route_native_auto, v2 = opencode_v2
        )
        command = [command_name, *opencode_args]
        with _session_config("opencode-subagent", launch, persist = persist) as cfg:
            config_path = cfg / "opencode.json"
            session_permission = write_opencode_config(
                base,
                key,
                entry,
                config_path,
                yolo = yolo and not native_auto,
                as_subagent = True,
                max_tokens = max_tokens,
                request_body = server_options.request_body(),
                headers = headers,
            )
            env = {
                "OPENCODE_CONFIG": str(config_path),
                **_opencode_output_env(entry, max_tokens),
            }
            inline_config = _opencode_subagent_inline_config(
                config_path,
                session_permission,
                command = command_name,
                v2 = opencode_v2,
            )
            # A project opencode.json outranks the session file and could field-merge its own agent.local over ours. Pin ours in the inline overlay so it wins.
            inline_config.setdefault("agent", {})[_SUBAGENT_NAME] = {
                "description": _SUBAGENT_DESCRIPTION,
                "mode": "subagent",
                "model": f"{_OPENCODE_PROVIDER}/{entry['id']}",
                "prompt": _SUBAGENT_INSTRUCTIONS,
            }
            env["OPENCODE_CONFIG_CONTENT"] = json.dumps(inline_config)
            typer.echo(f"The local model is available as @{_SUBAGENT_NAME} and in /models.")
            _run(
                base,
                entry,
                env,
                command,
                launch = launch,
                install_hint = install_hint,
            )
        return
    opencode_model = f"{_OPENCODE_PROVIDER}/{entry['id']}"
    # The inline OPENCODE_CONFIG_CONTENT below pins the model in the highest-priority layer, so the session model is forced without a --model flag. Only add --model for an interactive bare launch, as a convenience so the TUI opens on our model: it is omitted for passthrough, where inserting it before a subcommand can be misparsed, and for --no-launch, where the printed command is consumed by drivers that append a subcommand such as `run <prompt>` and a leading --model would land before it. Those paths rely on the inline pin instead.
    native_auto = False
    route_native_auto = yolo and _opencode_supports_native_auto(command_name)
    if ctx.args:
        opencode_args = list(ctx.args)
        if opencode_v2:
            opencode_args = _opencode_v2_standalone_args(opencode_args)
        opencode_args, native_auto = _opencode_native_auto_args(
            opencode_args, route_native_auto, v2 = opencode_v2
        )
        command = [command_name, *opencode_args]
    elif launch:
        opencode_args = [] if opencode_v2 else ["--model", opencode_model]
        if opencode_v2:
            opencode_args = _opencode_v2_standalone_args(opencode_args)
        opencode_args, native_auto = _opencode_native_auto_args(
            opencode_args,
            route_native_auto,
            v2 = opencode_v2,
        )
        command = [command_name, *opencode_args]
    else:
        # Append-safe base: `opencode --auto run ...` parses as the TUI with a project "run", not the run subcommand. The command is unknown here, so keep the config fallback.
        opencode_args = _opencode_v2_standalone_args([]) if opencode_v2 else []
        command = [command_name, *opencode_args]
    # opencode keeps sessions in ~/.local/share/opencode (never relocated), so resume already survives exit; reopen the last one by passing `opencode --continue` through.
    with _session_config("opencode", launch, persist = persist) as cfg:
        config_path = cfg / "opencode.json"
        # OPENCODE_CONFIG is an overlay, loaded between the user's global and project configs, so this adds the agent-switch provider/model for the session without changing the user's default model. The key lives in the config, not the env.
        session_permission = write_opencode_config(
            base,
            key,
            entry,
            config_path,
            yolo = yolo and not native_auto,
            max_tokens = max_tokens,
            request_body = server_options.request_body(),
            headers = headers,
            compact_at = compact_at,
        )
        # A project's own opencode.json outranks OPENCODE_CONFIG, so the session model pin would silently lose to a repo config; carry it in OPENCODE_CONFIG_CONTENT, which outranks project config, while the API key stays in the private file. Only the config fallback carries a permission: native --auto omits it (auto-approve asks, keep explicit denies) and a non-yolo session omits it too, honoring project rules. V1 filters are ordinary overlays, so scope that session to our provider; V2 turns filters into security policies where global/project rules intentionally win, so keep those policies intact and tell the user above that they must allow our provider. small_model is opencode's separate model for lightweight tasks; pin it to the session model too, or a user/project small_model on another (now filtered) provider would resolve a not-found error mid-session.
        inline_config: dict = {
            "model": opencode_model,
            "small_model": opencode_model,
        }
        if not opencode_v2:
            inline_config["enabled_providers"] = [_OPENCODE_PROVIDER]
            inline_config["disabled_providers"] = []
        if session_permission:
            inline_config["permission"] = session_permission
        env = {
            "OPENCODE_CONFIG": str(config_path),
            "OPENCODE_CONFIG_CONTENT": json.dumps(inline_config),
            **_opencode_output_env(entry, max_tokens),
        }
        _run(base, entry, env, command, launch = launch, install_hint = install_hint)


@start_app.command("pi", cls = _PassthroughCommand, context_settings = _PASSTHROUGH)
def pi(
    ctx: typer.Context,
    model: Optional[str] = _MODEL_OPTION,
    api_key: Optional[str] = _KEY_OPTION,
    header: Optional[list[str]] = _HEADER_OPTION,
    launch: bool = _LAUNCH_OPTION,
    max_seq_length: int = _CONTEXT_OPTION,
    max_tokens: Optional[int] = _MAX_TOKENS_OPTION,
    reasoning: Optional[Literal["on", "off", "auto"]] = _REASONING_OPTION,
    reasoning_effort: Optional[str] = _REASONING_EFFORT_OPTION,
    temperature: Optional[float] = _TEMPERATURE_OPTION,
    top_p: Optional[float] = _TOP_P_OPTION,
    top_k: Optional[int] = _TOP_K_OPTION,
    min_p: Optional[float] = _MIN_P_OPTION,
    repetition_penalty: Optional[float] = _REPETITION_PENALTY_OPTION,
    presence_penalty: Optional[float] = _PRESENCE_PENALTY_OPTION,
    compact_at: Optional[float] = _COMPACT_AT_OPTION,
    model_load: bool = _MODEL_LOAD_OPTION,
    url: Optional[str] = _URL_OPTION,
    provider: Optional[ProviderName] = _PROVIDER_OPTION,
    yolo: bool = _YOLO_OPTION,
    persist: bool = _PERSIST_OPTION,
    as_subagent: bool = _AS_SUBAGENT_OPTION,
):
    """Point Pi (coding agent) at a local model server and start it."""
    # Route a leading `org/name` positional to --model; forward the rest to the agent.
    model, ctx.args[:] = _consume_positional_model(model, ctx.args)
    headers = parse_headers(header)
    target = _resolve_target(url, provider, api_key, headers)
    install_hint = _npm_install_hint(
        "@earendil-works/pi-coding-agent",
        ignore_scripts = True,
    )
    if as_subagent and not _PI_SUBAGENT_EXTENSION.is_file():
        _fail(f"Missing Pi subagent extension: {_PI_SUBAGENT_EXTENSION}")
    _require_agent_for_launch("pi", install_hint, launch)
    server_options = ServerOptions(
        reasoning = reasoning,
        reasoning_effort = reasoning_effort,
        temperature = temperature,
        top_p = top_p,
        top_k = top_k,
        min_p = min_p,
        repetition_penalty = repetition_penalty,
        presence_penalty = presence_penalty,
        carried = _ALL_REQUEST_FIELDS,
        provider = target.name,
    )
    if server_options.request_body() and not _agent_version_at_least(
        "pi", _PI_SAMPLING_PARAMS_MIN_VERSION
    ):
        server_options = server_options._replace(carried = frozenset())
    base, key, entry = _connect(
        api_key,
        model,
        LoadOptions(max_seq_length, model_load),
        server_options = server_options,
        target = target,
    )
    _check_compact_at(compact_at, entry)
    if as_subagent:
        if compact_at is not None:
            typer.echo(
                "Warning: --compact-at does not apply with --as-subagent for Pi; ignoring it.",
                err = True,
            )
        extension = _agent_config_path(_PI_SUBAGENT_EXTENSION, ["pi"])
        with _session_config("pi-subagent", launch, persist = persist) as config:
            config_path = config / "subagent.json"
            write_pi_subagent_config(
                base,
                key,
                entry,
                config_path,
                approve = yolo,
                max_tokens = max_tokens,
                request_body = server_options.request_body(),
                headers = headers,
            )
            command = [
                "pi",
                "--extension",
                extension,
                *_yolo_command_flags("pi", yolo),
                *ctx.args,
            ]
            typer.echo(
                "A local agent is available, and the model is in /model. "
                "Ask Pi to spawn a local agent."
            )
            _run(
                base,
                entry,
                {"AGENT_SWITCH_PI_SUBAGENT_CONFIG": str(config_path)},
                command,
                launch = launch,
                install_hint = install_hint,
                clear_screen = True,
            )
        return
    # Pi defaults to the google provider, so pin our provider/model on the command line; the custom OpenAI-compatible endpoint itself is only configurable via ~/.pi/agent/models.json.
    command = [
        "pi",
        "--provider",
        _PI_PROVIDER,
        "--model",
        entry["id"],
        *_yolo_command_flags("pi", yolo),
        *ctx.args,
    ]
    # --ignore-scripts matches Pi's documented install recipe (its README notes Pi needs no install scripts), so accepting the prompt skips dependency lifecycle scripts.
    with _session_config("pi", launch, persist = persist) as home:
        # Pi resolves its config dir from PI_CODING_AGENT_DIR first (getAgentDir() prefers it over $HOME/.pi/agent), so pin it at the session dir: an inherited PI_CODING_AGENT_DIR in the user's shell would otherwise send Pi to their real config and skip our provider/key. HOME is relocated too so any other ~/.pi paths stay in the session. The key rides in the config rather than the env.
        pi_agent_dir = home / ".pi" / "agent"
        write_pi_config(
            base,
            key,
            entry,
            pi_agent_dir / "models.json",
            max_tokens = max_tokens,
            request_body = server_options.request_body(),
            headers = headers,
        )
        write_pi_user_resources(pi_agent_dir, home)
        write_pi_compaction(pi_agent_dir, entry, compact_at)
        env = {"HOME": str(home), "PI_CODING_AGENT_DIR": str(pi_agent_dir)}
        if os.name == "nt" or os.environ.get("WSL_DISTRO_NAME"):
            # Node resolves ~/.pi via USERPROFILE (then HOMEDRIVE + HOMEPATH) on Windows, not HOME. Set them whenever Pi may run as a Windows process: native Windows, or a /mnt Windows shim launched from WSL, where the WSLENV bridge then translates the path. Otherwise the Windows process falls back to the user's real %USERPROFILE%\\.pi. splitdrive yields no drive off a POSIX path, so HOMEDRIVE/HOMEPATH stay unset there.
            env["USERPROFILE"] = str(home)
            drive, tail = os.path.splitdrive(str(home))
            if drive:
                env["HOMEDRIVE"], env["HOMEPATH"] = drive, tail
        # Pi paints inline from the current cursor position (no alternate screen, no clear on first render), so give it the clean screen it assumes.
        _run(
            base,
            entry,
            env,
            command,
            launch = launch,
            install_hint = install_hint,
            clear_screen = True,
        )


@start_app.command("dsh", cls = _PassthroughCommand, context_settings = _PASSTHROUGH)
def dsh(
    ctx: typer.Context,
    model: Optional[str] = _MODEL_OPTION,
    api_key: Optional[str] = _KEY_OPTION,
    header: Optional[list[str]] = _HEADER_OPTION,
    launch: bool = _LAUNCH_OPTION,
    max_seq_length: int = _CONTEXT_OPTION,
    max_tokens: Optional[int] = _MAX_TOKENS_OPTION,
    reasoning: Optional[Literal["on", "off", "auto"]] = _REASONING_OPTION,
    reasoning_effort: Optional[str] = _REASONING_EFFORT_OPTION,
    temperature: Optional[float] = _TEMPERATURE_OPTION,
    top_p: Optional[float] = _TOP_P_OPTION,
    top_k: Optional[int] = _TOP_K_OPTION,
    min_p: Optional[float] = _MIN_P_OPTION,
    repetition_penalty: Optional[float] = _REPETITION_PENALTY_OPTION,
    presence_penalty: Optional[float] = _PRESENCE_PENALTY_OPTION,
    compact_at: Optional[float] = _COMPACT_AT_OPTION,
    model_load: bool = _MODEL_LOAD_OPTION,
    url: Optional[str] = _URL_OPTION,
    provider: Optional[ProviderName] = _PROVIDER_OPTION,
    yolo: bool = _YOLO_OPTION,
    persist: bool = _PERSIST_OPTION,
):
    """Point DeepSeek Harness (dsh) at a local model server and start it."""
    model, ctx.args[:] = _consume_positional_model(model, ctx.args)
    _reject_as_subagent("dsh", ctx.args)
    headers = parse_headers(header)
    target = _resolve_target(url, provider, api_key, headers)
    install_hint = _npm_install_hint(_DSH_PACKAGE)
    _require_agent_for_launch("dsh", install_hint, launch)
    # dsh sends reasoning only as chat_template_kwargs, which only these servers read.
    carried = (
        _REASONING_FIELDS
        if providers.get_has_template_kwargs(target.name)
        else frozenset()
    )
    server_options = ServerOptions(
        reasoning = reasoning,
        reasoning_effort = reasoning_effort,
        temperature = temperature,
        top_p = top_p,
        top_k = top_k,
        min_p = min_p,
        repetition_penalty = repetition_penalty,
        presence_penalty = presence_penalty,
        carried = carried,
        provider = target.name,
    )
    base, key, entry = _connect(
        api_key,
        model,
        LoadOptions(max_seq_length, model_load),
        server_options = server_options,
        target = target,
    )
    _check_compact_at(compact_at, entry)
    with _session_config("dsh", launch, persist = persist) as home:
        patch = home / _DSH_PATCH_FILE
        write_dsh_patch(
            base,
            entry,
            patch,
            # dsh wraps these in chat_template_kwargs itself, so pass the untranslated fields.
            server_options._replace(provider = None).request_body(),
            headers = headers,
            max_tokens = max_tokens,
            compact_at = compact_at,
        )
        # A Windows dsh under WSL gets DSH_HOME translated through WSLENV, but not argv.
        command = _dsh_command(ctx.args, _agent_config_path(patch, ["dsh"]))
        if compact_at is not None and _get_dsh_boot_profile(command) == "web":
            typer.echo(
                "Warning: --compact-at is ignored for dsh's web profile. "
                "Use --profile headless to apply it.",
                err = True,
            )
        env = {
            _DSH_ENV_KEY: key,
            "DSH_HOME": str(home),
            # dsh uploads session records once a user records /feedback.
            "DSH_TELEMETRY_DISABLED": "1",
            "DSH_PERMISSION_MODE": (
                _DSH_YOLO_PERMISSION_MODE if yolo else _DSH_SAFE_PERMISSION_MODE
            ),
        }
        _run(base, entry, env, command, launch = launch, install_hint = install_hint)
