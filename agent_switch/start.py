# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`agent-switch <agent>` — launch a coding agent against a running model server."""

import typer

from agent_switch.agents import claude, codex, dsh, opencode, pi
from agent_switch.core.options import _PASSTHROUGH, _PassthroughCommand
from agent_switch.core.storage import _agent_switch_home

start_app = typer.Typer(
    help = "Start a coding agent against a local model server: Ollama, LM Studio, "
    "llama-server, vLLM or any OpenAI-compatible server.",
    no_args_is_help = True,
    context_settings = {"help_option_names": ["-h", "--help"]},
)

# Typer keeps single newlines as line breaks and turns \n\n into a blank line, so the example
# lines are laid out one per line. Rich only reads [tag] markup starting lowercase/#//@, which
# the JSON brackets below never match.
_MCP_EPILOG = (
    f"MCP servers: define them once in {_agent_switch_home() / 'mcp.json'}, in the .mcp.json shape:\n"
    '{"mcpServers": {\n'
    '  "context7": {"command": "npx", "args": ["-y", "@upstash/context7-mcp"]},\n'
    '  "github": {"type": "http", "url": "https://api.githubcopilot.com/mcp/"}\n'
    "}}\n\n"
    "Mount them per session with --mcp NAME (repeatable) or --mcp-all; the mounted servers replace "
    "the agent's own MCP servers. An http server takes url, headers and oauth; ${VAR} in args, env, "
    "url and header values is expanded from your environment at launch.\n\n"
    "Examples: agent-switch claude --mcp context7\n"
    "agent-switch codex --mcp-all"
)

# Registration order is the --help order; the old dsh decorators applied bottom-up (dsh, dsh-tui, dst).
start_app.command(
    "claude", cls = _PassthroughCommand, context_settings = _PASSTHROUGH, epilog = _MCP_EPILOG
)(claude.claude)
start_app.command(
    "codex", cls = _PassthroughCommand, context_settings = _PASSTHROUGH, epilog = _MCP_EPILOG
)(codex.codex)
start_app.command(
    "opencode", cls = _PassthroughCommand, context_settings = _PASSTHROUGH, epilog = _MCP_EPILOG
)(opencode.opencode)
start_app.command(
    "pi", cls = _PassthroughCommand, context_settings = _PASSTHROUGH, epilog = _MCP_EPILOG
)(pi.pi)
start_app.command(
    "dsh", cls = _PassthroughCommand, context_settings = _PASSTHROUGH, epilog = _MCP_EPILOG
)(dsh.dsh)
start_app.command(
    "dsh-tui",
    cls = _PassthroughCommand,
    context_settings = _PASSTHROUGH,
    epilog = _MCP_EPILOG,
    help = "Point the DeepSeek Harness TUI (dsh-tui) at a local model server and start it.",
)(dsh.dsh)
start_app.command(
    "dst",
    cls = _PassthroughCommand,
    context_settings = _PASSTHROUGH,
    epilog = _MCP_EPILOG,
    help = "Alias of dsh-tui.",
)(dsh.dsh)
