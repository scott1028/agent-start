# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`agent-switch <agent>` — launch a coding agent against a running model server."""

import re
from typing import Optional

import click
import typer
from typer.core import TyperGroup

from agent_switch.agents import claude, codex, dsh, opencode, pi
from agent_switch.core.launch import _ALIAS_KIND_META, _ALIAS_META
from agent_switch.core.options import _PASSTHROUGH, _PassthroughCommand
from agent_switch.core.storage import _agent_switch_home

# `dsh-tui-<x>` must win over `dsh-<x>`, so the longer kinds come first; the suffix keeps the
# whole name inside the shell-safe characters _check_alias accepts.
_ALIAS_KIND = re.compile(r"(dsh-tui|dsh|claude|codex|opencode|pi)-[A-Za-z0-9._-]+")


class _AliasGroup(TyperGroup):
    """Resolve `<agent>-<x>` names to that agent's command, for the user's own shell alias."""

    def get_command(self, ctx: click.Context, name: str) -> Optional[click.Command]:
        command = super().get_command(ctx, name)
        if command is not None:
            return command
        match = _ALIAS_KIND.fullmatch(name)
        if match is None:
            return None
        # Click shares ctx.meta with the child context, so the command sees the alias it runs.
        ctx.meta[_ALIAS_META] = name
        ctx.meta[_ALIAS_KIND_META] = match.group(1)
        return super().get_command(ctx, match.group(1))


start_app = typer.Typer(
    help = "With --url/--provider the agent runs against that local model server (Ollama, LM "
    "Studio, llama-server, vLLM or any OpenAI-compatible server) with a throwaway, session-only "
    "configuration; without them it runs natively, on its own model, login and config, and only "
    "the shared flags (MCP mounting, --yolo) are added. Either way your own agent config is not "
    "modified. "
    "A name like claude-<x>, codex-<x>, opencode-<x>, pi-<x>, dsh-<x> or dsh-tui-<x> runs your "
    "own bash function, alias or script of that name as that agent.",
    no_args_is_help = True,
    context_settings = {"help_option_names": ["-h", "--help"]},
    cls = _AliasGroup,
)

# Typer keeps single newlines as line breaks and turns \n\n into a blank line, so the example
# lines are laid out one per line. Rich only reads [tag] markup starting lowercase/#//@, which
# the JSON brackets below never match.
_MCP_EPILOG = (
    "Without --url or --provider the agent runs natively: its own model, login and config, and "
    "only the MCP flags and --yolo apply; there the mounted servers are added to the agent's own.\n\n"
    f"MCP servers: define them once in {_agent_switch_home() / 'mcp.json'}, in the .mcp.json shape:\n"
    '{"mcpServers": {\n'
    '  "context7": {"command": "npx", "args": ["-y", "@upstash/context7-mcp"]},\n'
    '  "github": {"type": "http", "url": "https://api.githubcopilot.com/mcp/"}\n'
    "}}\n\n"
    "Mount them per session with --mcp NAME (repeatable) or --mcp-all; with --url/--provider the "
    "mounted servers replace the agent's own MCP servers. An http server takes url, headers and "
    "oauth; ${VAR} in args, env, url and header values is expanded from your environment at "
    "launch.\n\n"
    "An http server another app already runs needs no registry entry: --mcp-url [NAME=]URL, "
    "--mcp-oauth-url for one that signs in with oauth, and --mcp-header [NAME:]HEADER=VALUE for its "
    "headers; NAME defaults to the URL's host and port. A command-line server takes --mcp-stdio "
    "[NAME=]COMMAND and --mcp-env [NAME:]KEY=VALUE; NAME defaults to the command's last plain "
    "word. Double-quote COMMAND to run it through bash -ic, which expands ~ and ${VAR} (needs "
    "bash, one command only); unquoted it runs directly, so write ${HOME} not ~.\n\n"
    "Naming, for --mcp-env 'blender:BLENDER_MCP_PORT=9876':\n"
    '  blender           the --mcp-stdio server it applies to (omit "blender:" with only one --mcp-stdio)\n'
    "  BLENDER_MCP_PORT  the environment variable that server sees\n"
    "  9876              its value; '${VAR}' is expanded by agent-switch at launch\n"
    "--mcp-header works the same way for --mcp-url/--mcp-oauth-url servers.\n\n"
    "Examples: agent-switch claude --mcp context7\n"
    "agent-switch codex --mcp-all\n"
    "agent-switch claude --mcp-url tools=http://127.0.0.1:8931/mcp\n"
    "agent-switch codex --mcp-url api=http://127.0.0.1:9000/mcp --mcp-header 'api:Authorization=Bearer ${API_TOKEN}'\n"
    "agent-switch claude --mcp-stdio 'blender=\"uv run --directory ~/workspace/blender-mcp blender-mcp\"' \\\n"
    "  --mcp-env 'blender:BLENDER_MCP_PORT=9876'"
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
