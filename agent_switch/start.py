# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`agent-switch <agent>` — launch a coding agent against a running model server."""

import typer

from agent_switch.agents import claude, codex, dsh, opencode, pi
from agent_switch.core.options import _PASSTHROUGH, _PassthroughCommand

start_app = typer.Typer(
    help = "Start a coding agent against a local model server: Ollama, LM Studio, "
    "llama-server, vLLM or any OpenAI-compatible server.",
    no_args_is_help = True,
    context_settings = {"help_option_names": ["-h", "--help"]},
)

# Registration order is the --help order; the old dsh decorators applied bottom-up (dsh, dsh-tui, dst).
start_app.command("claude", cls = _PassthroughCommand, context_settings = _PASSTHROUGH)(claude.claude)
start_app.command("codex", cls = _PassthroughCommand, context_settings = _PASSTHROUGH)(codex.codex)
start_app.command("opencode", cls = _PassthroughCommand, context_settings = _PASSTHROUGH)(opencode.opencode)
start_app.command("pi", cls = _PassthroughCommand, context_settings = _PASSTHROUGH)(pi.pi)
start_app.command("dsh", cls = _PassthroughCommand, context_settings = _PASSTHROUGH)(dsh.dsh)
start_app.command(
    "dsh-tui",
    cls = _PassthroughCommand,
    context_settings = _PASSTHROUGH,
    help = "Point the DeepSeek Harness TUI (dsh-tui) at a local model server and start it.",
)(dsh.dsh)
start_app.command(
    "dst", cls = _PassthroughCommand, context_settings = _PASSTHROUGH, help = "Alias of dsh-tui."
)(dsh.dsh)
