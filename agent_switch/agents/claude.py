# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`agent-switch claude`: Claude Code settings overlay, flags and subagent plugin."""

import base64
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Literal, Optional

import typer

from agent_switch.core.install import (
    _probe_env,
    _require_agent_for_launch,
    _which_with_install_dirs,
)
from agent_switch.core.launch import _connect, _resolve_target, _run
from agent_switch.core.mcp import load_mcp_servers
from agent_switch.core.options import (
    LoadOptions,
    ProviderName,
    ServerOptions,
    _ALL_REQUEST_FIELDS,
    _AS_SUBAGENT_OPTION,
    _COMPACT_AT_OPTION,
    _CONTEXT_OPTION,
    _HEADER_OPTION,
    _KEY_OPTION,
    _LAUNCH_OPTION,
    _MCP_ALL_OPTION,
    _MCP_OPTION,
    _MIN_P_OPTION,
    _MODEL_LOAD_OPTION,
    _MODEL_OPTION,
    _PERSIST_OPTION,
    _PRESENCE_PENALTY_OPTION,
    _PROVIDER_OPTION,
    _REASONING_EFFORT_OPTION,
    _REASONING_OPTION,
    _REPETITION_PENALTY_OPTION,
    _SUBAGENT_DESCRIPTION,
    _TEMPERATURE_OPTION,
    _TOP_K_OPTION,
    _TOP_P_OPTION,
    _URL_OPTION,
    _YOLO_OPTION,
    _check_compact_at,
    _consume_positional_model,
    _fail,
    _yolo_command_flags,
    parse_headers,
)
from agent_switch.core.platform import _merge_wslenv, _wsl_bridge_names, _wsl_windows_executable
from agent_switch.core.session import _agent_config_path, _session_config
from agent_switch.core.storage import _write_private_json, _write_private_text
from agent_switch.providers.utils import get_has_custom_authorization


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


def _write_claude_mcp_config(path: Path, servers: dict) -> None:
    """The session mcp.json for --mcp/--mcp-all, in the common .mcp.json shape."""
    _write_private_json(
        path,
        {
            "mcpServers": {
                name: (
                    {
                        "type": "stdio",
                        "command": server["command"],
                        "args": server["args"],
                        "env": server["env"],
                    }
                    if server["transport"] == "stdio"
                    else {"type": "http", "url": server["url"], "headers": server["headers"]}
                )
                for name, server in servers.items()
            }
        },
    )


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


def _claude_local_command(
    model_id: str,
    settings: str,
    yolo: bool,
    passthrough: list,
    mcp_flags: Optional[list] = None,
) -> list:
    local_args = [
        "--model",
        model_id,
        *_claude_flags(model_id, settings),
        *_yolo_command_flags("claude", yolo),
        # The `=` form: --mcp-config is variadic in claude, so a detached value would swallow the
        # first forwarded positional, same reason as --allowedTools in the subagent command.
        *(mcp_flags or []),
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
    mcp: Optional[list[str]] = _MCP_OPTION,
    mcp_all: bool = _MCP_ALL_OPTION,
):
    """Point Claude Code at a local model server and start it."""
    # Route a leading `org/name` positional to --model; forward the rest to the agent.
    model, ctx.args[:] = _consume_positional_model(model, ctx.args)
    headers = parse_headers(header)
    if as_subagent and (mcp or mcp_all):
        _fail("--mcp/--mcp-all cannot be combined with --as-subagent.")
    # Validate the MCP selection before _connect, so a registry error fails fast.
    mcp_servers = load_mcp_servers(mcp, should_mount_all = mcp_all)
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
        mcp_flags = []
        mcp_config = config / "mcp.json"
        if mcp_servers:
            _write_claude_mcp_config(mcp_config, mcp_servers)
            # --strict-mcp-config is what "replace" means here: claude ignores its user- and
            # project-level MCP servers and reads only this file. The flags go only with mounted
            # servers, so a persisted session without MCP flags keeps claude's own behavior.
            mcp_path = _agent_config_path(mcp_config, ["claude"])
            mcp_flags = ["--strict-mcp-config", f"--mcp-config={mcp_path}"]
        else:
            # As in pi(): a persisted session dir must not keep an earlier mount's expanded secrets.
            mcp_config.unlink(missing_ok = True)
        command = _claude_local_command(
            model_id,
            _agent_config_path(settings, ["claude"]),
            yolo,
            ctx.args,
            mcp_flags,
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
