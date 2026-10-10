# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The agent-switch MCP registry: pick servers for one session, validate and expand them."""

import os
import re
import shutil
from pathlib import Path
from typing import Optional

from agent_switch.core.options import _fail
from agent_switch.core.storage import _agent_switch_home, _read_json_object, _write_private_text


# Pinned to the newest mcp-remote published more than two weeks before this pin (0.14.3, 2026-09-21),
# so an untested upstream release cannot break an --oauth mount between user runs.
_MCP_REMOTE_PACKAGE = "mcp-remote@0.14.3"


_MCP_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")


_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _mcp_registry_path() -> Path:
    return _agent_switch_home() / "mcp.json"


def _expand(value: str, server: str) -> str:
    # Expanded values reach only the private (0600) files written from this result — the session
    # configs and the oauth header file — never argv; only the oauth url is an mcp-remote argument.
    def replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in os.environ:
            _fail(f"MCP server {server!r} uses ${{{name}}}, but {name} is not set.")
        return os.environ[name]

    return _VAR_RE.sub(replace, value)


def _expand_map(values: dict, server: str) -> dict:
    return {name: _expand(value, server) for name, value in values.items()}


def _expand_headers(headers: dict, server: str) -> dict:
    expanded = _expand_map(headers, server)
    for header, value in expanded.items():
        if "\r" in value or "\n" in value:
            _fail(
                f"MCP server {server!r} header {header!r} must not contain a carriage return or newline."
            )
    return expanded


def _string_map(values: object, server: str, key: str) -> dict:
    if not isinstance(values, dict) or not all(
        isinstance(name, str) and isinstance(value, str) for name, value in values.items()
    ):
        _fail(f"MCP server {server!r} {key} must be an object of name/value strings.")
    return values


def _oauth_stdio(name: str, url: str, headers: dict) -> dict:
    """Turn an oauth http server into the stdio mcp-remote bridge that signs in for it."""
    if shutil.which("npx") is None:
        _fail(
            f"MCP server {name!r} uses oauth, which runs {_MCP_REMOTE_PACKAGE} through npx, "
            "but npx is not on PATH."
        )
    auth_dir = _agent_switch_home() / "mcp-auth"
    auth_dir.mkdir(parents = True, exist_ok = True, mode = 0o700)
    args = ["-y", _MCP_REMOTE_PACKAGE, _expand(url, name)]
    expanded = _expand_headers(headers, name)
    headers_path = auth_dir / f"{name}.headers"
    if expanded:
        # mcp-remote sends these before the sign-in, so a header-authed server works too.
        # --header-file (one "Name: value" per line) keeps the values out of the process arguments.
        _write_private_text(
            headers_path, "".join(f"{header}: {value}\n" for header, value in expanded.items())
        )
        args += ["--header-file", str(headers_path)]
    else:
        # A mount that dropped its headers must not leave the old ones for mcp-remote to send.
        headers_path.unlink(missing_ok = True)
    return {
        "transport": "stdio",
        "command": "npx",
        "args": args,
        # The tokens mcp-remote caches stay inside the agent-switch home, not ~/.mcp-auth.
        "env": {"MCP_REMOTE_CONFIG_DIR": str(auth_dir)},
    }


def _normalize_server(name: str, entry: object) -> dict:
    if not isinstance(entry, dict):
        _fail(f"MCP server {name!r} must be an object.")
    transport = entry.get("type", "stdio")
    if transport == "http":
        url = entry.get("url")
        if not isinstance(url, str) or not url:
            _fail(f"MCP server {name!r} is type http but has no url.")
        headers = _string_map(entry.get("headers") or {}, name, "headers")
        if entry.get("oauth"):
            return _oauth_stdio(name, url, headers)
        return {"transport": "http", "url": _expand(url, name), "headers": _expand_headers(headers, name)}
    if transport != "stdio":
        _fail(
            f"MCP server {name!r} has type {transport!r}; only stdio (command) and http (url) "
            "are supported."
        )
    command = entry.get("command")
    if not isinstance(command, str) or not command:
        _fail(f"MCP server {name!r} needs a command.")
    args = entry.get("args") or []
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        _fail(f"MCP server {name!r} args must be a list of strings.")
    env = _string_map(entry.get("env") or {}, name, "env")
    return {
        "transport": "stdio",
        "command": command,
        "args": [_expand(arg, name) for arg in args],
        "env": _expand_map(env, name),
    }


def load_mcp_servers(names: Optional[list], *, should_mount_all: bool) -> Optional[dict]:
    """The selected servers as {name: normalized}, or None when no MCP flag was given.

    Normalized is {"transport": "stdio", "command", "args", "env"} or
    {"transport": "http", "url", "headers"}, with ${VAR} already expanded.
    """
    if not names and not should_mount_all:
        return None
    if names and should_mount_all:
        _fail("--mcp and --mcp-all cannot be combined: name the servers, or mount them all.")
    path = _mcp_registry_path()
    if not path.exists():
        _fail(
            f"No MCP registry at {path}. Create it as, e.g.:\n"
            '  {"mcpServers": {"context7": {"command": "npx", '
            '"args": ["-y", "@upstash/context7-mcp"]}}}'
        )
    registry = _read_json_object(path)
    if registry is None:
        _fail(f"Could not parse the MCP registry {path} as a JSON object.")
    servers = registry.get("mcpServers") or {}
    if not isinstance(servers, dict):
        _fail(f"The mcpServers key in {path} must be an object.")
    for name in servers:
        if not isinstance(name, str) or not _MCP_NAME_RE.fullmatch(name):
            _fail(f"MCP server name {name!r} in {path} must use only letters, digits, _ and -.")
    if should_mount_all:
        if not servers:
            _fail(f"The MCP registry {path} holds no servers for --mcp-all.")
        selected = list(servers)
    else:
        missing = [name for name in names if name not in servers]
        if missing:
            _fail(
                f"MCP server(s) {', '.join(missing)} not in {path}. "
                f"Available: {', '.join(servers) or '(none)'}"
            )
        selected = list(dict.fromkeys(names))
    return {name: _normalize_server(name, servers[name]) for name in selected}
