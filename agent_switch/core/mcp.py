# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The agent-switch MCP registry: pick servers for one session, validate and expand them."""

import os
import re
import shlex
import shutil
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

import typer

from agent_switch.core.options import _HEADER_NAME_TOKEN, _HEADER_VALUE_TEXT, _fail
from agent_switch.core.storage import _agent_switch_home, _read_json_object, _write_private_text


# Pinned to the newest mcp-remote published more than two weeks before this pin (0.14.3, 2026-09-21),
# so an untested upstream release cannot break an --oauth mount between user runs.
_MCP_REMOTE_PACKAGE = "mcp-remote@0.14.3"


_MCP_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")


_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


_ENV_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


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


def _normalize_server(name: str, entry: object, expand_args: bool = True) -> dict:
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
        "args": [_expand(arg, name) for arg in args] if expand_args else list(args),
        "env": _expand_map(env, name),
    }


def _parse_name_value(value: str) -> tuple:
    # NAME= only when the text before the first = is a whole server name, so a URL carrying a
    # query string (?key=v) or a command carrying --opt=v is not split.
    name, separator, rest = value.partition("=")
    if separator and _MCP_NAME_RE.fullmatch(name):
        return name, rest
    return None, value


def _auto_mcp_name(url: str) -> str:
    # host-port of the expanded URL (just the host when no port is written), punctuation as -.
    parts = urlsplit(url)
    try:
        port = parts.port
    except ValueError:
        _fail(f"MCP url {url!r} has a port that is not a number.")
    label = parts.hostname or ""
    if port is not None:
        label = f"{label}-{port}"
    name = re.sub(r"[^A-Za-z0-9_-]", "-", label).strip("-")
    if not name:
        _fail(f"MCP url {url!r} has no host to name it from; write NAME={url}.")
    return name


def _split_stdio(command: str) -> list:
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        _fail(f"--mcp-stdio command {command!r} has unbalanced quotes: {exc}.")
    if not argv:
        _fail("--mcp-stdio command is empty; write COMMAND (or NAME=COMMAND).")
    return argv


_SHELL_OPERATORS = frozenset((";", "&&", "||", "|", "&"))


def _shell_stdio_argv(inner: str) -> list:
    """Validate the double-quoted --mcp-stdio command line and build the bash -ic argv."""
    if shutil.which("bash") is None:
        _fail(
            '--mcp-stdio "..." needs bash on PATH; drop the outer double quotes to run the '
            "command directly."
        )
    try:
        lexer = shlex.shlex(inner, posix = True, punctuation_chars = True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError as exc:
        _fail(f'--mcp-stdio command {inner!r} has unbalanced quotes: {exc}.')
    if not tokens:
        _fail('--mcp-stdio command is empty; write "COMMAND" (or NAME="COMMAND").')
    for token in tokens:
        if token in _SHELL_OPERATORS:
            _fail(f'--mcp-stdio "..." runs one command, not {token!r}; put multiple steps in a script.')
    # exec lets the server replace bash: an interactive bash ignores SIGTERM, so without it the
    # server could outlive the agent.
    return ["-ic", f"exec {inner}"]


def _auto_stdio_name(tokens: list) -> str:
    # The last token that is neither an option nor a bare number names the server: its basename,
    # minus a trailing @version and a known script extension, punctuation as -.
    token = ""
    for candidate in reversed(tokens):
        if not candidate.startswith("-") and not candidate.isdigit():
            token = candidate
            break
    label = re.split(r"[/\\]", token)[-1]
    at = label.find("@", 1)  # a leading @ is a package scope, not a version
    if at > 0:
        label = label[:at]
    label = re.sub(r"\.(py|js|mjs|cjs|ts|exe|cmd)$", "", label)
    name = re.sub(r"[^A-Za-z0-9_-]", "-", label).strip("-")
    if not name:
        _fail(f"The --mcp-stdio command {' '.join(tokens)!r} has no word to name it from; write NAME=...")
    return name


def _shortcut_servers(
    urls: Optional[list], oauth_urls: Optional[list], stdios: Optional[list], selected: list
) -> dict:
    """The --mcp-url/--mcp-oauth-url/--mcp-stdio values as registry-shaped entries, unexpanded."""
    shortcuts: dict = {}
    given = [(value, "http", False) for value in urls or []]
    given += [(value, "http", True) for value in oauth_urls or []]
    given += [(value, "stdio", False) for value in stdios or []]
    for value, transport, oauth in given:
        name, target = _parse_name_value(value)
        if transport == "http":
            expanded = _expand(target, name or target)
            if urlsplit(expanded).scheme.lower() not in ("http", "https"):
                _fail(f"MCP url {target!r} must be http or https.")
            if name is None:
                name = _auto_mcp_name(expanded)
            entry = {"type": "http", "url": target, "headers": {}}
            if oauth:
                entry["oauth"] = True
        else:
            if len(target) >= 2 and target.startswith('"') and target.endswith('"'):
                # Shell form: bash expands ~ and ${VAR} at start, so secrets never land in argv.
                inner = target[1:-1]
                argv = _shell_stdio_argv(inner)
                if name is None:
                    name = _auto_stdio_name(shlex.split(inner))
                entry = {"type": "stdio", "command": "bash", "args": argv, "env": {}, "shell": True}
            else:
                argv = _split_stdio(target)
                if name is None:
                    name = _auto_stdio_name([_expand(arg, target) for arg in argv])
                entry = {"type": "stdio", "command": argv[0], "args": argv[1:], "env": {}}
        if name in selected:
            _fail(f"The MCP shortcut {name!r} clashes with the registry server of the same name.")
        if name in shortcuts:
            _fail(f"Two MCP shortcuts are both named {name!r}; give each a distinct NAME= prefix.")
        shortcuts[name] = entry
    return shortcuts


def _shortcut_target(
    value: str, shortcuts: dict, selected: list, wants_http: bool, flag: str, pair: str, noun: str
) -> tuple:
    """Split [NAME:]KEY=VALUE and find the shortcut of the wanted transport that it names."""
    left, separator, key_value = value.partition("=")
    if not separator:
        _fail(f"{flag} needs {pair}, got {value!r}.")
    server, colon, key = left.partition(":")
    if not colon:
        server, key = None, left
    kind = "http" if wants_http else "stdio"
    if server is None:
        same_kind = [n for n, e in shortcuts.items() if (e["type"] == "http") == wants_http]
        if len(same_kind) != 1:
            _fail(f"{flag} without a NAME: prefix needs exactly one {kind} shortcut server.")
        server = same_kind[0]
    if server not in shortcuts:
        if server in selected:
            _fail(
                f"{flag} cannot add {noun} to registry server {server!r}: "
                f"keep its {noun} in the registry."
            )
        _fail(f"{flag} names unknown server {server!r}. Shortcut servers: {', '.join(shortcuts) or '(none)'}")
    if (shortcuts[server]["type"] == "http") != wants_http:
        _fail(f"{flag} only adds {noun} to {kind} shortcut servers; {server!r} is not one.")
    return server, key, key_value


def _shortcut_headers(headers: Optional[list], shortcuts: dict, selected: list) -> None:
    """Attach the --mcp-header values to the http shortcut entries they name."""
    for value in headers or []:
        server, header_name, header_value = _shortcut_target(
            value, shortcuts, selected, True, "--mcp-header", "[NAME:]HEADER=VALUE", "headers"
        )
        if not _HEADER_NAME_TOKEN.fullmatch(header_name):
            _fail(f"--mcp-header name {header_name!r} has characters an HTTP header name cannot carry.")
        if not _HEADER_VALUE_TEXT.fullmatch(header_value):
            _fail(
                f"--mcp-header value for {header_name!r} has characters an HTTP header value cannot carry."
            )
        if header_name.lower() == "authorization" and "${" not in header_value:
            # agent-switch stays running until the agent exits, so a literal value sits in the
            # process list for the whole session.
            typer.echo(
                f"Warning: --mcp-header {header_name}=... is a literal value, visible in the "
                "process list while the session runs; pass it as '${VAR}' instead.",
                err = True,
            )
        merged = {
            name: v for name, v in shortcuts[server]["headers"].items() if name.lower() != header_name.lower()
        }
        merged[header_name] = header_value
        shortcuts[server]["headers"] = merged


def _shortcut_envs(envs: Optional[list], shortcuts: dict, selected: list) -> None:
    """Attach the --mcp-env values to the stdio shortcut entries they name."""
    for value in envs or []:
        server, key, env_value = _shortcut_target(
            value, shortcuts, selected, False, "--mcp-env", "[NAME:]KEY=VALUE", "env"
        )
        if not _ENV_KEY_RE.fullmatch(key):
            _fail(f"--mcp-env key {key!r} must match [A-Za-z_][A-Za-z0-9_]*.")
        shortcuts[server]["env"][key] = env_value


def load_mcp_servers(
    names: Optional[list],
    *,
    should_mount_all: bool,
    urls: Optional[list] = None,
    oauth_urls: Optional[list] = None,
    headers: Optional[list] = None,
    stdios: Optional[list] = None,
    envs: Optional[list] = None,
    as_subagent: bool = False,
) -> Optional[dict]:
    """The selected servers as {name: normalized}, or None when no MCP flag was given.

    Normalized is {"transport": "stdio", "command", "args", "env"} or
    {"transport": "http", "url", "headers"}, with ${VAR} already expanded. --mcp-url and
    --mcp-oauth-url add http servers without a registry entry (--mcp-header adds headers to them);
    --mcp-stdio adds stdio servers (--mcp-env sets their environment). A double-quoted
    --mcp-stdio command runs through bash -ic instead, and bash does the expanding.
    """
    if (
        not names
        and not should_mount_all
        and not urls
        and not oauth_urls
        and not headers
        and not stdios
        and not envs
    ):
        return None
    if as_subagent:
        # Codex's subagent mode symlinks the parent CODEX_HOME into the real ~/.codex, so writing
        # MCP servers there would modify the user's own config; every agent refuses the same way.
        _fail(
            "--mcp/--mcp-all/--mcp-url/--mcp-oauth-url/--mcp-header/--mcp-stdio/--mcp-env cannot "
            "be combined with --as-subagent."
        )
    if names and should_mount_all:
        _fail("--mcp and --mcp-all cannot be combined: name the servers, or mount them all.")
    servers: dict = {}
    selected: list = []
    if names or should_mount_all:
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
    # Shortcut-only runs never read the registry file, so it need not exist then.
    shortcuts = _shortcut_servers(urls, oauth_urls, stdios, selected)
    _shortcut_headers(headers, shortcuts, selected)
    _shortcut_envs(envs, shortcuts, selected)
    mounted = {name: _normalize_server(name, servers[name]) for name in selected}
    for name, entry in shortcuts.items():
        # The "shell" flag is set only by _shortcut_servers, so a registry entry cannot reach it.
        mounted[name] = _normalize_server(name, entry, expand_args = "shell" not in entry)
    return mounted
