# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See LICENSE
# Ported from unsloth_cli/tests/test_start.py (unsloth commit 8e11ba15e); see NOTICE.md.

"""Shared helpers and constants for the agent-switch CLI tests."""

import json
import os
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import agent_switch.start as start
from agent_switch.core.storage import _agent_switch_home
from tests.start_split import set_start_attr


BASE = "http://127.0.0.1:8888"
MODEL = {"id": "org/gemma-4-26B-A4B-it-GGUF", "context_length": 131072}
KEY = "sk-test-feedfacefeedface"


# --no-launch prints shell setup as POSIX (export/unset) on Unix/WSL and
# PowerShell ($env:/Remove-Item) on native Windows; assert the host's form.
def _assert_env_set(output: str, name: str, value: str) -> None:
    needle = f'$env:{name} = "{value}"' if os.name == "nt" else f"export {name}={value}"
    assert needle in output, f"{needle!r} not found in:\n{output}"


def _assert_env_unset(output: str, name: str) -> None:
    needle = f"Remove-Item Env:{name}" if os.name == "nt" else f"unset {name}"
    assert needle in output, f"{needle!r} not found in:\n{output}"


def _launch_command(output: str) -> list:
    # The --no-launch recipe ends with a self-contained one-liner: inline NAME=value
    # assignments, then the command. Return just the command argv.
    last = [ln for ln in output.splitlines() if ln.strip()][-1]
    parts = shlex.split(last)
    for i, part in enumerate(parts):
        name = part.partition("=")[0]
        if "=" not in part or not name.replace("_", "").isalnum():
            return parts[i:]
    return []


def _path_aware_which(binaries: dict):
    # A shutil.which fake that resolves a name only when its directory is on the PATH passed to
    # which (os.environ's PATH when which gets none). Lets a test prove a version probe searches
    # the augmented PATH before resolving: an agent present only in an install dir (~/.local/bin,
    # %APPDATA%\npm) must still be found and version-checked.
    def _which(name, path = None):
        directory = binaries.get(name)
        if directory is None:
            return None
        entries = (os.environ.get("PATH", "") if path is None else path).split(os.pathsep)
        # os.path.join (not Path()) so this works when a test has flipped os.name to "nt": under
        # a simulated os.name, pathlib would build the non-native flavour and raise.
        return os.path.join(str(directory), name) if str(directory) in entries else None

    return _which


def _simulate_windows(monkeypatch) -> None:
    # Exercise the `os.name == "nt"` branch on any host. Flipping os.name alone makes pathlib
    # pick the non-native flavour (WindowsPath on POSIX, PosixPath on Windows) when a Path is
    # constructed, which raises; pin Path to the host-native class (captured before the flip)
    # so the branch logic runs without that crash. Keeps these tests green on Linux/Mac/WSL too.
    set_start_attr(monkeypatch, "Path", type(Path()))
    monkeypatch.setattr(os, "name", "nt")


def _parse_toml(text: str) -> dict:
    tomllib = pytest.importorskip("tomllib")
    return tomllib.loads(text)


_SESSION_FLAGS = ["--temperature", "0.3", "--top-k", "40", "--reasoning", "off"]


def _opencode_inline_config(output: str) -> dict:
    # --no-launch prints OPENCODE_CONFIG_CONTENT as a POSIX `export NAME=<shell-quoted>`
    # line on Unix/WSL and a PowerShell `$env:NAME = "<escaped>"` line on native Windows;
    # parse whichever the host emitted so the opencode tests are shell-agnostic.
    name = "OPENCODE_CONFIG_CONTENT"
    for raw in output.splitlines():
        line = raw.strip()
        if line.startswith(f"export {name}="):
            return json.loads(shlex.split(line.removeprefix(f"export {name}="))[0])
        prefix = f'$env:{name} = "'
        if line.startswith(prefix) and line.endswith('"'):
            escaped = line[len(prefix) : -1]
            # Reverse _print_env's PowerShell escaping (backtick is the escape char).
            value = escaped.replace("`$", "$").replace('`"', '"').replace("``", "`")
            return json.loads(value)
    raise AssertionError(f"{name} not found in:\n{output}")


def _dsh_entries(path):
    yaml = pytest.importorskip("yaml")
    entries = yaml.safe_load(path.read_text())
    # A loader patch is a top-level list of id-targeted entries, not a settings mapping.
    assert isinstance(entries, list), entries
    return {entry["id"]: entry for entry in entries}


# The temp-dir agents: --persist points each one's home/state env at the stable dir;
# without it, at an ephemeral temp path. opencode is handled separately (only its
# config overlay is relocated; its session data was never in the temp dir).
_RESUME_ENV_VAR = {
    "codex": "CODEX_HOME",
    "pi": "HOME",
    "dsh": "DSH_HOME",
}


def _capture_launch(monkeypatch, argv):
    captured = {}

    def run(
        command,
        env = None,
        **kwargs,
    ):
        captured["command"] = command
        captured["env"] = env
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(subprocess, "run", run)
    result = CliRunner().invoke(start.start_app, argv)
    assert result.exit_code == 0, result.output
    return captured


# One stdio and one http server, the http header carrying a ${VAR} so a test can prove the
# expanded secret reaches only the private session file, never the printed recipe.
_MCP_SERVERS = {
    "context7": {"command": "npx", "args": ["-y", "@upstash/context7-mcp"]},
    "github": {
        "type": "http",
        "url": "https://api.githubcopilot.com/mcp/",
        "headers": {"Authorization": "Bearer ${GITHUB_TOKEN}"},
    },
}


def _mcp_registry(servers: dict = None) -> Path:
    """Write the agent-switch MCP registry under the test AGENT_SWITCH_HOME."""
    path = _agent_switch_home() / "mcp.json"
    path.parent.mkdir(parents = True, exist_ok = True)
    data = _MCP_SERVERS if servers is None else servers
    path.write_text(json.dumps({"mcpServers": data}), encoding = "utf-8")
    return path
