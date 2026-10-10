# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The agent-switch MCP registry: selection, validation, ${VAR} expansion and oauth wrapping."""

import json
import os
import re
import shutil

import pytest
import typer

from agent_switch.core import mcp as mcp_core


def _write_registry(servers) -> None:
    path = mcp_core._mcp_registry_path()
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(json.dumps({"mcpServers": servers}), encoding = "utf-8")


def _expect_fail(call, capsys) -> str:
    with pytest.raises(typer.Exit) as caught:
        call()
    assert caught.value.exit_code == 1
    return capsys.readouterr().err


def test_no_mcp_flags_mount_nothing():
    assert mcp_core.load_mcp_servers(None, should_mount_all = False) is None
    assert mcp_core.load_mcp_servers([], should_mount_all = False) is None


def test_load_picks_by_name_and_normalizes():
    _write_registry(
        {
            "context7": {"command": "npx", "args": ["-y", "@upstash/context7-mcp"]},
            "github": {"type": "http", "url": "https://api.githubcopilot.com/mcp/"},
        }
    )
    assert mcp_core.load_mcp_servers(["github"], should_mount_all = False) == {
        "github": {"transport": "http", "url": "https://api.githubcopilot.com/mcp/", "headers": {}}
    }
    assert mcp_core.load_mcp_servers(["context7"], should_mount_all = False) == {
        "context7": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@upstash/context7-mcp"],
            "env": {},
        }
    }


def test_unknown_name_lists_available(capsys):
    _write_registry({"context7": {"command": "npx"}, "github": {"type": "http", "url": "x"}})
    error = _expect_fail(
        lambda: mcp_core.load_mcp_servers(["nope"], should_mount_all = False), capsys
    )
    assert "nope" in error
    assert "Available: context7, github" in error


def test_mcp_all_mounts_every_server_in_registry_order():
    _write_registry({"context7": {"command": "npx"}, "github": {"type": "http", "url": "x"}})
    servers = mcp_core.load_mcp_servers(None, should_mount_all = True)
    assert list(servers) == ["context7", "github"]


def test_mcp_all_with_names_fails(capsys):
    _write_registry({"context7": {"command": "npx"}})
    error = _expect_fail(
        lambda: mcp_core.load_mcp_servers(["context7"], should_mount_all = True), capsys
    )
    assert "--mcp and --mcp-all cannot be combined" in error


def test_mcp_all_with_empty_registry_fails(capsys):
    _write_registry({})
    error = _expect_fail(lambda: mcp_core.load_mcp_servers(None, should_mount_all = True), capsys)
    assert "holds no servers for --mcp-all" in error
    assert str(mcp_core._mcp_registry_path()) in error


def test_missing_registry_fails_with_its_path_and_an_example(capsys):
    path = mcp_core._mcp_registry_path()
    assert not path.exists()
    error = _expect_fail(lambda: mcp_core.load_mcp_servers(["x"], should_mount_all = False), capsys)
    assert str(path) in error
    assert '"mcpServers"' in error


def test_var_expansion_in_args_env_url_and_headers(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "gh-secret")
    monkeypatch.setenv("PROFILE", "work")
    _write_registry(
        {
            "srv": {
                "command": "npx",
                "args": ["-y", "--profile", "${PROFILE}", "@upstash/context7-mcp"],
                "env": {"TOKEN": "Bearer ${GITHUB_TOKEN}", "PLAIN": "plain"},
            },
            "http": {
                "type": "http",
                "url": "https://api.example.com/${PROFILE}/mcp",
                "headers": {"Authorization": "Bearer ${GITHUB_TOKEN}"},
            },
        }
    )
    servers = mcp_core.load_mcp_servers(["srv", "http"], should_mount_all = False)
    assert servers["srv"]["args"] == ["-y", "--profile", "work", "@upstash/context7-mcp"]
    assert servers["srv"]["env"] == {"TOKEN": "Bearer gh-secret", "PLAIN": "plain"}
    assert servers["http"]["url"] == "https://api.example.com/work/mcp"
    assert servers["http"]["headers"] == {"Authorization": "Bearer gh-secret"}


def test_unset_var_fails_naming_the_var_and_the_server(capsys, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising = False)
    _write_registry(
        {"github": {"type": "http", "url": "https://x/mcp", "headers": {"A": "Bearer ${GITHUB_TOKEN}"}}}
    )
    error = _expect_fail(lambda: mcp_core.load_mcp_servers(["github"], should_mount_all = False), capsys)
    assert "GITHUB_TOKEN" in error
    assert "github" in error


def test_bad_server_name_rejected(capsys):
    _write_registry({"bad name": {"command": "npx"}})
    error = _expect_fail(lambda: mcp_core.load_mcp_servers(None, should_mount_all = True), capsys)
    assert "bad name" in error


def test_sse_type_rejected(capsys):
    _write_registry({"old": {"type": "sse", "url": "https://x/sse"}})
    error = _expect_fail(lambda: mcp_core.load_mcp_servers(["old"], should_mount_all = False), capsys)
    assert "sse" in error
    assert "only stdio" in error


def test_stdio_needs_a_command(capsys):
    _write_registry({"broken": {"args": ["x"]}})
    error = _expect_fail(lambda: mcp_core.load_mcp_servers(["broken"], should_mount_all = False), capsys)
    assert "command" in error


def test_oauth_becomes_pinned_mcp_remote_stdio(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name, path = None: "/usr/bin/npx")
    _write_registry(
        {
            "atlassian": {
                "type": "http",
                "url": "https://mcp.atlassian.com/v1/mcp",
                "oauth": True,
                "headers": {"X-Trace": "on"},
            }
        }
    )
    servers = mcp_core.load_mcp_servers(["atlassian"], should_mount_all = False)
    assert re.fullmatch(r"mcp-remote@\d+\.\d+\.\d+", mcp_core._MCP_REMOTE_PACKAGE)
    auth_dir = mcp_core._agent_switch_home() / "mcp-auth"
    headers_path = auth_dir / "atlassian.headers"
    assert servers["atlassian"] == {
        "transport": "stdio",
        "command": "npx",
        "args": [
            "-y",
            mcp_core._MCP_REMOTE_PACKAGE,
            "https://mcp.atlassian.com/v1/mcp",
            "--header-file",
            str(headers_path),
        ],
        "env": {"MCP_REMOTE_CONFIG_DIR": str(auth_dir)},
    }
    assert headers_path.read_text() == "X-Trace: on\n"
    assert auth_dir.is_dir()
    if os.name != "nt":
        assert auth_dir.stat().st_mode & 0o777 == 0o700
        assert headers_path.stat().st_mode & 0o777 == 0o600


def test_oauth_header_values_stay_out_of_argv(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name, path = None: "/usr/bin/npx")
    monkeypatch.setenv("ATL_TOKEN", "s3cret")
    _write_registry(
        {
            "atlassian": {
                "type": "http",
                "url": "https://mcp.atlassian.com/v1/mcp",
                "oauth": True,
                "headers": {"Authorization": "Bearer ${ATL_TOKEN}"},
            }
        }
    )
    servers = mcp_core.load_mcp_servers(["atlassian"], should_mount_all = False)
    assert "s3cret" not in " ".join(servers["atlassian"]["args"])
    headers_path = mcp_core._agent_switch_home() / "mcp-auth" / "atlassian.headers"
    assert headers_path.read_text() == "Authorization: Bearer s3cret\n"
    if os.name != "nt":
        assert headers_path.stat().st_mode & 0o777 == 0o600


def test_oauth_without_headers_removes_stale_header_file(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name, path = None: "/usr/bin/npx")
    auth_dir = mcp_core._agent_switch_home() / "mcp-auth"
    auth_dir.mkdir(parents = True, exist_ok = True, mode = 0o700)
    stale = auth_dir / "atlassian.headers"
    stale.write_text("X-Old: gone\n")
    _write_registry({"atlassian": {"type": "http", "url": "https://x/mcp", "oauth": True}})
    servers = mcp_core.load_mcp_servers(["atlassian"], should_mount_all = False)
    assert not stale.exists()
    assert "--header-file" not in servers["atlassian"]["args"]


def test_newline_in_http_header_value_fails(capsys):
    _write_registry({"http": {"type": "http", "url": "x", "headers": {"A": "one\ntwo"}}})
    error = _expect_fail(lambda: mcp_core.load_mcp_servers(["http"], should_mount_all = False), capsys)
    assert "newline" in error
    assert "http" in error


def test_newline_in_oauth_header_value_fails(monkeypatch, capsys):
    monkeypatch.setattr(shutil, "which", lambda name, path = None: "/usr/bin/npx")
    _write_registry(
        {"atlassian": {"type": "http", "url": "x", "oauth": True, "headers": {"A": "one\r\ntwo"}}}
    )
    error = _expect_fail(
        lambda: mcp_core.load_mcp_servers(["atlassian"], should_mount_all = False), capsys
    )
    assert "newline" in error
    assert "atlassian" in error


def test_oauth_without_npx_on_path_fails(monkeypatch, capsys):
    monkeypatch.setattr(shutil, "which", lambda name, path = None: None)
    _write_registry({"atlassian": {"type": "http", "url": "https://x/mcp", "oauth": True}})
    error = _expect_fail(lambda: mcp_core.load_mcp_servers(["atlassian"], should_mount_all = False), capsys)
    assert "npx is not on PATH" in error
    assert mcp_core._MCP_REMOTE_PACKAGE in error
