# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See LICENSE

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


# ── --mcp-url / --mcp-oauth-url / --mcp-header shortcuts ─────────────────────


def test_shortcut_auto_names_from_host_and_port():
    servers = mcp_core.load_mcp_servers(
        None,
        should_mount_all = False,
        urls = ["http://127.0.0.1:8931/mcp", "http://localhost:8931/mcp", "https://mcp.example.com/mcp"],
    )
    assert list(servers) == ["127-0-0-1-8931", "localhost-8931", "mcp-example-com"]


def test_auto_name_uses_the_expanded_url(monkeypatch):
    monkeypatch.setenv("MCP_HOST", "localhost")
    servers = mcp_core.load_mcp_servers(
        None, should_mount_all = False, urls = ["http://${MCP_HOST}:8931/mcp"]
    )
    assert list(servers) == ["localhost-8931"]


def test_name_prefix_only_splits_a_whole_name():
    servers = mcp_core.load_mcp_servers(
        None,
        should_mount_all = False,
        urls = ["tools=http://127.0.0.1:8931/mcp", "http://127.0.0.1:9000/mcp?key=v"],
    )
    assert servers["tools"]["url"] == "http://127.0.0.1:8931/mcp"
    assert servers["127-0-0-1-9000"]["url"] == "http://127.0.0.1:9000/mcp?key=v"


def test_duplicate_shortcut_names_fail(capsys):
    error = _expect_fail(
        lambda: mcp_core.load_mcp_servers(
            None, should_mount_all = False, urls = ["http://x:1/mcp", "http://x:1/mcp"]
        ),
        capsys,
    )
    assert "distinct NAME=" in error


def test_shortcut_name_clash_with_registry_fails(capsys):
    _write_registry({"github": {"type": "http", "url": "https://api.githubcopilot.com/mcp/"}})
    error = _expect_fail(
        lambda: mcp_core.load_mcp_servers(
            ["github"], should_mount_all = False, urls = ["github=http://127.0.0.1:9/mcp"]
        ),
        capsys,
    )
    assert "clashes with the registry server" in error


def test_shortcut_non_http_scheme_fails(capsys):
    error = _expect_fail(
        lambda: mcp_core.load_mcp_servers(
            None, should_mount_all = False, urls = ["ftp://127.0.0.1:21/mcp"]
        ),
        capsys,
    )
    assert "must be http or https" in error


def test_shortcut_var_expansion_in_url_and_header(monkeypatch):
    monkeypatch.setenv("MCP_PORT", "8931")
    monkeypatch.setenv("PROBE", "abc")
    servers = mcp_core.load_mcp_servers(
        None,
        should_mount_all = False,
        urls = ["ev=http://127.0.0.1:${MCP_PORT}/mcp"],
        headers = ["ev:X-Probe=${PROBE}"],
    )
    assert servers["ev"]["url"] == "http://127.0.0.1:8931/mcp"
    assert servers["ev"]["headers"] == {"X-Probe": "abc"}


def test_shortcuts_combine_with_registry_selection():
    _write_registry({"github": {"type": "http", "url": "https://api.githubcopilot.com/mcp/"}})
    servers = mcp_core.load_mcp_servers(
        ["github"], should_mount_all = False, urls = ["tools=http://127.0.0.1:9/mcp"]
    )
    assert list(servers) == ["github", "tools"]


def test_shortcut_only_needs_no_registry():
    assert not mcp_core._mcp_registry_path().exists()
    servers = mcp_core.load_mcp_servers(
        None, should_mount_all = False, urls = ["http://127.0.0.1:9/mcp"]
    )
    assert servers == {
        "127-0-0-1-9": {"transport": "http", "url": "http://127.0.0.1:9/mcp", "headers": {}}
    }


def test_header_without_name_ok_with_one_server():
    servers = mcp_core.load_mcp_servers(
        None,
        should_mount_all = False,
        urls = ["tools=http://127.0.0.1:9/mcp"],
        headers = ["X-Key=v"],
    )
    assert servers["tools"]["headers"] == {"X-Key": "v"}


def test_header_without_name_fails_with_two_servers(capsys):
    error = _expect_fail(
        lambda: mcp_core.load_mcp_servers(
            None,
            should_mount_all = False,
            urls = ["a=http://127.0.0.1:1/mcp", "b=http://127.0.0.1:2/mcp"],
            headers = ["X-Key=v"],
        ),
        capsys,
    )
    assert "needs exactly one" in error


def test_header_for_unknown_server_fails(capsys):
    error = _expect_fail(
        lambda: mcp_core.load_mcp_servers(
            None,
            should_mount_all = False,
            urls = ["a=http://127.0.0.1:1/mcp"],
            headers = ["nope:X-Key=v"],
        ),
        capsys,
    )
    assert "unknown server 'nope'" in error


def test_header_for_registry_server_fails(capsys):
    _write_registry({"github": {"type": "http", "url": "https://api.githubcopilot.com/mcp/"}})
    error = _expect_fail(
        lambda: mcp_core.load_mcp_servers(
            ["github"],
            should_mount_all = False,
            urls = ["a=http://127.0.0.1:1/mcp"],
            headers = ["github:X-Key=v"],
        ),
        capsys,
    )
    assert "keep its headers in the registry" in error


def test_header_name_and_value_validated(capsys):
    error = _expect_fail(
        lambda: mcp_core.load_mcp_servers(
            None,
            should_mount_all = False,
            urls = ["a=http://127.0.0.1:1/mcp"],
            headers = ["Bad Name=v"],
        ),
        capsys,
    )
    assert "header name cannot carry" in error


def test_later_duplicate_header_wins():
    servers = mcp_core.load_mcp_servers(
        None,
        should_mount_all = False,
        urls = ["a=http://127.0.0.1:1/mcp"],
        headers = ["X-Key=first", "X-Key=second"],
    )
    assert servers["a"]["headers"] == {"X-Key": "second"}


def test_literal_authorization_warns(capsys):
    servers = mcp_core.load_mcp_servers(
        None,
        should_mount_all = False,
        urls = ["a=http://127.0.0.1:1/mcp"],
        headers = ["Authorization=Bearer literal"],
    )
    assert servers["a"]["headers"] == {"Authorization": "Bearer literal"}
    err = capsys.readouterr().err
    assert "Warning" in err
    assert "'${VAR}'" in err


def test_authorization_with_var_does_not_warn(capsys, monkeypatch):
    monkeypatch.setenv("API_TOKEN", "t")
    servers = mcp_core.load_mcp_servers(
        None,
        should_mount_all = False,
        urls = ["a=http://127.0.0.1:1/mcp"],
        headers = ["authorization=Bearer ${API_TOKEN}"],
    )
    assert servers["a"]["headers"] == {"authorization": "Bearer t"}
    assert "Warning" not in capsys.readouterr().err


def test_oauth_url_shortcut_wraps_like_the_registry(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name, path = None: "/usr/bin/npx")
    servers = mcp_core.load_mcp_servers(
        None,
        should_mount_all = False,
        oauth_urls = ["atlassian=https://mcp.atlassian.com/v1/mcp"],
        headers = ["atlassian:X-Trace=on"],
    )
    headers_path = mcp_core._agent_switch_home() / "mcp-auth" / "atlassian.headers"
    assert servers["atlassian"]["args"] == [
        "-y",
        mcp_core._MCP_REMOTE_PACKAGE,
        "https://mcp.atlassian.com/v1/mcp",
        "--header-file",
        str(headers_path),
    ]
    assert headers_path.read_text() == "X-Trace: on\n"
    if os.name != "nt":
        assert headers_path.stat().st_mode & 0o777 == 0o600


def test_as_subagent_refuses_every_mcp_flag(capsys):
    calls = [
        lambda: mcp_core.load_mcp_servers(["x"], should_mount_all = False, as_subagent = True),
        lambda: mcp_core.load_mcp_servers(None, should_mount_all = True, as_subagent = True),
        lambda: mcp_core.load_mcp_servers(
            None, should_mount_all = False, urls = ["http://x/mcp"], as_subagent = True
        ),
        lambda: mcp_core.load_mcp_servers(
            None, should_mount_all = False, oauth_urls = ["http://x/mcp"], as_subagent = True
        ),
        lambda: mcp_core.load_mcp_servers(
            None, should_mount_all = False, headers = ["X=v"], as_subagent = True
        ),
        lambda: mcp_core.load_mcp_servers(
            None, should_mount_all = False, stdios = ["npx x"], as_subagent = True
        ),
        lambda: mcp_core.load_mcp_servers(
            None, should_mount_all = False, envs = ["X=v"], as_subagent = True
        ),
    ]
    for call in calls:
        error = _expect_fail(call, capsys)
        assert "cannot be combined with --as-subagent" in error
        assert "--mcp-stdio" in error
        assert "--mcp-env" in error


# ── --mcp-stdio / --mcp-env shortcuts ────────────────────────────────────────


def test_stdio_shlex_split_keeps_quoted_args():
    servers = mcp_core.load_mcp_servers(
        None, should_mount_all = False, stdios = ["srv=sh -c \"echo hi there\""]
    )
    assert servers["srv"] == {
        "transport": "stdio",
        "command": "sh",
        "args": ["-c", "echo hi there"],
        "env": {},
    }


def test_stdio_unbalanced_quotes_fail(capsys):
    error = _expect_fail(
        lambda: mcp_core.load_mcp_servers(
            None, should_mount_all = False, stdios = ["srv=sh -c \"oops"]
        ),
        capsys,
    )
    assert "unbalanced quotes" in error


def test_stdio_empty_command_fails(capsys):
    error = _expect_fail(
        lambda: mcp_core.load_mcp_servers(None, should_mount_all = False, stdios = ["srv="]), capsys
    )
    assert "empty" in error


def test_stdio_auto_names():
    servers = mcp_core.load_mcp_servers(
        None,
        should_mount_all = False,
        stdios = [
            "uv run --directory /opt/x blender-mcp",
            "npx -y @modelcontextprotocol/server-everything@2026.8.31",
            "node /srv/server.js --port 3000",
            "python -m foo.server",
            "docker run -i --rm mcp/everything",
        ],
    )
    assert list(servers) == ["blender-mcp", "server-everything", "server", "foo-server", "everything"]


def test_stdio_auto_name_empty_result_fails(capsys):
    error = _expect_fail(
        lambda: mcp_core.load_mcp_servers(None, should_mount_all = False, stdios = ["--optx 42"]),
        capsys,
    )
    assert "NAME=" in error


def test_stdio_name_prefix_vs_option_value():
    servers = mcp_core.load_mcp_servers(
        None, should_mount_all = False, stdios = ["npx pkg --opt=v"]
    )
    assert servers["pkg"]["command"] == "npx"
    assert servers["pkg"]["args"] == ["pkg", "--opt=v"]


def test_stdio_var_expansion_in_name_and_args(monkeypatch):
    monkeypatch.setenv("MCP_DIR", "/opt/x")
    servers = mcp_core.load_mcp_servers(
        None, should_mount_all = False, stdios = ["uv run --directory ${MCP_DIR} blender-mcp"]
    )
    assert list(servers) == ["blender-mcp"]
    assert servers["blender-mcp"]["args"] == ["run", "--directory", "/opt/x", "blender-mcp"]


def test_env_without_name_ok_with_one_stdio_server():
    servers = mcp_core.load_mcp_servers(
        None, should_mount_all = False, stdios = ["srv=npx x"], envs = ["API_KEY=v"]
    )
    assert servers["srv"]["env"] == {"API_KEY": "v"}


def test_env_naming_http_shortcut_fails(capsys):
    error = _expect_fail(
        lambda: mcp_core.load_mcp_servers(
            None,
            should_mount_all = False,
            urls = ["api=http://127.0.0.1:1/mcp"],
            stdios = ["srv=npx x"],
            envs = ["api:KEY=v"],
        ),
        capsys,
    )
    assert "only adds env to stdio shortcut servers" in error


def test_env_naming_registry_server_fails(capsys):
    _write_registry({"github": {"type": "http", "url": "https://api.githubcopilot.com/mcp/"}})
    error = _expect_fail(
        lambda: mcp_core.load_mcp_servers(
            ["github"],
            should_mount_all = False,
            stdios = ["srv=npx x"],
            envs = ["github:KEY=v"],
        ),
        capsys,
    )
    assert "keep its env in the registry" in error


def test_env_bad_key_fails(capsys):
    error = _expect_fail(
        lambda: mcp_core.load_mcp_servers(
            None, should_mount_all = False, stdios = ["srv=npx x"], envs = ["srv:9BAD=v"]
        ),
        capsys,
    )
    assert "must match [A-Za-z_][A-Za-z0-9_]*" in error


def test_env_later_key_wins():
    servers = mcp_core.load_mcp_servers(
        None,
        should_mount_all = False,
        stdios = ["srv=npx x"],
        envs = ["srv:KEY=first", "srv:KEY=second"],
    )
    assert servers["srv"]["env"] == {"KEY": "second"}


def test_env_var_expansion_reaches_only_the_entry(monkeypatch):
    monkeypatch.setenv("CTX7_KEY", "s3cret")
    servers = mcp_core.load_mcp_servers(
        None,
        should_mount_all = False,
        stdios = ["ctx7=npx -y @upstash/context7-mcp"],
        envs = ["ctx7:API_KEY=${CTX7_KEY}"],
    )
    assert servers["ctx7"]["env"] == {"API_KEY": "s3cret"}


def test_header_naming_stdio_shortcut_fails(capsys):
    error = _expect_fail(
        lambda: mcp_core.load_mcp_servers(
            None,
            should_mount_all = False,
            urls = ["api=http://127.0.0.1:1/mcp"],
            stdios = ["srv=npx x"],
            headers = ["srv:X-Key=v"],
        ),
        capsys,
    )
    assert "only adds headers to http shortcut servers" in error


def test_header_and_env_without_name_work_with_one_of_each_transport():
    servers = mcp_core.load_mcp_servers(
        None,
        should_mount_all = False,
        urls = ["api=http://127.0.0.1:1/mcp"],
        stdios = ["srv=npx x"],
        headers = ["X-Key=v"],
        envs = ["API_KEY=v"],
    )
    assert servers["api"]["headers"] == {"X-Key": "v"}
    assert servers["srv"]["env"] == {"API_KEY": "v"}


def test_stdio_and_http_mix_in_one_run():
    servers = mcp_core.load_mcp_servers(
        None,
        should_mount_all = False,
        stdios = ["blender=uv run --directory /opt/blender-mcp blender-mcp"],
        urls = ["api=http://127.0.0.1:9000/mcp"],
    )
    assert servers["blender"]["transport"] == "stdio"
    assert servers["blender"]["args"] == ["run", "--directory", "/opt/blender-mcp", "blender-mcp"]
    assert servers["api"]["transport"] == "http"


def test_stdio_http_shortcut_name_clash_fails(capsys):
    error = _expect_fail(
        lambda: mcp_core.load_mcp_servers(
            None,
            should_mount_all = False,
            urls = ["api=http://127.0.0.1:1/mcp"],
            stdios = ["api=npx x"],
        ),
        capsys,
    )
    assert "both named 'api'" in error


# ── --mcp-stdio double-quoted shell form ─────────────────────────────────────


def test_stdio_shell_form_with_name():
    servers = mcp_core.load_mcp_servers(
        None,
        should_mount_all = False,
        stdios = ['blender="uv run --directory ~/workspace/blender-mcp blender-mcp"'],
    )
    assert servers["blender"] == {
        "transport": "stdio",
        "command": "bash",
        "args": ["-ic", "exec uv run --directory ~/workspace/blender-mcp blender-mcp"],
        "env": {},
    }


def test_stdio_shell_form_auto_name_keeps_tilde_and_var_for_bash(monkeypatch):
    monkeypatch.setenv("MCP_DIR", "/opt/x")
    servers = mcp_core.load_mcp_servers(
        None,
        should_mount_all = False,
        stdios = ['"uv run --directory ${MCP_DIR} ~/x blender-mcp"'],
    )
    assert list(servers) == ["blender-mcp"]
    assert servers["blender-mcp"]["args"] == [
        "-ic",
        "exec uv run --directory ${MCP_DIR} ~/x blender-mcp",
    ]


def test_stdio_shell_form_env_still_expands(monkeypatch):
    monkeypatch.setenv("CTX7_KEY", "s3cret")
    servers = mcp_core.load_mcp_servers(
        None,
        should_mount_all = False,
        stdios = ['ctx7="npx -y @upstash/context7-mcp"'],
        envs = ["ctx7:API_KEY=${CTX7_KEY}"],
    )
    assert servers["ctx7"]["env"] == {"API_KEY": "s3cret"}
    assert servers["ctx7"]["args"] == ["-ic", "exec npx -y @upstash/context7-mcp"]


@pytest.mark.parametrize(
    "command",
    ['"echo a; echo b"', '"echo a && echo b"', '"echo a || echo b"', '"echo a | cat"', '"echo a &"'],
)
def test_stdio_shell_form_operators_fail(capsys, command):
    error = _expect_fail(
        lambda: mcp_core.load_mcp_servers(None, should_mount_all = False, stdios = [command]),
        capsys,
    )
    assert "runs one command" in error


def test_stdio_shell_form_quoted_operator_ok():
    servers = mcp_core.load_mcp_servers(
        None,
        should_mount_all = False,
        stdios = ['srv="sh -c "echo a; echo b""'],
    )
    assert servers["srv"]["args"] == ["-ic", 'exec sh -c "echo a; echo b"']


def test_stdio_shell_form_empty_fails(capsys):
    error = _expect_fail(
        lambda: mcp_core.load_mcp_servers(None, should_mount_all = False, stdios = ['srv=""']),
        capsys,
    )
    assert "empty" in error


def test_stdio_shell_form_unbalanced_inner_quotes_fail(capsys):
    error = _expect_fail(
        lambda: mcp_core.load_mcp_servers(
            None, should_mount_all = False, stdios = ['"echo \'oops"']
        ),
        capsys,
    )
    assert "unbalanced quotes" in error


def test_stdio_shell_form_without_bash_fails(capsys, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name, path = None: None)
    error = _expect_fail(
        lambda: mcp_core.load_mcp_servers(None, should_mount_all = False, stdios = ['"npx x"']),
        capsys,
    )
    assert "needs bash on PATH" in error
