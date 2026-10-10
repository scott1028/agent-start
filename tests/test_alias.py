# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See LICENSE

"""`agent-switch <agent>-<x>`: alias resolution, the bash launch form and the refusals."""

import shutil
import subprocess
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

import agent_switch.start as start
from agent_switch.core.launch import _agent_command, _check_alias
from tests.cli_support import BASE, _assert_env_set, _launch_command
from tests.start_split import set_start_attr


def _alias_ok(monkeypatch):
    # These CLI cases assert the command shape; the bash probe has its own cases below.
    set_start_attr(monkeypatch, "_check_alias", lambda alias: None)


def _native_no_connect(monkeypatch):
    set_start_attr(monkeypatch, "_connect",
        lambda *args, **kwargs: pytest.fail("native launch must not connect"),
    )


def _expect_fail(call, capsys) -> str:
    with pytest.raises(typer.Exit) as caught:
        call()
    assert caught.value.exit_code == 1
    return capsys.readouterr().err


def _fake_bash(monkeypatch, kind, definition = "", path = "/usr/local/bin/x"):
    def run(command, **kwargs):
        script = command[-1]
        if script.startswith("type -t"):
            return SimpleNamespace(stdout = f"{kind}\n" if kind else "", returncode = int(not kind))
        if script.startswith("command -v"):
            return SimpleNamespace(stdout = f"{path}\n", returncode = 0)
        return SimpleNamespace(stdout = definition, returncode = 0)

    monkeypatch.setattr(subprocess, "run", run)


def _find_bash(monkeypatch):
    monkeypatch.setattr(
        shutil, "which", lambda name, path = None: "/bin/bash" if name == "bash" else None
    )


# ── Resolution ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    ["claude-personal", "codex-personal", "opencode-laptop", "pi-personal"],
)
def test_alias_runs_its_agent_kind_natively(fake_vllm, monkeypatch, name):
    _alias_ok(monkeypatch)
    _native_no_connect(monkeypatch)
    result = CliRunner().invoke(start.start_app, [name, "--no-launch"])
    assert result.exit_code == 0, result.output
    assert f"{name} runs with its own model, login and config" in result.output
    assert _launch_command(result.output) == ["bash", "-ic", f'{name} "$@"', "agent-switch"]


def test_dsh_alias_keeps_the_dsh_kind(fake_vllm, monkeypatch):
    _alias_ok(monkeypatch)
    _native_no_connect(monkeypatch)
    result = CliRunner().invoke(start.start_app, ["dsh-x", "--no-launch"])
    assert result.exit_code == 0, result.output
    # `web` is what agent-switch would have passed to the real dsh, so it rides along.
    assert _launch_command(result.output) == ["bash", "-ic", 'dsh-x "$@"', "agent-switch", "web"]


def test_dsh_tui_alias_keeps_the_dsh_tui_kind(fake_vllm, tmp_path, monkeypatch):
    _alias_ok(monkeypatch)
    _native_no_connect(monkeypatch)
    result = CliRunner().invoke(
        start.start_app, ["dsh-tui-x", "--no-launch", "--mcp-stdio", "ev=npx everything"]
    )
    assert result.exit_code == 0, result.output
    patch = tmp_path / "agents" / "dsh-tui-native" / "agent-switch.patch.yml"
    assert _launch_command(result.output) == [
        "bash", "-ic", 'dsh-tui-x "$@"', "agent-switch", "--patch", str(patch),
    ]


@pytest.mark.parametrize("name", ["claude", "codex", "opencode", "pi", "dsh", "dsh-tui", "dst"])
def test_exact_subcommands_stay_exact(fake_vllm, monkeypatch, name):
    _native_no_connect(monkeypatch)
    result = CliRunner().invoke(start.start_app, [name, "--no-launch"])
    assert result.exit_code == 0, result.output
    assert _launch_command(result.output)[0] != "bash"


@pytest.mark.parametrize("name", ["foo-x", "pii", "claude-", "claude-;evil", "dst-x", "personal"])
def test_other_names_stay_no_such_command(fake_vllm, name):
    result = CliRunner().invoke(start.start_app, [name, "--no-launch"])
    assert result.exit_code == 2
    assert "No such command" in result.output


# ── Launch command shape ─────────────────────────────────────────────


def test_native_claude_alias_flags_ride_the_bash_args(fake_vllm, tmp_path, monkeypatch):
    _alias_ok(monkeypatch)
    _native_no_connect(monkeypatch)
    result = CliRunner().invoke(
        start.start_app,
        ["claude-personal", "--no-launch", "--mcp-stdio", "ev=npx everything", "--yolo", "--print", "hi"],
    )
    assert result.exit_code == 0, result.output
    mcp_config = tmp_path / "agents" / "claude-native" / "mcp.json"
    assert _launch_command(result.output) == [
        "bash",
        "-ic",
        'claude-personal "$@"',
        "agent-switch",
        "--dangerously-skip-permissions",
        f"--mcp-config={mcp_config}",
        "--print",
        "hi",
    ]


def test_native_codex_alias_mcp_flags_ride_the_bash_args(fake_vllm, monkeypatch):
    _alias_ok(monkeypatch)
    _native_no_connect(monkeypatch)
    result = CliRunner().invoke(
        start.start_app, ["codex-personal", "--no-launch", "--mcp-stdio", "ev=npx everything"]
    )
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    assert command[:4] == ["bash", "-ic", 'codex-personal "$@"', "agent-switch"]
    assert command[4:] == ["-c", 'mcp_servers.ev={command = "npx", args = ["everything"]}']


def test_native_pi_alias_mcp_config_rides_the_bash_args(fake_vllm, tmp_path, monkeypatch):
    _alias_ok(monkeypatch)
    _native_no_connect(monkeypatch)
    result = CliRunner().invoke(
        start.start_app, ["pi-personal", "--no-launch", "--mcp-stdio", "ev=npx everything"]
    )
    assert result.exit_code == 0, result.output
    mcp_config = tmp_path / "agents" / "pi-native" / "mcp.json"
    assert _launch_command(result.output) == [
        "bash", "-ic", 'pi-personal "$@"', "agent-switch", "--mcp-config", str(mcp_config),
    ]


def test_local_claude_alias_wraps_the_local_command(fake_vllm, monkeypatch):
    _alias_ok(monkeypatch)
    result = CliRunner().invoke(
        start.start_app, ["claude-personal", "--url", BASE, "--no-launch"]
    )
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    assert command[:2] == ["bash", "-ic"]
    # The unset inside the bash string re-clears what the re-read shell config could re-export.
    assert command[2].startswith("unset ANTHROPIC_API_KEY")
    assert command[2].endswith('claude-personal "$@"')
    assert command[3] == "agent-switch"
    assert "--model" in command[4:]
    _assert_env_set(result.output, "ANTHROPIC_BASE_URL", BASE)


def test_local_opencode_alias_wraps_the_local_command(fake_vllm, monkeypatch):
    _alias_ok(monkeypatch)
    result = CliRunner().invoke(
        start.start_app, ["opencode-laptop", "--url", BASE, "--no-launch", "run", "hi"]
    )
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    assert command[:4] == ["bash", "-ic", 'opencode-laptop "$@"', "agent-switch"]
    assert command[4:] == ["run", "hi"]
    assert "OPENCODE_CONFIG=" in result.output


# ── Refusals ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "needle"),
    [
        ("codex-personal", "codex aliases run natively only"),
        ("pi-personal", "pi aliases run natively only"),
        ("dsh-x", "dsh aliases run natively only"),
        ("dsh-tui-x", "dsh-tui aliases run natively only"),
    ],
)
def test_local_mode_refused_for_home_owning_aliases(fake_vllm, monkeypatch, name, needle):
    _alias_ok(monkeypatch)
    result = CliRunner().invoke(start.start_app, [name, "--url", BASE, "--no-launch"])
    assert result.exit_code == 1
    assert needle in result.output


@pytest.mark.parametrize("name", ["claude-personal", "codex-personal", "opencode-laptop", "pi-personal"])
def test_alias_refuses_as_subagent(fake_vllm, monkeypatch, name):
    _alias_ok(monkeypatch)
    result = CliRunner().invoke(
        start.start_app, [name, "--url", BASE, "--as-subagent", "--no-launch"]
    )
    assert result.exit_code == 1
    assert "--as-subagent is not supported for agent aliases" in result.output


def test_check_alias_accepts_a_forwarding_function(monkeypatch):
    _find_bash(monkeypatch)
    _fake_bash(monkeypatch, "function", 'claude-personal () \n{\n\tcommand claude "$@"\n}\n')
    _check_alias("claude-personal")


def test_check_alias_needs_bash(monkeypatch, capsys):
    monkeypatch.setattr(shutil, "which", lambda name, path = None: None)
    error = _expect_fail(lambda: _check_alias("claude-personal"), capsys)
    assert "needs `bash` on PATH" in error


def test_check_alias_refused_on_windows(monkeypatch, capsys):
    monkeypatch.setattr("os.name", "nt")
    error = _expect_fail(lambda: _check_alias("claude-personal"), capsys)
    assert "Windows is not supported" in error


def test_check_alias_refuses_bad_name_characters(monkeypatch, capsys):
    _find_bash(monkeypatch)
    _fake_bash(monkeypatch, "function")
    error = _expect_fail(lambda: _check_alias("claude;evil"), capsys)
    assert "characters an agent alias name cannot have" in error


def test_check_alias_refuses_a_missing_name(monkeypatch, capsys):
    _find_bash(monkeypatch)
    _fake_bash(monkeypatch, "")
    error = _expect_fail(lambda: _check_alias("claude-personal"), capsys)
    assert "is not a bash function, alias or file" in error


def test_check_alias_refuses_a_recursive_function(monkeypatch, capsys):
    _find_bash(monkeypatch)
    _fake_bash(monkeypatch, "function", "opencode-local () \n{\n\tagent-switch opencode --url x\n}\n")
    error = _expect_fail(lambda: _check_alias("opencode-local"), capsys)
    assert "would recurse" in error


def test_check_alias_refuses_a_recursive_alias(monkeypatch, capsys):
    _find_bash(monkeypatch)
    _fake_bash(monkeypatch, "alias", "alias opencode-x='agent-switch opencode'")
    error = _expect_fail(lambda: _check_alias("opencode-x"), capsys)
    assert "would recurse" in error


def test_check_alias_refuses_a_recursive_script(monkeypatch, capsys, tmp_path):
    script = tmp_path / "opencode-x"
    script.write_text("#!/bin/sh\nagent-switch opencode --url x \"$@\"\n", encoding = "utf-8")
    _find_bash(monkeypatch)
    _fake_bash(monkeypatch, "file", path = str(script))
    error = _expect_fail(lambda: _check_alias("opencode-x"), capsys)
    assert "would recurse" in error


def test_check_alias_accepts_a_plain_script(monkeypatch, tmp_path):
    script = tmp_path / "opencode-x"
    script.write_text("#!/bin/sh\nexec opencode \"$@\"\n", encoding = "utf-8")
    _find_bash(monkeypatch)
    _fake_bash(monkeypatch, "file", path = str(script))
    _check_alias("opencode-x")


# ── _agent_command ───────────────────────────────────────────────────


def test_agent_command_without_alias_is_the_plain_argv():
    assert _agent_command(None, "claude", ["--x"]) == ["claude", "--x"]


def test_agent_command_clears_unset_names_inside_bash():
    assert _agent_command("claude-x", "claude", ["--model", "m"], ("ANTHROPIC_API_KEY",)) == [
        "bash", "-ic", 'unset ANTHROPIC_API_KEY; claude-x "$@"', "agent-switch", "--model", "m",
    ]
