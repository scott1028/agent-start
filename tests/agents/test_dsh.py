# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Ported from unsloth_cli/tests/test_start.py (unsloth commit 8e11ba15e); see NOTICE.md.

"""`agent-switch dsh`: patch writing, command selection and permission modes."""

import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import agent_switch.start as start
from agent_switch.agents import (
    dsh as dsh_agent,
)
from agent_switch.core import (
    install as core_install,
)
from tests.cli_support import (
    BASE,
    KEY,
    MODEL,
    _SESSION_FLAGS,
    _assert_env_set,
    _capture_launch,
    _dsh_entries,
    _launch_command,
)
from tests.start_split import set_start_attr


@pytest.mark.parametrize("agent", ["dsh"])
@pytest.mark.parametrize("flag", ["--as-subagent", "--as-subagent=true", "--as-subagent=false"])
def test_unsupported_agents_reject_as_subagent(agent, flag):
    result = CliRunner().invoke(start.start_app, [agent, flag])
    assert result.exit_code == 1
    assert f"--as-subagent is not supported for {agent}." in result.output


@pytest.mark.usefixtures("one_local_server")
def test_dsh_rejects_an_unrelated_executable_before_connect(monkeypatch):
    set_start_attr(monkeypatch, "_which_with_install_dirs", lambda _: "/usr/bin/dsh")
    set_start_attr(monkeypatch, "is_deepseek_harness_executable", lambda _: False)
    set_start_attr(monkeypatch, "_install_agent", lambda *_: None)
    set_start_attr(monkeypatch, "_connect",
        lambda *args, **kwargs: pytest.fail("the wrong dsh must be rejected before connection"),
    )

    result = CliRunner().invoke(start.start_app, ["dsh"])

    assert result.exit_code == 1
    assert "`/usr/bin/dsh` is not DeepSeek Harness" in result.output


def test_dsh_resolver_searches_past_an_unrelated_earlier_path_entry(monkeypatch, tmp_path):
    shadow_dir = tmp_path / "system-bin"
    harness_dir = tmp_path / "user-bin"
    shadow_dir.mkdir()
    harness_dir.mkdir()
    suffix = ".cmd" if os.name == "nt" else ""
    for directory in (shadow_dir, harness_dir):
        executable = directory / f"dsh{suffix}"
        executable.write_text("@echo off\n" if os.name == "nt" else "#!/bin/sh\n")
        if os.name != "nt":
            executable.chmod(0o755)

    monkeypatch.setenv("PATH", os.pathsep.join((str(shadow_dir), str(harness_dir))))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "missing-home")
    set_start_attr(monkeypatch, "is_deepseek_harness_executable",
        lambda executable: Path(executable).parent == harness_dir,
    )
    set_start_attr(monkeypatch, "_install_agent",
        lambda *_: pytest.fail("an existing later Harness must be used without reinstalling"),
    )

    resolved = core_install._resolve_or_install_agent(
        "dsh",
        "npm install -g @deepseek-ai/dsh",
        core_install._which_with_install_dirs,
    )

    assert Path(resolved).parent == harness_dir


def test_dsh_carries_reasoning_and_warns_about_sampling(fake_vllm, tmp_path, monkeypatch):
    yaml = pytest.importorskip("yaml")
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(start.start_app, ["dsh", "--no-launch", *_SESSION_FLAGS])
    assert result.exit_code == 0, result.output
    assert "can't send --temperature, --top-k itself" in result.output
    assert "--reasoning" not in result.output
    patch = tmp_path / "agents" / "dsh" / dsh_agent._DSH_PATCH_FILE
    provider = yaml.safe_load(patch.read_text())[0]["config"]["providers"][dsh_agent._DSH_PROVIDER]
    assert provider["compat"]["thinkingFormat"] == "chat-template"
    assert provider["compat"]["chatTemplateKwargs"] == {"enable_thinking": False}
    assert "off" in provider["models"][0]["reasoningEfforts"]


# ── DeepSeek Harness (OpenAI /v1, key via env, ~/.dsh relocated) ─────
@pytest.fixture()
def dsh_patch(tmp_path):
    return tmp_path / "agent-switch.patch.yml"


def test_write_dsh_patch_fresh(dsh_patch):
    dsh_agent.write_dsh_patch(BASE, MODEL, dsh_patch)
    entries = _dsh_entries(dsh_patch)
    assert set(entries) == {"llm-pi-ai", "agent-default-model"}
    assert entries["llm-pi-ai"]["name"] == "@deepseek-ai/dsh-llm-pi-ai"
    assert entries["agent-default-model"]["name"] == "@deepseek-ai/dsh-agent-default-model"
    provider = entries["llm-pi-ai"]["config"]["providers"]["agent-switch"]
    assert provider["api"] == "openai-completions"
    assert provider["baseURL"] == f"{BASE}/v1"
    assert provider["apiKeyEnv"] == "AGENT_SWITCH_API_KEY"
    assert "sk-test" not in dsh_patch.read_text()
    assert provider["compat"] == {"supportsDeveloperRole": False, "maxTokensField": "max_tokens"}
    assert provider["models"] == [
        {"id": MODEL["id"], "contextWindow": MODEL["context_length"], "maxTokens": 32000}
    ]
    assert entries["agent-default-model"]["config"] == {
        "provider": "agent-switch",
        "model": MODEL["id"],
    }


def test_write_dsh_patch_without_window_omits_limits(dsh_patch):
    dsh_agent.write_dsh_patch(BASE, {"id": "org/unknown-window"}, dsh_patch)
    provider = _dsh_entries(dsh_patch)["llm-pi-ai"]["config"]["providers"]["agent-switch"]
    assert provider["models"] == [{"id": "org/unknown-window"}]


def test_write_dsh_patch_is_idempotent_and_follows_the_server(dsh_patch, capsys):
    dsh_agent.write_dsh_patch(BASE, MODEL, dsh_patch)
    before = dsh_patch.read_text()
    capsys.readouterr()
    dsh_agent.write_dsh_patch(BASE, MODEL, dsh_patch)
    assert dsh_patch.read_text() == before
    assert "Updated" not in capsys.readouterr().out
    # agent-switch owns this file: a new server or model replaces the old one, it does not pile up.
    dsh_agent.write_dsh_patch("http://127.0.0.1:9999", {"id": "other"}, dsh_patch)
    entries = _dsh_entries(dsh_patch)
    provider = entries["llm-pi-ai"]["config"]["providers"]["agent-switch"]
    assert provider["baseURL"] == "http://127.0.0.1:9999/v1"
    assert provider["models"] == [{"id": "other"}]
    assert entries["agent-default-model"]["config"]["model"] == "other"


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ([], ["dsh", "web"]),
        (["--no-open"], ["dsh", "web", "--no-open"]),
        (["--port", "3099"], ["dsh", "web", "--port", "3099"]),
        (["--profile", "headless", "fix the bug"], ["dsh", "--profile", "headless", "fix the bug"]),
        (["--profile=headless", "fix"], ["dsh", "--profile=headless", "fix"]),
        (
            ["plugin", "--profile", "web", "add", "x"],
            ["dsh", "plugin", "--profile", "web", "add", "x"],
        ),
        (["web", "--no-open"], ["dsh", "web", "--no-open"]),
    ],
)
def test_dsh_command_selects_web_only_for_app_arguments(args, expected):
    assert dsh_agent._dsh_command(args) == expected


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        # A bare profile name must stay first so dsh expands it to --profile <name>.
        ([], ["dsh", "web", "--patch", "P"]),
        (["--no-open"], ["dsh", "web", "--patch", "P", "--no-open"]),
        (["web", "--no-open"], ["dsh", "web", "--patch", "P", "--no-open"]),
        (
            ["--profile", "headless", "fix the bug"],
            ["dsh", "--patch", "P", "--profile", "headless", "fix the bug"],
        ),
        (["--profile=headless", "fix"], ["dsh", "--patch", "P", "--profile=headless", "fix"]),
        (["--dump-config"], ["dsh", "--patch", "P", "--dump-config"]),
        # --patch repeats and composes in order, so a caller's own overlay lands after ours
        # and wins only on the keys it sets; the agent-switch provider stays defined.
        (
            ["--patch", "mine.yml", "--profile", "headless"],
            ["dsh", "--patch", "P", "--patch", "mine.yml", "--profile", "headless"],
        ),
        (["--patch=mine.yml", "web"], ["dsh", "--patch", "P", "--patch=mine.yml", "web"]),
        # Nothing boots a profile here, so there is nothing for an overlay to apply to.
        (
            ["plugin", "--profile", "web", "add", "x"],
            ["dsh", "plugin", "--profile", "web", "add", "x"],
        ),
        (["-V"], ["dsh", "-V"]),
        (["--version"], ["dsh", "--version"]),
    ],
)
def test_dsh_command_places_the_patch_where_dsh_parses_it(args, expected):
    assert dsh_agent._dsh_command(args, "P") == expected


def test_connect_dsh_no_launch(fake_vllm, tmp_path):
    yaml = pytest.importorskip("yaml")
    result = CliRunner().invoke(start.start_app, ["dsh", "--no-launch"])
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "AGENT_SWITCH_API_KEY", KEY)
    home = tmp_path / "agents" / "dsh"
    _assert_env_set(result.output, "DSH_HOME", str(home))
    _assert_env_set(result.output, "DSH_TELEMETRY_DISABLED", "1")
    patch = home / "agent-switch.patch.yml"
    assert _launch_command(result.output) == ["dsh", "web", "--patch", str(patch)]
    entries = {entry["id"]: entry for entry in yaml.safe_load(patch.read_text())}
    assert entries["agent-default-model"]["config"] == {
        "provider": "agent-switch",
        "model": MODEL["id"],
    }
    assert entries["llm-pi-ai"]["config"]["providers"]["agent-switch"]["baseURL"] == f"{BASE}/v1"
    # dsh 0.1.7 imports a settings.yaml into the profile only after boot, so none is written.
    assert not (home / "settings.yaml").exists()


@pytest.mark.skipif(os.name == "nt", reason = "WSL scenario")
def test_dsh_under_wsl_gets_the_windows_patch_path(fake_vllm, monkeypatch):
    # WSLENV translates DSH_HOME for a Windows dsh, but a path on the command line reaches
    # the Windows Node process verbatim, where a Linux path does not open.
    windows_path = r"\\wsl.localhost\Ubuntu\tmp\agent-switch.patch.yml"
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    shim = "/mnt/c/Users/x/AppData/Roaming/npm/dsh"
    monkeypatch.setattr(shutil, "which", lambda _: shim)
    set_start_attr(monkeypatch, "is_deepseek_harness_executable", lambda _: True)
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: windows_path)
    captured = _capture_launch(monkeypatch, ["dsh", "--profile", "headless", "hi"])
    command = captured["command"]
    assert command[command.index("--patch") + 1] == windows_path, command


def test_dsh_yolo_sets_permission_mode(fake_vllm):
    result = CliRunner().invoke(start.start_app, ["dsh", "--yolo", "--no-launch"])
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "DSH_PERMISSION_MODE", "danger-full-access")


def test_dsh_without_yolo_pins_the_safe_permission_mode(fake_vllm):
    result = CliRunner().invoke(start.start_app, ["dsh", "--no-launch"])
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "DSH_PERMISSION_MODE", "workspace-write")


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["dsh"], "workspace-write"),
        (["dsh", "--yolo"], "danger-full-access"),
    ],
)
def test_dsh_permission_mode_overrides_an_inherited_bypass(
    argv, expected, fake_vllm, monkeypatch
):
    # dsh reads DSH_PERMISSION_MODE with ??, so merely omitting it would let a
    # danger-full-access exported in the parent shell survive a run without --yolo.
    monkeypatch.setenv("DSH_PERMISSION_MODE", "danger-full-access")
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/local/bin/dsh")
    set_start_attr(monkeypatch, "is_deepseek_harness_executable", lambda _: True)
    captured = _capture_launch(monkeypatch, argv)
    assert captured["env"]["DSH_PERMISSION_MODE"] == expected


def test_start_dsh_forwards_reasoning_effort(fake_vllm, monkeypatch):
    # --reasoning-effort is a shared option: it must reach the dsh config rather than
    # pass through to `dsh web`, which does not accept it.
    yaml = pytest.importorskip("yaml")
    captured = {}
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/local/bin/dsh")
    set_start_attr(monkeypatch, "is_deepseek_harness_executable", lambda _: True)

    def run(
        command,
        env = None,
        **kwargs,
    ):
        captured["command"] = command
        captured["patch"] = Path(command[command.index("--patch") + 1]).read_text()
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(subprocess, "run", run)

    result = CliRunner().invoke(
        start.start_app,
        ["dsh", "--model", MODEL["id"], "--reasoning-effort", "high"],
    )
    assert result.exit_code == 0, result.output
    assert "--reasoning-effort" not in captured["command"]
    provider = yaml.safe_load(captured["patch"])[0]["config"]["providers"][dsh_agent._DSH_PROVIDER]
    assert provider["compat"]["chatTemplateKwargs"] == {"reasoning_effort": "high"}
