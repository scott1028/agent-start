# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See LICENSE

"""Tests for `agent-switch <agent>` — config merging and launch env, no network."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
from typer.testing import CliRunner

import agent_switch.start as start
from agent_switch.agents import dsh as dsh_agent, pi as pi_agent
from tests.start_split import set_start_attr
from tests.cli_support import (
    BASE,
    MODEL,
    _RESUME_ENV_VAR,
    _capture_launch,
    _dsh_entries,
    _launch_command,
    _opencode_inline_config,
    _parse_toml,
)


def _assert_env_cwd(output: str, name: str) -> None:
    needle = f"$env:{name} = (Get-Location).Path" if os.name == "nt" else f'export {name}="$PWD"'
    assert needle in output, f"{needle!r} not found in:\n{output}"


def test_project_declares_direct_cli_dependencies():
    project = _parse_toml((_REPO_ROOT / "pyproject.toml").read_text(encoding = "utf-8"))
    assert "click>=8.0" in project["project"]["dependencies"]


@pytest.mark.parametrize("window, expected", [(32_768, 8_192), (143_616, 32_000)])
def test_pi_and_dsh_output_limit_follows_the_context(tmp_path, window, expected):
    model = {**MODEL, "context_length": window}
    pi_agent.write_pi_config(BASE, "sk-test-abc", model, tmp_path / "models.json")
    pi_agent.write_pi_subagent_config(BASE, "sk-test-abc", model, tmp_path / "subagent.json")
    dsh_agent.write_dsh_patch(BASE, model, tmp_path / "agent-switch.patch.yml")
    pi = json.loads((tmp_path / "models.json").read_text())["providers"]["agent-switch"]["models"][0]
    subagent = json.loads((tmp_path / "subagent.json").read_text())
    patch = _dsh_entries(tmp_path / "agent-switch.patch.yml")
    dsh = patch["llm-pi-ai"]["config"]["providers"]["agent-switch"]["models"][0]
    assert pi["maxTokens"] == subagent["maxTokens"] == dsh["maxTokens"] == expected


# ── --yolo: one switch routed to each agent's own auto-approve form ──


# The native "run tools without prompting" CLI flag each agent should receive.
_NATIVE_YOLO = {
    "claude": "--dangerously-skip-permissions",
    "codex": "--dangerously-bypass-approvals-and-sandbox",
    "pi": "--approve",
}


@pytest.mark.parametrize("agent, native", sorted(_NATIVE_YOLO.items()))
def test_yolo_routes_to_native_flag(fake_vllm, agent, native):
    result = CliRunner().invoke(start.start_app, [agent, "--url", BASE, "--yolo", "--no-launch"])
    assert result.exit_code == 0, result.output
    assert native in result.output


@pytest.mark.parametrize("agent, native", sorted(_NATIVE_YOLO.items()))
def test_no_yolo_omits_native_flag(fake_vllm, agent, native):
    result = CliRunner().invoke(start.start_app, [agent, "--url", BASE, "--no-launch"])
    assert result.exit_code == 0, result.output
    # pi's --approve is a real flag only added under --yolo; assert it's absent here.
    command = _launch_command(result.output)
    assert command and command[0] == agent, result.output
    assert native not in command


@pytest.mark.parametrize(
    "alias",
    ["--yolo", "--dangerously-skip-permissions", "--dangerously-bypass-approvals-and-sandbox"],
)
def test_yolo_aliases_are_interchangeable(fake_vllm, alias):
    # Any spelling on any agent routes to that agent's own flag, even the "wrong" one.
    claude = CliRunner().invoke(start.start_app, ["claude", "--url", BASE, alias, "--no-launch"])
    assert claude.exit_code == 0, claude.output
    assert "--dangerously-skip-permissions" in claude.output
    # The codex spelling must not leak through to Claude's command line.
    assert "--dangerously-bypass-approvals-and-sandbox" not in claude.output

    codex = CliRunner().invoke(start.start_app, ["codex", "--url", BASE, alias, "--no-launch"])
    assert codex.exit_code == 0, codex.output
    assert "--dangerously-bypass-approvals-and-sandbox" in codex.output
    assert "--dangerously-skip-permissions" not in codex.output

    opencode = CliRunner().invoke(
        start.start_app,
        ["opencode", "--url", BASE, alias, "--no-launch", "run", "hello"],
    )
    assert opencode.exit_code == 0, opencode.output
    assert _launch_command(opencode.output) == ["opencode", "run", "hello", "--auto"]
    assert "permission" not in _opencode_inline_config(opencode.output)


def test_yolo_config_fallbacks_add_no_legacy_command_flag(fake_vllm):
    # OpenCode's append-safe bare recipe uses its config fallback, so it must not leak a legacy
    # yolo/dangerous alias onto argv.
    for agent in ("opencode",):
        result = CliRunner().invoke(start.start_app, [agent, "--url", BASE, "--yolo", "--no-launch"])
        assert result.exit_code == 0, result.output
        command = _launch_command(result.output)
        assert command and command[0] == agent, result.output
        assert not any("--yolo" in arg or "--dangerous" in arg for arg in command)


@pytest.mark.parametrize("agent", sorted(_RESUME_ENV_VAR))
def test_resume_persists_agent_home_to_stable_dir(agent, fake_vllm, tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name, path = None: f"/usr/local/bin/{agent}")
    if agent == "dsh":
        set_start_attr(monkeypatch, "is_deepseek_harness_executable", lambda _: True)
    captured = _capture_launch(monkeypatch, [agent, "--url", BASE, "--persist"])
    stable = tmp_path / "agents" / agent
    assert captured["env"][_RESUME_ENV_VAR[agent]] == str(stable)
    # The stable dir survives the agent exit, so the session can be resumed.
    assert stable.exists()


def test_resume_persist_only_agents_have_no_resume_token(fake_vllm, monkeypatch):
    # Persistence alone must not select a session.
    for agent in ("dsh",):
        monkeypatch.setattr(shutil, "which", lambda _, a = agent: f"/usr/local/bin/{a}")
        if agent == "dsh":
            set_start_attr(monkeypatch, "is_deepseek_harness_executable", lambda _: True)
        captured = _capture_launch(monkeypatch, [agent, "--url", BASE, "--persist"])
        assert "resume" not in captured["command"]
        assert "--continue" not in captured["command"]


# These also pin Codex's wording, now that the helpers are shared.


# Tolerance: callers tear the server down on any exception, so only "is_gguf": false rejects.


@pytest.mark.parametrize(
    ("agent", "unset"),
    [
        ("codex", ("OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN")),
    ],
)
def test_launch_drops_provider_credentials(agent, unset, fake_vllm, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name, path = None: f"/usr/local/bin/{agent}")
    for name in unset:
        monkeypatch.setenv(name, "sk-stale")
    captured = _capture_launch(monkeypatch, [agent, "--url", BASE])
    for name in unset:
        assert name not in captured["env"]
