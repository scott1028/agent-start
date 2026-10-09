# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the agent-switch authors. See LICENSE.

"""--compact-at scales each agent's auto-compaction trigger off the reported window."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

import agent_switch.start as start
from tests.cli_support import BASE, MODEL, _assert_env_set
from agent_switch.core import options as core_options
from agent_switch.agents import claude as claude_agent, codex as codex_agent, opencode as opencode_agent
from tests.start_split import set_start_attr


def test_claude_compact_at_sets_pct_override(fake_vllm):
    result = CliRunner().invoke(
        start.start_app, ["claude", "--no-launch", "--compact-at", "0.85"]
    )
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE", "85")


def test_claude_compact_at_unset_keeps_90(fake_vllm):
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE", "90")


def _codex_profile(tmp_path) -> str:
    return (tmp_path / "agents" / "codex" / f"{codex_agent._CODEX_PROFILE}.config.toml").read_text()


def test_codex_compact_at_sets_auto_compact_limit(fake_vllm, tmp_path):
    result = CliRunner().invoke(
        start.start_app, ["codex", "--no-launch", "--compact-at", "0.85"]
    )
    assert result.exit_code == 0, result.output
    assert "model_auto_compact_token_limit = 111411" in _codex_profile(tmp_path)  # 131072 * 0.85


def test_codex_compact_at_unset_omits_limit(fake_vllm, tmp_path):
    result = CliRunner().invoke(start.start_app, ["codex", "--no-launch"])
    assert result.exit_code == 0, result.output
    assert "model_auto_compact_token_limit" not in _codex_profile(tmp_path)


def test_opencode_compact_at_scales_reserved(fake_vllm, tmp_path):
    result = CliRunner().invoke(
        start.start_app, ["opencode", "--no-launch", "--compact-at", "0.85"]
    )
    assert result.exit_code == 0, result.output
    config = json.loads((tmp_path / "agents" / "opencode" / "opencode.json").read_text())
    assert config["compaction"] == {"auto": True, "reserved": 19660}  # 131072 * 0.15


def test_opencode_compact_at_unset_keeps_default_reserved(fake_vllm, tmp_path):
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])
    assert result.exit_code == 0, result.output
    config = json.loads((tmp_path / "agents" / "opencode" / "opencode.json").read_text())
    assert config["compaction"] == {"auto": True, "reserved": 131072 // 10}


def _pi_settings(tmp_path) -> dict:
    path = tmp_path / "agents" / "pi" / ".pi" / "agent" / "settings.json"
    return json.loads(path.read_text()) if path.exists() else {}


def test_pi_compact_at_sets_reserve(fake_vllm, tmp_path):
    result = CliRunner().invoke(
        start.start_app, ["pi", "--no-launch", "--compact-at", "0.85"]
    )
    assert result.exit_code == 0, result.output
    # Pi compacts once the context exceeds contextWindow - reserveTokens.
    assert _pi_settings(tmp_path)["compaction"] == {"enabled": True, "reserveTokens": 19660}  # 131072 * 0.15


def test_pi_compact_at_unset_omits_compaction(fake_vllm, tmp_path):
    result = CliRunner().invoke(start.start_app, ["pi", "--no-launch"])
    assert result.exit_code == 0, result.output
    assert "compaction" not in _pi_settings(tmp_path)


def test_pi_compact_at_cleared_on_rerun_without_flag(fake_vllm, tmp_path):
    result = CliRunner().invoke(start.start_app, ["pi", "--no-launch", "--compact-at", "0.85"])
    assert result.exit_code == 0, result.output
    assert "compaction" in _pi_settings(tmp_path)
    result = CliRunner().invoke(start.start_app, ["pi", "--no-launch"])
    assert result.exit_code == 0, result.output
    assert "compaction" not in _pi_settings(tmp_path)


@pytest.mark.parametrize("value", ["0.3", "0.99", "1.5", "-0.1"])
def test_compact_at_out_of_range_rejected(fake_vllm, value):
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch", "--compact-at", value])
    assert result.exit_code != 0
    assert "Invalid value for '--compact-at'" in result.output


@pytest.mark.parametrize("value", ["0.5", "0.95"])
def test_compact_at_range_accepted(fake_vllm, value):
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch", "--compact-at", value])
    assert result.exit_code == 0, result.output


def test_check_compact_at_warns_without_a_window(capsys):
    core_options._check_compact_at(0.85, {"id": "some-model"})
    assert "--compact-at is ignored" in capsys.readouterr().err


def test_check_compact_at_silent_with_a_window(capsys):
    core_options._check_compact_at(0.85, {"id": "some-model", "context_length": 32768})
    assert capsys.readouterr().err == ""


def test_claude_local_env_pct_from_compact_at():
    env = claude_agent._claude_local_env(
        BASE, "k", {"id": "m", "context_length": 32768}, compact_at = 0.85
    )
    assert env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] == "85"


def test_opencode_compaction_reserved_ratio_skips_cap_and_floor():
    # An explicit ratio is honored exactly: no output cap, no 8192 floor.
    assert opencode_agent.opencode_compaction_reserved(16_384, 4_096, 0.85) == 2_457  # 16384 * 0.15


def test_opencode_subagent_compact_at_warns(fake_vllm):
    result = CliRunner().invoke(
        start.start_app, ["opencode", "--no-launch", "--as-subagent", "--compact-at", "0.85"]
    )
    assert result.exit_code == 0, result.output
    assert "--compact-at does not apply with --as-subagent for OpenCode" in result.stderr


def test_pi_subagent_compact_at_warns(fake_vllm):
    result = CliRunner().invoke(
        start.start_app, ["pi", "--no-launch", "--as-subagent", "--compact-at", "0.85"]
    )
    assert result.exit_code == 0, result.output
    assert "--compact-at does not apply with --as-subagent for Pi" in result.stderr


def test_claude_subagent_plugin_settings_carry_pct(tmp_path):
    plugin = claude_agent.write_claude_subagent_plugin(
        tmp_path,
        {
            "AGENT_SWITCH_CLAUDE_SUBAGENT_BASE_URL": BASE,
            "AGENT_SWITCH_CLAUDE_SUBAGENT_API_KEY": "sk-test",
            "AGENT_SWITCH_CLAUDE_SUBAGENT_MODEL": "m1",
            "AGENT_SWITCH_CLAUDE_SUBAGENT_CONTEXT_WINDOW": "32768",
            "AGENT_SWITCH_CLAUDE_SUBAGENT_COMPACT_AT": "0.85",
        },
    )
    settings_file = next(plugin.glob("settings-*.json"))
    settings = json.loads(settings_file.read_text())
    assert settings["env"]["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] == "85"


def test_claude_subagent_child_gets_pct(monkeypatch, tmp_path):
    import agent_switch.claude_subagent_mcp as bridge

    captured = {}
    monkeypatch.setenv("AGENT_SWITCH_CLAUDE_SUBAGENT_BASE_URL", BASE)
    monkeypatch.setenv("AGENT_SWITCH_CLAUDE_SUBAGENT_API_KEY", "sk-test")
    monkeypatch.setenv("AGENT_SWITCH_CLAUDE_SUBAGENT_MODEL", "m1")
    monkeypatch.setenv("AGENT_SWITCH_CLAUDE_SUBAGENT_CONTEXT_WINDOW", "32768")
    monkeypatch.setenv("AGENT_SWITCH_CLAUDE_SUBAGENT_COMPACT_AT", "0.85")
    monkeypatch.setenv(bridge._CLAUDE_SUBAGENT_SETTINGS_ENV, str(tmp_path / "settings.json"))
    monkeypatch.setattr(bridge.shutil, "which", lambda _: "/usr/local/bin/claude")
    monkeypatch.setattr(bridge, "_claude_flags", lambda model, settings = None: ["--settings", settings])

    class Process:
        pid = 1
        returncode = 0

        def communicate(self, timeout):
            return json.dumps({"is_error": False, "result": "OK"}), ""

        def poll(self):
            return 0

    def popen(command, **kwargs):
        captured.update(kwargs)
        return Process()

    monkeypatch.setattr(bridge.subprocess, "Popen", popen)
    assert bridge.run_local_agent("ping") == "OK"
    assert captured["env"]["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] == "85"


def test_codex_subagent_bridge_carries_compact_at(tmp_path, monkeypatch):
    set_start_attr(monkeypatch, "_codex_supports_model_catalog", lambda: False)
    codex_agent.write_codex_subagent_bridge(
        BASE, "sk-test", MODEL, tmp_path, yolo = False, compact_at = 0.85
    )
    profile = (tmp_path / "child" / f"{codex_agent._CODEX_PROFILE}.config.toml").read_text()
    assert "model_auto_compact_token_limit = 111411" in profile