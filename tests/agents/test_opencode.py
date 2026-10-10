# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Ported from unsloth_cli/tests/test_start.py (unsloth commit 8e11ba15e); see NOTICE.md.

"""`agent-switch opencode`: config writing, output limits, yolo and subagent wiring."""

import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import agent_switch.start as start
from agent_switch.agents import (
    opencode as opencode_agent,
)
from agent_switch.core import (
    install as core_install,
    options as core_options,
)
from tests.cli_support import (
    BASE,
    KEY,
    MODEL,
    _assert_env_set,
    _capture_launch,
    _launch_command,
    _mcp_registry,
    _opencode_inline_config,
    _path_aware_which,
)
from tests.start_split import set_start_attr


def test_opencode_native_auto_probes_old_opencode_only_in_install_dir(monkeypatch, tmp_path):
    # Same ordering fix for opencode: an old opencode present only in an install dir is detected
    # so native --auto is not assumed (the old binary rejects it).
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents = True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))
    monkeypatch.setattr(shutil, "which", _path_aware_which({"opencode": local_bin}))
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "1.17.11")
    assert opencode_agent._opencode_supports_native_auto() is False


def test_opencode_command_prefers_installed_v2(monkeypatch):
    set_start_attr(monkeypatch, "_which_with_install_dirs",
        lambda name: "/usr/local/bin/opencode2" if name == "opencode2" else None,
    )
    assert opencode_agent._opencode_command() == ("/usr/local/bin/opencode2", True)


def test_opencode_command_falls_back_to_v1(monkeypatch):
    set_start_attr(monkeypatch, "_which_with_install_dirs", lambda _: None)
    assert opencode_agent._opencode_command() == ("opencode", False)


def test_opencode_command_finds_official_v2_install_dir(monkeypatch, tmp_path):
    install_dir = tmp_path / ".opencode" / "bin"
    install_dir.mkdir(parents = True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))
    monkeypatch.setattr(shutil, "which", _path_aware_which({"opencode2": install_dir}))

    assert opencode_agent._opencode_command() == (str(install_dir / "opencode2"), True)


@pytest.mark.usefixtures("one_local_server")
def test_declined_opencode_subagent_install_stops_before_connect(monkeypatch):
    installs = []
    set_start_attr(monkeypatch, "_which_with_install_dirs", lambda _: None)
    set_start_attr(monkeypatch, "_install_agent",
        lambda name, hint: installs.append((name, hint)),
    )
    set_start_attr(monkeypatch, "_connect",
        lambda *a, **k: pytest.fail("declined install must stop before model connection"),
    )

    result = CliRunner().invoke(start.start_app, ["opencode", "--as-subagent"])

    assert result.exit_code == 1
    assert len(installs) == 1
    assert installs[0][0] == "opencode"


@pytest.mark.usefixtures("one_local_server")
def test_opencode_no_launch_resolves_generation_without_installing(monkeypatch):
    resolved = []
    set_start_attr(monkeypatch, "_which_with_install_dirs",
        lambda name: resolved.append(name)
        or ("/home/me/.opencode/bin/opencode2" if name == "opencode2" else None),
    )
    set_start_attr(monkeypatch, "_install_agent",
        lambda *args: pytest.fail("--no-launch must not install an agent"),
    )
    set_start_attr(monkeypatch, "_connect", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError)
    )

    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])

    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)
    assert resolved == ["opencode2"]


def test_launch_native_posix_child_gets_current_pwd(fake_vllm, monkeypatch, tmp_path):
    captured = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PWD", "/stale/outer/repo")
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/local/bin/opencode")

    def run(command, env):
        captured["command"] = command
        captured["env"] = env
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(subprocess, "run", run)

    result = CliRunner().invoke(start.start_app, ["opencode"])

    assert result.exit_code == 0, result.output
    assert captured["command"][0] == "/usr/local/bin/opencode"
    if os.name != "nt":
        assert captured["env"]["PWD"] == os.getcwd()


def test_opencode_inline_config_beats_project_config(fake_vllm):
    # A project's opencode.json outranks OPENCODE_CONFIG, so the model pin (and --yolo
    # permissions) ride in OPENCODE_CONFIG_CONTENT, which outranks project config.
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch", "--yolo"])
    assert result.exit_code == 0, result.output
    inline = _opencode_inline_config(result.output)
    assert inline["model"] == f"{opencode_agent._OPENCODE_PROVIDER}/{MODEL['id']}"
    assert inline["permission"] == {
        "edit": "allow",
        "bash": "allow",
        "webfetch": "allow",
        "external_directory": {"*": "allow"},
    }
    assert KEY not in result.output  # key stays in the private file, not the env


def test_opencode_inline_config_omits_permission_without_yolo(fake_vllm):
    # A non-yolo session carries no permission inline. OPENCODE_CONFIG_CONTENT outranks the
    # project opencode.json we cannot read, so forcing any value there would override the
    # user's project rules; clearing our own config is the fix, and the inline pins the model.
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])
    assert result.exit_code == 0, result.output
    inline = _opencode_inline_config(result.output)
    assert inline["model"] == f"{opencode_agent._OPENCODE_PROVIDER}/{MODEL['id']}"
    assert "permission" not in inline


def test_opencode_session_temperature_needs_the_capability(fake_vllm, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "agents" / "opencode" / "opencode.json"
    for argv, capability in ((["--temperature", "0.3"], True), (["--top-k", "40"], None)):
        result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch", *argv])
        assert result.exit_code == 0, result.output
        provider = json.loads(config_path.read_text())["provider"][opencode_agent._OPENCODE_PROVIDER]
        assert provider["models"][MODEL["id"]].get("temperature") is capability


def test_session_carries_the_custom_header_to_the_agent(fake_vllm, tmp_path, monkeypatch):
    # A custom Authorization wins over the API key on agent-switch's own requests too, so the
    # agent config must carry it and drop the apiKey it no longer sends.
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        start.start_app, ["opencode", "--no-launch", "--header", "Authorization=Bearer gateway-token"]
    )
    assert result.exit_code == 0, result.output
    provider = json.loads((tmp_path / "agents" / "opencode" / "opencode.json").read_text())["provider"][
        opencode_agent._OPENCODE_PROVIDER
    ]
    assert provider["options"]["headers"] == {"Authorization": "Bearer gateway-token"}
    assert "apiKey" not in provider["options"]


def test_opencode_v1_reads_the_effort_as_reasoning_effort_option(
    fake_vllm, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        start.start_app, ["opencode", "--no-launch", "--reasoning-effort", "low"]
    )
    assert result.exit_code == 0, result.output
    provider = json.loads((tmp_path / "agents" / "opencode" / "opencode.json").read_text())
    provider = provider["provider"][opencode_agent._OPENCODE_PROVIDER]
    assert provider["models"][MODEL["id"]]["options"] == {"reasoningEffort": "low"}
    assert provider["options"]["body"] == {"reasoning_effort": "low"}


# ── OpenClaw (Anthropic /v1/messages) ────────────────────────────────

# ── OpenCode (OpenAI /v1/chat/completions) ───────────────────────────
def test_write_opencode_config_fresh(tmp_path):
    path = tmp_path / "opencode.json"
    opencode_agent.write_opencode_config(BASE, "sk-test-abc", MODEL, path)
    config = json.loads(path.read_text())
    provider = config["provider"][opencode_agent._OPENCODE_PROVIDER]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"] == {"baseURL": f"{BASE}/v1", "apiKey": "sk-test-abc"}
    # Context limit must be declared, or OpenCode treats it as 0 and disables compaction.
    assert provider["models"] == {
        MODEL["id"]: {
            "name": MODEL["id"],
            "limit": {"context": 131072, "input": 131072, "output": 32_000},
        }
    }
    assert config["model"] == f"{opencode_agent._OPENCODE_PROVIDER}/{MODEL['id']}"
    # Provider filters belong to the launch-time inline overlay, not this config writer.
    assert "disabled_providers" not in config
    # Compaction buffer scaled to ~10% of the window (compact near 90%).
    assert config["compaction"] == {"auto": True, "reserved": 131072 // 10}


@pytest.mark.parametrize(
    "window, max_tokens, expected",
    [
        (16_384, None, 4_096),
        (32_768, None, 8_192),
        # No longer pinned at 8,192 (#12009).
        (131_072, None, 32_000),
        (143_616, None, 32_000),
        (143_616, 65_536, 65_536),
        (143_616, 200_000, 71_808),
        (32_768, 4_000, 4_000),
    ],
)
def test_opencode_output_limit(window, max_tokens, expected):
    assert core_options.opencode_output_limit(window, max_tokens) == expected


def test_opencode_max_tokens_sets_limit_and_raises_opencode_ceiling(fake_vllm, tmp_path):
    result = CliRunner().invoke(
        start.start_app, ["opencode", "--no-launch", "--max-tokens", "65536"]
    )
    assert result.exit_code == 0, result.output
    config_path = tmp_path / "agents" / "opencode" / "opencode.json"
    config = json.loads(config_path.read_text())
    limit = config["provider"][opencode_agent._OPENCODE_PROVIDER]["models"][MODEL["id"]]["limit"]
    assert limit == {"context": 131072, "input": 131072, "output": 65536}
    _assert_env_set(result.output, "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX", "65536")


def test_opencode_limit_input_keeps_compaction_off_the_output_limit(tmp_path):
    # Without input, OpenCode compacts at context - output and a 65,536 limit compacts at half full.
    path = tmp_path / "opencode.json"
    opencode_agent.write_opencode_config(BASE, "sk-test-abc", MODEL, path, max_tokens = 65536)
    config = json.loads(path.read_text())
    limit = config["provider"][opencode_agent._OPENCODE_PROVIDER]["models"][MODEL["id"]]["limit"]
    assert limit["input"] == limit["context"] == 131072
    assert config["compaction"]["reserved"] == 131072 // 10


@pytest.mark.parametrize(
    "window, expected_reserved, expected_compacts_at",
    [
        (16_384, 4_096, 12_288),
        (32_768, 8_192, 24_576),
        (131_072, 13_107, 117_965),
        (262_144, 26_214, 235_930),
    ],
)
def test_opencode_compaction_reserved(window, expected_reserved, expected_compacts_at):
    reserved = opencode_agent.opencode_compaction_reserved(window, core_options.opencode_output_limit(window))
    assert reserved == expected_reserved
    assert window - reserved == expected_compacts_at


def test_opencode_subagent_drops_the_compaction_a_normal_session_wrote(tmp_path):
    path = tmp_path / "opencode.json"
    small = {**MODEL, "context_length": 16_384}
    opencode_agent.write_opencode_config(BASE, "sk-test-abc", small, path)
    assert json.loads(path.read_text())["compaction"] == {"auto": True, "reserved": 4_096}
    opencode_agent.write_opencode_config(BASE, "sk-test-abc", small, path, as_subagent = True)
    assert "compaction" not in json.loads(path.read_text())


def test_opencode_max_tokens_without_a_window_warns(capsys):
    assert opencode_agent._opencode_output_env({"id": "m"}, 65536) == {}
    assert "--max-tokens is ignored" in capsys.readouterr().err


def test_opencode_max_tokens_under_ceiling_leaves_opencode_env_alone(fake_vllm, tmp_path):
    result = CliRunner().invoke(
        start.start_app, ["opencode", "--no-launch", "--max-tokens", "16000"]
    )
    assert result.exit_code == 0, result.output
    config = json.loads((tmp_path / "agents" / "opencode" / "opencode.json").read_text())
    limit = config["provider"][opencode_agent._OPENCODE_PROVIDER]["models"][MODEL["id"]]["limit"]
    assert limit["output"] == 16000
    assert "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX" not in result.output


def test_opencode_max_tokens_raises_a_smaller_inherited_ceiling(fake_vllm, monkeypatch):
    monkeypatch.setenv("OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX", "8000")
    result = CliRunner().invoke(
        start.start_app, ["opencode", "--no-launch", "--max-tokens", "16000"]
    )
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX", "16000")


def test_opencode_max_tokens_recipe_keeps_a_larger_inherited_ceiling(fake_vllm, monkeypatch):
    # The --no-launch recipe must carry the ceiling, or a shell without the export reverts to 32,000.
    monkeypatch.setenv("OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX", "100000")
    result = CliRunner().invoke(
        start.start_app, ["opencode", "--no-launch", "--max-tokens", "65536"]
    )
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX", "100000")


def test_opencode_max_tokens_past_half_the_window_is_capped(fake_vllm, tmp_path):
    result = CliRunner().invoke(
        start.start_app, ["opencode", "--no-launch", "--max-tokens", "120000"]
    )
    assert result.exit_code == 0, result.output
    assert "leaves too little" in result.output
    config = json.loads((tmp_path / "agents" / "opencode" / "opencode.json").read_text())
    limit = config["provider"][opencode_agent._OPENCODE_PROVIDER]["models"][MODEL["id"]]["limit"]
    assert limit["output"] == 131072 // 2


def test_write_opencode_config_preserves_and_idempotent(tmp_path):
    path = tmp_path / "opencode.json"
    path.write_text(
        json.dumps(
            {
                "theme": "tokyonight",
                "disabled_providers": ["ollama", "agent-switch"],
                "provider": {"anthropic": {"name": "Anthropic"}},
            }
        )
    )
    opencode_agent.write_opencode_config(BASE, "sk-test-abc", MODEL, path)
    config = json.loads(path.read_text())
    assert config["theme"] == "tokyonight"
    # The overlay no longer edits disabled_providers; re-enabling agent-switch is done in
    # the inline layer, so an existing list here is preserved untouched.
    assert config["disabled_providers"] == ["ollama", "agent-switch"]
    assert config["provider"]["anthropic"]["name"] == "Anthropic"
    assert config["provider"][opencode_agent._OPENCODE_PROVIDER]["options"]["baseURL"] == f"{BASE}/v1"
    before = path.read_text()
    opencode_agent.write_opencode_config(BASE, "sk-test-abc", MODEL, path)
    assert path.read_text() == before


def test_write_opencode_config_keeps_foreign_disabled_providers(tmp_path):
    # A user who disabled other providers (but not ours) must keep them disabled:
    # the overlay must not rewrite disabled_providers, or those providers get silently
    # re-enabled for the session.
    path = tmp_path / "opencode.json"
    path.write_text(json.dumps({"disabled_providers": ["openai", "gemini"]}))
    opencode_agent.write_opencode_config(BASE, "sk-test-abc", MODEL, path)
    config = json.loads(path.read_text())
    assert config["disabled_providers"] == ["openai", "gemini"]


def test_write_opencode_config_as_subagent_preserves_parent_model(tmp_path):
    path = tmp_path / "opencode.json"
    path.write_text(
        json.dumps(
            {
                "model": "anthropic/claude-sonnet-4-5",
                "small_model": "anthropic/claude-haiku-4-5",
                "compaction": {"auto": False},
            }
        )
    )
    local = {**MODEL, "id": MODEL["id"] + ":UD-Q4_K_XL"}
    opencode_agent.write_opencode_config(
        BASE,
        "sk-test-abc",
        local,
        path,
        as_subagent = True,
    )
    config = json.loads(path.read_text())
    assert config["model"] == "anthropic/claude-sonnet-4-5"
    assert config["small_model"] == "anthropic/claude-haiku-4-5"
    assert config["compaction"] == {"auto": False}
    agent = config["agent"]["local"]
    assert agent["mode"] == "subagent"
    assert agent["model"] == f"{opencode_agent._OPENCODE_PROVIDER}/{local['id']}"
    assert "local agent" in agent["description"].lower()
    assert local["id"] in config["provider"][opencode_agent._OPENCODE_PROVIDER]["models"]


def test_opencode_subagent_inline_keeps_parent_provider_filters(monkeypatch, tmp_path):
    config_path = tmp_path / "opencode.json"
    inherited = {
        "theme": "tokyonight",
        "enabled_providers": ["anthropic"],
    }
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", json.dumps(inherited))
    set_start_attr(monkeypatch, "_which_with_install_dirs", lambda _: "/usr/bin/opencode")
    set_start_attr(monkeypatch, "_wsl_windows_executable", lambda _: None)
    captured = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(
            returncode = 0,
            stdout = json.dumps(
                {
                    "enabled_providers": ["opencode-go"],
                    "disabled_providers": ["ollama", opencode_agent._OPENCODE_PROVIDER],
                    "subagent_depth": 0,
                }
            ),
            stderr = "",
        )

    monkeypatch.setattr(subprocess, "run", run)
    permission = {"edit": "allow"}
    inline = opencode_agent._opencode_subagent_inline_config(config_path, permission)

    assert captured["command"] == ["/usr/bin/opencode", "debug", "config"]
    assert captured["env"]["OPENCODE_CONFIG"] == str(config_path)
    assert inline == {
        "theme": "tokyonight",
        "enabled_providers": [
            "anthropic",
            "opencode-go",
            opencode_agent._OPENCODE_PROVIDER,
        ],
        "disabled_providers": ["ollama"],
        "subagent_depth": 1,
        "permission": permission,
    }


def test_opencode_subagent_inline_preserves_positive_depth(monkeypatch, tmp_path):
    set_start_attr(monkeypatch, "_which_with_install_dirs", lambda _: "/usr/bin/opencode")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode = 0,
            stdout = json.dumps({"subagent_depth": 3}),
            stderr = "",
        ),
    )

    inline = opencode_agent._opencode_subagent_inline_config(tmp_path / "opencode.json", {})

    assert inline["subagent_depth"] == 3


def test_opencode_subagent_inline_merges_inherited_filters_without_binary(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "OPENCODE_CONFIG_CONTENT",
        json.dumps(
            {
                "enabled_providers": ["opencode-go"],
                "disabled_providers": ["ollama", opencode_agent._OPENCODE_PROVIDER],
            }
        ),
    )
    set_start_attr(monkeypatch, "_which_with_install_dirs", lambda _: None)

    inline = opencode_agent._opencode_subagent_inline_config(tmp_path / "opencode.json", {})

    assert inline["enabled_providers"] == ["opencode-go", opencode_agent._OPENCODE_PROVIDER]
    assert inline["disabled_providers"] == ["ollama"]
    assert inline["subagent_depth"] == 1


def test_opencode_v2_subagent_uses_native_depth_without_debug_probe(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "OPENCODE_CONFIG_CONTENT",
        json.dumps({"enabled_providers": ["anthropic"], "subagent_depth": 2}),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("V2 debug config is not a resolved object"),
    )

    inline = opencode_agent._opencode_subagent_inline_config(
        tmp_path / "opencode.json", {}, command = "opencode2", v2 = True
    )

    assert "subagent_depth" not in inline
    assert inline["enabled_providers"] == ["anthropic", opencode_agent._OPENCODE_PROVIDER]
    assert inline["experimental"] == {"subagent_depth": 2}


def test_opencode_v2_subagent_does_not_override_configured_depth(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENCODE_CONFIG_CONTENT", raising = False)

    inline = opencode_agent._opencode_subagent_inline_config(
        tmp_path / "opencode.json", {}, command = "opencode2", v2 = True
    )

    assert "experimental" not in inline


def test_opencode_inline_scopes_session_to_our_provider(fake_vllm):
    # opencode filters even config-defined providers through enabled/disabled_providers,
    # and a model pin does not bypass that gate. The inline overlay (session-only, highest
    # layer, arrays replace) allowlists our provider and clears the denylist so the local
    # model always loads regardless of the user's config, without reading or editing it.
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])
    assert result.exit_code == 0, result.output
    inline = _opencode_inline_config(result.output)
    assert inline["enabled_providers"] == [opencode_agent._OPENCODE_PROVIDER]
    assert inline["disabled_providers"] == []
    assert inline["model"] == f"{opencode_agent._OPENCODE_PROVIDER}/{MODEL['id']}"
    # small_model stays on the enabled provider too, so lightweight tasks do not resolve a
    # filtered provider mid-session.
    assert inline["small_model"] == f"{opencode_agent._OPENCODE_PROVIDER}/{MODEL['id']}"


def test_opencode_passthrough_flags_omit_model_flag(fake_vllm):
    # Any passthrough (top-level flags that may precede a subcommand, or a subcommand)
    # is left untouched; --model is not injected. The model is pinned by the inline
    # OPENCODE_CONFIG_CONTENT (highest layer) instead, so it is still forced.
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch", "--dir", "repo"])
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    assert command == ["opencode", "--dir", "repo"]
    assert "--model" not in command
    assert (
        _opencode_inline_config(result.output)["model"]
        == f"{opencode_agent._OPENCODE_PROVIDER}/{MODEL['id']}"
    )


def test_opencode_passthrough_subcommand_omits_model_flag(fake_vllm):
    # A passthrough subcommand (e.g. `serve`) takes the model from the pinned config;
    # inserting --model before it would break opencode's arg parsing.
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch", "serve"])
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    assert command[0] == "opencode"
    assert command[1] == "serve"
    assert "--model" not in command


def test_connect_opencode_no_launch(fake_vllm, tmp_path):
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])
    assert result.exit_code == 0, result.output
    assert "opencode" in result.output
    config_path = tmp_path / "agents" / "opencode" / "opencode.json"
    # OPENCODE_CONFIG overlay points at the session file, not the user's global config.
    _assert_env_set(result.output, "OPENCODE_CONFIG", str(config_path))
    inline_config = _opencode_inline_config(result.output)
    config = json.loads(config_path.read_text())
    provider = config["provider"][opencode_agent._OPENCODE_PROVIDER]
    assert provider["options"]["apiKey"] == KEY
    assert config["model"] == f"{opencode_agent._OPENCODE_PROVIDER}/{MODEL['id']}"
    # The session config file (a throwaway overlay, not the user's real config) does not
    # carry provider filters; the session scoping rides in the inline env layer only.
    assert "disabled_providers" not in config
    assert "enabled_providers" not in config
    assert inline_config == {
        "model": f"{opencode_agent._OPENCODE_PROVIDER}/{MODEL['id']}",
        "small_model": f"{opencode_agent._OPENCODE_PROVIDER}/{MODEL['id']}",
        "enabled_providers": [opencode_agent._OPENCODE_PROVIDER],
        "disabled_providers": [],
    }
    # --no-launch prints an append-safe base command (no --model before a subcommand a
    # driver may append); the model is forced by the inline pin above.
    assert _launch_command(result.output) == ["opencode"]


def test_connect_opencode_v2_no_launch_uses_private_server(fake_vllm, monkeypatch):
    set_start_attr(monkeypatch, "_opencode_command", lambda *_: ("opencode2", True))

    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])

    assert result.exit_code == 0, result.output
    assert _launch_command(result.output) == ["opencode2", "--standalone"]
    assert _opencode_inline_config(result.output)["model"] == (
        f"{opencode_agent._OPENCODE_PROVIDER}/{MODEL['id']}"
    )
    assert "enabled_providers" not in _opencode_inline_config(result.output)
    assert "disabled_providers" not in _opencode_inline_config(result.output)
    assert f"provider policies must allow '{opencode_agent._OPENCODE_PROVIDER}'" in result.output


def test_connect_opencode_v2_no_launch_uses_resolved_off_path_binary(
    fake_vllm, monkeypatch, tmp_path
):
    binary = tmp_path / ".opencode" / "bin" / "opencode2"
    set_start_attr(monkeypatch, "_opencode_command",
        lambda: (str(binary), True),
    )

    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])

    assert result.exit_code == 0, result.output
    assert _launch_command(result.output) == [str(binary), "--standalone"]


def test_connect_opencode_v2_models_uses_private_server(fake_vllm, monkeypatch):
    set_start_attr(monkeypatch, "_opencode_command", lambda *_: ("opencode2", True))

    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch", "models"])

    assert result.exit_code == 0, result.output
    assert _launch_command(result.output) == ["opencode2", "models", "--standalone"]


def test_connect_opencode_as_subagent_preserves_cloud_parent(fake_vllm, tmp_path, monkeypatch):
    set_start_attr(monkeypatch, "_opencode_subagent_inline_config", lambda path, permission, **kwargs: {}
    )
    result = CliRunner().invoke(
        start.start_app,
        [
            "opencode",
            "--as-subagent",
            "--no-launch",
            "--model",
            MODEL["id"],
        ],
    )
    assert result.exit_code == 0, result.output
    assert _launch_command(result.output) == ["opencode"]
    expected_model = f"{opencode_agent._OPENCODE_PROVIDER}/{MODEL['id']}"
    # The agent rides in the inline overlay; nothing else comes from the empty base.
    assert _opencode_inline_config(result.output) == {
        "agent": {
            "local": {
                "description": core_options._SUBAGENT_DESCRIPTION,
                "mode": "subagent",
                "model": expected_model,
                "prompt": core_options._SUBAGENT_INSTRUCTIONS,
            }
        }
    }
    path = tmp_path / "agents" / "opencode-subagent" / "opencode.json"
    config = json.loads(path.read_text())
    assert "model" not in config
    assert "small_model" not in config
    assert "compaction" not in config
    agent = config["agent"]["local"]
    assert agent["model"] == expected_model
    assert "The local model is available as @local and in /models." in result.output


def test_opencode_subagent_installs_binary_before_filter_inspection(fake_vllm, monkeypatch):
    # The effective-config inspection needs the opencode binary; a first launch must
    # offer the install before building the overlay, or a global allowlist read only
    # after _launch installs OpenCode would filter out the new provider.
    installed = {}
    set_start_attr(monkeypatch, "_which_with_install_dirs",
        lambda name: "/usr/local/bin/opencode" if installed.get("done") else None,
    )

    def require(name, hint, launch):
        assert launch is True
        installed["done"] = True
        installed["name"] = name

    set_start_attr(monkeypatch, "_require_agent_for_launch", require)
    inspected = {}

    def inline(path, permission, **kwargs):
        inspected["binary"] = core_install._which_with_install_dirs("opencode")
        return {}

    set_start_attr(monkeypatch, "_opencode_subagent_inline_config", inline)
    set_start_attr(monkeypatch, "_run", lambda *a, **k: None)

    result = CliRunner().invoke(start.start_app, ["opencode", "--as-subagent"])

    assert result.exit_code == 0, result.output
    assert installed["name"] == "opencode"
    assert inspected["binary"] == "/usr/local/bin/opencode"


def test_opencode_subagent_pins_agent_in_inline_overlay(fake_vllm, monkeypatch):
    # A project opencode.json outranks the session file, so the agent must ride in
    # OPENCODE_CONFIG_CONTENT where a repo's own agent.local cannot field-merge over it.
    set_start_attr(monkeypatch, "_opencode_subagent_inline_config", lambda path, permission, **kwargs: {}
    )
    result = CliRunner().invoke(
        start.start_app,
        ["opencode", "--as-subagent", "--no-launch", "--model", MODEL["id"]],
    )
    assert result.exit_code == 0, result.output
    agent = _opencode_inline_config(result.output)["agent"]["local"]
    assert agent["mode"] == "subagent"
    assert agent["model"] == f"{opencode_agent._OPENCODE_PROVIDER}/{MODEL['id']}"
    assert agent["prompt"] == core_options._SUBAGENT_INSTRUCTIONS
    assert agent["description"] == core_options._SUBAGENT_DESCRIPTION


def test_connect_opencode_subagent_yolo_no_launch_stays_append_safe(fake_vllm, monkeypatch):
    set_start_attr(monkeypatch, "_opencode_supports_native_auto", lambda *_: True)
    captured = {}

    def inline(path, permission, **kwargs):
        captured["permission"] = permission
        return {"permission": permission}

    set_start_attr(monkeypatch, "_opencode_subagent_inline_config", inline)
    result = CliRunner().invoke(
        start.start_app,
        ["opencode", "--as-subagent", "--no-launch", "--yolo"],
    )

    assert result.exit_code == 0, result.output
    assert _launch_command(result.output) == ["opencode"]
    assert "--auto" not in result.output
    assert captured["permission"] == {
        "edit": "allow",
        "bash": "allow",
        "webfetch": "allow",
        "task": "allow",
        "external_directory": {"*": "allow"},
    }
    assert _opencode_inline_config(result.output)["permission"] == captured["permission"]


def test_yolo_opencode_bare_no_launch_uses_permission_fallback(fake_vllm, tmp_path):
    # A bare --no-launch recipe stays append-safe (callers add a subcommand later);
    # `opencode --auto run ...` would select the TUI, not `run`, so keep the config fallback.
    result = CliRunner().invoke(start.start_app, ["opencode", "--yolo", "--no-launch"])
    assert result.exit_code == 0, result.output
    config = json.loads((tmp_path / "agents" / "opencode" / "opencode.json").read_text())
    assert config["permission"] == {
        "edit": "allow",
        "bash": "allow",
        "webfetch": "allow",
        "external_directory": {"*": "allow"},
    }


def test_yolo_opencode_run_uses_native_auto(fake_vllm):
    result = CliRunner().invoke(
        start.start_app,
        ["opencode", "--yolo", "--no-launch", "run", "hello"],
    )
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    assert command == ["opencode", "run", "hello", "--auto"]
    assert "permission" not in _opencode_inline_config(result.output)


def test_yolo_opencode_v2_run_uses_standalone_and_native_auto(fake_vllm, monkeypatch):
    set_start_attr(monkeypatch, "_opencode_command", lambda *_: ("opencode2", True))

    result = CliRunner().invoke(
        start.start_app,
        ["opencode", "--yolo", "--no-launch", "run", "hello"],
    )

    assert result.exit_code == 0, result.output
    assert _launch_command(result.output) == [
        "opencode2",
        "run",
        "--standalone",
        "hello",
        "--auto",
    ]
    assert "permission" not in _opencode_inline_config(result.output)


def test_yolo_opencode_v2_mini_uses_permission_fallback(fake_vllm, monkeypatch):
    set_start_attr(monkeypatch, "_opencode_command", lambda *_: ("opencode2", True))

    result = CliRunner().invoke(
        start.start_app,
        ["opencode", "--yolo", "--no-launch", "mini"],
    )

    assert result.exit_code == 0, result.output
    assert _launch_command(result.output) == ["opencode2", "mini", "--standalone"]
    assert _opencode_inline_config(result.output)["permission"]["edit"] == "allow"


def test_yolo_opencode_tui_resume_uses_native_auto(fake_vllm):
    result = CliRunner().invoke(
        start.start_app,
        ["opencode", "--yolo", "--no-launch", "--session", "sid"],
    )
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    assert command == ["opencode", "--session", "sid", "--auto"]
    assert "permission" not in _opencode_inline_config(result.output)


def test_no_yolo_opencode_run_omits_native_auto(fake_vllm):
    result = CliRunner().invoke(
        start.start_app,
        ["opencode", "--no-launch", "run", "hello"],
    )
    assert result.exit_code == 0, result.output
    assert _launch_command(result.output) == ["opencode", "run", "hello"]
    assert "permission" not in _opencode_inline_config(result.output)


def test_yolo_opencode_bare_launch_uses_native_auto(fake_vllm, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/local/bin/opencode")
    set_start_attr(monkeypatch, "_opencode_supports_native_auto", lambda *_: True)
    captured = _capture_launch(monkeypatch, ["opencode", "--yolo"])
    assert captured["command"][1:] == [
        "--model",
        f"{opencode_agent._OPENCODE_PROVIDER}/{MODEL['id']}",
        "--auto",
    ]
    assert "permission" not in json.loads(captured["env"]["OPENCODE_CONFIG_CONTENT"])


def test_yolo_opencode_v2_bare_launch_omits_root_model(fake_vllm, monkeypatch):
    set_start_attr(monkeypatch, "_opencode_command", lambda *_: ("opencode2", True))
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/local/bin/opencode2")
    captured = _capture_launch(monkeypatch, ["opencode", "--yolo"])

    assert captured["command"][0].endswith("opencode2")
    assert captured["command"][1:] == ["--standalone", "--auto"]
    assert "--model" not in captured["command"]


def test_yolo_opencode_native_auto_clears_prior_config_fallback(fake_vllm, tmp_path):
    fallback = CliRunner().invoke(
        start.start_app,
        ["opencode", "--yolo", "--no-launch"],
    )
    assert fallback.exit_code == 0, fallback.output

    native = CliRunner().invoke(
        start.start_app,
        ["opencode", "--yolo", "--no-launch", "run", "hello"],
    )
    assert native.exit_code == 0, native.output
    assert _launch_command(native.output) == ["opencode", "run", "hello", "--auto"]
    assert "permission" not in _opencode_inline_config(native.output)
    config = json.loads((tmp_path / "agents" / "opencode" / "opencode.json").read_text())
    assert config["permission"] == {
        "edit": "ask",
        "bash": "ask",
        "webfetch": "ask",
        "external_directory": {"*": "ask"},
    }


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("1.17.11", False),
        ("1.17.12", True),
        ("opencode 1.18.2", True),
        ("development build", False),
    ],
)
def test_opencode_native_auto_version_gate(monkeypatch, version, expected):
    monkeypatch.setattr(shutil, "which", lambda name, path = None: "/usr/local/bin/opencode")
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: version)
    assert opencode_agent._opencode_supports_native_auto() is expected


def test_opencode_native_auto_assumes_current_without_local_binary(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name, path = None: None)
    assert opencode_agent._opencode_supports_native_auto() is True


def test_yolo_opencode_old_version_uses_config_fallback(fake_vllm, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name, path = None: "/usr/local/bin/opencode")
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "1.17.11")
    result = CliRunner().invoke(
        start.start_app,
        ["opencode", "--yolo", "--no-launch", "run", "hello"],
    )
    assert result.exit_code == 0, result.output
    assert _launch_command(result.output) == ["opencode", "run", "hello"]
    assert _opencode_inline_config(result.output)["permission"] == {
        "edit": "allow",
        "bash": "allow",
        "webfetch": "allow",
        "external_directory": {"*": "allow"},
    }


@pytest.mark.parametrize(
    ("args", "expected", "native"),
    [
        ([], ["--auto"], True),
        (["run", "hello"], ["run", "hello", "--auto"], True),
        (
            ["run", "hello", "--", "--literal"],
            ["run", "hello", "--auto", "--", "--literal"],
            True,
        ),
        (["--print-logs", "run", "hello"], ["--print-logs", "run", "hello", "--auto"], True),
        (["--session", "serve"], ["--session", "serve", "--auto"], True),
        (["serve"], ["serve"], False),
        (["--print-logs", "serve"], ["--print-logs", "serve"], False),
        (["run", "--auto", "hello"], ["run", "--auto", "hello"], True),
        # Hidden commands that reject --auto fall back like the visible utility ones.
        (["generate"], ["generate"], False),
        (["console", "login"], ["console", "login"], False),
        # --mini ignores --auto (runMini forces auto=false), so use the config fallback.
        (["--mini"], ["--mini"], False),
        (["--session", "sid", "--mini"], ["--session", "sid", "--mini"], False),
    ],
)
def test_opencode_native_auto_args(args, expected, native):
    assert opencode_agent._opencode_native_auto_args(args, True) == (expected, native)
    assert opencode_agent._opencode_native_auto_args(args, False) == (args, False)


def test_yolo_opencode_non_agent_subcommand_uses_config_fallback(fake_vllm):
    result = CliRunner().invoke(
        start.start_app,
        ["opencode", "--yolo", "--no-launch", "serve"],
    )
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    assert command == ["opencode", "serve"]
    assert _opencode_inline_config(result.output)["permission"] == {
        "edit": "allow",
        "bash": "allow",
        "webfetch": "allow",
        "external_directory": {"*": "allow"},
    }


@pytest.mark.parametrize("passthrough", (["generate"], ["console", "login"], ["--mini"]))
def test_yolo_opencode_no_auto_command_uses_config_fallback(fake_vllm, passthrough):
    # generate/console are hidden and reject --auto, --mini ignores it: none get --auto,
    # all keep the config permission fallback.
    result = CliRunner().invoke(
        start.start_app,
        ["opencode", "--yolo", "--no-launch", *passthrough],
    )
    assert result.exit_code == 0, result.output
    assert _launch_command(result.output) == ["opencode", *passthrough]
    assert _opencode_inline_config(result.output)["permission"] == {
        "edit": "allow",
        "bash": "allow",
        "webfetch": "allow",
        "external_directory": {"*": "allow"},
    }


def test_no_yolo_opencode_has_no_permission_block(fake_vllm, tmp_path):
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])
    assert result.exit_code == 0, result.output
    config = json.loads((tmp_path / "agents" / "opencode" / "opencode.json").read_text())
    # A non-yolo run on a fresh config writes no permission block; it only flips a prior
    # --yolo run's explicit allow back to ask (see the yolo-then-plain test below).
    assert "permission" not in config


def test_no_yolo_opencode_flips_prior_yolo_allow_to_ask(fake_vllm, tmp_path):
    # The core reset: a --yolo run wrote explicit per-tool allow; a later non-yolo run
    # must flip exactly those back to ask so nothing stays auto-approved.
    yolo = CliRunner().invoke(start.start_app, ["opencode", "--yolo", "--no-launch"])
    assert yolo.exit_code == 0, yolo.output
    config_path = tmp_path / "agents" / "opencode" / "opencode.json"
    assert json.loads(config_path.read_text())["permission"] == {
        "edit": "allow",
        "bash": "allow",
        "webfetch": "allow",
        "external_directory": {"*": "allow"},
    }
    plain = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])
    assert plain.exit_code == 0, plain.output
    assert json.loads(config_path.read_text())["permission"] == {
        "edit": "ask",
        "bash": "ask",
        "webfetch": "ask",
        "external_directory": {"*": "ask"},
    }


def test_write_opencode_config_yolo_unit(tmp_path):
    path = tmp_path / "opencode.json"
    opencode_agent.write_opencode_config(BASE, "sk-test-abc", MODEL, path, yolo = True)
    config = json.loads(path.read_text())
    assert config["permission"] == {
        "edit": "allow",
        "bash": "allow",
        "webfetch": "allow",
        "external_directory": {"*": "allow"},
    }


def test_no_launch_rerun_clears_stale_opencode_yolo_permissions(fake_vllm, tmp_path):
    # The no-launch config dir is reused across runs, so a --yolo run persists its
    # auto-approve settings; a later run without --yolo must strip them, not leave
    # tool execution silently pre-approved.
    yolo = CliRunner().invoke(start.start_app, ["opencode", "--yolo", "--no-launch"])
    assert yolo.exit_code == 0, yolo.output
    config_path = tmp_path / "agents" / "opencode" / "opencode.json"
    assert "permission" in json.loads(config_path.read_text())
    plain = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])
    assert plain.exit_code == 0, plain.output
    config = json.loads(config_path.read_text())
    # The yolo allow policy is replaced by a prompting one, not deleted (which would
    # revert to OpenCode's permissive "allow" default).
    assert config["permission"] == {
        "edit": "ask",
        "bash": "ask",
        "webfetch": "ask",
        "external_directory": {"*": "ask"},
    }
    # The session provider survives the cleanup.
    assert opencode_agent._OPENCODE_PROVIDER in config["provider"]


def test_write_opencode_config_yolo_then_plain_unit(tmp_path):
    path = tmp_path / "opencode.json"
    opencode_agent.write_opencode_config(BASE, "sk-test-abc", MODEL, path, yolo = True)
    opencode_agent.write_opencode_config(BASE, "sk-test-abc", MODEL, path, yolo = False)
    config = json.loads(path.read_text())
    # A plain rerun replaces the yolo allow policy with a prompting one.
    assert config["permission"] == {
        "edit": "ask",
        "bash": "ask",
        "webfetch": "ask",
        "external_directory": {"*": "ask"},
    }


def test_opencode_non_yolo_flips_only_explicit_allow(tmp_path):
    # Only a tool explicitly set to "allow" (what --yolo writes) is flipped to "ask". A
    # deny/ask a user set is kept, and an absent tool is not added.
    path = tmp_path / "opencode.json"
    path.write_text(json.dumps({"permission": {"edit": "allow", "bash": "deny", "read": "ask"}}))
    session = opencode_agent.write_opencode_config(BASE, "sk-test-abc", MODEL, path, yolo = False)
    config = json.loads(path.read_text())
    assert config["permission"] == {"edit": "ask", "bash": "deny", "read": "ask"}
    assert session == {}  # a non-yolo session carries no permission inline


def test_opencode_subagent_non_yolo_clears_yolo_task_permission(tmp_path):
    path = tmp_path / "opencode.json"
    opencode_agent.write_opencode_config(
        BASE,
        "sk-test-abc",
        MODEL,
        path,
        yolo = True,
        as_subagent = True,
    )
    opencode_agent.write_opencode_config(
        BASE,
        "sk-test-abc",
        MODEL,
        path,
        as_subagent = True,
    )

    assert json.loads(path.read_text())["permission"]["task"] == "ask"


def test_opencode_non_yolo_leaves_string_permission(tmp_path):
    # A global string rule ("deny") is a user-managed catch-all; leave it untouched and
    # carry no inline override.
    path = tmp_path / "opencode.json"
    path.write_text(json.dumps({"permission": "deny"}))
    session = opencode_agent.write_opencode_config(BASE, "sk-test-abc", MODEL, path, yolo = False)
    assert json.loads(path.read_text())["permission"] == "deny"
    assert session == {}


def test_opencode_non_yolo_leaves_catch_all_and_flips_explicit_allow(tmp_path):
    # A "*" catch-all is the user's own rule, never something --yolo writes (yolo sets
    # explicit per-tool allow), so it is left intact; an explicit per-tool "allow" is still
    # flipped to "ask", but an absent tool inheriting the catch-all is not touched.
    path = tmp_path / "opencode.json"
    path.write_text(json.dumps({"permission": {"*": "allow", "bash": "allow"}}))
    session = opencode_agent.write_opencode_config(BASE, "sk-test-abc", MODEL, path, yolo = False)
    assert json.loads(path.read_text())["permission"] == {"*": "allow", "bash": "ask"}
    assert session == {}


def test_opencode_non_yolo_leaves_granular_object(tmp_path):
    # A granular object value is a user rule (yolo only ever writes a plain "allow" string),
    # so it is left in the file verbatim and never carried inline.
    path = tmp_path / "opencode.json"
    obj = {"read *": "deny", "git *": "ask"}
    path.write_text(json.dumps({"permission": {"bash": dict(obj)}}))
    session = opencode_agent.write_opencode_config(BASE, "sk-test-abc", MODEL, path, yolo = False)
    assert json.loads(path.read_text())["permission"]["bash"] == obj
    assert session == {}


def test_resume_opencode_config_in_stable_dir(fake_vllm, tmp_path, monkeypatch):
    # opencode's session data lives in ~/.local/share/opencode (never relocated), so
    # resume already survives exit; --persist also stabilizes its config overlay dir.
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/local/bin/opencode")
    captured = _capture_launch(monkeypatch, ["opencode", "--persist"])
    stable = tmp_path / "agents" / "opencode"
    assert captured["env"]["OPENCODE_CONFIG"] == str(stable / "opencode.json")
    assert stable.exists()


def test_persist_bare_opencode_launch_has_no_resume_token(fake_vllm, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/local/bin/opencode")
    captured = _capture_launch(monkeypatch, ["opencode", "--persist"])
    assert "--continue" not in captured["command"]
    assert captured["command"][1:] == ["--model", f"{opencode_agent._OPENCODE_PROVIDER}/{MODEL['id']}"]


# ── --mcp / --mcp-all: session-only MCP mounting ─────────────────────


def _opencode_user_global_mcp(tmp_path, monkeypatch) -> dict:
    # A global server the session does not mount must be copied and disabled, not deleted.
    global_dir = tmp_path / "xdg" / "opencode"
    global_dir.mkdir(parents = True)
    entry = {"type": "remote", "url": "http://127.0.0.1:9/mcp", "enabled": True}
    (global_dir / "opencode.json").write_text(json.dumps({"mcp": {"user-server": entry}}))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    return entry


def test_connect_opencode_mcp_writes_local_and_remote_entries(fake_vllm, tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "gh-secret")
    user_entry = _opencode_user_global_mcp(tmp_path, monkeypatch)
    _mcp_registry()
    result = CliRunner().invoke(
        start.start_app, ["opencode", "--no-launch", "--mcp", "context7", "--mcp", "github"]
    )
    assert result.exit_code == 0, result.output
    config = json.loads((tmp_path / "agents" / "opencode" / "opencode.json").read_text())
    assert config["mcp"]["context7"] == {
        "type": "local",
        "command": ["npx", "-y", "@upstash/context7-mcp"],
        "environment": {},
        "enabled": True,
    }
    assert config["mcp"]["github"] == {
        "type": "remote",
        "url": "https://api.githubcopilot.com/mcp/",
        "headers": {"Authorization": "Bearer gh-secret"},
        "enabled": True,
    }
    # The unmounted global entry is a full copy with enabled: false (a partial entry can fail
    # the per-layer schema check).
    assert config["mcp"]["user-server"] == {**user_entry, "enabled": False}
    assert "gh-secret" not in result.output


def test_connect_opencode_mcp_all_mounts_every_registry_server(fake_vllm, tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "gh-secret")
    _opencode_user_global_mcp(tmp_path, monkeypatch)
    _mcp_registry()
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch", "--mcp-all"])
    assert result.exit_code == 0, result.output
    config = json.loads((tmp_path / "agents" / "opencode" / "opencode.json").read_text())
    assert {name for name, entry in config["mcp"].items() if entry["enabled"]} == {
        "context7",
        "github",
    }


def test_connect_opencode_without_mcp_flags_has_no_mcp_key(fake_vllm, tmp_path, monkeypatch):
    _opencode_user_global_mcp(tmp_path, monkeypatch)
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])
    assert result.exit_code == 0, result.output
    config = json.loads((tmp_path / "agents" / "opencode" / "opencode.json").read_text())
    assert "mcp" not in config


def test_opencode_mcp_state_cleared_on_rerun_without_flags(fake_vllm, tmp_path, monkeypatch):
    # A --no-launch session config is reused, so the next run must not keep the earlier mount.
    monkeypatch.setenv("GITHUB_TOKEN", "gh-secret")
    _mcp_registry()
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch", "--mcp", "context7"])
    assert result.exit_code == 0, result.output
    config_path = tmp_path / "agents" / "opencode" / "opencode.json"
    assert "mcp" in json.loads(config_path.read_text())
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])
    assert result.exit_code == 0, result.output
    assert "mcp" not in json.loads(config_path.read_text())


def test_opencode_unparseable_global_config_warns_and_continues(fake_vllm, tmp_path, monkeypatch, capsys):
    # Still unparseable after the JSONC handling: warn, keep its servers.
    global_dir = tmp_path / "xdg" / "opencode"
    global_dir.mkdir(parents = True)
    (global_dir / "opencode.json").write_text("{ this is not json")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    _mcp_registry()
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch", "--mcp", "context7"])
    assert result.exit_code == 0, result.output
    assert "couldn't parse" in result.output
    config = json.loads((tmp_path / "agents" / "opencode" / "opencode.json").read_text())
    assert list(config["mcp"]) == ["context7"]


def test_opencode_commented_jsonc_global_mcp_is_disabled(fake_vllm, tmp_path, monkeypatch):
    # .jsonc with comments is OpenCode's documented format: parse it and disable its unmounted servers.
    global_dir = tmp_path / "xdg" / "opencode"
    global_dir.mkdir(parents = True)
    (global_dir / "opencode.jsonc").write_text(
        "{\n  // the user's own server\n  \"mcp\": {\"user-server\": {\"type\": \"remote\", "
        "\"url\": \"http://127.0.0.1:9/mcp\", /* block */ \"enabled\": true}},\n}\n"
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    _mcp_registry()
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch", "--mcp", "context7"])
    assert result.exit_code == 0, result.output
    config = json.loads((tmp_path / "agents" / "opencode" / "opencode.json").read_text())
    assert config["mcp"]["user-server"] == {
        "type": "remote",
        "url": "http://127.0.0.1:9/mcp",
        "enabled": False,
    }


def test_opencode_global_mcp_from_two_files_both_disabled(fake_vllm, tmp_path, monkeypatch):
    # OpenCode loads config.json, opencode.json and opencode.jsonc; every unmounted name is disabled.
    global_dir = tmp_path / "xdg" / "opencode"
    global_dir.mkdir(parents = True)
    (global_dir / "config.json").write_text(
        json.dumps({"mcp": {"old-server": {"type": "local", "command": ["true"]}}})
    )
    (global_dir / "opencode.jsonc").write_text(
        json.dumps({"mcp": {"new-server": {"type": "remote", "url": "http://127.0.0.1:9/mcp"}}})
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    _mcp_registry()
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch", "--mcp", "context7"])
    assert result.exit_code == 0, result.output
    config = json.loads((tmp_path / "agents" / "opencode" / "opencode.json").read_text())
    assert config["mcp"]["old-server"] == {"type": "local", "command": ["true"], "enabled": False}
    assert config["mcp"]["new-server"] == {
        "type": "remote",
        "url": "http://127.0.0.1:9/mcp",
        "enabled": False,
    }


def test_opencode_jsonc_comment_marker_in_string_survives(fake_vllm, tmp_path, monkeypatch):
    # Comment markers inside a string value must not be stripped as comments.
    global_dir = tmp_path / "xdg" / "opencode"
    global_dir.mkdir(parents = True)
    (global_dir / "opencode.jsonc").write_text(
        '{"mcp": {"odd-server": {"type": "remote", "url": "http://x//y/*z*/mcp"}}} // trailing'
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    _mcp_registry()
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch", "--mcp", "context7"])
    assert result.exit_code == 0, result.output
    config = json.loads((tmp_path / "agents" / "opencode" / "opencode.json").read_text())
    assert config["mcp"]["odd-server"] == {
        "type": "remote",
        "url": "http://x//y/*z*/mcp",
        "enabled": False,
    }


def test_opencode_mcp_with_as_subagent_fails(fake_vllm):
    _mcp_registry()
    result = CliRunner().invoke(
        start.start_app, ["opencode", "--as-subagent", "--no-launch", "--mcp", "context7"]
    )
    assert result.exit_code == 1
    assert "--mcp/--mcp-all cannot be combined with --as-subagent" in result.output
