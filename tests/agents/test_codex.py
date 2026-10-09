# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Ported from unsloth_cli/tests/test_start.py (unsloth commit 8e11ba15e); see NOTICE.md.

"""`agent-switch codex`: config and profile writing, overlays and subagent wiring."""

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import agent_switch.start as start
from agent_switch.agents import (
    codex as codex_agent,
)
from agent_switch.core import (
    session as core_session,
)
from tests.cli_support import (
    BASE,
    KEY,
    MODEL,
    _SESSION_FLAGS,
    _assert_env_set,
    _assert_env_unset,
    _capture_launch,
    _launch_command,
    _parse_toml,
    _path_aware_which,
)
from tests.start_split import set_start_attr


def _assert_env_kept(output: str, name: str) -> None:
    needle = f"Remove-Item Env:{name}" if os.name == "nt" else f"unset {name}"
    assert needle not in output, f"{needle!r} unexpectedly found in:\n{output}"


def test_codex_catalog_probes_old_codex_only_in_install_dir(monkeypatch, tmp_path):
    # Same ordering fix for codex: an old codex present only in an install dir is detected so the
    # model-catalog config is omitted (the old binary can't consume it).
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents = True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))
    monkeypatch.setattr(shutil, "which", _path_aware_which({"codex": local_bin}))
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "codex-cli 0.109.0")
    assert codex_agent._codex_supports_model_catalog() is False


def test_merge_codex_config_fresh():
    merged = codex_agent._merge_codex_config("", BASE)
    parsed = _parse_toml(merged)
    assert parsed["oss_provider"] == "agent_switch"
    provider = parsed["model_providers"]["agent_switch"]
    assert provider["base_url"] == f"{BASE}/v1"
    assert provider["wire_api"] == "responses"
    assert provider["requires_openai_auth"] is False


def test_merge_codex_config_raises_the_stream_idle_timeout():
    """Codex's 300s default cancels the stream while llama-server is still reading.

    llama-server emits nothing at all during prompt processing, so the whole wait counts
    as idle. Measured on a 2-core CI host: 16.1 tok/s against Codex's several-thousand
    token preamble is ~460s of silence before the first token exists, and the default
    trips at 300s. The reconnect then lands on a different parallel slot whose KV cache
    shares no prefix, so every retry restarts from zero and the turn never completes --
    a job hung for its full 600s cap with `Reconnecting... 1/5` and one request logged at
    exactly 300056ms.

    Asserted as a floor rather than an equality: raising it further is fine, and the
    number is not the contract. Losing it entirely is the regression.
    """
    provider = _parse_toml(codex_agent._merge_codex_config("", BASE))["model_providers"]["agent_switch"]
    assert "stream_idle_timeout_ms" in provider, (
        "the Codex provider block no longer sets stream_idle_timeout_ms, so Codex falls "
        "back to its 300s default and cancels any first turn whose prompt takes longer "
        "than that to process -- which is an ordinary local CPU host, not a corner case"
    )
    assert provider["stream_idle_timeout_ms"] > 300_000, (
        f"stream_idle_timeout_ms is {provider['stream_idle_timeout_ms']}, at or below "
        f"Codex's own 300000 default, so setting it changes nothing"
    )


def test_merge_codex_config_replaces_stale_block():
    existing = (
        'model = "gpt-5"\n'
        "\n"
        "[model_providers.agent_switch]\n"
        'base_url = "http://old-host:9999/v1"\n'
        'wire_api = "chat"\n'
        "\n"
        "[model_providers.agent_switch.http_headers]\n"
        'x-old = "1"\n'
        "\n"
        "[model_providers.ollama]\n"
        'base_url = "http://localhost:11434/v1"\n'
    )
    merged = codex_agent._merge_codex_config(existing, BASE)
    parsed = _parse_toml(merged)
    assert parsed["model"] == "gpt-5"
    assert parsed["model_providers"]["agent_switch"]["base_url"] == f"{BASE}/v1"
    assert parsed["model_providers"]["agent_switch"]["wire_api"] == "responses"
    assert "http_headers" not in parsed["model_providers"]["agent_switch"]
    assert parsed["model_providers"]["ollama"]["base_url"] == "http://localhost:11434/v1"
    assert codex_agent._merge_codex_config(merged, BASE) == merged


def test_merge_codex_config_keeps_user_oss_provider():
    merged = codex_agent._merge_codex_config('oss_provider = "ollama"\n', BASE)
    assert _parse_toml(merged)["oss_provider"] == "ollama"


def test_write_codex_config_profile(tmp_path, monkeypatch):
    set_start_attr(monkeypatch, "_codex_supports_model_catalog", lambda: True)
    set_start_attr(monkeypatch, "_codex_supports_patch_line_endings", lambda: True)
    codex_agent.write_codex_config(BASE, MODEL, tmp_path)
    profile = _parse_toml((tmp_path / "agent_switch.config.toml").read_text())
    assert profile["oss_provider"] == "agent_switch"
    assert profile["model_provider"] == "agent_switch"
    assert profile["model"] == MODEL["id"]
    assert profile["model_context_window"] == 131072
    assert profile["features"]["apply_patch_preserve_line_endings"] is True
    assert profile["suppress_unstable_features_warning"] is True

    catalog_path = Path(profile["model_catalog_json"])
    assert catalog_path == Path("model-catalog.json")
    catalog = json.loads((tmp_path / catalog_path).read_text())
    assert catalog["models"][0]["slug"] == MODEL["id"]
    assert catalog["models"][0]["context_window"] == 131072
    assert catalog["models"][0]["max_context_window"] == 131072
    assert catalog["models"][0]["supports_reasoning_summary_parameter"] is False
    assert catalog["models"][0]["supports_parallel_tool_calls"] is False
    assert catalog["models"][0]["apply_patch_tool_type"] == "freeform"

    assert catalog["models"][0]["base_instructions"] == codex_agent._CODEX_FALLBACK_PROMPT.read_text(
        encoding = "utf-8"
    )
    assert '{"command"' not in catalog["models"][0]["base_instructions"]
    config = _parse_toml((tmp_path / "config.toml").read_text())
    assert config["model_providers"]["agent_switch"]["env_key"] == "AGENT_SWITCH_AUTH_TOKEN"


def test_write_codex_config_catalog_without_context_length(tmp_path, monkeypatch):
    set_start_attr(monkeypatch, "_codex_supports_model_catalog", lambda: True)
    codex_agent.write_codex_config(BASE, {"id": "org/no-window"}, tmp_path)
    profile = _parse_toml((tmp_path / "agent_switch.config.toml").read_text())
    catalog = json.loads((tmp_path / profile["model_catalog_json"]).read_text())
    entry = catalog["models"][0]
    assert entry["slug"] == "org/no-window"
    assert "context_window" not in entry
    assert "max_context_window" not in entry


@pytest.mark.parametrize(
    ("version", "expected"),
    [("codex-cli 0.109.0", False), ("codex-cli 0.110.0", True), ("codex-cli 0.144.4", True)],
)
def test_codex_model_catalog_version_gate(monkeypatch, version, expected):
    monkeypatch.setattr(shutil, "which", lambda name, path = None: "/usr/local/bin/codex")
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: version)
    assert codex_agent._codex_supports_model_catalog() is expected


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("codex-cli 0.147.0", False),
        ("codex-cli 0.148.0", True),
        ("codex-cli 0.150.0", True),
        ("codex-cli 0.151.0", True),
        ("codex-cli 1.0.0", True),
    ],
)
def test_codex_patch_line_endings_version_gate(monkeypatch, version, expected):
    monkeypatch.setattr(shutil, "which", lambda name, path = None: "/usr/local/bin/codex")
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: version)
    assert codex_agent._codex_supports_patch_line_endings() is expected


def test_codex_patch_line_endings_assumes_current_when_not_installed(monkeypatch):
    set_start_attr(monkeypatch, "_which_with_install_dirs", lambda _: None)
    assert codex_agent._codex_supports_patch_line_endings() is True


def test_write_codex_config_omits_catalog_for_old_codex(tmp_path, monkeypatch):
    set_start_attr(monkeypatch, "_codex_supports_model_catalog", lambda: False)
    codex_agent.write_codex_config(BASE, MODEL, tmp_path)
    profile = _parse_toml((tmp_path / "agent_switch.config.toml").read_text())
    assert "model_catalog_json" not in profile
    assert not (tmp_path / "model-catalog.json").exists()


def test_write_codex_subagent_bridge_keeps_parent_credentials_out(tmp_path, monkeypatch):
    set_start_attr(monkeypatch, "_codex_supports_model_catalog", lambda: True)
    local = {**MODEL, "id": MODEL["id"] + ":UD-Q4_K_XL"}
    path = codex_agent.write_codex_subagent_bridge(
        BASE,
        "private-token",
        local,
        tmp_path,
        yolo = False,
    )
    assert json.loads(path.read_text(encoding = "utf-8")) == {
        "api_key": "private-token",
        "codex_home": str(tmp_path / "child"),
        "bypass_permissions": False,
    }
    assert path.stat().st_mode & 0o077 == 0
    profile = _parse_toml((tmp_path / "child" / "agent_switch.config.toml").read_text())
    assert profile["model"] == local["id"]
    assert profile["model_provider"] == codex_agent._CODEX_PROFILE
    assert profile["model_context_window"] == MODEL["context_length"]
    config = _parse_toml((tmp_path / "child" / "config.toml").read_text())
    assert config["model_providers"][codex_agent._CODEX_PROFILE]["base_url"] == f"{BASE}/v1"
    catalog = json.loads((tmp_path / "child" / profile["model_catalog_json"]).read_text())
    assert catalog["models"][0]["slug"] == local["id"]


def test_write_codex_parent_overlay_preserves_user_state_and_instructions(tmp_path, monkeypatch):
    source = tmp_path / "user-codex"
    source.mkdir()
    (source / "config.toml").write_text('model = "cloud-model"\n')
    (source / "auth.json").write_text('{"auth": "cloud"}\n')
    (source / "sessions").mkdir()
    (source / "AGENTS.override.md").write_text("Keep my existing instructions.\n")
    monkeypatch.setenv("CODEX_HOME", str(source))

    overlay = codex_agent.write_codex_parent_overlay(tmp_path / "managed" / "parent")

    assert (overlay / "config.toml").read_text() == 'model = "cloud-model"\n'
    assert (overlay / "auth.json").read_text() == '{"auth": "cloud"}\n'
    assert (overlay / "sessions").is_dir()
    instructions = (overlay / "AGENTS.override.md").read_text()
    assert instructions.startswith("Keep my existing instructions.\n")
    assert codex_agent._CODEX_SUBAGENT_ROUTING_INSTRUCTIONS in instructions
    assert not (overlay / "AGENTS.md").exists()
    assert (overlay / "AGENTS.override.md").stat().st_mode & 0o077 == 0
    assert (source / "AGENTS.override.md").read_text() == "Keep my existing instructions.\n"


def test_write_codex_parent_overlay_refreshes_reused_entries(tmp_path, monkeypatch):
    first = tmp_path / "first-codex"
    first.mkdir()
    (first / "auth.json").write_text('{"auth": "old"}\n')
    (first / "old-only.toml").write_text("old\n")
    second = tmp_path / "second-codex"
    second.mkdir()
    (second / "auth.json").write_text('{"auth": "new"}\n')
    overlay_path = tmp_path / "managed" / "parent"

    monkeypatch.setenv("CODEX_HOME", str(first))
    overlay = codex_agent.write_codex_parent_overlay(overlay_path)
    assert (overlay / "auth.json").read_text() == '{"auth": "old"}\n'
    assert (overlay / "old-only.toml").exists()

    monkeypatch.setenv("CODEX_HOME", str(second))
    overlay = codex_agent.write_codex_parent_overlay(overlay_path)
    assert (overlay / "auth.json").read_text() == '{"auth": "new"}\n'
    assert not (overlay / "old-only.toml").exists()


def test_write_codex_parent_overlay_does_not_use_itself_as_source(tmp_path, monkeypatch):
    source = tmp_path / "user-codex"
    source.mkdir()
    (source / "auth.json").write_text('{"auth": "cloud"}\n')
    overlay_path = tmp_path / "managed" / "parent"
    monkeypatch.setenv("CODEX_HOME", str(source))
    overlay = codex_agent.write_codex_parent_overlay(overlay_path)

    monkeypatch.setenv("CODEX_HOME", str(overlay))
    overlay = codex_agent.write_codex_parent_overlay(overlay_path)

    assert (overlay / "auth.json").read_text() == '{"auth": "cloud"}\n'
    manifest = json.loads((overlay / codex_agent._CODEX_PARENT_OVERLAY_MANIFEST).read_text())
    assert manifest["source_home"] == str(source)


def test_write_codex_parent_overlay_refreshes_fallback_copies(tmp_path, monkeypatch):
    source = tmp_path / "user-codex"
    source.mkdir()
    config = source / "config.toml"
    config.write_text('model = "first"\n')
    sessions = source / "sessions"
    sessions.mkdir()
    (sessions / "existing.jsonl").write_text("existing session\n")
    monkeypatch.setenv("CODEX_HOME", str(source))

    def deny_symlink(*args, **kwargs):
        raise OSError("symlinks unavailable")

    monkeypatch.setattr(Path, "symlink_to", deny_symlink)
    set_start_attr(monkeypatch, "_create_directory_junction", lambda source, target: False)
    overlay = codex_agent.write_codex_parent_overlay(tmp_path / "managed" / "parent")
    (overlay / "history.jsonl").write_text("session state\n")
    config.write_text('model = "second"\n')

    overlay = codex_agent.write_codex_parent_overlay(overlay)

    assert (overlay / "config.toml").read_text() == 'model = "second"\n'
    assert (overlay / "sessions" / "existing.jsonl").read_text() == "existing session\n"
    assert (overlay / "history.jsonl").read_text() == "session state\n"

    config.unlink()
    overlay = codex_agent.write_codex_parent_overlay(overlay)
    assert not (overlay / "config.toml").exists()
    assert (overlay / "history.jsonl").read_text() == "session state\n"


@pytest.mark.skipif(os.name == "nt", reason = "WSL scenario")
def test_write_codex_parent_overlay_uses_windows_home_for_windows_codex(tmp_path, monkeypatch):
    windows_profile = tmp_path / "windows-profile"
    source = windows_profile / ".codex"
    source.mkdir(parents = True)
    (source / "auth.json").write_text('{"auth": "windows"}\n')
    executable = "/mnt/c/Users/x/AppData/Roaming/npm/codex"
    monkeypatch.delenv("CODEX_HOME", raising = False)
    monkeypatch.delenv("USERPROFILE", raising = False)
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr(shutil, "which", lambda _: executable)

    def check_output(command, **kwargs):
        if command[0] == "cmd.exe":
            assert kwargs["cwd"] == str(Path(executable).parent)
            return r"C:\Users\x" + "\n"
        assert command == ["wslpath", "-u", r"C:\Users\x"]
        return str(windows_profile) + "\n"

    monkeypatch.setattr(subprocess, "check_output", check_output)

    overlay = codex_agent.write_codex_parent_overlay(tmp_path / "managed" / "parent")

    assert (overlay / "auth.json").read_text() == '{"auth": "windows"}\n'


def test_codex_parent_overlay_can_use_session_home(tmp_path, monkeypatch):
    source = tmp_path / "user-codex"
    source.mkdir()
    (source / "auth.json").write_text("{}\n")
    monkeypatch.setenv("CODEX_HOME", str(source))
    session_home = tmp_path / "session"

    overlay = codex_agent.write_codex_parent_overlay(session_home / "parent")

    assert overlay == session_home / "parent"
    assert codex_agent._CODEX_SUBAGENT_ROUTING_INSTRUCTIONS in (overlay / "AGENTS.md").read_text()
    assert overlay.exists()


def test_ephemeral_codex_parent_overlay_is_cleaned_with_session(tmp_path, monkeypatch):
    source = tmp_path / "user-codex"
    source.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(source))
    agents_root = tmp_path / "agents"
    set_start_attr(monkeypatch, "_agents_config_root", lambda: agents_root)

    with core_session._session_config("codex-subagent", launch = True) as session_home:
        overlay = codex_agent.write_codex_parent_overlay(session_home / "parent")
        assert overlay.exists()
        assert session_home.exists()

    assert not overlay.exists()
    assert not session_home.exists()


@pytest.mark.skipif(os.name == "nt", reason = "WSL scenario")
def test_codex_subagent_bridge_uses_wsl_for_windows_codex(monkeypatch, tmp_path):
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr(
        shutil,
        "which",
        lambda _: "/mnt/c/Users/x/AppData/Roaming/npm/codex.exe",
    )
    flags = codex_agent._codex_subagent_flags(tmp_path / "subagent.json")
    prefix = f"mcp_servers.{codex_agent._CODEX_SUBAGENT_MCP_SERVER}="
    override = next(value for value in flags if value.startswith(prefix))
    server = _parse_toml("server = " + override.removeprefix(prefix))["server"]
    assert server["command"] == "wsl.exe"
    assert server["args"] == [
        "-d",
        "Ubuntu",
        "--",
        sys.executable,
        "-c",
        server["args"][5],
        str(tmp_path / "subagent.json"),
    ]
    assert "sys.path.insert" in server["args"][5]
    assert f"from {codex_agent._CODEX_SUBAGENT_MCP_MODULE} import main" in server["args"][5]
    assert server["required"] is True
    assert server["enabled_tools"] == [codex_agent._CODEX_SUBAGENT_MCP_TOOL]
    assert server["default_tools_approval_mode"] == "approve"
    assert not any(value.startswith("developer_instructions=") for value in flags)


def test_codex_subagent_flags_bootstrap_path_contains_the_package(tmp_path):
    # The bootstrap sys.path.insert must name the directory that CONTAINS the agent_switch
    # package, so an interpreter without it installed (e.g. the WSL one) can still import
    # the bridge module from there.
    flags = codex_agent._codex_subagent_flags(tmp_path / "subagent.json")
    prefix = f"mcp_servers.{codex_agent._CODEX_SUBAGENT_MCP_SERVER}="
    override = next(value for value in flags if value.startswith(prefix))
    server = _parse_toml("server = " + override.removeprefix(prefix))["server"]
    bootstrap = next(arg for arg in server["args"] if "sys.path.insert" in arg)
    package_root = json.loads(bootstrap.split("sys.path.insert(0,", 1)[1].split(");", 1)[0])
    assert (Path(package_root) / "agent_switch" / "__init__.py").is_file()


@pytest.mark.skipif(os.name == "nt", reason = "WSL scenario")
def test_agent_config_path_translates_for_windows_agent(monkeypatch, tmp_path):
    windows_path = r"\\wsl.localhost\Ubuntu\tmp\agent-switch.toml"
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr(
        shutil,
        "which",
        lambda _: "/mnt/c/Users/x/AppData/Roaming/npm/codex",
    )
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: windows_path)

    assert core_session._agent_config_path(tmp_path / "agent-switch.toml", ["codex"]) == windows_path


def test_connect_codex_no_launch(fake_vllm, tmp_path):
    result = CliRunner().invoke(start.start_app, ["codex", "--no-launch"])
    assert result.exit_code == 0, result.output
    for name in codex_agent._CODEX_ENV_UNSET:
        _assert_env_unset(result.output, name)
    _assert_env_set(result.output, "AGENT_SWITCH_AUTH_TOKEN", KEY)
    assert "codex --oss --profile agent_switch" in result.output
    # Config lands in the session-scoped CODEX_HOME, not the user's ~/.codex.
    home = tmp_path / "agents" / "codex"
    _assert_env_set(result.output, "CODEX_HOME", str(home))
    assert (home / "config.toml").exists()
    assert (home / "agent_switch.config.toml").exists()


def test_connect_codex_as_subagent_preserves_cloud_parent(fake_vllm, tmp_path, monkeypatch):
    set_start_attr(monkeypatch, "_codex_supports_model_catalog", lambda: True)
    source_home = tmp_path / "user-codex"
    source_home.mkdir()
    (source_home / "config.toml").write_text('model = "cloud-model"\n')
    (source_home / "AGENTS.md").write_text("Keep the user's guidance.\n")
    monkeypatch.setenv("CODEX_HOME", str(source_home))
    result = CliRunner().invoke(
        start.start_app,
        [
            "codex",
            "--as-subagent",
            "--no-launch",
            "--model",
            MODEL["id"],
        ],
    )
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    assert command[0] == "codex"
    assert "--oss" not in command
    assert "--profile" not in command
    assert "--model" not in command
    parent_home = tmp_path / "agents" / "codex-subagent" / "parent"
    _assert_env_set(result.output, "CODEX_HOME", str(parent_home))
    for name in codex_agent._CODEX_ENV_UNSET:
        _assert_env_kept(result.output, name)
    assert codex_agent._CODEX_ENV_KEY not in result.output
    assert KEY not in result.output
    home = tmp_path / "agents" / "codex-subagent"
    bridge_path = home / "subagent.json"
    bridge = json.loads(bridge_path.read_text())
    assert bridge["api_key"] == KEY
    assert bridge["codex_home"] == str(home / "child")
    assert bridge["bypass_permissions"] is False
    profile = _parse_toml((home / "child" / "agent_switch.config.toml").read_text())
    assert profile["model"] == MODEL["id"]
    prefix = f"mcp_servers.{codex_agent._CODEX_SUBAGENT_MCP_SERVER}="
    override = next(value for value in command if value.startswith(prefix))
    assert override.startswith(prefix)
    server = _parse_toml("server = " + override.removeprefix(prefix))["server"]
    assert server["command"] == sys.executable
    assert server["args"] == ["-c", server["args"][1], str(bridge_path)]
    assert "sys.path.insert" in server["args"][1]
    assert f"from {codex_agent._CODEX_SUBAGENT_MCP_MODULE} import main" in server["args"][1]
    assert server["enabled_tools"] == [codex_agent._CODEX_SUBAGENT_MCP_TOOL]
    assert not any(value.startswith("developer_instructions=") for value in command)
    parent_instructions = (parent_home / "AGENTS.md").read_text()
    assert parent_instructions.startswith("Keep the user's guidance.\n")
    assert codex_agent._CODEX_SUBAGENT_ROUTING_INSTRUCTIONS in parent_instructions
    assert "Ask Codex to spawn a local agent." in result.output


def test_connect_codex_matches_requested_model_case_insensitively(fake_vllm, tmp_path):
    result = CliRunner().invoke(
        start.start_app,
        [
            "codex",
            "--no-launch",
            "--model",
            "org/gemma-4-26b-a4b-it-gguf",
        ],
    )
    assert result.exit_code == 0, result.output
    home = tmp_path / "agents" / "codex"
    profile = _parse_toml((home / "agent_switch.config.toml").read_text())
    assert profile["model"] == MODEL["id"]


def test_connect_codex_launch_uses_ephemeral_home(fake_vllm, monkeypatch):
    # Launch mode writes config to a throwaway temp CODEX_HOME and removes it after
    # the agent exits; the user's real ~/.codex is never the target.
    captured = {}
    monkeypatch.setattr(shutil, "which", lambda name, path = None: "/usr/local/bin/codex")

    def run(command, env):
        captured["home"] = env["CODEX_HOME"]
        captured["config_present"] = (Path(env["CODEX_HOME"]) / "config.toml").exists()
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(subprocess, "run", run)
    result = CliRunner().invoke(start.start_app, ["codex"])
    assert result.exit_code == 0, result.output
    home = Path(captured["home"])
    assert captured["config_present"]  # config existed while codex ran
    parent = core_session._ephemeral_session_parent("codex")
    assert home.name.startswith(core_session._ephemeral_session_prefix("codex", parent))
    assert not home.exists()  # cleaned up after the agent exits


@pytest.mark.skipif(
    os.name == "nt",
    reason = "the #6547 CI parser is bash-only; on Windows --no-launch prints PowerShell",
)
def test_no_launch_output_is_parseable(fake_vllm):
    # Mirror the #6547 CI parser: status lines, then `export`/`unset`, then exactly
    # one launch command on the last line (now an inline-env one-liner, so the parser
    # matches by substring rather than prefix).
    result = CliRunner().invoke(start.start_app, ["codex", "--no-launch"])
    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    skip = ("export ", "unset ", "vLLM ", "Updated ", "Disabled ", "Warning", "Loading")
    body = [ln for ln in lines if not ln.startswith(skip)]
    assert "codex --oss --profile agent_switch" in body[-1]
    assert any(ln.startswith("export CODEX_HOME=") for ln in lines)


def test_no_launch_last_line_is_self_contained(fake_vllm, tmp_path):
    # People copy just the last line. A bare `codex` there would run against the user's
    # real ~/.codex (e.g. a pre-existing damaged state DB) with zero isolation, so the
    # last line must inline every session env var ahead of the command.
    result = CliRunner().invoke(start.start_app, ["codex", "--no-launch"])
    assert result.exit_code == 0, result.output
    last = [ln for ln in result.output.splitlines() if ln.strip()][-1]
    parts = shlex.split(last)
    assignments = {}
    command = []
    for i, part in enumerate(parts):
        if "=" not in part:
            command = parts[i:]
            break
        name, _, value = part.partition("=")
        assignments[name] = value
    assert command and command[0] == "codex"
    assert assignments["CODEX_HOME"] == str(tmp_path / "agents" / "codex")
    assert assignments["AGENT_SWITCH_AUTH_TOKEN"] == KEY


def test_start_separator_preserves_model_shaped_agent_argument(fake_vllm):
    result = CliRunner().invoke(
        start.start_app,
        ["codex", "--no-launch", "--", "owner/repo"],
    )

    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    assert command[-2:] == ["--", "owner/repo"]

    result = CliRunner().invoke(
        start.start_app,
        ["codex", "--no-launch", MODEL["id"], "--", "--continue"],
    )
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    assert command[-2:] == ["--", "--continue"]


def test_codex_carries_reasoning_and_warns_about_sampling(fake_vllm, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(start.start_app, ["codex", "--no-launch", *_SESSION_FLAGS])
    assert result.exit_code == 0, result.output
    profile = (tmp_path / "agents" / "codex" / f"{codex_agent._CODEX_PROFILE}.config.toml").read_text()
    assert 'model_reasoning_effort = "none"' in profile
    assert "can't send --temperature, --top-k itself" in result.output
    assert "--reasoning" not in result.output
    result = CliRunner().invoke(start.start_app, ["codex", "--no-launch"])
    assert result.exit_code == 0, result.output
    profile = (tmp_path / "agents" / "codex" / f"{codex_agent._CODEX_PROFILE}.config.toml").read_text()
    assert "model_reasoning_effort" not in profile


def test_codex_warns_about_reasoning_it_cannot_express(fake_vllm, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(start.start_app, ["codex", "--no-launch", "--reasoning", "on"])
    assert result.exit_code == 0, result.output
    assert "can't send --reasoning itself" in result.output
    profile = (tmp_path / "agents" / "codex" / f"{codex_agent._CODEX_PROFILE}.config.toml").read_text()
    assert "model_reasoning_effort" not in profile


@pytest.mark.parametrize("mode", ["--persist", "--no-launch"])
@pytest.mark.parametrize("agent, version", [("codex", (0, 144, 0)), ("pi", (0, 83, 0))])
def test_agent_too_old_to_send_the_flags_warns_and_ignores(
    agent, version, mode, fake_vllm, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    set_start_attr(monkeypatch, "_which_with_install_dirs", lambda name: f"/bin/{name}")
    set_start_attr(monkeypatch, "_codex_executable_version", lambda executable: version)
    set_start_attr(monkeypatch, "_launch", lambda *args, **kwargs: None)
    result = CliRunner().invoke(start.start_app, [agent, mode, *_SESSION_FLAGS])
    assert result.exit_code == 0, result.output
    assert "can't send --temperature, --top-k, --reasoning itself" in result.output
    if agent == "pi":
        models = tmp_path / "agents" / "pi" / ".pi" / "agent" / "models.json"
        provider = json.loads(models.read_text())["providers"]["agent-switch"]
        assert "samplingParams" not in provider["models"][0]
    else:
        profile = (
            tmp_path / "agents" / "codex" / f"{codex_agent._CODEX_PROFILE}.config.toml"
        ).read_text()
        assert "model_reasoning_effort" not in profile


def test_session_config_codex_uses_short_ephemeral_parent(monkeypatch, tmp_path):
    # Windows Codex checks out its curated plugins under CODEX_HOME/.tmp/plugins.
    # Put its throwaway home outside the longer system temp path so that checkout
    # stays below legacy MAX_PATH and Codex does not reject temp-dir PATH helpers.
    short_parent = tmp_path / "u"
    short_parent.mkdir()
    set_start_attr(monkeypatch, "_ephemeral_session_parent",
        lambda agent: short_parent if agent == "codex" else None,
    )

    with core_session._session_config("codex", launch = True) as home:
        assert home.parent == short_parent
        assert home.name.startswith("a-codex-")
        assert home.exists()
    assert not home.exists()


@pytest.mark.parametrize("agent", ["codex", "codex-subagent"])
def test_windows_codex_homes_use_the_short_parent(monkeypatch, tmp_path, agent):
    # codex-subagent nests CODEX_HOME under <home>/parent, so it needs the short root even more.
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.delenv("AGENT_SWITCH_HOME", raising = False)

    assert core_session._ephemeral_session_parent(agent) == tmp_path / ".agent-switch" / ".tmp"
    parent = core_session._ephemeral_session_parent(agent)
    assert core_session._ephemeral_session_prefix(agent, parent) == "a-codex-"


def test_non_codex_agents_keep_the_agent_switch_root(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

    assert core_session._ephemeral_session_parent("claude") is None
    assert core_session._ephemeral_session_prefix("claude", None) == "agent-switch-claude-"


def test_session_config_reclaims_abandoned_homes_for_non_codex_agents(monkeypatch, tmp_path):
    # Nothing else prunes the agents tree, so a killed wrapper's home must be reclaimed.
    agents_root = tmp_path / "agents"
    temp_root = agents_root / ".tmp"
    temp_root.mkdir(parents = True)
    set_start_attr(monkeypatch, "_agents_config_root", lambda: agents_root)
    abandoned = temp_root / "agent-switch-claude-abandoned"
    abandoned.mkdir()
    (abandoned / ".active.lock").write_bytes(b"\0")
    (abandoned / "state.json").write_text("left behind")
    old = time.time() - core_session._CODEX_EPHEMERAL_STALE_SECONDS - 1
    os.utime(abandoned / ".active.lock", (old, old))
    recent = temp_root / "agent-switch-claude-still-running"
    recent.mkdir()
    (recent / ".active.lock").write_bytes(b"\0")

    with core_session._session_config("claude", launch = True) as home:
        assert not abandoned.exists()
        assert recent.exists()
        assert home.parent == temp_root
    assert not home.exists()


def test_persist_bare_codex_launch_has_no_resume_token(fake_vllm, monkeypatch):
    # A bare `--persist` only persists the session dir; it must NOT auto-append a native
    # resume token, or the very first launch (no session yet) would send codex down its
    # no-session error path. The user resumes explicitly: `agent-switch codex --persist resume`.
    monkeypatch.setattr(shutil, "which", lambda name, path = None: "/usr/local/bin/codex")
    captured = _capture_launch(monkeypatch, ["codex", "--persist"])
    assert "resume" not in captured["command"]
    # command[0] is the resolved executable path; assert the argv after it.
    assert captured["command"][1:] == ["--oss", "--profile", codex_agent._CODEX_PROFILE]


def test_resume_with_passthrough_does_not_auto_append(fake_vllm, monkeypatch):
    # When the caller drives their own subcommand, --persist only persists the dir; it
    # must not inject a resume token that would collide with the user's command.
    monkeypatch.setattr(shutil, "which", lambda name, path = None: "/usr/local/bin/codex")
    captured = _capture_launch(monkeypatch, ["codex", "--persist", "exec", "hello"])
    assert "resume" not in captured["command"]
    assert captured["command"][-2:] == ["exec", "hello"]


def test_default_launch_has_no_resume_token(fake_vllm, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name, path = None: "/usr/local/bin/codex")
    captured = _capture_launch(monkeypatch, ["codex"])
    assert "resume" not in captured["command"]
