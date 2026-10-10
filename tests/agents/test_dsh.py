# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See LICENSE
# Ported from unsloth_cli/tests/test_start.py (unsloth commit 8e11ba15e); see NOTICE.md.

"""`agent-switch dsh`: patch writing, command selection and permission modes."""

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
    dsh as dsh_agent,
)
from agent_switch.core import (
    install as core_install,
    session as core_session,
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
    _mcp_registry,
)
from tests.start_split import set_start_attr


@pytest.mark.parametrize("agent", ["dsh"])
@pytest.mark.parametrize("flag", ["--as-subagent", "--as-subagent=true", "--as-subagent=false"])
def test_unsupported_agents_reject_as_subagent(agent, flag):
    result = CliRunner().invoke(start.start_app, [agent, "--url", BASE, flag])
    assert result.exit_code == 1
    assert f"--as-subagent is not supported for {agent}." in result.output


def test_dsh_rejects_an_unrelated_executable_before_connect(monkeypatch):
    set_start_attr(monkeypatch, "_which_with_install_dirs", lambda _: "/usr/bin/dsh")
    set_start_attr(monkeypatch, "is_deepseek_harness_executable", lambda _: False)
    set_start_attr(monkeypatch, "_install_agent", lambda *_: None)
    set_start_attr(monkeypatch, "_connect",
        lambda *args, **kwargs: pytest.fail("the wrong dsh must be rejected before connection"),
    )

    result = CliRunner().invoke(start.start_app, ["dsh", "--provider", "vllm"])

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
    result = CliRunner().invoke(start.start_app, ["dsh", "--url", BASE, "--no-launch", *_SESSION_FLAGS])
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
    result = CliRunner().invoke(start.start_app, ["dsh", "--url", BASE, "--no-launch"])
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
    captured = _capture_launch(monkeypatch, ["dsh", "--url", BASE, "--profile", "headless", "hi"])
    command = captured["command"]
    assert command[command.index("--patch") + 1] == windows_path, command


def test_dsh_yolo_sets_permission_mode(fake_vllm):
    result = CliRunner().invoke(start.start_app, ["dsh", "--url", BASE, "--yolo", "--no-launch"])
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "DSH_PERMISSION_MODE", "danger-full-access")


def test_dsh_without_yolo_pins_the_safe_permission_mode(fake_vllm):
    result = CliRunner().invoke(start.start_app, ["dsh", "--url", BASE, "--no-launch"])
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "DSH_PERMISSION_MODE", "workspace-write")


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["dsh", "--url", BASE], "workspace-write"),
        (["dsh", "--url", BASE, "--yolo"], "danger-full-access"),
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
        ["dsh", "--url", BASE, "--model", MODEL["id"], "--reasoning-effort", "high"],
    )
    assert result.exit_code == 0, result.output
    assert "--reasoning-effort" not in captured["command"]
    provider = yaml.safe_load(captured["patch"])[0]["config"]["providers"][dsh_agent._DSH_PROVIDER]
    assert provider["compat"]["chatTemplateKwargs"] == {"reasoning_effort": "high"}


# ── the user's own AGENTS.md and skills linked into the session DSH home ──


def _dsh_user_home(tmp_path, monkeypatch) -> Path:
    # Point the user's home and DSH home at tmp dirs; dsh ~-expands through HOME.
    home = tmp_path / "user-home"
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("DSH_HOME", raising = False)
    return home


@pytest.mark.skipif(os.name == "nt", reason = "asserts POSIX symlinks")
def test_write_dsh_user_resources_links_agents_md_and_skills(tmp_path, monkeypatch):
    user_home = _dsh_user_home(tmp_path, monkeypatch)
    source = user_home / ".dsh"
    (source / "skills" / "my-skill").mkdir(parents = True)
    (source / "skills" / "my-skill" / "SKILL.md").write_text("skill body\n")
    (source / "AGENTS.md").write_text("user instructions\n")
    for private in (".credentials.yaml", "settings.yaml", "sessions", "storages"):
        (source / private).write_text("keep me out\n")
    dsh_home = tmp_path / "session"

    dsh_agent.write_dsh_user_resources(dsh_home, is_tui = False)

    assert (dsh_home / "AGENTS.md").is_symlink()
    assert (dsh_home / "skills").is_symlink()
    assert (dsh_home / "AGENTS.md").resolve() == (source / "AGENTS.md").resolve()
    assert (dsh_home / "skills").resolve() == (source / "skills").resolve()
    assert (dsh_home / "AGENTS.md").read_text() == "user instructions\n"
    assert (dsh_home / "skills" / "my-skill" / "SKILL.md").read_text() == "skill body\n"
    for private in (".credentials.yaml", "settings.yaml", "sessions", "storages"):
        assert not (dsh_home / private).exists()
    manifest = json.loads((dsh_home / dsh_agent._DSH_USER_RESOURCES_MANIFEST).read_text())
    assert manifest == {"entries": ["AGENTS.md", "skills"]}


@pytest.mark.skipif(os.name == "nt", reason = "asserts POSIX symlinks and ~ expansion")
def test_write_dsh_user_resources_uses_the_inherited_dsh_home(tmp_path, monkeypatch):
    user_home = _dsh_user_home(tmp_path, monkeypatch)
    configured = tmp_path / "custom-dsh"
    configured.mkdir()
    (configured / "AGENTS.md").write_text("custom\n")
    monkeypatch.setenv("DSH_HOME", str(configured))
    dsh_home = tmp_path / "session"

    dsh_agent.write_dsh_user_resources(dsh_home, is_tui = False)

    assert (dsh_home / "AGENTS.md").resolve() == (configured / "AGENTS.md").resolve()

    # dsh ~-expands DSH_HOME, so the source follows it; a gone entry is not recreated.
    (user_home / "tilde-dsh" / "skills").mkdir(parents = True)
    monkeypatch.setenv("DSH_HOME", "~/tilde-dsh")
    dsh_agent.write_dsh_user_resources(dsh_home, is_tui = False)

    assert not (dsh_home / "AGENTS.md").exists()
    assert (dsh_home / "skills").resolve() == (user_home / "tilde-dsh" / "skills").resolve()


@pytest.mark.skipif(os.name == "nt", reason = "asserts POSIX symlinks")
def test_write_dsh_user_resources_refuses_the_session_as_its_own_source(tmp_path, monkeypatch):
    user_home = _dsh_user_home(tmp_path, monkeypatch)
    (user_home / ".dsh").mkdir(parents = True)
    (user_home / ".dsh" / "AGENTS.md").write_text("real home\n")
    dsh_home = tmp_path / "session"
    monkeypatch.setenv("DSH_HOME", str(dsh_home))

    dsh_agent.write_dsh_user_resources(dsh_home, is_tui = False)

    assert (dsh_home / "AGENTS.md").resolve() == (user_home / ".dsh" / "AGENTS.md").resolve()


@pytest.mark.skipif(os.name == "nt", reason = "asserts POSIX symlinks")
def test_write_dsh_user_resources_refreshes_a_persisted_session(tmp_path, monkeypatch):
    user_home = _dsh_user_home(tmp_path, monkeypatch)
    source = user_home / ".dsh"
    (source / "skills").mkdir(parents = True)
    (source / "AGENTS.md").write_text("one\n")
    dsh_home = tmp_path / "session"
    dsh_agent.write_dsh_user_resources(dsh_home, is_tui = False)
    assert (dsh_home / "AGENTS.md").is_symlink()

    (source / "AGENTS.md").unlink()
    # A link that went stale between runs is replaced, not left dangling.
    (dsh_home / "skills").unlink()
    (dsh_home / "skills").symlink_to(tmp_path / "gone")

    dsh_agent.write_dsh_user_resources(dsh_home, is_tui = False)

    assert not (dsh_home / "AGENTS.md").exists()
    assert (dsh_home / "skills").resolve() == (source / "skills").resolve()
    manifest = json.loads((dsh_home / dsh_agent._DSH_USER_RESOURCES_MANIFEST).read_text())
    assert manifest == {"entries": ["skills"]}


def test_write_dsh_user_resources_keeps_session_owned_entries(tmp_path, monkeypatch):
    user_home = _dsh_user_home(tmp_path, monkeypatch)
    source = user_home / ".dsh"
    (source / "skills").mkdir(parents = True)
    (source / "AGENTS.md").write_text("user\n")
    dsh_home = tmp_path / "session"
    dsh_home.mkdir()
    (dsh_home / "AGENTS.md").write_text("session\n")
    (dsh_home / "skills").mkdir(parents = True)
    (dsh_home / "skills" / "mine.txt").write_text("mine\n")

    dsh_agent.write_dsh_user_resources(dsh_home, is_tui = False)

    assert not (dsh_home / "AGENTS.md").is_symlink()
    assert (dsh_home / "AGENTS.md").read_text() == "session\n"
    assert (dsh_home / "skills" / "mine.txt").read_text() == "mine\n"
    manifest = json.loads((dsh_home / dsh_agent._DSH_USER_RESOURCES_MANIFEST).read_text())
    assert manifest == {"entries": []}


def test_write_dsh_user_resources_falls_back_to_copy_and_junction(tmp_path, monkeypatch, capsys):
    user_home = _dsh_user_home(tmp_path, monkeypatch)
    source = user_home / ".dsh"
    (source / "skills").mkdir(parents = True)
    (source / "AGENTS.md").write_text("user\n")
    dsh_home = tmp_path / "session"

    def refuse_symlinks(self, target, target_is_directory = False):
        raise OSError("this filesystem does not allow symlinks")

    monkeypatch.setattr(Path, "symlink_to", refuse_symlinks)
    junctions = []

    def make_junction(source, target):
        junctions.append((source, target))
        return True

    set_start_attr(monkeypatch, "_create_directory_junction", make_junction)
    dsh_agent.write_dsh_user_resources(dsh_home, is_tui = False)

    assert (dsh_home / "AGENTS.md").is_file() and not (dsh_home / "AGENTS.md").is_symlink()
    assert (dsh_home / "AGENTS.md").read_text() == "user\n"
    assert junctions == [(source / "skills", dsh_home / "skills")]
    assert capsys.readouterr().err == ""

    # When both link forms fail for skills, warn once and keep the rest.
    set_start_attr(monkeypatch, "_create_directory_junction", lambda source, target: False)
    dsh_agent.write_dsh_user_resources(dsh_home, is_tui = False)

    assert not (dsh_home / "skills").exists()
    assert "couldn't link" in capsys.readouterr().err
    manifest = json.loads((dsh_home / dsh_agent._DSH_USER_RESOURCES_MANIFEST).read_text())
    assert manifest == {"entries": ["AGENTS.md"]}


@pytest.mark.skipif(os.name == "nt", reason = "WSL scenario")
def test_write_dsh_user_resources_clears_links_for_a_windows_dsh_under_wsl(
    tmp_path, monkeypatch, capsys
):
    user_home = _dsh_user_home(tmp_path, monkeypatch)
    source = user_home / ".dsh"
    (source / "skills").mkdir(parents = True)
    (source / "AGENTS.md").write_text("user\n")
    dsh_home = tmp_path / "session"
    dsh_agent.write_dsh_user_resources(dsh_home, is_tui = False)
    assert (dsh_home / "AGENTS.md").is_symlink()

    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr(shutil, "which", lambda name, path = None: "/mnt/c/npm/dsh")
    dsh_agent.write_dsh_user_resources(dsh_home, is_tui = False)

    assert not (dsh_home / "AGENTS.md").exists()
    assert not (dsh_home / "skills").exists()
    assert not (dsh_home / dsh_agent._DSH_USER_RESOURCES_MANIFEST).exists()
    assert "AGENTS.md and skills won't load" in capsys.readouterr().err
    # dsh-tui refuses a Windows launcher under WSL, so its session still gets the links.
    dsh_agent.write_dsh_user_resources(dsh_home, is_tui = True)
    assert (dsh_home / "AGENTS.md").is_symlink()
    assert (dsh_home / "skills").is_symlink()


@pytest.mark.skipif(os.name == "nt", reason = "WSL scenario")
def test_write_dsh_user_resources_stays_quiet_under_wsl_without_user_resources(
    tmp_path, monkeypatch, capsys
):
    # Warning only when there is something the session would have lost.
    _dsh_user_home(tmp_path, monkeypatch)
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr(shutil, "which", lambda name, path = None: "/mnt/c/npm/dsh")

    dsh_agent.write_dsh_user_resources(tmp_path / "session", is_tui = False)

    assert capsys.readouterr().err == ""
    assert not (tmp_path / "session").exists()


@pytest.mark.skipif(os.name == "nt", reason = "POSIX ephemeral sessions and symlinks")
@pytest.mark.parametrize(("agent", "is_tui"), [("dsh", False), ("dsh-tui", True)])
def test_ephemeral_session_removal_deletes_only_the_links(agent, is_tui, tmp_path, monkeypatch):
    # The ephemeral launch home goes with shutil.rmtree; that must drop the links into the
    # user's real ~/.dsh, never what they point at.
    user_home = _dsh_user_home(tmp_path, monkeypatch)
    source = user_home / ".dsh"
    (source / "skills" / "my-skill").mkdir(parents = True)
    (source / "skills" / "my-skill" / "SKILL.md").write_text("skill body\n")
    (source / "AGENTS.md").write_text("user instructions\n")
    set_start_attr(monkeypatch, "_agents_config_root", lambda: tmp_path / "agents")

    with core_session._session_config(agent, True) as home:
        dsh_home = home / ".dsh" if is_tui else home
        dsh_agent.write_dsh_user_resources(dsh_home, is_tui = is_tui)
        assert (dsh_home / "AGENTS.md").is_symlink()
        assert (dsh_home / "skills").is_symlink()
        assert (dsh_home / "skills" / "my-skill" / "SKILL.md").read_text() == "skill body\n"

    assert not home.exists()
    assert (source / "AGENTS.md").read_text() == "user instructions\n"
    assert (source / "skills" / "my-skill" / "SKILL.md").read_text() == "skill body\n"


@pytest.mark.skipif(os.name == "nt", reason = "asserts POSIX symlinks")
def test_dsh_no_launch_links_user_resources_into_the_session(fake_vllm, tmp_path, monkeypatch):
    user_home = _dsh_user_home(tmp_path, monkeypatch)
    source = user_home / ".dsh"
    (source / "skills" / "my-skill").mkdir(parents = True)
    (source / "AGENTS.md").write_text("user instructions\n")

    result = CliRunner().invoke(start.start_app, ["dsh", "--url", BASE, "--no-launch"])

    assert result.exit_code == 0, result.output
    dsh_home = tmp_path / "agents" / "dsh"
    assert (dsh_home / "AGENTS.md").resolve() == (source / "AGENTS.md").resolve()
    assert (dsh_home / "skills").resolve() == (source / "skills").resolve()


# ── --mcp / --mcp-all: session-only MCP mounting ─────────────────────


def _dsh_mcp_rows(patch_path):
    yaml = pytest.importorskip("yaml")
    rows = [entry["insert"] for entry in yaml.safe_load(patch_path.read_text()) if "insert" in entry]
    assert len(rows) <= 1
    return {row["id"]: row for row in rows[0]} if rows else {}


def test_connect_dsh_mcp_inserts_one_row_per_server(fake_vllm, tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "gh-secret")
    _mcp_registry()
    result = CliRunner().invoke(
        start.start_app, ["dsh", "--url", BASE, "--no-launch", "--mcp", "context7", "--mcp", "github"]
    )
    assert result.exit_code == 0, result.output
    rows = _dsh_mcp_rows(tmp_path / "agents" / "dsh" / dsh_agent._DSH_PATCH_FILE)
    assert set(rows) == {"mcp-context7", "mcp-github"}
    assert rows["mcp-context7"]["name"] == "@deepseek-ai/dsh-mcp-client"
    assert rows["mcp-context7"]["config"] == {
        "transport": "stdio",
        "serverName": "context7",
        "command": "npx",
        "args": ["-y", "@upstash/context7-mcp"],
        "env": {},
    }
    assert rows["mcp-github"]["config"] == {
        "transport": "streamable-http",
        "serverName": "github",
        "url": "https://api.githubcopilot.com/mcp/",
        "headers": {"Authorization": "Bearer gh-secret"},
    }
    assert "gh-secret" not in result.output


def test_connect_dsh_mcp_all_mounts_every_registry_server(fake_vllm, tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "gh-secret")
    _mcp_registry()
    result = CliRunner().invoke(start.start_app, ["dsh", "--url", BASE, "--no-launch", "--mcp-all"])
    assert result.exit_code == 0, result.output
    rows = _dsh_mcp_rows(tmp_path / "agents" / "dsh" / dsh_agent._DSH_PATCH_FILE)
    assert set(rows) == {"mcp-context7", "mcp-github"}


def test_connect_dsh_mcp_url_mounts_without_a_registry(fake_vllm, tmp_path):
    result = CliRunner().invoke(
        start.start_app, ["dsh", "--url", BASE, "--no-launch", "--mcp-url", "ev=http://127.0.0.1:18331/mcp"]
    )
    assert result.exit_code == 0, result.output
    rows = _dsh_mcp_rows(tmp_path / "agents" / "dsh" / dsh_agent._DSH_PATCH_FILE)
    assert rows["mcp-ev"]["config"] == {
        "transport": "streamable-http",
        "serverName": "ev",
        "url": "http://127.0.0.1:18331/mcp",
        "headers": {},
    }


def test_connect_dsh_mcp_stdio_mounts_without_a_registry(fake_vllm, tmp_path):
    result = CliRunner().invoke(
        start.start_app,
        ["dsh", "--url", BASE, "--no-launch", "--mcp-stdio", "ev=npx -y @modelcontextprotocol/server-everything@2026.8.31"],
    )
    assert result.exit_code == 0, result.output
    rows = _dsh_mcp_rows(tmp_path / "agents" / "dsh" / dsh_agent._DSH_PATCH_FILE)
    assert rows["mcp-ev"]["config"] == {
        "transport": "stdio",
        "serverName": "ev",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-everything@2026.8.31"],
        "env": {},
    }


def test_connect_dsh_mcp_stdio_shell_form(fake_vllm, tmp_path, monkeypatch):
    # fake_vllm stubs shutil.which to None; the shell form needs bash found.
    monkeypatch.setattr(shutil, "which", lambda name, path = None: "/bin/bash" if name == "bash" else None)
    result = CliRunner().invoke(
        start.start_app,
        ["dsh", "--url", BASE, "--no-launch", "--mcp-stdio", 'ev="npx -y @modelcontextprotocol/server-everything@2026.8.31"'],
    )
    assert result.exit_code == 0, result.output
    rows = _dsh_mcp_rows(tmp_path / "agents" / "dsh" / dsh_agent._DSH_PATCH_FILE)
    assert rows["mcp-ev"]["config"] == {
        "transport": "stdio",
        "serverName": "ev",
        "command": "bash",
        "args": ["-ic", "exec npx -y @modelcontextprotocol/server-everything@2026.8.31"],
        "env": {},
    }


def test_connect_dsh_without_mcp_flags_has_no_insert_row(fake_vllm, tmp_path):
    result = CliRunner().invoke(start.start_app, ["dsh", "--url", BASE, "--no-launch"])
    assert result.exit_code == 0, result.output
    patch = tmp_path / "agents" / "dsh" / dsh_agent._DSH_PATCH_FILE
    assert _dsh_mcp_rows(patch) == {}


def test_dsh_mcp_state_cleared_on_rerun_without_flags(fake_vllm, tmp_path, monkeypatch):
    # The patch is rewritten whole, so a run without MCP flags drops the insert row again.
    monkeypatch.setenv("GITHUB_TOKEN", "gh-secret")
    _mcp_registry()
    result = CliRunner().invoke(start.start_app, ["dsh", "--url", BASE, "--no-launch", "--mcp", "context7"])
    assert result.exit_code == 0, result.output
    patch = tmp_path / "agents" / "dsh" / dsh_agent._DSH_PATCH_FILE
    assert _dsh_mcp_rows(patch)
    result = CliRunner().invoke(start.start_app, ["dsh", "--url", BASE, "--no-launch"])
    assert result.exit_code == 0, result.output
    assert _dsh_mcp_rows(patch) == {}


def test_dsh_mcp_with_as_subagent_fails(fake_vllm):
    _mcp_registry()
    result = CliRunner().invoke(
        start.start_app, ["dsh", "--url", BASE, "--no-launch", "--mcp", "context7", "--as-subagent"]
    )
    assert result.exit_code == 1
    assert "--as-subagent is not supported for dsh." in result.output


# ── Native launch (no --url/--provider) ──────────────────────────────


def _native_no_connect_dsh(monkeypatch):
    set_start_attr(monkeypatch, "_connect",
        lambda *args, **kwargs: pytest.fail("native launch must not connect"),
    )


def test_native_dsh_bare_passes_no_patch(fake_vllm, monkeypatch):
    _native_no_connect_dsh(monkeypatch)
    result = CliRunner().invoke(start.start_app, ["dsh", "--no-launch"])
    assert result.exit_code == 0, result.output
    assert _launch_command(result.output) == ["dsh", "web"]
    assert "DSH_HOME" not in result.output


def test_native_dsh_mcp_patch_holds_only_the_insert_row(fake_vllm, tmp_path, monkeypatch):
    _native_no_connect_dsh(monkeypatch)
    result = CliRunner().invoke(
        start.start_app, ["dsh", "--no-launch", "--mcp-stdio", "ev=npx everything"]
    )
    assert result.exit_code == 0, result.output
    patch = tmp_path / "agents" / "dsh-native" / "agent-switch.patch.yml"
    assert _launch_command(result.output) == ["dsh", "web", "--patch", str(patch)]
    rows = _dsh_mcp_rows(patch)
    assert list(rows) == ["mcp-ev"]
    assert rows["mcp-ev"]["config"]["command"] == "npx"
    assert "DSH_HOME" not in result.output


def test_native_dsh_yolo_sets_only_the_permission_mode(fake_vllm, monkeypatch):
    _native_no_connect_dsh(monkeypatch)
    result = CliRunner().invoke(start.start_app, ["dsh", "--no-launch", "--yolo"])
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "DSH_PERMISSION_MODE", "danger-full-access")
    assert "DSH_HOME" not in result.output


@pytest.mark.parametrize(
    "flag",
    [
        ["--model", "org/model"],
        ["--compact-at", "0.8"],
        ["--no-persist"],
        ["--persist"],
        ["--max-tokens", "4096"],
    ],
)
def test_native_dsh_refuses_local_only_flags(fake_vllm, monkeypatch, flag):
    _native_no_connect_dsh(monkeypatch)
    result = CliRunner().invoke(start.start_app, ["dsh", "--no-launch", *flag])
    assert result.exit_code == 1
    assert "needs --url or --provider" in result.output
