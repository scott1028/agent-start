# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See LICENSE
# Ported from unsloth_cli/tests/test_start.py (unsloth commit 8e11ba15e); see NOTICE.md.

"""`agent-switch pi`: config, user resources, output limits and session flags."""

import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
from typer.testing import CliRunner

import agent_switch.start as start
from agent_switch.agents import (
    opencode as opencode_agent,
    pi as pi_agent,
)
from tests.cli_support import (
    BASE,
    KEY,
    MODEL,
    _SESSION_FLAGS,
    _assert_env_set,
    _launch_command,
    _mcp_registry,
    _simulate_windows,
)
from tests.start_split import set_start_attr


def test_missing_pi_subagent_extension_fails_before_install_or_connect(monkeypatch, tmp_path):
    set_start_attr(monkeypatch, "_PI_SUBAGENT_EXTENSION", tmp_path / "missing.ts")
    set_start_attr(monkeypatch, "_require_agent_for_launch",
        lambda *args: pytest.fail("local prerequisites must be checked before installation"),
    )
    set_start_attr(monkeypatch, "_connect",
        lambda *args, **kwargs: pytest.fail(
            "local prerequisites must be checked before connection"
        ),
    )

    result = CliRunner().invoke(start.start_app, ["pi", "--provider", "vllm", "--as-subagent"])

    assert result.exit_code == 1
    assert "Missing Pi subagent extension" in result.output


# vLLM reads the reasoning switch from chat_template_kwargs.
_SESSION_BODY = {"temperature": 0.3, "top_k": 40, "chat_template_kwargs": {"enable_thinking": False}}


def _session_request_body(agent, root, output):
    home = root / "agents" / agent
    if agent == "pi":
        config = json.loads((home / ".pi" / "agent" / "models.json").read_text())
        return config["providers"]["agent-switch"]["models"][0].get("samplingParams")
    if agent == "opencode":
        provider = json.loads((home / "opencode.json").read_text())["provider"]
        provider = provider[opencode_agent._OPENCODE_PROVIDER]
        options = provider["models"][MODEL["id"]].get("options")
        assert provider["options"].get("body") == options
        return options
    if agent == "claude":
        settings = Path(shlex.split(re.search(r"--settings (\S+)", output).group(1))[0])
        extra = json.loads(settings.read_text())["env"].get("CLAUDE_CODE_EXTRA_BODY")
        return json.loads(extra) if extra else None
    raise AssertionError(agent)


@pytest.mark.parametrize("agent", ["pi", "opencode", "claude"])
def test_session_flags_ride_in_the_agent_config_on_a_running_server(
    agent, fake_vllm, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(start.start_app, [agent, "--url", BASE, "--no-launch", *_SESSION_FLAGS])
    assert result.exit_code == 0, result.output
    assert "already running" not in result.output
    assert _session_request_body(agent, tmp_path, result.output) == _SESSION_BODY


@pytest.mark.parametrize("agent", ["pi", "opencode", "claude"])
def test_session_flags_from_an_earlier_run_do_not_stick(agent, fake_vllm, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for argv in (
        [agent, "--url", BASE, "--no-launch", *_SESSION_FLAGS],
        [agent, "--url", BASE, "--no-launch"],
    ):
        result = CliRunner().invoke(start.start_app, argv)
        assert result.exit_code == 0, result.output
    assert not _session_request_body(agent, tmp_path, result.output)


def test_pi_subagent_carries_the_session_flags(fake_vllm, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        start.start_app, ["pi", "--url", BASE, "--as-subagent", "--no-launch", *_SESSION_FLAGS]
    )
    assert result.exit_code == 0, result.output
    assert "already running" not in result.output
    config = json.loads((tmp_path / "agents" / "pi-subagent" / "subagent.json").read_text())
    assert config["samplingParams"] == _SESSION_BODY


# ── Hermes (OpenAI /v1/chat/completions, key via env) ────────────────

# ── Pi (OpenAI-compatible /v1, key in config, ~/.pi relocated via HOME) ──
def test_write_pi_config_fresh(tmp_path):
    path = tmp_path / ".pi" / "agent" / "models.json"
    pi_agent.write_pi_config(BASE, "sk-test-abc", MODEL, path)
    config = json.loads(path.read_text())
    provider = config["providers"]["agent-switch"]
    assert provider["api"] == "openai-completions"
    assert provider["baseUrl"] == f"{BASE}/v1"
    assert provider["apiKey"] == "sk-test-abc"
    # Pin the loaded window (and a sane output cap) so Pi compacts instead of
    # overflowing; without it Pi assumes its 128000 default.
    assert provider["models"] == [
        {"id": MODEL["id"], "contextWindow": MODEL["context_length"], "maxTokens": 32000}
    ]


def test_write_pi_config_preserves_and_idempotent(tmp_path):
    path = tmp_path / ".pi" / "agent" / "models.json"
    path.parent.mkdir(parents = True)
    path.write_text(json.dumps({"providers": {"google": {"api": "gemini"}}}))
    pi_agent.write_pi_config(BASE, "sk-test-abc", MODEL, path)
    config = json.loads(path.read_text())
    assert config["providers"]["google"] == {"api": "gemini"}  # unrelated provider kept
    assert config["providers"]["agent-switch"]["baseUrl"] == f"{BASE}/v1"
    before = path.read_text()
    pi_agent.write_pi_config(BASE, "sk-test-abc", MODEL, path)
    assert path.read_text() == before


def _pi_user_agent_dir(tmp_path, monkeypatch) -> Path:
    user_home = tmp_path / "user-home"
    agent_dir = user_home / ".pi" / "agent"
    agent_dir.mkdir(parents = True)
    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.setenv("USERPROFILE", str(user_home))
    monkeypatch.delenv("PI_CODING_AGENT_DIR", raising = False)
    return agent_dir


def test_connect_pi_no_launch(fake_vllm, tmp_path, monkeypatch):
    user_agent_dir = _pi_user_agent_dir(tmp_path, monkeypatch)
    (user_agent_dir / "extensions").mkdir()
    (user_agent_dir / "extensions" / "mine.ts").write_text("export default () => {};\n")
    (user_agent_dir / "settings.json").write_text(json.dumps({"packages": ["npm:pi-mine"]}))
    result = CliRunner().invoke(start.start_app, ["pi", "--url", BASE, "--no-launch"])
    assert result.exit_code == 0, result.output
    # Pi resolves its config dir from PI_CODING_AGENT_DIR first, so pin it at the session
    # dir (and relocate HOME) to keep the user's real ~/.pi untouched and their own
    # PI_CODING_AGENT_DIR from redirecting Pi away from our provider/key.
    home = tmp_path / "agents" / "pi"
    _assert_env_set(result.output, "HOME", str(home))
    _assert_env_set(result.output, "PI_CODING_AGENT_DIR", str(home / ".pi" / "agent"))
    # Provider/model pinned on the command (Pi defaults to google otherwise).
    assert f"pi --provider agent-switch --model {MODEL['id']}" in result.output
    config = json.loads((home / ".pi" / "agent" / "models.json").read_text())
    assert config["providers"]["agent-switch"]["apiKey"] == KEY
    assert config["providers"]["agent-switch"]["models"] == [
        {"id": MODEL["id"], "contextWindow": MODEL["context_length"], "maxTokens": 32000}
    ]
    assert (home / ".pi" / "agent" / "extensions" / "mine.ts").is_file()
    settings = json.loads((home / ".pi" / "agent" / "settings.json").read_text())
    assert settings == {"packages": ["npm:pi-mine"]}
    assert not (user_agent_dir / "models.json").exists()


@pytest.mark.skipif(os.name == "nt", reason = "asserts POSIX symlinks and path forms")
def test_write_pi_user_resources_links_resources_and_keeps_config_private(tmp_path, monkeypatch):
    user_agent_dir = _pi_user_agent_dir(tmp_path, monkeypatch)
    user_home = user_agent_dir.parent.parent
    for name in ("extensions", "skills", "npm", "git"):
        (user_agent_dir / name).mkdir()
    (user_home / ".agents" / "skills").mkdir(parents = True)
    (user_agent_dir / "auth.json").write_text('{"google": "user-key"}\n')
    (user_agent_dir / "models.json").write_text('{"providers": {"mine": {}}}\n')
    (user_agent_dir / "sessions").mkdir()
    (user_agent_dir / "settings.json").write_text(
        json.dumps(
            {
                "defaultProvider": "google",
                "theme": "light",
                "packages": [
                    "npm:pi-mine@1.2.3",
                    "git:github.com/me/pi-tools",
                    "../../src/local-extension",
                    {"source": "~/other-extension", "extensions": ["index.ts"]},
                ],
                "extensions": [
                    "extensions/extra.ts",
                    "-extensions/off.ts",
                    "root.ts",
                    "/abs/ext.ts",
                    "!local/old.ts",
                    "-~/literal.ts",
                ],
                "skills": ["skills/*", "../shared/*"],
            }
        )
    )
    session_home = tmp_path / "session"
    agent_dir = session_home / ".pi" / "agent"
    pi_agent.write_pi_config(BASE, "sk-test-abc", MODEL, agent_dir / "models.json")

    pi_agent.write_pi_user_resources(agent_dir, session_home)

    for name in ("extensions", "skills", "npm", "git"):
        assert (agent_dir / name).is_symlink()
        assert (agent_dir / name).resolve() == (user_agent_dir / name).resolve()
    assert not (agent_dir / "prompts").exists()
    assert (session_home / ".agents" / "skills").resolve() == (
        user_home / ".agents" / "skills"
    ).resolve()
    for private in ("auth.json", "sessions"):
        assert not (agent_dir / private).exists()
    assert "mine" not in json.loads((agent_dir / "models.json").read_text())["providers"]
    settings = json.loads((agent_dir / "settings.json").read_text())
    assert settings == {
        "packages": [
            "npm:pi-mine@1.2.3",
            "git:github.com/me/pi-tools",
            str(user_home / "src" / "local-extension"),
            {"source": str(user_home / "other-extension"), "extensions": ["index.ts"]},
        ],
        "extensions": [
            "extensions/extra.ts",
            "-extensions/off.ts",
            str(user_agent_dir / "root.ts"),
            "/abs/ext.ts",
            # Pi matches patterns relative to the agent directory, so add a user-anchored copy.
            "!local/old.ts",
            f"!{user_agent_dir / 'local' / 'old.ts'}",
            "-~/literal.ts",
        ],
        "skills": ["skills/*", "../shared/*", str(user_home / ".pi" / "shared" / "*")],
    }


def test_write_pi_user_resources_refreshes_a_persisted_session(tmp_path, monkeypatch):
    user_agent_dir = _pi_user_agent_dir(tmp_path, monkeypatch)
    (user_agent_dir / "extensions").mkdir()
    user_settings = user_agent_dir / "settings.json"
    user_settings.write_text(json.dumps({"packages": ["npm:old", "npm:kept"]}))
    session_home = tmp_path / "session"
    agent_dir = session_home / ".pi" / "agent"

    pi_agent.write_pi_user_resources(agent_dir, session_home)
    # Add session-only state, then change the user config.
    settings_path = agent_dir / "settings.json"
    settings = json.loads(settings_path.read_text())
    settings["packages"].append("npm:session-only")
    settings["theme"] = "dark"
    settings_path.write_text(json.dumps(settings))
    user_settings.write_text(json.dumps({"packages": ["npm:kept", "npm:new"]}))
    (user_agent_dir / "extensions").rmdir()

    pi_agent.write_pi_user_resources(agent_dir, session_home)

    # Pi keeps the first entry per package identity, so session-owned packages lead.
    assert json.loads(settings_path.read_text()) == {
        "packages": ["npm:session-only", "npm:kept", "npm:new"],
        "theme": "dark",
    }
    assert not (agent_dir / "extensions").exists() and not (agent_dir / "extensions").is_symlink()

    user_settings.unlink()
    pi_agent.write_pi_user_resources(agent_dir, session_home)
    assert json.loads(settings_path.read_text()) == {
        "packages": ["npm:session-only"],
        "theme": "dark",
    }
    assert not (agent_dir / pi_agent._PI_USER_RESOURCES_MANIFEST).exists()


def test_write_pi_user_resources_leaves_session_dirs_and_user_files_alone(tmp_path, monkeypatch):
    user_agent_dir = _pi_user_agent_dir(tmp_path, monkeypatch)
    (user_agent_dir / "npm" / "node_modules" / "pi-mine").mkdir(parents = True)
    (user_agent_dir / "extensions").mkdir()
    (user_agent_dir / "extensions" / "mine.ts").write_text("mine\n")
    session_home = tmp_path / "session"
    agent_dir = session_home / ".pi" / "agent"
    (agent_dir / "npm").mkdir(parents = True)
    (agent_dir / "npm" / "session.txt").write_text("session\n")

    pi_agent.write_pi_user_resources(agent_dir, session_home)

    assert not (agent_dir / "npm").is_symlink()
    assert (agent_dir / "npm" / "session.txt").read_text() == "session\n"
    assert (agent_dir / "extensions" / "mine.ts").read_text() == "mine\n"
    # Deleting the session must not delete linked user files.
    shutil.rmtree(session_home)
    assert (user_agent_dir / "extensions" / "mine.ts").read_text() == "mine\n"
    assert (user_agent_dir / "npm" / "node_modules" / "pi-mine").is_dir()


def test_write_pi_user_resources_uses_inherited_agent_dir(tmp_path, monkeypatch):
    _pi_user_agent_dir(tmp_path, monkeypatch)
    configured = tmp_path / "custom-pi"
    (configured / "extensions").mkdir(parents = True)
    session_home = tmp_path / "session"
    agent_dir = session_home / ".pi" / "agent"
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(configured))

    pi_agent.write_pi_user_resources(agent_dir, session_home)
    assert (agent_dir / "extensions").resolve() == (configured / "extensions").resolve()

    # Do not reuse the session as its own source.
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent_dir))
    pi_agent.write_pi_user_resources(agent_dir, session_home)
    assert not (agent_dir / "extensions").exists()


def test_write_pi_user_resources_resolves_a_relative_agent_dir_from_the_launch_dir(
    tmp_path, monkeypatch
):
    _pi_user_agent_dir(tmp_path, monkeypatch)
    configured = tmp_path / "launch" / "custom-pi"
    (configured / "extensions").mkdir(parents = True)
    (configured / "settings.json").write_text(json.dumps({"packages": ["../src/local-ext"]}))
    monkeypatch.chdir(tmp_path / "launch")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", "custom-pi")
    session_home = tmp_path / "session"
    agent_dir = session_home / ".pi" / "agent"

    pi_agent.write_pi_user_resources(agent_dir, session_home)

    assert (agent_dir / "extensions").resolve() == (configured / "extensions").resolve()
    settings = json.loads((agent_dir / "settings.json").read_text())
    assert settings == {"packages": [str(tmp_path / "launch" / "src" / "local-ext")]}


def test_link_user_dir_replaces_a_junction_from_an_earlier_run(tmp_path, monkeypatch):
    target = tmp_path / "session" / "extensions"
    target.mkdir(parents = True)  # stands in for a junction to a previous source
    source = tmp_path / "user" / "extensions"
    source.mkdir(parents = True)
    set_start_attr(monkeypatch, "_is_junction", lambda path: path == target and path.is_dir())

    pi_agent._link_user_dir(source, target)

    assert target.is_symlink()
    assert target.resolve() == source.resolve()


def test_write_pi_user_resources_skips_a_windows_pi_under_wsl(tmp_path, monkeypatch):
    user_agent_dir = _pi_user_agent_dir(tmp_path, monkeypatch)
    (user_agent_dir / "extensions").mkdir()
    (user_agent_dir / "settings.json").write_text(json.dumps({"packages": ["npm:pi-mine"]}))
    set_start_attr(monkeypatch, "_wsl_windows_executable", lambda _: "/mnt/c/npm/pi")
    session_home = tmp_path / "session"
    agent_dir = session_home / ".pi" / "agent"

    pi_agent.write_pi_user_resources(agent_dir, session_home)

    assert not agent_dir.exists()


@pytest.mark.skipif(os.name == "nt", reason = "asserts POSIX symlinks and path forms")
def test_write_pi_user_resources_clears_a_session_a_linux_pi_prepared(tmp_path, monkeypatch):
    # Switching a persisted session from a Linux pi to a Windows one must not hand
    # Windows Pi the WSL-backed links the skip exists to withhold.
    user_agent_dir = _pi_user_agent_dir(tmp_path, monkeypatch)
    (user_agent_dir / "extensions").mkdir()
    (user_agent_dir / "settings.json").write_text(
        json.dumps({"packages": ["npm:user-pkg"], "extensions": ["extensions/mine.ts"]})
    )
    session_home = tmp_path / "session"
    agent_dir = session_home / ".pi" / "agent"
    set_start_attr(monkeypatch, "_wsl_windows_executable", lambda _: None)
    pi_agent.write_pi_user_resources(agent_dir, session_home)
    assert (agent_dir / "extensions").is_symlink()
    # Something the session set for itself, which must survive.
    settings = json.loads((agent_dir / "settings.json").read_text())
    settings["packages"].append("npm:session-only")
    (agent_dir / "settings.json").write_text(json.dumps(settings))

    set_start_attr(monkeypatch, "_wsl_windows_executable", lambda _: "/mnt/c/npm/pi.cmd")
    pi_agent.write_pi_user_resources(agent_dir, session_home)

    assert not (agent_dir / "extensions").exists()
    assert json.loads((agent_dir / "settings.json").read_text()) == {
        "packages": ["npm:session-only"],
    }
    assert not (agent_dir / pi_agent._PI_USER_RESOURCES_MANIFEST).exists()
    # The user's own directory is untouched either way.
    assert (user_agent_dir / "extensions").is_dir()


@pytest.mark.parametrize("session_command", [None, ["npm", "--silent"]])
def test_write_pi_user_resources_clears_npm_command_whole(tmp_path, monkeypatch, session_command):
    # npmCommand is an argument vector: subtracting it entry by entry would leave a
    # command missing whatever the copied and session values happen to share.
    user_agent_dir = _pi_user_agent_dir(tmp_path, monkeypatch)
    (user_agent_dir / "settings.json").write_text(
        json.dumps({"npmCommand": ["npm", "--registry=x"]})
    )
    session_home = tmp_path / "session"
    agent_dir = session_home / ".pi" / "agent"
    set_start_attr(monkeypatch, "_wsl_windows_executable", lambda _: None)
    pi_agent.write_pi_user_resources(agent_dir, session_home)
    assert json.loads((agent_dir / "settings.json").read_text())["npmCommand"] == [
        "npm",
        "--registry=x",
    ]
    if session_command is not None:
        settings = json.loads((agent_dir / "settings.json").read_text())
        settings["npmCommand"] = session_command
        (agent_dir / "settings.json").write_text(json.dumps(settings))

    set_start_attr(monkeypatch, "_wsl_windows_executable", lambda _: "/mnt/c/npm/pi.cmd")
    pi_agent.write_pi_user_resources(agent_dir, session_home)

    settings = json.loads((agent_dir / "settings.json").read_text())
    # The copied command goes; one the session chose survives intact.
    assert settings.get("npmCommand") == session_command


@pytest.mark.skipif(os.name == "nt", reason = "asserts POSIX symlinks and path forms")
def test_write_pi_user_resources_reanchors_when_a_real_session_dir_blocks_the_link(
    tmp_path, monkeypatch
):
    # Pi creates <agent dir>/npm the first time a package is installed, so a
    # persisted session can already own that directory. The link is then skipped
    # and a session-relative entry would point into it instead of at the user's.
    user_agent_dir = _pi_user_agent_dir(tmp_path, monkeypatch)
    (user_agent_dir / "npm").mkdir()
    (user_agent_dir / "extensions").mkdir()
    (user_agent_dir / "settings.json").write_text(
        json.dumps({"packages": ["npm/pkg"], "extensions": ["extensions/mine.ts"]})
    )
    session_home = tmp_path / "session"
    agent_dir = session_home / ".pi" / "agent"
    (agent_dir / "npm").mkdir(parents = True)

    pi_agent.write_pi_user_resources(agent_dir, session_home)

    settings = json.loads((agent_dir / "settings.json").read_text())
    assert not (agent_dir / "npm").is_symlink()  # the session's own directory survives
    assert settings["packages"] == [str(user_agent_dir / "npm" / "pkg")]
    # extensions was linked, so entries under it stay relative and resolve through it.
    assert settings["extensions"] == ["extensions/mine.ts"]


@pytest.mark.skipif(os.name == "nt", reason = "asserts POSIX symlinks and path forms")
def test_write_pi_user_resources_keeps_a_disabled_global_skill_disabled(tmp_path, monkeypatch):
    # ~/.agents/skills is reached through HOME, which moved, so a rule naming the
    # user's copy has to follow it or Pi re-enables the skill inside the session.
    user_agent_dir = _pi_user_agent_dir(tmp_path, monkeypatch)
    user_home = user_agent_dir.parent.parent
    disabled = user_home / ".agents" / "skills" / "off" / "SKILL.md"
    disabled.parent.mkdir(parents = True)
    disabled.write_text("disabled\n")
    (user_agent_dir / "settings.json").write_text(json.dumps({"skills": [f"-{disabled}"]}))
    session_home = tmp_path / "session"
    agent_dir = session_home / ".pi" / "agent"

    pi_agent.write_pi_user_resources(agent_dir, session_home)

    session_skill = session_home / ".agents" / "skills" / "off" / "SKILL.md"
    assert session_skill.exists()
    settings = json.loads((agent_dir / "settings.json").read_text())
    # The original rule is kept and an alias for the session path is added beside it.
    assert settings["skills"] == [f"-{disabled}", f"-{session_skill}"]


def test_write_pi_user_resources_copies_the_npm_command(tmp_path, monkeypatch):
    # Pi runs every package lookup and install through npmCommand, so a session
    # that inherits the package list without it falls back to plain npm.
    user_agent_dir = _pi_user_agent_dir(tmp_path, monkeypatch)
    command = ["mise", "exec", "node@20", "--", "npm"]
    (user_agent_dir / "settings.json").write_text(
        json.dumps({"packages": ["npm:pi-mine"], "npmCommand": command})
    )
    session_home = tmp_path / "session"
    agent_dir = session_home / ".pi" / "agent"

    pi_agent.write_pi_user_resources(agent_dir, session_home)

    assert json.loads((agent_dir / "settings.json").read_text())["npmCommand"] == command
    # A command set inside the session is not overwritten on the next launch.
    settings_path = agent_dir / "settings.json"
    settings = json.loads(settings_path.read_text())
    settings["npmCommand"] = ["pnpm"]
    settings_path.write_text(json.dumps(settings))
    pi_agent.write_pi_user_resources(agent_dir, session_home)
    assert json.loads(settings_path.read_text())["npmCommand"] == ["pnpm"]


def test_write_pi_user_resources_warns_on_an_unusable_agent_dir_override(
    tmp_path, monkeypatch, capsys
):
    # Otherwise this looks exactly like the bug write_pi_user_resources exists to fix.
    _pi_user_agent_dir(tmp_path, monkeypatch)
    missing = tmp_path / "not-a-directory"
    monkeypatch.setenv("PI_CODING_AGENT_DIR", f"  {missing}  ")
    session_home = tmp_path / "session"
    agent_dir = session_home / ".pi" / "agent"
    agent_dir.mkdir(parents = True)

    pi_agent.write_pi_user_resources(agent_dir, session_home)

    assert "PI_CODING_AGENT_DIR" in capsys.readouterr().err


def test_write_pi_user_resources_keeps_a_non_list_setting(tmp_path, monkeypatch):
    user_agent_dir = _pi_user_agent_dir(tmp_path, monkeypatch)
    (user_agent_dir / "settings.json").write_text(json.dumps({"themes": ["npm:user-theme"]}))
    session_home = tmp_path / "session"
    agent_dir = session_home / ".pi" / "agent"
    agent_dir.mkdir(parents = True)
    (agent_dir / "settings.json").write_text(json.dumps({"themes": {"name": "dark"}}))

    pi_agent.write_pi_user_resources(agent_dir, session_home)

    # Not a shape we understand, so it is left alone rather than deleted.
    assert json.loads((agent_dir / "settings.json").read_text())["themes"] == {"name": "dark"}


@pytest.mark.parametrize("entry", ["", "   ", "."])
def test_pi_local_entry_leaves_degenerate_entries_alone(tmp_path, entry):
    # Anchoring these would name the user's whole agent directory.
    assert pi_agent._pi_local_entry(entry, tmp_path, tmp_path, frozenset()) == entry


@pytest.mark.parametrize("yolo", [False, True])
def test_connect_pi_as_subagent_preserves_cloud_parent(fake_vllm, tmp_path, yolo):
    args = [
        "pi",
        "--url",
        BASE,
        "--as-subagent",
        "--no-launch",
        "--model",
        MODEL["id"],
    ]
    if yolo:
        args.insert(3, "--yolo")
    result = CliRunner().invoke(
        start.start_app,
        args,
    )
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    assert command[:2] == ["pi", "--extension"]
    assert command[2].endswith("agent_switch/agents/pi_subagent.ts")
    assert ("--approve" in command) is yolo
    assert "--provider" not in command
    assert "--model" not in command
    assert "PI_CODING_AGENT_DIR" not in result.output
    assert "export HOME=" not in result.output
    assert "AGENT_SWITCH_PI_SUBAGENT_API_KEY" not in result.output
    assert KEY not in result.output
    config_path = tmp_path / "agents" / "pi-subagent" / "subagent.json"
    _assert_env_set(result.output, "AGENT_SWITCH_PI_SUBAGENT_CONFIG", str(config_path))
    assert json.loads(config_path.read_text()) == {
        "baseUrl": f"{BASE}/v1",
        "apiKey": KEY,
        "model": MODEL["id"],
        "contextWindow": MODEL["context_length"],
        "maxTokens": 32000,
        "approve": yolo,
    }
    assert "Ask Pi to spawn a local agent." in result.output


def _pi_generated_model(tmp_path, as_subagent):
    if as_subagent:
        return json.loads((tmp_path / "agents" / "pi-subagent" / "subagent.json").read_text())
    return json.loads((tmp_path / "agents" / "pi" / ".pi" / "agent" / "models.json").read_text())[
        "providers"
    ]["agent-switch"]["models"][0]


@pytest.mark.parametrize("as_subagent", [False, True])
def test_connect_pi_output_limit(fake_vllm, tmp_path, as_subagent):
    args = ["pi", "--url", BASE, "--no-launch", "--max-tokens", "40000"]
    if as_subagent:
        args.append("--as-subagent")
    # Each launch regenerates the config, so the second run must honour the flag too.
    for _ in range(2):
        result = CliRunner().invoke(start.start_app, args)
        assert result.exit_code == 0, result.output
        assert "--max-tokens" not in _launch_command(result.output)
        config = _pi_generated_model(tmp_path, as_subagent)
        assert config["maxTokens"] == 40000
        assert config["contextWindow"] == MODEL["context_length"]


@pytest.mark.parametrize("as_subagent", [False, True])
def test_connect_pi_output_limit_capped_at_half_window(fake_vllm, tmp_path, as_subagent):
    args = ["pi", "--url", BASE, "--no-launch", "--max-tokens", "200000"]
    if as_subagent:
        args.append("--as-subagent")
    result = CliRunner().invoke(start.start_app, args)
    assert result.exit_code == 0, result.output
    assert "leaves too little" in result.output
    assert _pi_generated_model(tmp_path, as_subagent)["maxTokens"] == MODEL["context_length"] // 2


@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_connect_pi_invalid_output_limit(fake_vllm, tmp_path, value):
    result = CliRunner().invoke(start.start_app, ["pi", "--url", BASE, "--no-launch", "--max-tokens", value])
    assert result.exit_code != 0
    assert "--max-tokens" in result.output
    assert not list((tmp_path / "agents").rglob("models.json"))


@pytest.mark.parametrize("context", [{}, {"max_context_length": 32768}])
def test_write_pi_output_limit_context_metadata(tmp_path, context):
    path = tmp_path / "models.json"
    pi_agent.write_pi_config(BASE, "test-key", {"id": "test-model", **context}, path, max_tokens = 10000)
    model = json.loads(path.read_text())["providers"]["agent-switch"]["models"][0]
    assert model["maxTokens"] == 10000
    if context:
        assert model["contextWindow"] == 32768
    else:
        assert "contextWindow" not in model


def test_connect_pi_no_launch_windows_relocates_userprofile(fake_vllm, tmp_path, monkeypatch):
    # On native Windows Node resolves ~/.pi via USERPROFILE, not HOME, so the session
    # must point USERPROFILE at the relocated home or Pi reads the user's real ~/.pi.
    user_home = _pi_user_agent_dir(tmp_path, monkeypatch).parent.parent
    # Avoid pathlib selecting WindowsPath on this POSIX runner.
    monkeypatch.setattr(Path, "home", lambda: user_home)
    _simulate_windows(monkeypatch)
    result = CliRunner().invoke(start.start_app, ["pi", "--url", BASE, "--no-launch"])
    assert result.exit_code == 0, result.output
    home = tmp_path / "agents" / "pi"
    assert f'$env:HOME = "{home}"' in result.output
    assert f'$env:USERPROFILE = "{home}"' in result.output


def test_pi_launch_clears_screen_first(fake_vllm, monkeypatch):
    # Pi paints inline from the current cursor position (no alternate screen, no
    # clear on its first render), so the launcher hands it a clean screen. The
    # clear must come BEFORE the exec, and only on the launch path.
    calls = []
    monkeypatch.setattr(click, "clear", lambda: calls.append("clear"))
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/local/bin/pi")

    def run(command, env):
        calls.append("exec")
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(subprocess, "run", run)
    result = CliRunner().invoke(start.start_app, ["pi", "--url", BASE])
    assert result.exit_code == 0, result.output
    assert calls == ["clear", "exec"]


def test_pi_no_launch_does_not_clear(fake_vllm, monkeypatch):
    # The --no-launch recipe is meant to be read (and piped); never wipe it.
    calls = []
    monkeypatch.setattr(click, "clear", lambda: calls.append("clear"))
    result = CliRunner().invoke(start.start_app, ["pi", "--url", BASE, "--no-launch"])
    assert result.exit_code == 0, result.output
    assert calls == []


@pytest.mark.skipif(
    os.name == "nt",
    reason = "WSL-from-Linux scenario: a Windows pi shim under /mnt called from WSL "
    "(os.name is 'posix' under WSL), so this can't run on a native Windows runner.",
)
def test_connect_pi_wsl_windows_shim_relocates_userprofile(fake_vllm, monkeypatch):
    captured = {}
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr(shutil, "which", lambda _: "/mnt/c/Users/x/AppData/Roaming/npm/pi")

    def run(command, env):
        captured["env"] = env
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(subprocess, "run", run)
    result = CliRunner().invoke(start.start_app, ["pi", "--url", BASE])
    assert result.exit_code == 0, result.output
    home = captured["env"]["HOME"]
    # A Windows pi shim resolves ~/.pi via USERPROFILE, so it must match the session
    # HOME and ride the WSLENV bridge (with /p) so the path is translated for Windows.
    assert captured["env"]["USERPROFILE"] == home
    wslenv = captured["env"]["WSLENV"].split(":")
    assert "HOME/p" in wslenv
    assert "USERPROFILE/p" in wslenv


# ── --mcp / --mcp-all: session-only MCP mounting ─────────────────────


def test_connect_pi_mcp_writes_agent_mcp_json(fake_vllm, tmp_path, monkeypatch):
    _pi_user_agent_dir(tmp_path, monkeypatch)
    monkeypatch.setenv("GITHUB_TOKEN", "gh-secret")
    _mcp_registry()
    result = CliRunner().invoke(
        start.start_app, ["pi", "--url", BASE, "--no-launch", "--mcp", "context7", "--mcp", "github"]
    )
    assert result.exit_code == 0, result.output
    # Pi 1.1.0's built-in MCP reads <agentDir>/mcp.json; PI_CODING_AGENT_DIR is the session's.
    mcp_path = tmp_path / "agents" / "pi" / ".pi" / "agent" / "mcp.json"
    servers = json.loads(mcp_path.read_text())["mcpServers"]
    assert servers["context7"] == {
        "command": "npx",
        "args": ["-y", "@upstash/context7-mcp"],
        "env": {},
    }
    assert servers["github"] == {
        "url": "https://api.githubcopilot.com/mcp/",
        "headers": {"Authorization": "Bearer gh-secret"},
    }
    if os.name != "nt":
        assert mcp_path.stat().st_mode & 0o777 == 0o600
    assert "gh-secret" not in result.output


def test_connect_pi_mcp_all_mounts_every_registry_server(fake_vllm, tmp_path, monkeypatch):
    _pi_user_agent_dir(tmp_path, monkeypatch)
    monkeypatch.setenv("GITHUB_TOKEN", "gh-secret")
    _mcp_registry()
    result = CliRunner().invoke(start.start_app, ["pi", "--url", BASE, "--no-launch", "--mcp-all"])
    assert result.exit_code == 0, result.output
    mcp_path = tmp_path / "agents" / "pi" / ".pi" / "agent" / "mcp.json"
    assert set(json.loads(mcp_path.read_text())["mcpServers"]) == {"context7", "github"}


def test_connect_pi_mcp_url_mounts_without_a_registry(fake_vllm, tmp_path, monkeypatch):
    _pi_user_agent_dir(tmp_path, monkeypatch)
    result = CliRunner().invoke(
        start.start_app, ["pi", "--url", BASE, "--no-launch", "--mcp-url", "ev=http://127.0.0.1:18331/mcp"]
    )
    assert result.exit_code == 0, result.output
    mcp_path = tmp_path / "agents" / "pi" / ".pi" / "agent" / "mcp.json"
    servers = json.loads(mcp_path.read_text())["mcpServers"]
    assert servers == {"ev": {"url": "http://127.0.0.1:18331/mcp", "headers": {}}}


def test_connect_pi_mcp_stdio_mounts_without_a_registry(fake_vllm, tmp_path, monkeypatch):
    _pi_user_agent_dir(tmp_path, monkeypatch)
    result = CliRunner().invoke(
        start.start_app,
        ["pi", "--url", BASE, "--no-launch", "--mcp-stdio", "ev=npx -y @modelcontextprotocol/server-everything@2026.8.31"],
    )
    assert result.exit_code == 0, result.output
    mcp_path = tmp_path / "agents" / "pi" / ".pi" / "agent" / "mcp.json"
    servers = json.loads(mcp_path.read_text())["mcpServers"]
    assert servers == {
        "ev": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-everything@2026.8.31"],
            "env": {},
        }
    }


def test_connect_pi_mcp_stdio_shell_form(fake_vllm, tmp_path, monkeypatch):
    # fake_vllm stubs shutil.which to None; the shell form needs bash found.
    monkeypatch.setattr(shutil, "which", lambda name, path = None: "/bin/bash" if name == "bash" else None)
    _pi_user_agent_dir(tmp_path, monkeypatch)
    result = CliRunner().invoke(
        start.start_app,
        ["pi", "--url", BASE, "--no-launch", "--mcp-stdio", 'ev="npx -y @modelcontextprotocol/server-everything@2026.8.31"'],
    )
    assert result.exit_code == 0, result.output
    mcp_path = tmp_path / "agents" / "pi" / ".pi" / "agent" / "mcp.json"
    servers = json.loads(mcp_path.read_text())["mcpServers"]
    assert servers == {
        "ev": {
            "command": "bash",
            "args": ["-ic", "exec npx -y @modelcontextprotocol/server-everything@2026.8.31"],
            "env": {},
        }
    }


def test_connect_pi_without_mcp_flags_writes_no_mcp_json(fake_vllm, tmp_path, monkeypatch):
    _pi_user_agent_dir(tmp_path, monkeypatch)
    result = CliRunner().invoke(start.start_app, ["pi", "--url", BASE, "--no-launch"])
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "agents" / "pi" / ".pi" / "agent" / "mcp.json").exists()


def test_pi_mcp_state_cleared_on_rerun_without_flags(fake_vllm, tmp_path, monkeypatch):
    # A --no-launch session dir is reused, so the next run must delete the earlier mount.
    _pi_user_agent_dir(tmp_path, monkeypatch)
    monkeypatch.setenv("GITHUB_TOKEN", "gh-secret")
    _mcp_registry()
    result = CliRunner().invoke(start.start_app, ["pi", "--url", BASE, "--no-launch", "--mcp", "context7"])
    assert result.exit_code == 0, result.output
    mcp_path = tmp_path / "agents" / "pi" / ".pi" / "agent" / "mcp.json"
    assert mcp_path.exists()
    result = CliRunner().invoke(start.start_app, ["pi", "--url", BASE, "--no-launch"])
    assert result.exit_code == 0, result.output
    assert not mcp_path.exists()


def test_pi_mcp_with_as_subagent_fails(fake_vllm):
    _mcp_registry()
    result = CliRunner().invoke(
        start.start_app, ["pi", "--url", BASE, "--as-subagent", "--no-launch", "--mcp", "context7"]
    )
    assert result.exit_code == 1
    assert "--mcp/--mcp-all/--mcp-url/--mcp-oauth-url/--mcp-header/--mcp-stdio/--mcp-env cannot be combined with --as-subagent" in result.output


# ── Native launch (no --url/--provider) ──────────────────────────────


def _native_no_connect_pi(monkeypatch):
    set_start_attr(monkeypatch, "_connect",
        lambda *args, **kwargs: pytest.fail("native launch must not connect"),
    )


def test_native_pi_bare_adds_nothing(fake_vllm, monkeypatch):
    _native_no_connect_pi(monkeypatch)
    result = CliRunner().invoke(start.start_app, ["pi", "--no-launch"])
    assert result.exit_code == 0, result.output
    assert _launch_command(result.output) == ["pi"]
    assert "PI_CODING_AGENT_DIR" not in result.output
    assert "export HOME=" not in result.output


def test_native_pi_adds_mcp_config_without_moving_home(fake_vllm, tmp_path, monkeypatch):
    _native_no_connect_pi(monkeypatch)
    result = CliRunner().invoke(
        start.start_app, ["pi", "--no-launch", "--mcp-stdio", "ev=npx everything"]
    )
    assert result.exit_code == 0, result.output
    mcp_path = tmp_path / "agents" / "pi-native" / "mcp.json"
    assert _launch_command(result.output) == ["pi", "--mcp-config", str(mcp_path)]
    assert json.loads(mcp_path.read_text()) == {
        "mcpServers": {"ev": {"command": "npx", "args": ["everything"], "env": {}}}
    }
    assert "PI_CODING_AGENT_DIR" not in result.output
    assert "export HOME=" not in result.output


@pytest.mark.parametrize(
    "flag",
    [
        ["--model", "org/model"],
        ["--max-tokens", "4096"],
        ["--min-p", "0.1"],
        ["--no-model-load"],
        ["--as-subagent"],
        ["--persist"],
    ],
)
def test_native_pi_refuses_local_only_flags(fake_vllm, monkeypatch, flag):
    _native_no_connect_pi(monkeypatch)
    result = CliRunner().invoke(start.start_app, ["pi", "--no-launch", *flag])
    assert result.exit_code == 1
    assert "needs --url or --provider" in result.output
