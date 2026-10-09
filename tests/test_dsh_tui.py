# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the agent-switch authors. See LICENSE.

"""`agent-switch dsh-tui` / `dst`: the DeepSeek Harness TUI's session home, route row and launch guards."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

import agent_switch.start as start
from tests.cli_support import (
    KEY,
    MODEL,
    _assert_env_set,
    _assert_env_unset,
    _launch_command,
    _simulate_windows,
)
import sys
from agent_switch.agents import dsh as dsh_agent
from tests.start_split import set_start_attr

_REAL_WHICH = shutil.which
_REAL_RUN = subprocess.run
_JS_TAG = "tag:yaml.org,2002:js"
_HANDOFF_ENV = (
    "DSH_TUI_WORKSPACE_TARGET",
    "DSH_TUI_RESUME_SESSION",
    "DSH_TUI_RESUME_BACKEND",
    "DSH_TUI_BACKEND_HANDOFF",
    "DSH_TUI_PRESET",
)


def _invoke(*args):
    return CliRunner().invoke(start.start_app, list(args))


def _home(tmp_path) -> Path:
    return tmp_path / "agents" / "dsh-tui"


def _tui_command(output: str) -> list:
    """The recipe's launcher argv; on POSIX it follows the `env -u ...` that clears the handoff."""
    command = _launch_command(output)
    return command[next(i for i, part in enumerate(command) if part in dsh_agent._DSH_TUI_COMMANDS):]


@pytest.fixture()
def short_root(monkeypatch):
    # An ephemeral home under pytest's tmp_path can be too deep for dsh-tui's socket path.
    root = Path(tempfile.mkdtemp(prefix = "as-", dir = "/tmp"))
    set_start_attr(monkeypatch, "_agents_config_root", lambda: root / "agents")
    yield root
    shutil.rmtree(root, ignore_errors = True)


def _tui_row(text: str) -> dict:
    # compose keeps !!js as a tag on the node instead of failing or flattening it to a string.
    for row in yaml.compose(text).value:
        fields = {key.value: value for key, value in row.value}
        if fields["id"].value == "dsh-tui":
            return {key.value: value for key, value in fields["config"].value}
    raise AssertionError(f"no dsh-tui row in:\n{text}")


@pytest.mark.parametrize("agent", ["dsh-tui", "dst"])
def test_aliases_share_one_isolated_home(agent, fake_vllm, tmp_path):
    result = _invoke(agent, "--no-launch")
    assert result.exit_code == 0, result.output
    home = _home(tmp_path)
    _assert_env_set(result.output, "HOME", str(home))
    _assert_env_set(result.output, "USERPROFILE", str(home))
    _assert_env_set(result.output, "DSH_HOME", str(home / ".dsh"))
    _assert_env_set(result.output, "DSH_TUI_SESSION_ROOT", str(home / ".dsh" / "sessions"))
    _assert_env_set(result.output, "AGENT_SWITCH_API_KEY", KEY)
    _assert_env_set(result.output, "DSH_TELEMETRY_DISABLED", "1")
    assert _tui_command(result.output) == [
        "dsh-tui",
        "--patch",
        str(home / dsh_agent._DSH_PATCH_FILE),
    ]


def test_patch_pins_route_model_and_backend_in_the_tui_row(fake_vllm, tmp_path):
    result = _invoke("dsh-tui", "--no-launch", "--header", "X-Test=1", "--reasoning-effort", "low")
    assert result.exit_code == 0, result.output
    text = (_home(tmp_path) / dsh_agent._DSH_PATCH_FILE).read_text()
    ids = [dict((k.value, v) for k, v in row.value)["id"].value for row in yaml.compose(text).value]
    assert ids == ["llm-pi-ai", "agent-default-model", "dsh-tui"]
    assert "X-Test" in text and "chatTemplateKwargs" in text
    row = _tui_row(text)
    plain = {name: node.value for name, node in row.items() if node.tag != _JS_TAG}
    assert plain == {
        "provider": "agent-switch",
        "model": MODEL["id"],
        "fullscreen": "true",
        "terminalImages": "true",
        "effort": "max",
        "backend": "dsh",
    }
    bindings = {name: node.value for name, node in row.items() if node.tag == _JS_TAG}
    assert bindings == {
        "preset": "process.env.DSH_TUI_PRESET ?? undefined",
        "workspace": "process.env.DSH_TUI_WORKSPACE_TARGET ?? undefined",
        "sessionId": "process.env.DSH_TUI_RESUME_SESSION ?? undefined",
    }


def test_env_pins_backend_and_clears_launcher_handoff(fake_vllm):
    result = _invoke("dsh-tui", "--no-launch")
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    for name in _HANDOFF_ENV:
        if os.name == "nt":
            _assert_env_unset(result.output, name)
        else:
            # A POSIX recipe's last line removes them with env -u rather than emptying them.
            assert command[command.index(name) - 1] == "-u", command
    _assert_env_set(result.output, "DSH_TUI_BACKEND", "dsh")
    _assert_env_set(result.output, "DSH_PERMISSION_MODE", "workspace-write")


def test_pnpm_store_and_cache_stay_in_the_session_home(fake_vllm, tmp_path, monkeypatch):
    monkeypatch.setenv("PNPM_HOME", "/real/pnpm")
    monkeypatch.setenv("pnpm_config_store_dir", "/real/store")
    result = _invoke("dsh-tui", "--no-launch")
    assert result.exit_code == 0, result.output
    home = _home(tmp_path)
    pnpm_home = home / ".local" / "share" / "pnpm"
    _assert_env_set(result.output, "PNPM_HOME", str(pnpm_home))
    for prefix in ("pnpm_config_", "npm_config_"):
        _assert_env_set(result.output, prefix + "store_dir", str(pnpm_home / "store"))
        _assert_env_set(result.output, prefix + "cache_dir", str(home / ".cache" / "pnpm"))


def test_first_run_gates_are_seeded_once_in_the_session_home(fake_vllm, tmp_path):
    state = _home(tmp_path) / ".dsh-tui"
    assert _invoke("dsh-tui", "--no-launch").exit_code == 0
    assert json.loads((state / "onboarding.json").read_text()) == {"completed": True, "version": 1}
    assert json.loads((state / "home.json").read_text()) == {"seen": True}
    (state / "home.json").write_text('{"seen": true, "kept": 1}')
    assert _invoke("dst", "--no-launch").exit_code == 0
    assert json.loads((state / "home.json").read_text()) == {"seen": True, "kept": 1}


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--as-subagent"], "--as-subagent is not supported for dsh-tui"),
        (["--compact-at", "0.8"], "--compact-at"),
        (["--profile", "web"], "--profile"),
        (["--profile=web"], "--profile"),
        (["--from-default-profile", "web"], "--from-default-profile"),
        (["--backend", "claude"], "--backend"),
        (["--backend=codex"], "--backend"),
        (["hi", "--backend", "codex"], "--backend"),
        (["update"], "update"),
        (["migrate", "claude-code"], "migrate"),
        (["doctor"], "doctor"),
        (["safe", "--rescue"], "safe"),
        (["version"], "version"),
        (["help"], "help"),
        (["-v"], "-v"),
        (["--version"], "--version"),
    ],
)
def test_rejected_before_any_server_traffic(args, message, fake_vllm):
    result = _invoke("dsh-tui", "--no-launch", *args)
    assert result.exit_code == 1, result.output
    assert message in result.output
    assert fake_vllm == []


@pytest.mark.parametrize(
    "args",
    [
        ["--backend", "dsh"],
        ["--", "help"],
        ["--", "--backend", "claude"],
        ["fix", "the", "help", "text"],
        ["--resume", "abc"],
        ["-c"],
    ],
)
def test_session_arguments_follow_our_patch_unchanged(args, fake_vllm, tmp_path):
    result = _invoke("dsh-tui", "--no-launch", *args)
    assert result.exit_code == 0, result.output
    patch = str(_home(tmp_path) / dsh_agent._DSH_PATCH_FILE)
    assert _tui_command(result.output) == ["dsh-tui", "--patch", patch, *args]
    assert "replaces whole config blocks" not in result.output


def test_caller_patch_follows_ours_and_warns_once(fake_vllm, tmp_path):
    result = _invoke("dsh-tui", "--no-launch", "--patch", "mine.yml", "--patch=two.yml")
    assert result.exit_code == 0, result.output
    patch = str(_home(tmp_path) / dsh_agent._DSH_PATCH_FILE)
    assert _tui_command(result.output) == [
        "dsh-tui", "--patch", patch, "--patch", "mine.yml", "--patch=two.yml",
    ]
    assert result.output.count("replaces whole config blocks") == 1


def test_native_windows_requires_yolo(fake_vllm, monkeypatch, tmp_path):
    _simulate_windows(monkeypatch)
    refused = _invoke("dsh-tui", "--no-launch")
    assert refused.exit_code == 1, refused.output
    assert "--yolo" in refused.output
    assert fake_vllm == []
    accepted = _invoke("dsh-tui", "--no-launch", "--yolo")
    assert accepted.exit_code == 0, accepted.output
    home = _home(tmp_path)
    assert f'$env:USERPROFILE = "{home}"' in accepted.output
    assert '$env:DSH_PERMISSION_MODE = "danger-full-access"' in accepted.output


@pytest.mark.skipif(os.name == "nt", reason = "WSL scenario")
@pytest.mark.parametrize("args", [[], ["--yolo"]])
def test_wsl_windows_tui_is_refused(args, fake_vllm, monkeypatch):
    # WSL can hand a Windows process a cleared variable only as "", which dsh-tui reads as set.
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    shim = "/mnt/c/Users/x/AppData/Roaming/npm/dsh-tui"
    monkeypatch.setattr(
        shutil, "which", lambda name, path = None: shim if name == "dsh-tui" else None
    )
    result = _invoke("dsh-tui", "--no-launch", *args)
    assert result.exit_code == 1, result.output
    assert "WSL" in result.output
    assert fake_vllm == []


def test_launch_without_a_terminal_is_refused_before_server_traffic(fake_vllm, monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: pytest.fail("must not launch")
    )
    result = _invoke("dsh-tui")
    assert result.exit_code == 1, result.output
    assert "terminal" in result.output
    assert fake_vllm == []


# Launch tests use real stub launchers on a private PATH: identity comes from their content.
_TUI_MARKER = "@deepseek-harness-tui/dsh-tui"
_HARNESS_MARKER = "@deepseek-ai/dsh"


def _stub(directory: Path, name: str, marker: str = "") -> Path:
    directory.mkdir(parents = True, exist_ok = True)
    path = directory / name
    path.write_text(f"#!/bin/sh\n# {marker}\nexit 0\n")
    path.chmod(0o755)
    return path


def _launchable(monkeypatch, tmp_path, *directories: Path) -> None:
    set_start_attr(monkeypatch, "_get_has_terminal", lambda: True)
    monkeypatch.setattr(shutil, "which", _REAL_WHICH)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "missing-home")
    monkeypatch.setenv("PATH", os.pathsep.join(str(d) for d in directories))


def _launch(monkeypatch, argv, returncode = 0):
    captured = {}

    def run(command, env = None, **kwargs):
        if env is None:
            # Identity probes of the stub launchers run for real; only the agent launch is caught.
            return _REAL_RUN(command, **kwargs)
        captured["command"] = command
        captured["env"] = env
        captured["home_existed"] = Path(env["HOME"]).is_dir()
        found = _REAL_WHICH("dsh", path = env["PATH"])
        captured["dsh"] = os.path.realpath(found) if found else None
        return SimpleNamespace(returncode = returncode)

    monkeypatch.setattr(subprocess, "run", run)
    return CliRunner().invoke(start.start_app, argv), captured


@pytest.mark.skipif(os.name == "nt", reason = "POSIX stub launchers")
@pytest.mark.parametrize(("agent", "returncode"), [("dsh-tui", 0), ("dst", 3)])
def test_launch_uses_the_real_tui_past_a_foreign_shadow(
    agent, returncode, fake_vllm, short_root, monkeypatch, tmp_path
):
    shadow, real = tmp_path / "shadow", tmp_path / "real"
    _stub(shadow, "dsh-tui")
    dst = _stub(real, "dst", _TUI_MARKER)
    harness = _stub(real, "dsh", _HARNESS_MARKER)
    _launchable(monkeypatch, tmp_path, shadow, real)
    monkeypatch.setenv("DSH_PERMISSION_MODE", "danger-full-access")
    monkeypatch.setenv("DSH_TUI_BACKEND_HANDOFF", "claude")
    monkeypatch.setenv("DSH_TUI_WORKSPACE_TARGET", "/elsewhere")
    result, captured = _launch(monkeypatch, [agent, "--no-persist"], returncode)
    assert result.exit_code == returncode, result.output
    command, env = captured["command"], captured["env"]
    assert command[:2] == [str(dst), "--patch"]
    assert command[2] == os.path.join(env["HOME"], dsh_agent._DSH_PATCH_FILE)
    assert env["DSH_PERMISSION_MODE"] == "workspace-write"
    assert env["DSH_TUI_BACKEND"] == "dsh"
    # Compare names only: a failing repr of the child env would print the caller's secrets.
    assert not {"DSH_TUI_BACKEND_HANDOFF", "DSH_TUI_WORKSPACE_TARGET"} & set(env)
    assert captured["dsh"] == os.path.realpath(harness)
    # The ephemeral home exists for the run and is gone once it ends, success or failure.
    assert captured["home_existed"]
    assert not Path(env["HOME"]).exists()


@pytest.mark.skipif(os.name == "nt", reason = "POSIX stub launchers")
@pytest.mark.parametrize("agent", ["dsh-tui", "dst"])
def test_launch_keeps_the_session_home_by_default(agent, fake_vllm, short_root, monkeypatch, tmp_path):
    real = tmp_path / "real"
    _stub(real, "dsh-tui", _TUI_MARKER)
    _stub(real, "dst", _TUI_MARKER)
    _stub(real, "dsh", _HARNESS_MARKER)
    _launchable(monkeypatch, tmp_path, real)
    result, captured = _launch(monkeypatch, [agent])
    assert result.exit_code == 0, result.output
    home = short_root / "agents" / "dsh-tui"
    assert captured["env"]["HOME"] == str(home)
    assert captured["home_existed"] and home.is_dir()

    # What the TUI saved during one run is there for the next one.
    (home / ".dsh").mkdir(parents = True, exist_ok = True)
    (home / ".dsh" / "settings.yaml").write_text("theme: dark\n")
    (home / ".dsh-tui").mkdir(parents = True, exist_ok = True)
    (home / ".dsh-tui" / "theme.json").write_text('{"theme": "dark"}\n')
    result, captured = _launch(monkeypatch, [agent])
    assert result.exit_code == 0, result.output
    assert captured["env"]["HOME"] == str(home)
    assert (home / ".dsh" / "settings.yaml").read_text() == "theme: dark\n"
    assert (home / ".dsh-tui" / "theme.json").read_text() == '{"theme": "dark"}\n'


@pytest.mark.skipif(os.name == "nt", reason = "POSIX stub launchers")
def test_no_persist_launch_uses_a_throwaway_home(fake_vllm, short_root, monkeypatch, tmp_path):
    real = tmp_path / "real"
    _stub(real, "dsh-tui", _TUI_MARKER)
    _stub(real, "dsh", _HARNESS_MARKER)
    _launchable(monkeypatch, tmp_path, real)
    result, captured = _launch(monkeypatch, ["dsh-tui", "--no-persist"])
    assert result.exit_code == 0, result.output
    home = Path(captured["env"]["HOME"])
    assert home.parent == short_root / "agents" / ".tmp"
    assert home.name.startswith("agent-switch-dsh-tui-")
    assert captured["home_existed"]
    assert not home.exists()


@pytest.mark.skipif(os.name == "nt", reason = "POSIX stub launchers")
def test_persist_launch_still_keeps_the_home(fake_vllm, short_root, monkeypatch, tmp_path):
    real = tmp_path / "real"
    _stub(real, "dsh-tui", _TUI_MARKER)
    _stub(real, "dsh", _HARNESS_MARKER)
    _launchable(monkeypatch, tmp_path, real)
    result, captured = _launch(monkeypatch, ["dsh-tui", "--persist"])
    assert result.exit_code == 0, result.output
    home = short_root / "agents" / "dsh-tui"
    assert captured["env"]["HOME"] == str(home)
    assert captured["home_existed"] and home.is_dir()


@pytest.mark.skipif(os.name == "nt", reason = "POSIX stub launchers")
def test_foreign_tui_alone_is_refused(fake_vllm, short_root, monkeypatch, tmp_path):
    shadow = tmp_path / "shadow"
    _stub(shadow, "dsh-tui")
    _stub(shadow, "dsh", _HARNESS_MARKER)
    _launchable(monkeypatch, tmp_path, shadow)
    set_start_attr(monkeypatch, "_install_agent", lambda *args: None)
    result, captured = _launch(monkeypatch, ["dsh-tui"])
    assert result.exit_code == 1, result.output
    assert "is not the DeepSeek Harness TUI" in result.output
    assert captured == {}
    assert fake_vllm == []


@pytest.mark.skipif(os.name == "nt", reason = "POSIX stub launchers")
def test_nested_dsh_shadow_is_bypassed_through_the_child_path(fake_vllm, monkeypatch, tmp_path):
    shadow, real = tmp_path / "shadow", tmp_path / "real"
    _stub(shadow, "dsh")
    _stub(real, "dsh-tui", _TUI_MARKER)
    harness = _stub(real, "dsh", _HARNESS_MARKER)
    _launchable(monkeypatch, tmp_path, shadow, real)
    original_path = os.environ["PATH"]
    result, captured = _launch(monkeypatch, ["dsh-tui", "--persist", "--resume", "abc"])
    assert result.exit_code == 0, result.output
    env = captured["env"]
    home = _home(tmp_path)
    assert env["HOME"] == str(home) and home.is_dir()
    assert captured["command"][3:] == ["--resume", "abc"]
    assert captured["dsh"] == os.path.realpath(harness)
    # Only `dsh` is redirected: the original PATH follows unchanged.
    assert env["PATH"].endswith(os.pathsep + original_path)


@pytest.mark.skipif(os.name == "nt", reason = "POSIX stub launchers")
def test_path_is_left_alone_when_the_first_dsh_is_the_harness(
    fake_vllm, short_root, monkeypatch, tmp_path
):
    real = tmp_path / "real"
    _stub(real, "dsh-tui", _TUI_MARKER)
    _stub(real, "dsh", _HARNESS_MARKER)
    _launchable(monkeypatch, tmp_path, real)
    result, captured = _launch(monkeypatch, ["dsh-tui"])
    assert result.exit_code == 0, result.output
    assert captured["env"]["PATH"] == os.environ["PATH"]


@pytest.mark.skipif(os.name == "nt", reason = "WSL scenario")
def test_wsl_windows_dsh_under_a_linux_tui_is_refused(fake_vllm, short_root, monkeypatch, tmp_path):
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    entries = {"/usr/local/bin/dsh-tui", "/mnt/c/npm/dsh"}

    def which(name, path = None):
        if os.path.isabs(name):
            return name if name in entries else None
        for directory in (path or os.environ.get("PATH", "")).split(os.pathsep):
            if f"{directory}/{name}" in entries:
                return f"{directory}/{name}"
        return None

    set_start_attr(monkeypatch, "_get_has_terminal", lambda: True)
    monkeypatch.setattr(shutil, "which", which)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "missing-home")
    monkeypatch.setenv("PATH", "/usr/local/bin:/mnt/c/npm")
    set_start_attr(monkeypatch, "get_is_deepseek_harness_tui_executable", lambda *_: True)
    set_start_attr(monkeypatch, "is_deepseek_harness_executable", lambda *_: True)
    result, captured = _launch(monkeypatch, ["dsh-tui"])
    assert result.exit_code == 1, result.output
    assert "inside WSL" in result.output
    assert captured == {}
    assert fake_vllm == []


def test_dsh_keeps_its_own_home_and_no_tui_env(fake_vllm, tmp_path):
    result = _invoke("dsh", "--no-launch")
    assert result.exit_code == 0, result.output
    assert "export HOME=" not in result.output
    assert "DSH_TUI" not in result.output
    assert "PNPM_HOME" not in result.output
    patch = tmp_path / "agents" / "dsh" / dsh_agent._DSH_PATCH_FILE
    assert _launch_command(result.output) == ["dsh", "web", "--patch", str(patch)]
    assert "dsh-tui" not in patch.read_text()


@pytest.mark.skipif(os.name == "nt", reason = "asserts POSIX symlinks")
def test_user_resources_are_linked_and_agents_skills_stays_out(fake_vllm, tmp_path, monkeypatch):
    user_home = tmp_path / "user-home"
    monkeypatch.setattr(Path, "home", lambda: user_home)
    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.delenv("DSH_HOME", raising = False)
    source = user_home / ".dsh"
    (source / "skills" / "my-skill").mkdir(parents = True)
    (source / "AGENTS.md").write_text("user instructions\n")

    result = _invoke("dsh-tui", "--no-launch")

    assert result.exit_code == 0, result.output
    home = _home(tmp_path)
    dsh_home = home / ".dsh"
    assert (dsh_home / "AGENTS.md").resolve() == (source / "AGENTS.md").resolve()
    assert (dsh_home / "skills").resolve() == (source / "skills").resolve()
    # HOME moved, but the TUI session is not given the user's ~/.agents/skills.
    assert not (home / ".agents").exists()
    assert not (user_home / ".agents").exists()


# ── dsh-tui's per-session socket must never land outside the session home ──


def _padded_root(tmp_path, char: str, home_bytes: int) -> Path:
    """An agents root whose dsh-tui session home (root/dsh-tui) is ``home_bytes`` long."""
    fixed = len(os.fsencode(str(tmp_path))) + len(b"/") + len(b"/agents/dsh-tui")
    count, rest = divmod(home_bytes - fixed, len(char.encode()))
    if count < 1:
        pytest.skip("tmp_path is too deep to build this home")
    return tmp_path / (char * count + "x" * rest) / "agents"


def test_deep_session_home_is_refused_before_server_traffic(fake_vllm, monkeypatch, tmp_path):
    deep = tmp_path / ("d" * 80) / "agents"
    set_start_attr(monkeypatch, "_agents_config_root", lambda: deep)
    recipe = _invoke("dsh-tui", "--no-launch")
    assert recipe.exit_code == 1, recipe.output
    assert "AGENT_SWITCH_HOME" in recipe.output
    set_start_attr(monkeypatch, "_get_has_terminal", lambda: True)
    launch = _invoke("dst")
    assert launch.exit_code == 1, launch.output
    assert "AGENT_SWITCH_HOME" in launch.output
    assert fake_vllm == []
    assert not deep.exists()


def test_socket_bound_counts_bytes_not_characters(fake_vllm, monkeypatch, tmp_path):
    # 90 bytes is one past the Linux bound (inject dir plus one name byte within 107), in fewer
    # characters.
    monkeypatch.setattr(sys, "platform", "linux")
    root = _padded_root(tmp_path, "é", 90)
    assert len(str(root / "dsh-tui")) < 89 < len(os.fsencode(str(root / "dsh-tui")))
    set_start_attr(monkeypatch, "_agents_config_root", lambda: root)
    result = _invoke("dsh-tui", "--no-launch")
    assert result.exit_code == 1, result.output
    assert "AGENT_SWITCH_HOME" in result.output


@pytest.mark.parametrize(("platform", "accepted"), [("linux", True), ("darwin", False)])
def test_socket_bound_follows_the_platform(platform, accepted, fake_vllm, monkeypatch, tmp_path):
    # macOS and the BSDs cut socket paths at 104 bytes, Linux at 108.
    monkeypatch.setattr(sys, "platform", platform)
    root = _padded_root(tmp_path, "h", 87)
    set_start_attr(monkeypatch, "_agents_config_root", lambda: root)
    result = _invoke("dsh-tui", "--no-launch")
    assert (result.exit_code == 0) is accepted, result.output


def test_windows_named_pipe_needs_no_home_bound(fake_vllm, monkeypatch, tmp_path):
    _simulate_windows(monkeypatch)
    set_start_attr(monkeypatch, "_agents_config_root", lambda: tmp_path / ("d" * 80) / "agents")
    result = _invoke("dsh-tui", "--no-launch", "--yolo")
    assert result.exit_code == 0, result.output


# ── inherited other-case spellings of the cache selectors must not beat the pins ──


def _node_windows_env(env: dict) -> dict:
    # Node 22 lib/child_process.js normalizeSpawnArguments on win32: sort the names and keep the
    # first spelling of each, compared upper-cased.
    kept = {}
    for name in sorted(env):
        kept.setdefault(name.upper(), env[name])
    return kept


@pytest.mark.skipif(os.name == "nt", reason = "POSIX stub launchers")
def test_linux_child_env_drops_upper_and_mixed_case_cache_selectors(
    fake_vllm, short_root, monkeypatch, tmp_path
):
    real = tmp_path / "real"
    _stub(real, "dsh-tui", _TUI_MARKER)
    _stub(real, "dsh", _HARNESS_MARKER)
    _launchable(monkeypatch, tmp_path, real)
    # pnpm 11 on Linux reads PNPM_CONFIG_STORE_DIR too, and prefers it over the lower-case form.
    monkeypatch.setenv("PNPM_CONFIG_STORE_DIR", "/real/store")
    monkeypatch.setenv("Pnpm_Config_Cache_Dir", "/real/cache")
    monkeypatch.setenv("NPM_CONFIG_STORE_DIR", "/real/npm-store")
    result, captured = _launch(monkeypatch, ["dsh-tui", "--persist"])
    assert result.exit_code == 0, result.output
    env = captured["env"]
    # Compare names only: a failing repr of the child env would print the caller's secrets.
    assert not {"PNPM_CONFIG_STORE_DIR", "Pnpm_Config_Cache_Dir", "NPM_CONFIG_STORE_DIR"} & set(env)
    store = str(short_root / "agents" / "dsh-tui" / ".local" / "share" / "pnpm" / "store")
    assert env["pnpm_config_store_dir"] == store


@pytest.mark.skipif(os.name == "nt", reason = "simulates Windows on a POSIX host")
def test_windows_child_env_keeps_the_cache_pins_after_node_normalization(
    fake_vllm, short_root, monkeypatch, tmp_path
):
    real = tmp_path / "real"
    _stub(real, "dsh-tui", _TUI_MARKER)
    _stub(real, "dsh", _HARNESS_MARKER)
    _launchable(monkeypatch, tmp_path, real)
    # Windows hands Python upper-case names only.
    for name in ("PNPM_CONFIG_STORE_DIR", "PNPM_CONFIG_CACHE_DIR", "NPM_CONFIG_STORE_DIR"):
        monkeypatch.setenv(name, "C:/real")
    _simulate_windows(monkeypatch)
    result, captured = _launch(monkeypatch, ["dsh-tui", "--persist", "--yolo"])
    assert result.exit_code == 0, result.output
    seen = _node_windows_env(captured["env"])
    home = short_root / "agents" / "dsh-tui"
    store = str(home / ".local" / "share" / "pnpm" / "store")
    assert seen["PNPM_CONFIG_STORE_DIR"] == seen["NPM_CONFIG_STORE_DIR"] == store
    assert seen["PNPM_CONFIG_CACHE_DIR"] == str(home / ".cache" / "pnpm")


# ── cleared launcher variables must arrive unset, never as "" ──


@pytest.mark.skipif(os.name == "nt", reason = "POSIX recipe")
def test_posix_recipe_removes_handoff_vars_instead_of_emptying_them(
    fake_vllm, monkeypatch, tmp_path
):
    monkeypatch.setenv("PNPM_CONFIG_STORE_DIR", "/real/store")
    result = _invoke("dsh-tui", "--no-launch", "--resume", "abc")
    assert result.exit_code == 0, result.output
    stub = _stub(tmp_path / "bin", "dsh-tui")
    stub.write_text('#!/bin/sh\nenv\necho "--argv--"\nprintf "%s\\n" "$@"\n')
    stale = {name: "stale" for name in (*_HANDOFF_ENV, "PNPM_CONFIG_STORE_DIR")}
    # Run the recipe's self-contained last line the way a user pastes it.
    ran = _REAL_RUN(
        ["/bin/sh", "-c", result.output.splitlines()[-1]],
        env = {"PATH": f"{stub.parent}:/usr/bin:/bin", **stale},
        capture_output = True,
        text = True,
    )
    env_text, _, argv_text = ran.stdout.partition("--argv--\n")
    received = dict(line.split("=", 1) for line in env_text.splitlines() if "=" in line)
    assert not set(stale) & set(received), {name: received[name] for name in set(stale) & set(received)}
    home = _home(tmp_path)
    assert received["HOME"] == str(home)
    patch = str(home / dsh_agent._DSH_PATCH_FILE)
    assert argv_text.splitlines() == ["--patch", patch, "--resume", "abc"]


@pytest.mark.skipif(os.name == "nt", reason = "POSIX recipe")
def test_recipe_clears_uppercase_cache_selectors_a_later_shell_sets(fake_vllm, monkeypatch, tmp_path):
    # pnpm 11 on Linux prefers PNPM_CONFIG_* over our lower-case pins, and the receiving shell can
    # set them after generation, so the last line clears them even when absent while generating.
    for name in ("PNPM_CONFIG_STORE_DIR", "PNPM_CONFIG_CACHE_DIR", "NPM_CONFIG_STORE_DIR", "NPM_CONFIG_CACHE_DIR"):
        monkeypatch.delenv(name, raising = False)
    result = _invoke("dsh-tui", "--no-launch")
    assert result.exit_code == 0, result.output
    stub = _stub(tmp_path / "bin", "dsh-tui")
    stub.write_text("#!/bin/sh\nenv\n")
    probe = tmp_path / "probe"
    foreign = {
        "PNPM_CONFIG_STORE_DIR": str(probe / "foreign-store"),
        "PNPM_CONFIG_CACHE_DIR": str(probe / "foreign-cache"),
        "NPM_CONFIG_STORE_DIR": str(probe / "foreign-npm-store"),
        "NPM_CONFIG_CACHE_DIR": str(probe / "foreign-npm-cache"),
    }
    # Paste the exact self-contained last line into a shell that only later sets the upper case.
    ran = _REAL_RUN(
        ["/bin/sh", "-c", result.output.splitlines()[-1]],
        env = {"PATH": f"{stub.parent}:/usr/bin:/bin", **foreign},
        capture_output = True,
        text = True,
    )
    received = dict(line.split("=", 1) for line in ran.stdout.splitlines() if "=" in line)
    home = str(_home(tmp_path))
    offenders = {
        name: value
        for name, value in received.items()
        if name.casefold().startswith(("pnpm_config_", "npm_config_")) and not value.startswith(home)
    }
    assert not offenders, offenders
    assert received["pnpm_config_store_dir"] == str(_home(tmp_path) / ".local" / "share" / "pnpm" / "store")
    assert received["npm_config_cache_dir"] == str(_home(tmp_path) / ".cache" / "pnpm")


def test_windows_recipe_removes_handoff_vars(fake_vllm, monkeypatch):
    _simulate_windows(monkeypatch)
    result = _invoke("dsh-tui", "--no-launch", "--yolo")
    assert result.exit_code == 0, result.output
    for name in _HANDOFF_ENV:
        assert f"Remove-Item Env:{name}" in result.output


# ── launcher option values and the literal `--` payload are raw tokens ──


@pytest.mark.parametrize(
    "args",
    [
        ["--patch", "--backend=claude"],
        ["--patch", "update"],
        ["--patch", "--as-subagent"],
        ["--patch", "--profile", "--patch=two.yml"],
    ],
)
def test_host_option_values_are_raw_tokens(args, fake_vllm):
    result = _invoke("dst", "--no-launch", *args)
    assert result.exit_code == 0, result.output
    assert _launch_command(result.output)[-len(args):] == args
    assert result.output.count("replaces whole config blocks") == 1


@pytest.mark.parametrize(
    "args",
    [
        ["--", "--as-subagent"],
        ["--", "--profile", "web"],
        ["--", "update"],
        ["hi", "--profile", "web"],
    ],
)
def test_prompt_text_is_never_refused(args, fake_vllm):
    result = _invoke("dsh-tui", "--no-launch", *args)
    assert result.exit_code == 0, result.output
    assert _launch_command(result.output)[-len(args):] == args


@pytest.mark.parametrize("args", [["hi", "--as-subagent"], ["WORKSPACE", "--profile", "web"]])
def test_genuine_disallowed_flags_are_still_refused(args, fake_vllm, tmp_path):
    # An existing path is the launcher's workspace target, so options after it stay launcher options.
    args = [str(tmp_path) if arg == "WORKSPACE" else arg for arg in args]
    result = _invoke("dsh-tui", "--no-launch", *args)
    assert result.exit_code == 1, result.output
    assert fake_vllm == []