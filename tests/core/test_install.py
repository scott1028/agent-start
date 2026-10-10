# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See LICENSE
# Ported from unsloth_cli/tests/test_start.py (unsloth commit 8e11ba15e); see NOTICE.md.

"""Install prompts, npm resolution and PATH-augmented version probes."""

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer

from agent_switch.core import (
    install as core_install,
)
from tests.cli_support import _path_aware_which, _simulate_windows
from tests.start_split import set_start_attr


def test_install_agent_prompts_then_installs(monkeypatch):
    # TTY + yes: run the documented install command, then re-resolve the now-present binary.
    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: True)
    set_start_attr(monkeypatch, "_npm_executable", lambda: "/usr/local/bin/npm")
    ran = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, *a, **k: ran.append(command) or SimpleNamespace(returncode = 0),
    )
    # _install_agent only re-resolves after installing (the pre-install check is the
    # caller's job), so `which` reports the now-present binary.
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/local/bin/codex")
    executable = core_install._install_agent("codex", "npm install -g @openai/codex")
    assert executable == "/usr/local/bin/codex"
    assert ran == [["/usr/local/bin/npm", "install", "-g", "@openai/codex"]]


def test_install_agent_uses_powershell_on_windows(monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: True)
    ran = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, *a, **k: ran.append(command) or SimpleNamespace(returncode = 0),
    )
    monkeypatch.setattr(shutil, "which", lambda _: r"C:\Users\samle\bin\hermes.exe")

    install_hint = "& ([scriptblock]::Create((irm https://x/install.ps1))) -SkipSetup"
    executable = core_install._install_agent("hermes", install_hint)

    assert executable == r"C:\Users\samle\bin\hermes.exe"
    # -ExecutionPolicy Bypass (process-scoped) lets npm's npm.ps1 wrapper and irm|iex
    # scripts run even when the machine policy is the Windows default Restricted.
    assert ran == [
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", install_hint]
    ]


def test_install_agent_windows_failure_hints_execution_policy(monkeypatch, capsys):
    # A failed install on Windows points the user at the per-user execution-policy fix:
    # our subprocess bypasses the policy, but their own shell may still block npm.ps1
    # (PSSecurityException) when they run the install by hand.
    _simulate_windows(monkeypatch)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: True)
    set_start_attr(monkeypatch, "_npm_executable", lambda: r"C:\Users\me\AppData\Roaming\npm\npm.cmd"
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode = 1),
    )
    monkeypatch.setattr(shutil, "which", lambda _: None)

    with pytest.raises(typer.Exit):
        core_install._install_agent("codex", "npm install -g @openai/codex")

    err = capsys.readouterr().err
    assert "Install command failed" in err
    assert "Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned" in err


def test_install_command_uses_resolved_npm_cmd_on_windows(monkeypatch):
    _simulate_windows(monkeypatch)
    set_start_attr(monkeypatch, "_npm_executable", lambda: r"C:\Program Files\nodejs\npm.cmd")

    command, env = core_install._install_command(core_install._npm_install_hint("@openai/codex"))

    assert command == [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        "& 'C:\\Program Files\\nodejs\\npm.cmd' install -g '@openai/codex'",
    ]
    assert env is not None


def test_install_agent_posix_failure_omits_execution_policy_hint(monkeypatch, capsys):
    # The execution-policy hint is Windows-only; a POSIX install failure must not mention it.
    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: True)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode = 1),
    )

    with pytest.raises(typer.Exit):
        core_install._install_agent("codex", "npm install -g @openai/codex")

    err = capsys.readouterr().err
    assert "Install command failed" in err
    assert "Set-ExecutionPolicy" not in err


def test_npm_install_hint_uses_user_prefix_on_posix(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    hint = core_install._npm_install_hint("@openai/codex")

    assert shlex.split(hint) == [
        "npm",
        "install",
        "-g",
        "--prefix",
        str(tmp_path / ".local"),
        "@openai/codex",
    ]


@pytest.mark.skipif(os.name == "nt", reason = "WSL scenario")
def test_npm_executable_skips_wsl_shim_and_finds_native_npm(monkeypatch):
    # WSL inherits the Windows PATH, so a shim can precede a usable native npm.
    native = "/usr/bin/npm"
    shim = "/mnt/c/Program Files/nodejs/npm"
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setenv("PATH", "/mnt/c/Program Files/nodejs:/usr/bin")

    found = {"/mnt/c/Program Files/nodejs": shim, "/usr/bin": native}

    def fake_which(name, path = None):
        if os.path.isabs(name):
            return name
        # No path given means search all of PATH, so the shim wins as it does in WSL.
        for entry in (path or os.environ["PATH"]).split(os.pathsep):
            if entry in found:
                return found[entry]
        return None

    monkeypatch.setattr(shutil, "which", fake_which)

    assert core_install._npm_executable() == native


@pytest.mark.skipif(os.name == "nt", reason = "POSIX hint form")
def test_npm_install_hint_without_resolvable_home(monkeypatch):
    # A bare container UID has no home; the hint must still build.
    def no_home():
        raise RuntimeError("no home directory")

    monkeypatch.setattr(Path, "home", staticmethod(no_home))

    assert core_install._npm_install_hint("@openai/codex") == "npm install -g @openai/codex"


def test_install_agent_missing_npm_names_node_requirement(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: True)
    set_start_attr(monkeypatch, "_npm_executable", lambda: None)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: pytest.fail("should not run an installer without npm"),
    )

    with pytest.raises(typer.Exit):
        core_install._install_agent("codex", "npm install -g @openai/codex")

    err = capsys.readouterr().err
    assert "npm is required" in err
    assert "no native system npm was found" in err
    assert "Install Node.js with npm" in err


def test_install_agent_reports_os_error_without_traceback(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: True)
    set_start_attr(monkeypatch, "_npm_executable", lambda: "/broken/npm")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("permission denied")),
    )

    with pytest.raises(typer.Exit):
        core_install._install_agent("codex", "npm install -g @openai/codex")

    err = capsys.readouterr().err
    assert "Could not run the install command: permission denied" in err
    assert "Run it yourself, then re-run" in err


@pytest.mark.skipif(os.name == "nt", reason = "POSIX install command")
def test_install_agent_runs_npm_with_its_node_on_path(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: True)
    npm = tmp_path / "node" / "bin" / "npm"
    set_start_attr(monkeypatch, "_npm_executable", lambda: str(npm))
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/local/bin/codex")
    captured = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(subprocess, "run", run)
    hint = core_install._npm_install_hint("@openai/codex")

    assert core_install._install_agent("codex", hint) == "/usr/local/bin/codex"
    assert captured["command"] == [
        str(npm),
        "install",
        "-g",
        "--prefix",
        str(tmp_path / ".local"),
        "@openai/codex",
    ]
    assert captured["env"]["PATH"].split(os.pathsep)[0] == str(npm.parent)


def test_install_agent_warns_remote_installer_is_unverified_third_party(monkeypatch, capsys):
    # Before the confirm, a remote installer must name the URL it fetches so the
    # user consents to a specific source rather than blindly accepting.
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: False)  # decline: nothing runs
    hint = "& ([scriptblock]::Create((irm https://hermes-agent.nousresearch.com/install.ps1))) -SkipSetup"
    assert core_install._install_agent("hermes", hint) is None
    err = capsys.readouterr().err
    assert "Security warning" in err
    assert "unverified third-party script" in err
    assert "https://hermes-agent.nousresearch.com/install.ps1" in err
    assert "agent-switch does not pin or verify the downloaded content" in err
    assert "Continue only if you trust this source" in err


def test_install_agent_reports_immutable_remote_installer_pin(monkeypatch, capsys):
    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: False)
    commit = "f1af945f6c576eccb126fa955edc9be258b33020"
    hint = (
        f"curl -fsSL https://raw.githubusercontent.com/example/agent/{commit}/scripts/install.sh"
        f" | bash -s -- --commit {commit}"
    )
    assert core_install._install_agent("agent", hint) is None
    err = capsys.readouterr().err
    assert commit in err
    assert "immutable upstream commit" in err
    assert "does not independently verify or sandbox it" in err
    assert "does not pin or verify" not in err


def test_install_agent_warns_for_package_installer(monkeypatch, capsys):
    # An npm-style installer has no URL to fetch, but still runs with the user's
    # privileges, so the warning names the command instead.
    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: False)
    assert core_install._install_agent("codex", "npm install -g @openai/codex") is None
    err = capsys.readouterr().err
    assert "npm install -g @openai/codex" in err
    assert "with your privileges" in err


def test_augment_path_adds_existing_local_bin(monkeypatch, tmp_path):
    # Claude's installer drops its binary in ~/.local/bin but only *suggests* adding it to
    # PATH, so agent-switch appends it in-process to resolve the freshly installed agent.
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents = True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))  # skip the npm candidate
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))
    core_install._augment_path_with_install_dirs()
    entries = os.environ["PATH"].split(os.pathsep)
    assert str(local_bin) in entries
    # Appended (lowest precedence), so it never shadows an existing PATH entry.
    assert entries[-1] == str(local_bin)


def test_augment_path_skips_missing_and_duplicate_dirs(monkeypatch, tmp_path):
    # A non-existent ~/.local/bin is not added; an already-present one is not duplicated.
    monkeypatch.setattr(Path, "home", lambda: tmp_path)  # no .local/bin created yet
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))  # skip the npm candidate
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))
    core_install._augment_path_with_install_dirs()
    assert os.environ["PATH"] == str(tmp_path / "existing")

    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents = True)
    monkeypatch.setenv("PATH", os.pathsep.join([str(tmp_path / "existing"), str(local_bin)]))
    core_install._augment_path_with_install_dirs()
    assert os.environ["PATH"].split(os.pathsep).count(str(local_bin)) == 1


def test_augment_path_adds_npm_global_bin_on_windows(monkeypatch, tmp_path):
    # npm -g shims (codex/opencode/pi) land in %APPDATA%\npm on Windows; add it so a freshly
    # installed npm agent resolves even when that dir isn't on PATH yet.
    _simulate_windows(monkeypatch)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)  # no ~/.local/bin created
    npm_dir = tmp_path / "Roaming" / "npm"
    npm_dir.mkdir(parents = True)
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))
    core_install._augment_path_with_install_dirs()
    assert str(npm_dir) in os.environ["PATH"].split(os.pathsep)


def test_which_with_install_dirs_finds_agent_and_restores_path(monkeypatch, tmp_path):
    # The probe helper resolves against the augmented PATH but must NOT persist it: only
    # _launch() should mutate PATH for the child process. Here `claude` is only in ~/.local/bin.
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents = True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))  # skip the npm candidate
    original = str(tmp_path / "existing")
    monkeypatch.setenv("PATH", original)  # local_bin NOT on PATH yet
    monkeypatch.setattr(shutil, "which", _path_aware_which({"claude": local_bin}))
    assert core_install._which_with_install_dirs("claude") == str(local_bin / "claude")
    assert os.environ["PATH"] == original  # restored, no global pollution


def test_augment_path_preserves_defpath_when_path_unset(monkeypatch, tmp_path):
    # PATH unset: shutil.which() and exec*p* fall back to os.defpath (e.g. /bin:/usr/bin), so the
    # augmentation must keep those default dirs instead of collapsing to just the install dir
    # (which would hide a system-installed agent and strip the launched child's normal PATH).
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents = True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))
    monkeypatch.delenv("PATH", raising = False)
    core_install._augment_path_with_install_dirs()
    entries = os.environ["PATH"].split(os.pathsep)
    for default_dir in os.defpath.split(os.pathsep):
        if default_dir:
            assert default_dir in entries
    assert str(local_bin) in entries


def test_which_with_install_dirs_keeps_defpath_when_path_unset(monkeypatch, tmp_path):
    # With PATH unset, a system agent on os.defpath (e.g. /usr/bin) must still resolve; the
    # install-dir augmentation must not drop the default search path. PATH is restored to unset.
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents = True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))
    monkeypatch.delenv("PATH", raising = False)
    sysdir = next(part for part in reversed(os.defpath.split(os.pathsep)) if part)
    monkeypatch.setattr(shutil, "which", _path_aware_which({"claude": Path(sysdir)}))
    assert core_install._which_with_install_dirs("claude") == os.path.join(sysdir, "claude")
    assert "PATH" not in os.environ


def test_install_agent_declined_returns_none(monkeypatch):
    # TTY + no: never runs anything; caller falls back to the print-hint failure.
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: False)
    monkeypatch.setattr(shutil, "which", lambda _: None)
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("should not install when declined")
    )
    assert core_install._install_agent("codex", "npm install -g @openai/codex") is None


def test_install_agent_non_interactive_returns_none(monkeypatch):
    # No TTY (piped stdin): cannot prompt, so don't install; return None silently.
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty = lambda: False))
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("should not install without a TTY")
    )
    assert core_install._install_agent("codex", "npm install -g @openai/codex") is None


def test_prefer_cmd_sibling_is_none_safe_and_posix_noop(monkeypatch, tmp_path):
    assert core_install._prefer_windows_cmd_sibling(None) is None
    # Pin os.name instead of relying on the host: on a Windows runner the rescue
    # would fire and this would assert the opposite of what it means to check.
    monkeypatch.setattr(os, "name", "posix")
    shim = tmp_path / "fake-agent"
    shim.write_text("#!/bin/sh\n", encoding = "utf-8")
    (tmp_path / "fake-agent.cmd").write_text("@ECHO off\n", encoding = "utf-8")
    assert core_install._prefer_windows_cmd_sibling(str(shim)) == str(shim)


def test_which_with_install_dirs_applies_the_cmd_sibling_preference(monkeypatch, tmp_path):
    # The version probes spawn this result directly, bypassing
    # _resolved_launch_command, so the rescue must happen here too.
    _simulate_windows(monkeypatch)
    posix_shim = tmp_path / "fake-agent"
    posix_shim.write_text("#!/bin/sh\n", encoding = "utf-8")
    cmd = tmp_path / "fake-agent.cmd"
    cmd.write_text("@ECHO off\n", encoding = "utf-8")
    set_start_attr(monkeypatch, "_get_augmented_path", lambda: None)
    monkeypatch.setattr(shutil, "which", lambda name, path = None: str(posix_shim))

    assert core_install._which_with_install_dirs("fake-agent") == str(cmd)


def test_augment_path_leaves_path_alone_when_nothing_to_add(monkeypatch):
    monkeypatch.setattr(
        Path,
        "home",
        staticmethod(lambda: (_ for _ in ()).throw(RuntimeError("no home directory"))),
    )
    monkeypatch.setenv("PATH", "/usr/bin")

    core_install._augment_path_with_install_dirs()

    assert os.environ["PATH"] == "/usr/bin"


def test_probe_env_carries_install_dirs_and_restores_path(monkeypatch, tmp_path):
    # A Node-backed shim whose node sits in an install dir needs that dir on PATH when it runs.
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents = True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    before = os.environ.get("PATH")

    env = core_install._probe_env(OPENCODE_CONFIG = "/tmp/cfg.json")

    assert str(local_bin) in env["PATH"]
    assert env["OPENCODE_CONFIG"] == "/tmp/cfg.json"
    assert os.environ.get("PATH") == before
