# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for `agent-switch <agent>` — config merging and launch env, no network."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import pytest
from typer.testing import CliRunner

import agent_switch.providers.utils as provider_utils
import agent_switch.start as start

BASE = "http://127.0.0.1:8888"
MODEL = {"id": "org/gemma-4-26B-A4B-it-GGUF", "context_length": 131072}
KEY = "sk-test-feedfacefeedface"


# --no-launch prints shell setup as POSIX (export/unset) on Unix/WSL and
# PowerShell ($env:/Remove-Item) on native Windows; assert the host's form.
def _assert_env_set(output: str, name: str, value: str) -> None:
    needle = f'$env:{name} = "{value}"' if os.name == "nt" else f"export {name}={value}"
    assert needle in output, f"{needle!r} not found in:\n{output}"


def _assert_env_unset(output: str, name: str) -> None:
    needle = f"Remove-Item Env:{name}" if os.name == "nt" else f"unset {name}"
    assert needle in output, f"{needle!r} not found in:\n{output}"


def _assert_env_kept(output: str, name: str) -> None:
    needle = f"Remove-Item Env:{name}" if os.name == "nt" else f"unset {name}"
    assert needle not in output, f"{needle!r} unexpectedly found in:\n{output}"


def _assert_env_cwd(output: str, name: str) -> None:
    needle = f"$env:{name} = (Get-Location).Path" if os.name == "nt" else f'export {name}="$PWD"'
    assert needle in output, f"{needle!r} not found in:\n{output}"


def _launch_command(output: str) -> list:
    # The --no-launch recipe ends with a self-contained one-liner: inline NAME=value
    # assignments, then the command. Return just the command argv.
    last = [ln for ln in output.splitlines() if ln.strip()][-1]
    parts = shlex.split(last)
    for i, part in enumerate(parts):
        name = part.partition("=")[0]
        if "=" not in part or not name.replace("_", "").isalnum():
            return parts[i:]
    return []


def _fake_claude(monkeypatch, version_output: str) -> None:
    monkeypatch.setattr(start, "_which_with_install_dirs", lambda _: "/usr/local/bin/claude")
    monkeypatch.setattr(start, "_probe_env", lambda **_: {})
    monkeypatch.setattr(
        start.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout = version_output),
    )


def _path_aware_which(binaries: dict):
    # A shutil.which fake that resolves a name only when its directory is on PATH at call time.
    # Lets a test prove a version probe augments PATH before resolving: an agent present only in
    # an install dir (~/.local/bin, %APPDATA%\npm) must still be found and version-checked.
    def _which(name):
        directory = binaries.get(name)
        if directory is None:
            return None
        entries = os.environ.get("PATH", "").split(os.pathsep)
        # os.path.join (not Path()) so this works when a test has flipped os.name to "nt": under
        # a simulated os.name, pathlib would build the non-native flavour and raise.
        return os.path.join(str(directory), name) if str(directory) in entries else None

    return _which


def _simulate_windows(monkeypatch) -> None:
    # Exercise the `os.name == "nt"` branch on any host. Flipping os.name alone makes pathlib
    # pick the non-native flavour (WindowsPath on POSIX, PosixPath on Windows) when a Path is
    # constructed, which raises; pin Path to the host-native class (captured before the flip)
    # so the branch logic runs without that crash. Keeps these tests green on Linux/Mac/WSL too.
    monkeypatch.setattr(start, "Path", type(Path()))
    monkeypatch.setattr(start.os, "name", "nt")


def test_claude_flags_passed_to_supported_claude(monkeypatch):
    _fake_claude(monkeypatch, "2.1.98 (Claude Code)\n")
    assert start._claude_flags(MODEL["id"]) == [
        "--exclude-dynamic-system-prompt-sections",
        "--settings",
        start._claude_settings_overlay(MODEL["id"]),
    ]


def test_claude_dynamic_sections_skipped_on_old_claude(monkeypatch):
    _fake_claude(monkeypatch, "2.0.14 (Claude Code)\n")
    assert start._claude_flags(MODEL["id"]) == [
        "--settings",
        start._claude_settings_overlay(MODEL["id"]),
    ]


def test_claude_settings_retained_on_unparseable_version(monkeypatch):
    _fake_claude(monkeypatch, "weird build string\n")
    assert start._claude_flags(MODEL["id"]) == [
        "--settings",
        start._claude_settings_overlay(MODEL["id"]),
    ]


def test_claude_flags_detected_when_version_not_first_token(monkeypatch):
    # The X.Y.Z is pulled from anywhere in the output, so a format change (version not
    # the first token) doesn't silently drop the optimization flags.
    _fake_claude(monkeypatch, "claude version 2.1.98\n")
    assert start._claude_flags(MODEL["id"]) == [
        "--exclude-dynamic-system-prompt-sections",
        "--settings",
        start._claude_settings_overlay(MODEL["id"]),
    ]


def test_claude_settings_overlay_pins_served_model():
    # The session overlay must pin availableModels to the served model: a user's allowlist
    # in ~/.claude/settings.json otherwise rejects the local --model ("restricted by your
    # organization's settings"), and no env var can bypass it. The override must be a
    # NON-EMPTY array to take effect (an empty [] is ignored and the user's list still
    # applies), so it lists exactly this model, for this session only.
    overlay = json.loads(start._claude_settings_overlay(MODEL["id"]))
    assert overlay["availableModels"] == [MODEL["id"]]


def test_claude_settings_overlay_pins_local_routing_and_auth():
    local_env = start._claude_local_env(BASE, "sk-test", MODEL)
    overlay = json.loads(start._claude_settings_overlay(MODEL["id"], local_env))
    for name, value in local_env.items():
        assert overlay["env"][name] == value
    assert overlay["env"]["ANTHROPIC_BASE_URL"] == BASE
    assert overlay["env"]["ANTHROPIC_AUTH_TOKEN"] == "sk-test"
    for name in start._CLAUDE_ENV_UNSET:
        assert overlay["env"][name] == ""
    # The attribution-header suppression is preserved alongside it.
    assert overlay["env"]["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0"
    assert overlay["env"]["CLAUDE_CODE_TOTAL_TOKENS_REMINDER"] == "off"
    # Subagents fall through to the served model instead of a user's opus/sonnet pin.
    assert overlay["env"]["CLAUDE_CODE_SUBAGENT_MODEL"] == "inherit"


def test_claude_settings_files_preserve_concurrent_sessions(tmp_path):
    first_env = start._claude_local_env("http://127.0.0.1:8001", "first-key", MODEL)
    second_env = start._claude_local_env("http://127.0.0.1:8002", "second-key", MODEL)
    first = start._write_claude_settings(tmp_path, MODEL["id"], first_env)
    second = start._write_claude_settings(tmp_path, MODEL["id"], second_env)
    assert first != second
    assert json.loads(first.read_text())["env"]["ANTHROPIC_AUTH_TOKEN"] == "first-key"
    assert json.loads(second.read_text())["env"]["ANTHROPIC_AUTH_TOKEN"] == "second-key"


def test_install_agent_prompts_then_installs(monkeypatch):
    # TTY + yes: run the documented install command, then re-resolve the now-present binary.
    monkeypatch.setattr(start.os, "name", "posix")
    monkeypatch.setattr(start.sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(start.typer, "confirm", lambda *a, **k: True)
    monkeypatch.setattr(start, "_npm_executable", lambda: "/usr/local/bin/npm")
    ran = []
    monkeypatch.setattr(
        start.subprocess,
        "run",
        lambda command, *a, **k: ran.append(command) or SimpleNamespace(returncode = 0),
    )
    # _install_agent only re-resolves after installing (the pre-install check is the
    # caller's job), so `which` reports the now-present binary.
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/codex")
    executable = start._install_agent("codex", "npm install -g @openai/codex")
    assert executable == "/usr/local/bin/codex"
    assert ran == [["/usr/local/bin/npm", "install", "-g", "@openai/codex"]]


def test_install_agent_uses_powershell_on_windows(monkeypatch):
    monkeypatch.setattr(start.os, "name", "nt")
    monkeypatch.setattr(start.sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(start.typer, "confirm", lambda *a, **k: True)
    ran = []
    monkeypatch.setattr(
        start.subprocess,
        "run",
        lambda command, *a, **k: ran.append(command) or SimpleNamespace(returncode = 0),
    )
    monkeypatch.setattr(start.shutil, "which", lambda _: r"C:\Users\samle\bin\hermes.exe")

    install_hint = "& ([scriptblock]::Create((irm https://x/install.ps1))) -SkipSetup"
    executable = start._install_agent("hermes", install_hint)

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
    monkeypatch.setattr(start.sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(start.typer, "confirm", lambda *a, **k: True)
    monkeypatch.setattr(
        start, "_npm_executable", lambda: r"C:\Users\me\AppData\Roaming\npm\npm.cmd"
    )
    monkeypatch.setattr(
        start.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode = 1),
    )
    monkeypatch.setattr(start.shutil, "which", lambda _: None)

    with pytest.raises(start.typer.Exit):
        start._install_agent("codex", "npm install -g @openai/codex")

    err = capsys.readouterr().err
    assert "Install command failed" in err
    assert "Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned" in err


def test_install_command_uses_resolved_npm_cmd_on_windows(monkeypatch):
    _simulate_windows(monkeypatch)
    monkeypatch.setattr(start, "_npm_executable", lambda: r"C:\Program Files\nodejs\npm.cmd")

    command, env = start._install_command(start._npm_install_hint("@openai/codex"))

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
    monkeypatch.setattr(start.os, "name", "posix")
    monkeypatch.setattr(start.sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(start.typer, "confirm", lambda *a, **k: True)
    monkeypatch.setattr(
        start.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode = 1),
    )

    with pytest.raises(start.typer.Exit):
        start._install_agent("codex", "npm install -g @openai/codex")

    err = capsys.readouterr().err
    assert "Install command failed" in err
    assert "Set-ExecutionPolicy" not in err


def test_npm_install_hint_uses_user_prefix_on_posix(monkeypatch, tmp_path):
    monkeypatch.setattr(start.os, "name", "posix")
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)

    hint = start._npm_install_hint("@openai/codex")

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

    monkeypatch.setattr(start.shutil, "which", fake_which)

    assert start._npm_executable() == native


@pytest.mark.skipif(os.name == "nt", reason = "POSIX hint form")
def test_npm_install_hint_without_resolvable_home(monkeypatch):
    # A bare container UID has no home; the hint must still build.
    def no_home():
        raise RuntimeError("no home directory")

    monkeypatch.setattr(start.Path, "home", staticmethod(no_home))

    assert start._npm_install_hint("@openai/codex") == "npm install -g @openai/codex"


def test_install_agent_missing_npm_names_node_requirement(monkeypatch, capsys):
    monkeypatch.setattr(start.sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(start.typer, "confirm", lambda *a, **k: True)
    monkeypatch.setattr(start, "_npm_executable", lambda: None)
    monkeypatch.setattr(
        start.subprocess,
        "run",
        lambda *a, **k: pytest.fail("should not run an installer without npm"),
    )

    with pytest.raises(start.typer.Exit):
        start._install_agent("codex", "npm install -g @openai/codex")

    err = capsys.readouterr().err
    assert "npm is required" in err
    assert "no native system npm was found" in err
    assert "Install Node.js with npm" in err


def test_install_agent_reports_os_error_without_traceback(monkeypatch, capsys):
    monkeypatch.setattr(start.sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(start.typer, "confirm", lambda *a, **k: True)
    monkeypatch.setattr(start, "_npm_executable", lambda: "/broken/npm")
    monkeypatch.setattr(
        start.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("permission denied")),
    )

    with pytest.raises(start.typer.Exit):
        start._install_agent("codex", "npm install -g @openai/codex")

    err = capsys.readouterr().err
    assert "Could not run the install command: permission denied" in err
    assert "Run it yourself, then re-run" in err


@pytest.mark.skipif(os.name == "nt", reason = "POSIX install command")
def test_install_agent_runs_npm_with_its_node_on_path(monkeypatch, tmp_path):
    monkeypatch.setattr(start.os, "name", "posix")
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(start.sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(start.typer, "confirm", lambda *a, **k: True)
    npm = tmp_path / "node" / "bin" / "npm"
    monkeypatch.setattr(start, "_npm_executable", lambda: str(npm))
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/codex")
    captured = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(start.subprocess, "run", run)
    hint = start._npm_install_hint("@openai/codex")

    assert start._install_agent("codex", hint) == "/usr/local/bin/codex"
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
    monkeypatch.setattr(start.os, "name", "nt")
    monkeypatch.setattr(start.sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(start.typer, "confirm", lambda *a, **k: False)  # decline: nothing runs
    hint = "& ([scriptblock]::Create((irm https://hermes-agent.nousresearch.com/install.ps1))) -SkipSetup"
    assert start._install_agent("hermes", hint) is None
    err = capsys.readouterr().err
    assert "Security warning" in err
    assert "unverified third-party script" in err
    assert "https://hermes-agent.nousresearch.com/install.ps1" in err
    assert "agent-switch does not pin or verify the downloaded content" in err
    assert "Continue only if you trust this source" in err


def test_install_agent_reports_immutable_remote_installer_pin(monkeypatch, capsys):
    monkeypatch.setattr(start.os, "name", "posix")
    monkeypatch.setattr(start.sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(start.typer, "confirm", lambda *a, **k: False)
    commit = "f1af945f6c576eccb126fa955edc9be258b33020"
    hint = (
        f"curl -fsSL https://raw.githubusercontent.com/example/agent/{commit}/scripts/install.sh"
        f" | bash -s -- --commit {commit}"
    )
    assert start._install_agent("agent", hint) is None
    err = capsys.readouterr().err
    assert commit in err
    assert "immutable upstream commit" in err
    assert "does not independently verify or sandbox it" in err
    assert "does not pin or verify" not in err


def test_install_agent_warns_for_package_installer(monkeypatch, capsys):
    # An npm-style installer has no URL to fetch, but still runs with the user's
    # privileges, so the warning names the command instead.
    monkeypatch.setattr(start.os, "name", "posix")
    monkeypatch.setattr(start.sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(start.typer, "confirm", lambda *a, **k: False)
    assert start._install_agent("codex", "npm install -g @openai/codex") is None
    err = capsys.readouterr().err
    assert "npm install -g @openai/codex" in err
    assert "with your privileges" in err


def test_refresh_windows_path_noop_off_windows(monkeypatch):
    monkeypatch.setattr(start.os, "name", "posix")
    before = os.environ.get("PATH", "")
    monkeypatch.setenv("PATH", before)
    start._refresh_windows_path()
    assert os.environ.get("PATH", "") == before


def test_refresh_windows_path_merges_registry_hives(monkeypatch):
    # Fake Windows registry PATH values written after this process started.
    hkcu, hklm = object(), object()
    reg = {
        (hkcu, "Environment"): r"C:\existing;C:\Users\me\hermes\bin",
        (
            hklm,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        ): r"C:\Windows\System32",
    }

    class _Key:
        def __init__(self, value):
            self._value = value

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def open_key(root, sub):
        if (root, sub) in reg:
            return _Key(reg[(root, sub)])
        raise OSError("missing hive")

    fake_winreg = SimpleNamespace(
        HKEY_CURRENT_USER = hkcu,
        HKEY_LOCAL_MACHINE = hklm,
        OpenKey = open_key,
        QueryValueEx = lambda key, name: (key._value, 1),
    )
    monkeypatch.setattr(start.os, "name", "nt")
    monkeypatch.setattr(start.os, "pathsep", ";")
    monkeypatch.setitem(sys.modules, "winreg", fake_winreg)
    monkeypatch.setenv("PATH", r"C:\custom;C:\existing")

    start._refresh_windows_path()

    assert os.environ["PATH"].split(";") == [
        r"C:\custom",
        r"C:\existing",
        r"C:\Users\me\hermes\bin",
        r"C:\Windows\System32",
    ]


def test_augment_path_adds_existing_local_bin(monkeypatch, tmp_path):
    # Claude's installer drops its binary in ~/.local/bin but only *suggests* adding it to
    # PATH, so agent-switch appends it in-process to resolve the freshly installed agent.
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents = True)
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))  # skip the npm candidate
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))
    start._augment_path_with_install_dirs()
    entries = os.environ["PATH"].split(os.pathsep)
    assert str(local_bin) in entries
    # Appended (lowest precedence), so it never shadows an existing PATH entry.
    assert entries[-1] == str(local_bin)


def test_augment_path_skips_missing_and_duplicate_dirs(monkeypatch, tmp_path):
    # A non-existent ~/.local/bin is not added; an already-present one is not duplicated.
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)  # no .local/bin created yet
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))  # skip the npm candidate
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))
    start._augment_path_with_install_dirs()
    assert os.environ["PATH"] == str(tmp_path / "existing")

    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents = True)
    monkeypatch.setenv("PATH", os.pathsep.join([str(tmp_path / "existing"), str(local_bin)]))
    start._augment_path_with_install_dirs()
    assert os.environ["PATH"].split(os.pathsep).count(str(local_bin)) == 1


def test_augment_path_adds_npm_global_bin_on_windows(monkeypatch, tmp_path):
    # npm -g shims (codex/opencode/pi) land in %APPDATA%\npm on Windows; add it so a freshly
    # installed npm agent resolves even when that dir isn't on PATH yet.
    _simulate_windows(monkeypatch)
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)  # no ~/.local/bin created
    npm_dir = tmp_path / "Roaming" / "npm"
    npm_dir.mkdir(parents = True)
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))
    start._augment_path_with_install_dirs()
    assert str(npm_dir) in os.environ["PATH"].split(os.pathsep)


def test_which_with_install_dirs_finds_agent_and_restores_path(monkeypatch, tmp_path):
    # The probe helper resolves against the augmented PATH but must NOT persist it: only
    # _launch() should mutate PATH for the child process. Here `claude` is only in ~/.local/bin.
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents = True)
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))  # skip the npm candidate
    original = str(tmp_path / "existing")
    monkeypatch.setenv("PATH", original)  # local_bin NOT on PATH yet
    monkeypatch.setattr(start.shutil, "which", _path_aware_which({"claude": local_bin}))
    assert start._which_with_install_dirs("claude") == str(local_bin / "claude")
    assert os.environ["PATH"] == original  # restored, no global pollution


def test_claude_flags_probes_old_agent_only_in_install_dir(monkeypatch, tmp_path):
    # Regression: the version probe must augment PATH before resolving, so an OLD claude present
    # only in ~/.local/bin (not yet on PATH) is detected as old and the unsupported flags are
    # dropped -- the same binary _launch() will run. Before the fix the probe saw no binary,
    # assumed a current build, and emitted flags the old claude rejects.
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents = True)
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))
    monkeypatch.setattr(start.shutil, "which", _path_aware_which({"claude": local_bin}))
    monkeypatch.setattr(
        start.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout = "2.0.14 (Claude Code)\n")
    )
    assert start._claude_flags(MODEL["id"]) == [
        "--settings",
        start._claude_settings_overlay(MODEL["id"]),
    ]


def test_claude_flags_detects_supported_agent_only_in_install_dir(monkeypatch, tmp_path):
    # The counterpart: a SUPPORTED claude present only in ~/.local/bin is now resolved and gets
    # the flags, instead of being missed and (coincidentally) also assumed current.
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents = True)
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))
    monkeypatch.setattr(start.shutil, "which", _path_aware_which({"claude": local_bin}))
    monkeypatch.setattr(
        start.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout = "2.1.98 (Claude Code)\n")
    )
    assert start._claude_flags(MODEL["id"]) == [
        "--exclude-dynamic-system-prompt-sections",
        "--settings",
        start._claude_settings_overlay(MODEL["id"]),
    ]


def test_claude_flags_probes_npm_install_dir_on_windows(monkeypatch, tmp_path):
    # npm -g shims land in %APPDATA%\npm on Windows; an old claude there (not on PATH) must still
    # be version-checked so the unsupported flags are dropped.
    _simulate_windows(monkeypatch)
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)  # no ~/.local/bin created
    npm_dir = tmp_path / "Roaming" / "npm"
    npm_dir.mkdir(parents = True)
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))
    monkeypatch.setattr(start.shutil, "which", _path_aware_which({"claude": npm_dir}))
    monkeypatch.setattr(
        start.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout = "2.0.14 (Claude Code)\n")
    )
    assert start._claude_flags(MODEL["id"]) == [
        "--settings",
        start._claude_settings_overlay(MODEL["id"]),
    ]


def test_codex_catalog_probes_old_codex_only_in_install_dir(monkeypatch, tmp_path):
    # Same ordering fix for codex: an old codex present only in an install dir is detected so the
    # model-catalog config is omitted (the old binary can't consume it).
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents = True)
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))
    monkeypatch.setattr(start.shutil, "which", _path_aware_which({"codex": local_bin}))
    monkeypatch.setattr(start.subprocess, "check_output", lambda *a, **k: "codex-cli 0.109.0")
    assert start._codex_supports_model_catalog() is False


def test_opencode_native_auto_probes_old_opencode_only_in_install_dir(monkeypatch, tmp_path):
    # Same ordering fix for opencode: an old opencode present only in an install dir is detected
    # so native --auto is not assumed (the old binary rejects it).
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents = True)
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))
    monkeypatch.setattr(start.shutil, "which", _path_aware_which({"opencode": local_bin}))
    monkeypatch.setattr(start.subprocess, "check_output", lambda *a, **k: "1.17.11")
    assert start._opencode_supports_native_auto() is False


def test_opencode_command_prefers_installed_v2(monkeypatch):
    monkeypatch.setattr(
        start,
        "_which_with_install_dirs",
        lambda name: "/usr/local/bin/opencode2" if name == "opencode2" else None,
    )
    assert start._opencode_command() == ("/usr/local/bin/opencode2", True)


def test_opencode_command_falls_back_to_v1(monkeypatch):
    monkeypatch.setattr(start, "_which_with_install_dirs", lambda _: None)
    assert start._opencode_command() == ("opencode", False)


def test_opencode_command_finds_official_v2_install_dir(monkeypatch, tmp_path):
    install_dir = tmp_path / ".opencode" / "bin"
    install_dir.mkdir(parents = True)
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))
    monkeypatch.setattr(start.shutil, "which", _path_aware_which({"opencode2": install_dir}))

    assert start._opencode_command() == (str(install_dir / "opencode2"), True)


def test_augment_path_preserves_defpath_when_path_unset(monkeypatch, tmp_path):
    # PATH unset: shutil.which() and exec*p* fall back to os.defpath (e.g. /bin:/usr/bin), so the
    # augmentation must keep those default dirs instead of collapsing to just the install dir
    # (which would hide a system-installed agent and strip the launched child's normal PATH).
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents = True)
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))
    monkeypatch.delenv("PATH", raising = False)
    start._augment_path_with_install_dirs()
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
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))
    monkeypatch.delenv("PATH", raising = False)
    sysdir = next(part for part in reversed(os.defpath.split(os.pathsep)) if part)
    monkeypatch.setattr(start.shutil, "which", _path_aware_which({"claude": Path(sysdir)}))
    assert start._which_with_install_dirs("claude") == os.path.join(sysdir, "claude")
    assert "PATH" not in os.environ


def test_install_agent_declined_returns_none(monkeypatch):
    # TTY + no: never runs anything; caller falls back to the print-hint failure.
    monkeypatch.setattr(start.sys, "stdin", SimpleNamespace(isatty = lambda: True))
    monkeypatch.setattr(start.typer, "confirm", lambda *a, **k: False)
    monkeypatch.setattr(start.shutil, "which", lambda _: None)
    monkeypatch.setattr(
        start.subprocess, "run", lambda *a, **k: pytest.fail("should not install when declined")
    )
    assert start._install_agent("codex", "npm install -g @openai/codex") is None


def test_install_agent_non_interactive_returns_none(monkeypatch):
    # No TTY (piped stdin): cannot prompt, so don't install; return None silently.
    monkeypatch.setattr(start.sys, "stdin", SimpleNamespace(isatty = lambda: False))
    monkeypatch.setattr(
        start.subprocess, "run", lambda *a, **k: pytest.fail("should not install without a TTY")
    )
    assert start._install_agent("codex", "npm install -g @openai/codex") is None


def _parse_toml(text: str) -> dict:
    tomllib = pytest.importorskip("tomllib")
    return tomllib.loads(text)


def test_project_declares_direct_cli_dependencies():
    project = _parse_toml((_REPO_ROOT / "pyproject.toml").read_text(encoding = "utf-8"))
    assert "click>=8.0" in project["project"]["dependencies"]


def test_agent_paths_use_agent_switch_home(monkeypatch, tmp_path):
    # Session homes and the per-server key cache both live under agent-switch's own root.
    monkeypatch.setenv("AGENT_SWITCH_HOME", str(tmp_path / "home"))

    assert start._provider_key_cache_path() == tmp_path / "home" / "api_keys.json"
    assert start._agents_config_root() == tmp_path / "home" / "agents"


def test_merge_codex_config_fresh():
    merged = start._merge_codex_config("", BASE)
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
    provider = _parse_toml(start._merge_codex_config("", BASE))["model_providers"]["agent_switch"]
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
    merged = start._merge_codex_config(existing, BASE)
    parsed = _parse_toml(merged)
    assert parsed["model"] == "gpt-5"
    assert parsed["model_providers"]["agent_switch"]["base_url"] == f"{BASE}/v1"
    assert parsed["model_providers"]["agent_switch"]["wire_api"] == "responses"
    assert "http_headers" not in parsed["model_providers"]["agent_switch"]
    assert parsed["model_providers"]["ollama"]["base_url"] == "http://localhost:11434/v1"
    assert start._merge_codex_config(merged, BASE) == merged


def test_merge_codex_config_keeps_user_oss_provider():
    merged = start._merge_codex_config('oss_provider = "ollama"\n', BASE)
    assert _parse_toml(merged)["oss_provider"] == "ollama"


def test_write_codex_config_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(start, "_codex_supports_model_catalog", lambda: True)
    monkeypatch.setattr(start, "_codex_supports_patch_line_endings", lambda: True)
    start.write_codex_config(BASE, MODEL, tmp_path)
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

    assert catalog["models"][0]["base_instructions"] == start._CODEX_FALLBACK_PROMPT.read_text(
        encoding = "utf-8"
    )
    assert '{"command"' not in catalog["models"][0]["base_instructions"]
    config = _parse_toml((tmp_path / "config.toml").read_text())
    assert config["model_providers"]["agent_switch"]["env_key"] == "AGENT_SWITCH_AUTH_TOKEN"


def test_write_codex_config_catalog_without_context_length(tmp_path, monkeypatch):
    monkeypatch.setattr(start, "_codex_supports_model_catalog", lambda: True)
    start.write_codex_config(BASE, {"id": "org/no-window"}, tmp_path)
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
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/codex")
    monkeypatch.setattr(start.subprocess, "check_output", lambda *args, **kwargs: version)
    assert start._codex_supports_model_catalog() is expected


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
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/codex")
    monkeypatch.setattr(start.subprocess, "check_output", lambda *args, **kwargs: version)
    assert start._codex_supports_patch_line_endings() is expected


def test_codex_patch_line_endings_assumes_current_when_not_installed(monkeypatch):
    monkeypatch.setattr(start, "_which_with_install_dirs", lambda _: None)
    assert start._codex_supports_patch_line_endings() is True


def test_write_codex_config_omits_catalog_for_old_codex(tmp_path, monkeypatch):
    monkeypatch.setattr(start, "_codex_supports_model_catalog", lambda: False)
    start.write_codex_config(BASE, MODEL, tmp_path)
    profile = _parse_toml((tmp_path / "agent_switch.config.toml").read_text())
    assert "model_catalog_json" not in profile
    assert not (tmp_path / "model-catalog.json").exists()


def test_write_codex_subagent_bridge_keeps_parent_credentials_out(tmp_path, monkeypatch):
    monkeypatch.setattr(start, "_codex_supports_model_catalog", lambda: True)
    local = {**MODEL, "id": MODEL["id"] + ":UD-Q4_K_XL"}
    path = start.write_codex_subagent_bridge(
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
    assert profile["model_provider"] == start._CODEX_PROFILE
    assert profile["model_context_window"] == MODEL["context_length"]
    config = _parse_toml((tmp_path / "child" / "config.toml").read_text())
    assert config["model_providers"][start._CODEX_PROFILE]["base_url"] == f"{BASE}/v1"
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

    overlay = start.write_codex_parent_overlay(tmp_path / "managed" / "parent")

    assert (overlay / "config.toml").read_text() == 'model = "cloud-model"\n'
    assert (overlay / "auth.json").read_text() == '{"auth": "cloud"}\n'
    assert (overlay / "sessions").is_dir()
    instructions = (overlay / "AGENTS.override.md").read_text()
    assert instructions.startswith("Keep my existing instructions.\n")
    assert start._CODEX_SUBAGENT_ROUTING_INSTRUCTIONS in instructions
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
    overlay = start.write_codex_parent_overlay(overlay_path)
    assert (overlay / "auth.json").read_text() == '{"auth": "old"}\n'
    assert (overlay / "old-only.toml").exists()

    monkeypatch.setenv("CODEX_HOME", str(second))
    overlay = start.write_codex_parent_overlay(overlay_path)
    assert (overlay / "auth.json").read_text() == '{"auth": "new"}\n'
    assert not (overlay / "old-only.toml").exists()


def test_write_codex_parent_overlay_does_not_use_itself_as_source(tmp_path, monkeypatch):
    source = tmp_path / "user-codex"
    source.mkdir()
    (source / "auth.json").write_text('{"auth": "cloud"}\n')
    overlay_path = tmp_path / "managed" / "parent"
    monkeypatch.setenv("CODEX_HOME", str(source))
    overlay = start.write_codex_parent_overlay(overlay_path)

    monkeypatch.setenv("CODEX_HOME", str(overlay))
    overlay = start.write_codex_parent_overlay(overlay_path)

    assert (overlay / "auth.json").read_text() == '{"auth": "cloud"}\n'
    manifest = json.loads((overlay / start._CODEX_PARENT_OVERLAY_MANIFEST).read_text())
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
    monkeypatch.setattr(start, "_create_directory_junction", lambda source, target: False)
    overlay = start.write_codex_parent_overlay(tmp_path / "managed" / "parent")
    (overlay / "history.jsonl").write_text("session state\n")
    config.write_text('model = "second"\n')

    overlay = start.write_codex_parent_overlay(overlay)

    assert (overlay / "config.toml").read_text() == 'model = "second"\n'
    assert (overlay / "sessions" / "existing.jsonl").read_text() == "existing session\n"
    assert (overlay / "history.jsonl").read_text() == "session state\n"

    config.unlink()
    overlay = start.write_codex_parent_overlay(overlay)
    assert not (overlay / "config.toml").exists()
    assert (overlay / "history.jsonl").read_text() == "session state\n"


def test_create_directory_junction_uses_windows_mklink(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(start.os, "name", "nt")

    def run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(start.subprocess, "run", run)
    source = tmp_path / "source"
    target = tmp_path / "target"

    assert start._create_directory_junction(source, target) is True
    assert captured["command"] == [
        "cmd.exe",
        "/d",
        "/c",
        "mklink",
        "/J",
        str(target),
        str(source),
    ]
    # Whole-dict equality made this a tripwire for any unrelated keyword. #10192
    # added encoding/errors so a localized Windows console cannot lose the child's
    # whole output stream, and this assertion went red on a change that had nothing
    # to do with junctions. Pin the values the call has to keep, and separately
    # refuse a shell, instead of forbidding every future keyword.
    kwargs = captured["kwargs"]
    for key, value in {
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": 30,
        "check": False,
    }.items():
        assert kwargs[key] == value, f"{key} = {kwargs.get(key)!r}"
    assert not kwargs.get("shell", False), "mklink must not go through a shell"


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
    monkeypatch.setattr(start.shutil, "which", lambda _: executable)

    def check_output(command, **kwargs):
        if command[0] == "cmd.exe":
            assert kwargs["cwd"] == str(Path(executable).parent)
            return r"C:\Users\x" + "\n"
        assert command == ["wslpath", "-u", r"C:\Users\x"]
        return str(windows_profile) + "\n"

    monkeypatch.setattr(start.subprocess, "check_output", check_output)

    overlay = start.write_codex_parent_overlay(tmp_path / "managed" / "parent")

    assert (overlay / "auth.json").read_text() == '{"auth": "windows"}\n'


def test_codex_parent_overlay_can_use_session_home(tmp_path, monkeypatch):
    source = tmp_path / "user-codex"
    source.mkdir()
    (source / "auth.json").write_text("{}\n")
    monkeypatch.setenv("CODEX_HOME", str(source))
    session_home = tmp_path / "session"

    overlay = start.write_codex_parent_overlay(session_home / "parent")

    assert overlay == session_home / "parent"
    assert start._CODEX_SUBAGENT_ROUTING_INSTRUCTIONS in (overlay / "AGENTS.md").read_text()
    assert overlay.exists()


def test_ephemeral_codex_parent_overlay_is_cleaned_with_session(tmp_path, monkeypatch):
    source = tmp_path / "user-codex"
    source.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(source))
    agents_root = tmp_path / "agents"
    monkeypatch.setattr(start, "_agents_config_root", lambda: agents_root)

    with start._session_config("codex-subagent", launch = True) as session_home:
        overlay = start.write_codex_parent_overlay(session_home / "parent")
        assert overlay.exists()
        assert session_home.exists()

    assert not overlay.exists()
    assert not session_home.exists()


@pytest.mark.skipif(os.name == "nt", reason = "WSL scenario")
def test_codex_subagent_bridge_uses_wsl_for_windows_codex(monkeypatch, tmp_path):
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr(
        start.shutil,
        "which",
        lambda _: "/mnt/c/Users/x/AppData/Roaming/npm/codex.exe",
    )
    flags = start._codex_subagent_flags(tmp_path / "subagent.json")
    prefix = f"mcp_servers.{start._CODEX_SUBAGENT_MCP_SERVER}="
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
    assert f"from {start._CODEX_SUBAGENT_MCP_MODULE} import main" in server["args"][5]
    assert server["required"] is True
    assert server["enabled_tools"] == [start._CODEX_SUBAGENT_MCP_TOOL]
    assert server["default_tools_approval_mode"] == "approve"
    assert not any(value.startswith("developer_instructions=") for value in flags)


@pytest.mark.skipif(os.name == "nt", reason = "WSL scenario")
def test_agent_config_path_translates_for_windows_agent(monkeypatch, tmp_path):
    windows_path = r"\\wsl.localhost\Ubuntu\tmp\agent-switch.toml"
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr(
        start.shutil,
        "which",
        lambda _: "/mnt/c/Users/x/AppData/Roaming/npm/codex",
    )
    monkeypatch.setattr(start.subprocess, "check_output", lambda *args, **kwargs: windows_path)

    assert start._agent_config_path(tmp_path / "agent-switch.toml", ["codex"]) == windows_path


@pytest.mark.parametrize("agent", ["dsh"])
@pytest.mark.parametrize("flag", ["--as-subagent", "--as-subagent=true", "--as-subagent=false"])
def test_unsupported_agents_reject_as_subagent(agent, flag):
    result = CliRunner().invoke(start.start_app, [agent, flag])
    assert result.exit_code == 1
    assert f"--as-subagent is not supported for {agent}." in result.output


@pytest.fixture()
def one_local_server(monkeypatch):
    """The no-url scan finds one server; enough for tests that stop before connecting."""
    monkeypatch.setattr(start.providers, "scan_local_servers", lambda: [start.Target("vllm", BASE)])


@pytest.mark.usefixtures("one_local_server")
@pytest.mark.parametrize(
    "agent", ["claude", "codex", "opencode", "pi", "dsh"]
)
def test_launch_preflights_agent_before_connect(agent, monkeypatch):
    events = []
    if agent == "opencode":
        monkeypatch.setattr(start, "_opencode_command", lambda *_: ("opencode", False))

    def require(name, hint, launch):
        assert name == agent
        assert hint
        assert launch is True
        events.append("agent")

    def connect(*args, **kwargs):
        events.append("connect")
        raise RuntimeError("stop after ordering check")

    monkeypatch.setattr(start, "_require_agent_for_launch", require)
    monkeypatch.setattr(start, "_connect", connect)

    result = CliRunner().invoke(start.start_app, [agent])

    assert result.exit_code == 1
    assert events == ["agent", "connect"]


@pytest.mark.usefixtures("one_local_server")
def test_dsh_rejects_an_unrelated_executable_before_connect(monkeypatch):
    monkeypatch.setattr(start, "_which_with_install_dirs", lambda _: "/usr/bin/dsh")
    monkeypatch.setattr(start, "is_deepseek_harness_executable", lambda _: False)
    monkeypatch.setattr(start, "_install_agent", lambda *_: None)
    monkeypatch.setattr(
        start,
        "_connect",
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
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path / "missing-home")
    monkeypatch.setattr(
        start,
        "is_deepseek_harness_executable",
        lambda executable: Path(executable).parent == harness_dir,
    )
    monkeypatch.setattr(
        start,
        "_install_agent",
        lambda *_: pytest.fail("an existing later Harness must be used without reinstalling"),
    )

    resolved = start._resolve_or_install_agent(
        "dsh",
        "npm install -g @deepseek-ai/dsh",
        start._which_with_install_dirs,
    )

    assert Path(resolved).parent == harness_dir


@pytest.mark.usefixtures("one_local_server")
def test_declined_opencode_subagent_install_stops_before_connect(monkeypatch):
    installs = []
    monkeypatch.setattr(start, "_which_with_install_dirs", lambda _: None)
    monkeypatch.setattr(
        start,
        "_install_agent",
        lambda name, hint: installs.append((name, hint)),
    )
    monkeypatch.setattr(
        start,
        "_connect",
        lambda *a, **k: pytest.fail("declined install must stop before model connection"),
    )

    result = CliRunner().invoke(start.start_app, ["opencode", "--as-subagent"])

    assert result.exit_code == 1
    assert len(installs) == 1
    assert installs[0][0] == "opencode"


@pytest.mark.usefixtures("one_local_server")
@pytest.mark.parametrize(
    "agent", ["claude", "codex", "opencode", "pi"]
)
def test_noninteractive_missing_agent_stops_before_connect(agent, monkeypatch):
    monkeypatch.setattr(start, "_which_with_install_dirs", lambda _: None)
    monkeypatch.setattr(
        start.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("non-interactive launch must not install"),
    )
    monkeypatch.setattr(
        start,
        "_connect",
        lambda *args, **kwargs: pytest.fail("missing agent must stop before connection"),
    )

    result = CliRunner().invoke(start.start_app, [agent])

    assert result.exit_code == 1
    assert f"`{agent}` not found on PATH" in result.output


@pytest.mark.usefixtures("one_local_server")
@pytest.mark.parametrize("agent", ["claude", "codex", "pi", "dsh"])
def test_no_launch_skips_agent_resolution(agent, monkeypatch):
    monkeypatch.setattr(
        start,
        "_which_with_install_dirs",
        lambda _: pytest.fail("--no-launch must not resolve an agent"),
    )
    monkeypatch.setattr(
        start,
        "_install_agent",
        lambda *args: pytest.fail("--no-launch must not install an agent"),
    )

    def stop_at_connect(*args, **kwargs):
        raise RuntimeError

    monkeypatch.setattr(start, "_connect", stop_at_connect)

    result = CliRunner().invoke(start.start_app, [agent, "--no-launch"])

    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)


@pytest.mark.usefixtures("one_local_server")
def test_opencode_no_launch_resolves_generation_without_installing(monkeypatch):
    resolved = []
    monkeypatch.setattr(
        start,
        "_which_with_install_dirs",
        lambda name: resolved.append(name)
        or ("/home/me/.opencode/bin/opencode2" if name == "opencode2" else None),
    )
    monkeypatch.setattr(
        start,
        "_install_agent",
        lambda *args: pytest.fail("--no-launch must not install an agent"),
    )
    monkeypatch.setattr(
        start, "_connect", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError)
    )

    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])

    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)
    assert resolved == ["opencode2"]


@pytest.mark.usefixtures("one_local_server")
def test_missing_pi_subagent_extension_fails_before_install_or_connect(monkeypatch, tmp_path):
    monkeypatch.setattr(start, "_PI_SUBAGENT_EXTENSION", tmp_path / "missing.ts")
    monkeypatch.setattr(
        start,
        "_require_agent_for_launch",
        lambda *args: pytest.fail("local prerequisites must be checked before installation"),
    )
    monkeypatch.setattr(
        start,
        "_connect",
        lambda *args, **kwargs: pytest.fail(
            "local prerequisites must be checked before connection"
        ),
    )

    result = CliRunner().invoke(start.start_app, ["pi", "--as-subagent"])

    assert result.exit_code == 1
    assert "Missing Pi subagent extension" in result.output


@pytest.fixture()
def fake_vllm(tmp_path, monkeypatch):
    """A vLLM-shaped server at BASE that the no-url scan finds, holding a key given before."""
    calls = []

    def request_json(method, url, key = None, payload = None, timeout = 10, headers = None):
        calls.append((method, url, payload))
        if method == "GET" and url == f"{BASE}/v1/models":
            listing = {"id": MODEL["id"], "owned_by": "vllm", "max_model_len": MODEL["context_length"]}
            return 200, {"object": "list", "data": [listing]}
        if method == "POST" and url in (f"{BASE}/v1/messages", f"{BASE}/v1/responses"):
            # A real route rejects the empty probe body.
            return 400, {"error": {"message": "model is required"}}
        return 404, {"detail": "Not Found"}

    monkeypatch.setattr(provider_utils, "request_json", request_json)
    monkeypatch.setattr(start.providers, "request_json", request_json)
    monkeypatch.setattr(start.providers, "scan_local_servers", lambda: [start.Target("vllm", BASE)])
    start._remember_key(start._provider_key_cache_path(), BASE, KEY)
    # --no-launch session configs land under tmp instead of the real agent-switch dir.
    monkeypatch.setattr(start, "_agents_config_root", lambda: tmp_path / "agents")
    monkeypatch.setattr(start, "_require_agent_for_launch", lambda *args: None)
    # Most existing assertions cover the stable V1 command; V2 has focused cases below.
    monkeypatch.setattr(start, "_opencode_command", lambda *_: ("opencode", False))
    # No `claude` on PATH, so _claude_flags never probes the real binary.
    monkeypatch.setattr(start.shutil, "which", lambda _: None)
    monkeypatch.delenv("AGENT_SWITCH_API_KEY", raising = False)
    return calls


def test_connect_claude_no_launch(fake_vllm):
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    assert result.exit_code == 0, result.output
    for name in start._CLAUDE_ENV_UNSET:
        _assert_env_unset(result.output, name)
    _assert_env_set(result.output, "ANTHROPIC_BASE_URL", BASE)
    _assert_env_set(result.output, "ANTHROPIC_AUTH_TOKEN", KEY)
    _assert_env_set(result.output, "ANTHROPIC_MODEL", MODEL["id"])
    _assert_env_set(result.output, "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
    _assert_env_set(result.output, "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS", "1")
    # Suppress the full-screen TUI redraw so a bursty local server doesn't flicker.
    _assert_env_set(result.output, "CLAUDE_CODE_NO_FLICKER", "1")
    # Attribution header is suppressed for the session via env + --settings, never
    # by writing the user's ~/.claude/settings.json.
    _assert_env_set(result.output, "CLAUDE_CODE_ATTRIBUTION_HEADER", "0")
    _assert_env_set(result.output, "CLAUDE_CODE_TOTAL_TOKENS_REMINDER", "off")
    # Claude assumes 200k for an unrecognized model id and clamps the auto-compact
    # window into [100k, that], so the real window has to be pinned as well.
    _assert_env_set(result.output, "CLAUDE_CODE_MAX_CONTEXT_TOKENS", str(MODEL["context_length"]))
    _assert_env_set(result.output, "CLAUDE_CODE_AUTO_COMPACT_WINDOW", str(MODEL["context_length"]))
    _assert_env_set(result.output, "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE", "90")
    assert f"claude --model {MODEL['id']} --exclude-dynamic-system-prompt-sections" in result.output
    # Overlay is session-only and lives outside the user's ~/.claude.
    command = _launch_command(result.output)
    settings_path = Path(command[command.index("--settings") + 1])
    settings = json.loads(settings_path.read_text())
    assert settings["env"]["CLAUDE_CODE_SUBAGENT_MODEL"] == "inherit"
    assert settings["env"]["ANTHROPIC_BASE_URL"] == BASE
    assert settings["env"]["ANTHROPIC_AUTH_TOKEN"] == KEY
    for name in start._CLAUDE_ENV_UNSET:
        assert settings["env"][name] == ""
    if os.name != "nt":
        assert settings_path.stat().st_mode & 0o777 == 0o600
    assert "--plugin-dir" not in command
    assert ".claude/settings.json" not in result.output


def test_connect_claude_session_settings_follow_forwarded_settings(fake_vllm):
    forwarded = json.dumps({"env": {"CLAUDE_CODE_USE_FOUNDRY": "1"}})
    result = CliRunner().invoke(
        start.start_app,
        ["claude", "--no-launch", "--settings", forwarded],
    )
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    positions = [index for index, arg in enumerate(command) if arg == "--settings"]
    assert len(positions) == 2
    assert command[positions[0] + 1] == forwarded
    assert Path(command[positions[1] + 1]).name.startswith("settings-")


@pytest.mark.parametrize(
    "settings_arg",
    [
        lambda value: ["--settings", value],
        lambda value: [f"--settings={value}"],
    ],
)
def test_connect_claude_session_settings_precede_subcommand(fake_vllm, settings_arg):
    forwarded = json.dumps({"env": {"CLAUDE_CODE_USE_FOUNDRY": "1"}})
    result = CliRunner().invoke(
        start.start_app,
        ["claude", "--no-launch", "mcp", "list", *settings_arg(forwarded)],
    )
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    subcommand = command.index("mcp")
    assert command.index("--model") < subcommand
    settings_positions = [
        index
        for index, arg in enumerate(command)
        if arg == "--settings" or arg.startswith("--settings=")
    ]
    assert len(settings_positions) == 2
    assert settings_positions[0] < settings_positions[1] < subcommand


def test_connect_claude_session_settings_precede_forwarded_delimiter(fake_vllm):
    forwarded = json.dumps({"env": {"CLAUDE_CODE_USE_FOUNDRY": "1"}})
    result = CliRunner().invoke(
        start.start_app,
        ["claude", "--no-launch", "--", "--settings", forwarded],
    )
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    positions = [index for index, arg in enumerate(command) if arg == "--settings"]
    assert len(positions) == 2
    assert positions[0] < command.index("--") < positions[1]
    assert Path(command[positions[0] + 1]).name.startswith("settings-")


def test_connect_claude_as_subagent_preserves_cloud_parent(fake_vllm, tmp_path):
    result = CliRunner().invoke(
        start.start_app,
        [
            "claude",
            "--as-subagent",
            "--no-launch",
            "--model",
            MODEL["id"],
            "hello",
        ],
    )
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    plugin = tmp_path / "agents" / "claude-subagent" / "local-agent"
    assert command == [
        "claude",
        "--plugin-dir",
        str(plugin),
        f"--allowedTools={start._CLAUDE_SUBAGENT_TOOL},{start._CLAUDE_SUBAGENT_PLAN_TOOL}",
        "hello",
    ]
    assert "--model" not in command
    parent_base = "$env:ANTHROPIC_BASE_URL" if os.name == "nt" else "export ANTHROPIC_BASE_URL="
    parent_token = (
        "$env:ANTHROPIC_AUTH_TOKEN" if os.name == "nt" else "export ANTHROPIC_AUTH_TOKEN="
    )
    assert parent_base not in result.output
    assert parent_token not in result.output
    assert "unset ANTHROPIC_API_KEY" not in result.output
    assert "AGENT_SWITCH_CLAUDE_SUBAGENT_API_KEY" not in result.output
    assert KEY not in result.output
    assert json.loads((plugin / ".claude-plugin" / "plugin.json").read_text())["name"] == (
        "local-agent"
    )
    mcp = json.loads((plugin / ".mcp.json").read_text())["mcpServers"]["local"]
    settings_path = next(plugin.glob("settings-*.json"))
    assert mcp["command"] == sys.executable
    assert mcp["args"] == ["-m", start._CLAUDE_SUBAGENT_MCP_MODULE]
    assert mcp["env"] == {
        "AGENT_SWITCH_CLAUDE_SUBAGENT_BASE_URL": BASE,
        "AGENT_SWITCH_CLAUDE_SUBAGENT_API_KEY": KEY,
        "AGENT_SWITCH_CLAUDE_SUBAGENT_MODEL": MODEL["id"],
        "AGENT_SWITCH_CLAUDE_SUBAGENT_BYPASS_PERMISSIONS": "0",
        "AGENT_SWITCH_CLAUDE_SUBAGENT_CONTEXT_WINDOW": str(MODEL["context_length"]),
        start._CLAUDE_SUBAGENT_SETTINGS_ENV: str(settings_path),
    }
    settings = json.loads(settings_path.read_text())
    assert settings["availableModels"] == [MODEL["id"]]
    assert settings["env"]["ANTHROPIC_BASE_URL"] == BASE
    assert settings["env"]["ANTHROPIC_AUTH_TOKEN"] == KEY
    for name in start._CLAUDE_ENV_UNSET:
        assert settings["env"][name] == ""
    skill = (plugin / "skills" / "local-agent" / "SKILL.md").read_text()
    assert "spawn a local agent" in skill
    assert "In plan mode" in skill
    assert "Ask Claude to spawn a local agent." in result.output


@pytest.mark.skipif(os.name == "nt", reason = "WSL scenario")
def test_claude_subagent_plugin_uses_wsl_for_windows_claude(monkeypatch, tmp_path):
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setenv("WSLENV", "EXISTING")
    monkeypatch.setattr(
        start.shutil,
        "which",
        lambda _: "/mnt/c/Users/x/AppData/Local/Programs/claude.exe",
    )
    server_env = {
        "AGENT_SWITCH_CLAUDE_SUBAGENT_BASE_URL": BASE,
        "AGENT_SWITCH_CLAUDE_SUBAGENT_API_KEY": "secret",
        "AGENT_SWITCH_CLAUDE_SUBAGENT_MODEL": MODEL["id"],
    }
    plugin = start.write_claude_subagent_plugin(tmp_path, server_env)
    mcp = json.loads((plugin / ".mcp.json").read_text())["mcpServers"]["local"]
    settings_path = next(plugin.glob("settings-*.json"))
    assert mcp["command"] == "wsl.exe"
    assert mcp["args"] == [
        "-d",
        "Ubuntu",
        "--",
        sys.executable,
        "-m",
        start._CLAUDE_SUBAGENT_MCP_MODULE,
    ]
    assert mcp["env"]["AGENT_SWITCH_CLAUDE_SUBAGENT_API_KEY"] == "secret"
    assert mcp["env"][start._CLAUDE_SUBAGENT_SETTINGS_ENV] == str(settings_path)
    assert mcp["env"]["WSLENV"].split(":") == [
        "EXISTING",
        "AGENT_SWITCH_CLAUDE_SUBAGENT_BASE_URL",
        "AGENT_SWITCH_CLAUDE_SUBAGENT_API_KEY",
        "AGENT_SWITCH_CLAUDE_SUBAGENT_MODEL",
        start._CLAUDE_SUBAGENT_SETTINGS_ENV,
    ]


def test_launch_native_posix_child_gets_current_pwd(fake_vllm, monkeypatch, tmp_path):
    captured = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PWD", "/stale/outer/repo")
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/opencode")

    def run(command, env):
        captured["command"] = command
        captured["env"] = env
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(start.subprocess, "run", run)

    result = CliRunner().invoke(start.start_app, ["opencode"])

    assert result.exit_code == 0, result.output
    assert captured["command"][0] == "/usr/local/bin/opencode"
    if os.name != "nt":
        assert captured["env"]["PWD"] == os.getcwd()


@pytest.mark.skipif(os.name == "nt", reason = "POSIX exec signal semantics")
def test_launch_leaves_child_able_to_handle_sigint(monkeypatch, tmp_path):
    # SIG_IGN here reached the agent too, so hermes could never be interrupted.
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import signal, sys\n"
        "sys.exit(17 if signal.getsignal(signal.SIGINT) == signal.SIG_IGN else 0)\n",
        encoding = "utf-8",
    )
    monkeypatch.setattr(start.shutil, "which", lambda _: sys.executable)
    monkeypatch.setattr(start, "_augment_path_with_install_dirs", lambda: None)
    before = signal.getsignal(signal.SIGINT)

    code = start._launch([sys.executable, str(probe)], {}, install_hint = "n/a")

    assert code == 0, "child saw SIG_IGN and could never be interrupted"
    assert signal.getsignal(signal.SIGINT) is before


def test_connect_claude_launch_scrubs_conflicting_auth_env(fake_vllm, monkeypatch):
    captured = {}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic-stale")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-stale")
    monkeypatch.setenv("ANTHROPIC_UNIX_SOCKET", "/tmp/remote-claude.sock")
    monkeypatch.setenv("CLAUDE_CODE_USE_FOUNDRY", "1")
    monkeypatch.setenv(
        "ANTHROPIC_FOUNDRY_BASE_URL",
        "https://corporate-gateway.azure-api.net/anthropic-stream",
    )
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_RESOURCE", "my-foundry-resource")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")
    monkeypatch.setenv("CLAUDE_CODE_USE_ANTHROPIC_AWS", "1")
    monkeypatch.setenv("CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD", "1")
    monkeypatch.setenv("CLAUDE_CODE_USE_MANTLE", "1")
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/claude")
    monkeypatch.setattr(start, "_claude_flags", lambda *a, **k: [])

    def run(command, env):
        captured["command"] = command
        captured["env"] = env
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(start.subprocess, "run", run)
    result = CliRunner().invoke(start.start_app, ["claude"])

    assert result.exit_code == 0, result.output
    assert captured["command"] == ["/usr/local/bin/claude", "--model", MODEL["id"]]
    for name in start._CLAUDE_ENV_UNSET:
        assert name not in captured["env"]
    assert captured["env"]["ANTHROPIC_AUTH_TOKEN"] == KEY
    assert captured["env"]["ANTHROPIC_BASE_URL"] == BASE
    assert captured["env"]["ANTHROPIC_MODEL"] == MODEL["id"]
    assert captured["env"]["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0"
    assert captured["env"]["CLAUDE_CODE_TOTAL_TOKENS_REMINDER"] == "off"


@pytest.mark.skipif(
    os.name == "nt",
    reason = "WSL-from-Linux scenario (calling a Windows agent .exe from inside WSL); "
    "os.name is 'posix' under WSL, so this path can't run on a native Windows runner.",
)
def test_connect_claude_windows_shim_from_wsl_bridges_env(fake_vllm, monkeypatch, tmp_path):
    captured = {}
    windows_settings = r"C:\\Users\\samle\\AppData\\Local\\agent-switch\\settings.json"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PWD", "/stale/outer/repo")
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic-stale")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-stale")
    monkeypatch.setattr(
        start.shutil, "which", lambda _: "/mnt/c/Users/samle/AppData/Roaming/npm/claude"
    )
    monkeypatch.setattr(start, "_wsl_windows_path", lambda _: windows_settings)
    monkeypatch.setattr(
        start,
        "_claude_flags",
        lambda model_id, settings: ["--settings", settings],
    )

    def run(command, env):
        captured["command"] = command
        captured["env"] = env
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(start.subprocess, "run", run)
    result = CliRunner().invoke(start.start_app, ["claude"])

    assert result.exit_code == 0, result.output
    assert captured["command"] == [
        "/mnt/c/Users/samle/AppData/Roaming/npm/claude",
        "--model",
        MODEL["id"],
        "--settings",
        windows_settings,
    ]
    for name in start._CLAUDE_ENV_UNSET:
        assert captured["env"][name] == ""
    assert captured["env"]["ANTHROPIC_AUTH_TOKEN"] == KEY
    assert captured["env"]["ANTHROPIC_BASE_URL"] == BASE
    assert captured["env"]["ANTHROPIC_MODEL"] == MODEL["id"]
    assert captured["env"]["PWD"] == str(tmp_path)

    assert "PWD/p" in captured["env"]["WSLENV"].split(":")
    for name in (
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
        *start._CLAUDE_ENV_UNSET,
    ):
        assert name in captured["env"]["WSLENV"].split(":")


def _npm_node_cmd_shim(
    target: str,
    *,
    node_args: str = "",
    environment: tuple[tuple[str, str], ...] = (),
    cmd_shim_version: int = 7,
) -> str:
    environment_lines = "".join(f"@SET {name}={value}\r\n" for name, value in environment)
    legacy_pathext = "  SET PATHEXT=%PATHEXT:;.JS;=;%\r\n" if cmd_shim_version < 9 else ""
    current_pathext = "set PATHEXT=%PATHEXT:;.JS;=;% & " if cmd_shim_version >= 9 else ""
    return (
        "@ECHO off\r\n"
        "GOTO start\r\n"
        ":find_dp0\r\n"
        "SET dp0=%~dp0\r\n"
        "EXIT /b\r\n"
        ":start\r\n"
        "SETLOCAL\r\n"
        "CALL :find_dp0\r\n"
        f"{environment_lines}"
        "\r\n"
        'IF EXIST "%dp0%\\node.exe" (\r\n'
        '  SET "_prog=%dp0%\\node.exe"\r\n'
        ") ELSE (\r\n"
        '  SET "_prog=node"\r\n'
        f"{legacy_pathext}"
        ")\r\n"
        "\r\n"
        f"endLocal & goto #_undefined_# 2>NUL || title %COMSPEC% & {current_pathext}"
        f'"%_prog%" {node_args} "%dp0%\\{target}" %*\r\n'
    )


def test_launch_windows_npm_shim_preserves_multiline_argument(monkeypatch, tmp_path):
    _simulate_windows(monkeypatch)
    cmd = tmp_path / "fake-agent.cmd"
    target = tmp_path / "node_modules" / "fake-agent" / "index.js"
    target.parent.mkdir(parents = True)
    target.write_text("", encoding = "utf-8")
    cmd.write_bytes(_npm_node_cmd_shim(r"node_modules\fake-agent\index.js").encode())
    captured = {}

    def which(name):
        return str(cmd) if name == "fake-agent" else r"C:\Program Files\nodejs\node.exe"

    def run(command, env):
        captured["command"] = command
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(start.shutil, "which", which)
    monkeypatch.setattr(start.subprocess, "run", run)

    code = start._launch(
        ["fake-agent", 'first line\nsecond "quoted" line'],
        {},
        install_hint = "unused",
    )

    assert code == 0
    assert captured["command"] == [
        r"C:\Program Files\nodejs\node.exe",
        str(target),
        'first line\nsecond "quoted" line',
    ]


def test_resolved_launch_command_handles_current_npm_node_shim(monkeypatch, tmp_path):
    _simulate_windows(monkeypatch)
    cmd = tmp_path / "fake-agent.cmd"
    target = tmp_path / "node_modules" / "fake-agent" / "index.js"
    target.parent.mkdir(parents = True)
    target.write_text("#!/usr/bin/env node\n", encoding = "utf-8")
    cmd.write_bytes(
        _npm_node_cmd_shim(
            r"node_modules\fake-agent\index.js",
            cmd_shim_version = 9,
        ).encode()
    )
    monkeypatch.setattr(start.shutil, "which", lambda name: r"C:\Program Files\nodejs\node.exe")

    assert start._resolved_launch_command(str(cmd), ["--flag"]) == [
        r"C:\Program Files\nodejs\node.exe",
        str(target),
        "--flag",
    ]


def test_resolved_launch_command_ignores_node_js_pathext_shadow(monkeypatch, tmp_path):
    _simulate_windows(monkeypatch)
    cmd = tmp_path / "fake-agent.cmd"
    target = tmp_path / "node_modules" / "fake-agent" / "index.js"
    target.parent.mkdir(parents = True)
    target.write_text("#!/usr/bin/env node\n", encoding = "utf-8")
    cmd.write_bytes(_npm_node_cmd_shim(r"node_modules\fake-agent\index.js").encode())
    resolved_names = []

    def which(name):
        resolved_names.append(name)
        return {
            "node": r"C:\shadow\node.js",
            "node.exe": r"C:\Program Files\nodejs\node.exe",
        }.get(name)

    monkeypatch.setattr(start.shutil, "which", which)

    assert start._resolved_launch_command(str(cmd), ["--flag"]) == [
        r"C:\Program Files\nodejs\node.exe",
        str(target),
        "--flag",
    ]
    assert resolved_names == ["node.exe"]


def test_resolved_launch_command_accepts_extensionless_node_target(monkeypatch, tmp_path):
    _simulate_windows(monkeypatch)
    cmd = tmp_path / "fake-agent.cmd"
    target = tmp_path / "node_modules" / "fake-agent" / "cli"
    target.parent.mkdir(parents = True)
    target.write_text("#!/usr/bin/env node\n", encoding = "utf-8")
    cmd.write_bytes(_npm_node_cmd_shim(r"node_modules\fake-agent\cli").encode())
    monkeypatch.setattr(start.shutil, "which", lambda name: r"C:\Program Files\nodejs\node.exe")

    assert start._resolved_launch_command(str(cmd), ["--flag"]) == [
        r"C:\Program Files\nodejs\node.exe",
        str(target),
        "--flag",
    ]


def test_resolved_launch_command_prefers_cmd_sibling_of_extensionless_shim(monkeypatch, tmp_path):
    # which() can resolve the extensionless POSIX shim ahead of its .cmd sibling;
    # CreateProcess rejects the former with WinError 193 (#9167).
    _simulate_windows(monkeypatch)
    posix_shim = tmp_path / "fake-agent"
    posix_shim.write_text("#!/bin/sh\n", encoding = "utf-8")
    cmd = tmp_path / "fake-agent.cmd"
    target = tmp_path / "node_modules" / "fake-agent" / "index.js"
    target.parent.mkdir(parents = True)
    target.write_text("#!/usr/bin/env node\n", encoding = "utf-8")
    cmd.write_bytes(_npm_node_cmd_shim(r"node_modules\fake-agent\index.js").encode())
    monkeypatch.setattr(start.shutil, "which", lambda name: r"C:\Program Files\nodejs\node.exe")

    assert start._resolved_launch_command(str(posix_shim), ["--flag"]) == [
        r"C:\Program Files\nodejs\node.exe",
        str(target),
        "--flag",
    ]


def test_resolved_launch_command_keeps_extensionless_shim_without_sibling(monkeypatch, tmp_path):
    # Nothing to substitute, so the path passes through unchanged.
    _simulate_windows(monkeypatch)
    executable = tmp_path / "fake-agent"
    executable.write_text("#!/bin/sh\n", encoding = "utf-8")

    assert start._resolved_launch_command(str(executable), ["--flag"]) == [
        str(executable),
        "--flag",
    ]


def test_prefer_cmd_sibling_leaves_an_unreadable_resolution_alone(monkeypatch, tmp_path):
    # A directory, an unreadable file, or a delete between resolve and open must
    # fall through instead of raising out of the launch path.
    _simulate_windows(monkeypatch)
    executable = tmp_path / "fake-agent"
    executable.mkdir()
    (tmp_path / "fake-agent.cmd").write_text("@ECHO off\n", encoding = "utf-8")

    assert start._resolved_launch_command(str(executable), ["--flag"]) == [
        str(executable),
        "--flag",
    ]


def test_prefer_cmd_sibling_is_none_safe_and_posix_noop(monkeypatch, tmp_path):
    assert start._prefer_windows_cmd_sibling(None) is None
    # Pin os.name instead of relying on the host: on a Windows runner the rescue
    # would fire and this would assert the opposite of what it means to check.
    monkeypatch.setattr(start.os, "name", "posix")
    shim = tmp_path / "fake-agent"
    shim.write_text("#!/bin/sh\n", encoding = "utf-8")
    (tmp_path / "fake-agent.cmd").write_text("@ECHO off\n", encoding = "utf-8")
    assert start._prefer_windows_cmd_sibling(str(shim)) == str(shim)


def test_resolved_launch_command_rescues_uppercase_cmd_sibling(monkeypatch, tmp_path):
    # #9167's pnpm dir holds pi.CMD. A case-sensitive volume needs the .CMD probe;
    # a case-insensitive one answers the earlier .cmd probe with the same file, so
    # compare identity rather than spelling or this passes only on Linux.
    _simulate_windows(monkeypatch)
    posix_shim = tmp_path / "fake-agent"
    posix_shim.write_text("#!/bin/sh\n", encoding = "utf-8")
    cmd = tmp_path / "fake-agent.CMD"
    cmd.write_text("@ECHO off\ncustom-wrapper %*\n", encoding = "utf-8")

    resolved = start._resolved_launch_command(str(posix_shim), ["--flag"])
    assert resolved[1:] == ["--flag"]
    assert os.path.samefile(resolved[0], str(cmd))


def test_which_with_install_dirs_applies_the_cmd_sibling_preference(monkeypatch, tmp_path):
    # The version probes spawn this result directly, bypassing
    # _resolved_launch_command, so the rescue must happen here too.
    _simulate_windows(monkeypatch)
    posix_shim = tmp_path / "fake-agent"
    posix_shim.write_text("#!/bin/sh\n", encoding = "utf-8")
    cmd = tmp_path / "fake-agent.cmd"
    cmd.write_text("@ECHO off\n", encoding = "utf-8")
    monkeypatch.setattr(start, "_augment_path_with_install_dirs", lambda: None)
    monkeypatch.setattr(start.shutil, "which", lambda name: str(posix_shim))

    assert start._which_with_install_dirs("fake-agent") == str(cmd)


def test_resolved_launch_command_rescues_dotted_bin_name_shim(monkeypatch, tmp_path):
    # cmd-shim appends .cmd to the whole bin name, so "foo.bar" pairs with
    # "foo.bar.cmd" and still needs the sibling preference.
    _simulate_windows(monkeypatch)
    posix_shim = tmp_path / "fake.agent"
    posix_shim.write_text("#!/bin/sh\n", encoding = "utf-8")
    cmd = tmp_path / "fake.agent.cmd"
    target = tmp_path / "node_modules" / "fake-agent" / "index.js"
    target.parent.mkdir(parents = True)
    target.write_text("#!/usr/bin/env node\n", encoding = "utf-8")
    cmd.write_bytes(_npm_node_cmd_shim(r"node_modules\fake-agent\index.js").encode())
    monkeypatch.setattr(start.shutil, "which", lambda name: r"C:\Program Files\nodejs\node.exe")

    assert start._resolved_launch_command(str(posix_shim), ["--flag"]) == [
        r"C:\Program Files\nodejs\node.exe",
        str(target),
        "--flag",
    ]


def test_resolved_launch_command_keeps_extensionless_pe_binary_over_stale_sibling(
    monkeypatch, tmp_path
):
    # CreateProcess runs a PE regardless of its name, so a real executable keeps
    # priority over a stale .cmd; only shebang files are treated as shims.
    _simulate_windows(monkeypatch)
    executable = tmp_path / "fake-agent"
    executable.write_bytes(b"MZ\x90\x00")
    stale = tmp_path / "fake-agent.cmd"
    stale.write_text("@ECHO off\nold-wrapper %*\n", encoding = "utf-8")

    assert start._resolved_launch_command(str(executable), ["--flag"]) == [
        str(executable),
        "--flag",
    ]


def test_resolved_launch_command_falls_through_to_non_npm_cmd_sibling(monkeypatch, tmp_path):
    # A non-npm .cmd sibling is still what cmd.exe would have picked, so it is
    # returned as-is.
    _simulate_windows(monkeypatch)
    posix_shim = tmp_path / "fake-agent"
    posix_shim.write_text("#!/bin/sh\n", encoding = "utf-8")
    cmd = tmp_path / "fake-agent.cmd"
    cmd.write_text("@ECHO off\ncustom-wrapper %*\n", encoding = "utf-8")

    assert start._resolved_launch_command(str(posix_shim), ["--flag"]) == [str(cmd), "--flag"]


def test_launch_windows_npm_shim_preserves_shebang_args_and_environment(monkeypatch, tmp_path):
    _simulate_windows(monkeypatch)
    cmd = tmp_path / "fake-agent.cmd"
    target = tmp_path / "node_modules" / "fake-agent" / "index.js"
    target.parent.mkdir(parents = True)
    target.write_text(
        "#!/usr/bin/env NODE_OPTIONS=--trace-warnings node --no-warnings\n",
        encoding = "utf-8",
    )
    cmd.write_bytes(
        _npm_node_cmd_shim(
            r"node_modules\fake-agent\index.js",
            node_args = "--no-warnings",
            environment = (("NODE_OPTIONS", "--trace-warnings"),),
        ).encode()
    )
    captured = {}

    def which(name):
        return str(cmd) if name == "fake-agent" else r"C:\Program Files\nodejs\node.exe"

    def run(command, env):
        captured["command"] = command
        captured["env"] = env
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(start.shutil, "which", which)
    monkeypatch.setattr(start.subprocess, "run", run)

    code = start._launch(
        ["fake-agent", 'first line\nsecond "quoted" line'],
        {},
        install_hint = "unused",
    )

    assert code == 0
    assert captured["command"] == [
        r"C:\Program Files\nodejs\node.exe",
        "--no-warnings",
        str(target),
        'first line\nsecond "quoted" line',
    ]
    assert captured["env"]["NODE_OPTIONS"] == "--trace-warnings"


def test_resolved_launch_command_uses_native_npm_entrypoint(monkeypatch, tmp_path):
    _simulate_windows(monkeypatch)
    cmd = tmp_path / "fake-agent.cmd"
    target = tmp_path / "node_modules" / "fake-agent" / "agent.exe"
    target.parent.mkdir(parents = True)
    target.write_bytes(b"")
    cmd.write_bytes(
        (
            "@ECHO off\r\n"
            "GOTO start\r\n"
            ":find_dp0\r\n"
            "SET dp0=%~dp0\r\n"
            "EXIT /b\r\n"
            ":start\r\n"
            "SETLOCAL\r\n"
            "CALL :find_dp0\r\n"
            '"%dp0%\\node_modules\\fake-agent\\agent.exe" %*\r\n'
        ).encode()
    )

    assert start._resolved_launch_command(str(cmd), ["--flag", "two words"]) == [
        str(target),
        "--flag",
        "two words",
    ]


def test_resolved_launch_command_handles_project_local_npm_shim(monkeypatch, tmp_path):
    _simulate_windows(monkeypatch)
    node_modules = tmp_path / "project" / "node_modules"
    cmd = node_modules / ".bin" / "fake-agent.cmd"
    target = node_modules / "fake-agent" / "index.js"
    cmd.parent.mkdir(parents = True)
    target.parent.mkdir(parents = True)
    target.write_text("", encoding = "utf-8")
    cmd.write_bytes(_npm_node_cmd_shim(r"..\fake-agent\index.js").encode())
    monkeypatch.setattr(start.shutil, "which", lambda name: r"C:\Program Files\nodejs\node.exe")

    assert start._resolved_launch_command(str(cmd), ["--flag"]) == [
        r"C:\Program Files\nodejs\node.exe",
        str(target),
        "--flag",
    ]


def test_resolved_launch_command_leaves_custom_npm_like_wrapper_unchanged(monkeypatch, tmp_path):
    _simulate_windows(monkeypatch)
    cmd = tmp_path / "custom-agent.cmd"
    target = tmp_path / "node_modules" / "fake-agent" / "index.js"
    target.parent.mkdir(parents = True)
    target.write_text("", encoding = "utf-8")
    contents = _npm_node_cmd_shim(r"node_modules\fake-agent\index.js")
    cmd.write_bytes(
        contents.replace(
            "CALL :find_dp0\r\n", "CALL :find_dp0\r\nSET AGENT_MODE=custom\r\n"
        ).encode()
    )

    assert start._resolved_launch_command(str(cmd), ["--flag"]) == [str(cmd), "--flag"]


def test_resolved_launch_command_leaves_non_npm_batch_file_unchanged(monkeypatch, tmp_path):
    _simulate_windows(monkeypatch)
    cmd = tmp_path / "custom-agent.cmd"
    cmd.write_bytes(b'@echo off\r\n"%dp0%\\custom.exe" %*\r\n')

    assert start._resolved_launch_command(str(cmd), ["--flag"]) == [str(cmd), "--flag"]


@pytest.mark.skipif(
    os.name == "nt",
    reason = "WSL-from-Linux scenario (calling a Windows agent .exe from inside WSL); "
    "os.name is 'posix' under WSL, so this path can't run on a native Windows runner.",
)
def test_connect_claude_no_launch_windows_shim_from_wsl_prints_wslenv(
    fake_vllm, monkeypatch, tmp_path
):
    windows_settings = r"C:\\Users\\samle\\AppData\\Local\\agent-switch\\settings.json"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PWD", "/stale/outer/repo")
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr(
        start.shutil, "which", lambda _: "/mnt/c/Users/samle/AppData/Roaming/npm/claude"
    )
    monkeypatch.setattr(start, "_wsl_windows_path", lambda _: windows_settings)

    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])

    assert result.exit_code == 0, result.output
    for name in start._CLAUDE_ENV_UNSET:
        assert f"export {name}=" in result.output
    assert "export WSLENV=" in result.output
    # PWD must NOT be frozen into the recipe (no `export PWD=`): WSLENV PWD/p translates the
    # shell's live PWD at run time, so a recipe reused from another dir resolves the project root.
    assert "export PWD=" not in result.output
    assert "PWD/p" in result.output
    assert "ANTHROPIC_AUTH_TOKEN" in result.output
    assert "CLAUDE_CODE_OAUTH_TOKEN" in result.output
    command = _launch_command(result.output)
    assert command[command.index("--settings") + 1] == windows_settings


def test_connect_codex_no_launch(fake_vllm, tmp_path):
    result = CliRunner().invoke(start.start_app, ["codex", "--no-launch"])
    assert result.exit_code == 0, result.output
    for name in start._CODEX_ENV_UNSET:
        _assert_env_unset(result.output, name)
    _assert_env_set(result.output, "AGENT_SWITCH_AUTH_TOKEN", KEY)
    assert "codex --oss --profile agent_switch" in result.output
    # Config lands in the session-scoped CODEX_HOME, not the user's ~/.codex.
    home = tmp_path / "agents" / "codex"
    _assert_env_set(result.output, "CODEX_HOME", str(home))
    assert (home / "config.toml").exists()
    assert (home / "agent_switch.config.toml").exists()


def test_connect_codex_as_subagent_preserves_cloud_parent(fake_vllm, tmp_path, monkeypatch):
    monkeypatch.setattr(start, "_codex_supports_model_catalog", lambda: True)
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
    for name in start._CODEX_ENV_UNSET:
        _assert_env_kept(result.output, name)
    assert start._CODEX_ENV_KEY not in result.output
    assert KEY not in result.output
    home = tmp_path / "agents" / "codex-subagent"
    bridge_path = home / "subagent.json"
    bridge = json.loads(bridge_path.read_text())
    assert bridge["api_key"] == KEY
    assert bridge["codex_home"] == str(home / "child")
    assert bridge["bypass_permissions"] is False
    profile = _parse_toml((home / "child" / "agent_switch.config.toml").read_text())
    assert profile["model"] == MODEL["id"]
    prefix = f"mcp_servers.{start._CODEX_SUBAGENT_MCP_SERVER}="
    override = next(value for value in command if value.startswith(prefix))
    assert override.startswith(prefix)
    server = _parse_toml("server = " + override.removeprefix(prefix))["server"]
    assert server["command"] == sys.executable
    assert server["args"] == ["-c", server["args"][1], str(bridge_path)]
    assert "sys.path.insert" in server["args"][1]
    assert f"from {start._CODEX_SUBAGENT_MCP_MODULE} import main" in server["args"][1]
    assert server["enabled_tools"] == [start._CODEX_SUBAGENT_MCP_TOOL]
    assert not any(value.startswith("developer_instructions=") for value in command)
    parent_instructions = (parent_home / "AGENTS.md").read_text()
    assert parent_instructions.startswith("Keep the user's guidance.\n")
    assert start._CODEX_SUBAGENT_ROUTING_INSTRUCTIONS in parent_instructions
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
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/codex")

    def run(command, env):
        captured["home"] = env["CODEX_HOME"]
        captured["config_present"] = (Path(env["CODEX_HOME"]) / "config.toml").exists()
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(start.subprocess, "run", run)
    result = CliRunner().invoke(start.start_app, ["codex"])
    assert result.exit_code == 0, result.output
    home = Path(captured["home"])
    assert captured["config_present"]  # config existed while codex ran
    parent = start._ephemeral_session_parent("codex")
    assert home.name.startswith(start._ephemeral_session_prefix("codex", parent))
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


def test_no_launch_claude_last_line_blanks_conflicting_auth(fake_vllm):
    # The unset vars must be neutralized inline too, or a partial copy would send the
    # user's own ANTHROPIC_API_KEY to the local base.
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    assert result.exit_code == 0, result.output
    last = [ln for ln in result.output.splitlines() if ln.strip()][-1]
    for name in start._CLAUDE_ENV_UNSET:
        assert f"{name}= " in last
    assert "ANTHROPIC_AUTH_TOKEN=" in last  # the real key still applied after the blanks


def test_opencode_inline_config_beats_project_config(fake_vllm):
    # A project's opencode.json outranks OPENCODE_CONFIG, so the model pin (and --yolo
    # permissions) ride in OPENCODE_CONFIG_CONTENT, which outranks project config.
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch", "--yolo"])
    assert result.exit_code == 0, result.output
    inline = _opencode_inline_config(result.output)
    assert inline["model"] == f"{start._OPENCODE_PROVIDER}/{MODEL['id']}"
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
    assert inline["model"] == f"{start._OPENCODE_PROVIDER}/{MODEL['id']}"
    assert "permission" not in inline


def test_connect_explicit_key_remembered_for_keyless_runs(fake_vllm, tmp_path):
    CliRunner().invoke(
        start.start_app,
        ["claude", "--no-launch", "--api-key", "sk-test-deadbeefdeadbeef"],
    )
    result = CliRunner().invoke(start.start_app, ["claude", "--no-launch"])
    assert result.exit_code == 0, result.output
    # Reused ahead of the key the server was given before.
    _assert_env_set(result.output, "ANTHROPIC_AUTH_TOKEN", "sk-test-deadbeefdeadbeef")
    cached = json.loads(start._provider_key_cache_path().read_text())
    assert cached["servers"][BASE]["saved"] == ["sk-test-deadbeefdeadbeef", KEY]


@pytest.mark.parametrize(
    "model, expected",
    [
        ("org/Qwen3-1.7B-GGUF:UD-Q4_K_XL", ("org/Qwen3-1.7B-GGUF", "UD-Q4_K_XL")),
        ("org/gemma-4-E2B-it-GGUF:Q8_0", ("org/gemma-4-E2B-it-GGUF", "Q8_0")),
        ("org/Qwen3-1.7B-GGUF", ("org/Qwen3-1.7B-GGUF", None)),  # no suffix
        ("/models/local.gguf", ("/models/local.gguf", None)),  # absolute path
        ("./rel.gguf", ("./rel.gguf", None)),  # relative path
        ("C:\\models\\x.gguf", ("C:\\models\\x.gguf", None)),  # Windows drive
        ("repo:with/slash", ("repo:with/slash", None)),  # slash in variant -> not a variant
        ("", ("", None)),
    ],
)
def test_split_repo_variant(model, expected):
    assert start._split_repo_variant(model) == expected


@pytest.mark.parametrize(
    "token, expected",
    [
        ("org/gemma-4-E2B-it-GGUF", True),
        ("org/gemma-4-E2B-it-GGUF:UD-Q4_K_XL", True),
        ("some-org/model.name_1", True),
        ("--continue", False),  # flag
        ("resume", False),  # single word, no slash
        ("/models/local.gguf", False),  # absolute path
        ("./rel.gguf", False),  # relative path
        ("C:\\models\\x.gguf", False),  # Windows drive
        ("my models/foo", False),  # has a space
        ("owner/repo/extra", False),  # too many segments
    ],
)
def test_looks_like_model(token, expected):
    assert start._looks_like_model(token) is expected


def test_consume_positional_model_leading_token():
    # A leading org/name positional routes to --model and is dropped from the passthrough.
    model, rest = start._consume_positional_model(None, ["org/Model-GGUF", "--continue"])
    assert model == "org/Model-GGUF"
    assert rest == ["--continue"]


def test_looks_like_model_leaves_existing_local_dir_for_agent(tmp_path, monkeypatch):
    # A relative `owner/repo` that actually exists (e.g. an OpenCode project dir) must
    # stay an agent argument, not be consumed as a model.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "owner" / "repo").mkdir(parents = True)
    assert start._looks_like_model("owner/repo") is False
    model, rest = start._consume_positional_model(None, ["owner/repo"])
    assert model is None and rest == ["owner/repo"]
    # The same shape, when it does not exist locally, is still treated as a model.
    assert start._looks_like_model("owner/absent-repo") is True


def test_consume_positional_model_ignores_non_leading_and_explicit_model():
    # An org/name that is an option value (not leading) is never stolen.
    model, rest = start._consume_positional_model(None, ["--profile", "owner/repo"])
    assert model is None and rest == ["--profile", "owner/repo"]
    # An explicit --model always wins; the positional is left untouched.
    model, rest = start._consume_positional_model("explicit/model", ["owner/repo"])
    assert model == "explicit/model" and rest == ["owner/repo"]


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


_SESSION_FLAGS = ["--temperature", "0.3", "--top-k", "40", "--reasoning", "off"]
# vLLM reads the reasoning switch from chat_template_kwargs.
_SESSION_BODY = {"temperature": 0.3, "top_k": 40, "chat_template_kwargs": {"enable_thinking": False}}


def _session_request_body(agent, root, output):
    home = root / "agents" / agent
    if agent == "pi":
        config = json.loads((home / ".pi" / "agent" / "models.json").read_text())
        return config["providers"]["agent-switch"]["models"][0].get("samplingParams")
    if agent == "opencode":
        provider = json.loads((home / "opencode.json").read_text())["provider"]
        provider = provider[start._OPENCODE_PROVIDER]
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
    result = CliRunner().invoke(start.start_app, [agent, "--no-launch", *_SESSION_FLAGS])
    assert result.exit_code == 0, result.output
    assert "already running" not in result.output
    assert _session_request_body(agent, tmp_path, result.output) == _SESSION_BODY


@pytest.mark.parametrize("agent", ["pi", "opencode", "claude"])
def test_session_flags_from_an_earlier_run_do_not_stick(agent, fake_vllm, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for argv in ([agent, "--no-launch", *_SESSION_FLAGS], [agent, "--no-launch"]):
        result = CliRunner().invoke(start.start_app, argv)
        assert result.exit_code == 0, result.output
    assert not _session_request_body(agent, tmp_path, result.output)


def test_opencode_session_temperature_needs_the_capability(fake_vllm, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "agents" / "opencode" / "opencode.json"
    for argv, capability in ((["--temperature", "0.3"], True), (["--top-k", "40"], None)):
        result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch", *argv])
        assert result.exit_code == 0, result.output
        provider = json.loads(config_path.read_text())["provider"][start._OPENCODE_PROVIDER]
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
        start._OPENCODE_PROVIDER
    ]
    assert provider["options"]["headers"] == {"Authorization": "Bearer gateway-token"}
    assert "apiKey" not in provider["options"]


def test_pi_subagent_carries_the_session_flags(fake_vllm, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        start.start_app, ["pi", "--as-subagent", "--no-launch", *_SESSION_FLAGS]
    )
    assert result.exit_code == 0, result.output
    assert "already running" not in result.output
    config = json.loads((tmp_path / "agents" / "pi-subagent" / "subagent.json").read_text())
    assert config["samplingParams"] == _SESSION_BODY


def test_codex_carries_reasoning_and_warns_about_sampling(fake_vllm, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(start.start_app, ["codex", "--no-launch", *_SESSION_FLAGS])
    assert result.exit_code == 0, result.output
    profile = (tmp_path / "agents" / "codex" / f"{start._CODEX_PROFILE}.config.toml").read_text()
    assert 'model_reasoning_effort = "none"' in profile
    assert "can't send --temperature, --top-k itself" in result.output
    assert "--reasoning" not in result.output
    result = CliRunner().invoke(start.start_app, ["codex", "--no-launch"])
    assert result.exit_code == 0, result.output
    profile = (tmp_path / "agents" / "codex" / f"{start._CODEX_PROFILE}.config.toml").read_text()
    assert "model_reasoning_effort" not in profile


def test_codex_warns_about_reasoning_it_cannot_express(fake_vllm, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(start.start_app, ["codex", "--no-launch", "--reasoning", "on"])
    assert result.exit_code == 0, result.output
    assert "can't send --reasoning itself" in result.output
    profile = (tmp_path / "agents" / "codex" / f"{start._CODEX_PROFILE}.config.toml").read_text()
    assert "model_reasoning_effort" not in profile


@pytest.mark.parametrize("mode", ["--persist", "--no-launch"])
@pytest.mark.parametrize("agent, version", [("codex", (0, 144, 0)), ("pi", (0, 83, 0))])
def test_agent_too_old_to_send_the_flags_warns_and_ignores(
    agent, version, mode, fake_vllm, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(start, "_which_with_install_dirs", lambda name: f"/bin/{name}")
    monkeypatch.setattr(start, "_codex_executable_version", lambda executable: version)
    monkeypatch.setattr(start, "_launch", lambda *args, **kwargs: None)
    result = CliRunner().invoke(start.start_app, [agent, mode, *_SESSION_FLAGS])
    assert result.exit_code == 0, result.output
    assert "can't send --temperature, --top-k, --reasoning itself" in result.output
    if agent == "pi":
        models = tmp_path / "agents" / "pi" / ".pi" / "agent" / "models.json"
        provider = json.loads(models.read_text())["providers"]["agent-switch"]
        assert "samplingParams" not in provider["models"][0]
    else:
        profile = (
            tmp_path / "agents" / "codex" / f"{start._CODEX_PROFILE}.config.toml"
        ).read_text()
        assert "model_reasoning_effort" not in profile


def test_opencode_v1_reads_the_effort_as_reasoning_effort_option(
    fake_vllm, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        start.start_app, ["opencode", "--no-launch", "--reasoning-effort", "low"]
    )
    assert result.exit_code == 0, result.output
    provider = json.loads((tmp_path / "agents" / "opencode" / "opencode.json").read_text())
    provider = provider["provider"][start._OPENCODE_PROVIDER]
    assert provider["models"][MODEL["id"]]["options"] == {"reasoningEffort": "low"}
    assert provider["options"]["body"] == {"reasoning_effort": "low"}


def test_dsh_carries_reasoning_and_warns_about_sampling(fake_vllm, tmp_path, monkeypatch):
    yaml = pytest.importorskip("yaml")
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(start.start_app, ["dsh", "--no-launch", *_SESSION_FLAGS])
    assert result.exit_code == 0, result.output
    assert "can't send --temperature, --top-k itself" in result.output
    assert "--reasoning" not in result.output
    patch = tmp_path / "agents" / "dsh" / start._DSH_PATCH_FILE
    provider = yaml.safe_load(patch.read_text())[0]["config"]["providers"][start._DSH_PROVIDER]
    assert provider["compat"]["thinkingFormat"] == "chat-template"
    assert provider["compat"]["chatTemplateKwargs"] == {"enable_thinking": False}
    assert "off" in provider["models"][0]["reasoningEfforts"]


def test_launch_prints_the_ready_line_and_exits_with_the_agent(fake_vllm, monkeypatch):
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/claude")
    monkeypatch.setattr(start, "_claude_flags", lambda *a, **k: [])
    monkeypatch.setattr(
        start.subprocess,
        "run",
        lambda command, env: SimpleNamespace(returncode = 0),
    )

    result = CliRunner().invoke(start.start_app, ["claude"])

    assert result.exit_code == 0, result.output
    assert f"vLLM ready at {BASE} · model {MODEL['id']}\n" in result.output
    assert "still running" not in result.output


def test_nonzero_agent_exit_notes_code(fake_vllm, monkeypatch):
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/claude")
    monkeypatch.setattr(start, "_claude_flags", lambda *a, **k: [])
    monkeypatch.setattr(
        start.subprocess,
        "run",
        lambda command, env: SimpleNamespace(returncode = 3),
    )

    result = CliRunner().invoke(start.start_app, ["claude"])

    assert result.exit_code == 3
    assert "The agent exited with code 3." in result.output


def test_connect_explicit_api_key_wins_over_a_saved_one(fake_vllm):
    result = CliRunner().invoke(
        start.start_app,
        ["claude", "--no-launch", "--api-key", "sk-test-deadbeefdeadbeef"],
    )
    assert result.exit_code == 0, result.output
    _assert_env_set(result.output, "ANTHROPIC_AUTH_TOKEN", "sk-test-deadbeefdeadbeef")


# ── OpenClaw (Anthropic /v1/messages) ────────────────────────────────


# ── OpenCode (OpenAI /v1/chat/completions) ───────────────────────────


def test_write_opencode_config_fresh(tmp_path):
    path = tmp_path / "opencode.json"
    start.write_opencode_config(BASE, "sk-test-abc", MODEL, path)
    config = json.loads(path.read_text())
    provider = config["provider"][start._OPENCODE_PROVIDER]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"] == {"baseURL": f"{BASE}/v1", "apiKey": "sk-test-abc"}
    # Context limit must be declared, or OpenCode treats it as 0 and disables compaction.
    assert provider["models"] == {
        MODEL["id"]: {
            "name": MODEL["id"],
            "limit": {"context": 131072, "input": 131072, "output": 32_000},
        }
    }
    assert config["model"] == f"{start._OPENCODE_PROVIDER}/{MODEL['id']}"
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
    assert start.opencode_output_limit(window, max_tokens) == expected


def test_opencode_max_tokens_sets_limit_and_raises_opencode_ceiling(fake_vllm, tmp_path):
    result = CliRunner().invoke(
        start.start_app, ["opencode", "--no-launch", "--max-tokens", "65536"]
    )
    assert result.exit_code == 0, result.output
    config_path = tmp_path / "agents" / "opencode" / "opencode.json"
    config = json.loads(config_path.read_text())
    limit = config["provider"][start._OPENCODE_PROVIDER]["models"][MODEL["id"]]["limit"]
    assert limit == {"context": 131072, "input": 131072, "output": 65536}
    _assert_env_set(result.output, "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX", "65536")


def test_opencode_limit_input_keeps_compaction_off_the_output_limit(tmp_path):
    # Without input, OpenCode compacts at context - output and a 65,536 limit compacts at half full.
    path = tmp_path / "opencode.json"
    start.write_opencode_config(BASE, "sk-test-abc", MODEL, path, max_tokens = 65536)
    config = json.loads(path.read_text())
    limit = config["provider"][start._OPENCODE_PROVIDER]["models"][MODEL["id"]]["limit"]
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
    reserved = start.opencode_compaction_reserved(window, start.opencode_output_limit(window))
    assert reserved == expected_reserved
    assert window - reserved == expected_compacts_at


def test_opencode_subagent_drops_the_compaction_a_normal_session_wrote(tmp_path):
    path = tmp_path / "opencode.json"
    small = {**MODEL, "context_length": 16_384}
    start.write_opencode_config(BASE, "sk-test-abc", small, path)
    assert json.loads(path.read_text())["compaction"] == {"auto": True, "reserved": 4_096}
    start.write_opencode_config(BASE, "sk-test-abc", small, path, as_subagent = True)
    assert "compaction" not in json.loads(path.read_text())


def test_opencode_max_tokens_without_a_window_warns(capsys):
    assert start._opencode_output_env({"id": "m"}, 65536) == {}
    assert "--max-tokens is ignored" in capsys.readouterr().err


def test_opencode_max_tokens_under_ceiling_leaves_opencode_env_alone(fake_vllm, tmp_path):
    result = CliRunner().invoke(
        start.start_app, ["opencode", "--no-launch", "--max-tokens", "16000"]
    )
    assert result.exit_code == 0, result.output
    config = json.loads((tmp_path / "agents" / "opencode" / "opencode.json").read_text())
    limit = config["provider"][start._OPENCODE_PROVIDER]["models"][MODEL["id"]]["limit"]
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
    limit = config["provider"][start._OPENCODE_PROVIDER]["models"][MODEL["id"]]["limit"]
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
    start.write_opencode_config(BASE, "sk-test-abc", MODEL, path)
    config = json.loads(path.read_text())
    assert config["theme"] == "tokyonight"
    # The overlay no longer edits disabled_providers; re-enabling agent-switch is done in
    # the inline layer, so an existing list here is preserved untouched.
    assert config["disabled_providers"] == ["ollama", "agent-switch"]
    assert config["provider"]["anthropic"]["name"] == "Anthropic"
    assert config["provider"][start._OPENCODE_PROVIDER]["options"]["baseURL"] == f"{BASE}/v1"
    before = path.read_text()
    start.write_opencode_config(BASE, "sk-test-abc", MODEL, path)
    assert path.read_text() == before


def test_write_opencode_config_keeps_foreign_disabled_providers(tmp_path):
    # A user who disabled other providers (but not ours) must keep them disabled:
    # the overlay must not rewrite disabled_providers, or those providers get silently
    # re-enabled for the session.
    path = tmp_path / "opencode.json"
    path.write_text(json.dumps({"disabled_providers": ["openai", "gemini"]}))
    start.write_opencode_config(BASE, "sk-test-abc", MODEL, path)
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
    start.write_opencode_config(
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
    assert agent["model"] == f"{start._OPENCODE_PROVIDER}/{local['id']}"
    assert "local agent" in agent["description"].lower()
    assert local["id"] in config["provider"][start._OPENCODE_PROVIDER]["models"]


def test_opencode_subagent_inline_keeps_parent_provider_filters(monkeypatch, tmp_path):
    config_path = tmp_path / "opencode.json"
    inherited = {
        "theme": "tokyonight",
        "enabled_providers": ["anthropic"],
    }
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", json.dumps(inherited))
    monkeypatch.setattr(start, "_which_with_install_dirs", lambda _: "/usr/bin/opencode")
    monkeypatch.setattr(start, "_wsl_windows_executable", lambda _: None)
    captured = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(
            returncode = 0,
            stdout = json.dumps(
                {
                    "enabled_providers": ["opencode-go"],
                    "disabled_providers": ["ollama", start._OPENCODE_PROVIDER],
                    "subagent_depth": 0,
                }
            ),
            stderr = "",
        )

    monkeypatch.setattr(start.subprocess, "run", run)
    permission = {"edit": "allow"}
    inline = start._opencode_subagent_inline_config(config_path, permission)

    assert captured["command"] == ["/usr/bin/opencode", "debug", "config"]
    assert captured["env"]["OPENCODE_CONFIG"] == str(config_path)
    assert inline == {
        "theme": "tokyonight",
        "enabled_providers": [
            "anthropic",
            "opencode-go",
            start._OPENCODE_PROVIDER,
        ],
        "disabled_providers": ["ollama"],
        "subagent_depth": 1,
        "permission": permission,
    }


def test_opencode_subagent_inline_preserves_positive_depth(monkeypatch, tmp_path):
    monkeypatch.setattr(start, "_which_with_install_dirs", lambda _: "/usr/bin/opencode")
    monkeypatch.setattr(
        start.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode = 0,
            stdout = json.dumps({"subagent_depth": 3}),
            stderr = "",
        ),
    )

    inline = start._opencode_subagent_inline_config(tmp_path / "opencode.json", {})

    assert inline["subagent_depth"] == 3


def test_opencode_subagent_inline_merges_inherited_filters_without_binary(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "OPENCODE_CONFIG_CONTENT",
        json.dumps(
            {
                "enabled_providers": ["opencode-go"],
                "disabled_providers": ["ollama", start._OPENCODE_PROVIDER],
            }
        ),
    )
    monkeypatch.setattr(start, "_which_with_install_dirs", lambda _: None)

    inline = start._opencode_subagent_inline_config(tmp_path / "opencode.json", {})

    assert inline["enabled_providers"] == ["opencode-go", start._OPENCODE_PROVIDER]
    assert inline["disabled_providers"] == ["ollama"]
    assert inline["subagent_depth"] == 1


def test_opencode_v2_subagent_uses_native_depth_without_debug_probe(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "OPENCODE_CONFIG_CONTENT",
        json.dumps({"enabled_providers": ["anthropic"], "subagent_depth": 2}),
    )
    monkeypatch.setattr(
        start.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("V2 debug config is not a resolved object"),
    )

    inline = start._opencode_subagent_inline_config(
        tmp_path / "opencode.json", {}, command = "opencode2", v2 = True
    )

    assert "subagent_depth" not in inline
    assert inline["enabled_providers"] == ["anthropic", start._OPENCODE_PROVIDER]
    assert inline["experimental"] == {"subagent_depth": 2}


def test_opencode_v2_subagent_does_not_override_configured_depth(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENCODE_CONFIG_CONTENT", raising = False)

    inline = start._opencode_subagent_inline_config(
        tmp_path / "opencode.json", {}, command = "opencode2", v2 = True
    )

    assert "experimental" not in inline


def _opencode_inline_config(output: str) -> dict:
    # --no-launch prints OPENCODE_CONFIG_CONTENT as a POSIX `export NAME=<shell-quoted>`
    # line on Unix/WSL and a PowerShell `$env:NAME = "<escaped>"` line on native Windows;
    # parse whichever the host emitted so the opencode tests are shell-agnostic.
    name = "OPENCODE_CONFIG_CONTENT"
    for raw in output.splitlines():
        line = raw.strip()
        if line.startswith(f"export {name}="):
            return json.loads(shlex.split(line.removeprefix(f"export {name}="))[0])
        prefix = f'$env:{name} = "'
        if line.startswith(prefix) and line.endswith('"'):
            escaped = line[len(prefix) : -1]
            # Reverse _print_env's PowerShell escaping (backtick is the escape char).
            value = escaped.replace("`$", "$").replace('`"', '"').replace("``", "`")
            return json.loads(value)
    raise AssertionError(f"{name} not found in:\n{output}")


def test_opencode_inline_scopes_session_to_our_provider(fake_vllm):
    # opencode filters even config-defined providers through enabled/disabled_providers,
    # and a model pin does not bypass that gate. The inline overlay (session-only, highest
    # layer, arrays replace) allowlists our provider and clears the denylist so the local
    # model always loads regardless of the user's config, without reading or editing it.
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])
    assert result.exit_code == 0, result.output
    inline = _opencode_inline_config(result.output)
    assert inline["enabled_providers"] == [start._OPENCODE_PROVIDER]
    assert inline["disabled_providers"] == []
    assert inline["model"] == f"{start._OPENCODE_PROVIDER}/{MODEL['id']}"
    # small_model stays on the enabled provider too, so lightweight tasks do not resolve a
    # filtered provider mid-session.
    assert inline["small_model"] == f"{start._OPENCODE_PROVIDER}/{MODEL['id']}"


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
        == f"{start._OPENCODE_PROVIDER}/{MODEL['id']}"
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
    provider = config["provider"][start._OPENCODE_PROVIDER]
    assert provider["options"]["apiKey"] == KEY
    assert config["model"] == f"{start._OPENCODE_PROVIDER}/{MODEL['id']}"
    # The session config file (a throwaway overlay, not the user's real config) does not
    # carry provider filters; the session scoping rides in the inline env layer only.
    assert "disabled_providers" not in config
    assert "enabled_providers" not in config
    assert inline_config == {
        "model": f"{start._OPENCODE_PROVIDER}/{MODEL['id']}",
        "small_model": f"{start._OPENCODE_PROVIDER}/{MODEL['id']}",
        "enabled_providers": [start._OPENCODE_PROVIDER],
        "disabled_providers": [],
    }
    # --no-launch prints an append-safe base command (no --model before a subcommand a
    # driver may append); the model is forced by the inline pin above.
    assert _launch_command(result.output) == ["opencode"]


def test_connect_opencode_v2_no_launch_uses_private_server(fake_vllm, monkeypatch):
    monkeypatch.setattr(start, "_opencode_command", lambda *_: ("opencode2", True))

    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])

    assert result.exit_code == 0, result.output
    assert _launch_command(result.output) == ["opencode2", "--standalone"]
    assert _opencode_inline_config(result.output)["model"] == (
        f"{start._OPENCODE_PROVIDER}/{MODEL['id']}"
    )
    assert "enabled_providers" not in _opencode_inline_config(result.output)
    assert "disabled_providers" not in _opencode_inline_config(result.output)
    assert f"provider policies must allow '{start._OPENCODE_PROVIDER}'" in result.output


def test_connect_opencode_v2_no_launch_uses_resolved_off_path_binary(
    fake_vllm, monkeypatch, tmp_path
):
    binary = tmp_path / ".opencode" / "bin" / "opencode2"
    monkeypatch.setattr(
        start,
        "_opencode_command",
        lambda: (str(binary), True),
    )

    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch"])

    assert result.exit_code == 0, result.output
    assert _launch_command(result.output) == [str(binary), "--standalone"]


def test_connect_opencode_v2_models_uses_private_server(fake_vllm, monkeypatch):
    monkeypatch.setattr(start, "_opencode_command", lambda *_: ("opencode2", True))

    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch", "models"])

    assert result.exit_code == 0, result.output
    assert _launch_command(result.output) == ["opencode2", "models", "--standalone"]


def test_connect_opencode_as_subagent_preserves_cloud_parent(fake_vllm, tmp_path, monkeypatch):
    monkeypatch.setattr(
        start, "_opencode_subagent_inline_config", lambda path, permission, **kwargs: {}
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
    expected_model = f"{start._OPENCODE_PROVIDER}/{MODEL['id']}"
    # The agent rides in the inline overlay; nothing else comes from the empty base.
    assert _opencode_inline_config(result.output) == {
        "agent": {
            "local": {
                "description": start._SUBAGENT_DESCRIPTION,
                "mode": "subagent",
                "model": expected_model,
                "prompt": start._SUBAGENT_INSTRUCTIONS,
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


def test_claude_subagent_allowed_tools_precede_forwarded_delimiter(fake_vllm):
    # A forwarded `--` makes everything after it positional; the tool pre-approval
    # must be parsed as an option, so it rides before ctx.args.
    result = CliRunner().invoke(
        start.start_app,
        ["claude", "--as-subagent", "--no-launch", "--", "--resume", "abc123"],
    )
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    allowed = next(arg for arg in command if arg.startswith("--allowedTools="))
    assert command.index(allowed) < command.index("--resume")


def test_claude_subagent_forwards_positional_prompt(fake_vllm):
    # --allowedTools is variadic: a detached value would consume the prompt.
    result = CliRunner().invoke(
        start.start_app,
        ["claude", "--as-subagent", "--no-launch", "fix the failing test"],
    )
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    assert command[-1] == "fix the failing test"
    assert "--allowedTools" not in command


def test_opencode_subagent_installs_binary_before_filter_inspection(fake_vllm, monkeypatch):
    # The effective-config inspection needs the opencode binary; a first launch must
    # offer the install before building the overlay, or a global allowlist read only
    # after _launch installs OpenCode would filter out the new provider.
    installed = {}
    monkeypatch.setattr(
        start,
        "_which_with_install_dirs",
        lambda name: "/usr/local/bin/opencode" if installed.get("done") else None,
    )

    def require(name, hint, launch):
        assert launch is True
        installed["done"] = True
        installed["name"] = name

    monkeypatch.setattr(start, "_require_agent_for_launch", require)
    inspected = {}

    def inline(path, permission, **kwargs):
        inspected["binary"] = start._which_with_install_dirs("opencode")
        return {}

    monkeypatch.setattr(start, "_opencode_subagent_inline_config", inline)
    monkeypatch.setattr(start, "_run", lambda *a, **k: None)

    result = CliRunner().invoke(start.start_app, ["opencode", "--as-subagent"])

    assert result.exit_code == 0, result.output
    assert installed["name"] == "opencode"
    assert inspected["binary"] == "/usr/local/bin/opencode"


def test_opencode_subagent_pins_agent_in_inline_overlay(fake_vllm, monkeypatch):
    # A project opencode.json outranks the session file, so the agent must ride in
    # OPENCODE_CONFIG_CONTENT where a repo's own agent.local cannot field-merge over it.
    monkeypatch.setattr(
        start, "_opencode_subagent_inline_config", lambda path, permission, **kwargs: {}
    )
    result = CliRunner().invoke(
        start.start_app,
        ["opencode", "--as-subagent", "--no-launch", "--model", MODEL["id"]],
    )
    assert result.exit_code == 0, result.output
    agent = _opencode_inline_config(result.output)["agent"]["local"]
    assert agent["mode"] == "subagent"
    assert agent["model"] == f"{start._OPENCODE_PROVIDER}/{MODEL['id']}"
    assert agent["prompt"] == start._SUBAGENT_INSTRUCTIONS
    assert agent["description"] == start._SUBAGENT_DESCRIPTION


def test_connect_opencode_subagent_yolo_no_launch_stays_append_safe(fake_vllm, monkeypatch):
    monkeypatch.setattr(start, "_opencode_supports_native_auto", lambda *_: True)
    captured = {}

    def inline(path, permission, **kwargs):
        captured["permission"] = permission
        return {"permission": permission}

    monkeypatch.setattr(start, "_opencode_subagent_inline_config", inline)
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


# ── Hermes (OpenAI /v1/chat/completions, key via env) ────────────────


# ── Pi (OpenAI-compatible /v1, key in config, ~/.pi relocated via HOME) ──


def test_write_pi_config_fresh(tmp_path):
    path = tmp_path / ".pi" / "agent" / "models.json"
    start.write_pi_config(BASE, "sk-test-abc", MODEL, path)
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
    start.write_pi_config(BASE, "sk-test-abc", MODEL, path)
    config = json.loads(path.read_text())
    assert config["providers"]["google"] == {"api": "gemini"}  # unrelated provider kept
    assert config["providers"]["agent-switch"]["baseUrl"] == f"{BASE}/v1"
    before = path.read_text()
    start.write_pi_config(BASE, "sk-test-abc", MODEL, path)
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
    result = CliRunner().invoke(start.start_app, ["pi", "--no-launch"])
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
    start.write_pi_config(BASE, "sk-test-abc", MODEL, agent_dir / "models.json")

    start.write_pi_user_resources(agent_dir, session_home)

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

    start.write_pi_user_resources(agent_dir, session_home)
    # Add session-only state, then change the user config.
    settings_path = agent_dir / "settings.json"
    settings = json.loads(settings_path.read_text())
    settings["packages"].append("npm:session-only")
    settings["theme"] = "dark"
    settings_path.write_text(json.dumps(settings))
    user_settings.write_text(json.dumps({"packages": ["npm:kept", "npm:new"]}))
    (user_agent_dir / "extensions").rmdir()

    start.write_pi_user_resources(agent_dir, session_home)

    # Pi keeps the first entry per package identity, so session-owned packages lead.
    assert json.loads(settings_path.read_text()) == {
        "packages": ["npm:session-only", "npm:kept", "npm:new"],
        "theme": "dark",
    }
    assert not (agent_dir / "extensions").exists() and not (agent_dir / "extensions").is_symlink()

    user_settings.unlink()
    start.write_pi_user_resources(agent_dir, session_home)
    assert json.loads(settings_path.read_text()) == {
        "packages": ["npm:session-only"],
        "theme": "dark",
    }
    assert not (agent_dir / start._PI_USER_RESOURCES_MANIFEST).exists()


def test_write_pi_user_resources_leaves_session_dirs_and_user_files_alone(tmp_path, monkeypatch):
    user_agent_dir = _pi_user_agent_dir(tmp_path, monkeypatch)
    (user_agent_dir / "npm" / "node_modules" / "pi-mine").mkdir(parents = True)
    (user_agent_dir / "extensions").mkdir()
    (user_agent_dir / "extensions" / "mine.ts").write_text("mine\n")
    session_home = tmp_path / "session"
    agent_dir = session_home / ".pi" / "agent"
    (agent_dir / "npm").mkdir(parents = True)
    (agent_dir / "npm" / "session.txt").write_text("session\n")

    start.write_pi_user_resources(agent_dir, session_home)

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

    start.write_pi_user_resources(agent_dir, session_home)
    assert (agent_dir / "extensions").resolve() == (configured / "extensions").resolve()

    # Do not reuse the session as its own source.
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent_dir))
    start.write_pi_user_resources(agent_dir, session_home)
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

    start.write_pi_user_resources(agent_dir, session_home)

    assert (agent_dir / "extensions").resolve() == (configured / "extensions").resolve()
    settings = json.loads((agent_dir / "settings.json").read_text())
    assert settings == {"packages": [str(tmp_path / "launch" / "src" / "local-ext")]}


def test_is_junction_reads_the_reparse_tag_before_python_3_12(tmp_path, monkeypatch):
    # Python 3.13 also defines is_junction on a pathlib base class.
    for cls in type(tmp_path).__mro__:
        if "is_junction" in vars(cls):
            monkeypatch.delattr(cls, "is_junction")
    tags = {"junction": 0xA0000003, "symlink": 0xA000000C}

    def lstat(path):
        name = Path(path).name
        if name not in tags:
            raise FileNotFoundError(path)
        return SimpleNamespace(st_reparse_tag = tags[name])

    monkeypatch.setattr(start.os, "lstat", lstat)
    assert start._is_junction(tmp_path / "junction")
    assert not start._is_junction(tmp_path / "symlink")
    assert not start._is_junction(tmp_path / "missing")


def test_link_user_dir_replaces_a_junction_from_an_earlier_run(tmp_path, monkeypatch):
    target = tmp_path / "session" / "extensions"
    target.mkdir(parents = True)  # stands in for a junction to a previous source
    source = tmp_path / "user" / "extensions"
    source.mkdir(parents = True)
    monkeypatch.setattr(start, "_is_junction", lambda path: path == target and path.is_dir())

    start._link_user_dir(source, target)

    assert target.is_symlink()
    assert target.resolve() == source.resolve()


def test_write_pi_user_resources_skips_a_windows_pi_under_wsl(tmp_path, monkeypatch):
    user_agent_dir = _pi_user_agent_dir(tmp_path, monkeypatch)
    (user_agent_dir / "extensions").mkdir()
    (user_agent_dir / "settings.json").write_text(json.dumps({"packages": ["npm:pi-mine"]}))
    monkeypatch.setattr(start, "_wsl_windows_executable", lambda _: "/mnt/c/npm/pi")
    session_home = tmp_path / "session"
    agent_dir = session_home / ".pi" / "agent"

    start.write_pi_user_resources(agent_dir, session_home)

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
    monkeypatch.setattr(start, "_wsl_windows_executable", lambda _: None)
    start.write_pi_user_resources(agent_dir, session_home)
    assert (agent_dir / "extensions").is_symlink()
    # Something the session set for itself, which must survive.
    settings = json.loads((agent_dir / "settings.json").read_text())
    settings["packages"].append("npm:session-only")
    (agent_dir / "settings.json").write_text(json.dumps(settings))

    monkeypatch.setattr(start, "_wsl_windows_executable", lambda _: "/mnt/c/npm/pi.cmd")
    start.write_pi_user_resources(agent_dir, session_home)

    assert not (agent_dir / "extensions").exists()
    assert json.loads((agent_dir / "settings.json").read_text()) == {
        "packages": ["npm:session-only"],
    }
    assert not (agent_dir / start._PI_USER_RESOURCES_MANIFEST).exists()
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
    monkeypatch.setattr(start, "_wsl_windows_executable", lambda _: None)
    start.write_pi_user_resources(agent_dir, session_home)
    assert json.loads((agent_dir / "settings.json").read_text())["npmCommand"] == [
        "npm",
        "--registry=x",
    ]
    if session_command is not None:
        settings = json.loads((agent_dir / "settings.json").read_text())
        settings["npmCommand"] = session_command
        (agent_dir / "settings.json").write_text(json.dumps(settings))

    monkeypatch.setattr(start, "_wsl_windows_executable", lambda _: "/mnt/c/npm/pi.cmd")
    start.write_pi_user_resources(agent_dir, session_home)

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

    start.write_pi_user_resources(agent_dir, session_home)

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

    start.write_pi_user_resources(agent_dir, session_home)

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

    start.write_pi_user_resources(agent_dir, session_home)

    assert json.loads((agent_dir / "settings.json").read_text())["npmCommand"] == command
    # A command set inside the session is not overwritten on the next launch.
    settings_path = agent_dir / "settings.json"
    settings = json.loads(settings_path.read_text())
    settings["npmCommand"] = ["pnpm"]
    settings_path.write_text(json.dumps(settings))
    start.write_pi_user_resources(agent_dir, session_home)
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

    start.write_pi_user_resources(agent_dir, session_home)

    assert "PI_CODING_AGENT_DIR" in capsys.readouterr().err


def test_write_pi_user_resources_keeps_a_non_list_setting(tmp_path, monkeypatch):
    user_agent_dir = _pi_user_agent_dir(tmp_path, monkeypatch)
    (user_agent_dir / "settings.json").write_text(json.dumps({"themes": ["npm:user-theme"]}))
    session_home = tmp_path / "session"
    agent_dir = session_home / ".pi" / "agent"
    agent_dir.mkdir(parents = True)
    (agent_dir / "settings.json").write_text(json.dumps({"themes": {"name": "dark"}}))

    start.write_pi_user_resources(agent_dir, session_home)

    # Not a shape we understand, so it is left alone rather than deleted.
    assert json.loads((agent_dir / "settings.json").read_text())["themes"] == {"name": "dark"}


@pytest.mark.parametrize("entry", ["", "   ", "."])
def test_pi_local_entry_leaves_degenerate_entries_alone(tmp_path, entry):
    # Anchoring these would name the user's whole agent directory.
    assert start._pi_local_entry(entry, tmp_path, tmp_path, frozenset()) == entry


@pytest.mark.parametrize("dangling", [False, True])
def test_remove_overlay_entry_rmdirs_a_windows_directory_symlink(tmp_path, monkeypatch, dangling):
    # unlink maps to DeleteFileW, which refuses a directory symlink with WinError 5,
    # and a dangling link is still one: is_dir() would follow the missing target.
    source = tmp_path / "source"
    source.mkdir()
    (source / "keep.txt").write_text("keep\n")
    target = tmp_path / "link"
    target.symlink_to(source, target_is_directory = True)
    if dangling:
        shutil.rmtree(source)
        source.mkdir()  # restore the sentinel dir so the assertion below still reads
        (source / "keep.txt").write_text("keep\n")
        target.unlink()
        target.symlink_to(tmp_path / "gone", target_is_directory = True)
    # POSIX lstat has no st_file_attributes; stand in for the Windows link attributes.
    real_lstat = os.lstat
    monkeypatch.setattr(
        start.os,
        "lstat",
        lambda p: SimpleNamespace(
            st_file_attributes = 0x10,
            st_reparse_tag = 0,
            st_mode = real_lstat(p).st_mode,
        ),
    )
    # rmdir on a link is POSIX-invalid, so record the routing instead of running it.
    monkeypatch.setattr(start.os, "name", "nt")
    calls = []
    monkeypatch.setattr(start.Path, "unlink", lambda self, **kw: calls.append("unlink"))
    monkeypatch.setattr(start.Path, "rmdir", lambda self: calls.append("rmdir"))

    start._remove_overlay_entry(target)

    assert calls == ["rmdir"]
    assert (source / "keep.txt").read_text() == "keep\n"


@pytest.mark.parametrize("yolo", [False, True])
def test_connect_pi_as_subagent_preserves_cloud_parent(fake_vllm, tmp_path, yolo):
    args = [
        "pi",
        "--as-subagent",
        "--no-launch",
        "--model",
        MODEL["id"],
    ]
    if yolo:
        args.insert(2, "--yolo")
    result = CliRunner().invoke(
        start.start_app,
        args,
    )
    assert result.exit_code == 0, result.output
    command = _launch_command(result.output)
    assert command[:2] == ["pi", "--extension"]
    assert command[2].endswith("agent_switch/pi_subagent.ts")
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
    args = ["pi", "--no-launch", "--max-tokens", "40000"]
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
    args = ["pi", "--no-launch", "--max-tokens", "200000"]
    if as_subagent:
        args.append("--as-subagent")
    result = CliRunner().invoke(start.start_app, args)
    assert result.exit_code == 0, result.output
    assert "leaves too little" in result.output
    assert _pi_generated_model(tmp_path, as_subagent)["maxTokens"] == MODEL["context_length"] // 2


@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_connect_pi_invalid_output_limit(fake_vllm, tmp_path, value):
    result = CliRunner().invoke(start.start_app, ["pi", "--no-launch", "--max-tokens", value])
    assert result.exit_code != 0
    assert "--max-tokens" in result.output
    assert not list((tmp_path / "agents").rglob("models.json"))


@pytest.mark.parametrize("context", [{}, {"max_context_length": 32768}])
def test_write_pi_output_limit_context_metadata(tmp_path, context):
    path = tmp_path / "models.json"
    start.write_pi_config(BASE, "test-key", {"id": "test-model", **context}, path, max_tokens = 10000)
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
    monkeypatch.setattr(start.Path, "home", lambda: user_home)
    _simulate_windows(monkeypatch)
    result = CliRunner().invoke(start.start_app, ["pi", "--no-launch"])
    assert result.exit_code == 0, result.output
    home = tmp_path / "agents" / "pi"
    assert f'$env:HOME = "{home}"' in result.output
    assert f'$env:USERPROFILE = "{home}"' in result.output


# ── DeepSeek Harness (OpenAI /v1, key via env, ~/.dsh relocated) ─────


@pytest.fixture()
def dsh_patch(tmp_path):
    return tmp_path / "agent-switch.patch.yml"


def _dsh_entries(path):
    yaml = pytest.importorskip("yaml")
    entries = yaml.safe_load(path.read_text())
    # A loader patch is a top-level list of id-targeted entries, not a settings mapping.
    assert isinstance(entries, list), entries
    return {entry["id"]: entry for entry in entries}


def test_write_dsh_patch_fresh(dsh_patch):
    start.write_dsh_patch(BASE, MODEL, dsh_patch)
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


@pytest.mark.parametrize("window, expected", [(32_768, 8_192), (143_616, 32_000)])
def test_pi_and_dsh_output_limit_follows_the_context(tmp_path, window, expected):
    model = {**MODEL, "context_length": window}
    start.write_pi_config(BASE, "sk-test-abc", model, tmp_path / "models.json")
    start.write_pi_subagent_config(BASE, "sk-test-abc", model, tmp_path / "subagent.json")
    start.write_dsh_patch(BASE, model, tmp_path / "agent-switch.patch.yml")
    pi = json.loads((tmp_path / "models.json").read_text())["providers"]["agent-switch"]["models"][0]
    subagent = json.loads((tmp_path / "subagent.json").read_text())
    patch = _dsh_entries(tmp_path / "agent-switch.patch.yml")
    dsh = patch["llm-pi-ai"]["config"]["providers"]["agent-switch"]["models"][0]
    assert pi["maxTokens"] == subagent["maxTokens"] == dsh["maxTokens"] == expected


def test_write_dsh_patch_without_window_omits_limits(dsh_patch):
    start.write_dsh_patch(BASE, {"id": "org/unknown-window"}, dsh_patch)
    provider = _dsh_entries(dsh_patch)["llm-pi-ai"]["config"]["providers"]["agent-switch"]
    assert provider["models"] == [{"id": "org/unknown-window"}]


def test_write_dsh_patch_is_idempotent_and_follows_the_server(dsh_patch, capsys):
    start.write_dsh_patch(BASE, MODEL, dsh_patch)
    before = dsh_patch.read_text()
    capsys.readouterr()
    start.write_dsh_patch(BASE, MODEL, dsh_patch)
    assert dsh_patch.read_text() == before
    assert "Updated" not in capsys.readouterr().out
    # agent-switch owns this file: a new server or model replaces the old one, it does not pile up.
    start.write_dsh_patch("http://127.0.0.1:9999", {"id": "other"}, dsh_patch)
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
    assert start._dsh_command(args) == expected


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
    assert start._dsh_command(args, "P") == expected


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
    monkeypatch.setattr(start.shutil, "which", lambda _: shim)
    monkeypatch.setattr(start, "is_deepseek_harness_executable", lambda _: True)
    monkeypatch.setattr(start.subprocess, "check_output", lambda *args, **kwargs: windows_path)
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
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/dsh")
    monkeypatch.setattr(start, "is_deepseek_harness_executable", lambda _: True)
    captured = _capture_launch(monkeypatch, argv)
    assert captured["env"]["DSH_PERMISSION_MODE"] == expected


def test_start_dsh_forwards_reasoning_effort(fake_vllm, monkeypatch):
    # --reasoning-effort is a shared option: it must reach the dsh config rather than
    # pass through to `dsh web`, which does not accept it.
    yaml = pytest.importorskip("yaml")
    captured = {}
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/dsh")
    monkeypatch.setattr(start, "is_deepseek_harness_executable", lambda _: True)

    def run(
        command,
        env = None,
        **kwargs,
    ):
        captured["command"] = command
        captured["patch"] = Path(command[command.index("--patch") + 1]).read_text()
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(start.subprocess, "run", run)

    result = CliRunner().invoke(
        start.start_app,
        ["dsh", "--model", MODEL["id"], "--reasoning-effort", "high"],
    )
    assert result.exit_code == 0, result.output
    assert "--reasoning-effort" not in captured["command"]
    provider = yaml.safe_load(captured["patch"])[0]["config"]["providers"][start._DSH_PROVIDER]
    assert provider["compat"]["chatTemplateKwargs"] == {"reasoning_effort": "high"}


# ── WSLENV path translation + PowerShell quoting (helper units) ──


def test_wsl_bridge_names_flags_paths_not_scalars():
    # WSLENV only translates a var to a Windows path when its entry carries /p.
    # Path-valued vars must get it; scalar knobs and URLs must not, or WSLENV would
    # mangle them when handing off to a Windows shim under /mnt.
    env = {
        "CODEX_HOME": "/tmp/sess/codex",
        "HOME": "/tmp/sess/pi",
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "4096",
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:8888",
        "USERPROFILE": r"C:\Users\x",
    }
    names = start._wsl_bridge_names(env, ("ANTHROPIC_API_KEY",))
    assert "CODEX_HOME/p" in names
    assert "HOME/p" in names
    assert "USERPROFILE/p" in names  # drive-qualified Windows path
    assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW" in names  # scalar: no /p
    assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW/p" not in names
    assert "ANTHROPIC_BASE_URL" in names  # URL is not a filesystem path
    assert "ANTHROPIC_API_KEY" in names  # cleared var carries no value to translate


def test_merge_wslenv_dedups_on_base_name():
    # An already-shared var must not be appended again just because the flag differs.
    merged = start._merge_wslenv("CODEX_HOME/p:FOO", ("CODEX_HOME/p", "BAR/p"))
    parts = merged.split(":")
    assert parts.count("CODEX_HOME/p") == 1
    assert "FOO" in parts and "BAR/p" in parts


def test_merge_wslenv_upgrades_existing_unflagged_entry():
    # A user's pre-existing bare "HOME" must be upgraded to "HOME/p" (not left bare or
    # duplicated), or the Windows shim gets the path without WSL translation.
    merged = start._merge_wslenv("HOME:FOO", ("HOME/p", "CODEX_HOME/p"))
    parts = merged.split(":")
    assert "HOME/p" in parts and "HOME" not in parts  # upgraded in place
    assert parts.count("HOME/p") == 1
    assert "FOO" in parts  # untouched user var preserved
    assert "CODEX_HOME/p" in parts


def test_powershell_quote_single_quotes_json():
    # Bare flags/paths pass through; JSON payloads get single-quoted so PowerShell
    # keeps the embedded double quotes literal (list2cmdline's backslashes would not).
    assert start._powershell_quote("--settings") == "--settings"
    assert start._powershell_quote("org/gemma-4-26B") == "org/gemma-4-26B"
    overlay = start._claude_settings_overlay("org/gemma-4-26B")
    quoted = start._powershell_quote(overlay)
    assert quoted == "'" + overlay + "'"
    assert "\\" not in quoted  # no cmd.exe backslash escaping
    assert start._powershell_quote("a'b") == "'a''b'"  # embedded quote doubled


# ── --yolo: one switch routed to each agent's own auto-approve form ──

# The native "run tools without prompting" CLI flag each agent should receive.
_NATIVE_YOLO = {
    "claude": "--dangerously-skip-permissions",
    "codex": "--dangerously-bypass-approvals-and-sandbox",
    "pi": "--approve",
}


@pytest.mark.parametrize("agent, native", sorted(_NATIVE_YOLO.items()))
def test_yolo_routes_to_native_flag(fake_vllm, agent, native):
    result = CliRunner().invoke(start.start_app, [agent, "--yolo", "--no-launch"])
    assert result.exit_code == 0, result.output
    assert native in result.output


@pytest.mark.parametrize("agent, native", sorted(_NATIVE_YOLO.items()))
def test_no_yolo_omits_native_flag(fake_vllm, agent, native):
    result = CliRunner().invoke(start.start_app, [agent, "--no-launch"])
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
    claude = CliRunner().invoke(start.start_app, ["claude", alias, "--no-launch"])
    assert claude.exit_code == 0, claude.output
    assert "--dangerously-skip-permissions" in claude.output
    # The codex spelling must not leak through to Claude's command line.
    assert "--dangerously-bypass-approvals-and-sandbox" not in claude.output

    codex = CliRunner().invoke(start.start_app, ["codex", alias, "--no-launch"])
    assert codex.exit_code == 0, codex.output
    assert "--dangerously-bypass-approvals-and-sandbox" in codex.output
    assert "--dangerously-skip-permissions" not in codex.output

    opencode = CliRunner().invoke(
        start.start_app,
        ["opencode", alias, "--no-launch", "run", "hello"],
    )
    assert opencode.exit_code == 0, opencode.output
    assert _launch_command(opencode.output) == ["opencode", "run", "hello", "--auto"]
    assert "permission" not in _opencode_inline_config(opencode.output)


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
    monkeypatch.setattr(start, "_opencode_command", lambda *_: ("opencode2", True))

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
    monkeypatch.setattr(start, "_opencode_command", lambda *_: ("opencode2", True))

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
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/opencode")
    monkeypatch.setattr(start, "_opencode_supports_native_auto", lambda *_: True)
    captured = _capture_launch(monkeypatch, ["opencode", "--yolo"])
    assert captured["command"][1:] == [
        "--model",
        f"{start._OPENCODE_PROVIDER}/{MODEL['id']}",
        "--auto",
    ]
    assert "permission" not in json.loads(captured["env"]["OPENCODE_CONFIG_CONTENT"])


def test_yolo_opencode_v2_bare_launch_omits_root_model(fake_vllm, monkeypatch):
    monkeypatch.setattr(start, "_opencode_command", lambda *_: ("opencode2", True))
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/opencode2")
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
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/opencode")
    monkeypatch.setattr(start.subprocess, "check_output", lambda *args, **kwargs: version)
    assert start._opencode_supports_native_auto() is expected


def test_opencode_native_auto_assumes_current_without_local_binary(monkeypatch):
    monkeypatch.setattr(start.shutil, "which", lambda _: None)
    assert start._opencode_supports_native_auto() is True


def test_yolo_opencode_old_version_uses_config_fallback(fake_vllm, monkeypatch):
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/opencode")
    monkeypatch.setattr(start.subprocess, "check_output", lambda *args, **kwargs: "1.17.11")
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
    assert start._opencode_native_auto_args(args, True) == (expected, native)
    assert start._opencode_native_auto_args(args, False) == (args, False)


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
    start.write_opencode_config(BASE, "sk-test-abc", MODEL, path, yolo = True)
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
    assert start._OPENCODE_PROVIDER in config["provider"]


def test_write_opencode_config_yolo_then_plain_unit(tmp_path):
    path = tmp_path / "opencode.json"
    start.write_opencode_config(BASE, "sk-test-abc", MODEL, path, yolo = True)
    start.write_opencode_config(BASE, "sk-test-abc", MODEL, path, yolo = False)
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
    session = start.write_opencode_config(BASE, "sk-test-abc", MODEL, path, yolo = False)
    config = json.loads(path.read_text())
    assert config["permission"] == {"edit": "ask", "bash": "deny", "read": "ask"}
    assert session == {}  # a non-yolo session carries no permission inline


def test_opencode_subagent_non_yolo_clears_yolo_task_permission(tmp_path):
    path = tmp_path / "opencode.json"
    start.write_opencode_config(
        BASE,
        "sk-test-abc",
        MODEL,
        path,
        yolo = True,
        as_subagent = True,
    )
    start.write_opencode_config(
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
    session = start.write_opencode_config(BASE, "sk-test-abc", MODEL, path, yolo = False)
    assert json.loads(path.read_text())["permission"] == "deny"
    assert session == {}


def test_opencode_non_yolo_leaves_catch_all_and_flips_explicit_allow(tmp_path):
    # A "*" catch-all is the user's own rule, never something --yolo writes (yolo sets
    # explicit per-tool allow), so it is left intact; an explicit per-tool "allow" is still
    # flipped to "ask", but an absent tool inheriting the catch-all is not touched.
    path = tmp_path / "opencode.json"
    path.write_text(json.dumps({"permission": {"*": "allow", "bash": "allow"}}))
    session = start.write_opencode_config(BASE, "sk-test-abc", MODEL, path, yolo = False)
    assert json.loads(path.read_text())["permission"] == {"*": "allow", "bash": "ask"}
    assert session == {}


def test_opencode_non_yolo_leaves_granular_object(tmp_path):
    # A granular object value is a user rule (yolo only ever writes a plain "allow" string),
    # so it is left in the file verbatim and never carried inline.
    path = tmp_path / "opencode.json"
    obj = {"read *": "deny", "git *": "ask"}
    path.write_text(json.dumps({"permission": {"bash": dict(obj)}}))
    session = start.write_opencode_config(BASE, "sk-test-abc", MODEL, path, yolo = False)
    assert json.loads(path.read_text())["permission"]["bash"] == obj
    assert session == {}


def test_yolo_command_flags_unmapped_agent_is_empty():
    # Placement-aware/config-based agents (and any typo) must yield no prefix flag.
    assert start._yolo_command_flags("opencode", True) == []
    assert start._yolo_command_flags("claude", True) == ["--dangerously-skip-permissions"]
    assert start._yolo_command_flags("claude", False) == []


def test_yolo_config_fallbacks_add_no_legacy_command_flag(fake_vllm):
    # OpenCode's append-safe bare recipe uses its config fallback, so it must not leak a legacy
    # yolo/dangerous alias onto argv.
    for agent in ("opencode",):
        result = CliRunner().invoke(start.start_app, [agent, "--yolo", "--no-launch"])
        assert result.exit_code == 0, result.output
        command = _launch_command(result.output)
        assert command and command[0] == agent, result.output
        assert not any("--yolo" in arg or "--dangerous" in arg for arg in command)


def test_pi_launch_clears_screen_first(fake_vllm, monkeypatch):
    # Pi paints inline from the current cursor position (no alternate screen, no
    # clear on its first render), so the launcher hands it a clean screen. The
    # clear must come BEFORE the exec, and only on the launch path.
    calls = []
    monkeypatch.setattr(start.click, "clear", lambda: calls.append("clear"))
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/pi")

    def run(command, env):
        calls.append("exec")
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(start.subprocess, "run", run)
    result = CliRunner().invoke(start.start_app, ["pi"])
    assert result.exit_code == 0, result.output
    assert calls == ["clear", "exec"]


def test_pi_no_launch_does_not_clear(fake_vllm, monkeypatch):
    # The --no-launch recipe is meant to be read (and piped); never wipe it.
    calls = []
    monkeypatch.setattr(start.click, "clear", lambda: calls.append("clear"))
    result = CliRunner().invoke(start.start_app, ["pi", "--no-launch"])
    assert result.exit_code == 0, result.output
    assert calls == []


def test_claude_launch_does_not_clear(fake_vllm, monkeypatch):
    # Alternate-screen agents manage the terminal themselves; leave it alone.
    calls = []
    monkeypatch.setattr(start.click, "clear", lambda: calls.append("clear"))
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/claude")
    monkeypatch.setattr(start, "_claude_flags", lambda *a, **k: [])
    monkeypatch.setattr(start.subprocess, "run", lambda command, env: SimpleNamespace(returncode = 0))
    result = CliRunner().invoke(start.start_app, ["claude"])
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
    monkeypatch.setattr(start.shutil, "which", lambda _: "/mnt/c/Users/x/AppData/Roaming/npm/pi")

    def run(command, env):
        captured["env"] = env
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(start.subprocess, "run", run)
    result = CliRunner().invoke(start.start_app, ["pi"])
    assert result.exit_code == 0, result.output
    home = captured["env"]["HOME"]
    # A Windows pi shim resolves ~/.pi via USERPROFILE, so it must match the session
    # HOME and ride the WSLENV bridge (with /p) so the path is translated for Windows.
    assert captured["env"]["USERPROFILE"] == home
    wslenv = captured["env"]["WSLENV"].split(":")
    assert "HOME/p" in wslenv
    assert "USERPROFILE/p" in wslenv


def test_session_config_no_launch_preserves_existing_state(fake_vllm, tmp_path):
    # A previously printed recipe may still be running an agent whose sessions
    # or sqlite state live in the stable home; a re-run must not wipe it.
    with start._session_config("codex", launch = False) as home:
        marker = home / "sessions" / "live.sqlite"
        marker.parent.mkdir(parents = True)
        marker.write_text("state")
    with start._session_config("codex", launch = False) as home2:
        assert home2 == home
        assert (home2 / "sessions" / "live.sqlite").read_text() == "state"


# ── --persist: persist the agent session so it can be resumed ────────────────
def test_session_config_persist_uses_stable_dir_and_survives(monkeypatch, tmp_path):
    # --persist routes a launch to the stable agent-switch agents dir (the one --no-launch
    # already uses) instead of a throwaway temp dir, and never wipes it on exit.
    monkeypatch.setattr(start, "_agents_config_root", lambda: tmp_path / "agents")
    with start._session_config("codex", launch = True, persist = True) as home:
        assert home == tmp_path / "agents" / "codex"
        (home / "marker").write_text("kept")
    assert home.exists()
    assert (home / "marker").read_text() == "kept"


def test_session_config_default_launch_is_ephemeral(monkeypatch, tmp_path):
    agents_root = tmp_path / "agents"
    monkeypatch.setattr(start, "_agents_config_root", lambda: agents_root)
    with start._session_config("codex", launch = True) as home:
        assert home.exists()
        parent = start._ephemeral_session_parent("codex")
        assert home.name.startswith(start._ephemeral_session_prefix("codex", parent))
        if parent is None:
            assert home.parent == agents_root / ".tmp"
    assert not home.exists()


def test_session_config_codex_uses_short_ephemeral_parent(monkeypatch, tmp_path):
    # Windows Codex checks out its curated plugins under CODEX_HOME/.tmp/plugins.
    # Put its throwaway home outside the longer system temp path so that checkout
    # stays below legacy MAX_PATH and Codex does not reject temp-dir PATH helpers.
    short_parent = tmp_path / "u"
    short_parent.mkdir()
    monkeypatch.setattr(
        start,
        "_ephemeral_session_parent",
        lambda agent: short_parent if agent == "codex" else None,
    )

    with start._session_config("codex", launch = True) as home:
        assert home.parent == short_parent
        assert home.name.startswith("a-codex-")
        assert home.exists()
    assert not home.exists()


def test_locked_file_windows_blocking_retries_until_acquired(monkeypatch, tmp_path):
    attempts = []
    sleeps = []

    def locking(_fd, mode, _length):
        if mode == 1:
            attempts.append(mode)
            if len(attempts) < 3:
                raise PermissionError(start.errno.EACCES, "busy")

    fake_msvcrt = SimpleNamespace(LK_NBLCK = 1, LK_UNLCK = 2, locking = locking)
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    _simulate_windows(monkeypatch)
    monkeypatch.setattr(start.time, "sleep", sleeps.append)

    with start._locked_file(tmp_path / "lock") as acquired:
        assert acquired
    assert len(attempts) == 3
    assert sleeps == [0.05, 0.05]


def test_session_config_reclaims_old_short_homes_but_keeps_recent_and_live(monkeypatch, tmp_path):
    short_parent = tmp_path / "u"
    short_parent.mkdir()
    stale = short_parent / "a-codex-abandoned"
    stale.mkdir()
    (stale / ".active.lock").write_bytes(b"\0")
    (stale / "plugin-checkout").write_text("left behind")
    old = time.time() - start._CODEX_EPHEMERAL_STALE_SECONDS - 1
    os.utime(stale / ".active.lock", (old, old))
    recent = short_parent / "a-codex-surviving-child"
    recent.mkdir()
    (recent / ".active.lock").write_bytes(b"\0")
    monkeypatch.setattr(
        start,
        "_ephemeral_session_parent",
        lambda agent: short_parent if agent == "codex" else None,
    )

    with start._session_config("codex", launch = True) as first:
        assert not stale.exists()
        assert recent.exists()
        with start._session_config("codex", launch = True) as second:
            assert first.exists()
            assert second.exists()
            assert first != second
        assert first.exists()
        assert not second.exists()
    assert not first.exists()


@pytest.mark.parametrize("agent", ["codex", "codex-subagent"])
def test_windows_codex_homes_use_the_short_parent(monkeypatch, tmp_path, agent):
    # codex-subagent nests CODEX_HOME under <home>/parent, so it needs the short root even more.
    monkeypatch.setattr(start.os, "name", "nt")
    monkeypatch.setattr(start.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.delenv("AGENT_SWITCH_HOME", raising = False)

    assert start._ephemeral_session_parent(agent) == tmp_path / ".agent-switch" / ".tmp"
    parent = start._ephemeral_session_parent(agent)
    assert start._ephemeral_session_prefix(agent, parent) == "a-codex-"


def test_non_codex_agents_keep_the_agent_switch_root(monkeypatch, tmp_path):
    monkeypatch.setattr(start.os, "name", "nt")
    monkeypatch.setattr(start.Path, "home", staticmethod(lambda: tmp_path))

    assert start._ephemeral_session_parent("claude") is None
    assert start._ephemeral_session_prefix("claude", None) == "agent-switch-claude-"


def test_session_config_falls_back_when_existing_temp_root_is_unwritable(monkeypatch, tmp_path):
    # mkdir(exist_ok = True) succeeds on an existing unwritable root, so the lock fails first.
    agents = tmp_path / "agents"
    temp_root = agents / ".tmp"
    temp_root.mkdir(parents = True)
    os.chmod(temp_root, 0o500)
    monkeypatch.setattr(start, "_agents_config_root", lambda: agents)

    try:
        with start._session_config("claude", launch = True) as home:
            assert home.exists()
            assert temp_root not in home.parents
    finally:
        os.chmod(temp_root, 0o700)
    assert not home.exists()


def test_augment_path_leaves_path_alone_when_nothing_to_add(monkeypatch):
    monkeypatch.setattr(
        start.Path,
        "home",
        staticmethod(lambda: (_ for _ in ()).throw(RuntimeError("no home directory"))),
    )
    monkeypatch.setenv("PATH", "/usr/bin")

    start._augment_path_with_install_dirs()

    assert os.environ["PATH"] == "/usr/bin"


def test_probe_env_carries_install_dirs_and_restores_path(monkeypatch, tmp_path):
    # A Node-backed shim whose node sits in an install dir needs that dir on PATH when it runs.
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents = True)
    monkeypatch.setattr(start.Path, "home", lambda: tmp_path)
    before = os.environ.get("PATH")

    env = start._probe_env(OPENCODE_CONFIG = "/tmp/cfg.json")

    assert str(local_bin) in env["PATH"]
    assert env["OPENCODE_CONFIG"] == "/tmp/cfg.json"
    assert os.environ.get("PATH") == before


def test_session_config_falls_back_when_the_agents_root_is_unwritable(monkeypatch, tmp_path):
    # A read-only agent-switch home must not stop a launch.
    readonly = tmp_path / "readonly"
    readonly.mkdir(mode = 0o500)
    monkeypatch.setattr(start, "_agents_config_root", lambda: readonly / "agents")

    with start._session_config("claude", launch = True) as home:
        assert home.exists()
        assert readonly not in home.parents
    assert not home.exists()


def test_session_config_reclaims_abandoned_homes_for_non_codex_agents(monkeypatch, tmp_path):
    # Nothing else prunes the agents tree, so a killed wrapper's home must be reclaimed.
    agents_root = tmp_path / "agents"
    temp_root = agents_root / ".tmp"
    temp_root.mkdir(parents = True)
    monkeypatch.setattr(start, "_agents_config_root", lambda: agents_root)
    abandoned = temp_root / "agent-switch-claude-abandoned"
    abandoned.mkdir()
    (abandoned / ".active.lock").write_bytes(b"\0")
    (abandoned / "state.json").write_text("left behind")
    old = time.time() - start._CODEX_EPHEMERAL_STALE_SECONDS - 1
    os.utime(abandoned / ".active.lock", (old, old))
    recent = temp_root / "agent-switch-claude-still-running"
    recent.mkdir()
    (recent / ".active.lock").write_bytes(b"\0")

    with start._session_config("claude", launch = True) as home:
        assert not abandoned.exists()
        assert recent.exists()
        assert home.parent == temp_root
    assert not home.exists()


def test_session_config_serializes_normal_short_home_deletion(monkeypatch, tmp_path):
    short_parent = tmp_path / "u"
    short_parent.mkdir()
    monkeypatch.setattr(start, "_ephemeral_session_parent", lambda _agent: short_parent)
    original_rmtree = start.shutil.rmtree

    def checked_rmtree(path, *args, **kwargs):
        if path.parent == short_parent and path.name.startswith("a-codex-"):
            with start._locked_file(short_parent / ".cleanup.lock", blocking = False) as unlocked:
                assert not unlocked
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(start.shutil, "rmtree", checked_rmtree)
    with start._session_config("codex", launch = True) as home:
        assert home.exists()
    assert not home.exists()


# The temp-dir agents: --persist points each one's home/state env at the stable dir;
# without it, at an ephemeral temp path. opencode is handled separately (only its
# config overlay is relocated; its session data was never in the temp dir).
_RESUME_ENV_VAR = {
    "codex": "CODEX_HOME",
    "pi": "HOME",
    "dsh": "DSH_HOME",
}


def _capture_launch(monkeypatch, argv):
    captured = {}

    def run(
        command,
        env = None,
        **kwargs,
    ):
        captured["command"] = command
        captured["env"] = env
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(start.subprocess, "run", run)
    result = CliRunner().invoke(start.start_app, argv)
    assert result.exit_code == 0, result.output
    return captured


@pytest.mark.parametrize("agent", sorted(_RESUME_ENV_VAR))
def test_resume_persists_agent_home_to_stable_dir(agent, fake_vllm, tmp_path, monkeypatch):
    monkeypatch.setattr(start.shutil, "which", lambda _: f"/usr/local/bin/{agent}")
    if agent == "dsh":
        monkeypatch.setattr(start, "is_deepseek_harness_executable", lambda _: True)
    captured = _capture_launch(monkeypatch, [agent, "--persist"])
    stable = tmp_path / "agents" / agent
    assert captured["env"][_RESUME_ENV_VAR[agent]] == str(stable)
    # The stable dir survives the agent exit, so the session can be resumed.
    assert stable.exists()


@pytest.mark.parametrize("agent", sorted(_RESUME_ENV_VAR))
def test_default_launch_home_is_ephemeral(agent, fake_vllm, tmp_path, monkeypatch):
    monkeypatch.setattr(start.shutil, "which", lambda _: f"/usr/local/bin/{agent}")
    if agent == "dsh":
        monkeypatch.setattr(start, "is_deepseek_harness_executable", lambda _: True)
    captured = _capture_launch(monkeypatch, [agent])
    home = captured["env"][_RESUME_ENV_VAR[agent]]
    parent = start._ephemeral_session_parent(agent)
    assert start._ephemeral_session_prefix(agent, parent) in home
    if parent is None:
        assert Path(home).parent == tmp_path / "agents" / ".tmp"


def test_resume_opencode_config_in_stable_dir(fake_vllm, tmp_path, monkeypatch):
    # opencode's session data lives in ~/.local/share/opencode (never relocated), so
    # resume already survives exit; --persist also stabilizes its config overlay dir.
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/opencode")
    captured = _capture_launch(monkeypatch, ["opencode", "--persist"])
    stable = tmp_path / "agents" / "opencode"
    assert captured["env"]["OPENCODE_CONFIG"] == str(stable / "opencode.json")
    assert stable.exists()


def test_persist_bare_codex_launch_has_no_resume_token(fake_vllm, monkeypatch):
    # A bare `--persist` only persists the session dir; it must NOT auto-append a native
    # resume token, or the very first launch (no session yet) would send codex down its
    # no-session error path. The user resumes explicitly: `agent-switch codex --persist resume`.
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/codex")
    captured = _capture_launch(monkeypatch, ["codex", "--persist"])
    assert "resume" not in captured["command"]
    # command[0] is the resolved executable path; assert the argv after it.
    assert captured["command"][1:] == ["--oss", "--profile", start._CODEX_PROFILE]


def test_persist_bare_opencode_launch_has_no_resume_token(fake_vllm, monkeypatch):
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/opencode")
    captured = _capture_launch(monkeypatch, ["opencode", "--persist"])
    assert "--continue" not in captured["command"]
    assert captured["command"][1:] == ["--model", f"{start._OPENCODE_PROVIDER}/{MODEL['id']}"]


def test_persist_bare_claude_launch_has_no_resume_token(fake_vllm, monkeypatch):
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/claude")
    monkeypatch.setattr(start, "_claude_flags", lambda *a, **k: [])
    captured = _capture_launch(monkeypatch, ["claude", "--persist"])
    assert "--continue" not in captured["command"]
    assert captured["command"][1:] == ["--model", MODEL["id"]]


def test_resume_with_passthrough_does_not_auto_append(fake_vllm, monkeypatch):
    # When the caller drives their own subcommand, --persist only persists the dir; it
    # must not inject a resume token that would collide with the user's command.
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/codex")
    captured = _capture_launch(monkeypatch, ["codex", "--persist", "exec", "hello"])
    assert "resume" not in captured["command"]
    assert captured["command"][-2:] == ["exec", "hello"]


def test_default_launch_has_no_resume_token(fake_vllm, monkeypatch):
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/codex")
    captured = _capture_launch(monkeypatch, ["codex"])
    assert "resume" not in captured["command"]


def test_resume_persist_only_agents_have_no_resume_token(fake_vllm, monkeypatch):
    # Persistence alone must not select a session.
    for agent in ("dsh",):
        monkeypatch.setattr(start.shutil, "which", lambda _, a = agent: f"/usr/local/bin/{a}")
        if agent == "dsh":
            monkeypatch.setattr(start, "is_deepseek_harness_executable", lambda _: True)
        captured = _capture_launch(monkeypatch, [agent, "--persist"])
        assert "resume" not in captured["command"]
        assert "--continue" not in captured["command"]


def test_native_resume_flag_passes_through_unchanged(fake_vllm, monkeypatch):
    # The persistence flag is --persist, NOT --resume, so an agent's own
    # `--resume <id>` (e.g. `agent-switch claude --resume <guid>`) still flows
    # through to the agent verbatim and is not swallowed as an agent-switch option.
    monkeypatch.setattr(start.shutil, "which", lambda _: "/usr/local/bin/claude")
    monkeypatch.setattr(start, "_claude_flags", lambda *a, **k: [])
    captured = _capture_launch(monkeypatch, ["claude", "--resume", "some-session-guid"])
    resume = captured["command"].index("--resume")
    assert captured["command"][resume : resume + 2] == ["--resume", "some-session-guid"]
    assert captured["command"].index("--model") < resume
    # agent-switch never auto-appends its own resume token when the user drives resume.
    assert captured["command"].count("--resume") == 1
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
    monkeypatch.setattr(start.shutil, "which", lambda _: f"/usr/local/bin/{agent}")
    for name in unset:
        monkeypatch.setenv(name, "sk-stale")
    captured = _capture_launch(monkeypatch, [agent])
    for name in unset:
        assert name not in captured["env"]
