# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Ported from unsloth_cli/tests/test_start.py (unsloth commit 8e11ba15e); see NOTICE.md.

"""Launch command resolution: npm cmd-shims, child env and signal handling."""

import os
import shutil
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from agent_switch.core import (
    launch as core_launch,
)
from tests.cli_support import _simulate_windows
from tests.start_split import set_start_attr


@pytest.mark.skipif(os.name == "nt", reason = "POSIX exec signal semantics")
def test_launch_leaves_child_able_to_handle_sigint(monkeypatch, tmp_path):
    # SIG_IGN here reached the agent too, so hermes could never be interrupted.
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import signal, sys\n"
        "sys.exit(17 if signal.getsignal(signal.SIGINT) == signal.SIG_IGN else 0)\n",
        encoding = "utf-8",
    )
    monkeypatch.setattr(shutil, "which", lambda _: sys.executable)
    set_start_attr(monkeypatch, "_augment_path_with_install_dirs", lambda: None)
    before = signal.getsignal(signal.SIGINT)

    code = core_launch._launch([sys.executable, str(probe)], {}, install_hint = "n/a")

    assert code == 0, "child saw SIG_IGN and could never be interrupted"
    assert signal.getsignal(signal.SIGINT) is before


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

    monkeypatch.setattr(shutil, "which", which)
    monkeypatch.setattr(subprocess, "run", run)

    code = core_launch._launch(
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
    monkeypatch.setattr(shutil, "which", lambda name: r"C:\Program Files\nodejs\node.exe")

    assert core_launch._resolved_launch_command(str(cmd), ["--flag"]) == [
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

    monkeypatch.setattr(shutil, "which", which)

    assert core_launch._resolved_launch_command(str(cmd), ["--flag"]) == [
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
    monkeypatch.setattr(shutil, "which", lambda name: r"C:\Program Files\nodejs\node.exe")

    assert core_launch._resolved_launch_command(str(cmd), ["--flag"]) == [
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
    monkeypatch.setattr(shutil, "which", lambda name: r"C:\Program Files\nodejs\node.exe")

    assert core_launch._resolved_launch_command(str(posix_shim), ["--flag"]) == [
        r"C:\Program Files\nodejs\node.exe",
        str(target),
        "--flag",
    ]


def test_resolved_launch_command_keeps_extensionless_shim_without_sibling(monkeypatch, tmp_path):
    # Nothing to substitute, so the path passes through unchanged.
    _simulate_windows(monkeypatch)
    executable = tmp_path / "fake-agent"
    executable.write_text("#!/bin/sh\n", encoding = "utf-8")

    assert core_launch._resolved_launch_command(str(executable), ["--flag"]) == [
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

    assert core_launch._resolved_launch_command(str(executable), ["--flag"]) == [
        str(executable),
        "--flag",
    ]


def test_resolved_launch_command_rescues_uppercase_cmd_sibling(monkeypatch, tmp_path):
    # #9167's pnpm dir holds pi.CMD. A case-sensitive volume needs the .CMD probe;
    # a case-insensitive one answers the earlier .cmd probe with the same file, so
    # compare identity rather than spelling or this passes only on Linux.
    _simulate_windows(monkeypatch)
    posix_shim = tmp_path / "fake-agent"
    posix_shim.write_text("#!/bin/sh\n", encoding = "utf-8")
    cmd = tmp_path / "fake-agent.CMD"
    cmd.write_text("@ECHO off\ncustom-wrapper %*\n", encoding = "utf-8")

    resolved = core_launch._resolved_launch_command(str(posix_shim), ["--flag"])
    assert resolved[1:] == ["--flag"]
    assert os.path.samefile(resolved[0], str(cmd))


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
    monkeypatch.setattr(shutil, "which", lambda name: r"C:\Program Files\nodejs\node.exe")

    assert core_launch._resolved_launch_command(str(posix_shim), ["--flag"]) == [
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

    assert core_launch._resolved_launch_command(str(executable), ["--flag"]) == [
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

    assert core_launch._resolved_launch_command(str(posix_shim), ["--flag"]) == [str(cmd), "--flag"]


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

    monkeypatch.setattr(shutil, "which", which)
    monkeypatch.setattr(subprocess, "run", run)

    code = core_launch._launch(
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

    assert core_launch._resolved_launch_command(str(cmd), ["--flag", "two words"]) == [
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
    monkeypatch.setattr(shutil, "which", lambda name: r"C:\Program Files\nodejs\node.exe")

    assert core_launch._resolved_launch_command(str(cmd), ["--flag"]) == [
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

    assert core_launch._resolved_launch_command(str(cmd), ["--flag"]) == [str(cmd), "--flag"]


def test_resolved_launch_command_leaves_non_npm_batch_file_unchanged(monkeypatch, tmp_path):
    _simulate_windows(monkeypatch)
    cmd = tmp_path / "custom-agent.cmd"
    cmd.write_bytes(b'@echo off\r\n"%dp0%\\custom.exe" %*\r\n')

    assert core_launch._resolved_launch_command(str(cmd), ["--flag"]) == [str(cmd), "--flag"]
