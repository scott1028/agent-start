# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Ported from unsloth_cli/tests/test_start.py (unsloth commit 8e11ba15e); see NOTICE.md.

"""Windows/WSL platform helpers: registry PATH, junctions, WSLENV, quoting."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_switch.core import (
    platform as core_platform,
)


def test_refresh_windows_path_noop_off_windows(monkeypatch):
    monkeypatch.setattr(os, "name", "posix")
    before = os.environ.get("PATH", "")
    monkeypatch.setenv("PATH", before)
    core_platform._refresh_windows_path()
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
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(os, "pathsep", ";")
    monkeypatch.setitem(sys.modules, "winreg", fake_winreg)
    monkeypatch.setenv("PATH", r"C:\custom;C:\existing")

    core_platform._refresh_windows_path()

    assert os.environ["PATH"].split(";") == [
        r"C:\custom",
        r"C:\existing",
        r"C:\Users\me\hermes\bin",
        r"C:\Windows\System32",
    ]


def test_create_directory_junction_uses_windows_mklink(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(os, "name", "nt")

    def run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(subprocess, "run", run)
    source = tmp_path / "source"
    target = tmp_path / "target"

    assert core_platform._create_directory_junction(source, target) is True
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

    monkeypatch.setattr(os, "lstat", lstat)
    assert core_platform._is_junction(tmp_path / "junction")
    assert not core_platform._is_junction(tmp_path / "symlink")
    assert not core_platform._is_junction(tmp_path / "missing")


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
        os,
        "lstat",
        lambda p: SimpleNamespace(
            st_file_attributes = 0x10,
            st_reparse_tag = 0,
            st_mode = real_lstat(p).st_mode,
        ),
    )
    # rmdir on a link is POSIX-invalid, so record the routing instead of running it.
    monkeypatch.setattr(os, "name", "nt")
    calls = []
    monkeypatch.setattr(Path, "unlink", lambda self, **kw: calls.append("unlink"))
    monkeypatch.setattr(Path, "rmdir", lambda self: calls.append("rmdir"))

    core_platform._remove_overlay_entry(target)

    assert calls == ["rmdir"]
    assert (source / "keep.txt").read_text() == "keep\n"


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
    names = core_platform._wsl_bridge_names(env, ("ANTHROPIC_API_KEY",))
    assert "CODEX_HOME/p" in names
    assert "HOME/p" in names
    assert "USERPROFILE/p" in names  # drive-qualified Windows path
    assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW" in names  # scalar: no /p
    assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW/p" not in names
    assert "ANTHROPIC_BASE_URL" in names  # URL is not a filesystem path
    assert "ANTHROPIC_API_KEY" in names  # cleared var carries no value to translate


def test_merge_wslenv_dedups_on_base_name():
    # An already-shared var must not be appended again just because the flag differs.
    merged = core_platform._merge_wslenv("CODEX_HOME/p:FOO", ("CODEX_HOME/p", "BAR/p"))
    parts = merged.split(":")
    assert parts.count("CODEX_HOME/p") == 1
    assert "FOO" in parts and "BAR/p" in parts


def test_merge_wslenv_upgrades_existing_unflagged_entry():
    # A user's pre-existing bare "HOME" must be upgraded to "HOME/p" (not left bare or
    # duplicated), or the Windows shim gets the path without WSL translation.
    merged = core_platform._merge_wslenv("HOME:FOO", ("HOME/p", "CODEX_HOME/p"))
    parts = merged.split(":")
    assert "HOME/p" in parts and "HOME" not in parts  # upgraded in place
    assert parts.count("HOME/p") == 1
    assert "FOO" in parts  # untouched user var preserved
    assert "CODEX_HOME/p" in parts
