# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See LICENSE.
# Ported from studio/backend/utils/coding_agents.py (unsloth commit 8e11ba15e); see NOTICE.md.

"""Tell DeepSeek Harness (`dsh`) and its TUI (`dsh-tui`) apart from other executables of that name."""

import os
import shutil
import subprocess
from typing import Optional


_DEEPSEEK_HARNESS_HELP_MARKER = "DeepSeek Harness"
_DEEPSEEK_HARNESS_FILE_MARKERS = (
    b"deepseek harness",
    b"@deepseek-ai/dsh",
    b"@deepseek-ai+dsh",
)


def is_deepseek_harness_executable(executable: str, *, allow_execution: bool = True) -> bool:
    """Return whether ``executable`` identifies itself as DeepSeek Harness."""
    try:
        with open(executable, "rb") as launcher:
            contents = launcher.read(256 * 1024).lower()
    except OSError:
        contents = b""
    if any(marker in contents for marker in _DEEPSEEK_HARNESS_FILE_MARKERS):
        return True
    if not allow_execution:
        return False
    try:
        result = subprocess.run(
            [executable, "--help"],
            check = False,
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0 and _DEEPSEEK_HARNESS_HELP_MARKER in output


# npm, Windows cmd-shim and pnpm spellings of the TUI package path inside a launcher.
_DEEPSEEK_HARNESS_TUI_MARKER = "@deepseek-harness-tui/dsh-tui"
_DEEPSEEK_HARNESS_TUI_FILE_MARKERS = (
    _DEEPSEEK_HARNESS_TUI_MARKER.encode(),
    b"@deepseek-harness-tui\\dsh-tui",
    b"@deepseek-harness-tui+dsh-tui",
)


def get_is_deepseek_harness_tui_executable(executable: str, *, allow_execution: bool = True) -> bool:
    """Return whether ``executable`` is the DeepSeek Harness TUI launcher (dsh-tui / dst)."""
    try:
        with open(executable, "rb") as launcher:
            contents = launcher.read(256 * 1024).lower()
    except OSError:
        contents = b""
    if any(marker in contents for marker in _DEEPSEEK_HARNESS_TUI_FILE_MARKERS):
        return True
    if not allow_execution:
        return False
    # `version` answers before any profile bootstrap or delegation, so it writes nothing.
    try:
        result = subprocess.run(
            [executable, "version"],
            check = False,
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0 and _DEEPSEEK_HARNESS_TUI_MARKER in output


def deepseek_harness_executables_on_path(path: Optional[str] = None) -> list[str]:
    """Return every distinct ``dsh`` executable in PATH order."""
    if path is None:
        path = os.environ.get("PATH")
    if path is None:
        path = os.defpath
    executables = []
    seen = set()
    for directory in path.split(os.pathsep):
        try:
            executable = shutil.which("dsh", path = directory)
        except OSError:
            continue
        if executable is None:
            continue
        key = os.path.normcase(os.path.abspath(executable))
        if key in seen:
            continue
        seen.add(key)
        executables.append(executable)
    return executables
