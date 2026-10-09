# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See LICENSE.
# Ported from studio/backend/utils/coding_agents.py (unsloth commit 8e11ba15e); see NOTICE.md.

"""Tell DeepSeek Harness apart from other executables named `dsh` on PATH."""

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
