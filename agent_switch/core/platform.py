# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""WSL and Windows paths, junctions, PowerShell quoting and npm cmd-shim parsing."""

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from agent_switch.core.options import _fail


def _is_junction(path: Path) -> bool:
    # Path.is_junction() was added in Python 3.12.
    if hasattr(path, "is_junction"):
        return path.is_junction()
    try:
        return (
            getattr(os.lstat(path), "st_reparse_tag", None) == 0xA0000003
        )  # IO_REPARSE_TAG_MOUNT_POINT
    except OSError:
        return False


def _is_directory_link(path: Path) -> bool:
    # lstat reads the link, so FILE_ATTRIBUTE_DIRECTORY answers even when dangling.
    try:
        return bool(getattr(os.lstat(path), "st_file_attributes", 0) & 0x10)
    except OSError:
        return False


def _remove_overlay_entry(path: Path) -> None:
    if _is_junction(path):
        path.rmdir()
    elif os.name == "nt" and path.is_symlink() and _is_directory_link(path):
        # DeleteFileW, which unlink maps to, refuses a directory entry; rmdir drops the link.
        path.rmdir()
    elif path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _create_directory_junction(source: Path, target: Path) -> bool:
    if os.name != "nt":
        return False
    try:
        result = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(target), str(source)],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 30,
            check = False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _wsl_windows_executable(command: list) -> Optional[str]:
    if os.name == "nt" or not os.environ.get("WSL_DISTRO_NAME"):
        return None
    executable = shutil.which(command[0])
    if executable and executable.startswith("/mnt/"):
        return executable
    return None


def _wsl_windows_path(path: Path) -> str:
    try:
        translated = subprocess.check_output(
            ["wslpath", "-w", str(path)], text = True, encoding = "utf-8", errors = "replace"
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        _fail(f"Could not translate WSL path {path}: {exc}")
    if not translated:
        _fail(f"Could not translate WSL path {path}")
    return translated


def _looks_like_path(value: str) -> bool:
    # A var only wants the WSLENV /p flag if its value is a filesystem path: an absolute POSIX path (/...), a UNC path (\\\\...), or a drive-qualified Windows path (C:...). Scalar knobs such as a numeric context window must pass through untranslated, so they get no flag.
    return bool(value) and (value.startswith(("/", "\\")) or (len(value) >= 2 and value[1] == ":"))


def _wsl_bridge_names(env: dict, unset_env: tuple) -> tuple:
    # Build the WSLENV share list for a Windows shim reached from WSL. Path-valued vars get /p so WSLENV translates them to the Windows path the /mnt shim can actually open; a cleared var carries no value to translate.
    names = [name + ("/p" if _looks_like_path(value) else "") for name, value in env.items()]
    names.extend(unset_env)
    return tuple(dict.fromkeys(names))


def _merge_wslenv(current: str, names: tuple) -> str:
    # Index WSLENV entries by bare var name, preserving first-seen order. The vars we bridge are applied last so our entry wins: a user's pre-existing unflagged "HOME" is upgraded to "HOME/p" rather than left as-is, since WSLENV ignores a duplicate name and a bare entry would leave the path untranslated for a Windows shim.
    ordered = []
    by_name = {}
    for entry in (*current.split(":"), *names):
        if not entry:
            continue
        base = entry.split("/", 1)[0]
        if base not in by_name:
            ordered.append(base)
        by_name[base] = entry
    return ":".join(by_name[base] for base in ordered)


def _powershell_quote(arg: str) -> str:
    # PowerShell reads single-quoted strings literally (an embedded ' is doubled), so JSON args such as `--settings {"env":...}` survive intact. list2cmdline's backslash-escaped double quotes are cmd.exe syntax and PowerShell mis-parses them.
    if arg and re.fullmatch(r"[A-Za-z0-9_./:=+-]+", arg):
        return arg
    return "'" + arg.replace("'", "''") + "'"


def _refresh_windows_path() -> None:
    # Merge Windows registry PATH hives after the current process PATH so a freshly installed agent is visible without changing existing precedence.
    if os.name != "nt":
        return
    try:
        import winreg
    except Exception:
        return

    entries = []
    seen = set()

    def add_path(value: str) -> bool:
        added = False
        for entry in str(value).split(os.pathsep):
            entry = entry.strip()
            if not entry:
                continue
            key = os.path.normcase(entry).casefold()
            if key in seen:
                continue
            seen.add(key)
            entries.append(entry)
            added = True
        return added

    add_path(os.environ.get("PATH", ""))
    added_registry = False
    hives = (
        (winreg.HKEY_CURRENT_USER, "Environment"),
        (
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        ),
    )
    for root, sub in hives:
        try:
            with winreg.OpenKey(root, sub) as key:
                value, _ = winreg.QueryValueEx(key, "Path")
        except OSError:
            continue
        if value:
            added_registry = add_path(os.path.expandvars(str(value))) or added_registry
    if added_registry:
        os.environ["PATH"] = os.pathsep.join(entries)


_NPM_CMD_SHIM_HEAD = (
    "@ECHO off\n"
    "GOTO start\n"
    ":find_dp0\n"
    "SET dp0=%~dp0\n"
    "EXIT /b\n"
    ":start\n"
    "SETLOCAL\n"
    "CALL :find_dp0\n"
)


_NPM_NODE_CMD_SHIM_PREFIX = (
    re.escape(_NPM_CMD_SHIM_HEAD)
    + r"(?P<environment>(?:@SET [^=\r\n]+=[^\r\n]+\n)*)"
    + re.escape(
        '\nIF EXIST "%dp0%\\node.exe" (\n'
        + '  SET "_prog=%dp0%\\node.exe"\n'
        + ") ELSE (\n"
        + '  SET "_prog=node"\n'
    )
)


_NPM_NODE_CMD_SHIM_SUFFIX = (
    r"(?P<node_args>[^\r\n]*?)[ \t]+" + r'"%dp0%\\(?P<target>[^"\r\n]+)"[ \t]+%\*'
)


_NPM_NODE_CMD_SHIMS = (
    re.compile(
        _NPM_NODE_CMD_SHIM_PREFIX
        + re.escape(
            "  SET PATHEXT=%PATHEXT:;.JS;=;%\n"
            + ")\n\n"
            + 'endLocal & goto #_undefined_# 2>NUL || title %COMSPEC% & "%_prog%"'
        )
        + _NPM_NODE_CMD_SHIM_SUFFIX,
        re.IGNORECASE,
    ),
    re.compile(
        _NPM_NODE_CMD_SHIM_PREFIX
        + re.escape(
            ")\n\n"
            + "endLocal & goto #_undefined_# 2>NUL || title %COMSPEC% & "
            + "set PATHEXT=%PATHEXT:;.JS;=;% & "
            + '"%_prog%"'
        )
        + _NPM_NODE_CMD_SHIM_SUFFIX,
        re.IGNORECASE,
    ),
)


_NPM_NATIVE_CMD_SHIM = re.compile(
    re.escape(_NPM_CMD_SHIM_HEAD) + r'"%dp0%\\(?P<target>[^"\r\n]+)"[ \t]+%\*',
    re.IGNORECASE,
)


_NPM_NODE_SHEBANG = re.compile(
    r"^#!\s*(?:/usr/bin/env\s+(?:-S\s+)?((?:[^ \t=]+=[^ \t=]+\s+)*))?([^ \t]+)(.*)$"
)


_NPM_SHEBANG_DOLLAR = re.compile(r"\$\{?([^$@#?\- \t{}:]+)\}?")


def _npm_batch_environment(declarations: str) -> str:
    lines = []
    for declaration in declarations.split():
        name, separator, value = declaration.partition("=")
        name = name.strip()
        value = value.strip()
        if separator and name and value:
            value = _NPM_SHEBANG_DOLLAR.sub(lambda match: f"%{match.group(1)}%", value)
            lines.append(f"@SET {name}={value}\n")
    return "".join(lines)


def _windows_expand_environment(value: str, environment: dict) -> str:
    folded = {name.casefold(): item for name, item in environment.items()}
    return re.sub(
        r"%([^%\r\n]+)%",
        lambda match: folded.get(match.group(1).casefold(), ""),
        value,
    )


def _npm_node_shim_metadata(target: Path, match, environment: dict) -> Optional[tuple]:
    environment_block = match.group("environment") or ""
    node_args_text = (match.group("node_args") or "").strip()
    known_node_suffix = target.suffix.lower() in {".js", ".cjs", ".mjs"}
    if not environment_block and not node_args_text and known_node_suffix:
        return [], {}

    first_line = target.read_text(encoding = "utf-8").splitlines()[0]
    shebang = _NPM_NODE_SHEBANG.fullmatch(first_line)
    if shebang is None or Path(shebang.group(2)).name.casefold() not in {"node", "node.exe"}:
        return None
    declarations = shebang.group(1) or ""
    if _npm_batch_environment(declarations).casefold() != environment_block.casefold():
        return None
    if (shebang.group(3) or "").strip() != node_args_text:
        return None
    try:
        node_args = shlex.split(node_args_text) if node_args_text else []
    except ValueError:
        return None

    updates = {}
    expanded_environment = dict(environment)
    for line in environment_block.splitlines():
        name, value = line.removeprefix("@SET ").split("=", 1)
        expanded = _windows_expand_environment(value, expanded_environment)
        expanded_environment[name] = expanded
        updates[name] = expanded
    return node_args, updates


def _apply_windows_environment(environment: dict, updates: dict) -> None:
    for name, value in updates.items():
        existing = next((key for key in environment if key.casefold() == name.casefold()), None)
        if existing is not None and existing != name:
            del environment[existing]
        environment[name] = value
