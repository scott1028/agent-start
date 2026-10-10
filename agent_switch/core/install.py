# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See LICENSE

"""PATH augmentation, agent resolution with install dirs, version probes and installs."""

import contextlib
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from agent_switch._coding_agents import (
    deepseek_harness_executables_on_path,
    is_deepseek_harness_executable,
)
from agent_switch.core.options import _fail
from agent_switch.core.platform import (
    _powershell_quote,
    _refresh_windows_path,
    _wsl_windows_executable,
)


def _npm_install_hint(package: str, *, ignore_scripts: bool = False) -> str:
    parts = ["npm", "install", "-g"]
    if os.name != "nt":
        # No home (bare container UID): fall back to npm's own prefix instead of failing.
        try:
            parts.extend(("--prefix", str(Path.home() / ".local")))
        except (RuntimeError, OSError):
            pass
    if ignore_scripts:
        parts.append("--ignore-scripts")
    parts.append(package)
    if os.name == "nt":
        return " ".join(_powershell_quote(part) for part in parts)
    return shlex.join(parts)


def _codex_executable_version(executable: str) -> Optional[tuple[int, int, int]]:
    try:
        output = subprocess.check_output(
            [executable, "--version"],
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 10,
            stderr = subprocess.DEVNULL,
            env = _probe_env(),
        )
    except Exception:
        return None
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", output)
    return tuple(int(part) for part in match.groups()) if match else None


def _agent_version_at_least(command: str, minimum: tuple) -> bool:
    executable = _which_with_install_dirs(command)
    if executable is None:
        # Only --no-launch gets here without the agent; its recipe may run elsewhere.
        return True
    version = _codex_executable_version(executable)
    return version is not None and version >= minimum


def _get_augmented_path() -> Optional[str]:
    """The PATH with known install dirs appended, or None when PATH is unset and nothing is added, so callers keep the shutil.which os.defpath fallback."""
    # Add known install dirs so a freshly installed agent resolves without a new shell. User dirs are appended, so existing tools keep precedence; a missing home (bare container UID) just means there are none.
    try:
        home = Path.home()
    except (RuntimeError, OSError):
        home = None
    candidates = [home / ".local" / "bin", home / ".opencode" / "bin"] if home is not None else []
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            candidates.append(Path(appdata) / "npm")
    current = os.environ.get("PATH")
    unset = current is None
    if unset:
        # PATH unset: shutil.which() and exec*p* fall back to os.defpath (/bin:/usr/bin), so keep that default instead of collapsing to just the install dirs, which would hide a system-installed agent and strip the launched child's normal PATH. An explicitly empty PATH is left as-is: like shutil.which, it means "search nothing", not os.defpath.
        current = os.defpath
    seen = {os.path.normcase(entry) for entry in current.split(os.pathsep) if entry}
    additions = [
        str(directory)
        for directory in candidates
        if directory.is_dir() and os.path.normcase(str(directory)) not in seen
    ]
    if not additions:
        return None if unset else current
    return os.pathsep.join(part for part in (current, *additions) if part)


def _augment_path_with_install_dirs() -> None:
    # Persist the augmented PATH in os.environ so a launched child inherits it; probes resolve against _get_augmented_path() instead of mutating PATH.
    augmented = _get_augmented_path()
    if augmented is not None:
        os.environ["PATH"] = augmented


def _probe_env(**extra: str) -> dict:
    """Environment for probes that RUN a resolved shim. The copy carries the augmented PATH, so a Node-backed shim whose node sits in an install dir finds it when executed."""
    env = os.environ.copy()
    augmented = _get_augmented_path()
    if augmented is not None:
        env["PATH"] = augmented
    env.update(extra)
    return env


def _prefer_windows_cmd_sibling(executable: Optional[str]) -> Optional[str]:
    """Prefer the sibling .cmd when Windows resolved an extensionless npm/pnpm shim. cmd-shim writes ``to``, ``to.cmd`` and ``to.ps1``, and shutil.which can return the extensionless POSIX shim, which CreateProcess rejects with WinError 193. Measured on windows-latest: 3.12.0 probes the bare name before PATHEXT (gh-109590) and 3.12.1 onwards do not, and a PATHEXT holding "." reaches the same place on any version. Substituted only when the file opens with a shebang, so a real PE keeps priority over a stale wrapper beside it; matched on not-a-Windows-suffix so a dotted bin name is caught too."""
    if executable is None or os.name != "nt":
        return executable
    if Path(executable).suffix.lower() in {".exe", ".com", ".cmd", ".bat", ".ps1"}:
        return executable
    with contextlib.suppress(OSError):
        with open(executable, "rb") as resolved_file:
            if resolved_file.read(2) == b"#!":
                # .CMD only matters on case-sensitive volumes; no writer emits .bat.
                for extension in (".cmd", ".CMD"):
                    sibling = Path(executable + extension)
                    if sibling.is_file():
                        return str(sibling)
    return executable


def _which_with_install_dirs(name: str) -> Optional[str]:
    # shutil.which(name), but searching the known agent install dirs too, so a version probe resolves the same binary _launch() will (it augments PATH before it runs). Without this an agent present only in ~/.local/bin / %APPDATA%
    # pm is missed, wrongly assumed current, and launched with flags an older build rejects. The augmented PATH is passed to shutil.which, never written to os.environ: only _launch() should persist the augmentation for the child process.
    # Callers spawn this result directly, so the shim rescue is needed here too, not only in _resolved_launch_command.
    return _prefer_windows_cmd_sibling(shutil.which(name, path = _get_augmented_path()))


def _which_deepseek_harness_with_install_dirs() -> Optional[str]:
    """Find the first valid DeepSeek Harness even when another ``dsh`` shadows it."""
    for executable in deepseek_harness_executables_on_path(_get_augmented_path()):
        executable = _prefer_windows_cmd_sibling(executable)
        if executable is not None and is_deepseek_harness_executable(executable):
            return executable
    return None


def _install_source(install_hint: str) -> Optional[str]:
    """The first http(s) URL an install hint fetches, or None (e.g. an npm install)."""
    match = re.search(r"https?://[^\s'\")]+", install_hint)
    return match.group(0) if match else None


def _pinned_raw_github_commit(source: str) -> Optional[str]:
    """Return the immutable full commit in a raw GitHub URL, if present."""
    match = re.match(
        r"^https://raw\.githubusercontent\.com/[^/]+/[^/]+/([0-9a-f]{40})/",
        source,
        flags = re.IGNORECASE,
    )
    return match.group(1).lower() if match else None


def _npm_executable() -> Optional[str]:
    executable = _prefer_windows_cmd_sibling(shutil.which("npm"))
    if executable and not _wsl_windows_executable([executable]):
        return executable
    if executable:
        # WSL inherits the Windows PATH, so the rejected shim may shadow a native npm.
        for directory in os.get_exec_path():
            candidate = _prefer_windows_cmd_sibling(shutil.which("npm", path = directory))
            if candidate and not _wsl_windows_executable([candidate]):
                return candidate
    return None


def _install_command(install_hint: str) -> tuple[list[str], Optional[dict]]:
    if not re.match(r"^\s*npm(?:\s|$)", install_hint):
        if os.name == "nt":
            return (
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    install_hint,
                ],
                None,
            )
        return ["/bin/sh", "-c", install_hint], None

    npm = _npm_executable()
    if npm is None:
        _fail(
            "npm is required to install this agent, but no native system npm was found. "
            "Install Node.js with npm, then re-run."
        )
    args = shlex.split(install_hint)
    env = dict(os.environ)
    # dirname, not Path().parent: Path picks its flavour from os.name, which the tests override. Empty means npm is a bare name; prepending "" would put the cwd on PATH.
    npm_dir = os.path.dirname(npm)
    current_path = env.get("PATH", "")
    if npm_dir:
        env["PATH"] = os.pathsep.join([npm_dir, current_path]) if current_path else npm_dir
    if os.name == "nt":
        command = "& " + " ".join(_powershell_quote(arg) for arg in [npm, *args[1:]])
        return (
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                command,
            ],
            env,
        )
    return [npm, *args[1:]], env


def _install_agent(name: str, install_hint: str) -> Optional[str]:
    # Missing agent under --launch: offer to run its documented install command, then re-resolve it on PATH. Consent-based, and a non-interactive stdin cannot answer the prompt, so both the no-TTY and declined cases return None and let the caller print the hint and exit.
    if not sys.stdin.isatty():
        return None
    typer.echo(f"`{name}` is not installed.")
    # Make the supply-chain risk explicit before the prompt: these are the vendors' own installers (curl | bash, irm | iex, npm), run with the user's privileges, and nothing checks a signature or hash on the fetched content. Naming the source turns a blind "yes" into informed consent.
    source = _install_source(install_hint)
    if source:
        pinned_commit = _pinned_raw_github_commit(source)
        if pinned_commit:
            warning = (
                "Security warning: This will download and execute a third-party script "
                f"from {source} with your privileges. agent-switch pins this content to "
                f"immutable upstream commit {pinned_commit}, but does not independently "
                "verify or sandbox it. Continue only if you trust this source and commit."
            )
        else:
            warning = (
                "Security warning: This will download and execute an unverified third-party "
                f"script from {source} with your privileges. agent-switch does not pin or verify "
                "the downloaded content. Continue only if you trust this source."
            )
    else:
        warning = (
            f"This will RUN `{install_hint}` with your privileges; "
            "there is no signature or hash check."
        )
    typer.secho(warning, fg = "yellow", err = True)
    if not typer.confirm(f"Install `{name}` now with `{install_hint}`?", default = False):
        return None
    install_command, install_env = _install_command(install_hint)
    try:
        result = subprocess.run(install_command, env = install_env)
    except OSError as exc:
        _fail(
            f"Could not run the install command: {exc}. "
            f"Run it yourself, then re-run: {install_hint}"
        )
    if result.returncode != 0:
        message = f"Install command failed. Run it yourself, then re-run: {install_hint}"
        if os.name == "nt":
            # A hand-run retry can still hit the policy; point at the one-time per-user fix.
            message += (
                "\nIf it fails because running scripts is disabled (PSSecurityException), "
                "allow local scripts for your user, then retry:\n"
                "  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned"
            )
        _fail(message)
    # Resolve the freshly installed agent without a shell restart: pull registry PATH (Windows) plus well-known install dirs the installer may not have added to PATH.
    _refresh_windows_path()
    _augment_path_with_install_dirs()
    executable = shutil.which(name)
    if executable is None:
        _fail(
            f"`{name}` installed but isn't on PATH yet. Open a new shell (or add it to "
            f"PATH), then re-run. Install command: {install_hint}"
        )
    return executable


def _resolve_or_install_agent(name: str, install_hint: str, resolver) -> str:
    executable = resolver(name)
    invalid_executable = None
    if executable is not None:
        if name != "dsh" or is_deepseek_harness_executable(executable):
            return executable
        invalid_executable = executable
        executable = _which_deepseek_harness_with_install_dirs()
        if executable is not None:
            return executable

    executable = _install_agent(name, install_hint)
    if executable is not None:
        if name != "dsh" or is_deepseek_harness_executable(executable):
            return executable
        invalid_executable = executable
    if name == "dsh":
        executable = _which_deepseek_harness_with_install_dirs()
        if executable is not None:
            return executable

    if invalid_executable is not None:
        _fail(
            f"`{invalid_executable}` is not DeepSeek Harness. Install DeepSeek Harness "
            f"with: {install_hint}"
        )
    _fail(f"`{name}` not found on PATH. Install it with: {install_hint}")


def _require_agent_for_launch(name: str, install_hint: str, launch: bool) -> Optional[str]:
    if not launch:
        return None
    return _resolve_or_install_agent(name, install_hint, _which_with_install_dirs)
