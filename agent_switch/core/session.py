# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Session config homes: ephemeral and persistent, locks and agent config paths."""

import contextlib
import errno
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from agent_switch.core.platform import _wsl_windows_executable, _wsl_windows_path
from agent_switch.core.storage import _agent_switch_home


_CODEX_EPHEMERAL_STALE_SECONDS = 24 * 60 * 60


_CODEX_EPHEMERAL_HEARTBEAT_SECONDS = 60


def _agent_config_path(path: Path, command: list) -> str:
    """Translate a generated config path when a Windows agent runs through WSL."""
    return _wsl_windows_path(path) if _wsl_windows_executable(command) else str(path)


def _agents_config_root() -> Path:
    return _agent_switch_home() / "agents"


@contextlib.contextmanager
def _temporary_agent_config(prefix: str):
    # Nothing else prunes the agents tree, so reuse the locked session helper: the next launch reclaims homes left by a killed wrapper, and the lock spares live sessions.
    temp_root = _agents_config_root() / ".tmp"
    with contextlib.ExitStack() as stack:
        try:
            temp_root.mkdir(parents = True, exist_ok = True, mode = 0o700)
            path = stack.enter_context(_short_ephemeral_session(temp_root, prefix))
        except OSError:
            # The agent-switch home may be absent or unwritable. Fall back to the system temp dir, as before: no reclamation there, but the OS prunes it.
            path = Path(tempfile.mkdtemp(prefix = prefix))
            stack.callback(shutil.rmtree, path, ignore_errors = True)
        yield path


# codex-subagent nests CODEX_HOME under <home>/parent, so it needs the short root too.
_CODEX_SHORT_HOME_AGENTS = ("codex", "codex-subagent")


def _ephemeral_session_parent(agent: str) -> Optional[Path]:
    """Return a non-system-temp parent when an agent needs one."""
    if os.name != "nt" or agent not in _CODEX_SHORT_HOME_AGENTS:
        return None
    # Codex creates a deeply nested curated-plugin checkout below CODEX_HOME. A normal %TEMP%\\agent-switch-codex-* home can exceed legacy Windows path limits during startup, and Codex also refuses to create its PATH helpers below the system temp directory. Keep the throwaway home short but still private to the current user; _session_config removes it on exit.
    root = _agent_switch_home() / ".tmp"
    root.mkdir(parents = True, exist_ok = True, mode = 0o700)
    return root


def _ephemeral_session_prefix(agent: str, parent: Optional[Path]) -> str:
    """Return the platform-specific prefix for an ephemeral agent home."""
    if agent in _CODEX_SHORT_HOME_AGENTS and parent is not None:
        return "a-codex-"
    return f"agent-switch-{agent}-"


@contextlib.contextmanager
def _locked_file(path: Path, blocking: bool = True):
    """Yield whether an advisory lock was acquired for the first byte of path."""
    handle = path.open("a+b")
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    acquired = True
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    if not blocking:
                        break
                    # LK_LOCK gives up after roughly ten seconds. Poll LK_NBLCK instead so a large stale plugin checkout cannot make a concurrent launch fail just because cleanup takes longer.
                    time.sleep(0.05)
        else:
            import fcntl
            mode = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            try:
                fcntl.flock(handle.fileno(), mode)
                acquired = True
            except BlockingIOError:
                acquired = False
        yield acquired
    finally:
        if acquired:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _reclaim_stale_ephemeral_sessions(parent: Path, prefix: str) -> None:
    """Remove abandoned session homes while preserving locked live sessions."""
    for path in parent.glob(f"{prefix}*"):
        if not path.is_dir():
            continue
        active_lock = path / ".active.lock"
        try:
            modified = active_lock.stat().st_mtime if active_lock.exists() else path.stat().st_mtime
        except FileNotFoundError:
            continue
        # The wrapper owns the advisory lock, not the Codex child. If only the wrapper is killed, its child may still be using CODEX_HOME; give that process a full day to finish before treating the unlocked home as stale.
        if time.time() - modified < _CODEX_EPHEMERAL_STALE_SECONDS:
            continue
        try:
            with _locked_file(active_lock, blocking = False) as stale:
                pass
        except FileNotFoundError:
            # A normally exiting session may have removed itself after the glob.
            continue
        if stale:
            shutil.rmtree(path, ignore_errors = True)


def _refresh_ephemeral_session_marker(path: Path, stop: threading.Event) -> None:
    """Keep the stale grace period relative to wrapper death, not session start."""
    while not stop.wait(_CODEX_EPHEMERAL_HEARTBEAT_SECONDS):
        with contextlib.suppress(OSError):
            os.utime(path, None)


@contextlib.contextmanager
def _short_ephemeral_session(parent: Path, prefix: str = "a-codex-"):
    """Create a session home whose lock makes crash cleanup concurrency-safe."""
    path = None
    active_lock = contextlib.ExitStack()
    heartbeat_stop = None
    heartbeat = None
    try:
        with _locked_file(parent / ".cleanup.lock") as cleanup_lock:
            if not cleanup_lock:  # The blocking acquisition should always succeed.
                raise RuntimeError(f"Could not lock ephemeral session root: {parent}")
            _reclaim_stale_ephemeral_sessions(parent, prefix)
            path = Path(tempfile.mkdtemp(prefix = prefix, dir = parent))
            locked = active_lock.enter_context(_locked_file(path / ".active.lock"))
            if not locked:
                raise RuntimeError(f"Could not lock ephemeral session home: {path}")
            heartbeat_stop = threading.Event()
            heartbeat = threading.Thread(
                target = _refresh_ephemeral_session_marker,
                args = (path / ".active.lock", heartbeat_stop),
                name = "agent-switch-home-heartbeat",
                daemon = True,
            )
            heartbeat.start()
        yield path
    finally:
        if heartbeat_stop is not None:
            heartbeat_stop.set()
        if heartbeat is not None:
            heartbeat.join(timeout = 1)
        try:
            with _locked_file(parent / ".cleanup.lock") as cleanup_lock:
                if not cleanup_lock:  # The blocking acquisition should always succeed.
                    raise RuntimeError(f"Could not lock ephemeral session root: {parent}")
                # Release the live marker only after deletion is serialized with startup scavenging, so no scanner can race this rmtree.
                active_lock.close()
                if path is not None:
                    shutil.rmtree(path, ignore_errors = True)
        finally:
            active_lock.close()


@contextlib.contextmanager
def _session_config(
    agent: str,
    launch: bool,
    persist: bool = False,
):
    """Yield a private directory for an agent's session config (never the user's own). launch (the default) uses an ephemeral temp dir removed after the agent process exits, so nothing persists; no-launch uses a stable agent-switch dir, since the printed recipe is run later on this machine; persist (from --persist) uses that same stable dir even for a launch, so the agent's session survives the exit and can be resumed. Either way the user's real ~/.<agent> config is left untouched."""
    if launch and not persist:
        # Windows codex keeps #7519's short home (MAX_PATH); everyone else uses the agent-switch root.
        parent = _ephemeral_session_parent(agent)
        prefix = _ephemeral_session_prefix(agent, parent)
        if parent is not None:
            with _short_ephemeral_session(parent, prefix) as path:
                yield path
        else:
            with _temporary_agent_config(prefix) as path:
                yield path
    else:
        # Never wipe this dir: a previously printed recipe may still be running an agent whose sessions and state live here, and every config writer merges idempotently into an existing home anyway. Writers must also reset any state a previous run's flags left behind (--yolo especially), since files here outlive the invocation that wrote them.
        path = _agents_config_root() / agent
        path.mkdir(parents = True, exist_ok = True, mode = 0o700)
        yield path
