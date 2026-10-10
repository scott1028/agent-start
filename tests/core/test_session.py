# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Ported from unsloth_cli/tests/test_start.py (unsloth commit 8e11ba15e); see NOTICE.md.

"""Session config homes: ephemeral, persistent, locks and unwritable-root fallbacks."""

import errno
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_switch.core import (
    session as core_session,
)
from tests.cli_support import BASE, _RESUME_ENV_VAR, _capture_launch, _simulate_windows
from tests.start_split import set_start_attr


def test_session_config_no_launch_preserves_existing_state(fake_vllm, tmp_path):
    # A previously printed recipe may still be running an agent whose sessions
    # or sqlite state live in the stable home; a re-run must not wipe it.
    with core_session._session_config("codex", launch = False) as home:
        marker = home / "sessions" / "live.sqlite"
        marker.parent.mkdir(parents = True)
        marker.write_text("state")
    with core_session._session_config("codex", launch = False) as home2:
        assert home2 == home
        assert (home2 / "sessions" / "live.sqlite").read_text() == "state"


# ── --persist: persist the agent session so it can be resumed ────────────────
def test_session_config_persist_uses_stable_dir_and_survives(monkeypatch, tmp_path):
    # --persist routes a launch to the stable agent-switch agents dir (the one --no-launch
    # already uses) instead of a throwaway temp dir, and never wipes it on exit.
    set_start_attr(monkeypatch, "_agents_config_root", lambda: tmp_path / "agents")
    with core_session._session_config("codex", launch = True, persist = True) as home:
        assert home == tmp_path / "agents" / "codex"
        (home / "marker").write_text("kept")
    assert home.exists()
    assert (home / "marker").read_text() == "kept"


def test_session_config_default_launch_is_ephemeral(monkeypatch, tmp_path):
    agents_root = tmp_path / "agents"
    set_start_attr(monkeypatch, "_agents_config_root", lambda: agents_root)
    with core_session._session_config("codex", launch = True) as home:
        assert home.exists()
        parent = core_session._ephemeral_session_parent("codex")
        assert home.name.startswith(core_session._ephemeral_session_prefix("codex", parent))
        if parent is None:
            assert home.parent == agents_root / ".tmp"
    assert not home.exists()


def test_locked_file_windows_blocking_retries_until_acquired(monkeypatch, tmp_path):
    attempts = []
    sleeps = []

    def locking(_fd, mode, _length):
        if mode == 1:
            attempts.append(mode)
            if len(attempts) < 3:
                raise PermissionError(errno.EACCES, "busy")

    fake_msvcrt = SimpleNamespace(LK_NBLCK = 1, LK_UNLCK = 2, locking = locking)
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    _simulate_windows(monkeypatch)
    monkeypatch.setattr(time, "sleep", sleeps.append)

    with core_session._locked_file(tmp_path / "lock") as acquired:
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
    old = time.time() - core_session._CODEX_EPHEMERAL_STALE_SECONDS - 1
    os.utime(stale / ".active.lock", (old, old))
    recent = short_parent / "a-codex-surviving-child"
    recent.mkdir()
    (recent / ".active.lock").write_bytes(b"\0")
    set_start_attr(monkeypatch, "_ephemeral_session_parent",
        lambda agent: short_parent if agent == "codex" else None,
    )

    with core_session._session_config("codex", launch = True) as first:
        assert not stale.exists()
        assert recent.exists()
        with core_session._session_config("codex", launch = True) as second:
            assert first.exists()
            assert second.exists()
            assert first != second
        assert first.exists()
        assert not second.exists()
    assert not first.exists()


def test_session_config_falls_back_when_existing_temp_root_is_unwritable(monkeypatch, tmp_path):
    # mkdir(exist_ok = True) succeeds on an existing unwritable root, so the lock fails first.
    agents = tmp_path / "agents"
    temp_root = agents / ".tmp"
    temp_root.mkdir(parents = True)
    os.chmod(temp_root, 0o500)
    set_start_attr(monkeypatch, "_agents_config_root", lambda: agents)

    try:
        with core_session._session_config("claude", launch = True) as home:
            assert home.exists()
            assert temp_root not in home.parents
    finally:
        os.chmod(temp_root, 0o700)
    assert not home.exists()


def test_session_config_falls_back_when_the_agents_root_is_unwritable(monkeypatch, tmp_path):
    # A read-only agent-switch home must not stop a launch.
    readonly = tmp_path / "readonly"
    readonly.mkdir(mode = 0o500)
    set_start_attr(monkeypatch, "_agents_config_root", lambda: readonly / "agents")

    with core_session._session_config("claude", launch = True) as home:
        assert home.exists()
        assert readonly not in home.parents
    assert not home.exists()


def test_session_config_serializes_normal_short_home_deletion(monkeypatch, tmp_path):
    short_parent = tmp_path / "u"
    short_parent.mkdir()
    set_start_attr(monkeypatch, "_ephemeral_session_parent", lambda _agent: short_parent)
    original_rmtree = shutil.rmtree

    def checked_rmtree(path, *args, **kwargs):
        if path.parent == short_parent and path.name.startswith("a-codex-"):
            with core_session._locked_file(short_parent / ".cleanup.lock", blocking = False) as unlocked:
                assert not unlocked
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", checked_rmtree)
    with core_session._session_config("codex", launch = True) as home:
        assert home.exists()
    assert not home.exists()


@pytest.mark.parametrize("agent", sorted(_RESUME_ENV_VAR))
def test_default_launch_home_is_ephemeral(agent, fake_vllm, tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name, path = None: f"/usr/local/bin/{agent}")
    if agent == "dsh":
        set_start_attr(monkeypatch, "is_deepseek_harness_executable", lambda _: True)
    captured = _capture_launch(monkeypatch, [agent, "--url", BASE])
    home = captured["env"][_RESUME_ENV_VAR[agent]]
    parent = core_session._ephemeral_session_parent(agent)
    assert core_session._ephemeral_session_prefix(agent, parent) in home
    if parent is None:
        assert Path(home).parent == tmp_path / "agents" / ".tmp"
