# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Ported from unsloth_cli/tests/test_start.py (unsloth commit 8e11ba15e); see NOTICE.md.

"""Agent config paths under the agent-switch home."""

from agent_switch.core import (
    session as core_session,
    storage as core_storage,
)


def test_agent_paths_use_agent_switch_home(monkeypatch, tmp_path):
    # Session homes and the per-server key cache both live under agent-switch's own root.
    monkeypatch.setenv("AGENT_SWITCH_HOME", str(tmp_path / "home"))

    assert core_storage._provider_key_cache_path() == tmp_path / "home" / "api_keys.json"
    assert core_session._agents_config_root() == tmp_path / "home" / "agents"
