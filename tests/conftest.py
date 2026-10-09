# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See LICENSE.
# Ported from unsloth_cli/tests/conftest.py (unsloth commit 8e11ba15e); see NOTICE.md.

"""Shared fixtures for the agent-switch tests."""

import pytest

from agent_switch import providers


@pytest.fixture(autouse = True)
def _plain_cli_output(monkeypatch):
    """Keep Typer/Rich from colouring the output these tests assert on.

    Typer renders usage and parameter errors through Rich, which emits ANSI
    escapes as soon as FORCE_COLOR is set -- and a runner that exports it (as
    ours does) splits a plain substring like "Invalid value for
    '--compact-at'" across escape sequences, so `in result.output` stops
    matching even though the message is right there. Setting NO_COLOR is not
    enough on its own: FORCE_COLOR still wins, so it has to be removed.
    """
    for var in ("FORCE_COLOR", "CLICOLOR_FORCE"):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "dumb")


@pytest.fixture(autouse = True)
def _no_local_servers(monkeypatch, tmp_path):
    """No test may probe the developer's real local ports or write their ~/.agent-switch."""
    import agent_switch.start as start

    monkeypatch.setattr(providers, "scan_local_servers", lambda: [])
    monkeypatch.setattr(start, "_active_target", None)
    monkeypatch.setenv("AGENT_SWITCH_HOME", str(tmp_path / "agent-switch-home"))
