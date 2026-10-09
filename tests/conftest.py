# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See LICENSE.
# Ported from unsloth_cli/tests/conftest.py (unsloth commit 8e11ba15e); see NOTICE.md.

"""Shared fixtures for the agent-switch tests."""

import shutil

import pytest

import agent_switch.providers.utils as provider_utils
from agent_switch import providers
from agent_switch.core import storage as core_storage
from agent_switch.providers.types import Target
from tests.cli_support import BASE, KEY, MODEL
from tests.start_split import set_start_attr


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
    monkeypatch.setattr(providers, "scan_local_servers", lambda: [])
    set_start_attr(monkeypatch, "_active_target", None)
    monkeypatch.setenv("AGENT_SWITCH_HOME", str(tmp_path / "agent-switch-home"))


@pytest.fixture()
def one_local_server(monkeypatch):
    """The no-url scan finds one server; enough for tests that stop before connecting."""
    monkeypatch.setattr(providers, "scan_local_servers", lambda: [Target("vllm", BASE)])


@pytest.fixture()
def fake_vllm(tmp_path, monkeypatch):
    """A vLLM-shaped server at BASE that the no-url scan finds, holding a key given before."""
    calls = []

    def request_json(method, url, key = None, payload = None, timeout = 10, headers = None):
        calls.append((method, url, payload))
        if method == "GET" and url == f"{BASE}/v1/models":
            listing = {"id": MODEL["id"], "owned_by": "vllm", "max_model_len": MODEL["context_length"]}
            return 200, {"object": "list", "data": [listing]}
        if method == "POST" and url in (f"{BASE}/v1/messages", f"{BASE}/v1/responses"):
            # A real route rejects the empty probe body.
            return 400, {"error": {"message": "model is required"}}
        return 404, {"detail": "Not Found"}

    monkeypatch.setattr(provider_utils, "request_json", request_json)
    monkeypatch.setattr(providers, "request_json", request_json)
    monkeypatch.setattr(providers, "scan_local_servers", lambda: [Target("vllm", BASE)])
    core_storage._remember_key(core_storage._provider_key_cache_path(), BASE, KEY)
    # --no-launch session configs land under tmp instead of the real agent-switch dir.
    set_start_attr(monkeypatch, "_agents_config_root", lambda: tmp_path / "agents")
    set_start_attr(monkeypatch, "_require_agent_for_launch", lambda *args: None)
    # Most existing assertions cover the stable V1 command; V2 has focused cases below.
    set_start_attr(monkeypatch, "_opencode_command", lambda *_: ("opencode", False))
    # No `claude` on PATH, so _claude_flags never probes the real binary.
    monkeypatch.setattr(shutil, "which", lambda name, path = None: None)
    monkeypatch.delenv("AGENT_SWITCH_API_KEY", raising = False)
    return calls
