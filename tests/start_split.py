# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the agent-switch authors. See LICENSE.

"""Patch helper for the modules the original start.py was split into."""

from agent_switch import start
from agent_switch.agents import claude as _claude
from agent_switch.agents import codex as _codex
from agent_switch.agents import dsh as _dsh
from agent_switch.agents import opencode as _opencode
from agent_switch.agents import pi as _pi
from agent_switch.core import install as _install
from agent_switch.core import launch as _launch
from agent_switch.core import options as _options
from agent_switch.core import platform as _platform
from agent_switch.core import session as _session
from agent_switch.core import storage as _storage

# Every module the original single start.py was split into.
_SPLIT_MODULES = (
    start,
    _options,
    _storage,
    _platform,
    _install,
    _session,
    _launch,
    _claude,
    _codex,
    _opencode,
    _pi,
    _dsh,
)


def set_start_attr(monkeypatch, name, value):
    """monkeypatch.setattr(start, name, value) for the modules split out of start.py.

    Patches `name` in every agent_switch.start / core.* / agents.* module that binds the same
    object, so each caller sees the fake exactly as when all of it lived in start.py.
    Fails if no module binds `name`.
    """
    owners = []
    shared = None
    for module in _SPLIT_MODULES:
        try:
            bound = getattr(module, name)
        except AttributeError:
            continue
        if owners and bound is not shared:
            raise AssertionError(f"{name} is bound to different objects in the split modules")
        shared = bound
        owners.append(module)
    if not owners:
        raise AssertionError(f"no split module binds {name}")
    for module in owners:
        monkeypatch.setattr(module, name, value)
