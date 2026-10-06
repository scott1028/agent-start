# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the agent-switch authors. See LICENSE.

"""Unsloth Studio. Connecting reuses the ported `unsloth start` flow in agent_switch.start."""

from typing import Optional

from agent_switch._inference import _STUDIO_SERVICE_MARKER
from agent_switch.providers.utils import get_json

LABEL = "Unsloth"


def fingerprint(base: str, key: Optional[str] = None, headers: Optional[dict] = None) -> bool:
    health = get_json(base, "/api/health", timeout = 3, headers = headers)
    return isinstance(health, dict) and health.get("service") == _STUDIO_SERVICE_MARKER
