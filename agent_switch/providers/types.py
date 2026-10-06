# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the agent-switch authors. See LICENSE.

"""Types shared by the model-server providers."""

from typing import NamedTuple, Optional


class ProviderError(Exception):
    """A user-facing failure to reach or use a model server."""


class Target(NamedTuple):
    """Which server to use. base None means "discover or auto-start Unsloth"."""

    name: str
    base: Optional[str] = None
    # Custom --header pairs, carried so every request to this server sends them. Never mutated.
    headers: dict = {}
