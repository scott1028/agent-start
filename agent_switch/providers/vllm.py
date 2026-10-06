# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the agent-switch authors. See LICENSE.

"""vLLM: serves the models it was started with; /v1/models reports each one's max_model_len."""

from typing import Optional

from agent_switch.providers.types import ProviderError
from agent_switch.providers.utils import get_json, require_json

LABEL = "vLLM"
DEFAULT_URL = "http://127.0.0.1:8000"
CAN_LOAD = False
SETS_CONTEXT = False
SUPPORTED_FIELDS = None
TEMPLATE_KWARGS = True


def fingerprint(base: str, key: Optional[str] = None, headers: Optional[dict] = None) -> bool:
    listing = get_json(base, "/v1/models", key, timeout = 3, headers = headers)
    data = listing.get("data") if isinstance(listing, dict) else None
    return isinstance(data, list) and any(
        isinstance(m, dict) and ("max_model_len" in m or m.get("owned_by") == "vllm") for m in data
    )


def models(base: str, key: Optional[str], headers: Optional[dict] = None) -> list:
    listing = require_json(LABEL, base, "/v1/models", key, headers)
    return [
        {"id": m["id"], "loaded": True, "context_length": m.get("max_model_len")}
        for m in (listing or {}).get("data") or []
        if isinstance(m, dict) and m.get("id")
    ]


def load(base: str, key: Optional[str], model: str, context_length: Optional[int], headers: Optional[dict] = None) -> str:
    served = ", ".join(m["id"] for m in models(base, key, headers)) or "nothing"
    raise ProviderError(f"{LABEL} at {base} serves {served} and can't load {model}; restart it with that model.")
