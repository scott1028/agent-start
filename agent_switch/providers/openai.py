# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the agent-switch authors. See LICENSE.

"""Any other OpenAI-compatible server: /v1/models, plus whatever window field it volunteers."""

from typing import Optional

from agent_switch.providers.types import ProviderError
from agent_switch.providers.utils import require_json

LABEL = "OpenAI-compatible server"
# The generic fallback uses only the common OpenAI API and its optional metadata, never native model
# management. It has no default URL, so --provider openai needs --url.
CAN_LOAD = False
SETS_CONTEXT = False
SUPPORTED_FIELDS = None
# Window fields used by servers that report one (OpenRouter-style, vLLM, llama.cpp).
_WINDOW_FIELDS = ("context_length", "max_context_length", "context_window", "max_model_len")


def _window(model: dict) -> Optional[int]:
    for field in _WINDOW_FIELDS:
        if isinstance(model.get(field), int) and model[field] > 0:
            return model[field]
    meta = model.get("meta")
    return meta.get("n_ctx") if isinstance(meta, dict) and isinstance(meta.get("n_ctx"), int) else None


def models(base: str, key: Optional[str], headers: Optional[dict] = None) -> list:
    listing = require_json(LABEL, base, "/v1/models", key, headers)
    return [
        {"id": m["id"], "loaded": True, "context_length": _window(m)}
        for m in (listing or {}).get("data") or []
        if isinstance(m, dict) and m.get("id")
    ]


def load(base: str, key: Optional[str], model: str, context_length: Optional[int], headers: Optional[dict] = None) -> str:
    served = ", ".join(m["id"] for m in models(base, key, headers)) or "nothing"
    raise ProviderError(f"The server at {base} serves {served}; it has no way to load {model}.")
