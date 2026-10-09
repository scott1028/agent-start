# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the agent-switch authors. See LICENSE.

"""LM Studio: loaded instances and their windows from its REST API, loads via /api/v1/models/load."""

from typing import Optional

from agent_switch.providers.types import ProviderError
from agent_switch.providers.utils import error_detail, get_json, request_json, require_json

LABEL = "LM Studio"
DEFAULT_URL = "http://127.0.0.1:1234"
# The OpenAI listing gives no reliable load state or loaded context length; the native REST API
# reports loaded instances and loads a model with a chosen context_length.
CAN_LOAD = True
SETS_CONTEXT = True
# Its OpenAI API reads repeat_penalty and has no min_p, enable_thinking or reasoning_effort.
SUPPORTED_FIELDS = frozenset({"temperature", "top_p", "top_k", "repetition_penalty", "presence_penalty"})
RENAMED_FIELDS = {"repetition_penalty": "repeat_penalty"}


def _v1_models(base: str, key: Optional[str], headers: Optional[dict] = None):
    listing = get_json(base, "/api/v1/models", key, timeout = 3, headers = headers)
    return listing.get("models") if isinstance(listing, dict) and isinstance(listing.get("models"), list) else None


def _v0_models(base: str, key: Optional[str], headers: Optional[dict] = None):
    listing = get_json(base, "/api/v0/models", key, timeout = 3, headers = headers)
    data = listing.get("data") if isinstance(listing, dict) else None
    if isinstance(data, list) and all(isinstance(m, dict) and "state" in m for m in data):
        return data
    return None


def fingerprint(base: str, key: Optional[str] = None, headers: Optional[dict] = None) -> bool:
    return _v1_models(base, key, headers) is not None or _v0_models(base, key, headers) is not None


def can_load(base: str, key: Optional[str], headers: Optional[dict] = None) -> bool:
    """Only servers with the v1 REST API take the load endpoint; older ones get the `lms load` hint."""
    return _v1_models(base, key, headers) is not None


def models(base: str, key: Optional[str], headers: Optional[dict] = None) -> list:
    require_json(LABEL, base, "/v1/models", key, headers)
    listing = _v1_models(base, key, headers)
    entries = []
    if listing is not None:
        for model in listing:
            if not isinstance(model, dict) or model.get("type", "llm") != "llm" or not model.get("key"):
                continue
            instances = model.get("loaded_instances") or []
            for instance in instances:
                config = instance.get("config") or {}
                entries.append(
                    {
                        "id": instance.get("id") or model["key"],
                        "loaded": True,
                        "context_length": config.get("context_length"),
                    }
                )
            if not instances:
                entries.append({"id": model["key"], "loaded": False, "context_length": None})
        return entries
    for model in _v0_models(base, key, headers) or []:
        if model.get("type", "llm") not in ("llm", "vlm") or not model.get("id"):
            continue
        loaded = model.get("state") == "loaded"
        entries.append(
            {
                "id": model["id"],
                "loaded": loaded,
                "context_length": model.get("loaded_context_length") if loaded else None,
            }
        )
    return entries


def load(base: str, key: Optional[str], model: str, context_length: Optional[int], headers: Optional[dict] = None) -> str:
    if _v1_models(base, key, headers) is None:
        raise ProviderError(
            f"This {LABEL} has no load API. Load {model} with `lms load {model}`"
            + (f" --context-length {context_length}" if context_length else "")
            + " and re-run."
        )
    payload = {"model": model}
    if context_length:
        payload["context_length"] = context_length
    status, body = request_json("POST", f"{base}/api/v1/models/load", key, payload, timeout = 900, headers = headers)
    if status != 200:
        raise ProviderError(f"{LABEL} couldn't load {model}: {error_detail(body)}")
    instance = body.get("instance_id") if isinstance(body, dict) else None
    return instance or model
