# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the agent-switch authors. See LICENSE.

"""Ollama: models from /v1/models, runtime windows from /api/ps, loads via /api/generate."""

import re
from typing import Optional

from agent_switch.providers.types import ProviderError
from agent_switch.providers.utils import error_detail, get_json, request_json, require_json

LABEL = "Ollama"
DEFAULT_URL = "http://127.0.0.1:11434"
# Preloading and setting num_ctx need the native API (/api/generate, /api/create): the OpenAI API
# can do neither.
CAN_LOAD = True
# --context-length becomes the loaded num_ctx rather than a cap.
SETS_CONTEXT = True
# Its OpenAI API ignores top_k, min_p and repeat_penalty, and has no enable_thinking switch.
SUPPORTED_FIELDS = frozenset({"temperature", "top_p", "presence_penalty", "reasoning_effort"})
_ALIAS_PREFIX = "agent-switch/"


def fingerprint(base: str, key: Optional[str] = None, headers: Optional[dict] = None) -> bool:
    version = get_json(base, "/api/version", key, timeout = 3, headers = headers)
    ps = get_json(base, "/api/ps", key, timeout = 3, headers = headers)
    return isinstance(version, dict) and "version" in version and isinstance(ps, dict)


def canonical(model: str) -> str:
    """Ollama names a bare model by its :latest tag."""
    last = model.rsplit("/", 1)[-1]
    return model if ":" in last else f"{model}:latest"


def models(base: str, key: Optional[str], headers: Optional[dict] = None) -> list:
    listed = require_json(LABEL, base, "/v1/models", key, headers)
    # /v1/models has no load state or window; /api/ps reports what runs and its context length.
    ps = require_json(LABEL, base, "/api/ps", key, headers)
    running = {}
    for item in (ps or {}).get("models") or []:
        for name in {item.get("name"), item.get("model")} - {None}:
            running[name] = item.get("context_length")
    ids = [m.get("id") for m in (listed or {}).get("data") or [] if isinstance(m, dict)]
    ids += [name for name in running if name not in ids]
    return [{"id": i, "loaded": i in running, "context_length": running.get(i)} for i in ids if i]


def _alias(model: str, context_length: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", model.lower().removesuffix(":latest")).strip("-")
    return f"{_ALIAS_PREFIX}{slug}-ctx{context_length}:latest"


def resident_alias(entries: list, model: str, context_length: Optional[int]) -> Optional[dict]:
    """The loaded ctx alias carrying this window, if one is already resident."""
    if not context_length:
        return None
    alias = _alias(canonical(model), context_length)
    return next((entry for entry in entries if entry["id"] == alias and entry["loaded"]), None)


def _not_found(model: str, body) -> ProviderError:
    return ProviderError(
        f"{LABEL} doesn't have {model}: {error_detail(body)}. Pull it first with `ollama pull {model}`."
    )


def load(base: str, key: Optional[str], model: str, context_length: Optional[int], headers: Optional[dict] = None) -> str:
    name = canonical(model)
    if context_length:
        # The OpenAI API cannot carry num_ctx, so an alias with it baked in keeps the window even
        # after Ollama unloads an idle model and reloads it on the next request. Reused by name.
        alias = _alias(name, context_length)
        status, body = request_json(
            "POST",
            f"{base}/api/create",
            key,
            {"model": alias, "from": name, "parameters": {"num_ctx": context_length}, "stream": False},
            timeout = 600,
            headers = headers,
        )
        if status == 404:
            raise _not_found(model, body)
        if status != 200:
            raise ProviderError(f"{LABEL} couldn't set up {model} with {context_length} tokens: {error_detail(body)}")
        name = alias
    status, body = request_json("POST", f"{base}/api/generate", key, {"model": name}, timeout = 900, headers = headers)
    if status == 404:
        raise _not_found(model, body)
    if status != 200:
        raise ProviderError(f"{LABEL} couldn't load {model}: {error_detail(body)}")
    return name
