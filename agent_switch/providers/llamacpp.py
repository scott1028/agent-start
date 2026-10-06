# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the agent-switch authors. See LICENSE.

"""llama-server: one model, or router mode (--models-dir) that loads models on request."""

import time
from typing import Optional

from agent_switch.providers.types import ProviderError
from agent_switch.providers.utils import error_detail, get_json, request_json, require_json

LABEL = "llama-server"
DEFAULT_URL = "http://127.0.0.1:8080"
CAN_LOAD = True
# The window is fixed at server start (-c / a preset), so --context-length only caps it.
SETS_CONTEXT = False
SUPPORTED_FIELDS = None
RENAMED_FIELDS = {"repetition_penalty": "repeat_penalty"}
TEMPLATE_KWARGS = True
_LOAD_TIMEOUT_S = 900
_LOAD_POLL_S = 1.0


def fingerprint(base: str, key: Optional[str] = None, headers: Optional[dict] = None) -> bool:
    listing = get_json(base, "/v1/models", key, timeout = 3, headers = headers)
    data = listing.get("data") if isinstance(listing, dict) else None
    if isinstance(data, list) and any(isinstance(m, dict) and m.get("owned_by") == "llamacpp" for m in data):
        return True
    props = get_json(base, "/props", key, timeout = 3, headers = headers)
    return isinstance(props, dict) and isinstance(props.get("default_generation_settings"), dict)


def _router(base: str, key: Optional[str], headers: Optional[dict] = None) -> bool:
    props = get_json(base, "/props", key, headers = headers)
    return isinstance(props, dict) and props.get("role") == "router"


def models(base: str, key: Optional[str], headers: Optional[dict] = None) -> list:
    listing = require_json(LABEL, base, "/v1/models", key, headers)
    props = get_json(base, "/props", key, headers = headers)
    # meta.n_ctx is the per-slot window a request gets; older builds only report it in /props.
    fallback = None
    if isinstance(props, dict) and props.get("role") != "router":
        fallback = (props.get("default_generation_settings") or {}).get("n_ctx") or None
    entries = []
    for model in (listing or {}).get("data") or []:
        if not isinstance(model, dict) or not model.get("id"):
            continue
        status = (model.get("status") or {}).get("value")
        meta = model.get("meta") or {}
        entries.append(
            {
                "id": model["id"],
                "aliases": [a for a in model.get("aliases") or [] if isinstance(a, str)],
                "loaded": status in (None, "loaded"),
                "context_length": meta.get("n_ctx") or fallback,
            }
        )
    return entries


def load(base: str, key: Optional[str], model: str, context_length: Optional[int], headers: Optional[dict] = None) -> str:
    if not _router(base, key, headers):
        served = ", ".join(m["id"] for m in models(base, key, headers)) or "nothing"
        raise ProviderError(
            f"{LABEL} at {base} serves {served} and can't switch models. Start it with {model}, "
            "or in router mode (--models-dir) so it can load models on request."
        )
    status, body = request_json("POST", f"{base}/models/load", key, {"model": model}, headers = headers)
    if status != 200:
        raise ProviderError(f"{LABEL} couldn't load {model}: {error_detail(body)}")
    # The router answers at once and loads in the background.
    deadline = time.monotonic() + _LOAD_TIMEOUT_S
    while time.monotonic() < deadline:
        listing = get_json(base, "/v1/models", key, headers = headers) or {}
        for entry in listing.get("data") or []:
            if isinstance(entry, dict) and entry.get("id") == model:
                state = (entry.get("status") or {}).get("value")
                if state == "loaded":
                    return model
                if state == "failed":
                    raise ProviderError(f"{LABEL} failed to load {model}.")
        time.sleep(_LOAD_POLL_S)
    raise ProviderError(f"{LABEL} didn't finish loading {model} within {_LOAD_TIMEOUT_S}s.")
