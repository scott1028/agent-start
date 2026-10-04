# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the agent-switch authors. See LICENSE.

"""HTTP helpers shared by the providers."""

import json
import urllib.error
import urllib.request
from typing import Optional

from agent_switch._inference import _USER_AGENT, urlopen_no_redirect

from agent_switch.providers.types import ProviderError

_TIMEOUT_S = 10


def request_json(
    method: str,
    url: str,
    key: Optional[str] = None,
    payload = None,
    timeout: float = _TIMEOUT_S,
) -> tuple:
    """(status, parsed body). A network failure raises OSError; an HTTP error status does not."""
    headers = {"User-Agent": _USER_AGENT}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data = data, headers = headers, method = method)
    try:
        # No redirects: a 3xx would hand the bearer token to a base nobody vetted.
        with urlopen_no_redirect(request, timeout) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as exc:
        status, raw = exc.code, exc.read() if exc.fp is not None else b""
    text = raw.decode("utf-8", "replace")
    try:
        return status, json.loads(text) if text.strip() else None
    except ValueError:
        return status, text


def get_json(base: str, path: str, key: Optional[str] = None, timeout: float = _TIMEOUT_S):
    """Parsed body of a 200 GET, else None. Unreachable also yields None."""
    try:
        status, body = request_json("GET", base + path, key, timeout = timeout)
    except OSError:
        return None
    return body if status == 200 else None


def require_json(label: str, base: str, path: str, key: Optional[str] = None) -> object:
    """Parsed body of a GET that must succeed, else a ProviderError naming the server."""
    try:
        status, body = request_json("GET", base + path, key)
    except OSError as exc:
        raise ProviderError(f"Couldn't reach {label} at {base}: {getattr(exc, 'reason', None) or exc}")
    if status in (401, 403):
        raise ProviderError(
            f"{label} at {base} needs an API key. Pass it with --api-key (or AGENT_SWITCH_API_KEY)."
        )
    if status != 200:
        raise ProviderError(f"{label} at {base} answered {path} with HTTP {status}: {error_detail(body)}")
    return body


def error_detail(body) -> str:
    if isinstance(body, dict):
        error = body.get("error") or body.get("detail") or body.get("message")
        if isinstance(error, dict):
            error = error.get("message")
        if error:
            return str(error)
    return str(body)[:200] if body else "no details"


def endpoint_exists(base: str, path: str, key: Optional[str] = None) -> bool:
    """POST an empty object: a route that exists rejects the body; a missing one is 404/405/501."""
    try:
        status, _ = request_json("POST", base + path, key, payload = {})
    except OSError:
        return False
    return status not in (404, 405, 501)
