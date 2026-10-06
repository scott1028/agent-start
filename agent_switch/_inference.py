# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See LICENSE.
# Ported from unsloth_cli/_inference.py (unsloth commit 8e11ba15e); see NOTICE.md.

"""HTTP and server-discovery helpers shared by `agent-switch` and its providers."""

import itertools
import json
import os
from typing import Optional

# Cloudflare (in front of remote Unsloth proxies like RunPod) 403s the default
# "Python-urllib/X.Y" User-Agent as a bot; send a real one on every request.
_USER_AGENT = "agent-switch"


# Built lazily; urllib stays function-local to match this module.
_no_redirect_opener = None


def urlopen_no_redirect(request, timeout):
    """urlopen that errors on any redirect: following a 3xx would send a bearer
    token (or accept an identity proof) to a base we never vetted, letting a port
    squatter relay a real Unsloth's response."""
    global _no_redirect_opener
    if _no_redirect_opener is None:
        import urllib.error
        import urllib.request

        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                raise urllib.error.HTTPError(
                    req.full_url, code, f"refusing redirect to {newurl}", headers, fp
                )

        _no_redirect_opener = urllib.request.build_opener(_NoRedirect)
    return _no_redirect_opener.open(request, timeout = timeout)


# /api/inference/load and /unload pad their body so a proxy cannot time a slow load out,
# committing the 200 before the work finishes. A failure found after that travels only in-band
# under this key, so a client that treats any 200 as success reports a failed load as a
# successful one.
_DEFERRED_ERROR_KEY = "_deferred_error"


def raise_for_deferred_error(url: str, body):
    """Raise the late failure a padded 200 body carries; else return ``body``.

    ``urllib.error.HTTPError`` specifically: it is the class every CLI caller already
    handles for a plain HTTP failure, so existing ``except`` blocks, messages and exit
    codes keep working, and ``.read()`` yields the same ``{"detail": ...}`` shape.
    """
    if not isinstance(body, dict):
        return body
    deferred = body.get(_DEFERRED_ERROR_KEY)
    if not isinstance(deferred, dict):
        return body

    import email.message
    import io
    import urllib.error

    status = deferred.get("status_code")
    if not isinstance(status, int) or isinstance(status, bool):
        status = 500
    detail = deferred.get("detail")
    if not isinstance(detail, str) or not detail:
        detail = "unknown error" if detail is None else json.dumps(detail)
    headers = email.message.Message()
    headers["Content-Type"] = "application/json"
    raise urllib.error.HTTPError(
        url, status, detail, headers, io.BytesIO(json.dumps({"detail": detail}).encode())
    )


def require_completed_padded_body(url: str, body):
    """Return ``body``, or raise if it is not the payload a padded route promised.

    A proxy that gives up mid-pad leaves a 200 with an empty or truncated body, so
    accepting it reports an unfinished load or unload as completed. Only the two padded
    routes commit their status that early, so only they require a payload; ``{}`` is
    rejected too, since that is what a blank body decodes to here. Mirrored by
    ``assertCompletedPaddedBody`` in studio/frontend/src/features/chat/api/padded-response.ts.
    """
    if isinstance(body, dict) and body:
        return body
    raise RuntimeError(
        f"{url} did not report completion: the connection closed before the "
        "server's reply arrived. Check the model's status before retrying."
    )


def _loopback_candidate_bases(base: str) -> list:
    """For a bare ``localhost`` base, the concrete IP bases to try, IPv4
    127.0.0.1 first (where ``unsloth studio`` binds by default). Pinning to one
    address up front means discovery, the identity check, and the credential we
    then send all target the same endpoint instead of racing IPv4/IPv6
    resolution -- which would otherwise let the health probe land on one address
    and the identity check on another. A literal IP or remote name is unchanged.
    """
    from urllib.parse import urlparse

    parsed = urlparse(base)
    if (parsed.hostname or "").lower() != "localhost":
        return [base]
    import socket

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        ips = {
            ai[4][0] for ai in socket.getaddrinfo(parsed.hostname, port, type = socket.SOCK_STREAM)
        }
    except Exception:
        return [base]
    ordered = sorted(ips, key = lambda ip: (ip != "127.0.0.1", ip))
    bases = [
        f"{parsed.scheme}://" + (f"[{ip}]:{port}" if ":" in ip else f"{ip}:{port}")
        for ip in ordered
    ]
    return bases or [base]


_STUDIO_SERVICE_MARKER = "Unsloth UI Backend"


def _recorded_studio_bases(tried: list):
    # A generator, so the bridge call only runs when the default candidates did not answer.
    from agent_switch.providers import unsloth_bridge

    yield from unsloth_bridge.recorded_studio_bases(tried)


def find_studio_server(timeout: float = 3.0, headers: Optional[dict] = None) -> Optional[str]:
    import urllib.request

    named = os.environ.get("UNSLOTH_STUDIO_URL")
    base = os.environ.get("UNSLOTH_STUDIO_URL", "http://127.0.0.1:8888").rstrip("/")
    candidates = _loopback_candidate_bases(base)
    if not named:
        candidates = itertools.chain(candidates, _recorded_studio_bases(candidates))
    # Custom --header pairs go only to the base the user named; the default port and the
    # pid-record bases stay credential-free probes.
    probe_headers = {"User-Agent": _USER_AGENT}
    probe = urllib.request.urlopen
    if named and headers:
        probe_headers.update(headers)
        # Those pairs can carry a credential, and urllib forwards it verbatim to the next host on a
        # 3xx (an oauth2-proxy or Cloudflare Access front redirects to its IdP). Same reason the
        # rest of this module refuses redirects.
        probe = urlopen_no_redirect
    # Try the concrete loopback addresses in order and return the first that answers, so the rest of
    # the flow talks to that exact address.
    for candidate in candidates:
        request = urllib.request.Request(f"{candidate}/api/health", headers = probe_headers)
        try:
            with probe(request, timeout = timeout) as response:
                # A live port is not Studio: a stranger answering every path would get our key.
                body = json.loads(response.read(65536).decode() or "{}")
                if body.get("service") == _STUDIO_SERVICE_MARKER:
                    return candidate
        except Exception:
            continue
    return None


def is_loopback_url(base: str) -> bool:
    """True only when *base* resolves to loopback. find_studio_server() trusts a
    base after only a health probe, so credentials are auto-sent only to loopback
    (a local Unsloth or an SSH tunnel on 127.0.0.1), the targets the auto flows mean."""
    from urllib.parse import urlparse

    host = (urlparse(base).hostname or "").lower()
    if host in ("localhost", "127.0.0.1", "::1"):
        return True
    try:
        import ipaddress
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False

