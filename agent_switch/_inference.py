# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See LICENSE.
# Ported from unsloth_cli/_inference.py (unsloth commit 8e11ba15e); see NOTICE.md.

"""HTTP helpers shared by `agent-switch` and its providers."""

# Cloudflare (in front of remote proxies like RunPod) 403s the default
# "Python-urllib/X.Y" User-Agent as a bot; send a real one on every request.
_USER_AGENT = "agent-switch"


# Built lazily; urllib stays function-local to match this module.
_no_redirect_opener = None


def urlopen_no_redirect(request, timeout):
    """urlopen that errors on any redirect: following a 3xx would send a bearer
    token to a base we never vetted."""
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
