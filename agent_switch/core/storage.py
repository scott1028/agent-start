# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The agent-switch home, the provider key cache and private file reads/writes."""

import json
import os
from pathlib import Path
from typing import Optional


def _agent_switch_home() -> Path:
    configured = os.environ.get("AGENT_SWITCH_HOME")
    return Path(configured) if configured else Path.home() / ".agent-switch"


def _provider_key_cache_path() -> Path:
    """API keys given with --api-key, remembered per server."""
    return _agent_switch_home() / "api_keys.json"


def _read_cache(cache: Path) -> dict:
    try:
        data = json.loads(cache.read_text(encoding = "utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _saved_keys(servers: object, base: str) -> list:
    # Tolerate a corrupt file: anything but {"saved": [str, ...]} for this base reads as no keys.
    entry = servers.get(base) if isinstance(servers, dict) else None
    saved = entry.get("saved") if isinstance(entry, dict) else None
    return [k for k in saved if isinstance(k, str)] if isinstance(saved, list) else []


def _cached_keys(cache: Path, base: str) -> list:
    # Keys are scoped per server, so a key given for one base is never sent to another.
    return _saved_keys(_read_cache(cache).get("servers"), base)


def _write_private_json(path: Path, data: dict) -> None:
    # O_CREAT with 0o600 so a file holding an API key is never world-readable, even briefly (existing files keep whatever perms the user set).
    path.parent.mkdir(parents = True, exist_ok = True, mode = 0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps(data, indent = 2) + "\n")


def _write_private_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents = True, exist_ok = True, mode = 0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding = "utf-8") as handle:
        handle.write(text)


def _read_yaml_object(path: Path) -> Optional[dict]:
    import yaml

    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding = "utf-8"))
    except (yaml.YAMLError, OSError):
        return None
    if data is None:
        return {}
    return data if isinstance(data, dict) else None


def _read_json_object(path: Path) -> Optional[dict]:
    # {} when missing, None when it cannot be parsed as an object, so the caller leaves a user-managed file untouched rather than clobbering it.
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding = "utf-8"))
    except (ValueError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _subdict(parent: dict, key: str) -> dict:
    child = parent.get(key)
    if not isinstance(child, dict):
        child = parent[key] = {}
    return child


def _remember_key(cache: Path, base: str, key: str) -> None:
    data = _read_cache(cache)
    servers = data.get("servers")
    if not isinstance(servers, dict):
        servers = data["servers"] = {}
    new_entry = {"saved": ([key] + [k for k in _saved_keys(servers, base) if k != key])[:8]}
    if servers.get(base) == new_entry:
        return
    servers[base] = new_entry
    try:
        _write_private_json(cache, data)
    except OSError:
        pass  # worst case the next launch needs --api-key again
