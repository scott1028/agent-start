# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the agent-switch authors. See LICENSE.

"""Ask the local Unsloth install's own interpreter for state only it can read.

Minting an API key needs Studio's private auth secret, and the managed Node, pid
records and Studio home live behind Unsloth's own path resolution. Calling its
public CLI helpers through its interpreter keeps that logic in one place instead
of copying Studio's database schema here.
"""

import functools
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

_PYTHON_ENV = "AGENT_SWITCH_UNSLOTH_PYTHON"
_TIMEOUT_S = 60
# Drop the working directory from sys.path, as Unsloth's own bootstrap does, so a checkout
# named unsloth_cli in the cwd cannot stand in for the installed package.
_PRELUDE = (
    "import json, os, sys\n"
    "sys.path[:1] = [x for x in sys.path[:1] if x not in ('', os.getcwd())]\n"
    "args = json.loads(sys.argv[1])\n"
)


@functools.cache
def unsloth_python() -> Optional[str]:
    """The interpreter that can import unsloth_cli, or None when Unsloth is not installed."""
    configured = os.environ.get(_PYTHON_ENV, "").strip()
    if configured:
        return configured
    launcher = shutil.which("unsloth")
    if launcher and os.name != "nt":
        try:
            with open(launcher, "rb") as handle:
                first = handle.readline().decode("utf-8", "replace").strip()
        except OSError:
            first = ""
        interpreter = first[2:].strip() if first.startswith("#!") else ""
        if interpreter and " " not in interpreter and os.path.isabs(interpreter):
            return interpreter
    venv = Path.home() / ".unsloth" / "studio" / "unsloth_studio"
    for candidate in (venv / "bin" / "python", venv / "Scripts" / "python.exe"):
        if candidate.is_file():
            return str(candidate)
    try:
        import importlib.util

        if importlib.util.find_spec("unsloth_cli") is not None:
            return sys.executable
    except (ImportError, ValueError):
        pass
    return None


def _call(snippet: str, *args, timeout: float = _TIMEOUT_S):
    """Run snippet (which sets `out`) in Unsloth's interpreter; None on any failure."""
    python = unsloth_python()
    if python is None:
        return None
    code = f"{_PRELUDE}{snippet}\nsys.stdout.write('\\n' + json.dumps(out))\n"
    try:
        result = subprocess.run(
            [python, "-X", "utf8", "-c", code, json.dumps(args)],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = timeout,
            stdin = subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    try:
        return json.loads(lines[-1]) if lines else None
    except ValueError:
        return None


def verify_studio_identity(base: str, timeout: float = 3.0) -> bool:
    """Unsloth's same-user HMAC check that `base` is this machine's Studio. Fails closed."""
    return (
        _call(
            "from unsloth_cli._inference import verify_studio_identity\n"
            "out = verify_studio_identity(args[0], timeout = args[1])",
            base,
            timeout,
        )
        is True
    )


def _studio_token() -> Optional[str]:
    """A JWT for Studio's owner, self-issued from the stored secret. None when unavailable."""
    token = _call("from unsloth_cli._inference import _studio_token\nout = _studio_token()")
    return token if isinstance(token, str) and token else None


@functools.cache
def studio_home() -> Optional[Path]:
    home = _call("from unsloth_cli.commands.studio import STUDIO_HOME\nout = str(STUDIO_HOME)")
    return Path(home) if isinstance(home, str) and home else None


def recorded_studio_bases(tried: list) -> list:
    """Loopback bases of live Studio servers from Unsloth's pid records, minus `tried`."""
    bases = _call(
        "from unsloth_cli._inference import _recorded_studio_bases\n"
        "out = list(_recorded_studio_bases(args[0]))",
        list(tried),
    )
    return [base for base in bases if isinstance(base, str)] if isinstance(bases, list) else []


@functools.cache
def managed_node_paths() -> Optional[tuple]:
    """(managed node binary, node that Unsloth resolves) or None when there is no managed Node."""
    paths = _call(
        "from unsloth_cli._inference import ensure_studio_backend_path\n"
        "ensure_studio_backend_path(seed_cache_env = False)\n"
        "from utils.node_runtime import managed_node_binary, resolve_node_executable\n"
        "node = str(managed_node_binary())\n"
        "try:\n"
        "    resolved = resolve_node_executable()\n"
        "    resolved = str(resolved) if resolved else None\n"
        "except Exception:\n"
        "    resolved = None\n"
        "out = [node, resolved]"
    )
    if not isinstance(paths, list) or len(paths) != 2 or not isinstance(paths[0], str):
        return None
    return paths[0], paths[1] if isinstance(paths[1], str) else None


def unsloth_launch_head() -> Optional[list]:
    """argv prefix that runs the `unsloth` CLI through its own interpreter (Windows, #8490)."""
    python = unsloth_python()
    if python is None:
        return None
    head = _call(
        "from pathlib import Path\n"
        "from unsloth_cli.commands.studio import _managed_cli_argv\n"
        "out = _managed_cli_argv(Path(args[0]))",
        python,
    )
    return head if isinstance(head, list) and all(isinstance(a, str) for a in head) else None
