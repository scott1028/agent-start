# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the agent-switch authors. See LICENSE.

"""Differential check: `unsloth start <agent> --no-launch` vs `agent-switch <agent> --no-launch`.

Both CLIs talk to the same fake Studio server, write into throwaway homes, and print a
recipe. After mapping Unsloth's agent-facing ids to agent-switch's neutral ones (see the
naming table in NOTICE.md), the recipes and every generated config file must match.

    uv run python tests/parity/compare_no_launch.py [--unsloth-repo DIR] [--unsloth-python PY] [-k filter]

The unsloth side runs the checkout at --unsloth-repo (the ported commit) through an interpreter
that has Unsloth's dependencies, since an installed `unsloth` may be a different release. The
real agent binaries are probed for versions by both sides.
"""

import argparse
import base64
import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODEL = {"id": "unsloth/gemma-4-26B-A4B-it-GGUF", "context_length": 131072, "loaded": True}
API_KEY = "sk-unsloth-paritytestkey0000"
REPO = Path(__file__).resolve().parents[2]

_SESSION = ["--temperature", "0.6", "--top-k", "20"]
CASES = {
    "claude": [[], ["--yolo"], _SESSION, ["--reasoning", "off"], ["--reasoning-effort", "high"],
               ["--as-subagent"], ["--as-subagent", "--yolo"], ["-m", MODEL["id"]],
               ["--resume", "abc"], ["--", "-p", "hello"]],
    "codex": [[], ["--yolo"], _SESSION, ["--reasoning", "off"], ["--reasoning-effort", "high"],
              ["--as-subagent"], ["-m", MODEL["id"]], ["exec", "hello"]],
    "opencode": [[], ["--yolo"], _SESSION, ["--reasoning", "off"], ["--max-tokens", "4096"],
                 ["--as-subagent"], ["--as-subagent", "--yolo"], ["run", "hello"],
                 ["--yolo", "run", "hello"]],
    "pi": [[], ["--yolo"], _SESSION, ["--reasoning", "off"], ["--max-tokens", "4096"],
           ["--as-subagent"], ["-p", "hello"]],
}

# Unsloth's agent-facing ids -> agent-switch's, applied to the unsloth side only. Longest first.
_TEXT_MAP = [
    ("mcp__plugin_unsloth-local-agent_unsloth__unsloth_plan_agent",
     "mcp__plugin_local-agent_local__local_plan_agent"),
    ("mcp__plugin_unsloth-local-agent_unsloth__unsloth_agent",
     "mcp__plugin_local-agent_local__local_agent"),
    ("unsloth_cli.claude_subagent_mcp", "agent_switch.claude_subagent_mcp"),
    ("unsloth_cli.codex_subagent_mcp", "agent_switch.codex_subagent_mcp"),
    ("unsloth-local-agent", "local-agent"),
    ("unsloth_local_agent", "local_agent"),
    ("unsloth_plan_agent", "local_plan_agent"),
    ("unsloth_agent", "local_agent"),
    ("unsloth_api", "agent_switch"),
    ("UNSLOTH_STUDIO_AUTH_TOKEN", "AGENT_SWITCH_AUTH_TOKEN"),
    ("UNSLOTH_CLAUDE_SUBAGENT_", "AGENT_SWITCH_CLAUDE_SUBAGENT_"),
    ("UNSLOTH_CODEX_SUBAGENT_CONFIG", "AGENT_SWITCH_CODEX_SUBAGENT_CONFIG"),
    ("UNSLOTH_PI_SUBAGENT_", "AGENT_SWITCH_PI_SUBAGENT_"),
    (".unsloth-parent-overlay.json", ".agent-switch-parent-overlay.json"),
    (".unsloth-user-resources.json", ".agent-switch-user-resources.json"),
    ("unsloth-studio", "agent-switch"),
    ("--provider unsloth ", "--provider agent-switch "),
    ("Model served by Unsloth Studio", "Model served by a local server"),
    ('name = "Unsloth Studio"', 'name = "agent-switch"'),
    ('"name": "Unsloth Studio"', '"name": "agent-switch"'),
    ('"agent": {"unsloth": ', '"agent": {"local": '),
    ("Local coding subagent powered by Unsloth for debugging, implementation, and codebase "
     "research. Use when the user asks to spawn an Unsloth or local agent.",
     "Local coding subagent running on a local model for debugging, implementation, and "
     "codebase research. Use when the user asks to spawn a local agent."),
    ("You are a local coding subagent powered by Unsloth.",
     "You are a local coding subagent running on a local model."),
    ("When the user asks to spawn an Unsloth agent or local agent, you must call the",
     "When the user asks to spawn a local agent, you must call the"),
    ("Unsloth is available as a local agent. Ask Claude to spawn an Unsloth or local agent.",
     "A local agent is available. Ask Claude to spawn a local agent."),
    ("Unsloth is available as a local agent. Ask Codex to spawn an Unsloth or local agent.",
     "A local agent is available. Ask Codex to spawn a local agent."),
    ("Unsloth is available as a local agent and in /model. Ask Pi to spawn an Unsloth or local agent.",
     "A local agent is available, and the model is in /model. Ask Pi to spawn a local agent."),
    ("Unsloth is available as @unsloth and in /models.",
     "The local model is available as @local and in /models."),
    ("Delegate a task to the local agent powered by Unsloth. Use when the user asks to spawn "
     "an Unsloth agent or local agent.",
     "Delegate a task to the local agent running on a local model. Use when the user asks to "
     "spawn a local agent."),
    ("Call the Unsloth local agent tool once with the complete task. In plan mode, call the "
     "read-only Unsloth plan agent instead.",
     "Call the local agent tool once with the complete task. In plan mode, call the read-only "
     "local plan agent instead."),
    ("Plan mode is active. Call the read-only Unsloth plan agent ",
     "Plan mode is active. Call the read-only local plan agent "),
    ('"name": "Unsloth AI"', '"name": "agent-switch"'),
]
_UNSLOTH_BOOTSTRAP = (
    "import sys; sys.argv[0] = 'unsloth'; from unsloth_cli import app; sys.exit(app())"
)
# A JSON object key "unsloth" maps by its parent key.
_KEY_MAP = {"providers": "agent-switch", "agent": "local", "mcpServers": "local"}


class _Studio(BaseHTTPRequestHandler):
    def _send(self, body, code = 200):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/health":
            return self._send({"status": "healthy", "service": "Unsloth UI Backend"})
        if path in ("/api/inference/loaded-models", "/v1/models"):
            return self._send({"object": "list", "data": [MODEL]})
        if path == "/api/inference/status":
            return self._send(
                {
                    "is_gguf": True,
                    "gguf_variant": "UD-Q4_K_XL",
                    "active_model": MODEL["id"],
                    "model_identifier": MODEL["id"],
                    "yours": True,
                }
            )
        self._send({"detail": "Not Found"}, 404)

    def do_POST(self):
        self._send({"detail": "Not Found"}, 404)

    def log_message(self, *args):
        pass


def _rename_keys(value, parent = None):
    if isinstance(value, dict):
        return {
            (_KEY_MAP.get(parent, key) if key == "unsloth" else key): _rename_keys(item, key)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rename_keys(item, parent) for item in value]
    return value


def _normalize(text: str, roots: list, unsloth_side: bool) -> str:
    if unsloth_side:
        for old, new in _TEXT_MAP:
            text = text.replace(old, new)
    for root, label in roots:
        text = text.replace(root, label)
    text = re.sub(r"settings-[0-9a-f]{16}\.json", "settings-<H>.json", text)
    text = re.sub(r"sys\.path\.insert\(0,\\?\"[^\"\\]*\\?\"\)", "sys.path.insert(0,<ROOT>)", text)
    return text


def _decode_b64_paths(text: str) -> str:
    def decode(match):
        try:
            return "b64:" + base64.b64decode(match.group(1)).decode()
        except Exception:
            return match.group(0)

    return re.sub(r"b64decode\('([A-Za-z0-9+/=]+)'\)", decode, text)


def _walk(root: Path):
    """Files and links under root, without descending into linked directories."""
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        for name in dirnames:
            if (here / name).is_symlink():
                yield here / name
        for name in filenames:
            if not name.endswith(".lock"):
                yield here / name


def _transcript(output: str, agents_root: Path, roots: list, unsloth_side: bool) -> str:
    parts = [_normalize(_decode_b64_paths(output), roots, unsloth_side)]
    files = []
    for path in _walk(agents_root):
        rel = path.relative_to(agents_root).as_posix()
        if unsloth_side:
            for old, new in _TEXT_MAP:
                rel = rel.replace(old, new)
        files.append((rel, path))
    for rel, path in sorted(files):
        if path.is_symlink():
            # The codex parent overlay links the user's own home: compare where, never what.
            target = _normalize(os.readlink(path), roots, unsloth_side)
            parts.append(f"--- link {rel} -> {target}")
            continue
        raw = path.read_text(encoding = "utf-8", errors = "replace")
        try:
            body = json.dumps(_rename_keys(json.loads(raw)), indent = 2, sort_keys = True)
        except ValueError:
            body = raw
        body = _normalize(_decode_b64_paths(body), roots, unsloth_side)
        parts.append(f"--- file {rel}\n{body}")
    return "\n".join(parts)


def _run(argv, env, cwd) -> str:
    result = subprocess.run(argv, env = env, cwd = cwd, capture_output = True, text = True, timeout = 120)
    return f"[exit {result.returncode}]\n{result.stdout}{result.stderr}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unsloth-repo", default = str(REPO.parent / "unsloth"))
    parser.add_argument(
        "--unsloth-python",
        default = str(Path.home() / ".unsloth" / "studio" / "unsloth_studio" / "bin" / "python"),
    )
    parser.add_argument("--agent-switch", default = str(REPO / ".venv" / "bin" / "agent-switch"))
    parser.add_argument("-k", default = "", help = "only cases whose label contains this")
    args = parser.parse_args()
    unsloth_repo = Path(args.unsloth_repo).resolve()
    if not (unsloth_repo / "unsloth_cli" / "commands" / "start.py").is_file():
        print(f"no unsloth checkout at {unsloth_repo}", file = sys.stderr)
        return 2

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Studio)
    threading.Thread(target = server.serve_forever, daemon = True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    failures = 0
    total = 0
    try:
        for agent, cases in CASES.items():
            for flags in cases:
                label = f"{agent} {' '.join(flags)}".strip()
                if args.k not in label:
                    continue
                total += 1
                with tempfile.TemporaryDirectory(prefix = "parity-") as tmp:
                    tmp = Path(tmp)
                    unsloth_home = tmp / "unsloth-home"
                    agent_switch_home = tmp / "agent-switch-home"
                    work = tmp / "work"
                    for d in (unsloth_home, agent_switch_home, work):
                        d.mkdir()
                    env = dict(os.environ)
                    for name in ("UNSLOTH_API_KEY", "AGENT_SWITCH_API_KEY", "OPENCODE_CONFIG_CONTENT"):
                        env.pop(name, None)
                    env.update(
                        PYTHONPATH = str(unsloth_repo),
                        UNSLOTH_STUDIO_URL = base,
                        UNSLOTH_STUDIO_HOME = str(unsloth_home),
                        AGENT_SWITCH_HOME = str(agent_switch_home),
                        HF_HUB_OFFLINE = "1",
                        NO_COLOR = "1",
                        TERM = "dumb",
                    )
                    argv = [agent, "--no-launch", "--api-key", API_KEY, *flags]
                    unsloth_cmd = [args.unsloth_python, "-c", _UNSLOTH_BOOTSTRAP, "start"]
                    out_u = _run([*unsloth_cmd, *argv], env, work)
                    agent_env = {k: v for k, v in env.items() if k != "PYTHONPATH"}
                    out_a = _run([args.agent_switch, *argv], agent_env, work)
                    roots = [
                        (args.unsloth_python, "<PYTHON>"),
                        (str(REPO / ".venv" / "bin" / "python3"), "<PYTHON>"),
                        (str(REPO / ".venv" / "bin" / "python"), "<PYTHON>"),
                        (str(unsloth_home / "auth" / "agents"), "<AGENTS>"),
                        (str(agent_switch_home / "agents"), "<AGENTS>"),
                        (str(REPO / "agent_switch"), "<PKG>"),
                        (str(unsloth_repo / "unsloth_cli"), "<PKG>"),
                        (str(REPO), "<ROOT>"),
                        (str(unsloth_repo), "<ROOT>"),
                        (str(work), "<WORK>"),
                        (base, "<BASE>"),
                    ]
                    left = _transcript(out_u, unsloth_home / "auth" / "agents", roots, True)
                    right = _transcript(out_a, agent_switch_home / "agents", roots, False)
                    if left == right:
                        print(f"PASS  {label}")
                        continue
                    failures += 1
                    print(f"DIFF  {label}")
                    sys.stdout.writelines(
                        difflib.unified_diff(
                            left.splitlines(keepends = True),
                            right.splitlines(keepends = True),
                            "unsloth start",
                            "agent-switch",
                        )
                    )
                    print()
    finally:
        server.shutdown()
    print(f"\n{total - failures}/{total} cases match")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
