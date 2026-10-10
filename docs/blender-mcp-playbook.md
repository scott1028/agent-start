# Blender MCP playbook

Mount the Blender MCP server into any agent through agent-switch, with `--mcp-stdio`.
Copy-paste steps. The commands were verified on 2026-10-10: Claude Code end to end with a real
tool call (`get_objects_summary` returning the live scene; that run added `--no-model-load` —
optional, add it when the model is already loaded and agent-switch must not load or unload
anything — and the read-only restriction flag from section 4), and the exact commands as written
here for all seven agents with `--no-launch`. The earlier end-to-end runs for every agent used
the unquoted form, kept as the alternative in section 2. HTTP mode is out of scope (one sentence
at the end).

## How it fits together

- Inside Blender runs the **"MCP" add-on** (Blender Lab extension, version 1.0.3, maintainer
  "Blender Lab"; installed at
  `~/.config/blender/5.2/extensions/lab_blender_org/mcp/` on the tested machine). Its Preferences
  default to Host `localhost`, Port `9876`, Auto Start on, Auto Start Delay 1 s (`__init__.py`
  properties `host`, `port`, `use_autostart`, `autostart_delay`).
- That `localhost:9876` socket is **not** an MCP endpoint. `mcp_to_blender_server.py` is a plain
  TCP server whose messages are JSON terminated by `\0` (`_encode_response` at line ~228; requests
  split on `b"\0"` at lines ~521-526).
- The **MCP server** is the separate Python program `blender-mcp` (FastMCP). It reaches the add-on
  through `BLENDER_MCP_HOST` / `BLENDER_MCP_PORT` (defaults `localhost` / `9876`) and connects only
  inside tool calls — `send_code` in `blmcp/tools_helpers/connection.py` sends
  `{"type": "execute", "code": "<python>", "strict_json": true}` + `\0`.
- So: agents speak MCP to `blender-mcp`; `blender-mcp` speaks the add-on's private protocol to
  port 9876. In agent-switch terms, `--url` is the **model** server, `--mcp-stdio` is the **MCP**
  server.

```text
+---------------------+  MCP (stdio)  +------------------------+   private TCP    +---------------------------+
| agent               | ────────────► | blender-mcp (FastMCP)  | ───────────────► | Blender add-on "MCP"      |
| claude/codex/pi/... |  --mcp-stdio  | uv run ... blender-mcp |  localhost:9876  | mcp_to_blender_server.py  |
|                     |               |                        |  JSON + "\0"     | (inside the Blender proc) |
+---------------------+               +------------------------+                  +---------------------------+
        ^
        └── --url points here: the model server (e.g. http://127.0.0.1:19090)
```

## Prerequisites

- Blender running with the MCP add-on installed and enabled: add the Blender Lab extensions
  repository `https://lab.blender.org/`, find the MCP add-on, install and enable it (the bundle
  README's add-on instructions).
- `uv` on PATH (the steps below use it).

## 1. Install blender-mcp

The downloadable bundle `blender-<version>.mcpb` (here `~/Downloads/blender-1.0.3.mcpb`) is a zip
with `pyproject.toml` (`[project.scripts] blender-mcp = "blmcp:main"`), `blmcp/` and
`manifest.json`. Unpack and sync it:

```sh
mkdir -p ~/workspace/blender-mcp
unzip -q ~/Downloads/blender-1.0.3.mcpb -d ~/workspace/blender-mcp
uv sync --directory ~/workspace/blender-mcp
```

Check: `~/workspace/blender-mcp/.venv/bin/blender-mcp` exists afterwards.

- The bundle README also documents installing from git instead:
  `pip install git+https://projects.blender.org/lab/blender_mcp.git#subdirectory=mcp`.
- Never copy a `.venv` to another directory — it holds absolute paths. Re-run
  `uv sync --directory <new-place>` instead.

## 2. Mount it in an agent

Shared form (all seven agents) — double-quote the command so it reads like a terminal one:

```sh
agent-switch <agent> --url http://127.0.0.1:19090 \
  --mcp-stdio 'blender="uv run --directory ~/workspace/blender-mcp blender-mcp"'
```

- The double-quoted command runs through `bash -ic 'exec ...'`, so `~` and `${VAR}` expand the
  way your terminal does them. Needs `bash` on PATH, and one command only: `;`, `&&`, `||`, `|`
  and `&` outside the quotes are rejected — put multiple steps in a script.
- Alternative, unquoted: `--mcp-stdio 'blender=uv run --directory /home/<you>/workspace/blender-mcp blender-mcp'`
  starts the server directly (no shell): a `~` inside it is **not** expanded — write an absolute
  path, or `${HOME}/workspace/blender-mcp`, which agent-switch expands itself.
- pi, dsh-tui and dst move `HOME` to the session dir, so there bash does not read your real
  `~/.bashrc` and `~` points at the session dir (inherited environment variables still apply) —
  the commands below use an absolute path for those three.
- Codex passes only a few environment variables to MCP servers; `bash -ic` re-reads `~/.bashrc`,
  so variables exported there come back.
- Your `~/.bashrc` must not print to stdout (stdout is the MCP connection itself); bash's two
  job-control warnings (`cannot set terminal process group`, `no job control in this shell`) show
  up in the MCP log (and on the dsh-tui/dst screen).
- `blender=` names the server; without it the auto name would be `blender-mcp`.
- Only if you changed the add-on's Port: add
  `--mcp-env 'blender:BLENDER_MCP_PORT=<port>'`.
- dsh-tui and dst keep their session directory by default (settings, history and the installed
  profile survive). `--no-persist` is optional: it uses a throwaway directory removed on exit and
  reinstalls the TUI profile with pnpm on every start; the test used it only to leave the saved
  session untouched.

The verified commands:

```sh
# Claude Code, one-shot
agent-switch claude   --url http://127.0.0.1:19090 \
  --mcp-stdio 'blender="uv run --directory ~/workspace/blender-mcp blender-mcp"' \
  -p "<prompt>"

# Codex, one-shot (exec is Codex's own subcommand)
agent-switch codex    --url http://127.0.0.1:19090 \
  --mcp-stdio 'blender="uv run --directory ~/workspace/blender-mcp blender-mcp"' \
  exec "<prompt>"

# Pi, one-shot (HOME is the session dir there: absolute path)
agent-switch pi       --url http://127.0.0.1:19090 \
  --mcp-stdio 'blender="uv run --directory /home/<you>/workspace/blender-mcp blender-mcp"' \
  -p "<prompt>"

# OpenCode, TUI
agent-switch opencode --url http://127.0.0.1:19090 \
  --mcp-stdio 'blender="uv run --directory ~/workspace/blender-mcp blender-mcp"'

# dsh, headless (--profile and the prompt go to dsh)
agent-switch dsh      --url http://127.0.0.1:19090 \
  --mcp-stdio 'blender="uv run --directory ~/workspace/blender-mcp blender-mcp"' \
  --profile headless "<prompt>"

# dsh-tui / dst, TUI (HOME is the session dir there: absolute path)
agent-switch dsh-tui  --url http://127.0.0.1:19090 \
  --mcp-stdio 'blender="uv run --directory /home/<you>/workspace/blender-mcp blender-mcp"'
agent-switch dst      --url http://127.0.0.1:19090 \
  --mcp-stdio 'blender="uv run --directory /home/<you>/workspace/blender-mcp blender-mcp"'
```

## 3. Check it works

- TUIs: OpenCode shows `⊙ 1 MCP /mcps` in the status bar; Claude Code and Pi use `/mcp`.
- Read-only test prompt:

  > Call the get_objects_summary tool of the blender MCP server exactly once, then list the scene
  > name and every object name it returned.

  The default scene answers `Scene` with `Camera`, `Cube`, `Light`.
- Watch the socket: `ss -tanp | grep 9876`. Blender's own `LISTEN` socket is always there; a tool
  call adds a short `ESTAB` owned by `blender-mcp` that turns into `TIME-WAIT`. No tool call, no
  connection.

## 4. Safety

- The server exposes `execute_blender_code` — arbitrary Python inside Blender. **Save your work
  first.**
- Restrict to read-only tools where the agent supports it (both verified):
  - Claude Code: `--allowedTools=mcp__blender__get_objects_summary`
  - Codex: `-c 'mcp_servers.blender.enabled_tools=["get_objects_summary"]'` — place it after
    `exec`.
  - OpenCode, dsh, dsh-tui and dst had no equivalent in the test.

## 5. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| OpenCode log `~/.local/share/opencode/log/opencode.log` shows `mcp connect failed ... MCP server process exited with code 2: error: No such file or directory (os error 2)` | The `--directory` path does not exist, or it starts with an unexpanded `~` (unquoted form, or the moved HOME of pi/dsh-tui/dst) | Use an absolute path; in the unquoted form `${HOME}/...` also works |
| `Failed to spawn: blender-mcp` | The directory is not the unpacked, `uv sync`ed bundle | Re-run step 1 in that directory |
| A tool call fails with `Cannot connect to Blender at localhost:<port>` | Blender not running, add-on not started, or `BLENDER_MCP_PORT` differs from the add-on's Port | Start Blender and the add-on; match the port (`--mcp-env 'blender:BLENDER_MCP_PORT=<port>'`) |
| `--mcp-url http://localhost:9876` cannot work | 9876 is the add-on's private socket, not MCP | Use `--mcp-stdio` (this playbook) |
| A scripted `codex exec` hangs with `Reading additional input from stdin...` | codex waits on stdin | Add `< /dev/null` |
| OpenCode v2.0.26 one-shot `run` answers that no blender tool exists, although the log says `mcp connected server=blender tools=26` | One-shot `run` quirk | Use the OpenCode TUI |
| Pi with `--tools '...,mcp__blender__...'` finds zero MCP tools | pi-mcp-adapter installed (it adds `"-builtin:mcp"` to Pi's `extensions` setting and replaces the built-in MCP — Pi `docs/mcp.md` "Replace the built-in MCP support", adapter `docs/configuration.md`); the allowlist hides the adapter's tools | Do not combine `--tools` with pi-mcp-adapter |
| dsh-tui / dst: blender-mcp's `INFO Processing request ...` log lines draw over the TUI | Cosmetic | Ignore |

## HTTP mode

`blender-mcp --transport http --port <p>` serves MCP at `/` (not `/mcp`; `blmcp/__init__.py` sets
`streamable_http_path = "/"`), reachable with `--mcp-url`, but it is not covered here — not tested
with Blender, and it enables CORS `*` with DNS-rebinding protection off
(`enable_dns_rebinding_protection=False`) while exposing `execute_blender_code`.

## Tested versions (2026-10-10)

Blender 5.2.2 LTS, add-on / bundle 1.0.3, agent-switch `cc91887`, Claude Code 2.1.296,
codex-cli 0.161.0, OpenCode v2.0.26, Pi 1.1.0, dsh 0.1.5-rc.2, dsh-tui 0.14.0.
