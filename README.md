# agent-switch

Launch Claude Code, Codex, OpenCode, Pi or DeepSeek Harness (dsh, or its TUI dsh-tui) against a
local model server: Ollama, LM Studio, llama-server, vLLM or any OpenAI-compatible server. The agent
gets a throwaway, session-only configuration; your own `~/.claude`, `~/.codex`, `~/.dsh`,
`~/.dsh-tui`, OpenCode and Pi settings are not modified.

## Tested Strata version

The [Strata quickstart](#quickstart-strata) was tested against
[Niko1221/Strata@6f32ec070f23ced9f50e704d854d775da52591ab](https://github.com/Niko1221/Strata/commit/6f32ec070f23ced9f50e704d854d775da52591ab).

## Install

From the repository root, pick one:

```sh
uv tool install .      # regular: installs a copy of the current source
uv tool install -e .   # editable: runs straight from this checkout
```

| | `uv tool install .` | `uv tool install -e .` |
|---|---|---|
| What gets installed | A built copy of the source as it is now | A link back to this checkout |
| After editing the code | Not picked up until you reinstall: `uv tool install --reinstall .` | Picked up on the next run |
| After moving or deleting this checkout | Keeps working | Breaks; reinstall from the new location |
| After changing dependencies in `pyproject.toml` | Reinstall | Reinstall too: editable only covers the code |
| Good for | Everyday use | Developing agent-switch |

Either way the tool gets its own isolated environment, and `agent-switch` is placed in uv's tool bin directory (`~/.local/bin` by default).

## Quickstart: Strata

[Strata](https://github.com/Niko1221/Strata) serves Qwen3.8-Flash-Next at `http://127.0.0.1:8080`;
agent-switch detects it as llama-server.

```sh
# 1. In the Strata checkout: start the server (--gguf-dir is optional, it reuses GGUF files you already have)
./setup.sh --setup --family qwen --model IQ3_S --gguf-dir /path/to/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF

# 2. In another terminal: launch an agent against it
agent-switch claude --url http://127.0.0.1:8080
agent-switch codex --url http://127.0.0.1:8080
```

`--url` can be left out while Strata is the only server answering on a usual local port.

## Use

```sh
agent-switch claude                                  # the one server found on a usual local port
agent-switch codex --url http://127.0.0.1:11434      # Ollama; provider detected from the URL
agent-switch opencode --provider lmstudio -m qwen/qwen3-8b --context-length 32768
agent-switch pi --url http://127.0.0.1:8000/v1 --no-launch   # print the env and command instead
agent-switch claude --as-subagent                    # keep Claude's cloud model, add a local subagent
agent-switch dsh --url http://127.0.0.1:8080         # DeepSeek Harness, web UI
agent-switch dsh --profile headless "fix the failing test"   # one task, print the result, exit
agent-switch dsh-tui --url http://127.0.0.1:8080     # DeepSeek Harness TUI (alias: dst)
```

Without `--url` or `--provider`, agent-switch checks the usual local ports (Ollama 11434, LM Studio
1234, llama-server 8080, vLLM 8000) and uses the one server it finds; when it finds none or several
it stops and asks for `--url` (or `--provider`). `--provider` without `--url` uses that server's
usual port; `--provider openai` always needs `--url`.

`--api-key` (or `AGENT_SWITCH_API_KEY`) is sent as `Bearer <api-key>` and remembered per server in
`~/.agent-switch/api_keys.json`, so later runs against that server can leave it out.
`AGENT_SWITCH_HOME` moves `~/.agent-switch`, which also holds the `--persist` and `--no-launch`
session dirs.

dsh starts its web profile (`dsh web`) unless the arguments name another, e.g. `--profile headless`.
Its `DSH_HOME` is moved to a session directory that is removed when dsh exits (`--persist` keeps
it), so a session sees none of the profiles, plugins or DeepSeek API key in your `~/.dsh`. Your
`AGENTS.md` and `skills/` from your DSH home (`$DSH_HOME` or `~/.dsh`) are the exception: they are
linked into the session and read through the link. dsh replaces any `User-Agent` passed with
`--header`. `--as-subagent` is not supported for dsh.

`dsh-tui` and its alias `dst` start the DeepSeek Harness TUI (`@deepseek-harness-tui/dsh-tui`,
tried with 0.14.0 on dsh 0.1.5-rc.2 and 0.2.0-rc.2) on the same route, in one session directory
shared by both names. The TUI keeps its state in `~/.dsh-tui` under the home directory, so `HOME`
and `USERPROFILE` move there along with `DSH_HOME`, and pnpm's store and cache stay inside it too:
your own dsh and dsh-tui profiles, accounts, history and pnpm store are neither read nor written.
The same `AGENTS.md` and `skills/` links from your DSH home apply there. `HOME` is moved, so
`~/.agents/skills` is not visible in a dsh-tui session; put such skills in `~/.dsh/skills` or
the project.
dsh-tui and dst keep their session directory (`~/.agent-switch/agents/dsh-tui`) by default, so
settings changed with `/settings`, the TUI's preferences, history and the installed profile
survive between runs, and `--resume <id>` or `-c` reopen a session; `--no-persist` uses a
throwaway directory removed on exit, which installs the TUI profile with pnpm again on start
(about 9 s and 89 MB here). The model stays pinned to the agent-switch route whatever the TUI's
model picker saved. It needs an interactive terminal; `--no-launch` prints the command, whose
last line clears the TUI's inherited handoff variables with `env -u` on Linux and macOS.

- The session is pinned to the dsh backend and the agent-switch model route. `--profile`,
  `--from-default-profile`, a `--backend` other than `dsh`, `--compact-at` (the TUI's agent presets
  own compaction), `--as-subagent` and the launcher's own `update`, `migrate`, `doctor`, `safe`,
  `version` and `help` commands are refused; put such a word after `--` to send it as a prompt.
  Switching the kernel inside the TUI leaves that route.
- A `--patch` you pass is applied after agent-switch's and replaces whole config blocks, so it can
  override the route: a `dsh-tui` row there must restate `provider`, `model`, `backend` and the
  `preset`/`workspace`/`sessionId` bindings. agent-switch warns when one is passed.
- On Windows, dsh-tui runs every tool with `danger-full-access` and no approval prompts whatever
  the permission mode says, so agent-switch requires `--yolo` there. This path is untested. From
  WSL, install dsh-tui and dsh inside WSL: a Windows dsh-tui or dsh is refused, because WSL can
  hand a Windows process a cleared variable only as an empty string, which dsh-tui reads as set.
- dsh-tui binds a per-session socket under the session directory, and a Unix socket path is cut
  at 108 bytes on Linux and 104 on macOS, which would put it outside that directory. agent-switch
  refuses a session directory too deep for the socket to stay inside: keep `AGENT_SWITCH_HOME`
  short. Editor integrations that reach a running TUI through `~/.dsh-tui/inject` (dsh.nvim) do
  not find it.
- With `HOME` moved, Git no longer finds `~/.gitconfig`, so tools the TUI runs lose the global
  identity and settings kept there (`git commit` reports "Author identity unknown") unless your
  environment points Git at a configuration itself, for example with `GIT_CONFIG_GLOBAL`, which
  is passed through unchanged. Set the identity per repository with `git config user.name` /
  `user.email`. How SSH finds keys under the moved home is untested. Moving the home keeps the
  TUI's own state apart; it does not hide every global configuration and is not a sandbox.
- To resume, use `agent-switch dsh-tui --resume <id>`: the TUI's own exit hint runs
  `dsh` directly, outside agent-switch.

`--header NAME=VALUE` (repeat the flag) adds an HTTP header to every request sent to the
model server, e.g. for a gateway that needs its own auth:
`agent-switch codex --url https://gateway.example/v1 --header "Authorization=...gateway token..."`.
An `Authorization` header there replaces the built-in `Bearer <api-key>`.
A server that only answers when they are present must be named with `--url` (or `--provider`):
they are never sent while agent-switch probes the usual local ports for an unnamed server.

Arguments agent-switch does not know are passed to the agent unchanged, e.g.
`agent-switch claude -p "..."` or `agent-switch codex exec "..."`.

The context window comes from the server: Ollama `/api/ps`, LM Studio's loaded instance,
llama-server `meta.n_ctx` or vLLM `max_model_len`. A generic OpenAI-compatible server that reports
none in its `/v1/models` listing needs `--context-length`.

agent-switch never starts a model server itself. By default it may load the `--model` you name
into a server that is already running; `--no-model-load` makes it a pure client that never loads,
reloads or unloads anything — the model must already be loaded on the server.

`--compact-at 0.85` starts the agent's auto-compaction once 85% of that window is used, scaled
per agent (Claude Code, Codex, OpenCode, Pi and DeepSeek Harness); accepted range 0.5–0.95, and
leaving it unset keeps each agent's own behavior. Claude Code applies the fraction to its own
effective window (the window minus its output reserve), so there it can only pull the built-in
trigger earlier, never later. DeepSeek Harness applies it only to its headless profile
(`--profile headless`): its web profile ignores it, and agent-switch warns before starting it.

## MCP servers

`--mcp NAME` (repeat the flag) and `--mcp-all` mount MCP servers into one agent session only, from
a registry agent-switch owns: `~/.agent-switch/mcp.json` (or `$AGENT_SWITCH_HOME/mcp.json`), in the
common `.mcp.json` shape:

```json
{
  "mcpServers": {
    "context7":  { "command": "npx", "args": ["-y", "@upstash/context7-mcp"] },
    "github":    { "type": "http", "url": "https://api.githubcopilot.com/mcp/",
                   "headers": { "Authorization": "Bearer ${GITHUB_TOKEN}" } },
    "atlassian": { "type": "http", "url": "https://mcp.atlassian.com/v1/mcp", "oauth": true }
  }
}
```

A stdio server needs `command` and may carry `args` and `env`; an http server uses
`"type": "http"`, `url`, `headers` and `oauth`. Server names must match `[A-Za-z0-9_-]+`; other
transports (e.g. `sse`) are refused. `${VAR}` in `args`, `env` values, `url` and `headers` values
is expanded from your environment at launch, and an unset variable stops the launch naming it. The
expanded values go only into the session's private (0600) config files, never into the printed
command.

```sh
agent-switch claude --mcp context7 --mcp github   # just these two, this session only
agent-switch codex --mcp-all                      # every server in the registry
```

An http server another app already runs needs no registry entry: `--mcp-url [NAME=]URL` (repeat
the flag), `--mcp-oauth-url` for one that signs in with oauth, and `--mcp-header
[NAME:]HEADER=VALUE` for its headers. The name defaults to the URL's host and port
(`http://127.0.0.1:8931/mcp` becomes `127-0-0-1-8931`). A command-line server mounts the same way:
`--mcp-stdio [NAME=]COMMAND` splits the command like a shell would (so single-quote it), does not
expand `~` (write `${HOME}`), and names itself from the command's last plain word
(`uv run --directory /opt/blender-mcp blender-mcp` becomes `blender-mcp`); if the auto name is
wrong, give `NAME=`. Double-quote the whole command to write it like in a terminal:
`--mcp-stdio 'blender="uv run --directory ~/workspace/blender-mcp blender-mcp"'` runs it as
`bash -ic 'exec uv run ...'`, so bash expands `~` and `${VAR}` itself (needs `bash` on PATH, and
one command only: `;`, `&&`, `||`, `|` and `&` outside quotes are rejected — put multiple steps in
a script). Pi and the dsh-tui/dst TUI move `HOME` to the session dir, so there `~` points at that
dir — use an absolute path with them. `--mcp-env [NAME:]KEY=VALUE` sets its environment; put
secrets in `--mcp-env`, not args: args are always visible in the server's process arguments. The
`NAME:` of `--mcp-header`/`--mcp-env` may be omitted only when exactly one shortcut of that
transport is given, and shortcuts combine with `--mcp`/`--mcp-all`. Single-quote the header and
env values so agent-switch, not your shell, expands `${VAR}`; a literal `Authorization` value sits
in the process list for the whole session, because agent-switch stays running until the agent
exits, so prefer `'${VAR}'`.

```sh
agent-switch claude --mcp-url http://127.0.0.1:8931/mcp                    # name 127-0-0-1-8931
agent-switch claude --mcp-url tools=http://127.0.0.1:8931/mcp
agent-switch codex  --mcp-url api=http://127.0.0.1:9000/mcp --mcp-header 'api:Authorization=Bearer ${API_TOKEN}'
agent-switch pi     --mcp-oauth-url https://mcp.example.com/mcp
agent-switch opencode --mcp-all --mcp-url tools=http://127.0.0.1:8931/mcp
agent-switch claude --mcp-stdio 'uv run --directory /opt/blender-mcp blender-mcp'  # name blender-mcp
agent-switch claude --mcp-stdio 'blender="uv run --directory ~/workspace/blender-mcp blender-mcp"'  # through bash
agent-switch codex  --mcp-stdio 'ctx7=npx -y @upstash/context7-mcp' --mcp-env 'ctx7:API_KEY=${CTX7_KEY}'
```

The mounted servers replace the agent's own MCP servers for the session: Claude Code gets
`--strict-mcp-config --mcp-config=<session file>`, and Codex, OpenCode, Pi and dsh take them through
their isolated session config (OpenCode additionally copies your global config's unmounted servers
with `enabled: false`). MCP config inside the project you launch from may still load: OpenCode's
project `opencode.json` outranks the session overlay, for example. An `oauth: true` http server is
mounted through the pinned `mcp-remote` (needs `npx`): it signs in with a browser on first use, keeps
the tokens under `~/.agent-switch/mcp-auth/mcp-remote-v1/<url-hash>_tokens.json` (delete that file to
sign in again), and takes its `headers` from a private `~/.agent-switch/mcp-auth/<name>.headers` file
rather than the command line. No MCP flag (`--mcp`, `--mcp-all`, `--mcp-url`, `--mcp-oauth-url`,
`--mcp-header`, `--mcp-stdio`, `--mcp-env`) is combined with `--as-subagent`. Each stdio
server starts once per session, when the agent spawns it, and stops when the session ends.

## Develop

```sh
uv sync
uv run pytest
uv run ruff check .
```

## Origin and porting from upstream

agent-switch started as a port of `unsloth start` from
[unslothai/unsloth@8e11ba15ef59b324b99ad55406e98b0391dcd7db](https://github.com/unslothai/unsloth/commit/8e11ba15ef59b324b99ad55406e98b0391dcd7db)
(AGPL-3.0-only; `NOTICE.md` maps each ported file to its source). It is now independent of
Unsloth at runtime: it has no Unsloth Studio integration, mints no keys, and reads no `UNSLOTH_*`
variables. A Studio is not detected; named with `--url`, it is treated like any other
OpenAI-compatible server.

To bring over a later upstream feature:

1. Diff the upstream sources from `8e11ba15e` to the new commit: `unsloth_cli/commands/start.py`,
   `unsloth_cli/_inference.py`, `unsloth_cli/claude_subagent_mcp.py`,
   `unsloth_cli/codex_subagent_mcp.py`, `unsloth_cli/pi_subagent.ts`,
   `studio/backend/utils/coding_agents.py` and `unsloth_cli/tests/`.
2. Port agent-side changes (config writers, launch and install plumbing, WSL/Windows handling,
   subagent bridges) into the matching file from `NOTICE.md`. Skip Studio-only paths: `/api/...`
   routes, API-key minting, model loading and download progress, GGUF checks and the managed Node.
   Server-specific behavior belongs in `agent_switch/providers/`.
3. Rename upstream's agent-facing ids:

   | upstream | agent-switch |
   |---|---|
   | `mcp__plugin_unsloth-local-agent_unsloth__unsloth_agent` (and `..._plan_agent`) | `mcp__plugin_local-agent_local__local_agent` (and `..._plan_agent`) |
   | `unsloth_cli.claude_subagent_mcp`, `unsloth_cli.codex_subagent_mcp` | `agent_switch.claude_subagent_mcp`, `agent_switch.codex_subagent_mcp` |
   | `UNSLOTH_CLAUDE_SUBAGENT_*`, `UNSLOTH_CODEX_SUBAGENT_CONFIG`, `UNSLOTH_PI_SUBAGENT_*` | `AGENT_SWITCH_CLAUDE_SUBAGENT_*`, `AGENT_SWITCH_CODEX_SUBAGENT_CONFIG`, `AGENT_SWITCH_PI_SUBAGENT_*` |
   | `UNSLOTH_STUDIO_AUTH_TOKEN`, `UNSLOTH_API_KEY` | `AGENT_SWITCH_AUTH_TOKEN`, `AGENT_SWITCH_API_KEY` |
   | Codex profile `unsloth_api`; OpenCode, Pi and dsh provider `unsloth`; `unsloth.patch.yml` | `agent_switch`; `agent-switch`; `agent-switch.patch.yml` |
   | OpenCode `agent.unsloth`; `.unsloth-parent-overlay.json`; `.unsloth-user-resources.json` | `agent.local`; `.agent-switch-parent-overlay.json`; `.agent-switch-user-resources.json` |

4. Port the matching upstream tests, run them against the fake vLLM-shaped server
   (`fake_vllm` in `tests/conftest.py`), then update the commit above and `NOTICE.md`.
