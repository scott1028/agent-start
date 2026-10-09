# agent-switch

Launch Claude Code, Codex, OpenCode, Pi or DeepSeek Harness (dsh) against a local model server:
Ollama, LM Studio, llama-server, vLLM or any OpenAI-compatible server. The agent gets a throwaway,
session-only configuration; your own `~/.claude`, `~/.codex`, `~/.dsh`, OpenCode and Pi settings
are not modified.

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
it), so a session sees none of the profiles, plugins or DeepSeek API key in your `~/.dsh`. dsh
replaces any `User-Agent` passed with `--header`. `--as-subagent` is not supported for dsh.

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
   (`fake_vllm` in `tests/test_start.py`), then update the commit above and `NOTICE.md`.
