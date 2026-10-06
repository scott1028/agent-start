# agent-switch

Launch Claude Code, Codex, OpenCode or Pi against a local model server: Unsloth Studio, Ollama,
LM Studio, llama-server, vLLM or any OpenAI-compatible server. The agent gets a throwaway,
session-only configuration; your own `~/.claude`, `~/.codex`, OpenCode and Pi settings are not
modified.

A port of `unsloth start` (see `NOTICE.md` and `PARITY.md`) that works with servers other than
Unsloth.

## Aligned Unsloth version

Ported from and checked for parity against `unsloth start` at
[unslothai/unsloth@8e11ba15ef59b324b99ad55406e98b0391dcd7db](https://github.com/unslothai/unsloth/commit/8e11ba15ef59b324b99ad55406e98b0391dcd7db).

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

Pass `--url`: without it, a running Unsloth Studio is picked before Strata.

## Use

```sh
agent-switch claude                                  # a running Unsloth, else the one local server found
agent-switch codex --url http://127.0.0.1:11434      # Ollama; provider detected from the URL
agent-switch opencode --provider lmstudio -m qwen/qwen3-8b --context-length 32768
agent-switch pi --url http://127.0.0.1:8000/v1 --no-launch   # print the env and command instead
agent-switch claude --as-subagent                    # keep Claude's cloud model, add a local subagent
```

`--header NAME=VALUE` (repeat the flag) adds an HTTP header to every request sent to the
model server, e.g. for a gateway that needs its own auth:
`agent-switch codex --url https://gateway.example/v1 --header "Authorization=...gateway token..."`.
An `Authorization` header there replaces the built-in `Bearer <api-key>`.
It applies to every server, Unsloth Studio included: there an `Authorization` header also stands in
for `--api-key`, so no Studio key is minted. Studio's own API-key listing, minting and identity check
never carry these headers, and neither do the download-progress polls of a server this command has
just started, which use that server's own start key. A server that only answers when they are present
must be named with `--url`, `--provider unsloth` or `UNSLOTH_STUDIO_URL`; they are never sent while
agent-switch probes for an unnamed server.

Arguments agent-switch does not know are passed to the agent unchanged, e.g.
`agent-switch claude -p "..."` or `agent-switch codex exec "..."`.

The context window comes from the server: Ollama `/api/ps`, LM Studio's loaded instance,
llama-server `meta.n_ctx`, vLLM `max_model_len`, or Unsloth. A generic server that reports none
needs `--context-length`.

## Develop

```sh
uv sync
uv run pytest
uv run ruff check .
uv run python tests/parity/compare_no_launch.py   # needs the unsloth checkout at ../unsloth
```
