# Parity with `unsloth start`

Reference: unsloth commit `8e11ba15e`, `unsloth_cli/commands/start.py` and its tests.
Markers: `[v]` handled and passed, `[x]` handled but not passed, `[ ]` not handled yet.
Out of scope for v1 (stays on `unsloth start`): the hermes, openclaw and dsh agents, and `--app`.

How each item was checked:

- **tests**: the ported upstream tests (`tests/test_*.py`) pass.
- **diff**: `tests/parity/compare_no_launch.py` matches `unsloth start --no-launch` (34/34 cases).
- **live**: run on this Linux host against a real server.
- **sim**: Windows/WSL behavior, checked only by tests that fake the platform; not run on a real Windows or WSL machine.

## Shared CLI surface (claude, codex, opencode, pi)

- [v] Positional `org/name(:variant)` is routed to `--model`; the rest passes through to the agent (tests, diff)
- [v] `--` separator is preserved when forwarding (`_PassthroughCommand`) (tests, diff)
- [v] `--model/-m`, `--gguf-variant`, `--max-seq-length/--context-length`, `--load-in-4bit/--no-load-in-4bit`, `--tensor-parallel`, `--gpu-memory-mode` (tests, diff)
- [v] `--serve/--no-serve` auto-starts `unsloth run` for `--model` when no server is found (tests)
- [v] `--enable-tools/--disable-tools`, tool-call healing and nudging flags for an auto-started server (tests)
- [v] `--reasoning on|off|auto`, `--reasoning-effort` (tests, diff)
- [v] Sampling pins: `--temperature`, `--top-p`, `--top-k`, `--min-p`, `--repetition-penalty`, `--presence-penalty` (tests, diff)
- [v] `--max-tokens` (opencode, pi) (tests, diff)
- [v] `--api-key` (env `UNSLOTH_API_KEY`, now also `AGENT_SWITCH_API_KEY`), remembered per server (tests, live)
- [v] `--launch/--no-launch`: POSIX, PowerShell and WSL recipes, self-contained last line (tests, diff; PowerShell/WSL sim)
- [v] `--yolo` and its two aliases, routed per agent (tests, diff)
- [v] `--persist/--no-persist` session dirs (tests)
- [v] `--as-subagent` (claude, codex, opencode, pi) (tests, diff; claude and codex live)
- [v] Help grouped into Model / Server / Sampling / Agent session panels (tests)

## Unsloth server handling

- [v] Discovery: `UNSLOTH_STUDIO_URL`, loopback candidates, pid records, service marker check (tests, live)
- [v] Auto-start: port follow, early key marker, download progress, no-progress timeout, teardown on failure, server kept after a good session (tests)
- [v] Server-wide pins only reach a server this command starts; warnings when attaching (tests, diff)
- [v] API key: explicit, saved per server, identity-verified loopback mint, minted-key replay, remote refusal (tests; shared cache replay live)
- [v] Model resolution: attach, load with knobs, eviction notices, inferred resident reload, trust_remote_code refusal, already_loaded reuse (tests)
- [v] GGUF preflight and attach check for claude and codex (tests)
- [v] Subagent model id pins the GGUF variant (tests, diff)
- [v] Memory warning passthrough (tests)

## Agent launch plumbing

- [v] Agent auto-install with consent and security warning (curl, irm, npm) (tests)
- [v] PATH augmentation (`~/.local/bin`, `~/.opencode/bin`, `%APPDATA%\npm`, managed Node) (tests; managed Node through the bridge, live)
- [v] npm executable selection (system, managed, WSL shim skip) (tests; WSL sim)
- [v] Windows npm `.cmd` shim resolution and PowerShell quoting (tests; sim)
- [v] WSL: Windows shim detection, WSLENV bridging, path translation (tests; sim)
- [v] Ctrl+C cancels a turn, not the wrapper; signal exit codes become 128+N (tests)
- [v] Ephemeral session homes: lock, heartbeat, stale reclamation, Windows short home for codex (tests; removal live)
- [v] Post-session notices (server still running, agent exit code) (tests)

## claude

- [v] Env: base URL, auth token, model, context window, auto-compact, attribution, non-essential traffic, betas, flicker, token reminder (tests, diff, live)
- [v] `--settings` overlay with env pins and `availableModels`; forwarded `--settings` kept first (tests, diff)
- [v] `--exclude-dynamic-system-prompt-sections` on claude ≥ 2.1.98 (tests, diff)
- [v] Provider-routing env vars unset (`_CLAUDE_ENV_UNSET`) (tests, diff)
- [v] `CLAUDE_CODE_EXTRA_BODY` for sampling and reasoning (tests, diff)
- [v] `--as-subagent`: MCP plugin, plan-mode gate hook, skill, allowed tools, WSL bridge (tests, diff, live; WSL sim)

## codex

- [v] `CODEX_HOME` session, provider table, profile, `--oss --profile` (tests, diff, live)
- [v] Model catalog with the fallback prompt on codex ≥ 0.110 (tests, diff)
- [v] `apply_patch_preserve_line_endings` on codex ≥ 0.148 (tests, diff)
- [v] `model_reasoning_effort` on codex ≥ 0.145 (tests, diff)
- [v] Context window, stream idle timeout (tests, diff)
- [v] `--as-subagent`: MCP bridge, parent overlay of the user's codex home with routing instructions, WSL (tests, diff, live; WSL sim)

## opencode

- [v] OpenCode V1 and V2 (`opencode2`) commands, V2 `--standalone` (tests, diff; V2 live)
- [v] Provider overlay file plus inline `OPENCODE_CONFIG_CONTENT` pin (model, small_model, provider filters) (tests, diff)
- [v] Output limit, compaction reserve, `OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX` (tests, diff)
- [v] Native `--auto` on opencode ≥ 1.17.12, subcommand aware; permission fallback; yolo state reset (tests, diff)
- [v] `--as-subagent`: `agent.<name>` subagent, provider filters merged via `opencode debug config` (tests, diff; not run live)

## pi

- [v] `models.json` provider with context window, max tokens, `samplingParams` on pi ≥ 0.84 (tests, diff, live)
- [v] `HOME` and `PI_CODING_AGENT_DIR` relocation, Windows profile vars (tests, live; Windows sim)
- [v] User resources linked (extensions, skills, prompts, themes, npm, git, `~/.agents/skills`) and settings entries re-anchored (tests)
- [v] Clean screen before launch (tests)
- [v] `--as-subagent`: bundled TS extension and bootstrap config (tests incl. Bun, diff; not run live)

## Beyond `unsloth start`: other model servers

- [v] `--url` / `--provider`; without them, a running Unsloth first, else the one server on the usual Ollama, LM Studio, llama-server or vLLM port (tests)
- [v] Ollama: runtime window from `/api/ps`, `--model` preload, `--context-length` through a reusable `agent-switch/<model>-ctx<N>` alias (tests, live)
- [v] LM Studio: loaded-instance window, `/api/v1/models/load` with `context_length`, v0 fallback (tests only; no LM Studio server was run)
- [v] llama-server: per-slot `meta.n_ctx`, router-mode `/models/load` (tests, live for both modes)
- [v] vLLM: `max_model_len` (tests only; no vLLM server was run)
- [v] Generic OpenAI-compatible server: reported window, else `--context-length` required (tests)
- [v] Endpoint check: claude needs `/v1/messages`, codex needs `/v1/responses` (tests, live)
- [v] Request bodies translated per server; unsupported fields warned and dropped (tests)
- [v] Unsloth-only flags refused for other servers (tests)
- [v] `--header NAME=VALUE` (repeatable): custom HTTP headers on agent-switch's own requests and on every agent's requests, including the subagent bridges; an `Authorization` header here replaces the built-in Bearer `<api-key>` (tests; claude token precedence and pi header merge verified live on claude 2.1.291 and pi 1.0.4); an Unsloth Studio target carries them too: an `Authorization` header there stands in for `--api-key` (no Studio key is minted), while Studio's own API-key listing, minting and identity check and the auto-start server's progress polls never carry them, and probing for an unnamed server never sends them (tests)
- [v] All four agents complete a turn on llama-server; the user's own agent config files are unchanged afterwards (live)

## Not ported (with reason)

- `unsloth_cli._inference.verify_studio_identity` / `_studio_token` / `connect_studio_server` tests: agent-switch calls Unsloth's own implementation through `providers/unsloth_bridge.py`, so those tests stay upstream.
- `unsloth connect` alias test: agent-switch has no `connect` alias.
- hermes, openclaw, dsh and `--app` tests: those features are out of scope for v1.
