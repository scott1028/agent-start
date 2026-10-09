# NOTICE

agent-switch is licensed under the GNU Affero General Public License v3.0 only (see `LICENSE`).

Most of its code is ported from [Unsloth](https://github.com/unslothai/unsloth),
Copyright 2026-present the Unsloth AI Inc. team, licensed AGPL-3.0-only, at commit
`8e11ba15ef59b324b99ad55406e98b0391dcd7db`.

| agent-switch file | Source in unsloth | Changes |
|---|---|---|
| `agent_switch/start.py` | `unsloth_cli/commands/start.py` | Removed the hermes and openclaw agents, `--app`, and the Unsloth Studio integration (server discovery, API-key minting, model loading and download progress, GGUF checks, managed Node); renamed agent-facing ids to neutral names; `_connect` goes through the model-server providers; `--header NAME=VALUE` adds custom headers to its own requests; dsh also takes `--header`, `--max-tokens` and `--compact-at`, writes its patch privately, and sends reasoning only to servers that read chat template kwargs |
| `agent_switch/_inference.py` | `unsloth_cli/_inference.py` | Only the User-Agent and the redirect-refusing `urlopen` |
| `agent_switch/_coding_agents.py` | `studio/backend/utils/coding_agents.py` | Only the DeepSeek Harness detection helpers |
| `agent_switch/claude_subagent_mcp.py` | `unsloth_cli/claude_subagent_mcp.py` | Imports and names |
| `agent_switch/codex_subagent_mcp.py` | `unsloth_cli/codex_subagent_mcp.py` | Imports and names |
| `agent_switch/pi_subagent.ts` | `unsloth_cli/pi_subagent.ts` | Names |
| `tests/test_start.py` and the other ported tests | `unsloth_cli/tests/` | Imports, names, and tests for removed features (the Studio integration included) dropped; launch and config tests run against a fake vLLM-shaped server instead of a fake Studio |

`agent_switch/codex_fallback_prompt.md` is copied (through Unsloth) from
[openai/codex](https://github.com/openai/codex) `rust-v0.144.0`
`codex-rs/models-manager/prompt.md`, Copyright OpenAI, licensed Apache-2.0.
