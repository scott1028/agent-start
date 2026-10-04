# NOTICE

agent-switch is licensed under the GNU Affero General Public License v3.0 only (see `LICENSE`).

Most of its code is ported from [Unsloth](https://github.com/unslothai/unsloth),
Copyright 2026-present the Unsloth AI Inc. team, licensed AGPL-3.0-only, at commit
`8e11ba15ef59b324b99ad55406e98b0391dcd7db`.

| agent-switch file | Source in unsloth | Changes |
|---|---|---|
| `agent_switch/start.py` | `unsloth_cli/commands/start.py` | Removed the hermes, openclaw and dsh agents and `--app`; renamed agent-facing ids to neutral names; Unsloth-internal imports go through `providers/unsloth_bridge.py`; `_connect` dispatches to a provider |
| `agent_switch/_inference.py` | `unsloth_cli/_inference.py` | Only the HTTP and server-discovery helpers |
| `agent_switch/claude_subagent_mcp.py` | `unsloth_cli/claude_subagent_mcp.py` | Imports and names |
| `agent_switch/codex_subagent_mcp.py` | `unsloth_cli/codex_subagent_mcp.py` | Imports and names |
| `agent_switch/pi_subagent.ts` | `unsloth_cli/pi_subagent.ts` | Names |
| `tests/test_start.py` and the other ported tests | `unsloth_cli/tests/` | Imports, names, and tests for removed features dropped |

`agent_switch/codex_fallback_prompt.md` is copied (through Unsloth) from
[openai/codex](https://github.com/openai/codex) `rust-v0.144.0`
`codex-rs/models-manager/prompt.md`, Copyright OpenAI, licensed Apache-2.0.
