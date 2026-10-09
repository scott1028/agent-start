# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the agent-switch authors. See LICENSE.

"""`agent-switch <agent> --url ...` against model servers, end to end through the CLI."""

import json
import shlex
import urllib.request

import pytest
from typer.testing import CliRunner

import agent_switch.start as start
from agent_switch import providers
from agent_switch.providers.types import Target


@pytest.fixture
def cli(tmp_path, monkeypatch):
    monkeypatch.setattr(start, "_agents_config_root", lambda: tmp_path / "agents")
    monkeypatch.setattr(start, "_require_agent_for_launch", lambda *args: None)
    monkeypatch.setattr(start, "_opencode_command", lambda *_: ("opencode", False))
    # No agent binaries: version probes assume a current build, as for a recipe run elsewhere.
    monkeypatch.setattr(start.shutil, "which", lambda _: None)
    monkeypatch.delenv("AGENT_SWITCH_API_KEY", raising = False)

    def invoke(*argv):
        return CliRunner().invoke(start.start_app, list(argv))

    return invoke


def _exports(output: str) -> dict:
    env = {}
    for line in output.splitlines():
        if line.startswith("export "):
            name, _, value = line[len("export "):].partition("=")
            env[name] = shlex.split(value)[0] if value else ""
    return env


def _ollama(server, model = "smollm2:135m", ctx = 8192, messages = True, responses = True):
    server.route("GET", "/api/version", body = {"version": "0.34.3"})
    server.route("GET", "/api/ps", body = {"models": [{"name": model, "model": model, "context_length": ctx}]})
    server.route("GET", "/v1/models", body = {"object": "list", "data": [{"id": model}]})
    if messages:
        server.route("POST", "/v1/messages", status = 400, body = {"error": "model is required"})
    if responses:
        server.route("POST", "/v1/responses", status = 400, body = {"error": "model is required"})


def _vllm(server, ctx = 32768):
    server.route("GET", "/v1/models", body = {"object": "list", "data": [{"id": "Qwen/Qwen3-8B", "owned_by": "vllm", "max_model_len": ctx}]})
    server.route("POST", "/v1/messages", status = 400, body = {"error": "bad request"})
    server.route("POST", "/v1/responses", status = 400, body = {"error": "bad request"})


def test_claude_against_ollama(cli, fake_server):
    _ollama(fake_server)
    result = cli("claude", "--url", fake_server.base, "--no-launch")
    assert result.exit_code == 0, result.output
    env = _exports(result.output)
    assert env["ANTHROPIC_BASE_URL"] == fake_server.base
    assert env["ANTHROPIC_MODEL"] == "smollm2:135m"
    assert env["ANTHROPIC_AUTH_TOKEN"] == providers.NO_KEY
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "8192"
    assert f"Ollama ready at {fake_server.base}" in result.output
    assert "Unsloth" not in result.output


def test_codex_against_vllm(cli, fake_server, tmp_path):
    _vllm(fake_server)
    result = cli("codex", "--url", fake_server.base + "/v1", "--no-launch")
    assert result.exit_code == 0, result.output
    config = (tmp_path / "agents" / "codex" / "config.toml").read_text()
    assert f'base_url = "{fake_server.base}/v1"' in config
    profile = (tmp_path / "agents" / "codex" / f"{start._CODEX_PROFILE}.config.toml").read_text()
    assert 'model = "Qwen/Qwen3-8B"' in profile
    assert "model_context_window = 32768" in profile


def test_opencode_against_llamacpp(cli, fake_server, tmp_path):
    fake_server.route("GET", "/props", body = {"default_generation_settings": {"n_ctx": 4096}})
    fake_server.route("GET", "/v1/models", body = {"data": [{"id": "/m/jan.gguf", "owned_by": "llamacpp", "meta": {"n_ctx": 4096}}]})
    result = cli("opencode", "--url", fake_server.base, "--no-launch")
    assert result.exit_code == 0, result.output
    config = json.loads((tmp_path / "agents" / "opencode" / "opencode.json").read_text())
    provider = config["provider"][start._OPENCODE_PROVIDER]
    assert provider["options"]["baseURL"] == f"{fake_server.base}/v1"
    assert provider["models"]["/m/jan.gguf"]["limit"]["context"] == 4096


def test_pi_against_lmstudio(cli, fake_server, tmp_path):
    fake_server.route(
        "GET",
        "/api/v1/models",
        body = {"models": [{"type": "llm", "key": "qwen/qwen3-8b", "loaded_instances": [{"id": "qwen/qwen3-8b", "config": {"context_length": 16384}}]}]},
    )
    fake_server.route("GET", "/v1/models", body = {"object": "list", "data": [{"id": "qwen/qwen3-8b"}]})
    result = cli("pi", "--url", fake_server.base, "--no-launch")
    assert result.exit_code == 0, result.output
    models = json.loads((tmp_path / "agents" / "pi" / ".pi" / "agent" / "models.json").read_text())
    entry = models["providers"][start._PI_PROVIDER]
    assert entry["baseUrl"] == f"{fake_server.base}/v1"
    assert entry["models"][0]["contextWindow"] == 16384


def test_colon_model_names_are_not_split_into_a_gguf_variant(cli, fake_server):
    _ollama(fake_server)
    result = cli("claude", "--url", fake_server.base, "-m", "smollm2:135m", "--no-launch")
    assert result.exit_code == 0, result.output
    assert _exports(result.output)["ANTHROPIC_MODEL"] == "smollm2:135m"


def test_unsupported_sampling_fields_warn_and_stay_out_of_the_body(cli, fake_server):
    _ollama(fake_server)
    result = cli("claude", "--url", fake_server.base, "--temperature", "0.6", "--top-k", "20", "--no-launch")
    assert result.exit_code == 0, result.output
    assert "Ollama ignores --top-k" in result.output
    assert json.loads(_exports(result.output)["CLAUDE_CODE_EXTRA_BODY"]) == {"temperature": 0.6}


def test_reasoning_off_rides_in_chat_template_kwargs_on_vllm(cli, fake_server):
    _vllm(fake_server)
    result = cli("claude", "--url", fake_server.base, "--reasoning", "off", "--no-launch")
    assert result.exit_code == 0, result.output
    body = json.loads(_exports(result.output)["CLAUDE_CODE_EXTRA_BODY"])
    assert body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_claude_subagent_uses_the_served_model_id(cli, fake_server, tmp_path):
    _ollama(fake_server)
    result = cli("claude", "--url", fake_server.base, "--as-subagent", "--no-launch")
    assert result.exit_code == 0, result.output
    plugin = tmp_path / "agents" / "claude-subagent" / "local-agent" / ".mcp.json"
    env = json.loads(plugin.read_text())["mcpServers"]["local"]["env"]
    assert env["AGENT_SWITCH_CLAUDE_SUBAGENT_MODEL"] == "smollm2:135m"
    assert env["AGENT_SWITCH_CLAUDE_SUBAGENT_BASE_URL"] == fake_server.base
    assert not any("/api/inference" in path for _, path, *_ in fake_server.requests)


def test_claude_needs_the_anthropic_endpoint(cli, fake_server):
    _ollama(fake_server, messages = False)
    result = cli("claude", "--url", fake_server.base, "--no-launch")
    assert result.exit_code == 1
    assert "/v1/messages" in result.output


def test_codex_needs_the_responses_endpoint(cli, fake_server):
    _ollama(fake_server, responses = False)
    result = cli("codex", "--url", fake_server.base, "--no-launch")
    assert result.exit_code == 1
    assert "/v1/responses" in result.output


def test_no_url_uses_the_only_local_server_found(cli, fake_server, monkeypatch):
    _ollama(fake_server)
    monkeypatch.setattr(start.providers, "scan_local_servers", lambda: [Target("ollama", fake_server.base)])
    result = cli("claude", "--no-launch")
    assert result.exit_code == 0, result.output
    assert _exports(result.output)["ANTHROPIC_BASE_URL"] == fake_server.base


def test_no_url_with_several_local_servers_asks_which(cli, monkeypatch):
    monkeypatch.setattr(
        start.providers,
        "scan_local_servers",
        lambda: [Target("ollama", "http://127.0.0.1:11434"), Target("vllm", "http://127.0.0.1:8000")],
    )
    result = cli("claude", "--no-launch")
    assert result.exit_code == 1
    assert "Ollama at http://127.0.0.1:11434" in result.output
    assert "vLLM at http://127.0.0.1:8000" in result.output
    assert "--url" in result.output


@pytest.mark.parametrize("agent", ["claude", "codex", "opencode", "pi", "dsh"])
def test_no_url_without_a_local_server_says_how_to_name_one(cli, monkeypatch, agent):
    # The usual-port scan (empty here) is the only default: no other server is probed.
    monkeypatch.setattr(
        urllib.request.OpenerDirector, "open", lambda *a, **k: pytest.fail("no other server may be probed")
    )
    result = cli(agent, "--no-launch")
    assert result.exit_code == 1
    assert "No model server found" in result.output
    assert "--url" in result.output
    assert "Unsloth" not in result.output


def test_api_key_is_remembered_for_that_server(cli, fake_server):
    fake_server.route("GET", "/v1/models", lambda _: (200, {"data": [{"id": "m", "context_length": 4096}]}))
    first = cli("pi", "--url", fake_server.base, "--api-key", "secret", "--no-launch")
    assert first.exit_code == 0, first.output
    fake_server.requests.clear()
    second = cli("pi", "--url", fake_server.base, "--no-launch")
    assert second.exit_code == 0, second.output
    assert all(auth == "Bearer secret" for _, path, _, auth in fake_server.requests if path == "/v1/models")


# "unsloth" names the removed Studio integration.
@pytest.mark.parametrize("name", ["nope", "unsloth"])
def test_unknown_provider_is_a_usage_error(cli, name):
    result = cli("claude", "--provider", name, "--no-launch")
    assert result.exit_code == 2
