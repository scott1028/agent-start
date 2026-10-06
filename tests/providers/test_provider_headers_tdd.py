# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the agent-switch authors. See LICENSE.

"""`--header NAME=VALUE` on agent-switch's own requests to the model server."""

import json
import shlex

import pytest
from typer.testing import CliRunner

import agent_switch.start as start
from agent_switch import providers
from agent_switch.providers.utils import request_json, require_json


@pytest.fixture
def cli(tmp_path, monkeypatch):
    """Same isolation as test_cli_providers_tdd.py's fixture, kept local to this TDD file."""
    monkeypatch.setattr(start, "_agents_config_root", lambda: tmp_path / "agents")
    monkeypatch.setattr(start, "_key_cache_path", lambda: tmp_path / "agent_api_key.json")
    monkeypatch.setattr(start, "_require_agent_for_launch", lambda *args: None)
    monkeypatch.setattr(start, "_opencode_command", lambda *_: ("opencode", False))
    monkeypatch.setattr(start.shutil, "which", lambda _: None)
    monkeypatch.setattr(start, "find_studio_server", lambda *a, **k: pytest.fail("no Unsloth lookup"))
    monkeypatch.delenv("UNSLOTH_API_KEY", raising = False)
    monkeypatch.delenv("AGENT_SWITCH_API_KEY", raising = False)

    def invoke(*argv):
        return CliRunner().invoke(start.start_app, list(argv))

    return invoke


def _exports(output):
    env = {}
    for line in output.splitlines():
        if line.startswith("export "):
            name, _, value = line[len("export "):].partition("=")
            env[name] = shlex.split(value)[0] if value else ""
    return env


def _ollama(server, model = "smollm2:135m", ctx = 8192, responses = False):
    server.route("GET", "/api/version", body = {"version": "0.34.3"})
    server.route("GET", "/api/ps", body = {"models": [{"name": model, "model": model, "context_length": ctx}]})
    server.route("GET", "/v1/models", body = {"object": "list", "data": [{"id": model}]})
    server.route("POST", "/v1/messages", status = 400, body = {"error": "model is required"})
    if responses:
        server.route("POST", "/v1/responses", status = 400, body = {"error": "model is required"})


def _last_headers(server, path):
    for _, logged_path, headers in reversed(server.header_logs):
        if logged_path == path:
            return headers
    raise AssertionError(f"no request to {path} recorded")


def test_request_json_sends_custom_headers(fake_server):
    fake_server.route("GET", "/v1/models", 200, {"data": []})
    status, _ = request_json(
        "GET", f"{fake_server.base}/v1/models", "sk-test", headers = {"X-Foo": "bar", "X-Tenant": "eng"}
    )
    assert status == 200
    headers = _last_headers(fake_server, "/v1/models")
    assert headers["X-Foo"] == "bar"
    assert headers["X-Tenant"] == "eng"
    assert headers["Authorization"] == "Bearer sk-test"


def test_request_json_custom_authorization_replaces_bearer(fake_server):
    fake_server.route("GET", "/v1/models", 200, {"data": []})
    status, _ = request_json(
        "GET", f"{fake_server.base}/v1/models", "sk-test", headers = {"Authorization": "AABBCC"}
    )
    assert status == 200
    assert _last_headers(fake_server, "/v1/models")["Authorization"] == "AABBCC"


def test_detect_and_connect_carry_headers(fake_server):
    fake_server.route("GET", "/v1/models", 200, {"object": "list", "data": [{"id": "m1", "context_length": 8192}]})
    target = providers.resolve_target(fake_server.base, "openai", None, {"X-Foo": "bar"})
    base, key, entry = providers.connect(target, None, None, None)
    assert entry["id"] == "m1"
    assert _last_headers(fake_server, "/v1/models")["X-Foo"] == "bar"


def test_claude_cli_header_reaches_discovery_and_env(cli, fake_server):
    _ollama(fake_server)
    result = cli("claude", "--url", fake_server.base, "--header", "X-Foo=bar", "--no-launch")
    assert result.exit_code == 0, result.output
    assert fake_server.header_logs and all(h.get("X-Foo") == "bar" for _, _, h in fake_server.header_logs)
    assert _exports(result.output)["ANTHROPIC_CUSTOM_HEADERS"] == "X-Foo: bar"


def test_codex_cli_custom_authorization_replaces_bearer(cli, fake_server, tmp_path):
    _ollama(fake_server, responses = True)
    result = cli("codex", "--url", fake_server.base, "--header", "Authorization=AABBCC", "--no-launch")
    assert result.exit_code == 0, result.output
    auth_values = [h.get("Authorization") for _, _, h in fake_server.header_logs]
    assert auth_values and all(value == "AABBCC" for value in auth_values)
    config = (tmp_path / "agents" / "codex" / "config.toml").read_text()
    assert 'http_headers = { "Authorization" = "AABBCC" }' in config
    assert "env_key" not in config


def test_opencode_cli_header_lands_in_config(cli, fake_server, tmp_path):
    fake_server.route("GET", "/props", body = {"default_generation_settings": {"n_ctx": 4096}})
    fake_server.route("GET", "/v1/models", body = {"data": [{"id": "/m/jan.gguf", "owned_by": "llamacpp", "meta": {"n_ctx": 4096}}]})
    result = cli("opencode", "--url", fake_server.base, "--header", "X-Foo=bar", "--no-launch")
    assert result.exit_code == 0, result.output
    config = json.loads((tmp_path / "agents" / "opencode" / "opencode.json").read_text())
    assert config["provider"][start._OPENCODE_PROVIDER]["options"]["headers"] == {"X-Foo": "bar"}


def test_pi_cli_header_lands_in_models_json(cli, fake_server, tmp_path):
    fake_server.route(
        "GET",
        "/api/v1/models",
        body = {"models": [{"type": "llm", "key": "qwen/qwen3-8b", "loaded_instances": [{"id": "qwen/qwen3-8b", "config": {"context_length": 16384}}]}]},
    )
    fake_server.route("GET", "/v1/models", body = {"object": "list", "data": [{"id": "qwen/qwen3-8b"}]})
    result = cli("pi", "--url", fake_server.base, "--header", "X-Foo=bar", "--no-launch")
    assert result.exit_code == 0, result.output
    models = json.loads((tmp_path / "agents" / "pi" / ".pi" / "agent" / "models.json").read_text())
    assert models["providers"][start._PI_PROVIDER]["headers"] == {"X-Foo": "bar"}


def test_invalid_header_pair_fails_before_contacting_the_server(cli, fake_server):
    result = cli("claude", "--url", fake_server.base, "--header", "Nope")
    assert result.exit_code == 1
    assert "NAME=VALUE" in result.output
    assert not fake_server.header_logs


def test_claude_cli_custom_authorization_blanks_the_token_env(cli, fake_server):
    _ollama(fake_server)
    result = cli("claude", "--url", fake_server.base, "--header", "Authorization=AABBCC", "--no-launch")
    assert result.exit_code == 0, result.output
    assert _exports(result.output)["ANTHROPIC_AUTH_TOKEN"] == ""
    assert _exports(result.output)["ANTHROPIC_CUSTOM_HEADERS"] == "Authorization: AABBCC"


def test_require_json_401_with_custom_authorization_names_the_header(fake_server):
    fake_server.route("GET", "/v1/models", 401, {"error": "no"})
    with pytest.raises(providers.ProviderError) as raised:
        require_json("Srv", fake_server.base, "/v1/models", "k", {"Authorization": "AABBCC"})
    assert "--header" in str(raised.value)
    assert "--api-key" not in str(raised.value)
