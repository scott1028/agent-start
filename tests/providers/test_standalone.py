# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the agent-switch authors. See LICENSE.

"""Standalone invariants: legacy Unsloth inputs neither authenticate nor route, and no vendor API is used."""

from agent_switch.providers.types import Target
from test_cli_providers import _vllm, cli  # noqa: F401  (fixture import)
from agent_switch import providers


def test_legacy_api_key_env_is_not_sent(cli, fake_server, monkeypatch):
    _vllm(fake_server)
    monkeypatch.setenv("UNSLOTH_API_KEY", "sk-legacy-secret")
    result = cli("claude", "--url", fake_server.base, "--no-launch")
    assert result.exit_code == 0, result.output
    assert "sk-legacy-secret" not in result.output
    assert all(auth != "Bearer sk-legacy-secret" for *_, auth in fake_server.requests)


def test_legacy_studio_url_env_does_not_route(cli, fake_server, monkeypatch):
    # With --header, a named UNSLOTH_STUDIO_URL used to pin the session to it instead of the scan.
    _vllm(fake_server)
    monkeypatch.setenv("UNSLOTH_STUDIO_URL", "http://127.0.0.1:9")
    monkeypatch.setattr(providers, "scan_local_servers", lambda: [Target("vllm", fake_server.base)])
    result = cli("claude", "--header", "X-Tenant=eng", "--no-launch")
    assert result.exit_code == 0, result.output
    assert f"vLLM ready at {fake_server.base}" in result.output


def test_url_server_is_used_through_its_openai_api_only(cli, fake_server):
    # A server that also answers a vendor health route is still a generic OpenAI-compatible server:
    # no vendor key minting or inference routes.
    fake_server.route("GET", "/api/health", body = {"status": "healthy", "service": "Unsloth UI Backend"})
    fake_server.route("GET", "/v1/models", body = {"object": "list", "data": [{"id": "org/model", "context_length": 8192}]})
    fake_server.route("POST", "/v1/responses", status = 400, body = {"error": "bad request"})
    result = cli("codex", "--url", fake_server.base, "--no-launch")
    assert result.exit_code == 0, result.output
    assert f"OpenAI-compatible server ready at {fake_server.base}" in result.output
    assert not any(path.startswith(("/api/auth", "/api/inference")) for _, path, *_ in fake_server.requests)