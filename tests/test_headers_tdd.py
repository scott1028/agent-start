# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the agent-switch authors. See LICENSE.

"""`--header NAME=VALUE`: parsing, validation, and injection into each agent's config."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import tomllib
import urllib.error
from pathlib import Path

import pytest
import typer

import agent_switch._inference as inference
import agent_switch.start as start
from agent_switch.providers import utils as provider_utils

BASE = "http://127.0.0.1:8888"
MODEL = {"id": "test/model", "context_length": 32768}


def test_parse_headers_pairs():
    assert start.parse_headers(["X-Foo=bar", "X-Tenant=eng"]) == {"X-Foo": "bar", "X-Tenant": "eng"}


def test_parse_headers_value_keeps_later_equals():
    assert start.parse_headers(["X-Auth=a=b"]) == {"X-Auth": "a=b"}


def test_parse_headers_later_wins_case_insensitively():
    assert start.parse_headers(["X-Foo=one", "x-foo=two"]) == {"x-foo": "two"}


def test_parse_headers_rejects_missing_equals():
    with pytest.raises(typer.Exit):
        start.parse_headers(["Nope"])


def test_parse_headers_rejects_invalid_name():
    for bad in ("X Foo=bar", "=value", "X:Foo=bar"):
        with pytest.raises(typer.Exit):
            start.parse_headers([bad])


def test_parse_headers_rejects_newline_in_value():
    for bad in ("X-Foo=line1\nline2", "X-Foo=line1\r\nline2"):
        with pytest.raises(typer.Exit):
            start.parse_headers([bad])


def test_parse_headers_rejects_non_printable_value():
    # urllib raises UnicodeEncodeError and codex's TOML rejects surrogates, so require visible ASCII (+ tab).
    for bad in ("X-Foo=café", "X-Foo=hi\ud83d\ude00", "X-Foo=hi\x00", "X-Foo=hi\x7f"):
        with pytest.raises(typer.Exit):
            start.parse_headers([bad])


def test_get_has_custom_authorization_case_insensitive():
    assert provider_utils.get_has_custom_authorization({"Authorization": "x"})
    assert provider_utils.get_has_custom_authorization({"authorization": "x"})
    assert not provider_utils.get_has_custom_authorization({"X-Foo": "x"})
    assert not provider_utils.get_has_custom_authorization({})


def test_claude_local_env_custom_headers():
    env = start._claude_local_env(BASE, "sk-test", MODEL, headers = {"X-Foo": "bar", "X-Tenant": "eng"})
    assert env["ANTHROPIC_CUSTOM_HEADERS"] == "X-Foo: bar\nX-Tenant: eng"


def test_claude_local_env_custom_authorization_goes_to_custom_headers():
    env = start._claude_local_env(BASE, "sk-test", MODEL, headers = {"Authorization": "AABBCC"})
    assert env["ANTHROPIC_CUSTOM_HEADERS"] == "Authorization: AABBCC"


def test_claude_local_env_custom_authorization_blanks_auth_token():
    # claude applies settings env after the process env, and ANTHROPIC_AUTH_TOKEN outranks
    # ANTHROPIC_CUSTOM_HEADERS for Authorization (verified on 2.1.291). An empty pin beats an
    # inherited or user-settings token without tripping claude's own auth requirement.
    env = start._claude_local_env(BASE, "sk-test", MODEL, headers = {"Authorization": "AABBCC"})
    assert env["ANTHROPIC_AUTH_TOKEN"] == ""


def test_claude_local_env_without_custom_authorization_keeps_auth_token():
    env = start._claude_local_env(BASE, "sk-test", MODEL, headers = {"X-Foo": "bar"})
    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-test"


def test_claude_local_env_without_headers_has_no_custom_headers():
    env = start._claude_local_env(BASE, "sk-test", MODEL)
    assert "ANTHROPIC_CUSTOM_HEADERS" not in env


def _codex_provider_config(tmp_path, monkeypatch, headers):
    monkeypatch.setattr(start, "_codex_supports_model_catalog", lambda: False)
    start.write_codex_config(BASE, MODEL, tmp_path, headers = headers)
    return tomllib.loads((tmp_path / "config.toml").read_text())["model_providers"]["agent_switch"]


def test_write_codex_config_http_headers(tmp_path, monkeypatch):
    provider = _codex_provider_config(tmp_path, monkeypatch, {"X-Foo": "bar", "X-Tenant": "eng"})
    assert provider["http_headers"] == {"X-Foo": "bar", "X-Tenant": "eng"}
    assert provider["env_key"] == "AGENT_SWITCH_AUTH_TOKEN"


def test_write_codex_config_custom_authorization_omits_env_key(tmp_path, monkeypatch):
    provider = _codex_provider_config(tmp_path, monkeypatch, {"Authorization": "AABBCC"})
    assert provider["http_headers"] == {"Authorization": "AABBCC"}
    assert "env_key" not in provider


def test_write_codex_config_without_headers_has_no_http_headers(tmp_path, monkeypatch):
    provider = _codex_provider_config(tmp_path, monkeypatch, None)
    assert "http_headers" not in provider
    assert provider["env_key"] == "AGENT_SWITCH_AUTH_TOKEN"


def test_opencode_provider_headers():
    provider = start._opencode_provider(BASE, "sk-test", MODEL, headers = {"X-Foo": "bar"})
    assert provider["options"]["headers"] == {"X-Foo": "bar"}
    assert provider["options"]["apiKey"] == "sk-test"


def test_opencode_provider_custom_authorization_omits_api_key():
    provider = start._opencode_provider(BASE, "sk-test", MODEL, headers = {"Authorization": "AABBCC"})
    assert provider["options"]["headers"] == {"Authorization": "AABBCC"}
    assert "apiKey" not in provider["options"]


def test_opencode_provider_without_headers_has_no_headers_option():
    provider = start._opencode_provider(BASE, "sk-test", MODEL)
    assert "headers" not in provider["options"]


def test_write_opencode_config_headers(tmp_path):
    path = tmp_path / "opencode.json"
    start.write_opencode_config(BASE, "sk-test", MODEL, path, headers = {"X-Foo": "bar"})
    config = json.loads(path.read_text())
    assert config["provider"]["agent-switch"]["options"]["headers"] == {"X-Foo": "bar"}


def test_write_pi_config_headers(tmp_path):
    path = tmp_path / "models.json"
    start.write_pi_config(BASE, "sk-unsloth-abc", MODEL, path, headers = {"X-Foo": "bar"})
    provider = json.loads(path.read_text())["providers"]["agent-switch"]
    assert provider["headers"] == {"X-Foo": "bar"}
    assert provider["apiKey"] == "sk-unsloth-abc"


def test_write_pi_config_headers_escape_dollar(tmp_path):
    # Pi interpolates $NAME in header values; $$ emits a literal $, so a literal value must double it.
    path = tmp_path / "models.json"
    start.write_pi_config(BASE, "sk-unsloth-abc", MODEL, path, headers = {"X-Foo": "a$b"})
    provider = json.loads(path.read_text())["providers"]["agent-switch"]
    assert provider["headers"] == {"X-Foo": "a$$b"}


def test_write_pi_config_headers_escape_leading_bang(tmp_path):
    # A value starting with ! runs as a shell command in pi; $! emits a literal !.
    path = tmp_path / "models.json"
    start.write_pi_config(BASE, "sk-unsloth-abc", MODEL, path, headers = {"X-Foo": "!whoami"})
    provider = json.loads(path.read_text())["providers"]["agent-switch"]
    assert provider["headers"] == {"X-Foo": "$!whoami"}


def test_write_pi_config_custom_authorization_keeps_api_key(tmp_path):
    # pi 1.0.4 refuses the prompt outright without an apiKey ("No API key found"); its OpenAI SDK
    # merges the custom Authorization over the Bearer later, so the key stays and one header wins.
    path = tmp_path / "models.json"
    start.write_pi_config(BASE, "sk-unsloth-abc", MODEL, path, headers = {"Authorization": "AABBCC"})
    provider = json.loads(path.read_text())["providers"]["agent-switch"]
    assert provider["headers"] == {"Authorization": "AABBCC"}
    assert provider["apiKey"] == "sk-unsloth-abc"


def test_write_pi_config_without_headers_has_no_headers(tmp_path):
    path = tmp_path / "models.json"
    start.write_pi_config(BASE, "sk-unsloth-abc", MODEL, path)
    provider = json.loads(path.read_text())["providers"]["agent-switch"]
    assert "headers" not in provider
    assert provider["apiKey"] == "sk-unsloth-abc"


def test_write_pi_subagent_config_headers(tmp_path):
    path = tmp_path / "subagent.json"
    start.write_pi_subagent_config(BASE, "sk-unsloth-abc", MODEL, path, headers = {"X-Foo": "a$b"})
    assert json.loads(path.read_text())["headers"] == {"X-Foo": "a$$b"}


def test_write_codex_subagent_bridge_carries_headers(tmp_path, monkeypatch):
    monkeypatch.setattr(start, "_codex_supports_model_catalog", lambda: False)
    start.write_codex_subagent_bridge(BASE, "sk-test", MODEL, tmp_path, yolo = False, headers = {"X-Foo": "bar"})
    config = tomllib.loads((tmp_path / "child" / "config.toml").read_text())
    assert config["model_providers"]["agent_switch"]["http_headers"] == {"X-Foo": "bar"}


def test_write_codex_config_header_values_stay_private(tmp_path, monkeypatch):
    import stat

    monkeypatch.setattr(start, "_codex_supports_model_catalog", lambda: False)
    start.write_codex_config(BASE, MODEL, tmp_path, headers = {"Authorization": "AABBCC"})
    mode = (tmp_path / "config.toml").stat().st_mode
    assert stat.S_IMODE(mode) == 0o600


def test_write_codex_config_headers_are_replaced_on_rerun(tmp_path, monkeypatch):
    # A rerun without --header must drop the stale http_headers line and restore env_key.
    monkeypatch.setattr(start, "_codex_supports_model_catalog", lambda: False)
    start.write_codex_config(BASE, MODEL, tmp_path, headers = {"X-Foo": "bar"})
    start.write_codex_config(BASE, MODEL, tmp_path)
    provider = tomllib.loads((tmp_path / "config.toml").read_text())["model_providers"]["agent_switch"]
    assert "http_headers" not in provider
    assert provider["env_key"] == "AGENT_SWITCH_AUTH_TOKEN"


def _studio_transport(monkeypatch, routes: dict, headers: dict) -> list:
    """Answer Studio endpoints from `routes` ("METHOD /suffix": body, or a status to raise).

    Returns the (url, lowercased headers) of every request, and makes `headers` the target's
    custom --header pairs.
    """
    sent = []

    def urlopen(request, timeout):
        sent.append((request.full_url, {n.lower(): v for n, v in request.headers.items()}))
        path = request.full_url.split("?")[0]
        for route, body in routes.items():
            method, _, suffix = route.partition(" ")
            if request.get_method() == method and path.endswith(suffix):
                if isinstance(body, int):
                    raise urllib.error.HTTPError(request.full_url, body, "Unauthorized", {}, None)
                return io.BytesIO(json.dumps(body).encode())
        raise AssertionError(f"unexpected request: {request.get_method()} {request.full_url}")

    monkeypatch.setattr(start, "urlopen_no_redirect", urlopen)
    monkeypatch.setattr(start, "_active_target", start.Target("unsloth", BASE, headers))
    return sent


def _local_studio_key_harness(monkeypatch, tmp_path) -> None:
    """Let the Studio key flow run against the stub transport: a fresh cache, identity verified."""
    monkeypatch.setattr(start, "_key_cache_path", lambda: tmp_path / "agent_api_key.json")
    monkeypatch.setattr(start, "verify_studio_identity", lambda base: True)
    monkeypatch.setattr(start, "_studio_token", lambda: "owner-jwt")


MINT_ROUTES = {
    "GET /api/auth/api-keys": {"api_keys": []},
    "POST /api/auth/api-keys": {"key": "sk-unsloth-minted"},
    "GET /api/inference/loaded-models": {"object": "list", "data": []},
}


def test_json_body_keeps_its_content_type(monkeypatch):
    sent = _studio_transport(
        monkeypatch, {"POST /api/inference/load": {"status": "loaded"}}, {"content-type": "text/plain"}
    )
    start._http_json("POST", f"{BASE}/api/inference/load", "sk-studio-key", payload = {})
    assert sent[0][1]["content-type"] == "application/json"


def test_unsloth_model_calls_carry_headers(monkeypatch):
    sent = _studio_transport(monkeypatch, {"GET /v1/models": {"data": []}}, {"X-Foo": "bar"})
    start._http_json("GET", f"{BASE}/v1/models", "sk-studio-key")
    assert sent[0][1]["x-foo"] == "bar"
    assert sent[0][1]["authorization"] == "Bearer sk-studio-key"


def test_unsloth_custom_authorization_replaces_bearer(monkeypatch):
    sent = _studio_transport(
        monkeypatch, {"GET /v1/models": {"data": []}}, {"Authorization": "Bearer gateway-token"}
    )
    start._http_json("GET", f"{BASE}/v1/models", "sk-studio-key")
    assert sent[0][1]["authorization"] == "Bearer gateway-token"


def test_unsloth_mint_keeps_the_owner_jwt(monkeypatch, tmp_path):
    # A non-Authorization --header belongs to the model API; listing and minting keys is Studio's
    # own owner handshake, which keeps the owner JWT and never sees the custom pair.
    sent = _studio_transport(monkeypatch, MINT_ROUTES, {"X-Tenant": "eng"})
    _local_studio_key_harness(monkeypatch, tmp_path)
    assert start._agent_api_key(BASE, None) == "sk-unsloth-minted"
    minted = [request_headers for url, request_headers in sent if url.endswith("/api/auth/api-keys")]
    assert len(minted) == 2
    assert all(request_headers["authorization"] == "Bearer owner-jwt" for request_headers in minted)
    assert all("x-tenant" not in request_headers for request_headers in minted)


def test_unsloth_custom_authorization_skips_the_key_flow(monkeypatch):
    # The custom Authorization is what every request sends, so the Studio key cache is never even
    # opened: no replay, no probe, no mint, not even for an --api-key passed alongside it.
    from agent_switch import providers

    _studio_transport(monkeypatch, MINT_ROUTES, {"Authorization": "Bearer gateway"})
    monkeypatch.setattr(
        start, "_key_cache_path", lambda: pytest.fail("must not touch the Studio key cache")
    )
    monkeypatch.setattr(start, "_key_accepted", lambda *args: pytest.fail("must not probe a key it never sends"))
    monkeypatch.setattr(start, "_studio_token", lambda: pytest.fail("must not mint a key it never sends"))
    assert start._agent_api_key(BASE, "sk-explicit-unused") == providers.NO_KEY


def test_unsloth_model_401_names_the_custom_header(monkeypatch, capsys):
    _studio_transport(monkeypatch, {"GET /api/inference/loaded-models": 401}, {"Authorization": "Bearer bad"})
    with pytest.raises(typer.Exit):
        start._loaded_models(BASE, "sk-studio-key")
    assert "--header was rejected" in capsys.readouterr().err


def test_unsloth_model_401_without_a_custom_authorization_stays_silent(monkeypatch, capsys):
    _studio_transport(monkeypatch, {"GET /api/inference/loaded-models": 401}, {"X-Tenant": "eng"})
    with pytest.raises(typer.Exit):
        start._loaded_models(BASE, "sk-studio-key")
    assert "--header" not in capsys.readouterr().err


def test_health_probe_carries_headers_only_for_the_named_base(monkeypatch):
    # A Studio behind a gateway answers /api/health only with the custom pair. The default port and
    # the pid-record bases stay credential-free probes.
    seen = []
    healthy = json.dumps({"service": "Unsloth UI Backend"}).encode()

    def urlopen(request, timeout = None):
        request_headers = {name.lower(): value for name, value in request.headers.items()}
        seen.append(request_headers)
        if request_headers.get("authorization") != "Bearer gateway":
            raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)
        return io.BytesIO(healthy)

    monkeypatch.setattr(start.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(inference, "urlopen_no_redirect", urlopen)
    monkeypatch.setenv("UNSLOTH_STUDIO_URL", BASE)
    assert start.find_studio_server(headers = {"Authorization": "Bearer gateway"}) == BASE
    assert seen[0]["authorization"] == "Bearer gateway"

    seen.clear()
    monkeypatch.delenv("UNSLOTH_STUDIO_URL")
    assert start.find_studio_server(headers = {"Authorization": "Bearer gateway"}) is None
    assert seen
    assert all("authorization" not in request_headers for request_headers in seen)


def test_health_probe_with_headers_refuses_a_redirect(monkeypatch):
    # A gateway 302s to its IdP when the token is stale; the custom credential must not follow it.
    used = []

    def no_redirect(request, timeout):
        used.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, 302, "Found", {}, None)

    monkeypatch.setattr(inference, "urlopen_no_redirect", no_redirect)
    monkeypatch.setenv("UNSLOTH_STUDIO_URL", "https://gateway.example")
    assert start.find_studio_server(headers = {"Authorization": "Bearer gateway"}) is None
    assert used == ["https://gateway.example/api/health"]


def test_require_studio_probes_the_named_base_with_the_custom_headers(monkeypatch):
    seen = []
    monkeypatch.setattr(start, "find_studio_server", lambda **kwargs: seen.append(kwargs) or BASE)
    monkeypatch.setattr(start, "_active_target", start.Target("unsloth", BASE, {"Authorization": "Bearer gateway"}))
    monkeypatch.setenv("UNSLOTH_STUDIO_URL", BASE)
    assert start._require_studio() == BASE
    assert seen[0]["headers"] == {"Authorization": "Bearer gateway"}


def test_named_studio_with_headers_does_not_scan_the_local_ports(monkeypatch):
    # An exported UNSLOTH_STUDIO_URL behind a gateway that did not answer must not hand the gateway
    # token to whatever else listens on a local port; _require_studio reports the named base instead.
    monkeypatch.setenv("UNSLOTH_STUDIO_URL", "https://studio.gw.example")
    monkeypatch.setattr(start, "find_studio_server", lambda **kwargs: None)
    monkeypatch.setattr(start.providers, "scan_local_servers", lambda: pytest.fail("no local scan"))
    target = start._resolve_target(None, None, None, {"Authorization": "Bearer gateway"})
    assert target.name == "unsloth"
    assert target.headers == {"Authorization": "Bearer gateway"}


def test_hub_listing_never_carries_the_custom_headers(monkeypatch):
    # _hub_gguf_files talks to huggingface.co, not the model server; --header must not follow it there.
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    sent = {}
    payload = {"siblings": [{"rfilename": "model.gguf"}]}

    def urlopen(request, timeout = None):
        sent.update({name.lower(): value for name, value in request.headers.items()})
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(start.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(
        start, "_active_target", start.Target("unsloth", BASE, {"Authorization": "Bearer gateway", "X-Tenant": "eng"})
    )
    assert start._hub_gguf_files("unsloth/Repo") == ["model.gguf"]
    assert set(sent) == {"user-agent"}


def test_scanned_target_carries_headers(monkeypatch):
    from agent_switch.providers.types import Target

    monkeypatch.setattr(start, "find_studio_server", lambda *a, **k: None)
    monkeypatch.setattr(
        start.providers, "scan_local_servers", lambda: [Target("vllm", "http://127.0.0.1:8000")]
    )
    target = start._resolve_target(None, None, None, {"X-Foo": "bar"})
    assert target.headers == {"X-Foo": "bar"}


def test_claude_subagent_child_gets_custom_headers(monkeypatch, tmp_path):
    import agent_switch.claude_subagent_mcp as bridge

    captured = {}
    monkeypatch.setenv("AGENT_SWITCH_CLAUDE_SUBAGENT_BASE_URL", BASE)
    monkeypatch.setenv("AGENT_SWITCH_CLAUDE_SUBAGENT_API_KEY", "sk-test")
    monkeypatch.setenv("AGENT_SWITCH_CLAUDE_SUBAGENT_MODEL", "m1")
    monkeypatch.setenv("AGENT_SWITCH_CLAUDE_SUBAGENT_HEADERS", json.dumps({"X-Foo": "bar"}))
    monkeypatch.setenv(bridge._CLAUDE_SUBAGENT_SETTINGS_ENV, str(tmp_path / "settings.json"))
    monkeypatch.setattr(bridge.shutil, "which", lambda _: "/usr/local/bin/claude")
    monkeypatch.setattr(bridge, "_claude_flags", lambda model, settings = None: ["--settings", settings])

    class Process:
        pid = 1
        returncode = 0

        def communicate(self, timeout):
            return json.dumps({"is_error": False, "result": "OK"}), ""

        def poll(self):
            return 0

    def popen(command, **kwargs):
        captured.update(kwargs)
        return Process()

    monkeypatch.setattr(bridge.subprocess, "Popen", popen)
    assert bridge.run_local_agent("ping") == "OK"
    assert captured["env"]["ANTHROPIC_CUSTOM_HEADERS"] == "X-Foo: bar"


def test_claude_subagent_child_custom_authorization_sheds_inherited_token(monkeypatch, tmp_path):
    import agent_switch.claude_subagent_mcp as bridge

    captured = {}
    monkeypatch.setenv("AGENT_SWITCH_CLAUDE_SUBAGENT_BASE_URL", BASE)
    monkeypatch.setenv("AGENT_SWITCH_CLAUDE_SUBAGENT_API_KEY", "sk-test")
    monkeypatch.setenv("AGENT_SWITCH_CLAUDE_SUBAGENT_MODEL", "m1")
    monkeypatch.setenv("AGENT_SWITCH_CLAUDE_SUBAGENT_HEADERS", json.dumps({"Authorization": "AABBCC"}))
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "inherited-token")
    monkeypatch.setenv(bridge._CLAUDE_SUBAGENT_SETTINGS_ENV, str(tmp_path / "settings.json"))
    monkeypatch.setattr(bridge.shutil, "which", lambda _: "/usr/local/bin/claude")
    monkeypatch.setattr(bridge, "_claude_flags", lambda model, settings = None: ["--settings", settings])

    class Process:
        pid = 1
        returncode = 0

        def communicate(self, timeout):
            return json.dumps({"is_error": False, "result": "OK"}), ""

        def poll(self):
            return 0

    def popen(command, **kwargs):
        captured.update(kwargs)
        return Process()

    monkeypatch.setattr(bridge.subprocess, "Popen", popen)
    assert bridge.run_local_agent("ping") == "OK"
    assert captured["env"]["ANTHROPIC_AUTH_TOKEN"] == ""
    assert captured["env"]["ANTHROPIC_CUSTOM_HEADERS"] == "Authorization: AABBCC"


def test_claude_subagent_plugin_settings_carry_headers(tmp_path):
    # The plugin's --settings env is applied after the process env, so it must not re-pin the token
    # and must carry the custom headers.
    plugin = start.write_claude_subagent_plugin(
        tmp_path,
        {
            "AGENT_SWITCH_CLAUDE_SUBAGENT_BASE_URL": BASE,
            "AGENT_SWITCH_CLAUDE_SUBAGENT_API_KEY": "sk-test",
            "AGENT_SWITCH_CLAUDE_SUBAGENT_MODEL": "m1",
            "AGENT_SWITCH_CLAUDE_SUBAGENT_HEADERS": json.dumps({"Authorization": "AABBCC"}),
        },
    )
    settings_file = next(plugin.glob("settings-*.json"))
    settings = json.loads(settings_file.read_text())
    assert settings["env"]["ANTHROPIC_AUTH_TOKEN"] == ""
    assert settings["env"]["ANTHROPIC_CUSTOM_HEADERS"] == "Authorization: AABBCC"


def _run_pi_extension(tmp_path, config_body, assertions):
    bun = shutil.which("bun")
    if bun is None:
        pytest.skip("Bun is required to execute the bundled Pi extension")
    config = tmp_path / "subagent.json"
    config.write_text(json.dumps(config_body), encoding = "utf-8")
    extension = Path(__file__).parents[1] / "agent_switch" / "pi_subagent.ts"
    test_file = tmp_path / "pi-headers.test.ts"
    test_file.write_text(
        f"""
import {{ expect, mock, test }} from "bun:test";
import {{ pathToFileURL }} from "node:url";

mock.module("typebox", () => ({{
    Type: {{
        Object: (value) => value,
        String: (value) => value,
        Optional: (value) => value,
        Array: (value) => value,
    }},
}}));

test("the extension registers the configured headers", async () => {{
    process.env.AGENT_SWITCH_PI_SUBAGENT_CONFIG = {str(config)!r};
    const loaded = await import(pathToFileURL({str(extension)!r}).href);
    let provider;
    loaded.default({{
        registerProvider(_name, value) {{ provider = value; }},
        registerTool() {{}},
    }});
{assertions}
}});
""",
        encoding = "utf-8",
    )
    completed = subprocess.run([bun, "test", str(test_file)], capture_output = True, text = True, timeout = 15)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_pi_subagent_extension_registers_custom_headers(tmp_path):
    _run_pi_extension(
        tmp_path,
        {
            "baseUrl": "http://127.0.0.1:8000/v1",
            "apiKey": "private-token",
            "model": "local-model",
            "headers": {"X-Foo": "bar"},
        },
        """    expect(provider.headers).toEqual({ "X-Foo": "bar" });
    expect(provider.apiKey).toBe("private-token");
    expect(provider.authHeader).toBe(true);""",
    )


def test_pi_subagent_extension_custom_authorization_replaces_bearer(tmp_path):
    _run_pi_extension(
        tmp_path,
        {
            "baseUrl": "http://127.0.0.1:8000/v1",
            "apiKey": "private-token",
            "model": "local-model",
            "headers": {"Authorization": "AABBCC"},
        },
        """    expect(provider.headers).toEqual({ Authorization: "AABBCC" });
    expect(provider.apiKey).toBe("private-token");
    expect(provider.authHeader).toBeFalsy();""",
    )
