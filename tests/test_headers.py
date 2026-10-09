# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the agent-switch authors. See LICENSE.

"""`--header NAME=VALUE`: parsing, validation, and injection into each agent's config."""

from __future__ import annotations

import json
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
import typer

from agent_switch.providers import utils as provider_utils
from agent_switch import providers
from agent_switch.core import launch as core_launch, options as core_options
from agent_switch.agents import claude as claude_agent, codex as codex_agent, opencode as opencode_agent, pi as pi_agent
from tests.start_split import set_start_attr

BASE = "http://127.0.0.1:8888"
MODEL = {"id": "test/model", "context_length": 32768}


def test_parse_headers_pairs():
    assert core_options.parse_headers(["X-Foo=bar", "X-Tenant=eng"]) == {"X-Foo": "bar", "X-Tenant": "eng"}


def test_parse_headers_value_keeps_later_equals():
    assert core_options.parse_headers(["X-Auth=a=b"]) == {"X-Auth": "a=b"}


def test_parse_headers_later_wins_case_insensitively():
    assert core_options.parse_headers(["X-Foo=one", "x-foo=two"]) == {"x-foo": "two"}


def test_parse_headers_rejects_missing_equals():
    with pytest.raises(typer.Exit):
        core_options.parse_headers(["Nope"])


def test_parse_headers_rejects_invalid_name():
    for bad in ("X Foo=bar", "=value", "X:Foo=bar"):
        with pytest.raises(typer.Exit):
            core_options.parse_headers([bad])


def test_parse_headers_rejects_newline_in_value():
    for bad in ("X-Foo=line1\nline2", "X-Foo=line1\r\nline2"):
        with pytest.raises(typer.Exit):
            core_options.parse_headers([bad])


def test_parse_headers_rejects_non_printable_value():
    # urllib raises UnicodeEncodeError and codex's TOML rejects surrogates, so require visible ASCII (+ tab).
    for bad in ("X-Foo=café", "X-Foo=hi\ud83d\ude00", "X-Foo=hi\x00", "X-Foo=hi\x7f"):
        with pytest.raises(typer.Exit):
            core_options.parse_headers([bad])


def test_get_has_custom_authorization_case_insensitive():
    assert provider_utils.get_has_custom_authorization({"Authorization": "x"})
    assert provider_utils.get_has_custom_authorization({"authorization": "x"})
    assert not provider_utils.get_has_custom_authorization({"X-Foo": "x"})
    assert not provider_utils.get_has_custom_authorization({})


def test_claude_local_env_custom_headers():
    env = claude_agent._claude_local_env(BASE, "sk-test", MODEL, headers = {"X-Foo": "bar", "X-Tenant": "eng"})
    assert env["ANTHROPIC_CUSTOM_HEADERS"] == "X-Foo: bar\nX-Tenant: eng"


def test_claude_local_env_custom_authorization_goes_to_custom_headers():
    env = claude_agent._claude_local_env(BASE, "sk-test", MODEL, headers = {"Authorization": "AABBCC"})
    assert env["ANTHROPIC_CUSTOM_HEADERS"] == "Authorization: AABBCC"


def test_claude_local_env_custom_authorization_blanks_auth_token():
    # claude applies settings env after the process env, and ANTHROPIC_AUTH_TOKEN outranks
    # ANTHROPIC_CUSTOM_HEADERS for Authorization (verified on 2.1.291). An empty pin beats an
    # inherited or user-settings token without tripping claude's own auth requirement.
    env = claude_agent._claude_local_env(BASE, "sk-test", MODEL, headers = {"Authorization": "AABBCC"})
    assert env["ANTHROPIC_AUTH_TOKEN"] == ""


def test_claude_local_env_without_custom_authorization_keeps_auth_token():
    env = claude_agent._claude_local_env(BASE, "sk-test", MODEL, headers = {"X-Foo": "bar"})
    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-test"


def test_claude_local_env_without_headers_has_no_custom_headers():
    env = claude_agent._claude_local_env(BASE, "sk-test", MODEL)
    assert "ANTHROPIC_CUSTOM_HEADERS" not in env


def _codex_provider_config(tmp_path, monkeypatch, headers):
    set_start_attr(monkeypatch, "_codex_supports_model_catalog", lambda: False)
    codex_agent.write_codex_config(BASE, MODEL, tmp_path, headers = headers)
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
    provider = opencode_agent._opencode_provider(BASE, "sk-test", MODEL, headers = {"X-Foo": "bar"})
    assert provider["options"]["headers"] == {"X-Foo": "bar"}
    assert provider["options"]["apiKey"] == "sk-test"


def test_opencode_provider_custom_authorization_omits_api_key():
    provider = opencode_agent._opencode_provider(BASE, "sk-test", MODEL, headers = {"Authorization": "AABBCC"})
    assert provider["options"]["headers"] == {"Authorization": "AABBCC"}
    assert "apiKey" not in provider["options"]


def test_opencode_provider_without_headers_has_no_headers_option():
    provider = opencode_agent._opencode_provider(BASE, "sk-test", MODEL)
    assert "headers" not in provider["options"]


def test_write_opencode_config_headers(tmp_path):
    path = tmp_path / "opencode.json"
    opencode_agent.write_opencode_config(BASE, "sk-test", MODEL, path, headers = {"X-Foo": "bar"})
    config = json.loads(path.read_text())
    assert config["provider"]["agent-switch"]["options"]["headers"] == {"X-Foo": "bar"}


def test_write_pi_config_headers(tmp_path):
    path = tmp_path / "models.json"
    pi_agent.write_pi_config(BASE, "sk-test-abc", MODEL, path, headers = {"X-Foo": "bar"})
    provider = json.loads(path.read_text())["providers"]["agent-switch"]
    assert provider["headers"] == {"X-Foo": "bar"}
    assert provider["apiKey"] == "sk-test-abc"


def test_write_pi_config_headers_escape_dollar(tmp_path):
    # Pi interpolates $NAME in header values; $$ emits a literal $, so a literal value must double it.
    path = tmp_path / "models.json"
    pi_agent.write_pi_config(BASE, "sk-test-abc", MODEL, path, headers = {"X-Foo": "a$b"})
    provider = json.loads(path.read_text())["providers"]["agent-switch"]
    assert provider["headers"] == {"X-Foo": "a$$b"}


def test_write_pi_config_headers_escape_leading_bang(tmp_path):
    # A value starting with ! runs as a shell command in pi; $! emits a literal !.
    path = tmp_path / "models.json"
    pi_agent.write_pi_config(BASE, "sk-test-abc", MODEL, path, headers = {"X-Foo": "!whoami"})
    provider = json.loads(path.read_text())["providers"]["agent-switch"]
    assert provider["headers"] == {"X-Foo": "$!whoami"}


def test_write_pi_config_custom_authorization_keeps_api_key(tmp_path):
    # pi 1.0.4 refuses the prompt outright without an apiKey ("No API key found"); its OpenAI SDK
    # merges the custom Authorization over the Bearer later, so the key stays and one header wins.
    path = tmp_path / "models.json"
    pi_agent.write_pi_config(BASE, "sk-test-abc", MODEL, path, headers = {"Authorization": "AABBCC"})
    provider = json.loads(path.read_text())["providers"]["agent-switch"]
    assert provider["headers"] == {"Authorization": "AABBCC"}
    assert provider["apiKey"] == "sk-test-abc"


def test_write_pi_config_without_headers_has_no_headers(tmp_path):
    path = tmp_path / "models.json"
    pi_agent.write_pi_config(BASE, "sk-test-abc", MODEL, path)
    provider = json.loads(path.read_text())["providers"]["agent-switch"]
    assert "headers" not in provider
    assert provider["apiKey"] == "sk-test-abc"


def test_write_pi_subagent_config_headers(tmp_path):
    path = tmp_path / "subagent.json"
    pi_agent.write_pi_subagent_config(BASE, "sk-test-abc", MODEL, path, headers = {"X-Foo": "a$b"})
    assert json.loads(path.read_text())["headers"] == {"X-Foo": "a$$b"}


def test_write_codex_subagent_bridge_carries_headers(tmp_path, monkeypatch):
    set_start_attr(monkeypatch, "_codex_supports_model_catalog", lambda: False)
    codex_agent.write_codex_subagent_bridge(BASE, "sk-test", MODEL, tmp_path, yolo = False, headers = {"X-Foo": "bar"})
    config = tomllib.loads((tmp_path / "child" / "config.toml").read_text())
    assert config["model_providers"]["agent_switch"]["http_headers"] == {"X-Foo": "bar"}


def test_write_codex_config_header_values_stay_private(tmp_path, monkeypatch):
    import stat

    set_start_attr(monkeypatch, "_codex_supports_model_catalog", lambda: False)
    codex_agent.write_codex_config(BASE, MODEL, tmp_path, headers = {"Authorization": "AABBCC"})
    mode = (tmp_path / "config.toml").stat().st_mode
    assert stat.S_IMODE(mode) == 0o600


def test_write_codex_config_headers_are_replaced_on_rerun(tmp_path, monkeypatch):
    # A rerun without --header must drop the stale http_headers line and restore env_key.
    set_start_attr(monkeypatch, "_codex_supports_model_catalog", lambda: False)
    codex_agent.write_codex_config(BASE, MODEL, tmp_path, headers = {"X-Foo": "bar"})
    codex_agent.write_codex_config(BASE, MODEL, tmp_path)
    provider = tomllib.loads((tmp_path / "config.toml").read_text())["model_providers"]["agent_switch"]
    assert "http_headers" not in provider
    assert provider["env_key"] == "AGENT_SWITCH_AUTH_TOKEN"


def test_scanned_target_carries_headers(monkeypatch):
    from agent_switch.providers.types import Target

    monkeypatch.setattr(
        providers, "scan_local_servers", lambda: [Target("vllm", "http://127.0.0.1:8000")]
    )
    target = core_launch._resolve_target(None, None, None, {"X-Foo": "bar"})
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
    plugin = claude_agent.write_claude_subagent_plugin(
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
    extension = Path(__file__).parents[1] / "agent_switch" / "agents" / "pi_subagent.ts"
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