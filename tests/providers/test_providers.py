# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the agent-switch authors. See LICENSE.

"""Provider detection, model resolution and request-body translation against fake servers.

Response shapes mirror what each real server returned when probed: llama-server b11160 (single
and router mode), Ollama 0.34, and the documented LM Studio v1 and vLLM listings.
"""

import pytest

from agent_switch import providers
from agent_switch.providers.types import ProviderError, Target


def _ollama(server, ps = (), tags = ("smollm2:135m",)):
    server.route("GET", "/api/version", body = {"version": "0.34.3"})
    server.route("GET", "/api/ps", body = {"models": [{"name": n, "model": n, "context_length": c} for n, c in ps]})
    server.route(
        "GET",
        "/v1/models",
        body = {"object": "list", "data": [{"id": t, "object": "model", "owned_by": "library"} for t in tags] or None},
    )


def _llamacpp_single(server, n_ctx = 4096):
    server.route("GET", "/props", body = {"default_generation_settings": {"n_ctx": n_ctx}, "total_slots": 2})
    server.route(
        "GET",
        "/v1/models",
        body = {"object": "list", "data": [{"id": "/models/jan.gguf", "owned_by": "llamacpp", "meta": {"n_ctx": n_ctx, "n_ctx_train": 262144}}]},
    )


def _vllm(server, max_model_len = 32768):
    server.route(
        "GET",
        "/v1/models",
        body = {"object": "list", "data": [{"id": "Qwen/Qwen3-8B", "owned_by": "vllm", "max_model_len": max_model_len}]},
    )


# ── detection ──


def test_detect_ollama(fake_server):
    _ollama(fake_server)
    assert providers.detect(fake_server.base) == "ollama"


def test_detect_lmstudio_v1(fake_server):
    fake_server.route("GET", "/api/v1/models", body = {"models": [{"type": "llm", "key": "qwen/qwen3-8b", "loaded_instances": []}]})
    fake_server.route("GET", "/v1/models", body = {"object": "list", "data": []})
    assert providers.detect(fake_server.base) == "lmstudio"


def test_detect_llamacpp(fake_server):
    _llamacpp_single(fake_server)
    assert providers.detect(fake_server.base) == "llamacpp"


def test_detect_vllm(fake_server):
    _vllm(fake_server)
    assert providers.detect(fake_server.base) == "vllm"


def test_detect_generic_openai(fake_server):
    fake_server.route("GET", "/v1/models", body = {"object": "list", "data": [{"id": "m"}]})
    assert providers.detect(fake_server.base) == "openai"


def test_detect_generic_openai_behind_auth(fake_server):
    fake_server.route("GET", "/v1/models", status = 401, body = {"error": "unauthorized"})
    assert providers.detect(fake_server.base) == "openai"


def test_detect_nothing_listening():
    assert providers.detect("http://127.0.0.1:9") is None


# ── target resolution ──


def test_resolve_target_strips_v1_and_detects(fake_server):
    _vllm(fake_server)
    assert providers.resolve_target(fake_server.base + "/v1/", None) == Target("vllm", fake_server.base)


def test_resolve_target_trusts_an_explicit_provider(fake_server):
    assert providers.resolve_target(fake_server.base, "ollama") == Target("ollama", fake_server.base)


def test_resolve_target_uses_the_provider_default_port():
    assert providers.resolve_target(None, "ollama") == Target("ollama", "http://127.0.0.1:11434")


def test_resolve_target_generic_openai_needs_a_url():
    with pytest.raises(ProviderError, match = "--url"):
        providers.resolve_target(None, "openai")


def test_resolve_target_unreachable_url_fails():
    with pytest.raises(ProviderError, match = "Couldn't reach"):
        providers.resolve_target("http://127.0.0.1:9", None)


# ── connect: fixed servers ──


def test_vllm_uses_the_only_served_model_and_its_window(fake_server):
    _vllm(fake_server, 40960)
    base, key, entry = providers.connect(Target("vllm", fake_server.base), None, None, None)
    assert base == fake_server.base
    assert key
    assert entry == {"id": "Qwen/Qwen3-8B", "context_length": 40960}


def test_vllm_cannot_switch_models(fake_server):
    _vllm(fake_server)
    with pytest.raises(ProviderError, match = "Qwen/Qwen3-8B"):
        providers.connect(Target("vllm", fake_server.base), None, "other/model", None)


def test_fixed_server_context_length_caps_but_cannot_exceed(fake_server):
    _vllm(fake_server, 32768)
    _, _, entry = providers.connect(Target("vllm", fake_server.base), None, None, 16384)
    assert entry["context_length"] == 16384
    with pytest.raises(ProviderError, match = "32,768"):
        providers.connect(Target("vllm", fake_server.base), None, None, 65536)


def test_generic_server_without_a_window_needs_context_length(fake_server):
    fake_server.route("GET", "/v1/models", body = {"object": "list", "data": [{"id": "m"}]})
    with pytest.raises(ProviderError, match = "--context-length"):
        providers.connect(Target("openai", fake_server.base), None, None, None)
    _, _, entry = providers.connect(Target("openai", fake_server.base), None, None, 8192)
    assert entry == {"id": "m", "context_length": 8192}


def test_generic_server_reads_a_reported_window(fake_server):
    fake_server.route("GET", "/v1/models", body = {"object": "list", "data": [{"id": "m", "context_length": 65536}]})
    _, _, entry = providers.connect(Target("openai", fake_server.base), None, None, None)
    assert entry["context_length"] == 65536


def test_server_that_needs_a_key_says_so(fake_server):
    fake_server.route("GET", "/v1/models", status = 401, body = {"error": "unauthorized"})
    with pytest.raises(ProviderError, match = "--api-key"):
        providers.connect(Target("openai", fake_server.base), None, None, 4096)


def test_explicit_key_is_sent_and_returned(fake_server):
    fake_server.route("GET", "/v1/models", body = {"object": "list", "data": [{"id": "m"}]})
    _, key, _ = providers.connect(Target("openai", fake_server.base), "secret", None, 4096)
    assert key == "secret"
    assert ("GET", "/v1/models", None, "Bearer secret") in fake_server.requests


def test_several_models_need_a_choice(fake_server):
    fake_server.route("GET", "/v1/models", body = {"object": "list", "data": [{"id": "a"}, {"id": "b"}]})
    with pytest.raises(ProviderError, match = "a, b"):
        providers.connect(Target("openai", fake_server.base), None, None, 4096)


# ── connect: llama-server ──


def test_llamacpp_single_model_reads_the_per_slot_window(fake_server):
    _llamacpp_single(fake_server, 4096)
    _, _, entry = providers.connect(Target("llamacpp", fake_server.base), None, None, None)
    assert entry == {"id": "/models/jan.gguf", "context_length": 4096}


def test_llamacpp_router_loads_the_requested_model(fake_server):
    state = {"loaded": False}

    def models(_):
        status = "loaded" if state["loaded"] else "unloaded"
        meta = {"n_ctx": 8192} if state["loaded"] else None
        return 200, {"data": [{"id": "jan-4b", "owned_by": "llamacpp", "status": {"value": status}, "meta": meta}]}

    def load(payload):
        assert payload == {"model": "jan-4b"}
        state["loaded"] = True
        return 200, {"success": True}

    fake_server.route("GET", "/props", body = {"role": "router", "default_generation_settings": {"n_ctx": 0}})
    fake_server.route("GET", "/v1/models", models)
    fake_server.route("POST", "/models/load", load)
    _, _, entry = providers.connect(Target("llamacpp", fake_server.base), None, "jan-4b", None)
    assert entry == {"id": "jan-4b", "context_length": 8192}


# ── connect: Ollama ──


def test_ollama_uses_the_loaded_model_and_its_runtime_window(fake_server):
    _ollama(fake_server, ps = [("smollm2:135m", 3072)])
    _, _, entry = providers.connect(Target("ollama", fake_server.base), None, None, None)
    assert entry == {"id": "smollm2:135m", "context_length": 3072}


def test_ollama_preloads_a_requested_model(fake_server):
    state = {"ps": []}

    def ps(_):
        return 200, {"models": [{"name": n, "model": n, "context_length": c} for n, c in state["ps"]]}

    def generate(payload):
        assert payload["model"] == "smollm2:135m" and "options" not in payload
        state["ps"] = [("smollm2:135m", 4096)]
        return 200, {"done": True, "done_reason": "load"}

    _ollama(fake_server)
    fake_server.route("GET", "/api/ps", ps)
    fake_server.route("POST", "/api/generate", generate)
    _, _, entry = providers.connect(Target("ollama", fake_server.base), None, "smollm2:135m", None)
    assert entry == {"id": "smollm2:135m", "context_length": 4096}


def test_ollama_context_length_goes_through_a_num_ctx_alias(fake_server):
    state = {"ps": [], "created": None}
    alias = "agent-switch/smollm2-135m-ctx8192:latest"

    def ps(_):
        return 200, {"models": [{"name": n, "model": n, "context_length": c} for n, c in state["ps"]]}

    def create(payload):
        state["created"] = payload
        return 200, {"status": "success"}

    def generate(payload):
        assert payload["model"] == alias
        state["ps"] = [(alias, 8192)]
        return 200, {"done": True}

    _ollama(fake_server)
    fake_server.route("GET", "/api/ps", ps)
    fake_server.route("POST", "/api/create", create)
    fake_server.route("POST", "/api/generate", generate)
    _, _, entry = providers.connect(Target("ollama", fake_server.base), None, "smollm2:135m", 8192)
    assert state["created"] == {
        "model": alias,
        "from": "smollm2:135m",
        "parameters": {"num_ctx": 8192},
        "stream": False,
    }
    assert entry == {"id": alias, "context_length": 8192}


def test_ollama_missing_model_points_at_pull(fake_server):
    _ollama(fake_server)
    fake_server.route("POST", "/api/generate", status = 404, body = {"error": "model 'nope:latest' not found"})
    with pytest.raises(ProviderError, match = "ollama pull nope"):
        providers.connect(Target("ollama", fake_server.base), None, "nope", None)


def test_ollama_nothing_loaded_needs_a_model(fake_server):
    _ollama(fake_server)
    with pytest.raises(ProviderError, match = "--model"):
        providers.connect(Target("ollama", fake_server.base), None, None, None)


# ── connect: LM Studio ──


def test_lmstudio_uses_the_loaded_instance_window(fake_server):
    fake_server.route(
        "GET",
        "/api/v1/models",
        body = {"models": [{"type": "llm", "key": "qwen/qwen3-8b", "max_context_length": 40960,
                            "loaded_instances": [{"id": "qwen/qwen3-8b", "config": {"context_length": 16384}}]}]},
    )
    fake_server.route("GET", "/v1/models", body = {"object": "list", "data": [{"id": "qwen/qwen3-8b"}]})
    _, _, entry = providers.connect(Target("lmstudio", fake_server.base), None, None, None)
    assert entry == {"id": "qwen/qwen3-8b", "context_length": 16384}


def test_lmstudio_loads_with_the_requested_context(fake_server):
    state = {"instances": []}

    def models(_):
        return 200, {"models": [{"type": "llm", "key": "qwen/qwen3-8b", "max_context_length": 40960,
                                 "loaded_instances": state["instances"]}]}

    def load(payload):
        assert payload == {"model": "qwen/qwen3-8b", "context_length": 32768}
        state["instances"] = [{"id": "qwen/qwen3-8b", "config": {"context_length": 32768}}]
        return 200, {"instance_id": "qwen/qwen3-8b", "status": "loaded"}

    fake_server.route("GET", "/api/v1/models", models)
    fake_server.route("GET", "/v1/models", body = {"object": "list", "data": [{"id": "qwen/qwen3-8b"}]})
    fake_server.route("POST", "/api/v1/models/load", load)
    _, _, entry = providers.connect(Target("lmstudio", fake_server.base), None, "qwen/qwen3-8b", 32768)
    assert entry == {"id": "qwen/qwen3-8b", "context_length": 32768}


# ── connect: --no-model-load ──


def test_no_model_load_refuses_an_unloaded_ollama_model(fake_server):
    _ollama(fake_server)
    with pytest.raises(ProviderError, match = "--no-model-load") as excinfo:
        providers.connect(
            Target("ollama", fake_server.base), None, "nope", None, allow_load = False
        )
    assert "Loaded: none" in str(excinfo.value)
    assert not any(method == "POST" for method, *_ in fake_server.requests)


def test_ollama_a_different_resident_window_goes_through_the_alias(fake_server):
    # A resident model at another window is stale for --context-length: the alias load still runs.
    state = {"ps": [("smollm2:135m", 16384)], "created": None}
    alias = "agent-switch/smollm2-135m-ctx8192:latest"

    def ps(_):
        return 200, {"models": [{"name": n, "model": n, "context_length": c} for n, c in state["ps"]]}

    def create(payload):
        state["created"] = payload
        return 200, {"status": "success"}

    def generate(payload):
        assert payload["model"] == alias
        state["ps"] = [(alias, 8192)]
        return 200, {"done": True}

    _ollama(fake_server)
    fake_server.route("GET", "/api/ps", ps)
    fake_server.route("POST", "/api/create", create)
    fake_server.route("POST", "/api/generate", generate)
    _, _, entry = providers.connect(Target("ollama", fake_server.base), None, "smollm2:135m", 8192)
    assert state["created"] is not None, "a resident model at another window must reload at the requested one"
    assert entry == {"id": alias, "context_length": 8192}


def test_no_model_load_refuses_a_resident_at_another_window(fake_server):
    _ollama(fake_server, ps = [("smollm2:135m", 16384)])
    with pytest.raises(ProviderError, match = "--no-model-load") as excinfo:
        providers.connect(
            Target("ollama", fake_server.base), None, "smollm2:135m", 8192, allow_load = False
        )
    assert "Loaded: smollm2:135m" in str(excinfo.value)
    assert not any(method == "POST" for method, *_ in fake_server.requests)


def test_no_model_load_attaches_a_resident_ollama_ctx_alias(fake_server):
    # A resident alias already carries the window, so --no-model-load can still honor --context-length.
    alias = "agent-switch/smollm2-135m-ctx8192:latest"
    _ollama(fake_server, ps = [(alias, 8192)], tags = ("smollm2:135m", alias))
    _, _, entry = providers.connect(
        Target("ollama", fake_server.base), None, "smollm2:135m", 8192, allow_load = False
    )
    assert entry == {"id": alias, "context_length": 8192}
    assert not any(method == "POST" for method, *_ in fake_server.requests)


def test_no_model_load_nothing_loaded_points_at_the_server(fake_server):
    _ollama(fake_server)
    with pytest.raises(ProviderError, match = "pick one it serves"):
        providers.connect(
            Target("ollama", fake_server.base), None, None, None, allow_load = False
        )


def test_no_model_load_refuses_an_unloaded_router_model(fake_server):
    fake_server.route("GET", "/props", body = {"role": "router", "default_generation_settings": {"n_ctx": 0}})
    fake_server.route(
        "GET",
        "/v1/models",
        body = {"data": [{"id": "jan-4b", "owned_by": "llamacpp", "status": {"value": "unloaded"}, "meta": None}]},
    )
    with pytest.raises(ProviderError, match = "--no-model-load"):
        providers.connect(
            Target("llamacpp", fake_server.base), None, "jan-4b", None, allow_load = False
        )
    assert not any(method == "POST" for method, *_ in fake_server.requests)


def test_no_model_load_keeps_the_plain_llama_server_error(fake_server):
    # A single-model llama-server can't load anything anyway; its own precise error stands.
    _llamacpp_single(fake_server)
    with pytest.raises(ProviderError, match = "can't switch models") as excinfo:
        providers.connect(
            Target("llamacpp", fake_server.base), None, "other-model", None, allow_load = False
        )
    assert "--no-model-load" not in str(excinfo.value)


def test_no_model_load_keeps_the_lmstudio_v0_hint(fake_server):
    fake_server.route("GET", "/api/v0/models", body = {"data": [{"id": "qwen/qwen3-8b", "state": "loaded", "max_context_length": 4096}]})
    fake_server.route("GET", "/v1/models", body = {"object": "list", "data": [{"id": "qwen/qwen3-8b"}]})
    with pytest.raises(ProviderError, match = "lms load") as excinfo:
        providers.connect(
            Target("lmstudio", fake_server.base), None, "qwen/other", None, allow_load = False
        )
    assert "--no-model-load" not in str(excinfo.value)


def test_lmstudio_a_different_resident_window_reloads(fake_server):
    # A resident instance at another window is stale for --context-length: the load still runs.
    state = {"instances": [{"id": "qwen/qwen3-8b", "config": {"context_length": 16384}}]}

    def models(_):
        return 200, {"models": [{"type": "llm", "key": "qwen/qwen3-8b", "max_context_length": 40960,
                                 "loaded_instances": state["instances"]}]}

    def load(payload):
        assert payload == {"model": "qwen/qwen3-8b", "context_length": 32768}
        state["instances"] = [{"id": "qwen/qwen3-8b", "config": {"context_length": 32768}}]
        return 200, {"instance_id": "qwen/qwen3-8b", "status": "loaded"}

    fake_server.route("GET", "/api/v1/models", models)
    fake_server.route("GET", "/v1/models", body = {"object": "list", "data": [{"id": "qwen/qwen3-8b"}]})
    fake_server.route("POST", "/api/v1/models/load", load)
    _, _, entry = providers.connect(Target("lmstudio", fake_server.base), None, "qwen/qwen3-8b", 32768)
    assert entry == {"id": "qwen/qwen3-8b", "context_length": 32768}


def test_no_model_load_refuses_a_resident_at_another_lmstudio_window(fake_server):
    fake_server.route(
        "GET",
        "/api/v1/models",
        body = {"models": [{"type": "llm", "key": "qwen/qwen3-8b", "max_context_length": 40960,
                            "loaded_instances": [{"id": "qwen/qwen3-8b", "config": {"context_length": 16384}}]}]},
    )
    fake_server.route("GET", "/v1/models", body = {"object": "list", "data": [{"id": "qwen/qwen3-8b"}]})
    with pytest.raises(ProviderError, match = "--no-model-load") as excinfo:
        providers.connect(
            Target("lmstudio", fake_server.base), None, "qwen/qwen3-8b", 32768, allow_load = False
        )
    assert "Loaded: qwen/qwen3-8b" in str(excinfo.value)
    assert not any(method == "POST" for method, *_ in fake_server.requests)


# ── endpoint capability ──


def test_missing_anthropic_endpoint_fails_for_claude(fake_server):
    _vllm(fake_server)
    with pytest.raises(ProviderError, match = "/v1/messages"):
        providers.connect(Target("vllm", fake_server.base), None, None, None, needs = ("/v1/messages",))


def test_present_endpoint_passes_on_a_validation_error(fake_server):
    _vllm(fake_server)
    fake_server.route("POST", "/v1/responses", status = 400, body = {"error": "input is required"})
    providers.connect(Target("vllm", fake_server.base), None, None, None, needs = ("/v1/responses",))


# ── request bodies ──

_BODY = {
    "temperature": 0.6,
    "top_p": 0.9,
    "top_k": 20,
    "min_p": 0.05,
    "repetition_penalty": 1.1,
    "presence_penalty": 0.5,
    "enable_thinking": False,
    "reasoning_effort": "high",
}


@pytest.mark.parametrize("name", ["openai"])
def test_request_body_passes_through(name):
    assert providers.request_body(name, _BODY) == (_BODY, [])


def test_request_body_for_llamacpp():
    body, dropped = providers.request_body("llamacpp", _BODY)
    assert dropped == []
    assert body == {
        "temperature": 0.6,
        "top_p": 0.9,
        "top_k": 20,
        "min_p": 0.05,
        "repeat_penalty": 1.1,
        "presence_penalty": 0.5,
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_effort": "high",
    }


def test_request_body_for_vllm():
    body, dropped = providers.request_body("vllm", _BODY)
    assert dropped == []
    assert body["repetition_penalty"] == 1.1
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "enable_thinking" not in body


def test_request_body_for_ollama_drops_what_its_openai_api_ignores():
    body, dropped = providers.request_body("ollama", _BODY)
    assert body == {"temperature": 0.6, "top_p": 0.9, "presence_penalty": 0.5, "reasoning_effort": "high"}
    assert dropped == ["top_k", "min_p", "repetition_penalty", "enable_thinking"]


def test_request_body_for_lmstudio():
    body, dropped = providers.request_body("lmstudio", _BODY)
    assert body == {"temperature": 0.6, "top_p": 0.9, "top_k": 20, "repeat_penalty": 1.1, "presence_penalty": 0.5}
    assert dropped == ["min_p", "enable_thinking", "reasoning_effort"]
