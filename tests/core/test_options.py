# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See LICENSE
# Ported from unsloth_cli/tests/test_start.py (unsloth commit 8e11ba15e); see NOTICE.md.

"""Option parsing: repo/model token splitting and yolo command maps."""

import pytest

from agent_switch.core import (
    options as core_options,
)


@pytest.mark.parametrize(
    "model, expected",
    [
        ("org/Qwen3-1.7B-GGUF:UD-Q4_K_XL", ("org/Qwen3-1.7B-GGUF", "UD-Q4_K_XL")),
        ("org/gemma-4-E2B-it-GGUF:Q8_0", ("org/gemma-4-E2B-it-GGUF", "Q8_0")),
        ("org/Qwen3-1.7B-GGUF", ("org/Qwen3-1.7B-GGUF", None)),  # no suffix
        ("/models/local.gguf", ("/models/local.gguf", None)),  # absolute path
        ("./rel.gguf", ("./rel.gguf", None)),  # relative path
        ("C:\\models\\x.gguf", ("C:\\models\\x.gguf", None)),  # Windows drive
        ("repo:with/slash", ("repo:with/slash", None)),  # slash in variant -> not a variant
        ("", ("", None)),
    ],
)
def test_split_repo_variant(model, expected):
    assert core_options._split_repo_variant(model) == expected


@pytest.mark.parametrize(
    "token, expected",
    [
        ("org/gemma-4-E2B-it-GGUF", True),
        ("org/gemma-4-E2B-it-GGUF:UD-Q4_K_XL", True),
        ("some-org/model.name_1", True),
        ("--continue", False),  # flag
        ("resume", False),  # single word, no slash
        ("/models/local.gguf", False),  # absolute path
        ("./rel.gguf", False),  # relative path
        ("C:\\models\\x.gguf", False),  # Windows drive
        ("my models/foo", False),  # has a space
        ("owner/repo/extra", False),  # too many segments
    ],
)
def test_looks_like_model(token, expected):
    assert core_options._looks_like_model(token) is expected


def test_consume_positional_model_leading_token():
    # A leading org/name positional routes to --model and is dropped from the passthrough.
    model, rest = core_options._consume_positional_model(None, ["org/Model-GGUF", "--continue"])
    assert model == "org/Model-GGUF"
    assert rest == ["--continue"]


def test_looks_like_model_leaves_existing_local_dir_for_agent(tmp_path, monkeypatch):
    # A relative `owner/repo` that actually exists (e.g. an OpenCode project dir) must
    # stay an agent argument, not be consumed as a model.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "owner" / "repo").mkdir(parents = True)
    assert core_options._looks_like_model("owner/repo") is False
    model, rest = core_options._consume_positional_model(None, ["owner/repo"])
    assert model is None and rest == ["owner/repo"]
    # The same shape, when it does not exist locally, is still treated as a model.
    assert core_options._looks_like_model("owner/absent-repo") is True


def test_consume_positional_model_ignores_non_leading_and_explicit_model():
    # An org/name that is an option value (not leading) is never stolen.
    model, rest = core_options._consume_positional_model(None, ["--profile", "owner/repo"])
    assert model is None and rest == ["--profile", "owner/repo"]
    # An explicit --model always wins; the positional is left untouched.
    model, rest = core_options._consume_positional_model("explicit/model", ["owner/repo"])
    assert model == "explicit/model" and rest == ["owner/repo"]


def test_yolo_command_flags_unmapped_agent_is_empty():
    # Placement-aware/config-based agents (and any typo) must yield no prefix flag.
    assert core_options._yolo_command_flags("opencode", True) == []
    assert core_options._yolo_command_flags("claude", True) == ["--dangerously-skip-permissions"]
    assert core_options._yolo_command_flags("claude", False) == []
