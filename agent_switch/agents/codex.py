# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`agent-switch codex`: Codex config/catalog writers, overlays and subagent bridge."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal, Optional

import typer

from agent_switch.core.install import (
    _agent_version_at_least,
    _codex_executable_version,
    _npm_install_hint,
    _require_agent_for_launch,
    _which_with_install_dirs,
)
from agent_switch.core.launch import _connect, _resolve_target, _run
from agent_switch.core.mcp import load_mcp_servers
from agent_switch.core.options import (
    LoadOptions,
    ProviderName,
    ServerOptions,
    _AS_SUBAGENT_OPTION,
    _COMPACT_AT_OPTION,
    _CONTEXT_OPTION,
    _HEADER_OPTION,
    _KEY_OPTION,
    _LAUNCH_OPTION,
    _MCP_ALL_OPTION,
    _MCP_OPTION,
    _MIN_P_OPTION,
    _MODEL_LOAD_OPTION,
    _MODEL_OPTION,
    _PERSIST_OPTION,
    _PRESENCE_PENALTY_OPTION,
    _PROVIDER_OPTION,
    _REASONING_EFFORT_OPTION,
    _REASONING_FIELDS,
    _REASONING_OPTION,
    _REPETITION_PENALTY_OPTION,
    _SUBAGENT_DESCRIPTION,
    _TEMPERATURE_OPTION,
    _TOP_K_OPTION,
    _TOP_P_OPTION,
    _URL_OPTION,
    _YOLO_OPTION,
    _check_compact_at,
    _consume_positional_model,
    _fail,
    _yolo_command_flags,
    parse_headers,
)
from agent_switch.core.platform import (
    _create_directory_junction,
    _looks_like_path,
    _remove_overlay_entry,
    _wsl_windows_executable,
)
from agent_switch.core.session import _session_config
from agent_switch.core.storage import _write_private_json, _write_private_text
from agent_switch.providers.utils import get_has_custom_authorization


_CODEX_PROFILE = "agent_switch"


_CODEX_ENV_KEY = "AGENT_SWITCH_AUTH_TOKEN"


# Codex treats an SSE stream with no bytes for this long as lost, cancels it and reconnects. Its default is 300000 (5 minutes), measured against the WHOLE quiet period, and llama-server sends nothing at all while it processes the prompt. A local CPU host chews through a prompt at low tens of tokens a second and Codex's own preamble is several thousand tokens before the user has typed anything: 16.1 tok/s measured on a 2-core box means ~460s of silence for a ~7300-token first turn, so the default trips before the first token exists. The reconnect is worse than the wait, because llama-server hands the retry a different parallel slot whose KV cache shares no prefix, so each attempt restarts prompt processing from zero and the five retries can never converge. Observed as a request completing in exactly 300056ms with `Reconnecting... 1/5` and no turn ever finishing. 20 minutes here, sized to be longer than a slow local first turn rather than to any server-side budget: nothing here bounds generation, and a genuinely dead stream is still caught, just later.
_CODEX_STREAM_IDLE_TIMEOUT_MS = 1_200_000


_CODEX_SUBAGENT_MCP_MODULE = "agent_switch.codex_subagent_mcp"


_CODEX_SUBAGENT_MCP_SERVER = "local_agent"


_CODEX_SUBAGENT_MCP_TOOL = "spawn_local_agent"


_CODEX_SUBAGENT_CONFIG_ENV = "AGENT_SWITCH_CODEX_SUBAGENT_CONFIG"


_CODEX_PARENT_OVERLAY_MANIFEST = ".agent-switch-parent-overlay.json"


_CODEX_SUBAGENT_TOOL_DESCRIPTION = (
    f"{_SUBAGENT_DESCRIPTION} Use this tool instead of the built-in spawn_agent tool for those "
    "requests. Other subagent requests may use the built-in tools normally."
)


_CODEX_SUBAGENT_ROUTING_INSTRUCTIONS = (
    "When the user asks to spawn a local agent, you must call the "
    "spawn_local_agent MCP tool once with the complete task. Do not answer, simulate the "
    "result, call wait, or use a built-in subagent before calling the tool. Use built-in "
    "subagents for other delegation requests."
)


_PROVIDER_HEADER = f"[model_providers.{_CODEX_PROFILE}]"


_CODEX_ENV_UNSET = ("OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN")


_CODEX_REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")


def _codex_reasoning_effort(
    reasoning: Optional[str], reasoning_effort: Optional[str]
) -> Optional[str]:
    if reasoning == "off":
        return "none"
    if reasoning_effort in _CODEX_REASONING_EFFORTS:
        return reasoning_effort
    return None


def _codex_provider_table(base: str, headers: Optional[dict] = None) -> str:
    table = (
        f"{_PROVIDER_HEADER}\n"
        'name = "agent-switch"\n'
        f"base_url = {json.dumps(base + '/v1')}\n"
    )
    # A custom Authorization header replaces the Bearer <env_key> auth; keeping both would send two.
    if not get_has_custom_authorization(headers or {}):
        table += f'env_key = "{_CODEX_ENV_KEY}"\n'
    table += (
        'wire_api = "responses"\n'
        "requires_openai_auth = false\n"
        f"stream_idle_timeout_ms = {_CODEX_STREAM_IDLE_TIMEOUT_MS}\n"
    )
    if headers:
        pairs = ", ".join(f"{json.dumps(name)} = {json.dumps(value)}" for name, value in headers.items())
        table += f"http_headers = {{ {pairs} }}\n"
    return table


_CODEX_PROVIDER_TABLES = (_PROVIDER_HEADER, _PROVIDER_HEADER[:-1] + ".")


def _codex_mcp_tables(servers: Optional[dict]) -> str:
    """[mcp_servers."<name>"] tables for the session config.toml; keys verified against the codex binary."""
    if not servers:
        return ""
    tables = ""
    for name, server in servers.items():
        tables += f'\n[mcp_servers.{json.dumps(name)}]\n'
        if server["transport"] == "stdio":
            tables += f"command = {json.dumps(server['command'])}\n"
            if server["args"]:
                tables += f"args = {json.dumps(server['args'])}\n"
            if server["env"]:
                pairs = ", ".join(
                    f"{json.dumps(key)} = {json.dumps(value)}"
                    for key, value in server["env"].items()
                )
                tables += f"env = {{ {pairs} }}\n"
        else:
            tables += f"url = {json.dumps(server['url'])}\n"
            if server["headers"]:
                pairs = ", ".join(
                    f"{json.dumps(key)} = {json.dumps(value)}"
                    for key, value in server["headers"].items()
                )
                tables += f"http_headers = {{ {pairs} }}\n"
    return tables


def _merge_codex_config(
    existing: str,
    base: str,
    headers: Optional[dict] = None,
    mcp_servers: Optional[dict] = None,
) -> str:
    chunks = re.split(r"(?m)^(?=\[)", existing)  # preamble, then one chunk per table
    if not re.search(r"(?m)^\s*oss_provider\s*=", chunks[0]):
        if chunks[0] and not chunks[0].endswith("\n"):
            chunks[0] += "\n"
        chunks[0] += f'oss_provider = "{_CODEX_PROFILE}"\n'
    # This config.toml is agent-switch's own; drop every mcp_servers chunk first so a persisted
    # session never keeps an earlier run's servers, then append the ones mounted now (if any).
    text = "".join(
        c
        for c in chunks
        if not c.startswith(_CODEX_PROVIDER_TABLES) and not c.startswith("[mcp_servers")
    )
    if not text.endswith("\n"):
        text += "\n"
    if not text.endswith("\n\n"):
        text += "\n"
    return text + _codex_provider_table(base, headers) + _codex_mcp_tables(mcp_servers)


# Keep custom-model behavior aligned with Codex's own unknown-model fallback. This Apache-2.0 prompt is copied from openai/codex rust-v0.144.0 models-manager/prompt.md.
_CODEX_FALLBACK_PROMPT = Path(__file__).parent / "codex_fallback_prompt.md"


_CODEX_MODEL_CATALOG_MIN_VERSION = (0, 110, 0)


_CODEX_PATCH_LINE_ENDINGS_MIN_VERSION = (0, 148, 0)


# Older Codex sends no reasoning for a model without reasoning summaries, and older Pi has no samplingParams.
_CODEX_REASONING_REQUEST_MIN_VERSION = (0, 145, 0)


def _codex_supports_model_catalog() -> bool:
    executable = _which_with_install_dirs("codex")
    if executable is None:
        # A --no-launch recipe may be copied to another machine; assume a current Codex.
        return True
    version = _codex_executable_version(executable)
    return version is not None and version >= _CODEX_MODEL_CATALOG_MIN_VERSION


def _codex_supports_patch_line_endings() -> bool:
    executable = _which_with_install_dirs("codex")
    if executable is None:
        # A normal launch installs Codex after this check; no-launch may run elsewhere.
        return True
    version = _codex_executable_version(executable)
    return version is not None and version >= _CODEX_PATCH_LINE_ENDINGS_MIN_VERSION


def _codex_model_catalog(model: dict) -> dict:
    """Return conservative metadata for a local model unknown to Codex's built-in catalog."""
    model_id = model["id"]
    window = model.get("context_length") or model.get("max_context_length")
    entry = {
        "slug": model_id,
        "display_name": model_id,
        "description": "Model served by a local server",
        "supported_reasoning_levels": [],
        "shell_type": "default",
        "visibility": "none",
        "supported_in_api": True,
        "priority": 99,
        "availability_nux": None,
        "upgrade": None,
        "base_instructions": _CODEX_FALLBACK_PROMPT.read_text(encoding = "utf-8"),
        "supports_reasoning_summaries": False,
        "supports_reasoning_summary_parameter": False,
        "support_verbosity": False,
        "default_verbosity": None,
        "apply_patch_tool_type": "freeform",
        "truncation_policy": {"mode": "bytes", "limit": 10_000},
        "supports_parallel_tool_calls": False,
        "experimental_supported_tools": [],
    }
    if window:
        entry["context_window"] = int(window)
        entry["max_context_window"] = int(window)
    return {"models": [entry]}


def write_codex_config(
    base: str,
    model: dict,
    home: Path,
    reasoning_effort: Optional[str] = None,
    headers: Optional[dict] = None,
    compact_at: Optional[float] = None,
    mcp_servers: Optional[dict] = None,
) -> None:
    home.mkdir(parents = True, exist_ok = True)

    config = home / "config.toml"
    existing = config.read_text(encoding = "utf-8") if config.exists() else ""
    merged = _merge_codex_config(existing, base, headers, mcp_servers)
    if merged != existing:
        config.write_text(merged, encoding = "utf-8")
        # http_headers can carry a secret Authorization, so keep the file owner-only like the JSON configs.
        config.chmod(0o600)
        typer.echo(f"Updated {config}")

    # oss_provider here too: codex --oss picks the provider from it, and the profile layer must beat a user-set value ("ollama") in config.toml.
    profile_text = (
        f'oss_provider = "{_CODEX_PROFILE}"\n'
        f'model_provider = "{_CODEX_PROFILE}"\n'
        f"model = {json.dumps(model['id'])}\n"
    )
    if _codex_supports_patch_line_endings():
        profile_text += (
            "suppress_unstable_features_warning = true\n"
            "features.apply_patch_preserve_line_endings = true\n"
        )
    if _codex_supports_model_catalog() and _CODEX_FALLBACK_PROMPT.is_file():
        catalog = home / "model-catalog.json"
        catalog_text = json.dumps(_codex_model_catalog(model), indent = 2) + "\n"
        if not catalog.exists() or catalog.read_text(encoding = "utf-8") != catalog_text:
            catalog.write_text(catalog_text, encoding = "utf-8")
            typer.echo(f"Updated {catalog}")
        # Resolve relative to the profile file. This also survives WSL launching a Windows Codex binary, where a Linux absolute path inside TOML would not be usable.
        profile_text += f"model_catalog_json = {json.dumps(catalog.name)}\n"

    window = model.get("context_length") or model.get("max_context_length")
    if window:
        profile_text += f"model_context_window = {int(window)}\n"
        if compact_at is not None:
            profile_text += f"model_auto_compact_token_limit = {int(int(window) * compact_at)}\n"
    if reasoning_effort:
        profile_text += f"model_reasoning_effort = {json.dumps(reasoning_effort)}\n"
    profile = home / f"{_CODEX_PROFILE}.config.toml"
    if not profile.exists() or profile.read_text(encoding = "utf-8") != profile_text:
        profile.write_text(profile_text, encoding = "utf-8")
        typer.echo(f"Updated {profile}")


def write_codex_subagent_bridge(
    base: str,
    key: str,
    model: dict,
    home: Path,
    *,
    yolo: bool,
    reasoning_effort: Optional[str] = None,
    headers: Optional[dict] = None,
    compact_at: Optional[float] = None,
) -> Path:
    """Write private config for an explicit local Codex child launched through MCP."""
    child_home = home / "child"
    write_codex_config(base, model, child_home, reasoning_effort, headers, compact_at)
    path = home / "subagent.json"
    _write_private_json(
        path,
        {
            "api_key": key,
            "codex_home": str(child_home),
            "bypass_permissions": yolo,
        },
    )
    return path


def _wsl_windows_user_profile(executable: str) -> Path:
    """Return the Windows user profile as a path accessible from WSL."""
    profile = os.environ.get("USERPROFILE", "").strip()
    if not profile:
        try:
            profile = subprocess.check_output(
                ["cmd.exe", "/d", "/c", "echo %USERPROFILE%"],
                text = True,
                encoding = "utf-8",
                # The path is the value: a corrupted home is worse than a loud failure.
                errors = "strict",
                stderr = subprocess.DEVNULL,
                cwd = str(Path(executable).parent),
            ).strip()
        except UnicodeDecodeError as exc:
            _fail(
                f"Could not read the Windows user profile for Codex ({exc}); "
                "set USERPROFILE in the WSL environment, for example through WSLENV."
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            _fail(f"Could not find the Windows user profile for Codex: {exc}")
    if not profile or profile == "%USERPROFILE%":
        _fail("Could not find the Windows user profile for Codex.")
    if profile.startswith("/"):
        return Path(profile)
    try:
        translated = subprocess.check_output(
            ["wslpath", "-u", profile],
            text = True,
            encoding = "utf-8",
            errors = "replace",
            stderr = subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        _fail(f"Could not translate Windows user profile {profile}: {exc}")
    if not translated:
        _fail(f"Could not translate Windows user profile {profile}.")
    return Path(translated)


def _codex_source_home(*, ignore_configured: bool = False) -> Path:
    configured = None if ignore_configured else os.environ.get("CODEX_HOME")
    if configured:
        if _wsl_windows_executable(["codex"]) and _looks_like_path(configured):
            if not configured.startswith("/"):
                try:
                    configured = subprocess.check_output(
                        ["wslpath", "-u", configured],
                        text = True,
                        encoding = "utf-8",
                        errors = "replace",
                        stderr = subprocess.DEVNULL,
                    ).strip()
                except (OSError, subprocess.CalledProcessError) as exc:
                    _fail(f"Could not translate Windows CODEX_HOME {configured}: {exc}")
                if not configured:
                    _fail("Could not translate Windows CODEX_HOME.")
        return Path(configured).expanduser()
    executable = _wsl_windows_executable(["codex"])
    if executable:
        return _wsl_windows_user_profile(executable) / ".codex"
    return Path.home() / ".codex"


def write_codex_parent_overlay(overlay: Path) -> Path:
    """Add local-agent routing without replacing the cloud parent's configuration."""
    overlay.mkdir(parents = True, exist_ok = True, mode = 0o700)

    manifest_path = overlay / _CODEX_PARENT_OVERLAY_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding = "utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        manifest = None
    source_home = _codex_source_home()
    overlay_key = str(overlay.resolve(strict = False))
    source_key = str(source_home.resolve(strict = False))
    if source_key == overlay_key:
        previous_source = manifest.get("source_home") if isinstance(manifest, dict) else None
        if isinstance(previous_source, str) and previous_source:
            candidate = Path(previous_source).expanduser()
            if str(candidate.resolve(strict = False)) != overlay_key:
                source_home = candidate
            else:
                source_home = _codex_source_home(ignore_configured = True)
        else:
            source_home = _codex_source_home(ignore_configured = True)
        source_key = str(source_home.resolve(strict = False))
    same_source = isinstance(manifest, dict) and manifest.get("source_home") == source_key
    if same_source:
        managed_entries = manifest.get("entries", [])
        if not isinstance(managed_entries, list):
            managed_entries = []
        for name in managed_entries:
            if isinstance(name, str) and name not in {"", ".", ".."} and Path(name).name == name:
                _remove_overlay_entry(overlay / name)
    else:
        # A reused overlay must never mix credentials, config, or plugins from two different Codex homes. Legacy overlays have no manifest, so rebuild them once.
        for target in list(overlay.iterdir()):
            _remove_overlay_entry(target)

    # Keep the user's auth, config, plugins, agents, skills, rules and session state visible. Symlinks make this an overlay rather than a stale copy; if Windows denies them, directory junctions keep large runtime state shared without a bulk copy, and the configuration surfaces and sessions are copied only when both link forms are unavailable.
    fallback_dirs = {"agents", "skills", "rules", "plugins", "marketplaces", "sessions"}
    entries = []
    if source_home.is_dir():
        for source in source_home.iterdir():
            if source.name in {
                "AGENTS.md",
                "AGENTS.override.md",
                _CODEX_PARENT_OVERLAY_MANIFEST,
            }:
                continue
            target = overlay / source.name
            _remove_overlay_entry(target)
            try:
                target.symlink_to(source, target_is_directory = source.is_dir())
                entries.append(source.name)
            except OSError:
                if source.is_file():
                    shutil.copy2(source, target)
                    entries.append(source.name)
                elif source.is_dir():
                    if _create_directory_junction(source, target):
                        entries.append(source.name)
                    elif source.name in fallback_dirs:
                        shutil.copytree(source, target)
                        entries.append(source.name)

    _write_private_json(
        manifest_path,
        {"source_home": source_key, "entries": sorted(entries)},
    )

    inherited = ""
    instruction_name = "AGENTS.md"
    for candidate in (source_home / "AGENTS.override.md", source_home / "AGENTS.md"):
        try:
            text = candidate.read_text(encoding = "utf-8")
        except FileNotFoundError:
            continue
        except OSError as exc:
            _fail(f"Could not preserve Codex instructions from {candidate}: {exc}")
        if text.strip():
            inherited = text.rstrip()
            instruction_name = candidate.name
            break

    other_name = "AGENTS.md" if instruction_name == "AGENTS.override.md" else "AGENTS.override.md"
    other = overlay / other_name
    if other.is_file() or other.is_symlink():
        other.unlink()
    routing = _CODEX_SUBAGENT_ROUTING_INSTRUCTIONS
    combined = f"{inherited}\n\n{routing}\n" if inherited else f"{routing}\n"
    _write_private_text(overlay / instruction_name, combined)
    return overlay


def _codex_subagent_flags(path: Path) -> list[str]:
    command = sys.executable
    package_root = str(Path(__file__).resolve().parents[2])
    bootstrap = (
        f"import sys;sys.path.insert(0,{json.dumps(package_root)});"
        f"from {_CODEX_SUBAGENT_MCP_MODULE} import main;main()"
    )
    args = ["-c", bootstrap, str(path)]
    if _wsl_windows_executable(["codex"]):
        command = "wsl.exe"
        args = [
            "-d",
            os.environ["WSL_DISTRO_NAME"],
            "--",
            sys.executable,
            "-c",
            bootstrap,
            str(path),
        ]
    server = (
        "{ "
        f"command = {json.dumps(command)}, "
        f"args = {json.dumps(args)}, "
        f"required = true, enabled_tools = [{json.dumps(_CODEX_SUBAGENT_MCP_TOOL)}], "
        'default_tools_approval_mode = "approve", '
        "startup_timeout_sec = 15, tool_timeout_sec = 3600 }"
    )
    return ["-c", f"mcp_servers.{_CODEX_SUBAGENT_MCP_SERVER}={server}"]


def codex(
    ctx: typer.Context,
    model: Optional[str] = _MODEL_OPTION,
    api_key: Optional[str] = _KEY_OPTION,
    header: Optional[list[str]] = _HEADER_OPTION,
    launch: bool = _LAUNCH_OPTION,
    max_seq_length: int = _CONTEXT_OPTION,
    reasoning: Optional[Literal["on", "off", "auto"]] = _REASONING_OPTION,
    reasoning_effort: Optional[str] = _REASONING_EFFORT_OPTION,
    temperature: Optional[float] = _TEMPERATURE_OPTION,
    top_p: Optional[float] = _TOP_P_OPTION,
    top_k: Optional[int] = _TOP_K_OPTION,
    min_p: Optional[float] = _MIN_P_OPTION,
    repetition_penalty: Optional[float] = _REPETITION_PENALTY_OPTION,
    presence_penalty: Optional[float] = _PRESENCE_PENALTY_OPTION,
    compact_at: Optional[float] = _COMPACT_AT_OPTION,
    model_load: bool = _MODEL_LOAD_OPTION,
    url: Optional[str] = _URL_OPTION,
    provider: Optional[ProviderName] = _PROVIDER_OPTION,
    yolo: bool = _YOLO_OPTION,
    persist: bool = _PERSIST_OPTION,
    as_subagent: bool = _AS_SUBAGENT_OPTION,
    mcp: Optional[list[str]] = _MCP_OPTION,
    mcp_all: bool = _MCP_ALL_OPTION,
):
    """Point OpenAI Codex at a local model server and start it."""
    # Route a leading `org/name` positional to --model; forward the rest to the agent.
    model, ctx.args[:] = _consume_positional_model(model, ctx.args)
    headers = parse_headers(header)
    if as_subagent and (mcp or mcp_all):
        # The subagent's parent CODEX_HOME symlinks into the real ~/.codex, so writing MCP
        # servers there would modify the user's own config.
        _fail("--mcp/--mcp-all cannot be combined with --as-subagent.")
    # Validate the MCP selection before _connect, so a registry error fails fast.
    mcp_servers = load_mcp_servers(mcp, should_mount_all = mcp_all)
    target = _resolve_target(url, provider, api_key, headers)
    install_hint = _npm_install_hint("@openai/codex")
    _require_agent_for_launch("codex", install_hint, launch)
    codex_effort = _codex_reasoning_effort(reasoning, reasoning_effort)
    if codex_effort and not _agent_version_at_least("codex", _CODEX_REASONING_REQUEST_MIN_VERSION):
        codex_effort = None
    server_options = ServerOptions(
        reasoning = reasoning,
        reasoning_effort = reasoning_effort,
        temperature = temperature,
        top_p = top_p,
        top_k = top_k,
        min_p = min_p,
        repetition_penalty = repetition_penalty,
        presence_penalty = presence_penalty,
        carried = _REASONING_FIELDS if codex_effort else frozenset(),
        provider = target.name,
    )
    base, key, entry = _connect(
        api_key,
        model,
        LoadOptions(max_seq_length, model_load),
        server_options = server_options,
        target = target,
        needs = ("/v1/responses",),
    )
    _check_compact_at(compact_at, entry)
    if as_subagent:
        with _session_config("codex-subagent", launch, persist = persist) as home:
            bridge_config = write_codex_subagent_bridge(
                base,
                key,
                entry,
                home,
                yolo = yolo,
                reasoning_effort = codex_effort,
                headers = headers,
                compact_at = compact_at,
            )
            parent_home = write_codex_parent_overlay(home / "parent")
            command = [
                "codex",
                *_codex_subagent_flags(bridge_config),
                *_yolo_command_flags("codex", yolo),
                *ctx.args,
            ]
            typer.echo(
                "A local agent is available. Ask Codex to spawn a local agent."
            )
            _run(
                base,
                entry,
                {"CODEX_HOME": str(parent_home)},
                command,
                launch = launch,
                install_hint = install_hint,
            )
        return
    command = [
        "codex",
        "--oss",
        "--profile",
        _CODEX_PROFILE,
        *_yolo_command_flags("codex", yolo),
        *ctx.args,
    ]
    with _session_config("codex", launch, persist = persist) as home:
        write_codex_config(base, entry, home, codex_effort, headers, compact_at, mcp_servers)
        env = {_CODEX_ENV_KEY: key, "CODEX_HOME": str(home)}
        _run(
            base,
            entry,
            env,
            command,
            launch = launch,
            install_hint = install_hint,
            unset_env = _CODEX_ENV_UNSET,
        )
