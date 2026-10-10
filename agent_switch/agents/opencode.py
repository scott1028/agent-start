# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See LICENSE

"""`agent-switch opencode`: command detection, native auto and config writing."""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Literal, Optional

import typer

from agent_switch.core.install import (
    _npm_install_hint,
    _probe_env,
    _require_agent_for_launch,
    _which_with_install_dirs,
)
from agent_switch.core.launch import (
    _ALIAS_META,
    _agent_command,
    _check_alias,
    _connect,
    _resolve_target,
    _run,
    _run_native,
)
from agent_switch.core.mcp import load_mcp_servers
from agent_switch.core.options import (
    LoadOptions,
    ProviderName,
    ServerOptions,
    _ALL_REQUEST_FIELDS,
    _AS_SUBAGENT_OPTION,
    _COMPACT_AT_OPTION,
    _CONTEXT_OPTION,
    _HEADER_OPTION,
    _KEY_OPTION,
    _LAUNCH_OPTION,
    _MAX_TOKENS_OPTION,
    _MCP_ALL_OPTION,
    _MCP_ENV_OPTION,
    _MCP_HEADER_OPTION,
    _MCP_OAUTH_URL_OPTION,
    _MCP_OPTION,
    _MCP_STDIO_OPTION,
    _MCP_URL_OPTION,
    _MIN_P_OPTION,
    _MODEL_LOAD_OPTION,
    _MODEL_OPTION,
    _OPENCODE_OUTPUT_TOKEN_MAX,
    _PERSIST_OPTION,
    _PRESENCE_PENALTY_OPTION,
    _PROVIDER_OPTION,
    _REASONING_EFFORT_OPTION,
    _REASONING_OPTION,
    _REPETITION_PENALTY_OPTION,
    _SUBAGENT_DESCRIPTION,
    _SUBAGENT_INSTRUCTIONS,
    _TEMPERATURE_OPTION,
    _TOP_K_OPTION,
    _TOP_P_OPTION,
    _URL_OPTION,
    _YOLO_OPTION,
    _agent_output_limit,
    _check_compact_at,
    _consume_positional_model,
    _fail,
    _get_compaction_reserve,
    _refuse_local_only,
    opencode_output_limit,
    parse_headers,
)
from agent_switch.core.session import _agent_config_path, _session_config
from agent_switch.core.storage import _read_json_object, _subdict, _write_private_json
from agent_switch.providers.utils import get_has_custom_authorization


_SUBAGENT_NAME = "local"


# OpenCode selects a model by "<providerID>/<modelID>". Use a dedicated id to avoid colliding with a user's providers; provider filters are set in the launch-time overlay.
_OPENCODE_PROVIDER = "agent-switch"


_OPENCODE_OUTPUT_TOKEN_MAX_ENV = "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"


# Subcommands that reject --auto (OpenCode exposes it only on the default TUI and `run`), so `opencode serve --auto` is never emitted. Includes console/generate, hidden from `opencode --help` but still registered. Unknown first positionals are TUI paths and get --auto.
_OPENCODE_NON_AUTO_SUBCOMMANDS = frozenset(
    "completion acp mcp attach debug providers auth agent upgrade uninstall serve web "
    "models stats export import github pr session plugin plug db console generate".split()
)


_OPENCODE_V2_SUBCOMMANDS = frozenset(
    "acp api debug console auth mcp plugin models export import mini run service pair serve".split()
)


_OPENCODE_V2_STANDALONE_SUBCOMMANDS = frozenset("api models export import mini run".split())


_OPENCODE_GLOBAL_BOOLEAN_OPTIONS = frozenset(
    "-h --help -v --version --print-logs --pure --mdns --standalone --wizard".split()
)


_OPENCODE_GLOBAL_VALUE_OPTIONS = frozenset(
    "--log-level --port --hostname --mdns-domain --cors --server --completions --cpu-profile".split()
)


_OPENCODE_NATIVE_AUTO_MIN_VERSION = (1, 17, 12)


def _opencode_command() -> tuple[str, bool]:
    resolved_v2 = _which_with_install_dirs("opencode2")
    if resolved_v2:
        return resolved_v2, True
    return "opencode", False


def _opencode_supports_native_auto(command: str = "opencode") -> bool:
    if Path(command).stem.lower() == "opencode2":
        return True
    executable = _which_with_install_dirs(command)
    if executable is None:
        # No local binary: a --no-launch recipe may run elsewhere, and _run installs the current release on launch, so either way assume native --auto is available.
        return True
    try:
        output = subprocess.check_output(
            [executable, "--version"],
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 10,
            stderr = subprocess.DEVNULL,
            env = _probe_env(),
        )
    except Exception:
        return False
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", output)
    return bool(match) and tuple(int(part) for part in match.groups()) >= (
        _OPENCODE_NATIVE_AUTO_MIN_VERSION
    )


def _opencode_subcommand(args: list[str]) -> tuple[Optional[str], Optional[int]]:
    """Return an explicit OpenCode subcommand and its index after global options."""
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--":
            return None, None
        if arg in _OPENCODE_GLOBAL_BOOLEAN_OPTIONS:
            index += 1
            continue
        if arg in _OPENCODE_GLOBAL_VALUE_OPTIONS:
            index += 2
            continue
        if any(arg.startswith(f"{option}=") for option in _OPENCODE_GLOBAL_VALUE_OPTIONS):
            index += 1
            continue
        # A non-global option (--session) is a TUI flag; stop before its value is mistaken for a subcommand.
        if arg.startswith("-"):
            return None, None
        return arg, index
    return None, None


def _opencode_native_auto_args(
    args: list[str],
    yolo: bool,
    *,
    v2: bool = False,
) -> tuple[list[str], bool]:
    """Add OpenCode's native --auto when the selected command supports it."""
    routed = list(args)
    if not yolo:
        return routed, False
    subcommand, _ = _opencode_subcommand(routed)
    if v2 and subcommand in _OPENCODE_V2_SUBCOMMANDS and subcommand != "run":
        return routed, False
    if not v2 and subcommand in _OPENCODE_NON_AUTO_SUBCOMMANDS:
        return routed, False
    separator = routed.index("--") if "--" in routed else len(routed)
    # --mini's runMini TUI forces auto=false and never forwards --auto, so appending it is useless; fall back to the config permission block so --yolo still auto-approves.
    if any(arg == "--mini" or arg.startswith("--mini=") for arg in routed[:separator]):
        return routed, False
    if "--auto" not in routed[:separator]:
        routed.insert(separator, "--auto")
    return routed, True


def _opencode_v2_standalone_args(args: list[str]) -> list[str]:
    """Keep session-only config on the V2 server that consumes it."""
    routed = list(args)
    separator = routed.index("--") if "--" in routed else len(routed)
    head = routed[:separator]
    if (
        "--standalone" in head
        or "--server" in head
        or any(arg.startswith("--server=") for arg in head)
    ):
        return routed
    subcommand, subcommand_index = _opencode_subcommand(routed)
    if (
        subcommand in _OPENCODE_V2_SUBCOMMANDS
        and subcommand not in _OPENCODE_V2_STANDALONE_SUBCOMMANDS
    ):
        return routed
    insert_at = (
        subcommand_index + 1 if subcommand in _OPENCODE_V2_STANDALONE_SUBCOMMANDS else separator
    )
    routed.insert(insert_at, "--standalone")
    return routed


def _opencode_subagent_inline_config(
    path: Path,
    permission: dict,
    command: str = "opencode",
    v2: bool = False,
) -> dict:
    """Keep the local provider visible without hiding the parent's allowed providers."""
    inline: dict = {}
    inherited = os.environ.get("OPENCODE_CONFIG_CONTENT")
    if inherited:
        try:
            parsed = json.loads(inherited)
        except ValueError:
            _fail("OPENCODE_CONFIG_CONTENT is not valid JSON.")
        if not isinstance(parsed, dict):
            _fail("OPENCODE_CONFIG_CONTENT must contain a JSON object.")
        inline.update(parsed)

    def merge_provider_filters(effective_config: dict) -> None:
        enabled = effective_config.get("enabled_providers")
        if isinstance(enabled, list):
            inherited_enabled = inline.get("enabled_providers")
            if not isinstance(inherited_enabled, list):
                inherited_enabled = []
            providers = [
                provider
                for provider in [*inherited_enabled, *enabled]
                if provider != _OPENCODE_PROVIDER
            ]
            inline["enabled_providers"] = list(dict.fromkeys([*providers, _OPENCODE_PROVIDER]))
        disabled = effective_config.get("disabled_providers")
        if isinstance(disabled, list) and _OPENCODE_PROVIDER in disabled:
            inline["disabled_providers"] = [
                provider for provider in disabled if provider != _OPENCODE_PROVIDER
            ]

    # Keep an inherited inline allowlist usable by the local provider. V2 turns these filters into policies where global/project rules still intentionally outrank this content; the message at launch makes that boundary explicit.
    merge_provider_filters(inline)
    effective = inline

    executable = None if v2 else _which_with_install_dirs(command)
    if v2:
        legacy_depth = inline.pop("subagent_depth", None)
        if (
            isinstance(legacy_depth, int)
            and not isinstance(legacy_depth, bool)
            and legacy_depth > 0
        ):
            experimental = inline.get("experimental")
            if not isinstance(experimental, dict):
                experimental = {}
            experimental.setdefault("subagent_depth", legacy_depth)
            inline["experimental"] = experimental
    elif executable is None:
        typer.echo(
            f"Warning: OpenCode is not installed, so provider filters could not be checked. "
            f"The target configuration must allow '{_OPENCODE_PROVIDER}'.",
            err = True,
        )
    else:
        env = _probe_env(OPENCODE_CONFIG = _agent_config_path(path, [command]))
        try:
            resolved = subprocess.run(
                [executable, "debug", "config"],
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 15,
                env = env,
            )
        except Exception as exc:
            _fail(f"Could not inspect OpenCode provider filters: {exc}")
        if resolved.returncode != 0:
            detail = resolved.stderr.strip() or resolved.stdout.strip()
            _fail(f"Could not inspect OpenCode provider filters: {detail or 'unknown error'}")
        try:
            effective = json.loads(resolved.stdout)
        except ValueError:
            _fail("Could not inspect OpenCode provider filters: invalid JSON response.")
        if not isinstance(effective, dict):
            _fail("Could not inspect OpenCode provider filters: expected a JSON object.")

        merge_provider_filters(effective)

    if not v2:
        depth = effective.get("subagent_depth")
        inline["subagent_depth"] = (
            depth if isinstance(depth, int) and not isinstance(depth, bool) and depth > 0 else 1
        )
    if permission:
        inline["permission"] = permission
    return inline


def opencode_compaction_reserved(window: int, output: int, ratio: Optional[float] = None) -> int:
    if ratio is not None:
        # An explicit ratio is honored exactly; the default's output cap and 8192 floor
        # would block the earlier compaction the flag is asked for.
        return _get_compaction_reserve(window, ratio)
    return max(1, min(output, max(window // 10, 8192)))


def _opencode_output_env(model: dict, max_tokens: Optional[int]) -> dict:
    """Lift OpenCode's output ceiling when --max-tokens exceeds it; re-emit an inherited one so a --no-launch recipe keeps it."""
    window = model.get("context_length") or model.get("max_context_length")
    if not max_tokens:
        return {}
    if not window:
        typer.echo(
            "Warning: the server did not report the model's context length, so --max-tokens is ignored.",
            err = True,
        )
        return {}
    output = _agent_output_limit(int(window), max_tokens)
    raw = os.environ.get(_OPENCODE_OUTPUT_TOKEN_MAX_ENV, "")
    inherited = int(raw) if raw.isdigit() and int(raw) > 0 else None
    ceiling = inherited or _OPENCODE_OUTPUT_TOKEN_MAX
    if output <= ceiling and inherited is None:
        return {}
    return {_OPENCODE_OUTPUT_TOKEN_MAX_ENV: str(max(output, ceiling))}


def _opencode_provider(
    base: str,
    key: str,
    model: dict,
    max_tokens: Optional[int] = None,
    request_body: Optional[dict] = None,
    headers: Optional[dict] = None,
) -> dict:
    model_entry = {"name": model["id"]}
    window = model.get("context_length") or model.get("max_context_length")
    if window:
        window = int(window)
        output = opencode_output_limit(window, max_tokens)
        # Without a limit OpenCode assumes context 0 and never compacts. Without input it compacts at context - output and ignores compaction.reserved.
        model_entry["limit"] = {"context": window, "input": window, "output": output}
    provider_options = {"baseURL": f"{base}/v1"}
    # A custom Authorization header replaces the SDK's Bearer <apiKey>; keeping both would send two.
    if not get_has_custom_authorization(headers or {}):
        provider_options["apiKey"] = key
    if headers:
        provider_options["headers"] = dict(headers)
    if request_body:
        # OpenCode 1.x sends model options, reading the effort only as reasoningEffort; 2.x sends only the provider body.
        model_entry["options"] = {
            "reasoningEffort" if name == "reasoning_effort" else name: value
            for name, value in request_body.items()
        }
        if "temperature" in request_body:
            model_entry["temperature"] = True
        provider_options["body"] = request_body
    return {
        "npm": "@ai-sdk/openai-compatible",
        "name": "agent-switch",
        "options": provider_options,
        "models": {model["id"]: model_entry},
    }


def _parse_jsonc(text: str) -> dict:
    """Parse the JSONC OpenCode accepts: drop // and /* */ comments outside strings and trailing commas."""
    kept: list[str] = []
    in_string = in_block = escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_block:
            if char == "*" and text[index + 1 : index + 2] == "/":
                in_block = False
                index += 2
            else:
                index += 1
            continue
        if in_string:
            kept.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
        elif char == "/" and text[index + 1 : index + 2] == "/":
            while index < len(text) and text[index] != "\n":
                index += 1
            continue
        elif char == "/" and text[index + 1 : index + 2] == "*":
            in_block = True
            index += 2
            continue
        elif char in "}]":
            # Drop a trailing comma before the closer; a regex could corrupt a string holding ", }".
            while kept and kept[-1].isspace():
                kept.pop()
            if kept and kept[-1] == ",":
                kept.pop()
        kept.append(char)
        index += 1
    return json.loads("".join(kept))


def _opencode_global_mcp() -> dict:
    """The mcp entries of the user's global opencode configs.

    The session overlay outranks the global config but cannot delete a global entry, so an
    unmounted one is disabled by a copy of itself (a bare {"enabled": false} can fail the
    per-layer schema check). OpenCode loads config.json, opencode.json and opencode.jsonc
    from its global dir, a later file winning per server, so all three are read here;
    disabling a name OpenCode never loads is harmless.
    """
    config_home = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    directory = Path(config_home) / "opencode"
    servers: dict = {}
    for name in ("config.json", "opencode.json", "opencode.jsonc"):
        path = directory / name
        if not path.is_file():
            continue
        try:
            data = _parse_jsonc(path.read_text(encoding = "utf-8"))
        except (ValueError, OSError):
            # Still unparseable after the JSONC handling; leave its servers on.
            typer.echo(
                f"Warning: couldn't parse {path}; its MCP servers stay enabled in this session.",
                err = True,
            )
            continue
        if isinstance(data, dict) and isinstance(data.get("mcp"), dict):
            servers.update(data["mcp"])
    return servers


def _opencode_mcp_entries(mcp_servers: dict) -> dict:
    """The opencode mcp map for the mounted servers."""
    entries: dict = {}
    for name, server in mcp_servers.items():
        if server["transport"] == "stdio":
            entries[name] = {
                "type": "local",
                "command": [server["command"], *server["args"]],
                "environment": server["env"],
                "enabled": True,
            }
        else:
            entries[name] = {
                "type": "remote",
                "url": server["url"],
                "headers": server["headers"],
                "enabled": True,
            }
    return entries


def write_opencode_config(
    base: str,
    key: str,
    model: dict,
    path: Path,
    yolo: bool = False,
    as_subagent: bool = False,
    max_tokens: Optional[int] = None,
    request_body: Optional[dict] = None,
    headers: Optional[dict] = None,
    compact_at: Optional[float] = None,
    mcp_servers: Optional[dict] = None,
) -> dict:
    config = _read_json_object(path)
    if config is None:
        typer.echo(
            f"Warning: couldn't parse {path} — add an '{_OPENCODE_PROVIDER}' provider "
            "there yourself, or move the file aside and re-run.",
            err = True,
        )
        return {}
    before = json.dumps(config, sort_keys = True)
    config.setdefault("$schema", "https://opencode.ai/config.json")
    window = model.get("context_length") or model.get("max_context_length")
    reserved = None
    if window:
        window = int(window)
        reserved = opencode_compaction_reserved(
            window, opencode_output_limit(window, max_tokens), compact_at
        )
    # Keep the provider definition in this private session file. The launch path adjusts effective provider filters in the higher-priority inline overlay.
    _subdict(config, "provider")[_OPENCODE_PROVIDER] = _opencode_provider(
        base, key, model, max_tokens, request_body, headers
    )
    # --mcp/--mcp-all rewrite the mcp map every run: mounted servers win by name in this
    # overlay, and every unmounted entry of the user's global config is copied with
    # enabled: false (this layer cannot delete a lower-layer entry). Without MCP flags the
    # key is dropped, so a persisted session keeps no earlier mount.
    if mcp_servers is None:
        config.pop("mcp", None)
    else:
        mounted: dict = _opencode_mcp_entries(mcp_servers)
        for name, entry in _opencode_global_mcp().items():
            if name not in mounted and isinstance(entry, dict):
                mounted[name] = {**entry, "enabled": False}
        config["mcp"] = mounted
    # Normal mode pins this as the session model. Subagent mode leaves the user's main/small models alone and exposes the local model to @local and /models.
    opencode_model = f"{_OPENCODE_PROVIDER}/{model['id']}"
    if as_subagent:
        for field in ("model", "small_model"):
            if str(config.get(field) or "").startswith(f"{_OPENCODE_PROVIDER}/"):
                config.pop(field, None)
        # Drop a managed compaction block, current or legacy value.
        managed = {reserved, max(1, window // 10)} if window else set()
        compaction = config.get("compaction")
        if (
            isinstance(compaction, dict)
            and compaction.keys() == {"auto", "reserved"}
            and compaction["auto"] is True
            and compaction["reserved"] in managed
        ):
            config.pop("compaction", None)
        _subdict(config, "agent")[_SUBAGENT_NAME] = {
            "description": _SUBAGENT_DESCRIPTION,
            "mode": "subagent",
            "model": opencode_model,
            "prompt": _SUBAGENT_INSTRUCTIONS,
        }
    else:
        config["model"] = opencode_model
        agents = config.get("agent")
        if isinstance(agents, dict):
            agents.pop(_SUBAGENT_NAME, None)
            if not agents:
                config.pop("agent", None)
    if window and not as_subagent:
        # The fixed 20k-token default buffer over-compacts, or never settles, on a small local context.
        compaction = _subdict(config, "compaction")
        compaction["auto"] = True
        compaction["reserved"] = reserved
    tools = ("edit", "bash", "webfetch", *(("task",) if as_subagent else ()))
    if yolo:
        # Fallback for commands without native --auto and for the append-safe bare --no-launch command, where the subcommand is not known yet. Rides inline (OPENCODE_CONFIG_CONTENT) so it wins over a project config. TUI and `run` launches use --auto and call here with yolo=False, letting OpenCode preserve explicit deny rules.
        session_permission = {t: "allow" for t in tools}
        session_permission["external_directory"] = {"*": "allow"}
        config["permission"] = dict(session_permission)
    else:
        # Undo only what --yolo wrote: our yolo sets an explicit per-tool "allow" for these three tools, so flip exactly those explicit allows back to "ask". A "deny"/"ask", a granular object, a string, or a "*" catch-all is the user's own rule and is left untouched. We do NOT carry a permission inline for a non-yolo session: since OPENCODE_CONFIG_CONTENT outranks the project opencode.json we cannot read, any value forced there would override the user's project rules, weakening a project deny or auto-approving through a granular object's permissive default. Clearing our own persisted yolo state is the fix.
        session_permission: dict = {}
        permission = config.get("permission")
        if isinstance(permission, dict):
            for tool in tools:
                if permission.get(tool) == "allow":
                    permission[tool] = "ask"
            if permission.get("external_directory") == {"*": "allow"}:
                permission["external_directory"] = {"*": "ask"}
    if json.dumps(config, sort_keys = True) != before:
        _write_private_json(path, config)
        typer.echo(f"Updated {path}")
    return session_permission


def opencode(
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
    max_tokens: Optional[int] = _MAX_TOKENS_OPTION,
    compact_at: Optional[float] = _COMPACT_AT_OPTION,
    model_load: bool = _MODEL_LOAD_OPTION,
    url: Optional[str] = _URL_OPTION,
    provider: Optional[ProviderName] = _PROVIDER_OPTION,
    yolo: bool = _YOLO_OPTION,
    persist: bool = _PERSIST_OPTION,
    as_subagent: bool = _AS_SUBAGENT_OPTION,
    mcp: Optional[list[str]] = _MCP_OPTION,
    mcp_all: bool = _MCP_ALL_OPTION,
    mcp_url: Optional[list[str]] = _MCP_URL_OPTION,
    mcp_oauth_url: Optional[list[str]] = _MCP_OAUTH_URL_OPTION,
    mcp_header: Optional[list[str]] = _MCP_HEADER_OPTION,
    mcp_stdio: Optional[list[str]] = _MCP_STDIO_OPTION,
    mcp_env: Optional[list[str]] = _MCP_ENV_OPTION,
):
    """Point OpenCode at a local model server and start it."""
    # Route a leading `org/name` positional to --model; forward the rest to the agent.
    model, ctx.args[:] = _consume_positional_model(model, ctx.args)
    headers = parse_headers(header)
    # Validate the MCP selection before _connect, so a registry error fails fast.
    mcp_servers = load_mcp_servers(
        mcp,
        should_mount_all = mcp_all,
        urls = mcp_url,
        oauth_urls = mcp_oauth_url,
        headers = mcp_header,
        stdios = mcp_stdio,
        envs = mcp_env,
        as_subagent = as_subagent,
    )
    target = _resolve_target(url, provider, api_key, headers)
    if target is None:
        _refuse_local_only(
            model = model,
            max_seq_length = max_seq_length,
            max_tokens = max_tokens,
            reasoning = reasoning,
            reasoning_effort = reasoning_effort,
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            min_p = min_p,
            repetition_penalty = repetition_penalty,
            presence_penalty = presence_penalty,
            compact_at = compact_at,
            api_key = api_key,
            header = header,
            model_load = model_load,
            as_subagent = as_subagent,
            persist = persist,
        )
    alias = ctx.meta.get(_ALIAS_META)
    if alias and as_subagent:
        _fail("--as-subagent is not supported for agent aliases.")
    command_name, opencode_v2 = _opencode_command()
    install_hint = _npm_install_hint("@opencode-ai/cli@beta" if opencode_v2 else "opencode-ai")
    if alias:
        _check_alias(alias)
    else:
        _require_agent_for_launch(command_name, install_hint, launch)
    if target is None:
        # Native launch: opencode keeps its own model, login and config; the session config
        # carries only the mounted MCP servers, and OPENCODE_CONFIG is the only env change.
        route_native_auto = yolo and _opencode_supports_native_auto(command_name)
        opencode_args = list(ctx.args)
        if opencode_v2:
            opencode_args = _opencode_v2_standalone_args(opencode_args)
        opencode_args, _ = _opencode_native_auto_args(
            opencode_args, route_native_auto, v2 = opencode_v2
        )
        command = _agent_command(alias, command_name, opencode_args)
        with _session_config("opencode-native", launch) as cfg:
            env = {}
            config_path = cfg / "opencode.json"
            if mcp_servers:
                _write_private_json(
                    config_path,
                    {
                        "$schema": "https://opencode.ai/config.json",
                        "mcp": _opencode_mcp_entries(mcp_servers),
                    },
                )
                env["OPENCODE_CONFIG"] = str(config_path)
            else:
                # Stable --no-launch dir: drop an earlier mount so a bare rerun stays native.
                config_path.unlink(missing_ok = True)
            _run_native(
                alias or "opencode",
                env,
                command,
                launch = launch,
                install_hint = install_hint,
            )
        return
    server_options = ServerOptions(
        reasoning = reasoning,
        reasoning_effort = reasoning_effort,
        temperature = temperature,
        top_p = top_p,
        top_k = top_k,
        min_p = min_p,
        repetition_penalty = repetition_penalty,
        presence_penalty = presence_penalty,
        carried = _ALL_REQUEST_FIELDS,
        provider = target.name,
    )
    base, key, entry = _connect(
        api_key,
        model,
        LoadOptions(max_seq_length, model_load),
        server_options = server_options,
        target = target,
    )
    _check_compact_at(compact_at, entry)
    if opencode_v2:
        typer.echo(
            f"OpenCode V2 provider policies must allow '{_OPENCODE_PROVIDER}'.",
            err = True,
        )
    if as_subagent:
        if compact_at is not None:
            typer.echo(
                "Warning: --compact-at does not apply with --as-subagent for OpenCode; ignoring it.",
                err = True,
            )
        # Stay append-safe for a bare no-launch recipe: a later `run <prompt>` would make `opencode --auto run ...` parse as the TUI, so keep yolo in the inline fallback.
        route_native_auto = (
            yolo and _opencode_supports_native_auto(command_name) and (launch or bool(ctx.args))
        )
        opencode_args = list(ctx.args)
        if opencode_v2:
            opencode_args = _opencode_v2_standalone_args(opencode_args)
        opencode_args, native_auto = _opencode_native_auto_args(
            opencode_args, route_native_auto, v2 = opencode_v2
        )
        command = [command_name, *opencode_args]
        with _session_config("opencode-subagent", launch, persist = persist) as cfg:
            config_path = cfg / "opencode.json"
            session_permission = write_opencode_config(
                base,
                key,
                entry,
                config_path,
                yolo = yolo and not native_auto,
                as_subagent = True,
                max_tokens = max_tokens,
                request_body = server_options.request_body(),
                headers = headers,
            )
            env = {
                "OPENCODE_CONFIG": str(config_path),
                **_opencode_output_env(entry, max_tokens),
            }
            inline_config = _opencode_subagent_inline_config(
                config_path,
                session_permission,
                command = command_name,
                v2 = opencode_v2,
            )
            # A project opencode.json outranks the session file and could field-merge its own agent.local over ours. Pin ours in the inline overlay so it wins.
            inline_config.setdefault("agent", {})[_SUBAGENT_NAME] = {
                "description": _SUBAGENT_DESCRIPTION,
                "mode": "subagent",
                "model": f"{_OPENCODE_PROVIDER}/{entry['id']}",
                "prompt": _SUBAGENT_INSTRUCTIONS,
            }
            env["OPENCODE_CONFIG_CONTENT"] = json.dumps(inline_config)
            typer.echo(f"The local model is available as @{_SUBAGENT_NAME} and in /models.")
            _run(
                base,
                entry,
                env,
                command,
                launch = launch,
                install_hint = install_hint,
            )
        return
    opencode_model = f"{_OPENCODE_PROVIDER}/{entry['id']}"
    # The inline OPENCODE_CONFIG_CONTENT below pins the model in the highest-priority layer, so the session model is forced without a --model flag. Only add --model for an interactive bare launch, as a convenience so the TUI opens on our model: it is omitted for passthrough, where inserting it before a subcommand can be misparsed, and for --no-launch, where the printed command is consumed by drivers that append a subcommand such as `run <prompt>` and a leading --model would land before it. Those paths rely on the inline pin instead.
    native_auto = False
    route_native_auto = yolo and _opencode_supports_native_auto(command_name)
    if ctx.args:
        opencode_args = list(ctx.args)
        if opencode_v2:
            opencode_args = _opencode_v2_standalone_args(opencode_args)
        opencode_args, native_auto = _opencode_native_auto_args(
            opencode_args, route_native_auto, v2 = opencode_v2
        )
        command = _agent_command(alias, command_name, opencode_args)
    elif launch:
        opencode_args = [] if opencode_v2 else ["--model", opencode_model]
        if opencode_v2:
            opencode_args = _opencode_v2_standalone_args(opencode_args)
        opencode_args, native_auto = _opencode_native_auto_args(
            opencode_args,
            route_native_auto,
            v2 = opencode_v2,
        )
        command = _agent_command(alias, command_name, opencode_args)
    else:
        # Append-safe base: `opencode --auto run ...` parses as the TUI with a project "run", not the run subcommand. The command is unknown here, so keep the config fallback.
        opencode_args = _opencode_v2_standalone_args([]) if opencode_v2 else []
        command = _agent_command(alias, command_name, opencode_args)
    # opencode keeps sessions in ~/.local/share/opencode (never relocated), so resume already survives exit; reopen the last one by passing `opencode --continue` through.
    with _session_config("opencode", launch, persist = persist) as cfg:
        config_path = cfg / "opencode.json"
        # OPENCODE_CONFIG is an overlay, loaded between the user's global and project configs, so this adds the agent-switch provider/model for the session without changing the user's default model. The key lives in the config, not the env.
        session_permission = write_opencode_config(
            base,
            key,
            entry,
            config_path,
            yolo = yolo and not native_auto,
            max_tokens = max_tokens,
            request_body = server_options.request_body(),
            headers = headers,
            compact_at = compact_at,
            mcp_servers = mcp_servers,
        )
        # A project's own opencode.json outranks OPENCODE_CONFIG, so the session model pin would silently lose to a repo config; carry it in OPENCODE_CONFIG_CONTENT, which outranks project config, while the API key stays in the private file. Only the config fallback carries a permission: native --auto omits it (auto-approve asks, keep explicit denies) and a non-yolo session omits it too, honoring project rules. V1 filters are ordinary overlays, so scope that session to our provider; V2 turns filters into security policies where global/project rules intentionally win, so keep those policies intact and tell the user above that they must allow our provider. small_model is opencode's separate model for lightweight tasks; pin it to the session model too, or a user/project small_model on another (now filtered) provider would resolve a not-found error mid-session.
        inline_config: dict = {
            "model": opencode_model,
            "small_model": opencode_model,
        }
        if not opencode_v2:
            inline_config["enabled_providers"] = [_OPENCODE_PROVIDER]
            inline_config["disabled_providers"] = []
        if session_permission:
            inline_config["permission"] = session_permission
        env = {
            "OPENCODE_CONFIG": str(config_path),
            "OPENCODE_CONFIG_CONTENT": json.dumps(inline_config),
            **_opencode_output_env(entry, max_tokens),
        }
        _run(base, entry, env, command, launch = launch, install_hint = install_hint)
