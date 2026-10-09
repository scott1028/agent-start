# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`agent-switch pi`: Pi config, compaction, user resources and subagent config."""

import json
import os
from pathlib import Path
from typing import Literal, Optional

import typer

from agent_switch.core.install import (
    _agent_version_at_least,
    _npm_install_hint,
    _require_agent_for_launch,
)
from agent_switch.core.launch import _connect, _resolve_target, _run
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
    _MIN_P_OPTION,
    _MODEL_LOAD_OPTION,
    _MODEL_OPTION,
    _PERSIST_OPTION,
    _PRESENCE_PENALTY_OPTION,
    _PROVIDER_OPTION,
    _REASONING_EFFORT_OPTION,
    _REASONING_OPTION,
    _REPETITION_PENALTY_OPTION,
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
    _yolo_command_flags,
    parse_headers,
)
from agent_switch.core.platform import (
    _create_directory_junction,
    _is_junction,
    _remove_overlay_entry,
    _wsl_windows_executable,
)
from agent_switch.core.session import _agent_config_path, _session_config
from agent_switch.core.storage import _read_json_object, _subdict, _write_private_json


_PI_PROVIDER = "agent-switch"


_PI_SUBAGENT_EXTENSION = Path(__file__).parent / "pi_subagent.ts"


_PI_USER_RESOURCE_DIRS = ("extensions", "skills", "prompts", "themes", "npm", "git")


_PI_USER_RESOURCE_SETTINGS = ("packages", "extensions", "skills", "prompts", "themes")


_PI_USER_VERBATIM_SETTINGS = ("npmCommand",)


_PI_USER_RESOURCES_MANIFEST = ".agent-switch-user-resources.json"


_PI_SAMPLING_PARAMS_MIN_VERSION = (0, 84, 0)


def _pi_header_values(headers: dict) -> dict:
    """Pi resolves $NAME and !command in header values; doubling $ keeps literal values intact,
    and a leading ! is prefixed with $ so pi's $! escape makes it literal instead of a shell command."""
    escaped = {}
    for name, value in headers.items():
        value = value.replace("$", "$$")
        escaped[name] = f"${value}" if value.startswith("!") else value
    return escaped


def write_pi_config(
    base: str,
    key: str,
    model: dict,
    path: Path,
    *,
    max_tokens: Optional[int] = None,
    request_body: Optional[dict] = None,
    headers: Optional[dict] = None,
) -> None:
    config = _read_json_object(path)
    if config is None:
        typer.echo(
            f"Warning: couldn't parse {path} — add an '{_PI_PROVIDER}' provider there "
            "yourself, or move the file aside and re-run.",
            err = True,
        )
        return
    before = json.dumps(config, sort_keys = True)
    # Pi reads custom providers from ~/.pi/agent/models.json (HOME-relocated for the session). The server is a generic OpenAI-compatible /v1 endpoint, and the key lives in the config rather than the env, matching opencode.
    provider_model = {"id": model["id"]}
    window = model.get("context_length") or model.get("max_context_length")
    if window:
        window = int(window)
        # An unspecified model defaults to contextWindow 128000 / maxTokens 16384, far larger than a small local context, so Pi compacts too late and overflows the server. Pin the real window and a sane output cap, mirroring OpenCode.
        provider_model["contextWindow"] = window
        provider_model["maxTokens"] = _agent_output_limit(window, max_tokens)
    elif max_tokens:
        provider_model["maxTokens"] = max_tokens
    if request_body:
        provider_model["samplingParams"] = request_body
    provider_entry = {
        "api": "openai-completions",
        "baseUrl": f"{base}/v1",
        # pi refuses the prompt without an apiKey ("No API key found for ..."); its OpenAI SDK merges
        # the custom Authorization over the Bearer later (case-insensitively), so one header wins.
        "apiKey": key,
        **({"headers": _pi_header_values(headers)} if headers else {}),
        "models": [provider_model],
    }
    _subdict(config, "providers")[_PI_PROVIDER] = provider_entry
    if json.dumps(config, sort_keys = True) != before:
        _write_private_json(path, config)
        typer.echo(f"Updated {path}")


def write_pi_compaction(agent_dir: Path, model: dict, compact_at: Optional[float]) -> None:
    """Scale Pi's auto-compaction trigger to --compact-at in the session settings."""
    path = agent_dir / "settings.json"
    settings = _read_json_object(path)
    if settings is None:
        return  # write_pi_user_resources already warned
    window = model.get("context_length") or model.get("max_context_length")
    before = json.dumps(settings, sort_keys = True)
    if compact_at is not None and window:
        # Pi compacts once the context exceeds contextWindow - reserveTokens.
        settings["compaction"] = {
            "enabled": True,
            "reserveTokens": _get_compaction_reserve(int(window), compact_at),
        }
    else:
        # Undo a --compact-at block an earlier run left in a persisted session; compaction
        # is not a key this session inherits from the user, so this exact shape is ours.
        compaction = settings.get("compaction")
        if (
            isinstance(compaction, dict)
            and compaction.keys() == {"enabled", "reserveTokens"}
            and compaction["enabled"] is True
            and isinstance(compaction["reserveTokens"], int)
        ):
            settings.pop("compaction", None)
    if json.dumps(settings, sort_keys = True) != before:
        _write_private_json(path, settings)


def _link_user_dir(source: Path, target: Path) -> bool:
    """Expose source at target. True once target resolves to source."""
    # Refresh links, but preserve real session directories.
    if target.is_symlink() or _is_junction(target):
        _remove_overlay_entry(target)
    if target.exists() or not source.is_dir():
        return False
    target.parent.mkdir(parents = True, exist_ok = True, mode = 0o700)
    try:
        target.symlink_to(source, target_is_directory = True)
    except OSError:
        if not _create_directory_junction(source, target):
            typer.echo(f"Warning: couldn't link {source} into the Pi session.", err = True)
            return False
    return True


def _pi_local_entry(
    entry: str,
    source: Path,
    home: Path,
    linked: frozenset,
    agents_skills = None,
) -> str:
    """Re-anchor a user path from the original Pi agent directory."""
    value = entry.strip()
    if not value or value == "." or value.startswith("file:"):
        # Nothing to anchor: "" and "." would name the whole agent directory.
        return entry
    if value == "~" or value.startswith(("~/", "~" + os.sep)):
        target = os.path.join(home, value[2:])
    else:
        # Pi stores local packages relative to its agent directory.
        target = os.path.join(source, value)
    target = os.path.normpath(target)
    if agents_skills is not None:
        # Pi reads ~/.agents/skills through HOME, which moved, so a rule naming the
        # user's copy must follow it or it stops matching.
        user_root, session_root = agents_skills
        try:
            inside = os.path.relpath(target, user_root)
        except ValueError:  # on another Windows drive
            inside = os.pardir
        if inside == os.curdir:
            return session_root
        if inside != os.pardir and not inside.startswith(os.pardir + os.sep):
            return os.path.join(session_root, inside)
    try:
        relative = os.path.relpath(target, source)
    except ValueError:  # on another Windows drive
        return target
    # Session-relative only where the link landed: a real session directory blocks
    # the link, and the entry would then point into it instead of at the user's.
    if relative.split(os.sep)[0] in linked:
        return relative
    return target


def _pi_settings_entries(
    key: str,
    entries,
    source: Path,
    home: Path,
    linked: frozenset,
    agents_skills = None,
) -> list:
    if not isinstance(entries, list):
        return []
    result = []
    for entry in entries:
        if key == "packages":
            spec = entry.get("source") if isinstance(entry, dict) else entry
            # All other package sources are local paths.
            if isinstance(spec, str) and not spec.strip().startswith(
                ("npm:", "git:", "github:", "http:", "https:", "ssh:")
            ):
                spec = _pi_local_entry(spec, source, home, linked, agents_skills)
                entry = {**entry, "source": spec} if isinstance(entry, dict) else spec
        elif isinstance(entry, str):
            prefix = entry[:1] if entry.startswith(("!", "+", "-")) else ""
            pattern = entry[len(prefix) :]
            if not prefix and "*" not in entry and "?" not in entry:
                entry = _pi_local_entry(entry, source, home, linked, agents_skills)
            elif not pattern.strip().startswith("~"):  # Pi does not expand ~ in patterns
                # Pi matches patterns against paths relative to the agent directory, which moved.
                # Keep the original too: it still matches basenames and linked directories.
                anchored = prefix + _pi_local_entry(
                    pattern,
                    source,
                    home,
                    linked,
                    agents_skills,
                )
                if anchored != entry:
                    result.append(entry)
                    entry = anchored
        result.append(entry)
    return result


def _clear_pi_user_resources(agent_dir: Path, home: Path) -> None:
    """Undo what an earlier launch linked and copied, leaving session state alone."""
    targets = [agent_dir / name for name in _PI_USER_RESOURCE_DIRS]
    targets.append(home / ".agents" / "skills")
    for target in targets:
        if target.is_symlink() or _is_junction(target):
            _remove_overlay_entry(target)
    manifest_path = agent_dir / _PI_USER_RESOURCES_MANIFEST
    previous = _read_json_object(manifest_path)
    if not previous:
        return
    settings_path = agent_dir / "settings.json"
    settings = _read_json_object(settings_path)
    if settings is None:
        return
    before = json.dumps(settings, sort_keys = True)
    for key, copied in previous.items():
        own = settings.get(key)
        if key in _PI_USER_VERBATIM_SETTINGS:
            # An argument vector, not entries: subtracting drops whatever the two share.
            if own == copied:
                settings.pop(key, None)
        elif isinstance(copied, list) and isinstance(own, list):
            rest = [item for item in own if item not in copied]
            if rest:
                settings[key] = rest
            else:
                settings.pop(key, None)
        elif own == copied:
            settings.pop(key, None)
    if json.dumps(settings, sort_keys = True) != before:
        _write_private_json(settings_path, settings)
    manifest_path.unlink(missing_ok = True)


def write_pi_user_resources(agent_dir: Path, home: Path) -> None:
    """Expose selected user Pi resources inside an isolated session."""
    if _wsl_windows_executable(["pi"]):
        # Windows Pi cannot reliably follow WSL links into mounted drives, and a session
        # an earlier Linux pi prepared still holds them, so drop those before returning.
        _clear_pi_user_resources(agent_dir, home)
        return
    user_home = Path.home()
    configured = os.environ.get("PI_CODING_AGENT_DIR")
    configured = configured.strip() if configured else ""
    # Pi resolves a relative override from the launch directory.
    source = (
        Path(os.path.abspath(os.path.expanduser(configured)))
        if configured
        else user_home / ".pi" / "agent"
    )
    if source.resolve(strict = False) == agent_dir.resolve(strict = False):
        # Do not treat this session as its own resource source.
        source = user_home / ".pi" / "agent"
    if configured and not source.is_dir():
        # Otherwise this looks exactly like the bug this function exists to fix.
        typer.echo(
            f"Warning: PI_CODING_AGENT_DIR points at {source}, which is not a directory; "
            "no Pi extensions or packages will load in this session.",
            err = True,
        )
    linked = frozenset(
        name for name in _PI_USER_RESOURCE_DIRS if _link_user_dir(source / name, agent_dir / name)
    )
    # HOME is relocated, so link Pi's other global skill directory too.
    user_skills = user_home / ".agents" / "skills"
    session_skills = home / ".agents" / "skills"
    agents_skills = (
        (str(user_skills), str(session_skills))
        if _link_user_dir(user_skills, session_skills)
        else None
    )

    user_settings_path = source / "settings.json"
    user_settings = _read_json_object(user_settings_path)
    if user_settings is None:
        typer.echo(
            f"Warning: couldn't parse {user_settings_path}; "
            "Pi packages listed there won't load in this session.",
            err = True,
        )
        user_settings = {}
    settings_path = agent_dir / "settings.json"
    settings = _read_json_object(settings_path)
    if settings is None:
        typer.echo(
            f"Warning: couldn't parse {settings_path}; your Pi packages won't load in this session.",
            err = True,
        )
        return
    manifest_path = agent_dir / _PI_USER_RESOURCES_MANIFEST
    previous = _read_json_object(manifest_path)
    if previous is None:
        # Provenance is lost, so entries the user has since removed cannot be reconciled.
        typer.echo(
            f"Warning: couldn't parse {manifest_path}; Pi resources copied by an earlier "
            "launch stay in this session even if you removed them since.",
            err = True,
        )
        previous = {}
    before = json.dumps(settings, sort_keys = True)
    copied = {}
    for key in _PI_USER_RESOURCE_SETTINGS:
        entries = _pi_settings_entries(
            key,
            user_settings.get(key),
            source,
            user_home,
            linked,
            agents_skills,
        )
        # Refresh copied entries while preserving settings added inside the session.
        stale = previous.get(key) if isinstance(previous.get(key), list) else []
        own = settings.get(key)
        if own is not None and not isinstance(own, list):
            # Pi types these as arrays; leave a shape we do not understand alone.
            typer.echo(
                f"Warning: {settings_path} has a non-list {key!r}; "
                "leaving it as is, so your Pi entries for it won't load in this session.",
                err = True,
            )
            continue
        own = [item for item in own or [] if item not in stale and item not in entries]
        if entries or own:
            # Pi de-dupes packages by identity keeping the FIRST, so session entries
            # lead. Patterns apply in order instead, so those stay user-first.
            settings[key] = own + entries if key == "packages" else entries + own
        else:
            settings.pop(key, None)
        if entries:
            copied[key] = entries
    for key in _PI_USER_VERBATIM_SETTINGS:
        # Pi runs every package lookup and install through npmCommand, so inheriting
        # the package list without it falls back to an npm that cannot find them.
        value = user_settings.get(key)
        own = settings.get(key)
        if own is not None and own != previous.get(key):
            continue  # changed inside the session, so the session owns it now
        if isinstance(value, list) and value and all(isinstance(arg, str) for arg in value):
            settings[key] = value
            copied[key] = value
        else:
            settings.pop(key, None)
    if json.dumps(settings, sort_keys = True) != before:
        _write_private_json(settings_path, settings)
    if copied != previous:
        if copied:
            _write_private_json(manifest_path, copied)
        else:
            manifest_path.unlink(missing_ok = True)


def write_pi_subagent_config(
    base: str,
    key: str,
    model: dict,
    path: Path,
    approve: bool = False,
    max_tokens: Optional[int] = None,
    request_body: Optional[dict] = None,
    headers: Optional[dict] = None,
) -> None:
    """Write private bootstrap data for the bundled Pi extension."""
    window = model.get("context_length") or model.get("max_context_length")
    window = int(window) if window else 32768
    _write_private_json(
        path,
        {
            "baseUrl": f"{base}/v1",
            "apiKey": key,
            "model": model["id"],
            "contextWindow": window,
            "maxTokens": _agent_output_limit(window, max_tokens),
            "approve": approve,
            **({"headers": _pi_header_values(headers)} if headers else {}),
            **({"samplingParams": request_body} if request_body else {}),
        },
    )


def pi(
    ctx: typer.Context,
    model: Optional[str] = _MODEL_OPTION,
    api_key: Optional[str] = _KEY_OPTION,
    header: Optional[list[str]] = _HEADER_OPTION,
    launch: bool = _LAUNCH_OPTION,
    max_seq_length: int = _CONTEXT_OPTION,
    max_tokens: Optional[int] = _MAX_TOKENS_OPTION,
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
):
    """Point Pi (coding agent) at a local model server and start it."""
    # Route a leading `org/name` positional to --model; forward the rest to the agent.
    model, ctx.args[:] = _consume_positional_model(model, ctx.args)
    headers = parse_headers(header)
    target = _resolve_target(url, provider, api_key, headers)
    install_hint = _npm_install_hint(
        "@earendil-works/pi-coding-agent",
        ignore_scripts = True,
    )
    if as_subagent and not _PI_SUBAGENT_EXTENSION.is_file():
        _fail(f"Missing Pi subagent extension: {_PI_SUBAGENT_EXTENSION}")
    _require_agent_for_launch("pi", install_hint, launch)
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
    if server_options.request_body() and not _agent_version_at_least(
        "pi", _PI_SAMPLING_PARAMS_MIN_VERSION
    ):
        server_options = server_options._replace(carried = frozenset())
    base, key, entry = _connect(
        api_key,
        model,
        LoadOptions(max_seq_length, model_load),
        server_options = server_options,
        target = target,
    )
    _check_compact_at(compact_at, entry)
    if as_subagent:
        if compact_at is not None:
            typer.echo(
                "Warning: --compact-at does not apply with --as-subagent for Pi; ignoring it.",
                err = True,
            )
        extension = _agent_config_path(_PI_SUBAGENT_EXTENSION, ["pi"])
        with _session_config("pi-subagent", launch, persist = persist) as config:
            config_path = config / "subagent.json"
            write_pi_subagent_config(
                base,
                key,
                entry,
                config_path,
                approve = yolo,
                max_tokens = max_tokens,
                request_body = server_options.request_body(),
                headers = headers,
            )
            command = [
                "pi",
                "--extension",
                extension,
                *_yolo_command_flags("pi", yolo),
                *ctx.args,
            ]
            typer.echo(
                "A local agent is available, and the model is in /model. "
                "Ask Pi to spawn a local agent."
            )
            _run(
                base,
                entry,
                {"AGENT_SWITCH_PI_SUBAGENT_CONFIG": str(config_path)},
                command,
                launch = launch,
                install_hint = install_hint,
                clear_screen = True,
            )
        return
    # Pi defaults to the google provider, so pin our provider/model on the command line; the custom OpenAI-compatible endpoint itself is only configurable via ~/.pi/agent/models.json.
    command = [
        "pi",
        "--provider",
        _PI_PROVIDER,
        "--model",
        entry["id"],
        *_yolo_command_flags("pi", yolo),
        *ctx.args,
    ]
    # --ignore-scripts matches Pi's documented install recipe (its README notes Pi needs no install scripts), so accepting the prompt skips dependency lifecycle scripts.
    with _session_config("pi", launch, persist = persist) as home:
        # Pi resolves its config dir from PI_CODING_AGENT_DIR first (getAgentDir() prefers it over $HOME/.pi/agent), so pin it at the session dir: an inherited PI_CODING_AGENT_DIR in the user's shell would otherwise send Pi to their real config and skip our provider/key. HOME is relocated too so any other ~/.pi paths stay in the session. The key rides in the config rather than the env.
        pi_agent_dir = home / ".pi" / "agent"
        write_pi_config(
            base,
            key,
            entry,
            pi_agent_dir / "models.json",
            max_tokens = max_tokens,
            request_body = server_options.request_body(),
            headers = headers,
        )
        write_pi_user_resources(pi_agent_dir, home)
        write_pi_compaction(pi_agent_dir, entry, compact_at)
        env = {"HOME": str(home), "PI_CODING_AGENT_DIR": str(pi_agent_dir)}
        if os.name == "nt" or os.environ.get("WSL_DISTRO_NAME"):
            # Node resolves ~/.pi via USERPROFILE (then HOMEDRIVE + HOMEPATH) on Windows, not HOME. Set them whenever Pi may run as a Windows process: native Windows, or a /mnt Windows shim launched from WSL, where the WSLENV bridge then translates the path. Otherwise the Windows process falls back to the user's real %USERPROFILE%\\.pi. splitdrive yields no drive off a POSIX path, so HOMEDRIVE/HOMEPATH stay unset there.
            env["USERPROFILE"] = str(home)
            drive, tail = os.path.splitdrive(str(home))
            if drive:
                env["HOMEDRIVE"], env["HOMEPATH"] = drive, tail
        # Pi paints inline from the current cursor position (no alternate screen, no clear on first render), so give it the clean screen it assumes.
        _run(
            base,
            entry,
            env,
            command,
            launch = launch,
            install_hint = install_hint,
            clear_screen = True,
        )
