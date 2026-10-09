# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`agent-switch dsh` / `dsh-tui` / `dst`: DeepSeek Harness patch, TUI guards and state."""

import os
import re
import shutil
import sys
from pathlib import Path
from typing import Literal, Optional

import typer

from agent_switch import providers
from agent_switch._coding_agents import get_is_deepseek_harness_tui_executable
from agent_switch.core.install import (
    _get_augmented_path,
    _install_agent,
    _npm_install_hint,
    _prefer_windows_cmd_sibling,
    _require_agent_for_launch,
    _resolve_or_install_agent,
    _which_with_install_dirs,
)
from agent_switch.core.launch import _connect, _resolve_target, _run
from agent_switch.core.options import (
    LoadOptions,
    ProviderName,
    ServerOptions,
    _COMPACT_AT_OPTION,
    _CONTEXT_OPTION,
    _HEADER_OPTION,
    _KEY_OPTION,
    _LAUNCH_OPTION,
    _MAX_TOKENS_OPTION,
    _MIN_P_OPTION,
    _MODEL_LOAD_OPTION,
    _MODEL_OPTION,
    _PANEL_SESSION,
    _PRESENCE_PENALTY_OPTION,
    _PROVIDER_OPTION,
    _REASONING_EFFORT_OPTION,
    _REASONING_FIELDS,
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
    parse_headers,
)
from agent_switch.core.platform import (
    _create_directory_junction,
    _remove_overlay_entry,
    _wsl_windows_executable,
)
from agent_switch.core.session import (
    _agent_config_path,
    _agents_config_root,
    _ephemeral_session_prefix,
    _session_config,
)
from agent_switch.core.storage import _read_json_object, _write_private_json, _write_private_text


_DSH_PROVIDER = "agent-switch"


_DSH_ENV_KEY = "AGENT_SWITCH_API_KEY"


_DSH_PATCH_FILE = "agent-switch.patch.yml"


_DSH_PACKAGE = "@deepseek-ai/dsh"


# dsh picks its sandbox+approval preset from DSH_PERMISSION_MODE via ??, so omitting it would inherit a danger-full-access exported in the parent shell, and "" is not unset to ??. Pin the mode in both directions instead of only setting it for --yolo.
_DSH_SAFE_PERMISSION_MODE = "workspace-write"


_DSH_YOLO_PERMISSION_MODE = "danger-full-access"


_DSH_TUI_PACKAGE = "@deepseek-harness-tui/dsh-tui"


# `dsh-tui` and its `dst` alias are one integration with one session home; dsh-tui is preferred.
_DSH_TUI_AGENT = "dsh-tui"


_DSH_TUI_COMMANDS = ("dsh-tui", "dst")


_DSH_LAUNCHER_ARGS = frozenset(
    "--profile --patch --dump-config --dump-default-config -V --version plugin web".split()
)


# Launcher invocations that boot no profile, so they take no --patch overlay.
_DSH_NO_PROFILE_ARGS = frozenset("-V --version plugin".split())


_DSH_USER_RESOURCES_MANIFEST = ".agent-switch-user-resources.json"


# The only user DSH home entries a session sees; profiles, plugins, credentials and
# settings stay isolated.
_DSH_USER_RESOURCE_ENTRIES = ("AGENTS.md", "skills")


# dsh-tui/dst keep their session home by default: settings, preferences, history and the
# pnpm-installed profile live in it, so the shared --persist default does not fit them.
_DSH_PERSIST_OPTION = typer.Option(
    None,
    "--persist/--no-persist",
    rich_help_panel = _PANEL_SESSION,
    help = (
        "The session dir under the agent-switch agents dir keeps settings, preferences, "
        "history and the installed profile, so `--resume <id>` / `-c` can reopen a session. "
        "dsh-tui and dst keep it by default; --no-persist makes it a throwaway dir removed "
        "on exit. dsh uses a throwaway dir unless --persist."
    ),
)


def _dsh_command(args: list[str], patch: Optional[str] = None) -> list[str]:
    head = args[0] if args else ""
    if head in _DSH_LAUNCHER_ARGS or head.startswith(("--profile=", "--patch=")):
        command = ["dsh", *args]
    else:
        command = ["dsh", "web", *args]
    if patch is not None and command[1] not in _DSH_NO_PROFILE_ARGS:
        # `dsh <name>` only expands to `--profile <name>` when the name comes first, so the
        # overlay goes after a bare profile name and ahead of a leading launcher option.
        at = 1 if command[1].startswith("-") else 2
        command[at:at] = ["--patch", patch]
    return command


# Launcher options that take a value, as `--name value` or `--name=value`.
_DSH_VALUE_ARGS = frozenset("--profile --patch --from-default-profile".split())


# Launcher options that print the profile tree and exit instead of booting it.
_DSH_DUMP_ARGS = frozenset("--dump-config --dump-default-config".split())


def _get_dsh_boot_profile(command: list[str]) -> Optional[str]:
    """The profile a `_dsh_command` argv boots, or None when it boots none (plugin, version, dump).

    Like the launcher, read only its own leading options: they end at `--` or at the first
    other argument, and everything from there on belongs to the app. The `web` alias takes
    only --patch and the dumps.
    """
    args = command[1:]
    if args[:1] == ["web"]:
        profile, value_args, exit_args, args = "web", {"--patch"}, _DSH_DUMP_ARGS, args[1:]
    else:
        profile, value_args, exit_args = None, _DSH_VALUE_ARGS, _DSH_DUMP_ARGS | {"-V", "--version"}
    while args:
        name, equals, value = args[0].partition("=")
        if name in exit_args:
            return None
        if name not in value_args:
            break
        if not equals:
            value = args[1] if len(args) > 1 else None
            args = args[1:]
        args = args[1:]
        if name == "--profile":
            profile = value
    return profile


# The dsh-tui launcher reads these only as its first argument, which our leading --patch takes, so
# passed through they would silently become prompt text instead of maintenance commands.
_DSH_TUI_SUBCOMMANDS = frozenset(
    "update migrate doctor safe version --version -v help --help -h".split()
)


# The launcher's own leading options take the next token raw, even a flag-shaped one, and are read
# only until the first app argument.
_DSH_TUI_HOST_VALUE_ARGS = frozenset("--profile --from-default-profile --patch".split())


_DSH_TUI_HOST_SWITCHES = frozenset(
    "--dump-config --dump-default-config --dump-config-schema -V --version".split()
)


_DSH_TUI_RESUME_ARGS = frozenset("--resume -c --continue".split())


_URL_TARGET = re.compile(r"[a-z][a-z0-9+.-]*://", re.IGNORECASE)


def _check_dsh_tui_args(agent: str, args: list[str]) -> None:
    """Refuse arguments that would bypass the agent-switch route, read as the launcher reads them.

    Mirrors the argument loop of dsh-tui's bin/dsh-tui.js: only the first argument can be a
    maintenance command, option values are raw tokens, a first existing path is the workspace
    target, and everything from a literal `--` on is the app's prompt.
    """
    if args[:1] and args[0] in _DSH_TUI_SUBCOMMANDS:
        _fail(
            f"`{args[0]}` is a dsh-tui maintenance command, which {agent} does not run; "
            f"run `dsh-tui {args[0]}` directly. Use `-- {args[0]}` to send it as a prompt."
        )
    has_patch = has_app_args = has_workspace = False
    index = 0
    while index < len(args) and args[index] != "--":
        arg = args[index]
        name, equals, value = arg.partition("=")
        has_next_value = not equals and index + 1 < len(args)
        if not has_app_args and name in _DSH_TUI_HOST_VALUE_ARGS:
            if name != "--patch":
                _fail(
                    f"{name} is not supported for {agent}: the session always boots the dsh-tui "
                    "profile."
                )
            has_patch = True
            index += 2 if has_next_value else 1
            continue
        if name == "--backend":
            value = args[index + 1] if has_next_value else value
            if value.strip() != "dsh":
                _fail(
                    f"--backend {value} is not supported for {agent}: only the DeepSeek Harness "
                    "backend (dsh) talks to the model server."
                )
            index += 2 if has_next_value else 1
            continue
        _reject_as_subagent(agent, [arg])
        if arg == "--resume" and has_next_value and not args[index + 1].startswith("-"):
            index += 2
            continue
        if arg in _DSH_TUI_RESUME_ARGS or name == "--resume":
            pass
        elif not has_app_args and arg in _DSH_TUI_HOST_SWITCHES:
            pass
        elif (
            not has_workspace
            and not arg.startswith("-")
            and (os.path.isabs(arg) or _URL_TARGET.match(arg) or os.path.exists(arg))
        ):
            has_workspace = True
        else:
            has_app_args = True
        index += 1
    if has_patch:
        # Callers own their overlays, so they keep their order after ours; say what that can undo.
        typer.echo(
            "Warning: a --patch you pass is applied after agent-switch's and replaces whole config "
            "blocks; a dsh-tui row there must restate provider, model, backend and the "
            "preset/workspace/sessionId bindings, or it overrides the agent-switch route.",
            err = True,
        )


def _reject_as_subagent(agent: str, args: list) -> None:
    # Reject early, or the flag reaches the agent binary after the server loaded the model.
    if any(arg == "--as-subagent" or arg.startswith("--as-subagent=") for arg in args):
        _fail(f"--as-subagent is not supported for {agent}.")


class _JsExpression(str):
    """A dsh loader `!!js` value: evaluated at boot, so it must not be written as a plain string."""


def write_dsh_patch(
    base: str,
    model: dict,
    path: Path,
    request_body: Optional[dict] = None,
    *,
    headers: Optional[dict] = None,
    max_tokens: Optional[int] = None,
    compact_at: Optional[float] = None,
    is_tui: bool = False,
) -> None:
    """Write the dsh loader patch that points the booted profile at the model server.

    dsh 0.1.7 dropped `settings.yaml`: it now imports a leftover one into the profile only
    after the first boot has settled, so that boot still runs on the DeepSeek default. A
    `--patch` overlay is read at boot on every dsh release this supports, and the file is
    agent-switch's own, so it is rewritten whole rather than merged.
    """
    import yaml

    model_entry = {"id": model["id"]}
    window = model.get("context_length") or model.get("max_context_length")
    if window:
        window = int(window)
        model_entry["contextWindow"] = window
        model_entry["maxTokens"] = _agent_output_limit(window, max_tokens)
    elif max_tokens:
        model_entry["maxTokens"] = max_tokens
    compat = {"supportsDeveloperRole": False, "maxTokensField": "max_tokens"}
    if request_body:
        # dsh sends template kwargs only for a model that declares reasoning levels.
        model_entry["reasoningEfforts"] = {
            "off": None,
            "low": "low",
            "medium": "medium",
            "high": "high",
        }
        compat["thinkingFormat"] = "chat-template"
        compat["chatTemplateKwargs"] = request_body
    entries = [
        {
            "id": "llm-pi-ai",
            "name": "@deepseek-ai/dsh-llm-pi-ai",
            "config": {
                "providers": {
                    _DSH_PROVIDER: {
                        "displayName": "agent-switch",
                        "api": "openai-completions",
                        "baseURL": f"{base}/v1",
                        "apiKeyEnv": _DSH_ENV_KEY,
                        # Values go out verbatim: unlike Pi, dsh resolves no $NAME or !command.
                        **({"headers": headers} if headers else {}),
                        # pi-ai reads an unknown base URL as OpenAI itself.
                        "compat": compat,
                        "models": [model_entry],
                    }
                }
            },
        },
        {
            "id": "agent-default-model",
            "name": "@deepseek-ai/dsh-agent-default-model",
            "config": {"provider": _DSH_PROVIDER, "model": model["id"]},
        },
    ]
    if compact_at is not None and window:
        # dsh compacts once the context passes thresholdRatio of the model's window.
        entries.append(
            {
                "id": "compaction-basic",
                "name": "@deepseek-ai/dsh-compaction-basic",
                "config": {"thresholdRatio": compact_at},
            }
        )
    # TODO: dsh >= 0.1.7 keeps /settings in the profile patch, and this row replaces the whole
    # dsh-tui block, so settings changed there may be overridden on boot; verify on a 0.2.x harness.
    if is_tui:
        # A patch row replaces dsh-tui's whole config block (no deep merge), so restate the six
        # keys of the dsh-tui 0.14.0 row; the !!js bindings carry the launcher's workspace and
        # resume handoff. An explicit provider/model pair outranks ~/.dsh-tui/model.json, and
        # backend: dsh outranks DSH_TUI_BACKEND and the remembered kernel, so the first boot
        # always lands on this route.
        entries.append(
            {
                "id": "dsh-tui",
                "config": {
                    "provider": _DSH_PROVIDER,
                    "model": model["id"],
                    "fullscreen": True,
                    "terminalImages": True,
                    "effort": "max",
                    "preset": _JsExpression("process.env.DSH_TUI_PRESET ?? undefined"),
                    "workspace": _JsExpression("process.env.DSH_TUI_WORKSPACE_TARGET ?? undefined"),
                    "sessionId": _JsExpression("process.env.DSH_TUI_RESUME_SESSION ?? undefined"),
                    "backend": "dsh",
                },
            }
        )

    class _PatchDumper(yaml.SafeDumper):
        pass

    _PatchDumper.add_representer(
        _JsExpression,
        lambda dumper, value: dumper.represent_scalar("tag:yaml.org,2002:js", str(value)),
    )
    text = yaml.dump(entries, Dumper = _PatchDumper, sort_keys = False)
    if not path.exists() or path.read_text(encoding = "utf-8") != text:
        # Private: a --header value may be a token.
        _write_private_text(path, text)
        typer.echo(f"Updated {path}")


# Launcher handoff variables dsh-tui reads with ??, so an inherited one must be removed, not emptied.
_DSH_TUI_ENV_UNSET = (
    "DSH_TUI_WORKSPACE_TARGET",
    "DSH_TUI_RESUME_SESSION",
    "DSH_TUI_RESUME_BACKEND",
    # A one-shot kernel-switch handoff outranks the patch row's backend.
    "DSH_TUI_BACKEND_HANDOFF",
    "DSH_TUI_PRESET",
)


def _get_has_terminal() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _which_dsh_tui() -> Optional[str]:
    """Find the first real dsh-tui (else dst) launcher even when another executable shadows it."""
    augmented = _get_augmented_path()
    # os.get_exec_path() semantics: an unset PATH searches os.defpath.
    for name in _DSH_TUI_COMMANDS:
        for directory in (augmented if augmented is not None else os.defpath).split(os.pathsep):
            executable = _prefer_windows_cmd_sibling(shutil.which(name, path = directory))
            if executable and get_is_deepseek_harness_tui_executable(executable):
                return executable
    return None


def _resolve_dsh_tui(install_hint: str) -> str:
    executable = _which_dsh_tui()
    if executable is None:
        _install_agent(_DSH_TUI_AGENT, install_hint)
        executable = _which_dsh_tui()
    if executable is not None:
        return executable
    shadow = next(filter(None, map(_which_with_install_dirs, _DSH_TUI_COMMANDS)), None)
    if shadow is not None:
        _fail(f"`{shadow}` is not the DeepSeek Harness TUI. Install it with: {install_hint}")
    _fail(f"`{_DSH_TUI_AGENT}` not found on PATH. Install it with: {install_hint}")


def _check_dsh_tui_home(agent: str, home: Path) -> None:
    """Refuse a session home too deep for dsh-tui's per-session socket to stay inside it."""
    if os.name == "nt":
        # Windows gives each session a named pipe, not a path under the home.
        return
    # dsh-tui binds <home>/.dsh-tui/inject/<session id>.sock, and Node cuts an overlong Unix socket
    # path at the size of sun_path (108 bytes on Linux, 104 on macOS and the BSDs) and binds
    # wherever the cut lands, even beside the home. Keep the inject directory plus one name byte
    # within that size, less a terminator byte, so a cut socket stays in the home and goes with it.
    limit = (108 if sys.platform.startswith("linux") else 104) - 1
    inject = os.fsencode(os.path.join(home, ".dsh-tui", "inject", ""))
    if len(inject) + 1 > limit:
        _fail(
            f"{agent} cannot use the session directory {home}: dsh-tui binds a socket under "
            f"{os.fsdecode(inject)}, a {len(inject)}-byte path, and socket paths stop at "
            f"{limit + 1} bytes here, so the socket would land outside the session directory. "
            "Set AGENT_SWITCH_HOME to a shorter path."
        )


def _check_dsh_tui_start(
    agent: str,
    args: list,
    compact_at: Optional[float],
    yolo: bool,
    launch: bool,
    persist: bool,
) -> tuple[str, Optional[str]]:
    """Refuse an unsupported dsh-tui start before any server traffic.

    Returns the launcher to run and, for a launch, the DeepSeek Harness it must start.
    """
    _check_dsh_tui_args(agent, args)
    if compact_at is not None:
        # dsh-tui disables the host compaction row and its agent presets own compaction with no
        # threshold seam, so a ratio here would only look applied.
        _fail(f"--compact-at is not supported for {agent}: its agent presets own compaction.")
    # dsh-tui's own patch pins danger-full-access with approval never on Windows, whatever
    # DSH_PERMISSION_MODE says, because dsh has no Windows sandbox; there is no safe mode to pick.
    if os.name == "nt" and not yolo:
        _fail(
            f"{agent} on Windows always runs tools without a sandbox or approval prompts; "
            "pass --yolo to accept that."
        )
    if launch and not _get_has_terminal():
        _fail(
            f"{agent} is a full-screen terminal app and needs an interactive terminal; "
            "use --no-launch to print the command instead."
        )
    # The home _session_config will make; tempfile names an ephemeral one with 8 characters.
    root = _agents_config_root()
    if launch and not persist:
        _check_dsh_tui_home(
            agent, root / ".tmp" / (_ephemeral_session_prefix(_DSH_TUI_AGENT, None) + "x" * 8)
        )
    else:
        _check_dsh_tui_home(agent, root / _DSH_TUI_AGENT)
    if launch:
        launcher = _resolve_dsh_tui(_npm_install_hint(_DSH_TUI_PACKAGE))
    else:
        # A recipe may run elsewhere, so name the launcher without requiring or probing it.
        launcher = next(
            (name for name in _DSH_TUI_COMMANDS if _which_with_install_dirs(name)), _DSH_TUI_AGENT
        )
    if _wsl_windows_executable([launcher]):
        # WSL can hand a Windows process a cleared variable only as "", and dsh-tui reads its
        # workspace, resume and preset handoff with ??, where "" still counts as set.
        _fail(
            f"`{launcher}` is a Windows dsh-tui, which {agent} does not start from WSL: WSL cannot "
            "clear its handoff variables for it. Install dsh-tui inside WSL."
        )
    if not launch:
        return launcher, None
    harness = _resolve_or_install_agent(
        "dsh", _npm_install_hint(_DSH_PACKAGE), _which_with_install_dirs
    )
    if _wsl_windows_executable([harness]):
        _fail(
            f"`{harness}` is a Windows dsh, which the Linux dsh-tui cannot start under WSL; "
            "install DeepSeek Harness inside WSL."
        )
    return launcher, harness


def _get_dsh_tui_child_path(home: Path, harness: str) -> Optional[str]:
    """A child PATH whose first `dsh` is ``harness``, or None when the inherited PATH finds it."""
    augmented = _get_augmented_path()
    # An unset PATH (None) keeps shutil.which's os.defpath fallback and adds no entry to the child PATH.
    path = augmented if augmented is not None else ""
    first = _prefer_windows_cmd_sibling(shutil.which("dsh", path = augmented))
    harness = os.path.abspath(harness)
    if first is not None and os.path.normcase(os.path.abspath(first)) == os.path.normcase(harness):
        return None
    if os.name == "nt":
        _fail(
            f"`{first}` comes before DeepSeek Harness `{harness}` on PATH, and dsh-tui starts the "
            f"first `dsh`; put `{os.path.dirname(harness)}` earlier on PATH."
        )
    # The launcher starts `dsh` by name: shadow only that name, so pnpm, node and git still
    # resolve exactly as in the caller's PATH.
    shim = home / ".agent-switch-bin"
    shim.mkdir(exist_ok = True)
    link = shim / "dsh"
    link.unlink(missing_ok = True)
    link.symlink_to(harness)
    return os.pathsep.join(filter(None, (str(shim), path)))


def _get_dsh_tui_env(home: Path) -> dict:
    """Env that moves dsh-tui's user state, and the pnpm that installs its profile, into ``home``.

    dsh-tui keeps ~/.dsh-tui under os.homedir() (USERPROFILE on Windows), not DSH_HOME, so only
    moving the home isolates it. pnpm would otherwise follow an inherited PNPM_HOME or store
    setting into the user's own store; pnpm 11 reads pnpm_config_*, older releases npm_config_*.
    """
    pnpm_home = home / ".local" / "share" / "pnpm"
    env = {
        "HOME": str(home),
        "USERPROFILE": str(home),
        # Read with ??, so an inherited root would move session records out of DSH_HOME.
        "DSH_TUI_SESSION_ROOT": str(home / ".dsh" / "sessions"),
        # Pairs with the row's backend: dsh; the launcher's bare --resume and crash retry read it.
        "DSH_TUI_BACKEND": "dsh",
        "PNPM_HOME": str(pnpm_home),
        "COREPACK_HOME": str(home / ".cache" / "node" / "corepack"),
    }
    for prefix in ("pnpm_config_", "npm_config_"):
        env[prefix + "store_dir"] = str(pnpm_home / "store")
        env[prefix + "cache_dir"] = str(home / ".cache" / "pnpm")
    drive, tail = os.path.splitdrive(str(home))
    if drive:
        env["HOMEDRIVE"], env["HOMEPATH"] = drive, tail
    return env


def _get_dsh_tui_unset_env(env: dict) -> tuple:
    """The launcher handoff names, plus other-case spellings of every name set or cleared.

    Windows names are case-insensitive and Node keeps the first spelling in sorted order, so an
    inherited PNPM_CONFIG_STORE_DIR would beat our pnpm_config_store_dir; pnpm 11 on Linux reads
    both spellings and prefers the upper-case one too. A recipe's receiving shell can set those
    known upper-case cache selectors after generation, so clear them even when absent right now.
    """
    names = {name.casefold() for name in (*env, *_DSH_TUI_ENV_UNSET)}
    known = [name.upper() for name in env if name.startswith(("pnpm_config_", "npm_config_"))]
    aliases = [
        name
        for name in os.environ
        if name.casefold() in names and name not in env and name not in _DSH_TUI_ENV_UNSET
    ]
    return (*_DSH_TUI_ENV_UNSET, *dict.fromkeys((*known, *aliases)))


def _seed_dsh_tui_state(home: Path) -> None:
    # A fresh home opens the first-run guide (DeepSeek key, model) and the launchpad, which take
    # the first prompt; the connection is already supplied, so mark both one-shot gates seen.
    state = home / ".dsh-tui"
    for name, content in (
        ("onboarding.json", {"completed": True, "version": 1}),
        ("home.json", {"seen": True}),
    ):
        if not (state / name).exists():
            _write_private_json(state / name, content)


def _get_dsh_source_home(dsh_home: Path) -> Path:
    """The user's own DSH home, resolved the way dsh resolves its own."""
    configured = os.environ.get("DSH_HOME") or ""
    # dsh trims only to decide the variable is set, then uses the raw value, ~-expanded.
    if configured.strip():
        source = Path(os.path.abspath(os.path.expanduser(configured)))
    else:
        source = Path.home() / ".dsh"
    if source.resolve(strict = False) == dsh_home.resolve(strict = False):
        # Do not treat this session as its own resource source.
        return Path.home() / ".dsh"
    return source


def write_dsh_user_resources(dsh_home: Path, *, is_tui: bool) -> None:
    """Link the user's AGENTS.md and skills into an isolated session DSH home."""
    manifest_path = dsh_home / _DSH_USER_RESOURCES_MANIFEST
    previous = _read_json_object(manifest_path)
    managed = previous.get("entries") if isinstance(previous, dict) else []
    if not isinstance(managed, list):
        managed = []
    for name in managed:
        if name in _DSH_USER_RESOURCE_ENTRIES:
            _remove_overlay_entry(dsh_home / name)
    source_home = _get_dsh_source_home(dsh_home)
    if not is_tui and _wsl_windows_executable(["dsh"]):
        # A Windows dsh cannot follow WSL links, and a persisted session an earlier Linux dsh
        # prepared still holds them, so clear those and link nothing.
        manifest_path.unlink(missing_ok = True)
        if any((source_home / name).exists() for name in _DSH_USER_RESOURCE_ENTRIES):
            typer.echo(
                "Warning: your AGENTS.md and skills won't load in this session: a Windows dsh "
                "under WSL cannot follow links made from WSL.",
                err = True,
            )
        return
    dsh_home.mkdir(parents = True, exist_ok = True, mode = 0o700)
    created = []
    for name in _DSH_USER_RESOURCE_ENTRIES:
        source = source_home / name
        target = dsh_home / name
        if not source.exists() or target.is_symlink() or target.exists():
            # Gone from the source, or a real session entry owns the name now.
            continue
        try:
            target.symlink_to(source, target_is_directory = source.is_dir())
        except OSError:
            if source.is_dir():
                if not _create_directory_junction(source, target):
                    typer.echo(
                        f"Warning: couldn't link {source} into the dsh session.", err = True
                    )
                    continue
            else:
                shutil.copy2(source, target)
        created.append(name)
    _write_private_json(manifest_path, {"entries": created})


def dsh(
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
    persist: Optional[bool] = _DSH_PERSIST_OPTION,
):
    """Point DeepSeek Harness (dsh) at a local model server and start it."""
    # One handler for dsh and the TUI: they share the server route and differ in launch and home.
    agent = ctx.info_name
    is_tui = agent != "dsh"
    # dsh-tui/dst keep their session home unless --no-persist; dsh stays throwaway unless --persist.
    persist = is_tui if persist is None else persist
    model, ctx.args[:] = _consume_positional_model(model, ctx.args)
    if is_tui:
        launcher, harness = _check_dsh_tui_start(
            agent, ctx.args, compact_at, yolo, launch, persist
        )
    else:
        _reject_as_subagent("dsh", ctx.args)
    headers = parse_headers(header)
    target = _resolve_target(url, provider, api_key, headers)
    install_hint = _npm_install_hint(_DSH_TUI_PACKAGE if is_tui else _DSH_PACKAGE)
    if not is_tui:
        _require_agent_for_launch("dsh", install_hint, launch)
    # dsh sends reasoning only as chat_template_kwargs, which only these servers read.
    carried = (
        _REASONING_FIELDS
        if providers.get_has_template_kwargs(target.name)
        else frozenset()
    )
    server_options = ServerOptions(
        reasoning = reasoning,
        reasoning_effort = reasoning_effort,
        temperature = temperature,
        top_p = top_p,
        top_k = top_k,
        min_p = min_p,
        repetition_penalty = repetition_penalty,
        presence_penalty = presence_penalty,
        carried = carried,
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
    with _session_config(_DSH_TUI_AGENT if is_tui else "dsh", launch, persist = persist) as home:
        if is_tui:
            # Recheck the real home: an unwritable agents dir falls back to the system temp dir.
            _check_dsh_tui_home(agent, home)
        dsh_home = home / ".dsh" if is_tui else home
        patch = home / _DSH_PATCH_FILE
        write_dsh_patch(
            base,
            entry,
            patch,
            # dsh wraps these in chat_template_kwargs itself, so pass the untranslated fields.
            server_options._replace(provider = None).request_body(),
            headers = headers,
            max_tokens = max_tokens,
            compact_at = compact_at,
            is_tui = is_tui,
        )
        # A Windows dsh under WSL gets DSH_HOME translated through WSLENV, but not argv.
        if is_tui:
            # The launcher forwards only its leading options to dsh, so ours goes first and a
            # caller's own --patch keeps its place after it.
            command = [launcher, "--patch", _agent_config_path(patch, [launcher]), *ctx.args]
        else:
            command = _dsh_command(ctx.args, _agent_config_path(patch, ["dsh"]))
        if compact_at is not None and _get_dsh_boot_profile(command) == "web":
            typer.echo(
                "Warning: --compact-at is ignored for dsh's web profile. "
                "Use --profile headless to apply it.",
                err = True,
            )
        write_dsh_user_resources(dsh_home, is_tui = is_tui)
        env = {
            _DSH_ENV_KEY: key,
            "DSH_HOME": str(dsh_home),
            # dsh uploads session records once a user records /feedback.
            "DSH_TELEMETRY_DISABLED": "1",
            "DSH_PERMISSION_MODE": (
                _DSH_YOLO_PERMISSION_MODE if yolo else _DSH_SAFE_PERMISSION_MODE
            ),
        }
        unset_env = ()
        if is_tui:
            _seed_dsh_tui_state(home)
            env.update(_get_dsh_tui_env(home))
            child_path = _get_dsh_tui_child_path(home, harness) if harness else None
            if child_path:
                env["PATH"] = child_path
            unset_env = _get_dsh_tui_unset_env(env)
            if not launch and os.name != "nt":
                # A POSIX recipe's self-contained last line can only empty a variable inline, and
                # dsh-tui reads its handoff with ??, where "" still counts as set: env -u removes them.
                command = ["env", *(arg for name in unset_env for arg in ("-u", name)), *command]
                unset_env = ()
        _run(
            base,
            entry,
            env,
            command,
            launch = launch,
            install_hint = install_hint,
            unset_env = unset_env,
        )
