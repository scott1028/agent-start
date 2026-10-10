# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Target resolution, env printing and the launch of the agent child process."""

import contextlib
import os
import re
import shlex
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Optional

import click
import typer

from agent_switch import providers
from agent_switch.core.install import (
    _augment_path_with_install_dirs,
    _prefer_windows_cmd_sibling,
    _resolve_or_install_agent,
)
from agent_switch.core.options import LoadOptions, ServerOptions, _SAMPLING_FIELDS, _fail
from agent_switch.core.platform import (
    _NPM_NATIVE_CMD_SHIM,
    _NPM_NODE_CMD_SHIMS,
    _apply_windows_environment,
    _merge_wslenv,
    _npm_node_shim_metadata,
    _powershell_quote,
    _wsl_bridge_names,
    _wsl_windows_executable,
)
from agent_switch.core.storage import _cached_keys, _provider_key_cache_path, _remember_key
from agent_switch.providers.types import ProviderError, Target


def _print_env(
    env: dict,
    command: list,
    unset_env: tuple = (),
    wsl_env_bridge: tuple = (),
) -> None:
    if os.name == "nt":
        for name in unset_env:
            typer.echo(f"Remove-Item Env:{name} -ErrorAction SilentlyContinue")
        for name, value in env.items():
            # PowerShell: ` is the escape char, and $ triggers expansion inside "".
            escaped = value.replace("`", "``").replace('"', '`"').replace("$", "`$")
            typer.echo(f'$env:{name} = "{escaped}"')
        typer.echo(" ".join(_powershell_quote(arg) for arg in command))
        return
    for name in unset_env:
        typer.echo(f"export {name}=" if wsl_env_bridge else f"unset {name}")
    for name, value in env.items():
        typer.echo(f"export {name}={shlex.quote(value)}")
    if wsl_env_bridge:
        typer.echo(
            f"export WSLENV={shlex.quote(_merge_wslenv(os.environ.get('WSLENV', ''), wsl_env_bridge))}"
        )
    # The final line is a SELF-CONTAINED one-liner (inline env, VAR=... cmd) rather than a bare command. People copy just the last line, and a bare `codex`/`claude` would then run against their real ~/.codex or Anthropic credentials with zero isolation, for example inheriting a pre-existing damaged ~/.codex state DB and blaming the recipe. Inline assignments scope every var, and empty-string the conflicting ones, to this single invocation, so a partial copy behaves the same as pasting the whole block.
    inline = [f"{name}=" for name in unset_env]
    inline += [f"{name}={shlex.quote(value)}" for name, value in env.items()]
    if wsl_env_bridge:
        inline.append(
            f"WSLENV={shlex.quote(_merge_wslenv(os.environ.get('WSLENV', ''), wsl_env_bridge))}"
        )
    typer.echo(" ".join((*inline, shlex.join(command))))


def _wsl_shim_env(command: list, env: dict, unset_env: tuple) -> tuple[dict, tuple]:
    if not _wsl_windows_executable(command):
        return env, ()
    wsl_env_bridge = _wsl_bridge_names(env, unset_env)
    if not wsl_env_bridge:
        return env, ()
    # Bridge PWD via WSLENV (PWD/p) so the Windows shim finds its project root from the live cwd, not a stale inherited Linux PWD. Do not freeze env["PWD"]: a --no-launch recipe must translate the live PWD when run, not when generated; _launch overrides it.
    return env, tuple(dict.fromkeys((*wsl_env_bridge, "PWD/p")))


def _resolved_launch_command(
    executable: str,
    arguments: list,
    environment: Optional[dict] = None,
) -> list:
    """Return an argv that preserves arguments through standard Windows npm shims."""
    # _launch resolves with raw shutil.which, so rescue here too; the sibling then enters the parser below.
    executable = _prefer_windows_cmd_sibling(executable)
    if os.name == "nt" and Path(executable).suffix.lower() in {".cmd", ".bat"}:
        # cmd.exe treats CR/LF inside `%*` as command separators, and Windows PowerShell's native-command bridge also rewrites embedded quotes. Match complete cmd-shim templates so custom wrappers keep their setup behavior.
        with contextlib.suppress(OSError, UnicodeError, IndexError):
            shim = Path(executable)
            contents = shim.read_text(encoding = "utf-8").replace("\r\n", "\n").strip()
            for pattern in _NPM_NODE_CMD_SHIMS:
                match = pattern.fullmatch(contents)
                if match is None:
                    continue
                relative = Path(*re.split(r"[\\/]+", match.group("target")))
                target = (shim.parent / relative).resolve()
                if not target.is_file() or not any(
                    part.casefold() == "node_modules" for part in target.parts
                ):
                    continue
                metadata = _npm_node_shim_metadata(target, match, environment or os.environ)
                if metadata is None:
                    continue
                node_args, environment_updates = metadata
                bundled_node = shim.parent / "node.exe"
                node = str(bundled_node) if bundled_node.is_file() else shutil.which("node.exe")
                if node:
                    if environment is not None:
                        _apply_windows_environment(environment, environment_updates)
                    return [node, *node_args, str(target), *arguments]

            match = _NPM_NATIVE_CMD_SHIM.fullmatch(contents)
            if match is not None:
                relative = Path(*re.split(r"[\\/]+", match.group("target")))
                target = (shim.parent / relative).resolve()
                if (
                    target.is_file()
                    and any(part.casefold() == "node_modules" for part in target.parts)
                    and target.suffix.lower() in {".exe", ".com"}
                ):
                    return [str(target), *arguments]
    return [executable, *arguments]


def _launch(
    command: list,
    env: dict,
    install_hint: str,
    unset_env: tuple = (),
) -> int:
    # Resolve well-known install dirs (~/.local/bin) first, so an already-installed agent not yet on PATH is found instead of prompting a needless reinstall.
    _augment_path_with_install_dirs()
    executable = _resolve_or_install_agent(command[0], install_hint, shutil.which)
    env, wsl_env_bridge = _wsl_shim_env(command, env, unset_env)
    child_env = dict(os.environ)
    if wsl_env_bridge:
        # Override stale inherited PWD with the real cwd so the shim resolves the project root.
        env = {**env, "PWD": os.getcwd()}
        child_env["WSLENV"] = _merge_wslenv(child_env.get("WSLENV", ""), wsl_env_bridge)
        for name in unset_env:
            child_env[name] = ""
    else:
        for name in unset_env:
            child_env.pop(name, None)
    child_env.update(env)
    if os.name != "nt" and not wsl_env_bridge:
        # Keep POSIX child processes from seeing a stale inherited PWD when subprocess cwd was changed by the caller. Some Node CLIs use PWD for project-root discovery instead of process.cwd().
        child_env["PWD"] = os.getcwd()
    # Ctrl+C cancels a turn inside the agent; do not let it kill this wrapper. A no-op handler, not SIG_IGN: exec preserves an ignored signal but resets a caught one.
    previous = signal.signal(signal.SIGINT, lambda *_: None)
    try:
        launch_command = _resolved_launch_command(executable, command[1:], child_env)
        code = subprocess.run(launch_command, env = child_env).returncode
    finally:
        signal.signal(signal.SIGINT, previous)
    # Negative returncode means killed by signal N; shells expect 128+N.
    return code if code >= 0 else 128 - code


# The server this invocation talks to, for the status lines _run prints. Set by _resolve_target and _connect.
_active_target: Optional[Target] = None


# Set by the group in start.py when the invoked name is an agent alias like `claude-personal`:
# the user's own shell name, and the agent kind it was matched to. Click shares ctx.meta with
# child contexts, so the command function reads both.
_ALIAS_META = "agent_switch.alias"


_ALIAS_KIND_META = "agent_switch.alias_kind"


# The alias name goes into a `bash -ic` string, so it must stay shell-word safe.
_ALIAS_NAME = re.compile(r"[A-Za-z0-9._-]+")


def _agent_command(alias: Optional[str], agent: str, args: list, unset_env: tuple = ()) -> list:
    """The agent's argv; an alias replaces only argv[0], and the env still reaches the bash process."""
    if alias is None:
        return [agent, *args]
    script = f'{alias} "$@"'
    if unset_env:
        # bash -ic re-reads the shell config, which could re-export what the session cleared.
        script = "unset " + " ".join(unset_env) + "; " + script
    # `agent-switch` is $0, so "$@" is exactly the agent's arguments. No exec: bash cannot exec a function.
    return ["bash", "-ic", script, "agent-switch", *args]


def _check_alias(alias: str) -> None:
    """Fail unless bash can name and start the user's own function, alias or script."""
    if os.name == "nt":
        _fail(
            f"`{alias}` is an agent alias, which agent-switch starts through bash; "
            "Windows is not supported."
        )
    if shutil.which("bash") is None:
        _fail(f"`{alias}` is an agent alias, which needs `bash` on PATH.")
    if not _ALIAS_NAME.fullmatch(alias):
        _fail(f"`{alias}` carries characters an agent alias name cannot have.")
    probed = subprocess.run(
        ["bash", "-ic", f"type -t {alias}"],
        capture_output = True,
        text = True,
    )
    kind = probed.stdout.strip()
    if kind not in ("function", "alias", "file"):
        _fail(
            f"`{alias}` is not a bash function, alias or file in `bash -ic`'s shell; define it "
            'in your shell config forwarding "$@" to the agent, or use the plain subcommand.'
        )
    if kind == "file":
        path = subprocess.run(
            ["bash", "-ic", f"command -v {alias}"],
            capture_output = True,
            text = True,
        ).stdout.strip()
        try:
            definition = Path(path).read_text(encoding = "utf-8", errors = "replace")
        except OSError:
            definition = ""
    else:
        printer = "declare -f" if kind == "function" else "alias"
        definition = subprocess.run(
            ["bash", "-ic", f"{printer} {alias}"],
            capture_output = True,
            text = True,
        ).stdout
    if "agent-switch" in definition:
        _fail(
            f"`{alias}` runs agent-switch itself, so `agent-switch {alias}` would recurse; "
            "run what its definition runs instead."
        )


_REQUEST_FLAGS = {
    **{name: "--" + name.replace("_", "-") for name in _SAMPLING_FIELDS},
    "enable_thinking": "--reasoning",
    "reasoning": "--reasoning",
    "reasoning_effort": "--reasoning-effort",
}


def _resolve_target(
    url: Optional[str],
    provider: Optional[str],
    api_key: Optional[str] = None,
    headers: Optional[dict] = None,
) -> Optional[Target]:
    """The server named by --url/--provider; None when neither was given, which means native launch."""
    global _active_target
    if url and not api_key:
        api_key = next(iter(_cached_keys(_provider_key_cache_path(), providers.root_url(url))), None)
    try:
        target = providers.resolve_target(url, provider, api_key, headers)
    except ProviderError as exc:
        _fail(str(exc))
    _active_target = target
    return target


def _warn_unsent_pins(server_options: ServerOptions) -> None:
    """Sampling/reasoning pins this agent cannot carry in its own requests are ignored."""
    sent = server_options.sent_by_agent()
    unsent = [
        _REQUEST_FLAGS[name]
        for name in (*_SAMPLING_FIELDS, "reasoning", "reasoning_effort")
        if getattr(server_options, name) is not None
        and name not in sent
        and not (name == "reasoning" and server_options.reasoning == "auto")
    ]
    if unsent:
        typer.echo(f"Warning: this agent can't send {', '.join(unsent)} itself, so it is ignored.", err = True)


def _connect(
    api_key: Optional[str],
    model: Optional[str],
    load: LoadOptions = LoadOptions(),
    *,
    server_options: ServerOptions = ServerOptions(),
    target: Target,
    needs: tuple = (),
) -> tuple:
    global _active_target
    _active_target = target
    label = providers.label(target.name)
    _warn_unsent_pins(server_options)
    _, dropped = providers.request_body(target.name, server_options._replace(provider = None).request_body())
    for name in dropped:
        typer.echo(f"Warning: {label} ignores {_REQUEST_FLAGS[name]}, so it is left out.", err = True)
    cache = _provider_key_cache_path()
    key = api_key or next(iter(_cached_keys(cache, target.base)), None)
    try:
        base, key, entry = providers.connect(
            target, key, model, load.max_seq_length or None, needs, allow_load = load.allow_load
        )
    except ProviderError as exc:
        _fail(str(exc))
    if api_key:
        _remember_key(cache, base, api_key)
    return base, key, entry


def _run(
    base: str,
    entry: dict,
    env: dict,
    command: list,
    *,
    launch: bool,
    install_hint: str,
    unset_env: tuple = (),
    clear_screen: bool = False,
) -> None:
    # Some agents (Pi) render inline from wherever the cursor sits: their first paint assumes a clean screen rather than clearing or entering the alternate screen themselves. Hand them one so the session does not start mid-scroll under our connection output. click.clear() is cross-platform and a no-op when stdout is not a terminal, so transcripts and --no-launch recipes stay intact.
    if launch and clear_screen:
        click.clear()
    typer.echo(f"{providers.label(_active_target.name)} ready at {base} · model {entry['id']}")
    if not launch:
        env, wsl_env_bridge = _wsl_shim_env(command, env, unset_env)
        _print_env(
            env,
            command,
            unset_env = unset_env,
            wsl_env_bridge = wsl_env_bridge,
        )
        return
    code = _launch(
        command,
        env,
        install_hint = install_hint,
        unset_env = unset_env,
    )
    if code:
        # A failed session must not end silently behind the agent's own output.
        typer.echo(f"The agent exited with code {code}.")
    raise typer.Exit(code = code)


def _run_native(
    agent: str,
    env: dict,
    command: list,
    *,
    launch: bool,
    install_hint: str,
    unset_env: tuple = (),
    clear_screen: bool = False,
) -> None:
    """Native launch: the agent keeps its own model, login and config; only shared flags were added."""
    if launch and clear_screen:
        click.clear()
    typer.echo(f"{agent} runs with its own model, login and config.")
    if not launch:
        env, wsl_env_bridge = _wsl_shim_env(command, env, unset_env)
        _print_env(
            env,
            command,
            unset_env = unset_env,
            wsl_env_bridge = wsl_env_bridge,
        )
        return
    code = _launch(
        command,
        env,
        install_hint = install_hint,
        unset_env = unset_env,
    )
    if code:
        typer.echo(f"The agent exited with code {code}.")
    raise typer.Exit(code = code)
