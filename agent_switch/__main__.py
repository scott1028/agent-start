# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the agent-switch authors. See LICENSE.

"""`agent-switch <agent>` entry point."""

import sys

from agent_switch.start import start_app


def main() -> None:
    sys.exit(start_app(prog_name = "agent-switch"))


if __name__ == "__main__":
    main()
