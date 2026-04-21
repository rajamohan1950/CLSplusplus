"""CLI entry point:

    python -m clsplusplus.metering_v2 healthcheck [--json]

Exit code 0 = healthy, 1 = one or more checks failed.
"""

from __future__ import annotations

import sys


def main() -> None:
    argv = sys.argv[1:]
    if not argv:
        print(
            "Usage: python -m clsplusplus.metering_v2 healthcheck [--json]",
            file=sys.stderr,
        )
        sys.exit(2)

    command = argv[0]
    if command == "healthcheck":
        from clsplusplus.metering_v2.healthcheck import main as _run
        _run()
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
