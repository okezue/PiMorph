"""Command-line entry point for PiMorph.

Subcommands are registered lazily so that importing the CLI does not pull in
torch or cellpose.
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Dict, List, Optional


def _cmd_version(_: argparse.Namespace) -> int:
    from pimorph import __version__

    print(__version__)
    return 0


def _registry() -> Dict[str, Callable[[argparse.ArgumentParser], Callable[[argparse.Namespace], int]]]:
    """Map subcommand name -> function that configures a subparser and returns the handler."""
    reg: Dict[str, Callable[[argparse.ArgumentParser], Callable[[argparse.Namespace], int]]] = {}

    def version(p: argparse.ArgumentParser) -> Callable[[argparse.Namespace], int]:
        return _cmd_version

    reg["version"] = version
    return reg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pimorph", description="PiMorph endothelial cell-complex inference")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, configure in _registry().items():
        sp = sub.add_parser(name)
        sp.set_defaults(_handler=configure(sp))
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args._handler(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
