from __future__ import annotations
from .cliconfig import parse_args
from .pipeline import run

if __name__ == "__main__":
    run(parse_args())
