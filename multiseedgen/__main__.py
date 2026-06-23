from __future__ import annotations

from .cliconfig import parse_args
from .errors import MultiSeedGenError
from .logging_setup import configure_logging
from .pipeline import run


def main() -> None:
    """Console-script entry point (``seedgen`` / ``python -m multiseedgen``).

    Configures stdout logging and translates the library's typed errors back into the
    old CLI behavior: a clean stderr message + non-zero exit (no traceback).
    """
    configure_logging()
    try:
        run(parse_args())
    except MultiSeedGenError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
