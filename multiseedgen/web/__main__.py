"""Launch the MultiSeedGen web UI:  python -m multiseedgen.web [--host H --port P]"""

from __future__ import annotations

import argparse


def main() -> None:
    import os

    ap = argparse.ArgumentParser(description="MultiSeedGen web UI")
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    ap.add_argument("--reload", action="store_true")
    a = ap.parse_args()
    import uvicorn

    uvicorn.run("multiseedgen.web.app:app", host=a.host, port=a.port, reload=a.reload)


if __name__ == "__main__":
    main()
