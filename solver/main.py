"""
Oracle solver server entry point.
Starts uvicorn on localhost:7432.

Usage (development):
    docker compose up solver          # recommended — has dolfinx pre-installed
    docker compose up solver --build  # after Dockerfile changes

Usage (as Tauri sidecar):
    Spawned by Tauri. SIGTERM causes graceful shutdown.
"""

from __future__ import annotations

import logging
import signal
import sys

import uvicorn

from solver.server import app  # noqa: F401 — re-exported for uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

log = logging.getLogger(__name__)

PORT = 7432
HOST = "127.0.0.1"


def _handle_sigterm(signum, frame):
    log.info("SIGTERM received — shutting down Oracle solver")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _handle_sigterm)
    log.info(f"Starting Oracle solver on {HOST}:{PORT}")
    uvicorn.run(
        "solver.main:app",
        host=HOST,
        port=PORT,
        log_level="info",
        reload=False,  # set to True for dev via CLI flag
    )
