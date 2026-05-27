"""
FastAPI route definitions for the Oracle solver server.
Listens on localhost:7432.

Routes:
  GET  /health    — liveness check
  GET  /fields    — list available field names
  POST /solve     — full solve + slice extraction
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from solver.models.params import (
    FieldName,
    HealthResponse,
    SolveRequest,
    SolveResult,
)

log = logging.getLogger(__name__)

app = FastAPI(
    title="Oracle Solver",
    description="EED/DDOF field simulation server for the Oracle desktop app",
    version="0.1.0",
)

# Allow Tauri webview origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["tauri://localhost", "http://localhost:1420", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness check. Tauri polls this on startup before enabling the Solve button."""
    return HealthResponse()


@app.get("/fields", response_model=list[str])
async def list_fields() -> list[str]:
    """Return the list of field names the solver can produce."""
    return list(FieldName.__args__)  # type: ignore[attr-defined]


@app.post("/solve", response_model=SolveResult)
async def solve(request: SolveRequest) -> SolveResult:
    """
    Full solve pipeline. Accepts SolveRequest, returns SolveResult with
    slice data and field maxima.

    Can take seconds (coarse) to minutes (fine). Use a long HTTP timeout on the client.
    """
    log.info(
        f"Solve request: {request.formulation} | "
        f"{request.coil.coil_type} r={request.coil.radius_m}m n={request.coil.turns} | "
        f"mesh={request.mesh_resolution}"
    )

    try:
        from solver.fields.solver import run_solve
        from solver.fields.postproc import extract_results

        solve_output = run_solve(request)
        result = extract_results(
            solve_output,
            request.slices,
            request.request_volume,
            volume_field=request.volume_field,
        )
        log.info(f"Solve complete: {result.solve_time_s:.2f}s, {result.mesh_nodes} nodes")
        return result

    except ImportError as exc:
        log.error(f"Import error (missing dependency?): {exc}")
        raise HTTPException(
            status_code=503,
            detail=f"Solver dependency not available: {exc}. Run: docker compose up solver (see SETUP.md)",
        )
    except Exception as exc:
        log.error(f"Solve failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Solve failed: {exc}",
        )
