# solver/server.py

## Purpose
FastAPI app instance with route definitions. Entry point for all HTTP traffic from the Tauri shell.

## Routes
- `GET /health` → `HealthResponse` — polled by Tauri on startup
- `GET /fields` → `list[str]` — field names available
- `POST /solve` → `SolveResult` — full pipeline

## Decisions
- CORS allows tauri://localhost and the Vite dev server ports — required for Tauri webview
- Imports of dolfinx/solver deferred into route handlers — allows server to start even if the venv is incomplete, returning a 503 with a clear message rather than a crash

## Contracts
- `/solve` can take minutes for fine meshes — Tauri client must use a long timeout
- 503 = dependency missing (dolfinx not installed — run `uv sync`); 500 = solver runtime error
