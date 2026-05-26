# src-tauri/src/solver_client.rs

## Purpose
reqwest HTTP client wrapping the Python solver server. Handles health polling, long-timeout solves, and typed error reporting.

## Components
- `SolverClient` — thin wrapper over `reqwest::Client` with base URL
- `health_check()` — single GET /health → bool
- `wait_until_ready()` — polls until ready or 30s timeout; returns SolverStatus
- `solve(request)` — POST /solve with 600s timeout; returns SolveResult or SolverError
- `SolverError` — typed errors with actionable messages (e.g. "not reachable" includes start command)
- `CoilType/FormulationType/MeshResolution::label()` — display labels for logging

## Decisions
- Separate `reqwest::Client` for solve requests with 600s timeout — avoids polluting the general client used for health checks
- `SolverError::NotReachable` message includes the exact start command — faster feedback loop in dev
- `impl From<SolverError> for String` — Tauri commands return `Result<T, String>`, so errors convert automatically

## Contracts
- `wait_until_ready()` never panics — always returns a SolverStatus
- `solve()` returns `SolverError::Timeout` after 600s — fine meshes should complete within that window
- Caller is responsible for spawning the solver process; this client only communicates with it
