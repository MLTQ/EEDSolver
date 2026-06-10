# src-tauri/src/lib.rs

## Purpose
Tauri application entry point. Wires together managed solver state, Tauri commands, and the Gruve HTTP bridge.

## Components
- `run()` — configures and starts the Tauri app
- `Arc<OracleSolver>` as managed state — shared across Tauri commands and the Gruve HTTP server
- `gruve::start()` — serves the built frontend/API on localhost and announces to Gruve

## Decisions
- Solver initialization remains fail-fast in `setup` so the UI never starts without a usable backend.
- The same solver instance is used for IPC and HTTP to keep desktop and Gruve behavior consistent.

## Contracts
- `Arc<OracleSolver>` must be managed before `invoke_handler` is called.
- `gruve::start()` must run after solver init so announced apps have a live API upstream.
