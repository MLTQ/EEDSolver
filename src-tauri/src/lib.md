# src-tauri/src/lib.rs

## Purpose
Tauri application entry point. Wires together: managed state, plugins, commands, and the startup health-poll background task.

## Components
- `run()` — configures and starts the Tauri app
- `tauri_plugin_shell::init()` — enables sidecar process management (future: auto-launch solver)
- `SolverClient::new()` as managed state — shared across all async commands
- Background `wait_until_ready()` task — pre-warms the solver connection before the first user click

## Decisions
- Background health poll on `setup` — means the frontend's first `get_solver_status` call likely returns `Ready` immediately instead of making the user wait for a poll
- `tauri_plugin_shell` included now even though sidecar auto-launch isn't implemented yet — avoids a later Cargo.toml + capabilities change

## Contracts
- `SolverClient` must be managed before `invoke_handler` is called (Tauri enforces this at runtime)
- The background health task is fire-and-forget — it logs but doesn't surface errors to the UI (the frontend polls `get_solver_status` for that)
