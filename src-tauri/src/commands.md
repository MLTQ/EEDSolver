# src-tauri/src/commands.rs

## Purpose
Tauri commands exposed to the desktop frontend. Shared helper functions also back the Gruve HTTP bridge so IPC and HTTP keep identical behavior.

## Commands
| Command | Args | Returns | Notes |
|---|---|---|---|
| `solve` | `SolveRequest` | `SolveResult` | Health-checks first; long-running |
| `get_solver_status` | — | `SolverStatus` | Poll on startup |
| `save_hypothesis` | name, request, result, notes? | id string | Writes ~/.oracle/hypotheses/ |
| `load_hypotheses` | — | `Vec<HypothesisEntry>` | Sorted newest first |
| `delete_hypothesis` | id | — | Removes .json file |

## Components
- `solve_request` — shared solve helper used by Tauri IPC and HTTP
- `solver_status` — shared readiness payload builder
- `save_hypothesis_entry`, `load_hypothesis_entries`, `delete_hypothesis_entry` — shared filesystem persistence helpers

## Decisions
- Hypothesis storage uses `std::fs` + `~/.oracle/hypotheses/` flat JSON files — no database needed at this scale; files are human-readable
- `slug()` produces safe filename components from arbitrary hypothesis names
- All commands return `Result<T, String>` — Tauri serializes errors as `{ error: "..." }` to the frontend

## Contracts
- `Arc<OracleSolver>` must be registered as Tauri managed state before commands are registered
- Hypothesis IDs are `<timestamp>-<name-slug>` — sortable by filename, human-readable
- `load_hypotheses` silently skips unparseable JSON files (warns to log)
- `gruve.rs` depends on helper semantics matching these commands
