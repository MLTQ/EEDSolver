# src-tauri/src/commands.rs

## Purpose
Tauri commands exposed to the frontend. All IPC between the React UI and the solver goes through here.

## Commands
| Command | Args | Returns | Notes |
|---|---|---|---|
| `solve` | `SolveRequest` | `SolveResult` | Health-checks first; long-running |
| `get_solver_status` | — | `SolverStatus` | Poll on startup |
| `save_hypothesis` | name, request, result, notes? | id string | Writes ~/.oracle/hypotheses/ |
| `load_hypotheses` | — | `Vec<HypothesisEntry>` | Sorted newest first |
| `delete_hypothesis` | id | — | Removes .json file |

## Decisions
- `solve` does a quick health check before dispatching — gives a fast, clear error if solver isn't up instead of a 10s connection timeout
- Hypothesis storage uses `std::fs` + `~/.oracle/hypotheses/` flat JSON files — no database needed at this scale; files are human-readable
- `slug()` produces safe filename components from arbitrary hypothesis names
- All commands return `Result<T, String>` — Tauri serializes errors as `{ error: "..." }` to the frontend

## Contracts
- `SolverClient` must be registered as Tauri managed state before commands are registered
- Hypothesis IDs are `<timestamp>-<name-slug>` — sortable by filename, human-readable
- `load_hypotheses` silently skips unparseable JSON files (warns to log)
