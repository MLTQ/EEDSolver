# api.ts

## Purpose
Central transport layer for frontend-to-solver communication. It chooses Tauri IPC inside the desktop webview and HTTP JSON endpoints for Gruve or direct browser viewers.

## Components

### `solve`, `getSolverStatus`
- **Does**: Run solver requests and poll readiness.
- **Interacts with**: Tauri commands in `commands.rs`, HTTP routes in `gruve.rs`.

### `saveHypothesis`, `loadHypotheses`, `deleteHypothesis`
- **Does**: Persist and manage saved run metadata through the active transport.
- **Interacts with**: `HypothesisLog` and the Rust hypothesis helpers.

### `canUseTauriInvoke`
- **Does**: Detects whether the current page can safely call Tauri IPC.
- **Rationale**: Remote Gruve viewers get a normal browser page where `invoke()` is unavailable.

### `apiRequest`
- **Does**: Shared JSON fetch helper with consistent error text handling.
- **Interacts with**: `apiBase("api")` from `gruve-sdk`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `App.tsx` | `solve()` returns a full `SolveResult` and throws readable errors | Return type or thrown value changes |
| `HypothesisLog` | Hypothesis CRUD matches Tauri command semantics | Route response shape changes |
| `gruve.rs` | HTTP paths remain `/api/solver-status`, `/api/solve`, `/api/hypotheses` | Route shape changes |

## Notes
- The Gruve fallback is same-origin (`""`) because the Rust bridge serves frontend and API on one port.
