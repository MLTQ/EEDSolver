# gruve.rs

## Purpose
Provides the HTTP surface Gruve needs for multiplayer access to the Tauri app. It serves the built frontend, mirrors Tauri commands under `/api/*`, and heartbeats an Oracle lobby tile to the local Gruve agent.

## Components

### `start`
- **Does**: Binds an ephemeral localhost port, starts the Axum server, and starts the Gruve announce loop.
- **Interacts with**: `OracleSolver` from `solver-gpu`, `commands.rs` helper functions, `lib.rs` setup.
- **Rationale**: One port serves UI and API so Gruve can proxy the whole app by name.

### HTTP handlers
- **Does**: Expose solver status, solve, hypothesis save/load/delete as JSON endpoints.
- **Interacts with**: `SolveRequest`, `SolveResult`, and `HypothesisEntry` from `types.rs`.

### `get_static_asset_http` / `static_relative_path`
- **Does**: Serve `dist/` with an `index.html` fallback for client routes and strip Gruve app prefixes before resolving assets.
- **Interacts with**: Vite build output and Tauri resource directories.
- **Rationale**: Gruve serves apps under `/apps/<id>/` or peer paths containing `/apps/<id>/`; the backend must map those URLs back to files under `dist/`.

### `resolve_dist_dir`
- **Does**: Finds the built frontend in source-tree `dist/` or Tauri resource `_up_/dist` locations.
- **Rationale**: Raw Tauri binaries and packaged `.app` bundles do not run from the repository root, so the bridge must not rely on current working directory alone.

### `announce_loop`
- **Does**: Re-posts app metadata to `http://127.0.0.1:8088/gruve/announce` every TTL/3 seconds.
- **Interacts with**: Gruve agent; degrades quietly when the agent is not running.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Gruve agent | `id`, `port`, and `upstreams.api` point to a listening localhost server | Renaming `APP_ID`, removing heartbeat, changing port ownership |
| `src/lib/api.ts` | JSON endpoints live at `/api/*` behind the declared `api` upstream | Route shape or response JSON changes |
| Remote browsers | `dist/index.html` is found in source-tree and Tauri resource layouts; relative assets are served from any sub-path | Absolute asset paths or missing built assets |

## Notes
- Tauri copies the configured `../dist` resource as `_up_/dist`; raw binaries rely on executable-relative lookup for that shape.
