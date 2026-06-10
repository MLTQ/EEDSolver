# vite.config.ts

## Purpose
Vite configuration for the React frontend. It keeps the dev server deterministic for Tauri and builds assets that can be served from Gruve sub-paths.

## Components

### `defineConfig`
- **Does**: Registers React, sets a relative asset base, and fixes the dev server on port 5173.
- **Interacts with**: Tauri `devUrl`, Gruve static serving, and `gruve doctor`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `src-tauri/tauri.conf.json` | Dev server listens on `http://localhost:5173` | Port changes |
| Gruve remote viewers | Built asset URLs are relative via `base: "./"` | Absolute base paths |

## Notes
- Relative base is required because Gruve opens apps under `/apps/<id>/` or `/peer/.../apps/<id>/`.
