# App.tsx

## Purpose
Top-level Oracle UI shell. It owns solver request/result state, field selection, auto-solving, hypothesis saving, and now Gruve session synchronization for shared viewers.

## Components

### `App`
- **Does**: Coordinates solver status polling, parameter changes, solve execution, shared session publish/subscribe, and the main layout.
- **Interacts with**: `GeometryPanel`, `VolumeViewer`, `SliceViewer`, `LegendPanel`, `HypothesisLog`, `api.ts`, and `multiplayer.ts`.

### Gruve session effects
- **Does**: Apply remote request/field/result updates and publish local changes with echo suppression.
- **Rationale**: Only the local initiator auto-solves after edits; remote viewers apply retained state without stampeding the solver.

### `StatusDot`
- **Does**: Compact readiness indicator for the solver.

### `SaveModal`
- **Does**: Modal form for naming and annotating a solved run.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `GeometryPanel` | `request` and `onChange` stay canonical for parameter editing | Replacing setter semantics |
| `api.ts` | Solve requests include selected field in slices and volume before solving | Removing field normalization |
| Gruve session | Publishes `oracle.request`, `oracle.selectedField`, and `oracle.result` | Renaming session keys without migration |

## Notes
- Remote-applied changes suppress auto-solve briefly so one user action does not fan out into many duplicate solves.
