# multiplayer.ts

## Purpose
Small Gruve session adapter for Oracle's shared app state. It centralizes session keys and validates remote values before `App.tsx` applies them.

## Components

### `ORACLE_SESSION_KEYS`
- **Does**: Defines stable keys for request, selected field, and result state.
- **Interacts with**: `App.tsx` publish/subscribe logic and Gruve retained session state.

### `joinOracleSession`
- **Does**: Joins the Gruve app session and forwards peer-count updates.
- **Interacts with**: `joinSession` from `gruve-sdk`.

### `isGruveSharedSession`
- **Does**: Distinguishes shared Gruve openings from standalone or `gruve-solo` openings.

### `isFieldName`, `isSolveRequest`, `isSolveResult`
- **Does**: Lightweight runtime guards for values received from the session.
- **Rationale**: Session data is remote input and should not be trusted blindly.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `App.tsx` | Guards only accept state safe enough to feed existing setters | Weakening guards can crash the UI on malformed session state |
| Gruve session | Keys remain stable for retained state replay | Renaming keys loses late-join state |

## Notes
- Validation is intentionally structural rather than exhaustive; final physics validation still belongs to the solver.
