# src-tauri/src/types.rs

## Purpose
Rust mirror of `solver/models/params.py`. Defines all request/response types with serde for JSON transport over Tauri IPC.

## Components
- Enums: `CoilType`, `FormulationType`, `MeshResolution`, `SliceAxis`, `FieldName`
- Request structs: `CoilParams`, `EedParams`, `SliceRequest`, `SolveRequest`
- Result structs: `SliceData`, `FieldMaximum`, `SolveResult`
- App state: `SolverStatus`, `SolverState`
- `HypothesisEntry` — local experiment log entry with timestamp + notes

## Decisions
- `serde(rename_all = "snake_case")` on enums: Python uses snake_case string values ("toroid_poloidal"), Rust uses PascalCase variants — serde bridges them
- `current_a` field (not `current_A`) — Rust field names must be valid identifiers; JSON key is still `current_a` which Python receives fine
- `chrono::DateTime<Utc>` for timestamps — serializes to ISO 8601 for JS compatibility

## Contracts
- **Sync requirement**: any field change must be reflected in `solver/models/params.py` AND `src/lib/fieldTypes.ts`
- `FieldName::AMagnitude` → JSON `"a_magnitude"` → Python `"A_magnitude"` — check this on first integration test
- `EedParams` TODO: β/γ coupling term structure pending DDOF paper citation (see formulation.py)
