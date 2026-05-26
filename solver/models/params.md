# solver/models/params.py

## Purpose
Pydantic models defining the API contract between the Python solver, Rust Tauri shell (`types.rs`), and TypeScript frontend (`fieldTypes.ts`). Single source of truth for request/response shapes.

## Components
- `CoilParams` — coil geometry inputs (radius, turns, pitch, wire radius, current, type)
- `EEDParams` — coupling constants α, β, γ with physics documentation inline
- `SliceRequest` — which 2D cross-section slice to extract post-solve
- `SolveRequest` — full solve input, composes the above
- `SliceData` — one extracted 2D slice as flat float array
- `FieldMaximum` — location and value of a field maximum
- `SolveResult` — full solve output

## Decisions
- `EEDParams` split out of `CoilParams` to make it clear these are physics constants, not geometry
- α=0 default: massless scalar gives maximum predicted extent for first experiments
- `FieldName` literal union enforces valid field names at the boundary

## Contracts
- **Sync requirement**: any field added here must be added to `src-tauri/src/types.rs` AND `src/lib/fieldTypes.ts`
- `data` in `SliceData` is row-major flattened, shape is [rows, cols]
- `position` in `SliceRequest` is normalized 0–1 along the axis, not meters
