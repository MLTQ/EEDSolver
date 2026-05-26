# solver/fields/postproc.py

## Purpose
Converts dolfinx solve output → JSON-serializable SliceData and FieldMaximum lists. The boundary between FEniCSx land and the API transport layer.

## Components
- `extract_results(solve_output, slice_requests)` — top-level, returns full `SolveResult`
- `extract_slice(solve_output, request)` — samples one 2D plane as a regular grid
- `extract_maxima(solve_output)` — finds global max of each field on a coarse volume grid
- `_get_field_function` — resolves field name to dolfinx Function (handles derived fields like |A|, |B|)
- `_make_magnitude_function` — computes |v| = sqrt(v·v) as a CG1 scalar
- `_sample_field_at_points` — BoundingBoxTree point evaluation with NaN handling

## Decisions
- BoundingBoxTree point evaluation: faster than FEniCSx Expression grid sampling for irregular grids; handles out-of-domain points gracefully
- Maxima use a 32³ coarse grid — sufficient to locate spatial region of maximum, fast enough for every solve
- Slice errors return zero-filled SliceData rather than crashing the response — UI can show a warning
- |A|, |B|, |J| interpolated to CG1 scalar space for consistent sampling

## Contracts
- `field_name` must be one of: "phi", "A_magnitude", "B_magnitude", "J_magnitude"
- `phi` is None for maxwell_only — callers must handle
- `A` (and derived B) is None for scalar_only — callers must handle
- SliceData.data is row-major (u varies in rows, v varies in columns), shape [n, n]
