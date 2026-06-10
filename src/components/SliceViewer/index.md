# index.tsx

## Purpose
Renders 2-D solver slices as interactive heatmaps. It lets the user switch slice axes and inspect field ranges for the selected result.

## Components

### `SliceViewer`
- **Does**: Selects the best slice for the active field/axis, reshapes flat data into Plotly heatmap rows, and renders plot controls/statistics.
- **Interacts with**: `SolveResult` from `fieldTypes.ts`, `FIELD_COLORMAP` from `colormap.ts`.
- **Rationale**: Uses Plotly's basic bundle through the React factory so Gruve builds do not include unused Mapbox absolute font paths.

### `Placeholder`
- **Does**: Empty-state message for missing result or slice data.

### `linspace`
- **Does**: Builds axis coordinate arrays from slice ranges.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `App.tsx` | Accepts nullable `result` and current `selectedField` | Prop shape changes |
| Gruve build | Bundle avoids absolute public asset paths | Reintroducing full Plotly bundle |

## Notes
- Heatmap is the only Plotly trace currently used; add a larger Plotly bundle only if a new trace requires it.
