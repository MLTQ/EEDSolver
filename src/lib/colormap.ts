// Colorscale definitions for Plotly heatmaps and UI labels.

import type { FieldName } from "./fieldTypes";

// Plotly colorscale names
export type ColormapName = "Viridis" | "Plasma" | "RdBu" | "Hot" | "Cividis";

export const FIELD_COLORMAP: Record<FieldName, ColormapName> = {
  phi:         "Viridis",   // EED scalar — viridis standard
  A_magnitude: "Plasma",    // vector potential magnitude
  B_magnitude: "Hot",       // magnetic flux density — hot for intensity
  J_magnitude: "Cividis",   // current density
};

// Plotly colorscale entry: [position 0–1, css color]
export type PlotlyColorscale = [number, string][];

// Viridis (12-stop approximation)
export const VIRIDIS: PlotlyColorscale = [
  [0.0,  "#440154"], [0.1,  "#482878"], [0.2,  "#3e4989"],
  [0.3,  "#31688e"], [0.4,  "#26828e"], [0.5,  "#1f9e89"],
  [0.6,  "#35b779"], [0.7,  "#6ece58"], [0.8,  "#b5de2b"],
  [0.9,  "#dde318"], [1.0,  "#fde725"],
];

// Plasma (12-stop)
export const PLASMA: PlotlyColorscale = [
  [0.0,  "#0d0887"], [0.1,  "#41049d"], [0.2,  "#6a00a8"],
  [0.3,  "#8f0da4"], [0.4,  "#b12a90"], [0.5,  "#cc4778"],
  [0.6,  "#e16462"], [0.7,  "#f2844b"], [0.8,  "#fca636"],
  [0.9,  "#fcce25"], [1.0,  "#f0f921"],
];

export function getColorscale(name: ColormapName): PlotlyColorscale | string {
  switch (name) {
    case "Viridis": return VIRIDIS;
    case "Plasma":  return PLASMA;
    default:        return name; // Plotly built-in
  }
}

// Accent colors for field name chips in the UI
export const FIELD_CHIP_COLOR: Record<FieldName, string> = {
  phi:         "text-sky-300 bg-sky-900/40",
  A_magnitude: "text-violet-300 bg-violet-900/40",
  B_magnitude: "text-amber-300 bg-amber-900/40",
  J_magnitude: "text-green-300 bg-green-900/40",
};
