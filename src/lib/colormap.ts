// Colorscale definitions for Plotly heatmaps and field chip UI colours.

import type { FieldName } from "./fieldTypes";

export type ColormapName = "Viridis" | "Plasma" | "RdBu" | "Hot" | "Cividis" | "Greens";

// Plotly colorscale entry: [position 0–1, css color]
export type PlotlyColorscale = [number, string][];

// ── Per-field colormap assignment ─────────────────────────────────────────────

export const FIELD_COLORMAP: Partial<Record<FieldName, ColormapName>> = {
  // EED potentials
  phi:           "Viridis",
  A_magnitude:   "Plasma",
  // EM derived
  E_magnitude:   "Hot",
  B_magnitude:   "Hot",
  J_magnitude:   "Cividis",
  // EED scalar — diverging around zero
  C_field:       "RdBu",
  // Energy observables
  poynting_mag:  "Plasma",
  energy_density:"Viridis",
  // GEM sector
  phi_g:         "Greens",
  E_g_magnitude: "Greens",
  B_g_magnitude: "Greens",
};

export const DEFAULT_COLORMAP: ColormapName = "Viridis";

// ── Per-field chip colours (Tailwind utility classes) ─────────────────────────

export const FIELD_CHIP_COLOR: Partial<Record<FieldName, string>> = {
  phi:           "text-sky-300 bg-sky-900/40",
  A_magnitude:   "text-violet-300 bg-violet-900/40",
  E_magnitude:   "text-orange-300 bg-orange-900/40",
  B_magnitude:   "text-amber-300 bg-amber-900/40",
  J_magnitude:   "text-green-300 bg-green-900/40",
  // C field — distinctive teal (the deleted DOF)
  C_field:       "text-teal-300 bg-teal-900/40",
  poynting_mag:  "text-rose-300 bg-rose-900/40",
  energy_density:"text-yellow-300 bg-yellow-900/40",
  // GEM sector — emerald
  phi_g:         "text-emerald-300 bg-emerald-900/40",
  E_g_magnitude: "text-emerald-300 bg-emerald-900/40",
  B_g_magnitude: "text-emerald-300 bg-emerald-900/40",
};

export const DEFAULT_CHIP_COLOR = "text-slate-300 bg-slate-800/40";

// ── Colorscale data ───────────────────────────────────────────────────────────

export const VIRIDIS: PlotlyColorscale = [
  [0.0, "#440154"], [0.1, "#482878"], [0.2, "#3e4989"],
  [0.3, "#31688e"], [0.4, "#26828e"], [0.5, "#1f9e89"],
  [0.6, "#35b779"], [0.7, "#6ece58"], [0.8, "#b5de2b"],
  [0.9, "#dde318"], [1.0, "#fde725"],
];

export const PLASMA: PlotlyColorscale = [
  [0.0, "#0d0887"], [0.1, "#41049d"], [0.2, "#6a00a8"],
  [0.3, "#8f0da4"], [0.4, "#b12a90"], [0.5, "#cc4778"],
  [0.6, "#e16462"], [0.7, "#f2844b"], [0.8, "#fca636"],
  [0.9, "#fcce25"], [1.0, "#f0f921"],
];

export function getColorscale(name: ColormapName = DEFAULT_COLORMAP): PlotlyColorscale | string {
  switch (name) {
    case "Viridis": return VIRIDIS;
    case "Plasma":  return PLASMA;
    default:        return name; // Plotly built-in name
  }
}
