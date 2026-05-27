import React, { useMemo } from "react";
// @ts-expect-error — react-plotly.js types are loose
import Plot from "react-plotly.js";
import type { FieldName, SliceAxis, SliceData, SolveResult } from "../../lib/fieldTypes";
import { FIELD_LABELS } from "../../lib/fieldTypes";
import { FIELD_COLORMAP, DEFAULT_COLORMAP, getColorscale } from "../../lib/colormap";

interface Props {
  result:        SolveResult | null;
  selectedField: FieldName;
}

export function SliceViewer({ result, selectedField }: Props) {
  const [axis, setAxis] = React.useState<SliceAxis>("z");

  const slice = useMemo<SliceData | null>(() => {
    if (!result) return null;
    return (
      result.slices.find(s => s.axis === axis && s.field === selectedField) ??
      result.slices.find(s => s.axis === axis) ??
      result.slices[0] ??
      null
    );
  }, [result, axis, selectedField]);

  if (!result) {
    return <Placeholder text="Run a solve to see slice data" />;
  }
  if (!slice) {
    return <Placeholder text={`No ${axis.toUpperCase()} slice in this result.`} />;
  }

  const [rows, cols] = slice.shape;
  const z: number[][] = [];
  for (let r = 0; r < rows; r++) {
    z.push(slice.data.slice(r * cols, (r + 1) * cols) as unknown as number[]);
  }

  const cmapName  = FIELD_COLORMAP[slice.field] ?? DEFAULT_COLORMAP;
  const colorscale = getColorscale(cmapName);
  const axisPair  = axis === "z" ? ["x (m)", "y (m)"] : axis === "x" ? ["y (m)", "z (m)"] : ["x (m)", "z (m)"];
  const fieldLabel = FIELD_LABELS[slice.field] ?? slice.field;

  // For C field (RdBu): diverge around zero — symmetric colour limits.
  const useSymmetric = slice.field === "C_field";
  const absMax = Math.max(Math.abs(slice.field_min), Math.abs(slice.field_max));
  const zmin = useSymmetric ? -absMax : slice.field_min;
  const zmax = useSymmetric ?  absMax : slice.field_max;

  return (
    <div className="flex flex-col h-full bg-panel">
      {/* Axis tabs */}
      <div className="flex items-center gap-1 px-3 pt-2 pb-1 border-b border-rim shrink-0">
        {(["x", "y", "z"] as SliceAxis[]).map(a => (
          <button
            key={a}
            onClick={() => setAxis(a)}
            className={`px-3 py-0.5 rounded text-xs transition-colors
              ${axis === a
                ? "bg-accent/20 text-accent border border-accent/40"
                : "text-slate-500 hover:text-slate-300 border border-transparent"}`}
          >
            {a.toUpperCase()}
          </button>
        ))}
        <span className="ml-auto text-xs text-slate-500">
          {fieldLabel}
          <span className="ml-2 text-slate-600">
            [{slice.field_min.toExponential(2)}, {slice.field_max.toExponential(2)}]
          </span>
        </span>
      </div>

      {/* Heatmap */}
      <div className="flex-1 min-h-0">
        <Plot
          data={[{
            type:       "heatmap",
            z,
            x:          linspace(slice.x_range[0], slice.x_range[1], cols),
            y:          linspace(slice.y_range[0], slice.y_range[1], rows),
            colorscale,
            zmin,
            zmax,
            colorbar: {
              thickness: 12,
              len:       0.8,
              tickfont:  { color: "#94a3b8", size: 10, family: "JetBrains Mono, monospace" },
              outlinecolor: "transparent",
            },
          }]}
          layout={{
            autosize:     true,
            margin:       { t: 8, r: 60, b: 36, l: 48 },
            paper_bgcolor:"transparent",
            plot_bgcolor: "#09090d",
            xaxis: {
              title:    { text: axisPair[0], font: { color: "#64748b", size: 10 } },
              tickfont: { color: "#64748b", size: 9 },
              gridcolor:"rgba(255,255,255,0.05)",
              zerolinecolor:"rgba(255,255,255,0.1)",
            },
            yaxis: {
              title:    { text: axisPair[1], font: { color: "#64748b", size: 10 } },
              tickfont: { color: "#64748b", size: 9 },
              gridcolor:"rgba(255,255,255,0.05)",
              zerolinecolor:"rgba(255,255,255,0.1)",
              scaleanchor:"x",
            },
          }}
          config={{ displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] }}
          useResizeHandler
          style={{ width: "100%", height: "100%" }}
        />
      </div>

      {/* Stats bar */}
      <div className="px-3 py-1 border-t border-rim text-xs text-slate-600 shrink-0 flex gap-4">
        <span>pos {(slice.position * 100).toFixed(0)}%</span>
        <span>{rows}×{cols}</span>
        {result.maxima
          .filter(m => m.field === slice.field)
          .map(m => (
            <span key={m.field} className="text-slate-500">
              max {m.max_value.toExponential(3)} @ [
              {m.max_location.map(v => v.toFixed(3)).join(", ")}
              ]
            </span>
          ))}
      </div>
    </div>
  );
}

function Placeholder({ text }: { text: string }) {
  return (
    <div className="flex items-center justify-center h-full text-slate-600 text-xs bg-panel">
      {text}
    </div>
  );
}

function linspace(min: number, max: number, n: number): number[] {
  if (n <= 1) return [min];
  return Array.from({ length: n }, (_, i) => min + (i / (n - 1)) * (max - min));
}
