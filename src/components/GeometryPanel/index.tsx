import React from "react";
import type { CoilParams, EedParams, FormulationType, MeshResolution, SolveRequest } from "../../lib/fieldTypes";
import { COIL_LABELS } from "../../lib/fieldTypes";

interface Props {
  request: SolveRequest;
  onChange: (r: SolveRequest) => void;
  disabled: boolean;
}

export function GeometryPanel({ request, onChange, disabled }: Props) {
  const set = (patch: Partial<SolveRequest>) => onChange({ ...request, ...patch });
  const setCoil = (patch: Partial<CoilParams>) => set({ coil: { ...request.coil, ...patch } });
  const setEed  = (patch: Partial<EedParams>)  => set({ eed:  { ...request.eed,  ...patch } });

  const showAlpha = request.formulation !== "maxwell_only";
  const showBeta  = request.formulation === "eed_coupled";

  return (
    <div className={`flex flex-col gap-5 ${disabled ? "opacity-60 pointer-events-none" : ""}`}>

      {/* Coil type */}
      <Section label="Coil type">
        <select
          className="select-dark w-full"
          value={request.coil.coil_type}
          onChange={e => setCoil({ coil_type: e.target.value as CoilParams["coil_type"] })}
        >
          {(Object.keys(COIL_LABELS) as Array<keyof typeof COIL_LABELS>).map(k => (
            <option key={k} value={k}>{COIL_LABELS[k]}</option>
          ))}
        </select>
      </Section>

      {/* Geometry */}
      <Section label="Geometry">
        <Slider label="Radius" unit="m"  value={request.coil.radius_m}      min={0.005} max={0.5}   step={0.005} onChange={v => setCoil({ radius_m: v })} />
        <Slider label="Turns"  unit=""   value={request.coil.turns}          min={1}     max={100}   step={1}     onChange={v => setCoil({ turns: Math.round(v) })} />
        <Slider label="Pitch"  unit="m"  value={request.coil.pitch_m}        min={0.001} max={0.05}  step={0.001} onChange={v => setCoil({ pitch_m: v })} />
        <Slider label="Wire r" unit="m"  value={request.coil.wire_radius_m}  min={0.0002} max={0.005} step={0.0002} onChange={v => setCoil({ wire_radius_m: v })} fmt={v => v.toFixed(4)} />
        <Slider label="Current" unit="A" value={request.coil.current_A}      min={0.1}   max={100}   step={0.1}   onChange={v => setCoil({ current_A: v })} />
      </Section>

      {/* Domain */}
      <Section label="Domain">
        <Slider label="Radius" unit="m" value={request.domain_radius_m} min={0.05} max={2.0} step={0.05} onChange={v => set({ domain_radius_m: v })} />
      </Section>

      {/* EED parameters */}
      {showAlpha && (
        <Section label="EED parameters">
          <Slider label="α" unit="1/m" value={request.eed.alpha} min={0} max={50} step={0.5} onChange={v => setEed({ alpha: v })} />
          {showBeta && <>
            <Slider label="β" unit="" value={request.eed.beta} min={-5} max={5} step={0.01} onChange={v => setEed({ beta: v })} />
            <Slider label="γ" unit="" value={request.eed.gamma} min={-5} max={5} step={0.01} onChange={v => setEed({ gamma: v })} />
          </>}
        </Section>
      )}

      {/* Solver settings */}
      <Section label="Solver">
        <div className="flex flex-col gap-2">
          <div className="flex gap-2">
            {(["coarse", "medium", "fine"] as MeshResolution[]).map(m => (
              <button
                key={m}
                onClick={() => set({ mesh_resolution: m })}
                className={`flex-1 py-1 rounded text-xs border transition-colors
                  ${request.mesh_resolution === m
                    ? "bg-accent/20 border-accent/50 text-accent"
                    : "border-rim text-slate-400 hover:border-white/20"}`}
              >
                {m}
              </button>
            ))}
          </div>
          <div className="flex flex-col gap-1 mt-1">
            {(["scalar_only", "maxwell_only", "eed_coupled"] as FormulationType[]).map(f => (
              <label key={f} className="flex items-center gap-2 cursor-pointer group">
                <input
                  type="radio"
                  name="formulation"
                  value={f}
                  checked={request.formulation === f}
                  onChange={() => set({ formulation: f })}
                  className="accent-accent"
                />
                <span className={`text-xs ${request.formulation === f ? "text-slate-200" : "text-slate-500 group-hover:text-slate-400"}`}>
                  {f === "scalar_only" ? "scalar only (φ)" : f === "maxwell_only" ? "maxwell (A,B)" : "EED coupled (φ,A)"}
                </span>
              </label>
            ))}
          </div>
        </div>
      </Section>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="label mb-2">{label}</div>
      <div className="flex flex-col gap-2">{children}</div>
    </div>
  );
}

interface SliderProps {
  label: string;
  unit:  string;
  value: number;
  min:   number;
  max:   number;
  step:  number;
  onChange: (v: number) => void;
  fmt?: (v: number) => string;
}

function Slider({ label, unit, value, min, max, step, onChange, fmt }: SliderProps) {
  const display = fmt ? fmt(value) : value % 1 === 0 ? String(value) : value.toFixed(3);
  return (
    <div className="flex items-center gap-2">
      <span className="w-14 text-xs text-slate-400 shrink-0">{label}</span>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        className="flex-1 h-1 accent-accent cursor-pointer"
      />
      <span className="w-16 text-right text-xs text-slate-300 tabular-nums shrink-0">
        {display}{unit && <span className="text-slate-500 ml-0.5">{unit}</span>}
      </span>
    </div>
  );
}
