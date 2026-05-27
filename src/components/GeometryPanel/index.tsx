import React, { useState } from "react";
import type {
  CoilEntity, CoilParams, CoilType,
  EedParams, FieldName, GemParams, HolonomyPath,
  SolveRequest, SolverConfig, SolverMode,
} from "../../lib/fieldTypes";
import {
  AC_CAPABLE_TYPES, CAPACITOR_TYPES, COIL_LABELS,
  FIELD_CHIP, PHASE1_FIELDS,
  defaultCapacitorEntity,
} from "../../lib/fieldTypes";

interface Props {
  request:  SolveRequest;
  onChange: (r: SolveRequest) => void;
  disabled: boolean;
}

export function GeometryPanel({ request, onChange, disabled }: Props) {
  // All edits go through typed helpers so nothing gets lost.
  const set      = (patch: Partial<SolveRequest>) => onChange({ ...request, ...patch });
  const setSolver = (patch: Partial<SolverConfig>) =>
    set({ solver: { ...request.solver, ...patch } });
  const setEed   = (patch: Partial<EedParams>) =>
    set({ eed: { ...request.eed, ...patch } });
  const setGem   = (patch: Partial<GemParams>) =>
    set({ gem: { ...request.gem, ...patch } });

  // Entity management — track which entity is selected for editing.
  const [activeEntity, setActiveEntity] = useState(0);
  const entityIdx = Math.min(activeEntity, request.entities.length - 1);
  const entity    = request.entities[entityIdx];
  const setEntity = (patch: Partial<CoilEntity>) =>
    set({ entities: request.entities.map((e, i) => i === entityIdx ? { ...e, ...patch } : e) });
  const setCoil   = (patch: Partial<CoilParams>) =>
    setEntity({ coil: { ...entity.coil, ...patch } });

  const addEntity = () => {
    const newE: CoilEntity = {
      coil: {
        coil_type: "solenoid", radius_m: 0.05, turns: 10, pitch_m: 0.005,
        wire_radius_m: 0.001, current_A: 1.0,
        voltage_v: 0, frequency_hz: 0, plate_gap_m: 0, plate_aspect: 5,
      },
      position_m:    [0, 0, 0.1],
      orientation:   [0, 0, 0, 1],
      superconducting: false,
    };
    set({ entities: [...request.entities, newE] });
    setActiveEntity(request.entities.length);
  };
  const removeEntity = (idx: number) => {
    if (request.entities.length <= 1) return;
    const next = request.entities.filter((_, i) => i !== idx);
    set({ entities: next });
    setActiveEntity(Math.min(activeEntity, next.length - 1));
  };

  // Holonomy path management.
  const addHolonomyPath = () => {
    const p: HolonomyPath = { z_circle: { z_m: 0, radius_m: entity.coil.radius_m * 0.5 } };
    set({ holonomy_paths: [...request.holonomy_paths, p] });
  };
  const removeHolonomyPath = (idx: number) =>
    set({ holonomy_paths: request.holonomy_paths.filter((_, i) => i !== idx) });
  const updateHolonomyPath = (idx: number, p: HolonomyPath) =>
    set({ holonomy_paths: request.holonomy_paths.map((hp, i) => i === idx ? p : hp) });

  return (
    <div className={`flex flex-col gap-5 ${disabled ? "opacity-60 pointer-events-none" : ""}`}>

      {/* ── Entity selector ───────────────────────────────────────────── */}
      {request.entities.length > 1 && (
        <Section label="Entities">
          <div className="flex flex-wrap gap-1">
            {request.entities.map((_, i) => (
              <button
                key={i}
                onClick={() => setActiveEntity(i)}
                className={`text-xs px-2 py-0.5 rounded border transition-colors ${
                  i === entityIdx
                    ? "bg-accent/20 text-accent border-accent/40"
                    : "text-slate-500 border-rim hover:text-slate-300"
                }`}
              >
                #{i + 1}
              </button>
            ))}
          </div>
          {request.entities.length > 1 && (
            <button
              onClick={() => removeEntity(entityIdx)}
              className="text-xs text-red-500 hover:text-red-300 text-left"
            >
              ✕ Remove coil #{entityIdx + 1}
            </button>
          )}
        </Section>
      )}

      {/* ── Coil type ─────────────────────────────────────────────────── */}
      <Section label={`Coil ${request.entities.length > 1 ? `#${entityIdx + 1} ` : ""}type`}>
        <select
          className="select-dark w-full"
          value={entity.coil.coil_type}
          onChange={e => {
            const t = e.target.value as CoilType;
            // Pre-fill sensible defaults when switching to capacitor types.
            if (CAPACITOR_TYPES.includes(t)) {
              setCoil({
                coil_type:   t,
                current_A:   0,
                voltage_v:   entity.coil.voltage_v ?? 1000,
                plate_gap_m: entity.coil.plate_gap_m && entity.coil.plate_gap_m > 0
                               ? entity.coil.plate_gap_m : 0.02,
                plate_aspect: t === "capacitor_asymmetric"
                               ? (entity.coil.plate_aspect ?? 5) : 1,
              });
            } else {
              setCoil({ coil_type: t });
            }
          }}
        >
          {(Object.keys(COIL_LABELS) as CoilType[]).map(k => (
            <option key={k} value={k}>{COIL_LABELS[k]}</option>
          ))}
        </select>

        {/* Coil type description */}
        {entity.coil.coil_type === "open_helix" && (
          <div className="text-xs text-amber-600/80 mt-0.5">
            Non-closed wire — charge at tips → φ ≠ 0 with AC drive
          </div>
        )}
        {entity.coil.coil_type === "capacitor_symmetric" && (
          <div className="text-xs text-sky-600/80 mt-0.5">
            Parallel plate capacitor — uniform E between plates
          </div>
        )}
        {entity.coil.coil_type === "capacitor_asymmetric" && (
          <div className="text-xs text-violet-600/80 mt-0.5">
            TTB asymmetric — large plate + pointed electrode → non-uniform φ
          </div>
        )}

        <div className="flex gap-1.5 mt-0.5 flex-wrap">
          <button
            onClick={addEntity}
            className="text-xs text-slate-500 hover:text-slate-300 text-left"
          >
            + Add coil
          </button>
          <button
            onClick={() => {
              const cap = defaultCapacitorEntity(true);
              cap.position_m = [0, 0, 0.05];
              set({ entities: [...request.entities, cap] });
              setActiveEntity(request.entities.length);
            }}
            className="text-xs text-sky-700 hover:text-sky-400"
          >
            + Add capacitor
          </button>
        </div>
      </Section>

      {/* ── Geometry ──────────────────────────────────────────────────── */}
      {(() => {
        const isCap = CAPACITOR_TYPES.includes(entity.coil.coil_type);
        const isAC  = AC_CAPABLE_TYPES.includes(entity.coil.coil_type);
        const isAsym = entity.coil.coil_type === "capacitor_asymmetric";
        return (
          <Section label="Geometry">
            <Slider label="Radius" unit="m" value={entity.coil.radius_m}
              min={0.005} max={0.5} step={0.005}
              onChange={v => setCoil({ radius_m: v })}
              hint={isCap ? (isAsym ? "Large electrode radius" : "Plate radius") : undefined}
            />
            {!isCap && (
              <>
                <Slider label="Turns"  unit=""  value={entity.coil.turns}
                  min={1} max={100} step={1}
                  onChange={v => setCoil({ turns: Math.round(v) })} />
                <Slider label="Pitch"  unit="m" value={entity.coil.pitch_m}
                  min={0.001} max={0.05} step={0.001}
                  onChange={v => setCoil({ pitch_m: v })} />
                <Slider label="Wire r" unit="m" value={entity.coil.wire_radius_m}
                  min={0.0002} max={0.005} step={0.0002} fmt={v => v.toFixed(4)}
                  onChange={v => setCoil({ wire_radius_m: v })} />
              </>
            )}

            {/* Capacitor-specific controls */}
            {isCap && (
              <>
                <Slider label="Plate gap" unit="m" value={entity.coil.plate_gap_m ?? 0.02}
                  min={0.005} max={0.2} step={0.005}
                  onChange={v => setCoil({ plate_gap_m: v })}
                />
                {isAsym && (
                  <Slider label="Asymmetry" unit="×" value={entity.coil.plate_aspect ?? 5}
                    min={1} max={20} step={0.5} fmt={v => v.toFixed(1)}
                    onChange={v => setCoil({ plate_aspect: v })}
                    hint="Large/small electrode radius ratio"
                  />
                )}
                <Slider label="Voltage V" unit="V" value={entity.coil.voltage_v ?? 0}
                  min={0} max={50000} step={100}
                  onChange={v => setCoil({ voltage_v: v })}
                />
              </>
            )}

            {/* Current source (non-capacitor) */}
            {!isCap && (
              <Slider label="Current" unit="A" value={entity.coil.current_A}
                min={0.1} max={1000} step={0.5}
                onChange={v => setCoil({ current_A: v })} />
            )}

            {/* AC frequency (for current-carrying types) */}
            {isAC && (
              <Slider
                label="Frequency" unit="Hz"
                value={entity.coil.frequency_hz ?? 0}
                min={0} max={1e9} step={1e6}
                fmt={v => v === 0 ? "DC" : v >= 1e9 ? `${(v/1e9).toFixed(2)} GHz`
                          : v >= 1e6 ? `${(v/1e6).toFixed(1)} MHz`
                          : v >= 1e3 ? `${(v/1e3).toFixed(1)} kHz` : `${v.toFixed(0)} Hz`}
                onChange={v => setCoil({ frequency_hz: v })}
                hint={entity.coil.frequency_hz ?? 0 > 0
                  ? "AC — J(t) injected each FDTD step; φ≠0 from EED coupling"
                  : "DC — static source (no AC injection)"}
              />
            )}
          </Section>
        );
      })()}

      {/* ── Superconducting toggle (for Li-Torr GEM) ──────────────────── */}
      <Toggle
        label="Superconducting"
        hint="Enables Li-Torr gravitomagnetic coupling"
        checked={entity.superconducting}
        onChange={v => setEntity({ superconducting: v })}
      />

      {/* ── Domain & solver ───────────────────────────────────────────── */}
      <Section label="Domain">
        <Slider label="Radius"     unit="m"  value={request.solver.domain_radius_m} min={0.05} max={2.0} step={0.05} onChange={v => setSolver({ domain_radius_m: v })} />
        <Slider
          label="Grid"  unit="³"
          value={request.solver.cells_per_axis}
          min={16} max={256} step={16}
          onChange={v => setSolver({ cells_per_axis: Math.round(v) })}
          fmt={v => `${Math.round(v)}`}
        />
        <Toggle
          label="Lorenz gauge"
          hint="Force C=0 — Maxwell baseline comparison"
          checked={request.solver.lorenz_gauge}
          onChange={v => setSolver({ lorenz_gauge: v })}
        />
        <Toggle
          label="3-D volume"
          hint="Extract normalised volume for ray-march viewer"
          checked={request.request_volume}
          onChange={v => set({ request_volume: v })}
        />
        {request.request_volume && (
          <div className="flex items-center gap-1 flex-wrap pl-0.5">
            {PHASE1_FIELDS.map(f => (
              <button
                key={f}
                onClick={() => set({ volume_field: f as FieldName })}
                className={`text-xs px-2 py-0.5 rounded transition-colors border ${
                  request.volume_field === f
                    ? "bg-accent/20 text-accent border-accent/40"
                    : "text-slate-500 border-rim hover:text-slate-300 hover:border-white/20"
                }`}
              >
                {FIELD_CHIP[f as FieldName] ?? f}
              </button>
            ))}
          </div>
        )}
      </Section>

      {/* ── Solver mode ───────────────────────────────────────────────── */}
      <Section label="Mode">
        <Toggle
          label="Time-domain FDTD"
          hint="Leapfrog propagation of φ and A — reveals C-field dynamics"
          checked={request.solver.mode.mode === "time_domain"}
          onChange={v => setSolver({
            mode: v
              ? { mode: "time_domain", dt_s: 0, n_steps: 64 }
              : { mode: "static" }
          } as Partial<SolverConfig>)}
        />
        {request.solver.mode.mode === "time_domain" && (
          <>
            <Slider
              label="Steps"  unit=""
              value={(request.solver.mode as Extract<SolverMode, { mode: "time_domain" }>).n_steps}
              min={16} max={512} step={16}
              onChange={v => setSolver({ mode: { ...request.solver.mode, n_steps: Math.round(v) } as SolverMode })}
              fmt={v => String(Math.round(v))}
            />
            <div className="text-xs text-slate-600 pl-0.5">
              dt auto-set to CFL limit (dx/c√3)
            </div>
          </>
        )}
      </Section>

      {/* ── EED parameters ────────────────────────────────────────────── */}
      <Section label="EED coupling">
        <Slider label="α" unit="1/m" value={request.eed.alpha} min={0}   max={50} step={0.5}  onChange={v => setEed({ alpha: v })} />
        <Slider label="β" unit=""    value={request.eed.beta}  min={-2}  max={2}  step={0.01} onChange={v => setEed({ beta: v })}  />
        <Slider label="γ" unit=""    value={request.eed.gamma} min={0}   max={2}  step={0.01} onChange={v => setEed({ gamma: v })}
          hint="1 = full EED · 0 = standard Maxwell"
        />
      </Section>

      {/* ── Coil position (multi-entity) ──────────────────────────────── */}
      {request.entities.length > 1 && (
        <Section label={`Coil #${entityIdx + 1} position`}>
          <Slider
            label="x" unit="m"
            value={entity.position_m[0]} min={-0.5} max={0.5} step={0.01}
            onChange={v => setEntity({ position_m: [v, entity.position_m[1], entity.position_m[2]] })}
          />
          <Slider
            label="y" unit="m"
            value={entity.position_m[1]} min={-0.5} max={0.5} step={0.01}
            onChange={v => setEntity({ position_m: [entity.position_m[0], v, entity.position_m[2]] })}
          />
          <Slider
            label="z" unit="m"
            value={entity.position_m[2]} min={-0.5} max={0.5} step={0.01}
            onChange={v => setEntity({ position_m: [entity.position_m[0], entity.position_m[1], v] })}
          />
        </Section>
      )}

      {/* ── Holonomy paths ─────────────────────────────────────────────── */}
      <Section label="Holonomy paths">
        <div className="text-xs text-slate-600 mb-1">∮ A·dl along closed loops (AB phase)</div>
        {request.holonomy_paths.map((hp, i) => {
          if (!("z_circle" in hp)) return null; // only ZCircle UI for now
          const zc = hp.z_circle;
          return (
            <div key={i} className="border border-rim rounded p-2 flex flex-col gap-1.5 text-xs">
              <div className="flex justify-between items-center">
                <span className="text-slate-400">ZCircle #{i + 1}</span>
                <button
                  onClick={() => removeHolonomyPath(i)}
                  className="text-red-600 hover:text-red-400"
                >✕</button>
              </div>
              <Slider
                label="radius" unit="m"
                value={zc.radius_m} min={0.005} max={0.2} step={0.005}
                onChange={v => updateHolonomyPath(i, { z_circle: { ...zc, radius_m: v } })}
              />
              <Slider
                label="z" unit="m"
                value={zc.z_m} min={-0.1} max={0.1} step={0.005}
                onChange={v => updateHolonomyPath(i, { z_circle: { ...zc, z_m: v } })}
              />
            </div>
          );
        })}
        <button
          onClick={addHolonomyPath}
          className="text-xs text-slate-500 hover:text-slate-300 text-left"
        >
          + Add loop
        </button>
      </Section>

      {/* ── GEM gravitational sector ──────────────────────────────────── */}
      <Section label="GEM sector">
        <Toggle
          label="Enable GEM"
          hint="Gravitoelectromagnetic coupling via C field"
          checked={request.gem.enabled}
          onChange={v => setGem({ enabled: v })}
        />
        {request.gem.enabled && (
          <>
            <div className="flex flex-col gap-0.5">
              <div className="flex justify-between items-baseline">
                <span className="text-xs text-slate-400">κ<sub>g</sub></span>
                <span className="text-xs text-slate-200 tabular-nums">
                  {request.gem.kappa_g.toExponential(2)}
                </span>
              </div>
              <div className="text-xs text-slate-600 mb-1">
                KK: 7.4×10⁻²⁸ · Li-Torr: 1.14×10⁻¹¹
              </div>
              <div className="flex gap-1">
                {KAPPA_PRESETS.map(([label, val]) => (
                  <button
                    key={label}
                    onClick={() => setGem({ kappa_g: val })}
                    className={`flex-1 py-0.5 rounded text-xs border transition-colors
                      ${request.gem.kappa_g === val
                        ? "bg-emerald-900/40 border-emerald-600/50 text-emerald-300"
                        : "border-rim text-slate-400 hover:border-white/20"}`}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <input
                type="number"
                value={request.gem.kappa_g}
                step={1e-12}
                onChange={e => setGem({ kappa_g: parseFloat(e.target.value) || 0 })}
                className="bg-card border border-rim rounded px-2 py-1 text-xs mt-1
                           focus:outline-none focus:border-accent/50 tabular-nums"
                placeholder="κ_g value…"
              />
            </div>
            <Toggle
              label="Li-Torr mode"
              hint="Rotating superconductors source B_g = -(2mₑ/e)ω"
              checked={request.gem.li_torr_mode}
              onChange={v => setGem({ li_torr_mode: v })}
            />
          </>
        )}
      </Section>
    </div>
  );
}

// ── κ_g presets ───────────────────────────────────────────────────────────────

const KAPPA_PRESETS: [string, number][] = [
  ["KK",   7.4e-28],
  ["L-T",  1.14e-11],
  ["off",  0.0],
];

// ── Sub-components ────────────────────────────────────────────────────────────

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="label mb-2">{label}</div>
      <div className="flex flex-col gap-2">{children}</div>
    </div>
  );
}

interface SliderProps {
  label:    string;
  unit:     string;
  value:    number;
  min:      number;
  max:      number;
  step:     number;
  hint?:    string;
  onChange: (v: number) => void;
  fmt?:     (v: number) => string;
}

function Slider({ label, unit, value, min, max, step, hint, onChange, fmt }: SliderProps) {
  const display = fmt ? fmt(value) : value % 1 === 0 ? String(value) : value.toFixed(3);
  return (
    <div className="flex flex-col gap-0.5">
      <div className="flex justify-between items-baseline">
        <span className="text-xs text-slate-400">{label}</span>
        <span className="text-xs text-slate-200 tabular-nums">
          {display}{unit && <span className="text-slate-500 ml-0.5">{unit}</span>}
        </span>
      </div>
      {hint && <div className="text-xs text-slate-600">{hint}</div>}
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        className="w-full h-1 accent-accent cursor-pointer"
      />
    </div>
  );
}

function Toggle({
  label, hint, checked, onChange,
}: {
  label: string; hint?: string; checked: boolean; onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex items-start gap-2 cursor-pointer group">
      <div className="relative mt-0.5 shrink-0">
        <input
          type="checkbox"
          className="sr-only peer"
          checked={checked}
          onChange={e => onChange(e.target.checked)}
        />
        <div className={`w-7 h-4 rounded-full border transition-colors
          ${checked ? "bg-accent/40 border-accent/60" : "bg-white/5 border-rim"}`}
        />
        <div className={`absolute top-0.5 left-0.5 w-3 h-3 rounded-full bg-slate-300 transition-transform
          ${checked ? "translate-x-3" : ""}`}
        />
      </div>
      <div>
        <div className={`text-xs ${checked ? "text-slate-200" : "text-slate-500 group-hover:text-slate-400"}`}>
          {label}
        </div>
        {hint && <div className="text-xs text-slate-600">{hint}</div>}
      </div>
    </label>
  );
}
