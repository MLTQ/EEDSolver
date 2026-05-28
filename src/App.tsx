import { useCallback, useEffect, useRef, useState } from "react";
import { getSolverStatus, saveHypothesis, solve } from "./lib/api";
import type { FieldName, SolveRequest, SolveResult, SolverStatus } from "./lib/fieldTypes";
import { defaultSolveRequest, FIELD_CHIP, FIELD_UNITS, PHASE1_FIELDS } from "./lib/fieldTypes";
import { FIELD_CHIP_COLOR, DEFAULT_CHIP_COLOR } from "./lib/colormap";
import { GeometryPanel } from "./components/GeometryPanel";
import { LegendPanel }   from "./components/LegendPanel";
import { SliceViewer }   from "./components/SliceViewer";
import { VolumeViewer }  from "./components/VolumeViewer";
import { HypothesisLog } from "./components/HypothesisLog";

const DEFAULT_FIELD: FieldName = "B_magnitude";

export default function App() {
  const [request,       setRequest]       = useState<SolveRequest>(defaultSolveRequest());
  const [result,        setResult]        = useState<SolveResult | null>(null);
  const [isSolving,     setIsSolving]     = useState(false);
  const [solverStatus,  setSolverStatus]  = useState<SolverStatus>({
    state: "initialising", message: "Checking…", gpu_name: null,
  });
  const [selectedField, setSelectedField] = useState<FieldName>(DEFAULT_FIELD);
  const [saveModalOpen, setSaveModalOpen] = useState(false);
  const [hypRefresh,    setHypRefresh]    = useState(0);
  const [error,         setError]         = useState<string | null>(null);
  const [showHistory,   setShowHistory]   = useState(false);
  const [showLegend,    setShowLegend]    = useState(false);

  const isSolvingRef = useRef(false);

  // Poll solver status until ready
  useEffect(() => {
    let active = true;
    async function poll() {
      while (active) {
        try {
          const s = await getSolverStatus();
          if (active) setSolverStatus(s);
          if (s.state === "ready") break;
        } catch { /* ignore */ }
        await sleep(1500);
      }
    }
    poll();
    return () => { active = false; };
  }, []);

  const handleSolve = useCallback(async () => {
    if (isSolvingRef.current) return;
    isSolvingRef.current = true;
    setIsSolving(true);
    setError(null);
    try {
      // Drive all slice requests with the currently selected field.
      const req: SolveRequest = {
        ...request,
        slices:       request.slices.map(s => ({ ...s, field: selectedField })),
        volume_field: selectedField,
      };
      const r = await solve(req);
      setResult(r);
      if (r.warnings.length) setError(`Warnings: ${r.warnings.join("; ")}`);
    } catch (e) {
      setError(String(e));
    } finally {
      isSolvingRef.current = false;
      setIsSolving(false);
    }
  }, [request, selectedField]);

  const solverReady = solverStatus.state === "ready";

  // Auto-solve 750 ms after any parameter change (while solver is ready).
  useEffect(() => {
    if (!solverReady) return;
    const t = setTimeout(handleSolve, 750);
    return () => clearTimeout(t);
  }, [handleSolve, solverReady]);

  const handleSave = async (name: string, notes: string) => {
    if (!result) return;
    await saveHypothesis(name, request, result, notes || undefined)
      .catch(e => setError(String(e)));
    setHypRefresh(n => n + 1);
    setSaveModalOpen(false);
  };

  // ── Layout ─────────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col h-screen bg-app text-slate-200 font-mono overflow-hidden select-none">

      {/* ── Header ─────────────────────────────────────────────────────── */}
      <header className="h-10 flex items-center gap-3 px-4 border-b border-rim shrink-0">
        <span className="text-accent font-semibold text-sm tracking-wide">Oracle</span>
        <StatusDot status={solverStatus} />
        <div className="w-px h-4 bg-rim" />

        {/* Phase 1 field chips — B, |A|, C */}
        <div className="flex gap-0.5">
          {PHASE1_FIELDS.map(f => (
            <button
              key={f}
              onClick={() => setSelectedField(f)}
              className={`text-xs px-2 py-0.5 rounded transition-colors ${
                selectedField === f
                  ? (FIELD_CHIP_COLOR[f] ?? DEFAULT_CHIP_COLOR)
                  : "text-slate-600 hover:text-slate-400"
              }`}
            >
              {FIELD_CHIP[f] ?? f}
            </button>
          ))}
        </div>

        <div className="ml-auto flex items-center gap-2">
          {/* Max value chip for selected field */}
          {result && (() => {
            const mx = result.maxima.find(m => m.field === selectedField);
            if (!mx) return null;
            const v = mx.max_value;
            const fmtV = (Math.abs(v) >= 1e3 || (Math.abs(v) < 1e-3 && v !== 0))
              ? v.toExponential(2) : v.toPrecision(3);
            const unit = FIELD_UNITS[selectedField] ?? "";
            return (
              <span className={`text-xs tabular-nums px-2 py-0.5 rounded ${
                FIELD_CHIP_COLOR[selectedField] ?? DEFAULT_CHIP_COLOR
              }`}>
                max {fmtV} {unit}
              </span>
            );
          })()}

          {/* Magnetic helicity */}
          {result && Math.abs(result.magnetic_helicity) > 1e-40 && (
            <span className="text-xs text-violet-400/80 tabular-nums" title="Magnetic helicity ∫A·B d³x">
              H={result.magnetic_helicity.toExponential(2)}
            </span>
          )}

          {/* Solve stats */}
          {result && (
            <span className="text-xs text-slate-600 tabular-nums">
              {result.solve_time_s.toFixed(2)}s · {result.grid_cells.toLocaleString()} cells
            </span>
          )}

          {result && (
            <button onClick={() => setSaveModalOpen(true)} className="btn-ghost text-xs">
              Save
            </button>
          )}
          <button
            onClick={handleSolve}
            disabled={!solverReady || isSolving}
            className="btn-primary text-xs"
          >
            {isSolving ? "Solving…" : "▶ Solve"}
          </button>
        </div>
      </header>

      {/* ── Error / warning banner ─────────────────────────────────────── */}
      {error && (
        <div className="bg-red-950/60 border-b border-red-800/40 px-4 py-1 text-xs text-red-300 flex items-center gap-2 shrink-0">
          <span className="flex-1 truncate">{error}</span>
          <button onClick={() => setError(null)} className="text-red-500 hover:text-red-300 shrink-0">✕</button>
        </div>
      )}

      {/* ── Main layout ────────────────────────────────────────────────── */}
      <div className="flex flex-1 min-h-0">

        {/* ── Controls sidebar ─────────────────────────────────────────── */}
        <aside className="w-60 shrink-0 border-r border-rim flex flex-col overflow-hidden bg-panel">
          <div className="flex-1 overflow-y-auto px-3 py-3 min-h-0">
            <GeometryPanel
              request={request}
              onChange={setRequest}
              disabled={isSolving}
            />
          </div>

          {/* Hypothesis history — collapsible at bottom */}
          <div className="shrink-0 border-t border-rim">
            <button
              onClick={() => setShowHistory(h => !h)}
              className="w-full px-3 py-2 text-xs flex items-center gap-1.5 text-slate-500 hover:text-slate-300 transition-colors"
            >
              <span>{showHistory ? "▾" : "▸"}</span>
              <span>Hypotheses</span>
              <span className="ml-auto text-slate-700">{showHistory ? "hide" : "show"}</span>
            </button>
            {showHistory && (
              <div className="h-52 overflow-hidden border-t border-rim">
                <HypothesisLog
                  onRestoreParams={r => { setRequest(r); setShowHistory(false); }}
                  refreshTrigger={hypRefresh}
                />
              </div>
            )}
          </div>
        </aside>

        {/* ── Right pane: 3-D viewer top, 2-D slices bottom ───────────── */}
        <div className="flex-1 min-w-0 flex flex-col">
          {/* 3-D volume viewer */}
          <div className="flex-1 relative min-h-0">
            <VolumeViewer
              volume={result?.volume ?? null}
              selectedField={selectedField}
              isSolving={isSolving}
              maxima={result?.maxima ?? []}
              entity={request.entities[0]}
              domainRadius={request.solver.domain_radius_m}
              leadPoints={result?.lead_points}
            />
          </div>

          {/* 2-D slice panel — shown when we have results */}
          {result && result.slices.length > 0 && (
            <div className="h-72 border-t border-rim shrink-0">
              <SliceViewer result={result} selectedField={selectedField} />
            </div>
          )}

          {/* Holonomy results panel */}
          {result && result.holonomies.length > 0 && (
            <div className="border-t border-rim shrink-0 px-4 py-2 flex items-center gap-4 flex-wrap">
              <span className="text-xs text-slate-600 shrink-0">∮ A·dl</span>
              {result.holonomies.map((h, i) => {
                const v = h.value;
                const fmtV = (Math.abs(v) >= 1e-3 || v === 0)
                  ? v.toPrecision(4) : v.toExponential(3);
                const pathLabel = "z_circle" in h.path
                  ? `r=${h.path.z_circle.radius_m.toFixed(3)}m z=${h.path.z_circle.z_m.toFixed(3)}m`
                  : "toroid" in h.path ? "toroidal"
                  : "poloidal";
                return (
                  <span key={i}
                    className="text-xs font-mono tabular-nums px-2 py-0.5 rounded bg-violet-950/40 border border-violet-800/30 text-violet-300"
                    title={pathLabel}
                  >
                    {fmtV} V·s
                  </span>
                );
              })}
            </div>
          )}
        </div>

        {/* ── Legend sidebar ───────────────────────────────────────────── */}
        <LegendPanel open={showLegend} onToggle={() => setShowLegend(v => !v)} />
      </div>

      {/* ── Save modal ─────────────────────────────────────────────────── */}
      {saveModalOpen && (
        <SaveModal
          onSave={handleSave}
          onCancel={() => setSaveModalOpen(false)}
        />
      )}
    </div>
  );
}

// ── Small UI helpers ──────────────────────────────────────────────────────────

function StatusDot({ status }: { status: SolverStatus }) {
  const { state, message, gpu_name } = status;
  const color =
    state === "ready"       ? "bg-green-400" :
    state === "error"       ? "bg-red-400"   :
                              "bg-amber-400 animate-pulse";
  const tip = gpu_name ? `${message} (${gpu_name})` : message;
  return (
    <span title={tip} className="flex items-center gap-1.5 text-xs text-slate-500 cursor-default">
      <span className={`w-1.5 h-1.5 rounded-full ${color}`} />
      {state}
    </span>
  );
}

function SaveModal({
  onSave, onCancel,
}: {
  onSave:   (name: string, notes: string) => void;
  onCancel: () => void;
}) {
  const [name,  setName]  = useState("");
  const [notes, setNotes] = useState("");

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-panel border border-rim rounded-lg p-5 w-80 flex flex-col gap-3">
        <div className="text-sm font-medium">Save hypothesis</div>
        <input
          autoFocus
          placeholder="Run name…"
          value={name}
          onChange={e => setName(e.target.value)}
          onKeyDown={e => e.key === "Enter" && name && onSave(name, notes)}
          className="bg-card border border-rim rounded px-3 py-1.5 text-sm focus:outline-none focus:border-accent/50"
        />
        <textarea
          placeholder="Notes (optional)…"
          value={notes}
          onChange={e => setNotes(e.target.value)}
          rows={2}
          className="bg-card border border-rim rounded px-3 py-1.5 text-sm resize-none focus:outline-none focus:border-accent/50"
        />
        <div className="flex justify-end gap-2">
          <button onClick={onCancel} className="btn-ghost text-xs">Cancel</button>
          <button
            onClick={() => name && onSave(name, notes)}
            disabled={!name}
            className="btn-primary text-xs"
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
}

function sleep(ms: number) { return new Promise(r => setTimeout(r, ms)); }
