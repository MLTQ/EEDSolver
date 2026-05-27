import { useCallback, useEffect, useRef, useState } from "react";
import { getSolverStatus, saveHypothesis, solve } from "./lib/api";
import type { FieldName, SolveRequest, SolveResult, SolverStatus } from "./lib/fieldTypes";
import { defaultSolveRequest, PRIMARY_FIELD } from "./lib/fieldTypes";
import { FIELD_CHIP_COLOR } from "./lib/colormap";
import { GeometryPanel } from "./components/GeometryPanel";
import { VolumeViewer }  from "./components/VolumeViewer";
import { HypothesisLog } from "./components/HypothesisLog";

export default function App() {
  const [request,      setRequest]      = useState<SolveRequest>(defaultSolveRequest());
  const [result,       setResult]       = useState<SolveResult | null>(null);
  const [isSolving,    setIsSolving]    = useState(false);
  const [solverStatus, setSolverStatus] = useState<SolverStatus>({ state: "starting", message: "Checking…" });
  const [selectedField, setSelectedField] = useState<FieldName>("phi");
  const [saveModalOpen, setSaveModalOpen] = useState(false);
  const [hypRefresh,   setHypRefresh]   = useState(0);
  const [error,        setError]        = useState<string | null>(null);
  const [showHistory,  setShowHistory]  = useState(false);

  // Ref so handleSolve can check without being in its dep array
  const isSolvingRef = useRef(false);

  // Sync selected field + volume_field when formulation changes
  useEffect(() => {
    const f = PRIMARY_FIELD[request.formulation];
    setSelectedField(f);
    setRequest(r => ({ ...r, volume_field: f }));
  }, [request.formulation]);

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
    if (isSolvingRef.current) return;   // skip if a solve is already running
    isSolvingRef.current = true;
    setIsSolving(true);
    setError(null);
    try {
      const req: SolveRequest = {
        ...request,
        slices: request.slices.map(s => ({ ...s, field: selectedField })),
        volume_field: selectedField,   // drive the 3-D viewer with the active chip
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

  // ── Auto-solve on parameter change (debounced 750 ms) ──────────────────────
  // handleSolve recreates whenever request or selectedField changes, so this
  // effect fires naturally on every param edit. The 750 ms timer resets on
  // each keystroke / slider drag, then fires once the user pauses.
  useEffect(() => {
    if (!solverReady) return;
    const t = setTimeout(handleSolve, 750);
    return () => clearTimeout(t);
  }, [handleSolve, solverReady]);

  const handleSave = async (name: string, notes: string) => {
    if (!result) return;
    await saveHypothesis(name, request, result, notes || undefined).catch(e => setError(String(e)));
    setHypRefresh(n => n + 1);
    setSaveModalOpen(false);
  };

  return (
    <div className="flex flex-col h-screen bg-app text-slate-200 font-mono overflow-hidden select-none">

      {/* ── Header ─────────────────────────────────────────────────────── */}
      <header className="h-10 flex items-center gap-3 px-4 border-b border-rim shrink-0">
        <span className="text-accent font-semibold text-sm tracking-wide">Oracle</span>
        <StatusDot state={solverStatus.state} title={solverStatus.message} />
        <div className="w-px h-4 bg-rim" />

        {/* Field selector */}
        <div className="flex gap-0.5">
          {(["phi", "A_magnitude", "B_magnitude", "J_magnitude"] as FieldName[]).map(f => (
            <button
              key={f}
              onClick={() => setSelectedField(f)}
              className={`text-xs px-2 py-0.5 rounded transition-colors ${
                selectedField === f
                  ? FIELD_CHIP_COLOR[f]
                  : "text-slate-600 hover:text-slate-400"
              }`}
            >
              {f === "phi" ? "φ" : f === "A_magnitude" ? "|A|" : f === "B_magnitude" ? "|B|" : "|J|"}
            </button>
          ))}
        </div>

        <div className="ml-auto flex items-center gap-2">
          {result && (() => {
            const mx = result.maxima.find(m => m.field === selectedField);
            if (!mx) return null;
            const v = mx.max_value;
            const fmtV = (Math.abs(v) >= 1e3 || (Math.abs(v) < 1e-3 && v !== 0))
              ? v.toExponential(2)
              : v.toPrecision(3);
            const units: Record<string, string> = {
              phi: "V", A_magnitude: "Wb/m", B_magnitude: "T", J_magnitude: "A/m²",
            };
            return (
              <span className={`text-xs tabular-nums px-2 py-0.5 rounded ${FIELD_CHIP_COLOR[selectedField]}`}>
                max {fmtV} {units[selectedField] ?? ""}
              </span>
            );
          })()}
          {result && (
            <span className="text-xs text-slate-600 tabular-nums">
              {result.solve_time_s.toFixed(1)}s · {result.mesh_nodes.toLocaleString()} nodes
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

      {/* ── Error banner ───────────────────────────────────────────────── */}
      {error && (
        <div className="bg-red-950/60 border-b border-red-800/40 px-4 py-1 text-xs text-red-300 flex items-center gap-2 shrink-0">
          <span className="flex-1">{error}</span>
          <button onClick={() => setError(null)} className="text-red-500 hover:text-red-300">✕</button>
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

          {/* Hypothesis history — collapsible at the bottom */}
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

        {/* ── 3-D viewer — fills everything right of the sidebar ───────── */}
        <div className="flex-1 min-w-0 relative">
          <VolumeViewer
            volume={result?.volume ?? null}
            selectedField={selectedField}
            isSolving={isSolving}
            maxima={result?.maxima ?? []}
            coilParams={request.coil}
            domainRadius={request.domain_radius_m}
          />
        </div>
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

// ---------------------------------------------------------------------------
// Small UI helpers
// ---------------------------------------------------------------------------

function StatusDot({ state, title }: { state: string; title: string }) {
  const color =
    state === "ready"   ? "bg-green-400" :
    state === "error"   ? "bg-red-400"   :
                          "bg-amber-400 animate-pulse";
  return (
    <span title={title} className="flex items-center gap-1.5 text-xs text-slate-500">
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
