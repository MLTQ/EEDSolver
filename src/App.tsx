import { useCallback, useEffect, useState } from "react";
import { getSolverStatus, saveHypothesis, solve } from "./lib/api";
import type { FieldName, SolveRequest, SolveResult, SolverStatus } from "./lib/fieldTypes";
import { defaultSolveRequest, PRIMARY_FIELD } from "./lib/fieldTypes";
import { FIELD_CHIP_COLOR } from "./lib/colormap";
import { GeometryPanel } from "./components/GeometryPanel";
import { SliceViewer }   from "./components/SliceViewer";
import { VolumeViewer }  from "./components/VolumeViewer";
import { HypothesisLog } from "./components/HypothesisLog";

type BottomTab = "slices" | "hypotheses";

export default function App() {
  const [request,       setRequest]       = useState<SolveRequest>(defaultSolveRequest());
  const [result,        setResult]        = useState<SolveResult | null>(null);
  const [isSolving,     setIsSolving]     = useState(false);
  const [solverStatus,  setSolverStatus]  = useState<SolverStatus>({ state: "starting", message: "Checking…" });
  const [selectedField, setSelectedField] = useState<FieldName>("phi");
  const [bottomTab,     setBottomTab]     = useState<BottomTab>("slices");
  const [saveModalOpen, setSaveModalOpen] = useState(false);
  const [hypRefresh,    setHypRefresh]    = useState(0);
  const [error,         setError]         = useState<string | null>(null);

  // Sync selectedField when formulation changes
  useEffect(() => {
    setSelectedField(PRIMARY_FIELD[request.formulation]);
  }, [request.formulation]);

  // Poll solver status on mount
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
    setIsSolving(true);
    setError(null);
    try {
      // Ensure slices use the current field
      const req: SolveRequest = {
        ...request,
        slices: request.slices.map(s => ({ ...s, field: selectedField })),
      };
      const r = await solve(req);
      setResult(r);
      if (r.warnings.length) setError(`Warnings: ${r.warnings.join("; ")}`);
    } catch (e) {
      setError(String(e));
    } finally {
      setIsSolving(false);
    }
  }, [request, selectedField]);

  const handleSave = async (name: string, notes: string) => {
    if (!result) return;
    await saveHypothesis(name, request, result, notes || undefined).catch(e => setError(String(e)));
    setHypRefresh(n => n + 1);
    setSaveModalOpen(false);
  };

  const solverReady = solverStatus.state === "ready";

  return (
    <div className="flex flex-col h-screen bg-app text-slate-200 font-mono overflow-hidden select-none">

      {/* ── Header ─────────────────────────────────────────────────────── */}
      <header className="h-11 flex items-center gap-3 px-4 border-b border-rim shrink-0">
        <span className="text-accent font-semibold text-sm tracking-wide mr-1">Oracle</span>

        {/* Solver status indicator */}
        <StatusDot state={solverStatus.state} title={solverStatus.message} />

        <div className="w-px h-4 bg-rim mx-1" />

        {/* Field selector */}
        <div className="flex gap-1">
          {(["phi", "A_magnitude", "B_magnitude", "J_magnitude"] as FieldName[]).map(f => (
            <button
              key={f}
              onClick={() => setSelectedField(f)}
              className={`text-xs px-2 py-1 rounded transition-colors ${
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
          {result && (
            <span className="text-xs text-slate-600">
              {result.solve_time_s.toFixed(1)}s · {result.mesh_nodes.toLocaleString()} nodes
            </span>
          )}
          {result && (
            <button
              onClick={() => setSaveModalOpen(true)}
              className="btn-ghost text-xs"
            >
              Save run
            </button>
          )}
          <button
            onClick={handleSolve}
            disabled={!solverReady || isSolving}
            className="btn-primary text-sm"
          >
            {isSolving ? "Solving…" : "▶ Solve"}
          </button>
        </div>
      </header>

      {/* ── Error banner ──────────────────────────────────────────────── */}
      {error && (
        <div className="bg-red-950/60 border-b border-red-800/40 px-4 py-1.5 text-xs text-red-300 flex items-center gap-2 shrink-0">
          <span className="flex-1">{error}</span>
          <button onClick={() => setError(null)} className="text-red-500 hover:text-red-300">✕</button>
        </div>
      )}

      {/* ── Main layout ───────────────────────────────────────────────── */}
      <div className="flex flex-1 min-h-0">

        {/* Sidebar */}
        <aside className="w-64 shrink-0 border-r border-rim flex flex-col overflow-hidden bg-panel">
          <div className="flex-1 overflow-y-auto px-4 py-4">
            <GeometryPanel
              request={request}
              onChange={setRequest}
              disabled={isSolving}
            />
          </div>
        </aside>

        {/* Content column */}
        <div className="flex-1 flex flex-col min-w-0">

          {/* 3D Volume viewer — primary view */}
          <div className="flex-1 min-h-0">
            <VolumeViewer
              volume={result?.volume ?? null}
              selectedField={selectedField}
              isSolving={isSolving}
            />
          </div>

          {/* Bottom panel — slices + hypotheses */}
          <div className="h-64 shrink-0 border-t border-rim flex flex-col bg-panel">
            {/* Tab bar */}
            <div className="flex border-b border-rim shrink-0">
              {(["slices", "hypotheses"] as BottomTab[]).map(t => (
                <button
                  key={t}
                  onClick={() => setBottomTab(t)}
                  className={`px-4 py-2 text-xs transition-colors border-r border-rim
                    ${bottomTab === t
                      ? "text-slate-200 bg-white/4"
                      : "text-slate-500 hover:text-slate-300"}`}
                >
                  {t === "slices" ? "2D Slices" : "Hypotheses"}
                </button>
              ))}
              {/* Maxima summary */}
              {result && (
                <div className="ml-auto px-3 flex items-center gap-3 text-xs text-slate-600 overflow-x-auto">
                  {result.maxima.slice(0, 3).map(m => (
                    <span key={m.field} className={`shrink-0 ${FIELD_CHIP_COLOR[m.field as FieldName]}`}>
                      {m.field === "phi" ? "φ" : m.field === "A_magnitude" ? "|A|" : m.field === "B_magnitude" ? "|B|" : "|J|"}
                      {" "}{m.max_value.toExponential(2)}
                    </span>
                  ))}
                </div>
              )}
            </div>

            {/* Tab content */}
            <div className="flex-1 min-h-0 overflow-hidden">
              {bottomTab === "slices" ? (
                <SliceViewer result={result} selectedField={selectedField} />
              ) : (
                <HypothesisLog
                  onRestoreParams={r => { setRequest(r); }}
                  refreshTrigger={hypRefresh}
                />
              )}
            </div>
          </div>
        </div>
      </div>

      {/* ── Save modal ────────────────────────────────────────────────── */}
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
// Small UI components
// ---------------------------------------------------------------------------

function StatusDot({ state, title }: { state: string; title: string }) {
  const color = state === "ready" ? "bg-green-400"
              : state === "error" ? "bg-red-400"
              : "bg-amber-400 animate-pulse";
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
  onSave: (name: string, notes: string) => void;
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
