import { useEffect, useState } from "react";
import { deleteHypothesis, loadHypotheses } from "../../lib/api";
import type { FieldName, HypothesisEntry, SolveRequest } from "../../lib/fieldTypes";
import { FIELD_CHIP_COLOR, DEFAULT_CHIP_COLOR } from "../../lib/colormap";

interface Props {
  onRestoreParams: (r: SolveRequest) => void;
  refreshTrigger:  number;
}

export function HypothesisLog({ onRestoreParams, refreshTrigger }: Props) {
  const [entries, setEntries] = useState<HypothesisEntry[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    loadHypotheses()
      .then(setEntries)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [refreshTrigger]);

  const handleDelete = async (id: string) => {
    await deleteHypothesis(id).catch(console.error);
    setEntries(prev => prev.filter(e => e.id !== id));
  };

  if (loading) return <div className="p-3 text-xs text-slate-600">Loading…</div>;
  if (!entries.length) return (
    <div className="p-3 text-xs text-slate-600">
      No saved runs yet. Solve and click Save.
    </div>
  );

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="overflow-y-auto flex-1">
        {entries.map(e => (
          <EntryRow
            key={e.id}
            entry={e}
            onRestore={() => onRestoreParams(e.request)}
            onDelete={() => handleDelete(e.id)}
          />
        ))}
      </div>
    </div>
  );
}

function EntryRow({
  entry, onRestore, onDelete,
}: {
  entry: HypothesisEntry;
  onRestore: () => void;
  onDelete: () => void;
}) {
  const primaryMax = entry.maxima[0];
  const ts = new Date(entry.timestamp);
  const coilType = entry.request.entities[0]?.coil?.coil_type ?? "?";
  const n3 = entry.request.solver.cells_per_axis;
  const gemOn = entry.request.gem.enabled;

  return (
    <div className="flex items-start gap-2 px-3 py-2 border-b border-rim hover:bg-white/3 group">
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-0.5">
          <span className="text-slate-200 text-xs font-medium truncate">{entry.name}</span>
          <span className="text-slate-600 text-xs shrink-0">
            {ts.toLocaleDateString()} {ts.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
          </span>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <Chip label={coilType} />
          <Chip label={`${n3}³`} />
          {gemOn && <Chip label="GEM" color="text-emerald-400 bg-emerald-900/30" />}
          {primaryMax && (
            <span className={`text-xs px-1.5 py-0 rounded ${
              FIELD_CHIP_COLOR[primaryMax.field as FieldName] ?? DEFAULT_CHIP_COLOR
            }`}>
              max {primaryMax.max_value.toExponential(2)}
            </span>
          )}
        </div>
        {entry.notes && (
          <div className="text-xs text-slate-500 mt-1 truncate">{entry.notes}</div>
        )}
      </div>
      <div className="flex gap-1 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
        <button
          onClick={onRestore}
          className="text-xs text-accent hover:text-sky-300 px-1.5 py-0.5 rounded hover:bg-white/5"
          title="Restore params"
        >
          ↩
        </button>
        <button
          onClick={onDelete}
          className="text-xs text-slate-500 hover:text-red-400 px-1.5 py-0.5 rounded hover:bg-white/5"
          title="Delete"
        >
          ✕
        </button>
      </div>
    </div>
  );
}

function Chip({ label, color }: { label: string; color?: string }) {
  return (
    <span className={`text-xs px-1.5 py-0 rounded ${color ?? "text-slate-500 bg-white/5"}`}>
      {label}
    </span>
  );
}
