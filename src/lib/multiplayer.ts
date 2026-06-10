import { isServedByGruve, joinSession } from "gruve-sdk";
import type { FieldName, SolveRequest, SolveResult } from "./fieldTypes";
import { PHASE1_FIELDS } from "./fieldTypes";

export const ORACLE_SESSION_KEYS = {
  request: "oracle.request",
  selectedField: "oracle.selectedField",
  result: "oracle.result",
} as const;

export type OracleSession = ReturnType<typeof joinSession>;
export type OracleSessionKey = typeof ORACLE_SESSION_KEYS[keyof typeof ORACLE_SESSION_KEYS];

export function joinOracleSession(onPeers: (count: number) => void): OracleSession {
  return joinSession({ onPeers });
}

export function isGruveSharedSession(): boolean {
  if (!isServedByGruve()) return false;
  return !new URLSearchParams(location.search).has("gruve-solo");
}

export function isFieldName(value: unknown): value is FieldName {
  return typeof value === "string" && PHASE1_FIELDS.includes(value as FieldName);
}

export function isSolveRequest(value: unknown): value is SolveRequest {
  if (!isRecord(value)) return false;
  return (
    Array.isArray(value.entities) &&
    isRecord(value.eed) &&
    isRecord(value.gem) &&
    isRecord(value.solver) &&
    Array.isArray(value.slices) &&
    typeof value.request_volume === "boolean" &&
    typeof value.volume_field === "string" &&
    Array.isArray(value.holonomy_paths)
  );
}

export function isSolveResult(value: unknown): value is SolveResult {
  if (!isRecord(value)) return false;
  return (
    typeof value.solve_time_s === "number" &&
    typeof value.grid_cells === "number" &&
    Array.isArray(value.slices) &&
    Array.isArray(value.maxima) &&
    Array.isArray(value.holonomies) &&
    Array.isArray(value.warnings)
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
