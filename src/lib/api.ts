// Typed Tauri invoke wrappers.
// All solver communication goes through here — no direct HTTP from components.

import { invoke } from "@tauri-apps/api/core";
import type {
  HypothesisEntry,
  SolveRequest,
  SolveResult,
  SolverStatus,
} from "./fieldTypes";

export async function solve(request: SolveRequest): Promise<SolveResult> {
  return invoke<SolveResult>("solve", { request });
}

export async function getSolverStatus(): Promise<SolverStatus> {
  return invoke<SolverStatus>("get_solver_status");
}

export async function saveHypothesis(
  name: string,
  request: SolveRequest,
  result: SolveResult,
  notes?: string,
): Promise<string> {
  return invoke<string>("save_hypothesis", { name, request, result, notes: notes ?? null });
}

export async function loadHypotheses(): Promise<HypothesisEntry[]> {
  return invoke<HypothesisEntry[]>("load_hypotheses");
}

export async function deleteHypothesis(id: string): Promise<void> {
  return invoke<void>("delete_hypothesis", { id });
}
