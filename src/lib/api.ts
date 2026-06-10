// Typed Tauri invoke wrappers.
// All solver communication goes through here: Tauri IPC for the desktop webview,
// HTTP for Gruve/browser viewers.

import { invoke } from "@tauri-apps/api/core";
import { apiBase, isServedByGruve } from "gruve-sdk";
import type {
  HypothesisEntry,
  SolveRequest,
  SolveResult,
  SolverStatus,
} from "./fieldTypes";

declare global {
  interface Window {
    __TAURI__?: unknown;
    __TAURI_INTERNALS__?: unknown;
  }
}

const API_BASE = apiBase("api", { fallback: "" });

export async function solve(request: SolveRequest): Promise<SolveResult> {
  if (canUseTauriInvoke()) {
    return invoke<SolveResult>("solve", { request });
  }

  return apiRequest<SolveResult>("/api/solve", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export async function getSolverStatus(): Promise<SolverStatus> {
  if (canUseTauriInvoke()) {
    return invoke<SolverStatus>("get_solver_status");
  }

  return apiRequest<SolverStatus>("/api/solver-status");
}

export async function saveHypothesis(
  name: string,
  request: SolveRequest,
  result: SolveResult,
  notes?: string,
): Promise<string> {
  if (canUseTauriInvoke()) {
    return invoke<string>("save_hypothesis", { name, request, result, notes: notes ?? null });
  }

  return apiRequest<string>("/api/hypotheses", {
    method: "POST",
    body: JSON.stringify({ name, request, result, notes: notes ?? null }),
  });
}

export async function loadHypotheses(): Promise<HypothesisEntry[]> {
  if (canUseTauriInvoke()) {
    return invoke<HypothesisEntry[]>("load_hypotheses");
  }

  return apiRequest<HypothesisEntry[]>("/api/hypotheses");
}

export async function deleteHypothesis(id: string): Promise<void> {
  if (canUseTauriInvoke()) {
    return invoke<void>("delete_hypothesis", { id });
  }

  await apiRequest<void>(`/api/hypotheses/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
}

function canUseTauriInvoke(): boolean {
  if (isServedByGruve()) return false;
  if (typeof window === "undefined") return false;
  return Boolean(window.__TAURI__ || window.__TAURI_INTERNALS__);
}

async function apiRequest<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body && !headers.has("content-type")) {
    headers.set("content-type", "application/json");
  }

  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers,
  });

  if (!response.ok) {
    const message = await response.text().catch(() => response.statusText);
    throw new Error(message || response.statusText);
  }

  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}
