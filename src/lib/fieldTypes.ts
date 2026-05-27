// ════════════════════════════════════════════════════════════════════════════
// fieldTypes.ts — v2 Oracle type system
// ════════════════════════════════════════════════════════════════════════════
//
// Mirrors solver-gpu/src/types.rs.  Keep in sync — serde field names are
// the JSON keys, TypeScript field names must match exactly.

// ── Primitives ───────────────────────────────────────────────────────────────

export type CoilType =
  | "solenoid"
  | "toroid"
  | "toroid_poloidal"
  | "flat_spiral"
  | "rodin";

export type SliceAxis = "x" | "y" | "z";

export type SolverState = "initialising" | "ready" | "solving" | "error";

/** All field quantities the solver can output. */
export type FieldName =
  // EED potentials (primary state)
  | "phi"             // φ  [V]
  | "A_magnitude"     // |A|  [V·s/m]
  // EED derived EM
  | "E_magnitude"     // |E|  [V/m]
  | "B_magnitude"     // |B|  [T]
  | "J_magnitude"     // |J|  [A/m²]
  // EED scalar — the deleted DOF
  | "C_field"         // C = ∇·A + (1/c²)∂φ/∂t  [1/m]
  // EED energy
  | "poynting_mag"    // |P| = |E×B − EC|  [W/m²]
  | "energy_density"  // u = ½(E²+B²+C²)  [J/m³]
  // GEM gravitational sector
  | "phi_g"           // Φ_g  [m²/s²]
  | "E_g_magnitude"   // |E_g|  [m/s²]
  | "B_g_magnitude";  // |B_g|  [1/s]

// ── Physics types ─────────────────────────────────────────────────────────────

export interface CoilParams {
  coil_type:     CoilType;
  radius_m:      number;
  turns:         number;
  pitch_m:       number;
  wire_radius_m: number;
  current_A:     number;
}

/** One current-carrying entity in the simulation. */
export interface CoilEntity {
  coil:            CoilParams;
  /** Centre position [x, y, z] metres. */
  position_m:      [number, number, number];
  /** Unit quaternion [x, y, z, w].  Identity = [0,0,0,1]. */
  orientation:     [number, number, number, number];
  /** Superconducting body for Li-Torr GEM coupling. */
  superconducting: boolean;
}

/** EED coupling constants (Stueckelberg Lagrangian). */
export interface EedParams {
  /** Yukawa scalar mass [1/m].  0 = massless C field. */
  alpha: number;
  /** A→φ coupling.  0 = decoupled. */
  beta:  number;
  /** φ→A coupling.  1 = full EED;  0 = standard Maxwell. */
  gamma: number;
}

/** GEM (gravitoelectromagnetic) sector parameters. */
export interface GemParams {
  enabled:      boolean;
  /** C-field → GEM coupling κ_g. */
  kappa_g:      number;
  /** Enable Li-Torr gravitomagnetic London moment. */
  li_torr_mode: boolean;
}

/** Solve mode — tagged by `mode` field (matches Rust serde tag). */
export type SolverMode =
  | { mode: "static" }
  | { mode: "time_domain"; dt_s: number; n_steps: number };

export interface SolverConfig {
  mode:            SolverMode;
  /** Yee cells per axis.  Grid has (cells+1)³ vertices. */
  cells_per_axis:  number;
  domain_radius_m: number;
  /** Force Lorenz gauge (C=0) — standard Maxwell baseline. */
  lorenz_gauge:    boolean;
}

// ── Output spec ───────────────────────────────────────────────────────────────

export interface SliceRequest {
  axis:       SliceAxis;
  position:   number;   // 0–1 normalised position along axis
  field:      FieldName;
  resolution: number;   // grid points per side in the output slice
}

export type HolonomyPath =
  | { z_circle:      { z_m: number; radius_m: number } }
  | { toroidal_loop: { centre_m: [number,number,number]; major_radius_m: number } }
  | { poloidal_loop: { centre_m: [number,number,number]; major_radius_m: number; minor_radius_m: number } };

// ── Top-level request ─────────────────────────────────────────────────────────

export interface SolveRequest {
  entities:       CoilEntity[];
  eed:            EedParams;
  gem:            GemParams;
  solver:         SolverConfig;
  slices:         SliceRequest[];
  request_volume: boolean;
  volume_field:   FieldName;
  holonomy_paths: HolonomyPath[];
}

// ── Results ───────────────────────────────────────────────────────────────────

export interface SliceData {
  axis:      SliceAxis;
  position:  number;
  field:     FieldName;
  shape:     [number, number];
  data:      number[];
  x_range:   [number, number];
  y_range:   [number, number];
  field_min: number;
  field_max: number;
}

export interface VolumeData {
  field:     FieldName;
  shape:     [number, number, number];
  data:      number[];   // normalised [0,1], row-major
  x_range:   [number, number];
  y_range:   [number, number];
  z_range:   [number, number];
  field_min: number;
  field_max: number;
}

export interface FieldMaximum {
  field:        FieldName;
  max_value:    number;
  max_location: [number, number, number];
}

export interface HolonomyResult {
  path:  HolonomyPath;
  /** ∮ A·dl  [V·s] */
  value: number;
}

export interface SolveResult {
  solve_time_s: number;
  grid_cells:   number;   // (cells_per_axis)³
  slices:       SliceData[];
  volume:       VolumeData | null;
  maxima:       FieldMaximum[];
  holonomies:   HolonomyResult[];
  warnings:     string[];
}

export interface SolverStatus {
  state:    SolverState;
  message:  string;
  gpu_name: string | null;
}

export interface HypothesisEntry {
  id:        string;
  name:      string;
  timestamp: string;
  request:   SolveRequest;
  maxima:    FieldMaximum[];
  notes?:    string;
}

// ── Labels & UI helpers ───────────────────────────────────────────────────────

export const FIELD_LABELS: Partial<Record<FieldName, string>> = {
  phi:           "φ  (scalar pot.)",
  A_magnitude:   "|A|  (vector pot.)",
  E_magnitude:   "|E|  (electric)",
  B_magnitude:   "|B|  (magnetic)",
  C_field:       "C  (EED scalar ∇·A)",
  poynting_mag:  "|P|  (Poynting)",
  energy_density:"u  (energy dens.)",
  phi_g:         "Φ_g  (grav. scalar)",
  E_g_magnitude: "|E_g|  (gravito-E)",
  B_g_magnitude: "|B_g|  (gravito-B)",
};

/** Short chip labels for the field selector bar. */
export const FIELD_CHIP: Partial<Record<FieldName, string>> = {
  phi:          "φ",
  A_magnitude:  "|A|",
  B_magnitude:  "|B|",
  C_field:      "C",
  poynting_mag: "|P|",
  phi_g:        "Φ_g",
  B_g_magnitude:"|B_g|",
};

export const FIELD_UNITS: Partial<Record<FieldName, string>> = {
  phi:           "V",
  A_magnitude:   "V·s/m",
  E_magnitude:   "V/m",
  B_magnitude:   "T",
  J_magnitude:   "A/m²",
  C_field:       "m⁻¹",
  poynting_mag:  "W/m²",
  energy_density:"J/m³",
  phi_g:         "m²/s²",
  E_g_magnitude: "m/s²",
  B_g_magnitude: "s⁻¹",
};

export const COIL_LABELS: Record<CoilType, string> = {
  solenoid:        "Solenoid",
  toroid:          "Toroid (azimuthal)",
  toroid_poloidal: "Toroid (poloidal)",
  flat_spiral:     "Flat spiral",
  rodin:           "Rodin coil",
};

/** The EM fields available in Phase 1 (Biot-Savart, B and C computed). */
export const PHASE1_FIELDS: FieldName[] = ["B_magnitude", "A_magnitude", "C_field"];

/** All EM fields (add as phases complete). */
export const EM_FIELDS: FieldName[] = [
  "phi", "A_magnitude", "E_magnitude", "B_magnitude", "J_magnitude",
  "C_field", "poynting_mag", "energy_density",
];

export const GEM_FIELDS: FieldName[] = ["phi_g", "E_g_magnitude", "B_g_magnitude"];

// ── Defaults ──────────────────────────────────────────────────────────────────

export function defaultCoilEntity(): CoilEntity {
  return {
    coil: {
      coil_type:     "solenoid",
      radius_m:      0.05,
      turns:         10,
      pitch_m:       0.005,
      wire_radius_m: 0.001,
      current_A:     1.0,
    },
    position_m:      [0, 0, 0],
    orientation:     [0, 0, 0, 1],
    superconducting: false,
  };
}

export function defaultSolveRequest(): SolveRequest {
  return {
    entities: [defaultCoilEntity()],
    eed: { alpha: 0.0, beta: 0.1, gamma: 1.0 },
    gem: { enabled: false, kappa_g: 0.0, li_torr_mode: false },
    solver: {
      mode:            { mode: "static" },
      cells_per_axis:  64,
      domain_radius_m: 0.2,
      lorenz_gauge:    false,
    },
    slices: [
      { axis: "z", position: 0.5, field: "B_magnitude", resolution: 128 },
      { axis: "x", position: 0.5, field: "B_magnitude", resolution: 128 },
    ],
    request_volume: false,
    volume_field:   "B_magnitude",
    holonomy_paths: [],
  };
}
