// TypeScript mirror of solver/models/params.py and src-tauri/src/types.rs.
// Sync rule: any change here must be reflected in both of those files.

export type CoilType =
  | "solenoid"
  | "toroid"
  | "toroid_poloidal"
  | "flat_spiral"
  | "rodin";

export type FormulationType = "scalar_only" | "maxwell_only" | "eed_coupled";
export type MeshResolution  = "coarse" | "medium" | "fine";
export type SliceAxis        = "x" | "y" | "z";
export type FieldName        = "phi" | "A_magnitude" | "B_magnitude" | "J_magnitude";
export type SolverState      = "starting" | "ready" | "solving" | "error";

export const FIELD_LABELS: Record<FieldName, string> = {
  phi:         "φ (EED scalar)",
  A_magnitude: "|A| (vector pot.)",
  B_magnitude: "|B| (flux density)",
  J_magnitude: "|J| (current density)",
};

export const COIL_LABELS: Record<CoilType, string> = {
  solenoid:        "Solenoid",
  toroid:          "Toroid (azimuthal)",
  toroid_poloidal: "Toroid (poloidal)",
  flat_spiral:     "Flat spiral",
  rodin:           "Rodin coil",
};

// Primary field for each formulation (drives default 3D view)
export const PRIMARY_FIELD: Record<FormulationType, FieldName> = {
  scalar_only:  "phi",
  maxwell_only: "B_magnitude",
  eed_coupled:  "phi",
};

export interface CoilParams {
  radius_m:      number;
  turns:         number;
  pitch_m:       number;
  wire_radius_m: number;
  current_A:     number;
  coil_type:     CoilType;
}

export interface EedParams {
  alpha: number;
  beta:  number;
  gamma: number;
}

export interface SliceRequest {
  axis:       SliceAxis;
  position:   number;   // 0–1 normalized
  field:      FieldName;
  resolution: number;
}

export interface SolveRequest {
  coil:            CoilParams;
  eed:             EedParams;
  domain_radius_m: number;
  mesh_resolution: MeshResolution;
  formulation:     FormulationType;
  slices:          SliceRequest[];
  request_volume:  boolean;
}

export interface SliceData {
  axis:       SliceAxis;
  position:   number;
  field:      FieldName;
  shape:      [number, number];
  data:       number[];
  x_range:    [number, number];
  y_range:    [number, number];
  field_min:  number;
  field_max:  number;
}

export interface VolumeData {
  field:     FieldName;
  shape:     [number, number, number];  // [nx, ny, nz]
  data:      number[];                  // Normalized [0,1], row-major
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

export interface SolveResult {
  solve_time_s: number;
  mesh_nodes:   number;
  slices:       SliceData[];
  volume:       VolumeData | null;
  maxima:       FieldMaximum[];
  warnings:     string[];
}

export interface SolverStatus {
  state:   SolverState;
  message: string;
}

export interface HypothesisEntry {
  id:        string;
  name:      string;
  timestamp: string;
  request:   SolveRequest;
  maxima:    FieldMaximum[];
  notes?:    string;
}

// Default solve request — used on app start
export function defaultSolveRequest(formulation: FormulationType = "scalar_only"): SolveRequest {
  const field = PRIMARY_FIELD[formulation];
  return {
    coil: {
      radius_m: 0.05, turns: 10, pitch_m: 0.005,
      wire_radius_m: 0.001, current_A: 1.0, coil_type: "solenoid",
    },
    eed: { alpha: 0.0, beta: 0.1, gamma: 0.1 },
    domain_radius_m: 0.2,
    mesh_resolution: "coarse",
    formulation,
    slices: [
      { axis: "z", position: 0.5, field, resolution: 128 },
      { axis: "x", position: 0.5, field, resolution: 128 },
      { axis: "y", position: 0.5, field, resolution: 128 },
    ],
    request_volume: true,
  };
}
