//! Biot-Savart sources: wire segment geometry and GPU data layout.
//!
//! # Data flow
//! `CoilEntity` (from the frontend) → `entity_to_segments()` → `Vec<WireSegment>`
//! → uploaded to GPU storage buffer → `biot.wgsl` kernel evaluates A(r) at every
//! Yee-grid vertex.
//!
//! # Wire segment integration
//! The kernel numerically integrates A(r) = (μ₀/4π) ∫ I dl / |r − r′| by
//! subdividing each segment into `ndiv` equal pieces and summing midpoint
//! contributions.  `ndiv` is chosen so each sub-element is ≤ 1 mm long,
//! giving < 0.1% relative error at distances > 1 cm from the wire.

use bytemuck::{Pod, Zeroable};

mod builders;
pub use builders::{entity_to_segments, entity_lead_points, OPEN_HELIX_Z_REF};

// ─────────────────────────────────────────────────────────────────────────────
// WireSegment — GPU-ready (Pod, 32 bytes, matches WGSL struct layout)
// ─────────────────────────────────────────────────────────────────────────────
//
// WGSL std430 layout (vec3<f32> align=16, size=12):
//   start:   vec3<f32>  @  0  (bytes 0-11)
//   current: f32        @ 12  (bytes 12-15)
//   end:     vec3<f32>  @ 16  (bytes 16-27)
//   ndiv:    u32        @ 28  (bytes 28-31)
//   struct size: 32 bytes
//
// The Rust repr(C) layout is identical — fields in the same order.

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct WireSegment {
    /// Segment start point [m].
    pub start:   [f32; 3],
    /// Current in the segment [A] (signed; direction = start → end).
    pub current: f32,
    /// Segment end point [m].
    pub end:     [f32; 3],
    /// Sub-divisions for numerical integration (auto-set from length).
    pub ndiv:    u32,
}

/// Target sub-element length for integration accuracy [m].
/// 1 mm gives < 0.1% error at distances > 10 mm from the wire.
const TARGET_ELEM_M: f32 = 1e-3;

impl WireSegment {
    pub fn new(start: [f32; 3], end: [f32; 3], current_a: f32) -> Self {
        let d   = [end[0]-start[0], end[1]-start[1], end[2]-start[2]];
        let len = (d[0]*d[0] + d[1]*d[1] + d[2]*d[2]).sqrt();
        // Clamp: 4 ≤ ndiv ≤ 512 — keeps the inner GPU loop bounded.
        let ndiv = ((len / TARGET_ELEM_M).ceil() as u32).clamp(4, 512);
        Self { start, current: current_a, end, ndiv }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// J-source grid — voxelise wire segments into a current-density volume
// ─────────────────────────────────────────────────────────────────────────────

/// Voxelise wire segments into a 3-D current-density grid for AC injection.
///
/// Returns a `Vec<f32>` of length n1³ × 4 (stride 4: [Jx, Jy, Jz, 0]).
/// The current density J is normalised so that `I₀ = 1 A` throughout.
///
/// Each segment's current is distributed to grid cells via nearest-cell
/// deposit along the segment at sub-cell resolution (dx/4 steps).
/// J has units A/m² (current density for I₀ = 1 A).
///
/// Multiply by `I₀ · sin(ωt)` at each FDTD step to obtain the time-varying
/// source J(t) for injection into the A-velocity update.
///
/// # Arguments
/// * `segments` — normalised wire segments (current is used for direction only)
/// * `n1`       — vertices per axis (grid is n1³)
/// * `dx`       — cell size [m]
/// * `origin`   — lower corner of the grid [m]
pub fn segments_to_j_grid(
    segments: &[WireSegment],
    n1:       usize,
    dx:       f32,
    origin:   [f32; 3],
) -> Vec<f32> {
    let total = n1 * n1 * n1;
    let mut j_grid = vec![0.0f32; total * 4];

    // Sub-step resolution: 4 sub-steps per cell = dx/4.
    let sub_step = dx * 0.25;
    // Volume of one grid cell [m³] — normalises deposited current to density.
    let cell_vol = dx * dx * dx;

    for seg in segments {
        let dx_seg = [seg.end[0] - seg.start[0],
                      seg.end[1] - seg.start[1],
                      seg.end[2] - seg.start[2]];
        let seg_len = (dx_seg[0]*dx_seg[0] + dx_seg[1]*dx_seg[1] + dx_seg[2]*dx_seg[2]).sqrt();
        if seg_len < 1e-12 { continue; }

        // Unit direction vector (current sign irrelevant for AC — only geometry matters).
        let dir = [dx_seg[0] / seg_len, dx_seg[1] / seg_len, dx_seg[2] / seg_len];

        // Walk along segment at sub_step resolution.
        let n_sub = ((seg_len / sub_step).ceil() as usize).max(1);
        for k in 0..n_sub {
            let t = (k as f32 + 0.5) / n_sub as f32;
            let r = [
                seg.start[0] + t * dx_seg[0],
                seg.start[1] + t * dx_seg[1],
                seg.start[2] + t * dx_seg[2],
            ];

            // Grid coordinates (vertex-aligned).
            let gi_f = [(r[0] - origin[0]) / dx,
                        (r[1] - origin[1]) / dx,
                        (r[2] - origin[2]) / dx];

            let ix = gi_f[0].round() as i64;
            let iy = gi_f[1].round() as i64;
            let iz = gi_f[2].round() as i64;

            if ix < 0 || ix >= n1 as i64
            || iy < 0 || iy >= n1 as i64
            || iz < 0 || iz >= n1 as i64 { continue; }

            let flat = (ix as usize) + (iy as usize) * n1 + (iz as usize) * n1 * n1;

            // Deposit: J_k = I₀ · dl_k / V_cell
            // dl_k for this sub-element = dir_k × (seg_len / n_sub)
            let dl = seg_len / n_sub as f32;
            j_grid[flat * 4]     += dir[0] * dl / cell_vol;
            j_grid[flat * 4 + 1] += dir[1] * dl / cell_vol;
            j_grid[flat * 4 + 2] += dir[2] * dl / cell_vol;
            // j_grid[flat * 4 + 3] = 0 (padding)
        }
    }

    j_grid
}
