//! Coil geometry builders: CoilEntity → Vec<WireSegment>.
//!
//! Each builder traces the 3-D wire path of one coil type and samples it into
//! straight segments.  The segment list is then uploaded to the GPU for
//! Biot-Savart evaluation.
//!
//! # Coordinate convention
//! All coils are built centred at the origin, then transformed by the entity's
//! `position_m` (translation) and `orientation` (unit quaternion rotation).
//!
//! # Sampling density
//! Points per turn is chosen so that the angular step is ≤ 2° (180 pts/turn).
//! WireSegment::new() further subdivides each segment if it exceeds 1 mm,
//! so the Biot-Savart integration is always accurate regardless of coil size.

use std::f64::consts::TAU;

use crate::types::{CoilEntity, CoilType};
use super::WireSegment;

/// Convert one `CoilEntity` into a list of GPU-ready wire segments.
pub fn entity_to_segments(entity: &CoilEntity) -> Vec<WireSegment> {
    let raw = build_path(entity);
    let current = entity.coil.current_a as f32;

    // Apply rigid transform: rotate by quaternion, then translate.
    let pts: Vec<[f32; 3]> = raw.iter()
        .map(|&p| transform(p, entity.position_m, entity.orientation))
        .collect();

    // Convert consecutive point pairs to segments.
    pts.windows(2)
        .map(|w| WireSegment::new(w[0], w[1], current))
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-coil-type path builders (unrotated, centred at origin)
// ─────────────────────────────────────────────────────────────────────────────

fn build_path(e: &CoilEntity) -> Vec<[f64; 3]> {
    let c = &e.coil;
    match c.coil_type {
        CoilType::Solenoid       => solenoid(c.radius_m, c.turns, c.pitch_m),
        CoilType::Toroid         => toroid(c.radius_m, c.turns, c.pitch_m),
        CoilType::ToroidPoloidal => toroid_poloidal(c.radius_m, c.turns, c.pitch_m),
        CoilType::FlatSpiral     => flat_spiral(c.radius_m, c.turns, c.pitch_m),
        CoilType::Rodin          => rodin(c.radius_m, c.turns, c.pitch_m),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Solenoid — helical coil along Z
// ─────────────────────────────────────────────────────────────────────────────
//
//   x(t) = R cos(N·2π·t)
//   y(t) = R sin(N·2π·t)
//   z(t) = pitch·N·(t − 0.5)          centred at origin
//   t ∈ [0, 1]

fn solenoid(radius: f64, turns: u32, pitch: f64) -> Vec<[f64; 3]> {
    let n    = turns as f64;
    let pts  = (turns * 180).max(20) as usize;  // 180 pts/turn
    (0..=pts)
        .map(|i| {
            let t   = i as f64 / pts as f64;
            let phi = TAU * n * t;
            [radius * phi.cos(), radius * phi.sin(), pitch * n * (t - 0.5)]
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Toroid — single-layer winding on a torus
// ─────────────────────────────────────────────────────────────────────────────
//
//   Major radius R = `radius_m`  (axis ‖ Z)
//   Minor radius r = `pitch_m`   (tube cross-section)
//   N = `turns`  poloidal winds per one toroidal trip
//
//   φ(t) = 2π t                   (toroidal angle, 0→2π)
//   θ(t) = 2π N t                 (poloidal angle, N full winds)
//   x(t) = (R + r cosθ) cosφ
//   y(t) = (R + r cosθ) sinφ
//   z(t) = r sinθ

fn toroid(radius: f64, turns: u32, pitch: f64) -> Vec<[f64; 3]> {
    let r    = pitch.max(radius * 0.05); // minor radius
    let n    = turns as f64;
    let pts  = (turns * 180).max(36) as usize;
    (0..=pts)
        .map(|i| {
            let t   = i as f64 / pts as f64;
            let phi = TAU * t;
            let th  = TAU * n * t;
            let rho = radius + r * th.cos();
            [rho * phi.cos(), rho * phi.sin(), r * th.sin()]
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Toroid–Poloidal — like toroid but M toroidal trips for N poloidal winds
// (N/M turns per trip, M trips total)
// ─────────────────────────────────────────────────────────────────────────────
//
// Uses M = 2 toroidal trips so the winding crosses the torus twice.
// Practical approximation to the full bifilar toroid winding.

fn toroid_poloidal(radius: f64, turns: u32, pitch: f64) -> Vec<[f64; 3]> {
    let r     = pitch.max(radius * 0.05);
    let trips = 2u32;                         // toroidal trips
    let n_pol = turns as f64 / trips as f64;  // poloidal winds per trip
    let pts   = (turns * 180).max(36) as usize;
    (0..=pts)
        .map(|i| {
            let t   = i as f64 / pts as f64;
            let phi = TAU * trips as f64 * t;
            let th  = TAU * n_pol * t;
            let rho = radius + r * th.cos();
            [rho * phi.cos(), rho * phi.sin(), r * th.sin()]
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Flat spiral — Archimedes spiral in the XY plane
// ─────────────────────────────────────────────────────────────────────────────
//
//   r(t)  = (pitch·t·N + inner_r)
//   φ(t)  = 2π N t
//   x     = r cosφ,  y = r sinφ,  z = 0
//
// The inner radius is `pitch` (first turn gap equals the pitch).

fn flat_spiral(radius: f64, turns: u32, pitch: f64) -> Vec<[f64; 3]> {
    let n     = turns as f64;
    let r_in  = pitch;
    let r_out = radius;
    let dr    = (r_out - r_in).max(0.0);
    let pts   = (turns * 180).max(20) as usize;
    (0..=pts)
        .map(|i| {
            let t   = i as f64 / pts as f64;
            let r   = r_in + dr * t;
            let phi = TAU * n * t;
            [r * phi.cos(), r * phi.sin(), 0.0]
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Rodin coil — figure-8 winding on a torus
// ─────────────────────────────────────────────────────────────────────────────
//
// The Rodin coil winds with poloidal angle advancing 2× faster than toroidal,
// crossing the torus surface in a figure-8 pattern.  This creates balanced
// north/south winding that is claimed to produce anomalous field-line geometry.
//
//   φ(t) = 2π N t                 (toroidal)
//   θ(t) = 2 × 2π N t            (poloidal, 2× faster)
//   x(t) = (R + r cosθ) cosφ
//   y(t) = (R + r cosθ) sinφ
//   z(t) = r sinθ

fn rodin(radius: f64, turns: u32, pitch: f64) -> Vec<[f64; 3]> {
    let r    = pitch.max(radius * 0.1);
    let n    = turns as f64;
    let pts  = (turns * 360).max(72) as usize; // finer for the figure-8
    (0..=pts)
        .map(|i| {
            let t   = i as f64 / pts as f64;
            let phi = TAU * n * t;
            let th  = 2.0 * TAU * n * t;
            let rho = radius + r * th.cos();
            [rho * phi.cos(), rho * phi.sin(), r * th.sin()]
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Rigid transform helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Rotate `p` by unit quaternion `q = [x, y, z, w]`, then add `t`.
fn transform(p: [f64; 3], t: [f64; 3], q: [f64; 4]) -> [f32; 3] {
    let r = quat_rotate(p, q);
    [(r[0] + t[0]) as f32, (r[1] + t[1]) as f32, (r[2] + t[2]) as f32]
}

/// Sandwich product: q p q* (pure quaternion rotation).
fn quat_rotate(p: [f64; 3], q: [f64; 4]) -> [f64; 3] {
    let [qx, qy, qz, qw] = q;
    let [px, py, pz] = p;

    // q × p_quat
    let ix =  qw*px + qy*pz - qz*py;
    let iy =  qw*py + qz*px - qx*pz;
    let iz =  qw*pz + qx*py - qy*px;
    let iw = -qx*px - qy*py - qz*pz;

    // (q × p_quat) × q*
    [
        ix*qw + iw*(-qx) + iy*(-qz) - iz*(-qy),
        iy*qw + iw*(-qy) + iz*(-qx) - ix*(-qz),
        iz*qw + iw*(-qz) + ix*(-qy) - iy*(-qx),
    ]
}
