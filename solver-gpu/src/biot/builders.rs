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

/// Reference impedance used to convert open-helix drive voltage → feed current.
///
/// An open helix is antenna-like: you drive it with a voltage V₀ at the feed
/// terminals; the peak feed current is I₀ = V₀ / Z_REF.  50 Ω is the standard
/// RF reference impedance and gives a reasonable first-order conversion without
/// needing a full antenna impedance calculation.
pub const OPEN_HELIX_Z_REF: f64 = 50.0;

/// Convert one `CoilEntity` into a list of GPU-ready wire segments.
pub fn entity_to_segments(entity: &CoilEntity) -> Vec<WireSegment> {
    let raw = build_path(entity);

    // Open circuits (open helix, or any winding with open_circuit set) are
    // antenna-like: voltage-driven, with peak feed current I₀ = V₀ / Z_ref.  A
    // closed loop instead carries the directly-specified loop current.
    let current = if entity.coil.coil_type == CoilType::OpenHelix || entity.coil.open_circuit {
        (entity.coil.voltage_v / OPEN_HELIX_Z_REF) as f32
    } else {
        entity.coil.current_a as f32
    };

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
    let path = match c.coil_type {
        CoilType::Solenoid            => solenoid(c.radius_m, c.turns, c.pitch_m),
        CoilType::Toroid              => toroid(c.radius_m, c.turns, c.pitch_m),
        CoilType::ToroidPoloidal      => toroid_poloidal(c.radius_m, c.turns, c.pitch_m),
        CoilType::FlatSpiral          => flat_spiral(c.radius_m, c.turns, c.pitch_m),
        CoilType::Rodin               => rodin(c.radius_m, c.turns, c.pitch_m),
        CoilType::OpenHelix           => open_helix(c.radius_m, c.turns, c.pitch_m),
        // Capacitor types carry no current → Biot-Savart is zero.
        // Their φ field is initialised separately by `initialize_phi_capacitor`.
        CoilType::CapacitorSymmetric  => vec![],
        CoilType::CapacitorAsymmetric => vec![],
        // Pure mass source: no current, no Biot-Savart path.  Its Φ_g/A_g come
        // from the GEM mass-source kernel (ORC-0tl), not from any winding.
        CoilType::MassSphere          => vec![],
    };

    // Open-circuit option (ORC-09r): break a *closed* winding so the wire has
    // free tips.  Under AC drive the oscillating charge at the tips gives
    // ∂µJµ ≠ 0 → sources the EED scalar C (the only way a toroidal topology
    // produces Φ_g — closed loops have ∇·J ≈ 0).  Already-open types
    // (solenoid/open_helix/flat_spiral) have tips by construction, so leave them.
    if c.open_circuit
        && matches!(c.coil_type,
            CoilType::Toroid | CoilType::ToroidPoloidal | CoilType::Rodin)
        && path.len() > 4
    {
        let gap  = c.open_gap_fraction.clamp(0.02, 0.6);
        let keep = ((path.len() as f64 * (1.0 - gap)) as usize).max(2);
        path[..keep].to_vec()
    } else {
        path
    }
}

/// Return the two physical lead attachment points for an entity [m].
/// For current sources: [wire_start, wire_end].
/// For capacitors: [anode_center, cathode_center].
pub fn entity_lead_points(entity: &CoilEntity) -> [[f64; 3]; 2] {
    let raw = build_path(entity);
    let c   = &entity.coil;

    let gap = if c.plate_gap_m > 0.0 { c.plate_gap_m } else { 2.0 * c.pitch_m };
    match c.coil_type {
        CoilType::CapacitorSymmetric | CoilType::CapacitorAsymmetric => {
            // Anode (top, +V/2) and cathode (bottom, −V/2).
            let anode   = transform([0.0, 0.0,  gap * 0.5], entity.position_m, entity.orientation);
            let cathode = transform([0.0, 0.0, -gap * 0.5], entity.position_m, entity.orientation);
            [anode.map(|v| v as f64), cathode.map(|v| v as f64)]
        }
        _ => {
            // Wire source: first and last point on the path.
            if raw.len() < 2 {
                [entity.position_m, entity.position_m]
            } else {
                let p0 = transform(raw[0],           entity.position_m, entity.orientation);
                let p1 = transform(*raw.last().unwrap(), entity.position_m, entity.orientation);
                [p0.map(|v| v as f64), p1.map(|v| v as f64)]
            }
        }
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
// Toroid–Poloidal — poloidal-field winding (transpose of the azimuthal toroid)
// ─────────────────────────────────────────────────────────────────────────────
//
// Poloidal-field toroid — the genuine *transpose* of the azimuthal toroid.
// The wire winds TOROIDALLY (the long way, N trips around the major circle, like
// a solenoid bent into a ring) and advances just ONE slow poloidal turn over the
// whole coil.  Current is predominantly toroidal ⇒ B is POLOIDAL (threads the
// hole), the physical contrast to the azimuthal toroid's confined toroidal B.
// Torus winding T(N, 1) vs the azimuthal T(1, N).
fn toroid_poloidal(radius: f64, turns: u32, pitch: f64) -> Vec<[f64; 3]> {
    let r   = pitch.max(radius * 0.05);
    let n   = turns as f64;
    let pts = (turns * 180).max(36) as usize;
    (0..=pts)
        .map(|i| {
            let t   = i as f64 / pts as f64;
            let phi = TAU * n * t;   // N toroidal trips (the long way around)
            let th  = TAU * t;       // ONE slow poloidal advance over the coil
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
// Rodin / Marko coil — star (node-skip) winding on a torus
// ─────────────────────────────────────────────────────────────────────────────
//
// Place M nodes evenly around the torus and connect every K-th node — the star
// polygon {M/K}.  Between nodes the wire dips through a full poloidal loop, so
// (with a LARGE minor radius) the chords sweep near the hub and cross over/under
// each other: the figure-8 woven pattern.  Continuously this is the torus
// winding T(K, M): K toroidal trips, M poloidal dips.
//
//   φ(t) = 2π K t                 (K toroidal trips to close the {M/K} star)
//   θ(t) = 2π M t                 (M poloidal dips — one per chord)
//   ρ    = R + r cosθ,  r ≈ 0.8 R  (large ⇒ inner dips reach the hub)
//
// M is forced odd so the skip-2 star is a single continuous wire (gcd(M,2)=1).
fn rodin(radius: f64, turns: u32, _pitch: f64) -> Vec<[f64; 3]> {
    let m_raw = turns.max(5);
    let m     = if m_raw % 2 == 0 { m_raw + 1 } else { m_raw }; // odd ⇒ single wire
    let k     = 2u32;                          // skip every other node → {M/2} star
    let r     = radius * 0.8;                  // large minor radius ⇒ visible star crossings
    let pts   = (m * 240).max(240) as usize;   // fine sampling for the crossings
    (0..=pts)
        .map(|i| {
            let t   = i as f64 / pts as f64;
            let phi = TAU * k as f64 * t;       // K toroidal trips
            let th  = TAU * m as f64 * t;       // M poloidal dips
            let rho = radius + r * th.cos();
            [rho * phi.cos(), rho * phi.sin(), r * th.sin()]
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Open helix — like a solenoid but geometrically open (non-closed)
// ─────────────────────────────────────────────────────────────────────────────
//
// Identical wire path to the solenoid but with a fewer turns (fewer wraps)
// by default to keep the tips clearly separated.  The key semantic difference
// is that this coil does NOT close back to its origin — the first and last
// points are the physical wire tips where charge accumulates when AC-driven.
//
// When frequency_hz > 0 in time-domain FDTD, the oscillating current creates
// ∇·J ≠ 0 at the tips, producing the EED electroscalar (C) mode.

fn open_helix(radius: f64, turns: u32, pitch: f64) -> Vec<[f64; 3]> {
    // Same path as solenoid — the "open" property comes from not closing the loop.
    solenoid(radius, turns, pitch)
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

#[cfg(test)]
mod tests {
    use super::{toroid, toroid_poloidal, rodin};
    use std::f64::consts::PI;

    /// Total toroidal winding number: unwrapped Σ|Δφ| / 2π of the path's
    /// azimuthal angle φ = atan2(y, x).
    fn toroidal_turns(path: &[[f64; 3]]) -> f64 {
        let mut total = 0.0;
        for w in path.windows(2) {
            let a0 = w[0][1].atan2(w[0][0]);
            let a1 = w[1][1].atan2(w[1][0]);
            let mut d = a1 - a0;
            while d >  PI { d -= 2.0 * PI; }
            while d < -PI { d += 2.0 * PI; }
            total += d.abs();
        }
        total / (2.0 * PI)
    }

    /// Smallest cylindrical radius √(x²+y²) the wire reaches.
    fn min_rho(path: &[[f64; 3]]) -> f64 {
        path.iter().map(|p| (p[0]*p[0] + p[1]*p[1]).sqrt()).fold(f64::MAX, f64::min)
    }

    #[test]
    fn toroid_poloidal_rodin_are_distinct_topologies() {
        let (r, n, pitch) = (0.05_f64, 12_u32, 0.006_f64);
        let az  = toroid(r, n, pitch);            // T(1,N) — azimuthal
        let pol = toroid_poloidal(r, n, pitch);   // T(N,1) — poloidal transpose
        let rod = rodin(r, n, pitch);             // T(2,M) — star

        let (t_az, t_pol, t_rod) = (toroidal_turns(&az), toroidal_turns(&pol), toroidal_turns(&rod));
        println!("toroidal turns — azimuthal {t_az:.2}, poloidal {t_pol:.2}, rodin {t_rod:.2}");
        println!("min ρ/R — azimuthal {:.2}, poloidal {:.2}, rodin {:.2}",
            min_rho(&az)/r, min_rho(&pol)/r, min_rho(&rod)/r);

        // Azimuthal makes ~1 toroidal trip; poloidal makes ~N (it's the transpose).
        assert!(t_az < 2.0, "azimuthal toroid should make ~1 toroidal trip, got {t_az:.2}");
        assert!(t_pol > 6.0, "poloidal toroid should make ~N toroidal trips, got {t_pol:.2}");
        assert!(t_pol > 3.0 * t_az,
            "azimuthal and poloidal toroids are not distinct (turns {t_az:.2} vs {t_pol:.2})");

        // Rodin's large minor radius makes the wire dip near the hub; the toroids
        // (thin tube) never approach the centre.
        assert!(min_rho(&rod) < 0.5 * r,
            "rodin should dip near the hub (min ρ {:.3} ≥ 0.5R)", min_rho(&rod));
        assert!(min_rho(&az)  > 0.7 * r && min_rho(&pol) > 0.7 * r,
            "toroids should stay near the major radius (az {:.3}, pol {:.3})",
            min_rho(&az), min_rho(&pol));
    }
}
