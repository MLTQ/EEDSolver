//! Integration test: how much of the open-helix C signal survives when the
//! circuit is completed with return leads? (ORC-mlo)
//!
//! The EED literature sources C only through 4-current non-conservation
//! (□C = µ₀∂µJµ in the decoupled reading).  Oracle injects J with no
//! compensating ρ, so an *open* helix has ∂µJµ = ∇·J ≠ 0 at its tips — but a
//! real energised coil has feed leads completing the circuit, giving
//! ∇·J = 0 along the whole loop.  This test drives the SAME helix both ways:
//!
//!   open:   bare helix, free tips              (∇·J ≠ 0 at the tips)
//!   closed: helix + radial/axial return leads   (∇·J ≈ 0 everywhere)
//!
//! and compares (a) the deposited grid-level |∇·J| and (b) the resulting
//! max |C| after identical AC drive at γ=1.
//!
//! This is a CHARACTERIZATION test: whatever C the closed circuit still
//! produces comes from the bespoke coupled dynamics (which generates
//! longitudinal ∇·A from any oscillating A, tips or no tips), not from the
//! ∂µJµ channel.  The measured open/closed ratio quantifies how much of the
//! "end-cap/tip" lab signature is an artifact of truncating the circuit.
//! Result is logged in FIELD_THEORY.md (decision log 2026-07-02).

use solver_gpu::{
    biot::{segments_to_j_grid, WireSegment},
    context::GpuContext,
    grid::{GpuGridState, YeeGrid},
};

const CELLS:    u32 = 48;
const DOMAIN_R: f64 = 0.10;

const HELIX_R:  f64 = 0.03;   // helix radius [m]
const TURNS:    u32 = 6;
const PITCH:    f64 = 0.01;   // total height 0.06 m
const RETURN_R: f64 = 0.06;   // return-lead radius [m] — inside the sponge layer

/// Helix path, identical to builders::solenoid (which is private).
fn helix_path() -> Vec<[f64; 3]> {
    use std::f64::consts::TAU;
    let n   = TURNS as f64;
    let pts = (TURNS * 180) as usize;
    (0..=pts)
        .map(|i| {
            let t   = i as f64 / pts as f64;
            let phi = TAU * n * t;
            [HELIX_R * phi.cos(), HELIX_R * phi.sin(), PITCH * n * (t - 0.5)]
        })
        .collect()
}

/// Points → consecutive-pair segments with unit current (J is normalised;
/// the drive amplitude is applied by run_fdtd_ac).
fn to_segments(path: &[[f64; 3]]) -> Vec<WireSegment> {
    path.windows(2)
        .map(|w| WireSegment::new(
            [w[0][0] as f32, w[0][1] as f32, w[0][2] as f32],
            [w[1][0] as f32, w[1][1] as f32, w[1][2] as f32],
            1.0,
        ))
        .collect()
}

/// Closed variant: helix plus a return circuit — radially out from the top
/// tip, down along z at RETURN_R, radially back in to the bottom tip.
fn closed_path() -> Vec<[f64; 3]> {
    let mut path = helix_path();
    let first = path[0];
    let last  = *path.last().unwrap();
    // Both tips sit at azimuth 0 (integer turns): (HELIX_R, 0, ±h/2).
    path.push([RETURN_R, last[1], last[2]]);   // radial out at top
    path.push([RETURN_R, first[1], first[2]]); // axial down at RETURN_R
    path.push(first);                          // radial in — loop closed
    path
}

/// Monopole strength [A] of the deposited J around a point: ∫∇·J dV over a
/// ball of radius `r_cells` centred on `p`.  Pointwise |∇·J| is useless here —
/// nearest-vertex rasterisation makes it huge everywhere along the wire — but
/// the noise integrates to ≈ 0, while a genuine open tip integrates to ±I₀
/// (the terminating current).  A closed circuit has no monopole anywhere.
fn tip_monopole_a(j: &[f32], n1: usize, dx: f64, origin: f64, p: [f64; 3], r_cells: f64) -> f64 {
    let inv2dx = 0.5 / dx;
    let vol    = dx * dx * dx;
    let idx = |x: usize, y: usize, z: usize| (x + y * n1 + z * n1 * n1) * 4;
    let pc: Vec<f64> = p.iter().map(|v| (v - origin) / dx).collect();
    let mut net = 0.0f64;
    for iz in 1..n1 - 1 {
        for iy in 1..n1 - 1 {
            for ix in 1..n1 - 1 {
                let r = ((ix as f64 - pc[0]).powi(2)
                       + (iy as f64 - pc[1]).powi(2)
                       + (iz as f64 - pc[2]).powi(2)).sqrt();
                if r > r_cells { continue; }
                let div = (j[idx(ix + 1, iy, iz)]     as f64 - j[idx(ix - 1, iy, iz)]     as f64) * inv2dx
                        + (j[idx(ix, iy + 1, iz) + 1] as f64 - j[idx(ix, iy - 1, iz) + 1] as f64) * inv2dx
                        + (j[idx(ix, iy, iz + 1) + 2] as f64 - j[idx(ix, iy, iz - 1) + 2] as f64) * inv2dx;
                net += div * vol;
            }
        }
    }
    net
}

/// Max |C| over the interior, excluding the sponge layer plus a 2-cell margin.
fn max_c_interior(c: &[f32], n1: usize, sponge_cells: usize) -> f64 {
    let m = sponge_cells + 2;
    let mut cmax = 0.0f64;
    for iz in m..n1 - m {
        for iy in m..n1 - m {
            for ix in m..n1 - m {
                let v = c[ix + iy * n1 + iz * n1 * n1].abs() as f64;
                if v > cmax { cmax = v; }
            }
        }
    }
    cmax
}

/// Drive the given wire path with 1 GHz AC at γ=1; return (|tip monopole| [A], max|C|).
async fn drive(path: &[[f64; 3]]) -> (f64, f64) {
    let grid = YeeGrid::new(CELLS, DOMAIN_R);
    let n1   = (grid.n + 1) as usize;
    let dt   = (grid.cfl_dt() * 0.5) as f32;

    let ctx    = GpuContext::new().await.expect("GPU init");
    let gstate = GpuGridState::new(&ctx, &grid);

    let segs   = to_segments(path);
    let origin = [-grid.extent as f32; 3];
    let j_grid = segments_to_j_grid(&segs, n1, grid.dx as f32, origin);
    // Monopole strength at the helix top tip (R, 0, +h/2): ±1 A if the wire
    // terminates there, ≈ 0 if the return lead carries the current onward.
    let tip = *helix_path().last().unwrap();
    let div_j = tip_monopole_a(&j_grid, n1, grid.dx, -grid.extent, tip, 4.0).abs();
    gstate.upload_j_source(&ctx, &j_grid);

    // ~1.5 periods of the 1 GHz drive; default absorbing sponge.
    let n_steps = 400_u32;
    gstate
        .run_fdtd_ac(&ctx, &grid, dt, n_steps, 1.0, None, 1.0, 1.0e9, 0.0, None)
        .unwrap();

    let c = gstate.readback(&ctx, &gstate.c_fld, gstate.scalar_len()).unwrap();
    let sponge_cells = ((n1 as u32) / 8).max(4) as usize;
    (div_j, max_c_interior(&c, n1, sponge_cells))
}

#[tokio::test]
async fn test_closing_the_circuit_reduces_but_does_not_kill_c() {
    println!("Open helix vs closed circuit (leads), 48³, γ=1, 1 GHz, 400 steps:");

    let (div_open,   c_open)   = drive(&helix_path()).await;
    let (div_closed, c_closed) = drive(&closed_path()).await;

    let div_ratio = div_open / div_closed.max(1e-30);
    let c_ratio   = c_open / c_closed.max(1e-30);

    println!("\n  tip ∫∇·J dV  open = {div_open:.3e} A   closed = {div_closed:.3e} A   (open/closed = {div_ratio:.2})");
    println!("  max|C|       open = {c_open:.3e}   closed = {c_closed:.3e}   (open/closed = {c_ratio:.2})");

    assert!(c_open.is_finite() && c_closed.is_finite(), "C blew up (NaN/Inf)");
    assert!(c_open > 0.0 && c_closed > 0.0, "no C produced at all — drive broken?");

    // Sanity on the source model: the open tip is a genuine current monopole
    // (∫∇·J dV ≈ ±I₀ = ±1 A); with the return lead the current flows onward
    // and the ball integral collapses to rasterisation residue.  If this
    // fails, the leads didn't actually close the circuit.
    assert!(div_open > 0.5 && div_open < 2.0,
        "open tip monopole {div_open:.3e} A is not ≈ I₀ = 1 A — tip not where expected?");
    assert!(div_closed < 0.3 * div_open,
        "closing the circuit did not remove the tip monopole (open {div_open:.3e} A vs closed {div_closed:.3e} A)");

    // Characterization (not a physics theorem): the tip channel should make the
    // open helix C larger, mirroring the ORC-09r open-toroid result (~3×).
    assert!(c_open > c_closed,
        "open helix C ({c_open:.3e}) not above closed circuit ({c_closed:.3e})");

    println!(
        "\n  → {:.0}% of the open-helix C survives circuit closure — that fraction comes\n    \
         from the coupled dynamics, not the ∂µJµ tip channel; the rest is the\n    \
         truncated-circuit idealization.  Log in FIELD_THEORY.md.",
        100.0 * c_closed / c_open
    );
}
