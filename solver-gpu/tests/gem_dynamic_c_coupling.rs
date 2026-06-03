//! ORC-x7m: KK-direct must couple to the *dynamical* EED scalar C for a
//! sustained (AC) drive — not the static Coulomb-gauge snapshot.
//!
//! # The oversight this guards against
//!
//! The KK identification is algebraic: Φ_g = κ_G·C, A_g = κ_G·A.  Originally
//! the EED potentials were snapshotted *before* the FDTD loop (ORC-bn6) — the
//! right call for an unsustained DC source, whose un-fed static field the loop
//! radiates to ~0.  But that pre-loop snapshot is the **Coulomb-gauge**
//! Biot-Savart field (A from a current integral), so C = ∇·A there is the very
//! quantity the gauge sets to zero.  For a *driven* source that throws away the
//! whole point of EED: the deleted 7th DOF C is *dynamical*, sourced by ∂µJµ,
//! and only the FDTD evolution produces it.
//!
//! An **open helix** is the clean probe: its open tips have ∂µJµ ≠ 0, so the
//! FDTD develops a genuinely non-zero C.  The fix (ORC-x7m) snapshots the EED
//! potentials *after* the loop for sustained AC drives, so KK-direct couples to
//! that live C.
//!
//! # What this test asserts
//!
//! With κ_G = 1 the KK-direct map is the identity Φ_g = C pointwise, so on a
//! fixed slice the Φ_g peak must equal the **live post-FDTD** C peak.
//! `FieldName::CField` reads the evolved `c_fld`; `FieldName::PhiG` reads the
//! KK-direct output.  If the coupling still sourced from the pre-loop static
//! snapshot, Φ_g would track the static C (different distribution/magnitude)
//! and the ratio would diverge from 1.  We also assert the live C is genuinely
//! non-zero — i.e. the 7th DOF is alive, not floating-point dust.

use solver_gpu::{
    OracleSolver,
    types::{
        CoilEntity, CoilParams, CoilType, CouplingMode, EedParams, FieldName,
        GemParams, SliceAxis, SliceRequest, SolveRequest, SolverConfig, SolverMode,
    },
};

fn slice_req(field: FieldName) -> SliceRequest {
    SliceRequest { axis: SliceAxis::Z, position: 0.5, field, resolution: 64 }
}

fn global_max(result: &solver_gpu::types::SolveResult, f: FieldName) -> f64 {
    result.maxima.iter().find(|m| m.field == f).map(|m| m.max_value).unwrap_or(0.0)
}

#[tokio::test]
async fn kk_direct_couples_to_dynamical_c_for_ac_drive() {
    // Open helix, voltage-driven AC: non-closed wire ⇒ ∂µJµ ≠ 0 at the tips ⇒
    // the FDTD develops a real dynamical C.
    let entity = CoilEntity {
        coil: CoilParams {
            coil_type:     CoilType::OpenHelix,
            radius_m:      0.05,
            turns:         12,
            pitch_m:       0.006,
            wire_radius_m: 0.001,
            current_a:     0.0,        // open helix is voltage-driven
            voltage_v:     10_000.0,   // I₀ = V₀/Z_ref at the feed
            frequency_hz:  1.0e9,      // ← AC: sustains the field through FDTD
            ..Default::default()
        },
        position_m:    [0.0, 0.0, 0.0],
        orientation:   [0.0, 0.0, 0.0, 1.0],
        superconducting: false,
        angular_velocity_rad_s: [0.0; 3],
    };

    let request = SolveRequest {
        entities: vec![entity],
        eed:      EedParams { alpha: 0.0, beta: 0.0, gamma: 1.0 },
        gem:      GemParams {
            enabled:       true,
            kappa_g:       1.0,                  // identity map: Φ_g = C, A_g = A
            li_torr_mode:  false,
            coupling_mode: CouplingMode::KkDirect,
        },
        solver: SolverConfig {
            mode:            SolverMode::TimeDomain { dt_s: 0.0, n_steps: 150 },
            cells_per_axis:  32,
            domain_radius_m: 0.10,
            lorenz_gauge:    false,
        },
        slices: vec![
            slice_req(FieldName::BMagnitude),
            slice_req(FieldName::CField),  // live post-FDTD C
            slice_req(FieldName::PhiG),    // KK-direct output
        ],
        request_volume: false,
        volume_field:   FieldName::PhiG,
        holonomy_paths: vec![],
    };

    let solver = OracleSolver::new().await.expect("GPU init failed");
    let result = solver.solve(&request).await.expect("AC open-helix solve failed");

    // Use the global 3-D maxima, not a single slice: the open helix's tips —
    // where ∂µJµ sources C — sit at the axial extremes, off the mid-plane.
    let a_peak  = global_max(&result, FieldName::AMagnitude);
    let b_peak  = global_max(&result, FieldName::BMagnitude);
    let c_peak  = global_max(&result, FieldName::CField);
    let pg_peak = global_max(&result, FieldName::PhiG);
    let bg_peak = global_max(&result, FieldName::BgMagnitude);

    println!("\n── ORC-x7m dynamical-C coupling (AC open helix, κ_G=1, global 3-D max) ──");
    println!("  |A|   peak = {:.4e}  (post-FDTD a_vec)", a_peak);
    println!("  |B|   peak = {:.4e} T", b_peak);
    println!("  |C|   peak = {:.4e} 1/m   (live, post-FDTD)", c_peak);
    println!("  Φ_g   peak = {:.4e} m²/s² (KK-direct)", pg_peak);
    println!("  |B_g| peak = {:.4e} 1/s", bg_peak);

    // (1) The AC drive must produce a magnetic field at all.
    assert!(b_peak > 1e-9, "open helix produced no B field — AC drive broken");

    // (2) Stability guard (ORC-4eg): if the EED γ=1 FDTD had blown up to NaN, the
    // max-reduction would report 0 for the evolved fields.  A genuinely non-zero,
    // finite C means the leapfrog stayed stable AND the deleted 7th DOF is alive —
    // sourced by ∂µJµ at the open-helix tips, exactly as EED predicts.
    assert!(c_peak.is_finite() && a_peak.is_finite(),
        "evolved fields are non-finite — EED FDTD blew up (ORC-4eg regression)");
    assert!(c_peak > 1e-6,
        "live post-FDTD C is ~0 ({c_peak:.3e}) — dynamical 7th DOF did not develop");

    // (3) KK-direct identity (ORC-x7m): with κ_G = 1 the map is Φ_g = C pointwise,
    // so the Φ_g peak must equal the *live post-FDTD* C peak.  If the coupling
    // still sourced from the pre-loop static (Coulomb-gauge) snapshot, Φ_g would
    // track a different field and the ratio would diverge from 1.
    let ratio = pg_peak / c_peak;
    assert!((ratio - 1.0).abs() < 1e-3,
        "Φ_g peak ({pg_peak:.3e}) != live C peak ({c_peak:.3e}); ratio={ratio:.6} — \
         KK-direct is not coupling to the dynamical post-FDTD C");
}
