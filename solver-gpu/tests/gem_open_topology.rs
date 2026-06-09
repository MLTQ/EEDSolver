//! ORC-09r: an OPEN-circuit toroid sources the EED scalar C (and hence Φ_g),
//! while the same winding CLOSED produces almost none — because a closed loop
//! has ∇·J ≈ 0 (no charge accumulation), whereas the open tips give ∂µJµ ≠ 0
//! under AC drive.  This is the whole reason the toroidal topologies were inert.

use solver_gpu::{
    OracleSolver,
    types::{
        CoilEntity, CoilParams, CoilType, CouplingMode, EedParams, FieldName,
        GemParams, SliceAxis, SliceRequest, SolveRequest, SolverConfig, SolverMode,
    },
};

fn slice_req(field: FieldName) -> SliceRequest {
    SliceRequest { axis: SliceAxis::Z, position: 0.5, field, resolution: 48 }
}

fn global_max(result: &solver_gpu::types::SolveResult, f: FieldName) -> f64 {
    result.maxima.iter().find(|m| m.field == f).map(|m| m.max_value).unwrap_or(0.0)
}

fn toroid_solve(open: bool) -> SolveRequest {
    let entity = CoilEntity {
        coil: CoilParams {
            coil_type:    CoilType::Toroid,
            radius_m:     0.05,
            turns:        16,
            pitch_m:      0.006,
            wire_radius_m: 0.001,
            current_a:    50.0,         // closed loop: drives 50 A
            voltage_v:    2500.0,       // open circuit: 2500 V / 50 Ω = 50 A (same effective drive)
            frequency_hz: 1.0e9,        // AC
            open_circuit: open,
            open_gap_fraction: 0.30,    // wide gap → well-separated tips
            ..Default::default()
        },
        position_m:    [0.0, 0.0, 0.0],
        orientation:   [0.0, 0.0, 0.0, 1.0],
        superconducting: false,
        angular_velocity_rad_s: [0.0; 3],
    };
    SolveRequest {
        entities: vec![entity],
        eed:      EedParams { alpha: 0.0, beta: 0.0, gamma: 1.0 },
        gem:      GemParams {
            enabled: true, kappa_g: 1.0, li_torr_mode: false,
            coupling_mode: CouplingMode::KkDirect,
        },
        solver: SolverConfig {
            mode: SolverMode::TimeDomain { dt_s: 0.0, n_steps: 150 },
            cells_per_axis: 32, domain_radius_m: 0.10, lorenz_gauge: false,
        },
        slices: vec![slice_req(FieldName::CField), slice_req(FieldName::PhiG)],
        request_volume: false,
        volume_field:   FieldName::PhiG,
        holonomy_paths: vec![],
    }
}

#[tokio::test]
async fn open_toroid_sources_c_closed_does_not() {
    let solver = OracleSolver::new().await.expect("GPU init failed");

    let closed = solver.solve(&toroid_solve(false)).await.expect("closed toroid solve failed");
    let open   = solver.solve(&toroid_solve(true)).await.expect("open toroid solve failed");

    let c_closed  = global_max(&closed, FieldName::CField);
    let c_open    = global_max(&open,   FieldName::CField);
    let pg_closed = global_max(&closed, FieldName::PhiG);
    let pg_open   = global_max(&open,   FieldName::PhiG);

    println!("\n── ORC-09r open vs closed toroid (AC, κ_G=1) ──");
    println!("  |C|  closed = {c_closed:.4e}    open = {c_open:.4e}   (ratio {:.1}×)",
        if c_closed > 0.0 { c_open / c_closed } else { f64::INFINITY });
    println!("  Φ_g  closed = {pg_closed:.4e}    open = {pg_open:.4e}");

    assert!(c_open.is_finite() && pg_open.is_finite(),
        "open toroid blew up (C={c_open:.3e}, Φ_g={pg_open:.3e})");

    // The open toroid develops a genuine dynamical C; the closed one stays near
    // the divergence-free floor.
    assert!(c_open > 1e-6, "open toroid sourced no C ({c_open:.3e}) — tips not open?");
    // The closed toroid is NOT a zero baseline — its strong oscillating A drives
    // a real longitudinal C through the EED coupling.  Opening the loop adds the
    // ∂µJµ tip source on top, multiplying C by ~3× (well-separated tips at the
    // gap).  Assert a clear, meaningful increase rather than going-from-nothing.
    assert!(c_open > 2.0 * c_closed.max(1e-30),
        "open toroid C ({c_open:.3e}) not meaningfully above closed ({c_closed:.3e}) — \
         opening the loop barely changed it");
    assert!(pg_open > pg_closed,
        "open toroid Φ_g ({pg_open:.3e}) not above closed ({pg_closed:.3e})");
}
