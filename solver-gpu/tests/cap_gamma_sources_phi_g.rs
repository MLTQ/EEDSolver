//! ORC-3yn: an AC-driven (Biefeld-Brown) capacitor sources the EED scalar C and
//! hence Φ_g, while the same capacitor at DC produces none — because a static
//! plate field has ∂φ/∂t = 0 (and no current), so C = ∇·A + (1/c²)∂φ/∂t = 0.
//! Oscillating the plate voltage gives ∂φ/∂t ≠ 0 → C → Φ_g.

use solver_gpu::{
    OracleSolver,
    types::{
        CoilEntity, CoilParams, CoilType, CouplingMode, EedParams, FieldName,
        GemParams, SliceAxis, SliceRequest, SolveRequest, SolverConfig, SolverMode,
    },
};

fn global_max(result: &solver_gpu::types::SolveResult, f: FieldName) -> f64 {
    result.maxima.iter().find(|m| m.field == f).map(|m| m.max_value).unwrap_or(0.0)
}

fn ttb_solve(gamma: f64) -> SolveRequest {
    let entity = CoilEntity {
        coil: CoilParams {
            coil_type:    CoilType::CapacitorAsymmetric,
            radius_m:     0.05,
            plate_gap_m:  0.02,
            plate_aspect: 5.0,
            voltage_v:    50_000.0,
            current_a:    0.0,
            frequency_hz: 0.0,
            ..Default::default()
        },
        position_m:    [0.0, 0.0, 0.0],
        orientation:   [0.0, 0.0, 0.0, 1.0],
        superconducting: false,
        angular_velocity_rad_s: [0.0; 3],
    };
    SolveRequest {
        entities: vec![entity],
        eed:      EedParams { alpha: 0.0, beta: 0.0, gamma },
        gem:      GemParams {
            enabled: true, kappa_g: 1.0, li_torr_mode: false,
            coupling_mode: CouplingMode::KkDirect,
        },
        solver: SolverConfig {
            mode: SolverMode::Static,   // quasi-static drive, no FDTD ramp
            cells_per_axis: 48, domain_radius_m: 0.10, lorenz_gauge: false,
        },
        slices: vec![
            SliceRequest { axis: SliceAxis::Z, position: 0.5, field: FieldName::Phi,  resolution: 48 },
            SliceRequest { axis: SliceAxis::Z, position: 0.5, field: FieldName::PhiG, resolution: 48 },
        ],
        request_volume: false,
        volume_field:   FieldName::PhiG,
        holonomy_paths: vec![],
    }
}

#[tokio::test]
async fn static_ttb_sources_phi_g_via_gamma_coupling() {
    let solver = OracleSolver::new().await.expect("GPU init failed");

    let g0 = solver.solve(&ttb_solve(0.0)).await.expect("γ=0 solve failed");
    let g1 = solver.solve(&ttb_solve(1.0)).await.expect("γ=1 solve failed");

    let phi  = global_max(&g1, FieldName::Phi);
    let pg_0 = global_max(&g0, FieldName::PhiG);
    let pg_1 = global_max(&g1, FieldName::PhiG);

    println!("\n── static TTB capacitor: Φ_g from the γ-coupling (κ_G=1, 50 kV, DC) ──");
    println!("  φ   peak       = {phi:.4e} V");
    println!("  Φ_g peak γ=0   = {pg_0:.4e}");
    println!("  Φ_g peak γ=1   = {pg_1:.4e} m²/s²");

    assert!(phi > 1e-3, "capacitor produced no φ field ({phi:.3e})");
    // With γ=1 the static EED A-correction imposes A ← γ∇φ, so ∇·A ≠ 0 → C → Φ_g.
    assert!(pg_1.is_finite() && pg_1 > 1e-3,
        "static TTB sourced no Φ_g at γ=1 ({pg_1:.3e})");
    // Turning the coupling off (γ=0) removes it — confirming the γ∇φ path is the source.
    assert!(pg_1 > 1e3 * pg_0.max(1e-30),
        "Φ_g(γ=1)={pg_1:.3e} not dominated by the γ-coupling vs γ=0 ({pg_0:.3e})");
}
