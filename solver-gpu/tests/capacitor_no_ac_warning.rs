//! ORC-1sw: a voltage-driven capacitor with a *stale* frequency_hz must NOT be
//! routed through the AC current-injection path, and must not emit the
//! "effective current = 0 A — set Current > 0 A" warning (a capacitor has no
//! current slider and is voltage-driven).

use solver_gpu::{
    OracleSolver,
    types::{
        CoilEntity, CoilParams, CoilType, EedParams, FieldName, GemParams,
        SliceAxis, SliceRequest, SolveRequest, SolverConfig, SolverMode,
    },
};

#[tokio::test]
async fn capacitor_with_stale_frequency_emits_no_current_warning() {
    // TTB asymmetric capacitor, voltage-driven, with a leftover AC frequency
    // (as if switched from a solenoid that had f=7 MHz set).
    let entity = CoilEntity {
        coil: CoilParams {
            coil_type:     CoilType::CapacitorAsymmetric,
            radius_m:      0.05,
            plate_gap_m:   0.02,
            plate_aspect:  5.0,
            voltage_v:     50_000.0,
            current_a:     0.0,
            frequency_hz:  7.0e6,      // ← stale: capacitors show no frequency slider
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
        gem:      GemParams::default(),
        solver: SolverConfig {
            mode:            SolverMode::TimeDomain { dt_s: 0.0, n_steps: 20 },
            cells_per_axis:  32,
            domain_radius_m: 0.10,
            lorenz_gauge:    false,
        },
        slices: vec![SliceRequest {
            axis: SliceAxis::Z, position: 0.5, field: FieldName::Phi, resolution: 32,
        }],
        request_volume: false,
        volume_field:   FieldName::Phi,
        holonomy_paths: vec![],
    };

    let solver = OracleSolver::new().await.expect("GPU init failed");
    let result = solver.solve(&request).await.expect("capacitor solve failed");

    println!("\n── ORC-1sw capacitor stale-frequency warnings ──");
    for w in &result.warnings {
        println!("  warning: {w}");
    }

    let bad = result.warnings.iter().any(|w|
        w.contains("effective current = 0") || w.contains("no source injected"));
    assert!(!bad,
        "voltage-driven capacitor wrongly emitted the AC current-injection warning: {:?}",
        result.warnings);
}
