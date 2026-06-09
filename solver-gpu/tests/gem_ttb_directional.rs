//! ORC-3yn follow-up: the Townsend-Brown asymmetric capacitor must source a
//! *directional* Φ_g — stronger on one side of the device than the other —
//! because its φ (point electrode vs. large plate) is asymmetric, so the
//! γ-coupling's ∇·A (= C → Φ_g) inherits that asymmetry.  A *symmetric*
//! capacitor (equal plates) is the control: its Φ_g should be ~balanced.
//!
//! This is the number that matters for a thrust-direction prediction: which way,
//! and how strongly, the gravitational-scalar field leans.

use solver_gpu::{
    OracleSolver,
    types::{
        CoilEntity, CoilParams, CoilType, CouplingMode, EedParams, FieldName,
        GemParams, SliceAxis, SliceData, SliceRequest, SolveRequest, SolverConfig,
        SolverMode,
    },
};

/// Nearest-pixel sample of a slice at in-plane world coords (a → x_range,
/// b → y_range).  For an X-axis slice that is (a = y, b = z).
fn sample(slice: &SliceData, a: f64, b: f64) -> f64 {
    let [rows, cols] = slice.shape;
    let fa = (a - slice.x_range[0]) / (slice.x_range[1] - slice.x_range[0]);
    let fb = (b - slice.y_range[0]) / (slice.y_range[1] - slice.y_range[0]);
    let col = ((fa * (cols - 1) as f64).round() as i64).clamp(0, cols as i64 - 1) as usize;
    let row = ((fb * (rows - 1) as f64).round() as i64).clamp(0, rows as i64 - 1) as usize;
    slice.data[row * cols as usize + col] as f64
}

fn capacitor_request(coil_type: CoilType, plate_aspect: f64) -> SolveRequest {
    let entity = CoilEntity {
        coil: CoilParams {
            coil_type,
            radius_m:     0.05,
            plate_gap_m:  0.02,
            plate_aspect,
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
        eed:      EedParams { alpha: 0.0, beta: 0.0, gamma: 1.0 },  // γ-coupling on
        gem:      GemParams {
            enabled: true, kappa_g: 1.0, li_torr_mode: false,
            coupling_mode: CouplingMode::KkDirect,
        },
        solver: SolverConfig {
            mode: SolverMode::Static, cells_per_axis: 64, domain_radius_m: 0.10,
            lorenz_gauge: false,
        },
        // X-axis slice (the plane x=0, spanning y,z) so we can sample along z.
        slices: vec![SliceRequest {
            axis: SliceAxis::X, position: 0.5, field: FieldName::PhiG, resolution: 96,
        }],
        request_volume: false,
        volume_field:   FieldName::PhiG,
        holonomy_paths: vec![],
    }
}

/// |Φ_g| at (y=0, z=+d) and (y=0, z=−d), and their asymmetry ratio (≥1).
fn directional(slice: &SliceData, d: f64) -> (f64, f64, f64) {
    let plus  = sample(slice, 0.0,  d).abs();
    let minus = sample(slice, 0.0, -d).abs();
    let ratio = plus.max(minus) / plus.min(minus).max(1e-30);
    (plus, minus, ratio)
}

#[tokio::test]
async fn ttb_phi_g_is_directional() {
    let solver = OracleSolver::new().await.expect("GPU init failed");

    let asym = solver.solve(&capacitor_request(CoilType::CapacitorAsymmetric, 5.0)).await
        .expect("TTB solve failed");
    let sym  = solver.solve(&capacitor_request(CoilType::CapacitorSymmetric, 1.0)).await
        .expect("symmetric solve failed");

    let slice_a = asym.slices.iter().find(|s| s.field == FieldName::PhiG).expect("no Φ_g slice");
    let slice_s = sym.slices.iter().find(|s| s.field == FieldName::PhiG).expect("no Φ_g slice");

    let d = 0.025;  // 2.5 cm each side (just beyond the ±1 cm plates)
    let (a_plus, a_minus, a_ratio) = directional(slice_a, d);
    let (s_plus, s_minus, s_ratio) = directional(slice_s, d);

    let lean = if a_plus > a_minus { "+z (point electrode)" } else { "−z (large plate)" };

    println!("\n── ORC-3yn TTB Φ_g directionality (static, γ=1, 50 kV, ±{d} m along z) ──");
    println!("  asymmetric (TTB):  Φ_g(+z)={a_plus:.3e}  Φ_g(−z)={a_minus:.3e}  ratio={a_ratio:.2}×  → leans {lean}");
    println!("  symmetric (ctrl):  Φ_g(+z)={s_plus:.3e}  Φ_g(−z)={s_minus:.3e}  ratio={s_ratio:.2}×");

    // The TTB must be meaningfully directional: Φ_g differs by >30% across the device.
    assert!(a_ratio > 1.3,
        "TTB Φ_g is not directional (ratio {a_ratio:.2}×) — no thrust-direction asymmetry");
    // The symmetric capacitor is the control: it should be ~balanced.
    assert!(s_ratio < 1.2,
        "symmetric capacitor Φ_g is unexpectedly lopsided (ratio {s_ratio:.2}×) — \
         geometry/coupling artifact?");
    // And the asymmetry is geometry-driven, not numerical: TTB ≫ symmetric.
    assert!(a_ratio > s_ratio,
        "TTB ({a_ratio:.2}×) not more directional than symmetric control ({s_ratio:.2}×)");

    // ── Backend asymmetry metric (ORC-65c): the half-space Σ|Φ_g| readout ──────
    let fa = asym.phi_g_asymmetry.as_ref().expect("no Φ_g asymmetry metric (TTB)");
    let fs = sym.phi_g_asymmetry.as_ref().expect("no Φ_g asymmetry metric (sym)");
    println!("  backend metric:  TTB {:.2}× lean=[{:+.2},{:+.2},{:+.2}]   sym {:.2}×",
        fa.ratio, fa.lean[0], fa.lean[1], fa.lean[2], fs.ratio);

    // TTB: the integrated Φ_g leans clearly toward the +z (point-electrode) half.
    assert!(fa.ratio > 1.3, "backend Φ_g asymmetry not directional ({:.2}×)", fa.ratio);
    assert!(fa.lean[2] > 0.9,
        "TTB Φ_g should lean +z (point electrode); lean=[{:+.2},{:+.2},{:+.2}]",
        fa.lean[0], fa.lean[1], fa.lean[2]);
    // Symmetric control: integrated Φ_g is balanced across the split plane.
    assert!(fs.ratio < 1.2,
        "symmetric capacitor backend asymmetry should be ~balanced ({:.2}×)", fs.ratio);
}
