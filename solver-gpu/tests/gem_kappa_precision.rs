//! Precision probe (ORC-gru): does the KK-direct algebraic coupling survive the
//! *physical* Kaluza-Klein κ_G ≈ 7.4×10⁻²⁸ in f32, or does it silently round to
//! zero?
//!
//! # Context
//!
//! ORC-gru was filed against the *wave-source* reading of the KK coupling, where
//! κ_G·(source) is added to the GEM equation of motion alongside the leading
//! c²∇²Φ_g term (c² ≈ 9×10¹⁶).  In f32, κ_G·source (~10⁻³³) added to a c²∇²
//! term (~10⁺) is annihilated by rounding — hence the bead's proposed fix of
//! evolving rescaled variables Φ̃_g = Φ_g/κ_G with O(1) coefficients.
//!
//! But ORC-bn6 implemented KK-direct as a *pointwise algebraic* assignment
//!     A_g = κ_G · A,   Φ_g = κ_G · C
//! with **no** c²∇² term to lose precision against.  The single multiply
//! κ_G·A = 7.4e-28 × ~1e-5 ≈ 7e-33 is comfortably above the f32 normal floor
//! (~1.18e-38), and B_g = ∇×A_g ≈ 7e-33 / dx stays representable too.
//!
//! This test confirms that empirically: at physical κ_G the algebraic B_g must
//! be finite, non-zero, and scale *linearly* with κ_G (B_g = κ_G·B).  If true,
//! the EoM-rescaling in ORC-gru is unnecessary for the KK-direct channel — the
//! only remaining concern is *display* scaling (B_g ~ 1e-30 is below colormap
//! range, which the "Exp" κ_G preset already addresses).

use solver_gpu::{
    OracleSolver,
    types::{
        CoilEntity, CoilParams, CoilType, CouplingMode, EedParams, FieldName,
        GemParams, SliceAxis, SliceRequest, SolveRequest, SolverConfig, SolverMode,
    },
};

const KAPPA_KK: f64 = 7.4e-28; // Kaluza-Klein G/c² coupling

fn toroid_request(kappa_g: f64) -> SolveRequest {
    let entity = CoilEntity {
        coil: CoilParams {
            coil_type:     CoilType::Toroid,
            radius_m:      0.05,
            turns:         20,
            pitch_m:       0.012,
            wire_radius_m: 0.001,
            current_a:     10.0,
            frequency_hz:  0.0,
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
            enabled:       true,
            kappa_g,
            li_torr_mode:  false,
            coupling_mode: CouplingMode::KkDirect,
        },
        solver: SolverConfig {
            // Static mode: KK-direct needs no time stepping, and this keeps the
            // probe a pure test of the algebraic-assignment precision.
            mode:            SolverMode::TimeDomain { dt_s: 0.0, n_steps: 1 },
            cells_per_axis:  32,
            domain_radius_m: 0.10,
            lorenz_gauge:    false,
        },
        slices: vec![SliceRequest {
            axis: SliceAxis::Z, position: 0.5, field: FieldName::BgMagnitude, resolution: 64,
        }],
        request_volume: false,
        volume_field:   FieldName::BgMagnitude,
        holonomy_paths: vec![],
    }
}

fn global_max(result: &solver_gpu::types::SolveResult, f: FieldName) -> f64 {
    result.maxima.iter().find(|m| m.field == f).map(|m| m.max_value).unwrap_or(0.0)
}

#[tokio::test]
async fn kk_direct_survives_physical_kappa() {
    let solver = OracleSolver::new().await.expect("GPU init failed");

    let phys = solver.solve(&toroid_request(KAPPA_KK)).await.expect("physical-κ solve failed");
    let unit = solver.solve(&toroid_request(1.0)).await.expect("unit-κ solve failed");

    let b      = global_max(&unit, FieldName::BMagnitude);
    let bg_1   = global_max(&unit, FieldName::BgMagnitude);
    let bg_phys = global_max(&phys, FieldName::BgMagnitude);

    println!("\n── ORC-gru KK-direct f32 precision probe (DC toroid) ──");
    println!("  |B|                 = {:.4e} T", b);
    println!("  |B_g| (κ_G=1)       = {:.4e} 1/s", bg_1);
    println!("  |B_g| (κ_G=7.4e-28) = {:.4e} 1/s", bg_phys);
    println!("  ratio bg_phys/bg_1  = {:.4e}  (expect ≈ κ_G = 7.4e-28)", bg_phys / bg_1);

    // At κ_G=1 the algebraic identity gives B_g = B (validated in ORC-bn6).
    assert!(bg_1 > 1e-9, "κ_G=1 KK-direct produced no B_g — coupling broken");

    // The headline check: at *physical* κ_G the result is finite and non-zero,
    // i.e. it did NOT silently round to 0 in f32.
    assert!(
        bg_phys.is_finite() && bg_phys > 0.0,
        "physical-κ_G |B_g| = {:.3e} underflowed to zero/NaN in f32 — \
         EoM rescaling (ORC-gru) WOULD be needed",
        bg_phys,
    );

    // And it must scale *linearly* with κ_G (algebraic, not lost to a large
    // c²∇² term).  B_g(κ_G) / B_g(1) should equal κ_G to within f32 rounding.
    let ratio = bg_phys / bg_1;
    let rel_err = (ratio - KAPPA_KK).abs() / KAPPA_KK;
    assert!(
        rel_err < 1e-3,
        "B_g did not scale linearly with κ_G: ratio={:.6e} vs κ_G={:.6e} (rel err {:.2e}) \
         — precision is being lost, EoM rescaling needed",
        ratio, KAPPA_KK, rel_err,
    );

    println!(
        "\n✓ KK-direct B_g scales linearly with κ_G down to the physical value \
         ({:.2e} 1/s at κ_G=7.4e-28) with no f32 underflow.",
        bg_phys,
    );
    println!("  → the algebraic channel needs no EoM rescaling; only display scaling.");
}
