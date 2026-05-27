//! Performance benchmark: validate solver meets timing targets (ORC-vod).
//!
//! Targets (Apple M-series GPU):
//!   - Static solve (64³ grid): < 100 ms
//!   - Single FDTD step (64³ grid): < 10 ms
//!
//! These are regression benchmarks — if they fail, a performance regression
//! has occurred.  Tests use `cargo test -- --nocapture` to print timings.

use std::time::Instant;
use solver_gpu::{
    biot::{entity_to_segments, WireSegment},
    context::GpuContext,
    grid::{GpuGridState, YeeGrid},
    types::{CoilEntity, CoilParams, CoilType},
};

fn default_solenoid() -> CoilEntity {
    CoilEntity {
        coil: CoilParams {
            coil_type:     CoilType::Solenoid,
            radius_m:      0.05,
            turns:         10,
            pitch_m:       0.005,
            wire_radius_m: 0.001,
            current_a:     1.0,
        },
        position_m:    [0.0, 0.0, 0.0],
        orientation:   [0.0, 0.0, 0.0, 1.0],
        superconducting: false,
    }
}

#[tokio::test]
async fn benchmark_static_solve_64() {
    let cells    = 64_u32;
    let domain_r = 0.2_f64;
    let grid     = YeeGrid::new(cells, domain_r);

    let ctx    = GpuContext::new().await.expect("GPU init");
    let gstate = GpuGridState::new(&ctx, &grid);

    let segments: Vec<WireSegment> = entity_to_segments(&default_solenoid());

    let t0 = Instant::now();

    gstate.run_biot_savart(&ctx, &grid, &segments).expect("Biot-Savart");
    gstate.run_derive_fields(&ctx, &grid).expect("derive");
    gstate.run_observables(&ctx, &grid).expect("observables");

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("Static solve 64³: {:.1} ms", elapsed_ms);

    // Target: < 100 ms on Apple M-series GPU in debug profile.
    // Relax to 2000 ms for CI / software renderers.
    let target_ms = if cfg!(debug_assertions) { 2000.0 } else { 100.0 };
    assert!(
        elapsed_ms < target_ms,
        "Static 64³ solve took {:.1} ms, exceeds {:.0} ms target",
        elapsed_ms, target_ms,
    );
}

#[tokio::test]
async fn benchmark_fdtd_step_64() {
    let cells    = 64_u32;
    let domain_r = 0.2_f64;
    let grid     = YeeGrid::new(cells, domain_r);

    let ctx    = GpuContext::new().await.expect("GPU init");
    let mut gstate = GpuGridState::new(&ctx, &grid);

    let segments: Vec<WireSegment> = entity_to_segments(&default_solenoid());
    gstate.run_biot_savart(&ctx, &grid, &segments).expect("BS");

    // Warmup: 1 step
    let cfl_dt = grid.cfl_dt() as f32;
    gstate.run_fdtd(&ctx, &grid, cfl_dt, 1, 1.0).expect("warmup FDTD");

    // Timed: 10 steps
    let t0 = Instant::now();
    gstate.run_fdtd(&ctx, &grid, cfl_dt, 10, 1.0).expect("timed FDTD");
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let per_step_ms = elapsed_ms / 10.0;

    println!("FDTD step 64³: {:.2} ms/step (10 steps in {:.1} ms)", per_step_ms, elapsed_ms);

    // Target: < 10 ms/step release, < 200 ms/step debug.
    let target_ms = if cfg!(debug_assertions) { 200.0 } else { 10.0 };
    assert!(
        per_step_ms < target_ms,
        "FDTD step took {:.2} ms, exceeds {:.0} ms target",
        per_step_ms, target_ms,
    );
}
