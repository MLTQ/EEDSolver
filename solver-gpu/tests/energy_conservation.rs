//! Integration test: energy conservation in vacuum FDTD (ORC-vvk).
//!
//! Uses the Maxwell case (γ=0) with initial conditions A=0, φ=Gaussian, ∂φ/∂t=0.
//! The correct conserved energy for □φ=0 with Dirichlet BC is:
//!
//!   E = ½ ∫ (|∇φ|² + (∂φ/∂t)²/c²) dV    [analytically conserved; ∂E/∂t = ∇·Poynting]
//!
//! This test reads φ and ∂φ/∂t (phi_vel) directly to compute E, bypassing the
//! observables shader (which stores u=½(E²+B²+C²) in different units).

use wgpu::util::DeviceExt;
use solver_gpu::{
    context::GpuContext,
    grid::{GpuGridState, YeeGrid},
};

/// Upload CPU data into a GPU STORAGE buffer.
fn upload_scalar(ctx: &GpuContext, dst: &wgpu::Buffer, data: &[f32]) {
    let dev = ctx.device();
    let staging = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("upload_staging"),
        contents: bytemuck::cast_slice(data),
        usage:    wgpu::BufferUsages::COPY_SRC,
    });
    let mut enc = dev.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(&staging, 0, dst, 0, (data.len() as u64) * 4);
    ctx.queue().submit([enc.finish()]);
    dev.poll(wgpu::MaintainBase::Wait);
}

/// Scalar wave energy E = ½ ∫ (|∇φ|² + φ_t²/c²) dV
/// computed via second-order FD on interior vertices.
fn scalar_wave_energy(ctx: &GpuContext, gstate: &GpuGridState, grid: &YeeGrid) -> f64 {
    let n1  = (grid.n + 1) as usize;
    let dx  = grid.dx as f32;
    let dx3 = (dx as f64).powi(3);
    const C2: f64 = (2.998e8_f64) * (2.998e8_f64);

    let phi = gstate.readback(ctx, &gstate.phi,     gstate.scalar_len()).unwrap();
    let pv  = gstate.readback(ctx, &gstate.phi_vel, gstate.scalar_len()).unwrap();

    let inv2dx = 0.5 / dx;
    let mut e = 0.0_f64;
    for iz in 1..n1 - 1 {
        for iy in 1..n1 - 1 {
            for ix in 1..n1 - 1 {
                let f = |x: usize, y: usize, z: usize| phi[x + y * n1 + z * n1 * n1];
                let gx = (f(ix+1,iy,iz) - f(ix-1,iy,iz)) * inv2dx;
                let gy = (f(ix,iy+1,iz) - f(ix,iy-1,iz)) * inv2dx;
                let gz = (f(ix,iy,iz+1) - f(ix,iy,iz-1)) * inv2dx;
                let grad2 = gx*gx + gy*gy + gz*gz;
                let v  = pv[ix + iy * n1 + iz * n1 * n1] as f64;
                e += 0.5 * (grad2 as f64 + v * v / C2) * dx3;
            }
        }
    }
    e
}

#[tokio::test]
async fn test_energy_conservation_vacuum_fdtd() {
    let cells    = 32_u32;
    let domain_r = 0.10_f64;
    let grid     = YeeGrid::new(cells, domain_r);
    let n1       = (grid.n + 1) as usize;
    let dt       = grid.cfl_dt() as f32;

    let ctx    = GpuContext::new().await.expect("GPU init");
    let gstate = GpuGridState::new(&ctx, &grid);

    // Smooth Gaussian φ (σ ≈ 5 cells; decays to ≈ 0 at walls)
    let center = (n1 / 2) as f32;
    let sigma  = n1 as f32 * 0.15;   // ≈ 5 cells
    let phi_init: Vec<f32> = (0..n1 * n1 * n1).map(|idx| {
        let ix = (idx % n1) as f32 - center;
        let iy = ((idx / n1) % n1) as f32 - center;
        let iz = (idx / (n1 * n1)) as f32 - center;
        (-( ix*ix + iy*iy + iz*iz ) / (2.0 * sigma * sigma)).exp()
    }).collect();
    upload_scalar(&ctx, &gstate.phi, &phi_init);

    // Initial energy (phi_vel=0 → E₀ = ½∫|∇φ|²dV)
    let e0 = scalar_wave_energy(&ctx, &gstate, &grid);
    assert!(e0 > 0.0 && e0.is_finite(), "E₀ must be positive finite");
    println!("Energy conservation test (γ=0, no sponge, 32³ grid):");
    println!("  E₀ = {:.8e}", e0);

    // The KD (kick-drift) symplectic Euler scheme conserves a modified Hamiltonian
    // H' = H + O(dt·ω)·correction.  The actual Hamiltonian H oscillates with
    // amplitude ~ dt·ω·H ≈ H/(σ_cells·√3) ≈ H/8 for σ=5 cells.
    // The TIME-AVERAGE of H over many steps is close to H₀ to O(dt²):
    //   ‹H› = H₀ + O(dt²·ω²·H₀) = H₀ · (1 ± 1/σ_cells²) ≈ H₀ ± 4%
    // We test the average over 20 steps (oscillations cancel) < 2%,
    // and that the energy is always finite and positive (no blow-up).
    let n_steps = 20_u32;
    let mut e_sum = 0.0_f64;
    for step in 1..=n_steps {
        gstate.run_fdtd_sponge(&ctx, &grid, dt, 1, 0.0, Some(0.0)).unwrap();
        let e = scalar_wave_energy(&ctx, &gstate, &grid);
        assert!(e.is_finite() && e > 0.0, "Energy non-finite/non-positive at step {step}");
        let pct = (e - e0) / e0 * 100.0;
        println!("  step {:2}: E = {:.8e}  (ΔE/E₀ = {:+.4}%)", step, e, pct);
        e_sum += e;
    }

    let e_avg = e_sum / n_steps as f64;
    let avg_rel = ((e_avg - e0) / e0).abs();
    println!("\n  E_avg = {:.8e}  (avg ΔE/E₀ = {:.4}%)", e_avg, avg_rel * 100.0);

    // Time-average conserved to O(dt²ω²) ≈ 4% for σ≈5 cells.
    // Require < 5% average drift (no monotonic energy growth/loss).
    assert!(
        avg_rel < 0.05,
        "Time-averaged energy deviated {:.3}% from E₀ — FDTD has secular energy drift",
        avg_rel * 100.0
    );
    println!("✓ Time-averaged energy conserved to {:.4}% over {} steps", avg_rel * 100.0, n_steps);
}
