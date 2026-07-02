//! Integration test: longitudinal propagation speed of the coupled EED system
//! at γ=1 (ORC-0r9).
//!
//! The implemented (bespoke, coupled) equations of motion are
//!
//!   ∂²φ/∂t² = c²∇²φ − γ·c²·∂(∇·A)/∂t
//!   ∂²A/∂t² = c²∇²A − γ·∇(∂φ/∂t)
//!
//! Fourier analysis of the longitudinal sector (d ≡ ∇·A):
//!
//!   φ̈ = −c²k²φ − γc²ḋ
//!   d̈ = −c²k²d + γk²φ̇
//!
//! gives (c²k² − ω²)² = γ²ω²c²k², i.e. two non-dispersive branches
//!
//!   ω± = ck·(√(γ²+4) ∓ γ)/2
//!
//! At γ=1 the phase/group speeds are (√5−1)/2·c ≈ 0.618c and
//! (√5+1)/2·c ≈ 1.618c — golden-ratio multiples of c, NOT c.  (The earlier
//! FIELD_THEORY.md claim that the longitudinal ∇·A half of C "radiates at c"
//! was wrong; only the γ=0 limit propagates at c.)
//!
//! Method: initialise a purely longitudinal wave packet A = ∇G (G a Gaussian),
//! φ = 0, all velocities zero.  This splits 50/50 between the two branches
//! (eigenvectors P = ±cA).  Track the outward front of |C| over time and fit
//! its speed:
//!   γ=0 control → front at c   (plain wave equation for every A component)
//!   γ=1        → front at 1.618c (the fast branch leads)

use wgpu::util::DeviceExt;
use solver_gpu::{
    context::GpuContext,
    grid::{GpuGridState, YeeGrid},
};

const C_LIGHT: f64 = 2.998e8;

fn upload(ctx: &GpuContext, dst: &wgpu::Buffer, data: &[f32]) {
    let dev = ctx.device();
    let stg = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("upload_staging"),
        contents: bytemuck::cast_slice(data),
        usage:    wgpu::BufferUsages::COPY_SRC,
    });
    let mut enc = dev.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(&stg, 0, dst, 0, (data.len() as u64) * 4);
    ctx.queue().submit([enc.finish()]);
    dev.poll(wgpu::MaintainBase::Wait);
}

/// Radial profile P(r) = ⟨C²⟩_shell · r², 1-cell bins, lightly smoothed.
///
/// The r² weight compensates the 1/r² geometric decay of an outgoing 3-D
/// shell, so the shell's profile peak stays at roughly constant height as it
/// propagates — which lets a single ABSOLUTE threshold track the front across
/// snapshots.  (A relative-to-max threshold fails here: at γ=1 the global max
/// sits in the slow branch near the origin, hiding the fast front.)
fn radial_profile(c: &[f32], n1: usize) -> Vec<f64> {
    let center = (n1 / 2) as f64;
    let n_bins = n1 / 2 + 1;
    let mut sum   = vec![0.0f64; n_bins];
    let mut count = vec![0u32;   n_bins];
    for iz in 1..n1 - 1 {
        for iy in 1..n1 - 1 {
            for ix in 1..n1 - 1 {
                let r = ((ix as f64 - center).powi(2)
                       + (iy as f64 - center).powi(2)
                       + (iz as f64 - center).powi(2)).sqrt();
                let bin = r.round() as usize;
                if bin < n_bins {
                    let v = c[ix + iy * n1 + iz * n1 * n1] as f64;
                    sum[bin]   += v * v;
                    count[bin] += 1;
                }
            }
        }
    }
    let raw: Vec<f64> = (0..n_bins)
        .map(|b| if count[b] > 0 {
            (sum[b] / count[b] as f64) * (b as f64).max(1.0).powi(2)
        } else { 0.0 })
        .collect();
    // 3-bin moving average to kill single-bin noise.
    (0..n_bins)
        .map(|b| {
            let lo = b.saturating_sub(1);
            let hi = (b + 1).min(n_bins - 1);
            raw[lo..=hi].iter().sum::<f64>() / (hi - lo + 1) as f64
        })
        .collect()
}

/// Outermost radius [cells] where P(r) crosses the absolute threshold,
/// linearly interpolated between bins for sub-cell resolution.
fn front_radius_cells(profile: &[f64], thresh: f64) -> f64 {
    for b in (1..profile.len()).rev() {
        if profile[b] >= thresh {
            // Interpolate the crossing between bin b and bin b+1 (below thresh).
            if b + 1 < profile.len() && profile[b + 1] < profile[b] {
                let t = (profile[b] - thresh) / (profile[b] - profile[b + 1]).max(1e-300);
                return b as f64 + t.clamp(0.0, 1.0);
            }
            return b as f64;
        }
    }
    0.0
}

/// Least-squares slope of y over x.
fn slope(x: &[f64], y: &[f64]) -> f64 {
    let n  = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let num: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let den: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
    num / den
}

/// Launch the longitudinal packet at coupling `gamma`, return fitted front
/// speed [m/s] of the |C| wavefront.
async fn front_speed(gamma: f32) -> f64 {
    let cells    = 96_u32;
    let domain_r = 0.10_f64;
    let grid     = YeeGrid::new(cells, domain_r);
    let n1       = (grid.n + 1) as usize;
    let dx       = grid.dx;
    // 0.5×CFL safety factor: the γ cross-coupling bound (β ≤ 2, ORC-4eg) is
    // marginal at full CFL for the highest grid modes.
    let dt       = (grid.cfl_dt() * 0.5) as f32;

    let ctx    = GpuContext::new().await.expect("GPU init");
    let gstate = GpuGridState::new(&ctx, &grid);

    // A = ∇G, G = exp(−r²/2σ²): purely longitudinal (curl ∇G = 0), compact.
    let sigma  = 3.0 * dx;
    let center = (n1 / 2) as f64;
    let mut a_init = vec![0.0f32; n1 * n1 * n1 * 4];
    for iz in 0..n1 {
        for iy in 0..n1 {
            for ix in 0..n1 {
                let x = (ix as f64 - center) * dx;
                let y = (iy as f64 - center) * dx;
                let z = (iz as f64 - center) * dx;
                let g = (-(x * x + y * y + z * z) / (2.0 * sigma * sigma)).exp();
                let base = (ix + iy * n1 + iz * n1 * n1) * 4;
                a_init[base]     = (-(x / (sigma * sigma)) * g) as f32;
                a_init[base + 1] = (-(y / (sigma * sigma)) * g) as f32;
                a_init[base + 2] = (-(z / (sigma * sigma)) * g) as f32;
            }
        }
    }
    upload(&ctx, &gstate.a_vec, &a_init);

    // Sample the front radius at several times.  No sponge; the run ends
    // before the fast front reaches the walls (48 cells from center).
    // c·dt = dx/(2√3) ≈ 0.289·dx per step → fast branch ≈ 0.47 cells/step.
    // Start at step 30: on a 64³ grid with earlier sampling the fast shell is
    // still emerging from under the slow branch and the fitted speed reads
    // low (measured 1.11c→1.52c per-interval, still rising, at steps 16–44).
    // The absolute threshold is anchored to the first sample's profile peak.
    let sample_steps: [u32; 5] = [30, 40, 50, 60, 70];
    let mut t_s    = Vec::new();
    let mut r_m    = Vec::new();
    let mut done   = 0u32;
    let mut thresh = 0.0f64;
    for &s in &sample_steps {
        gstate.run_fdtd_sponge(&ctx, &grid, dt, s - done, gamma, Some(0.0), None).unwrap();
        done = s;
        let c = gstate.readback(&ctx, &gstate.c_fld, gstate.scalar_len()).unwrap();
        let profile = radial_profile(&c, n1);
        if thresh == 0.0 {
            // 1% of the first sample's peak.  The at-rest IC (p=0, ṗ=0, d=d₀,
            // ḋ=0) splits C unevenly between the branches: the fast branch
            // carries only ~16% of the slow branch's C amplitude (~2.5% in
            // C² — verified by eigen-decomposing the discrete one-step update
            // matrix), so a 10% threshold sees only the slow shell.  Outside
            // the fast front there is causally nothing but f32 noise, many
            // orders below 1%, so the low threshold is safe.
            thresh = 0.01 * profile.iter().cloned().fold(0.0, f64::max);
            assert!(thresh > 0.0, "profile is identically zero — no wave launched");
        }
        let rf = front_radius_cells(&profile, thresh);
        assert!(rf < (cells as f64) / 2.0 - 2.0,
            "front reached the wall at step {s} (r={rf:.1} cells) — shorten the run");
        t_s.push(s as f64 * dt as f64);
        r_m.push(rf * dx);
        println!("  γ={gamma}: step {s:2}  front r = {rf:5.2} cells");
    }

    slope(&t_s, &r_m)
}

#[tokio::test]
async fn test_gamma1_longitudinal_front_speed_is_golden_ratio_not_c() {
    println!("Longitudinal C-front speed, 96³ grid, A=∇G packet:");

    let v0 = front_speed(0.0).await;
    let v1 = front_speed(1.0).await;

    let v0_c   = v0 / C_LIGHT;
    let v1_c   = v1 / C_LIGHT;
    let ratio  = v1 / v0;
    const PHI: f64 = 1.618033988749895;

    println!("\n  v(γ=0) = {v0_c:.3} c   (control — expect ≈ 1.0 c)");
    println!("  v(γ=1) = {v1_c:.3} c   (docs used to claim c; dispersion predicts {PHI:.3} c)");
    println!("  v(γ=1)/v(γ=0) = {ratio:.3}   (expect ≈ φ = {PHI:.3})");

    // Control: γ=0 is Maxwell-in-Lorenz-gauge; every A component obeys □A=0,
    // so ∇·A (and hence C) propagates at c.  Generous bounds: the front
    // detector has ~1-cell quantisation and the packet has finite width.
    assert!((0.80..=1.20).contains(&v0_c),
        "γ=0 control front speed {v0_c:.3}c is not ≈ c — measurement machinery is broken");

    // The refuted claim: γ=1 longitudinal C radiates at c.  If this assert
    // fires with v1 ≈ 1.0c, the dispersion derivation in FIELD_THEORY.md
    // (2026-07-02) is wrong and the doc needs re-revision.
    assert!(v1_c > 1.25,
        "γ=1 front speed {v1_c:.3}c is consistent with c — golden-ratio dispersion NOT observed");

    // The positive claim: fast branch at ω₊ = ck(√5+1)/2 → front at 1.618c.
    // Ratio v1/v0 cancels systematic threshold/quantisation bias.
    assert!((1.40..=1.85).contains(&ratio),
        "v(γ=1)/v(γ=0) = {ratio:.3}, expected ≈ φ = {PHI:.3} — coupled dispersion differs from derivation");

    println!("✓ γ=1 longitudinal front at {v1_c:.3}c ≈ φ·c — matches ω± = ck(√(γ²+4)∓γ)/2");
}
