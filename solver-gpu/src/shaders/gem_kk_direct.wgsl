// ─────────────────────────────────────────────────────────────────────────────
// gem_kk_direct.wgsl — Kaluza-Klein DIRECT (algebraic) GEM coupling
//
// Wilhelm 2026 (§4.10 / Fig. 9) identifies the EM four-potential A_μ with the
// off-diagonal 5D Kaluza-Klein metric component g_5μ.  This is a *direct
// algebraic identification*, not a derivative wave coupling:
//
//     Φ_g(x)  =  κ_G · C(x)          (gravitational scalar ← EED scalar C = ∇·A)
//     A_g(x)  =  κ_G · A(x)          (gravitomagnetic potential ← vector potential A)
//
// Unlike the SLW-mediated channel (κ_G·∂C/∂t, κ_G·∇C in fdtd_gem.wgsl), this
// responds to *static* configurations.  The canonical case is the DC toroid:
// B is confined to the tube but A is non-zero everywhere (Aharonov-Bohm region),
// so A_g = κ_G·A is non-zero everywhere and B_g = ∇×A_g lights up — even though
// ∂C/∂t = 0 and the derivative coupling sees nothing (proven by ORC-0km).
//
// This is a pointwise assignment (no stencil, no leapfrog) so it is
// unconditionally stable and visible at κ_G = 1, independent of dt.  The source
// fields a_src / c_src are *snapshots* of the static EED potentials taken before
// the EM FDTD radiates the DC field away (see GpuGridState::snapshot_gem_sources).
//
// `additive = 0` overwrites Φ_g / A_g (KkDirect mode).
// `additive = 1` adds on top of the existing FDTD result (Both mode: SLW + KK).
//
// Bindings:
//   0  phi_g    storage read_write   n1³ × f32       (gravitational scalar)
//   1  a_g_vec  storage read_write   n1³ × 4·f32     (gravitomagnetic potential, stride 4)
//   2  a_src    storage read         n1³ × 4·f32     (snapshot of static A, stride 4)
//   3  c_src    storage read         n1³ × f32       (snapshot of static C = ∇·A)
//   4  params   uniform
// ─────────────────────────────────────────────────────────────────────────────

struct KkParams {
    n1:       u32,
    kappa_g:  f32,
    additive: u32,   // 0 = overwrite, 1 = accumulate
    _pad:     u32,
}

@group(0) @binding(0) var<storage, read_write> phi_g:   array<f32>;
@group(0) @binding(1) var<storage, read_write> a_g_vec: array<f32>;  // stride 4
@group(0) @binding(2) var<storage, read>       a_src:   array<f32>;  // stride 4
@group(0) @binding(3) var<storage, read>       c_src:   array<f32>;
@group(0) @binding(4) var<uniform>             params:  KkParams;

@compute @workgroup_size(256)
fn kk_direct(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n     = params.n1;
    let total = n * n * n;
    let i     = gid.x;
    if i >= total { return; }

    let kappa = params.kappa_g;
    let base  = i * 4u;

    let pg  = kappa * c_src[i];
    let agx = kappa * a_src[base];
    let agy = kappa * a_src[base + 1u];
    let agz = kappa * a_src[base + 2u];

    if params.additive == 1u {
        phi_g[i]          += pg;
        a_g_vec[base]     += agx;
        a_g_vec[base + 1u] += agy;
        a_g_vec[base + 2u] += agz;
    } else {
        phi_g[i]           = pg;
        a_g_vec[base]      = agx;
        a_g_vec[base + 1u] = agy;
        a_g_vec[base + 2u] = agz;
        a_g_vec[base + 3u] = 0.0;
    }
}
