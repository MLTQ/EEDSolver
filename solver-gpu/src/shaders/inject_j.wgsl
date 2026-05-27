// ─────────────────────────────────────────────────────────────────────────────
// inject_j.wgsl — AC current-source injection into A velocity
//
// Adds the J-source contribution to a_vel at each FDTD step.
// Dispatched BEFORE vel_step so the external driving enters the leapfrog.
//
// Physics:
//   ∂²A/∂t² = c²∇²A − γc²∇(∂φ/∂t) − μ₀c² · J(r,t)
//
//   The external contribution to ∂A/∂t per step:
//     Δ(a_vel_k) = −dt · μ₀c² · J_k(r) · I₀ · sin(ωt)
//
//   With J_k normalised for I₀ = 1 A and the time-varying amplitude folded
//   into `source_amp`:
//     source_amp = −dt · μ₀c² · I₀ · sin(ωt)   [units: V·m·s/(A) · A = V·m·s / m ... → V/m after × J [A/m²]]
//
//   Net: a_vel_k += source_amp · j_src_k
//
// Bindings:
//   0  j_src   storage read        n1³ × 4·f32  (Jx, Jy, Jz, pad)
//   1  a_vel   storage read_write  n1³ × 4·f32
//   2  params  uniform
// ─────────────────────────────────────────────────────────────────────────────

struct InjectParams {
    source_amp: f32,   // −dt · μ₀c² · I₀ · sin(ωt)
    n1:         u32,
    _pad:       vec2<u32>,
}

@group(0) @binding(0) var<storage, read>       j_src:  array<f32>;
@group(0) @binding(1) var<storage, read_write> a_vel:  array<f32>;
@group(0) @binding(2) var<uniform>             params: InjectParams;

@compute @workgroup_size(256)
fn inject_j(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n    = i32(params.n1);
    let flat = gid.x;
    if flat >= u32(n * n * n) { return; }

    // Skip boundary vertices (Dirichlet: velocities zero at walls).
    let ix = i32(flat) % n;
    let iy = (i32(flat) / n) % n;
    let iz = i32(flat) / (n * n);
    if ix == 0 || ix == n - 1 || iy == 0 || iy == n - 1 || iz == 0 || iz == n - 1 { return; }

    let j_base = flat * 4u;
    let a_base = flat * 4u;

    a_vel[a_base]      += params.source_amp * j_src[j_base];
    a_vel[a_base + 1u] += params.source_amp * j_src[j_base + 1u];
    a_vel[a_base + 2u] += params.source_amp * j_src[j_base + 2u];
    // a_vel[a_base + 3u] stays 0 (padding component)
}
