// ─────────────────────────────────────────────────────────────────────────────
// cg_scalar.wgsl — GPU Preconditioned Conjugate Gradient (PCG) for scalar φ
//
// Solves (−∇² + α²)u = b  with Jacobi (diagonal) preconditioner.
//
// Four entry points dispatched per-iteration from Rust:
//   1. cg_matvec    — compute ap = A·p
//   2. cg_dot       — partial workgroup dot product a_dot · b_dot → partial[]
//   3. cg_update_xr — x += α·p;  r -= α·ap   (keeps p unchanged)
//   4. cg_update_p  — p = M⁻¹r + β·p         (uses updated r)
//
// Per-iteration algorithm:
//   ap      = A·p                       (cg_matvec)
//   pAp     = dot(p, ap)                (cg_dot, CPU sum)
//   α       = ρ_old / pAp              (CPU)
//   x, r    = updated                   (cg_update_xr)
//   ‖r‖²    = dot(r, r)                (cg_dot, CPU sum)
//   ρ_new   = m_inv · ‖r‖²            (CPU; m_inv = dx²/(6+α²dx²))
//   β       = ρ_new / ρ_old           (CPU)
//   p       = M⁻¹r + β·p              (cg_update_p)
//   ρ_old   = ρ_new                    (CPU)
//
// Bindings (shared layout across all entry points):
//   0  u_sol    storage read_write  n1³ × f32
//   1  r_vec    storage read_write  n1³ × f32
//   2  p_vec    storage read_write  n1³ × f32
//   3  ap_vec   storage read_write  n1³ × f32
//   4  a_dot    storage read        n1³ × f32  (re-pointed per cg_dot call)
//   5  b_dot    storage read        n1³ × f32  (re-pointed per cg_dot call)
//   6  partial  storage read_write  n_wg × f32
//   7  params   uniform
// ─────────────────────────────────────────────────────────────────────────────

struct CgParams {
    dx:    f32,
    m2:    f32,    // α²
    n1:    u32,
    alpha: f32,    // CG step length (CPU sets this before cg_update_xr)
    beta:  f32,    // CG direction mix (CPU sets this before cg_update_p)
    // Three scalar u32 pads (not vec3<u32>) — vec3 has 16-byte WGSL alignment
    // which would inflate the struct to 48 bytes and mismatch the Rust side.
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> u_sol:   array<f32>;
@group(0) @binding(1) var<storage, read_write> r_vec:   array<f32>;
@group(0) @binding(2) var<storage, read_write> p_vec:   array<f32>;
@group(0) @binding(3) var<storage, read_write> ap_vec:  array<f32>;
@group(0) @binding(4) var<storage, read>       a_dot:   array<f32>;
@group(0) @binding(5) var<storage, read>       b_dot:   array<f32>;
@group(0) @binding(6) var<storage, read_write> partial: array<f32>;
@group(0) @binding(7) var<uniform>             params:  CgParams;

var<workgroup> wg_partial: array<f32, 256>;

// ── Boundary test ─────────────────────────────────────────────────────────────

fn is_boundary(flat: u32) -> bool {
    let n  = i32(params.n1);
    let ix = i32(flat) % n;
    let iy = (i32(flat) / n) % n;
    let iz = i32(flat) / (n * n);
    return ix == 0 || ix == n - 1 || iy == 0 || iy == n - 1 || iz == 0 || iz == n - 1;
}

// ── Entry 1: cg_matvec ────────────────────────────────────────────────────────
// Computes ap_vec[i] = (−∇² + α²)p_vec[i]  for all interior vertices.

fn p_at(ix: i32, iy: i32, iz: i32, n: i32) -> f32 {
    let cx = clamp(ix, 0, n - 1);
    let cy = clamp(iy, 0, n - 1);
    let cz = clamp(iz, 0, n - 1);
    return p_vec[u32(cx + cy * n + cz * n * n)];
}

@compute @workgroup_size(256)
fn cg_matvec(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n    = i32(params.n1);
    let flat = gid.x;
    if flat >= u32(n * n * n) { return; }

    if is_boundary(flat) { ap_vec[flat] = 0.0; return; }

    let ix  = i32(flat) % n;
    let iy  = (i32(flat) / n) % n;
    let iz  = i32(flat) / (n * n);
    let inv_dx2 = 1.0 / (params.dx * params.dx);
    let p_c = p_vec[flat];

    let neg_lap = (6.0 * p_c - (
        p_at(ix+1, iy,   iz,   n) + p_at(ix-1, iy,   iz,   n) +
        p_at(ix,   iy+1, iz,   n) + p_at(ix,   iy-1, iz,   n) +
        p_at(ix,   iy,   iz+1, n) + p_at(ix,   iy,   iz-1, n)
    )) * inv_dx2;

    ap_vec[flat] = neg_lap + params.m2 * p_c;
}

// ── Entry 2: cg_dot ───────────────────────────────────────────────────────────
// Computes partial[workgroup_id] = Σ_{i in wg} a_dot[i]·b_dot[i].
// CPU accumulates these into the total dot product.

@compute @workgroup_size(256)
fn cg_dot(
    @builtin(global_invocation_id)   gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id)           wid: vec3<u32>,
) {
    let n    = i32(params.n1);
    let flat = gid.x;
    var val = 0.0f;
    if flat < u32(n * n * n) { val = a_dot[flat] * b_dot[flat]; }
    wg_partial[lid] = val;
    workgroupBarrier();

    for (var stride = 128u; stride >= 1u; stride >>= 1u) {
        if lid < stride { wg_partial[lid] += wg_partial[lid + stride]; }
        workgroupBarrier();
    }
    if lid == 0u { partial[wid.x] = wg_partial[0]; }
}

// ── Entry 3: cg_update_xr ─────────────────────────────────────────────────────
// Updates solution and residual only; leaves p unchanged.
//   u += α · p
//   r -= α · ap

@compute @workgroup_size(256)
fn cg_update_xr(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n    = i32(params.n1);
    let flat = gid.x;
    if flat >= u32(n * n * n) { return; }
    if is_boundary(flat) { u_sol[flat] = 0.0; r_vec[flat] = 0.0; return; }

    u_sol[flat] += params.alpha * p_vec[flat];
    r_vec[flat] -= params.alpha * ap_vec[flat];
}

// ── Entry 4: cg_update_p ──────────────────────────────────────────────────────
// Updates search direction using updated residual.
//   z = M⁻¹ r = r · dx²/(6 + α²·dx²)  (Jacobi preconditioner)
//   p = z + β · p

@compute @workgroup_size(256)
fn cg_update_p(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n    = i32(params.n1);
    let flat = gid.x;
    if flat >= u32(n * n * n) { return; }
    if is_boundary(flat) { p_vec[flat] = 0.0; return; }

    let dx2   = params.dx * params.dx;
    let m_inv = dx2 / (6.0 + params.m2 * dx2);
    let z     = r_vec[flat] * m_inv;
    p_vec[flat] = z + params.beta * p_vec[flat];
}
