// ─────────────────────────────────────────────────────────────────────────────
// fdtd_gem.wgsl — GEM (gravitoelectromagnetic) FDTD leapfrog
//
// Evolves the gravitomagnetic 4-potential (Φ_g, A_g) coupled to the EED
// C-field via the κ_G coupling constant.
//
// GEM equations of motion (linearised GR + EED coupling):
//
//   ∂²Φ_g/∂t² = c²∇²Φ_g + 4πG·ρ_m  +  κ_G · ∂C/∂t
//   ∂²A_g/∂t² = c²∇²A_g − ∇(∂Φ_g/∂t) − (4πG/c)J_m  +  κ_G · ∇C
//
// The EED→GEM coupling (κ_G terms) is the key novel physics:
//   - ∂C/∂t  sources  Φ_g  (temporal C oscillations drive gravitational waves)
//   - ∇C     sources  A_g  (spatial C gradients drive gravitomagnetic field)
//
// # κ_G guide
//   KK prediction : G/c² ≈ 7.4e-28  m/kg    (Kaluza-Klein)
//   Li-Torr       : 2mₑ/e ≈ 1.14e-11         (superconductor London moment)
//
// Same two-pass leapfrog as fdtd_em.wgsl (vel_gem, pos_gem).
//
// Bindings:
//   0  phi_g      storage read_write   n1³ × f32
//   1  a_g_vec    storage read_write   n1³ × 4·f32  [Agx,Agy,Agz,0]
//   2  phi_g_vel  storage read_write   n1³ × f32
//   3  a_g_vel    storage read_write   n1³ × 4·f32
//   4  c_fld      storage read         n1³ × f32    (EED C-field, read-only)
//   5  c_fld_prev storage read         n1³ × f32    (C from previous step for ∂C/∂t)
//   6  params     uniform
// ─────────────────────────────────────────────────────────────────────────────

struct GemParams {
    dx:       f32,
    dt:       f32,
    n1:       u32,
    kappa_g:  f32,   // κ_G  [dimensionless × physical constant baked in]
}

@group(0) @binding(0) var<storage, read_write> phi_g:      array<f32>;
@group(0) @binding(1) var<storage, read_write> a_g_vec:    array<f32>;  // stride 4
@group(0) @binding(2) var<storage, read_write> phi_g_vel:  array<f32>;
@group(0) @binding(3) var<storage, read_write> a_g_vel:    array<f32>;  // stride 4
@group(0) @binding(4) var<storage, read>       c_fld:      array<f32>;  // current C
@group(0) @binding(5) var<storage, read>       c_fld_prev: array<f32>;  // previous C
@group(0) @binding(6) var<uniform>             params:     GemParams;

const C2: f32 = 8.9875517873681764e16;  // c² [m²/s²]

// ── Index helpers ─────────────────────────────────────────────────────────────

fn pg_at(ix: i32, iy: i32, iz: i32, n: i32) -> f32 {
    let cx = clamp(ix, 0, n-1); let cy = clamp(iy, 0, n-1); let cz = clamp(iz, 0, n-1);
    return phi_g[u32(cx + cy*n + cz*n*n)];
}

fn pgv_at(ix: i32, iy: i32, iz: i32, n: i32) -> f32 {
    let cx = clamp(ix, 0, n-1); let cy = clamp(iy, 0, n-1); let cz = clamp(iz, 0, n-1);
    return phi_g_vel[u32(cx + cy*n + cz*n*n)];
}

fn ag_comp(ix: i32, iy: i32, iz: i32, n: i32, comp: u32) -> f32 {
    let cx = clamp(ix, 0, n-1); let cy = clamp(iy, 0, n-1); let cz = clamp(iz, 0, n-1);
    return a_g_vec[u32(cx + cy*n + cz*n*n)*4u + comp];
}

fn agv_comp(ix: i32, iy: i32, iz: i32, n: i32, comp: u32) -> f32 {
    let cx = clamp(ix, 0, n-1); let cy = clamp(iy, 0, n-1); let cz = clamp(iz, 0, n-1);
    return a_g_vel[u32(cx + cy*n + cz*n*n)*4u + comp];
}

fn c_at(ix: i32, iy: i32, iz: i32, n: i32) -> f32 {
    let cx = clamp(ix, 0, n-1); let cy = clamp(iy, 0, n-1); let cz = clamp(iz, 0, n-1);
    return c_fld[u32(cx + cy*n + cz*n*n)];
}

// ── Pass 1: GEM velocity update ───────────────────────────────────────────────

@compute @workgroup_size(256)
fn vel_gem(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n    = i32(params.n1);
    let flat = gid.x;
    if flat >= u32(n*n*n) { return; }

    let ix = i32(flat) % n;
    let iy = (i32(flat) / n) % n;
    let iz = i32(flat) / (n*n);

    if ix == 0 || ix == n-1 || iy == 0 || iy == n-1 || iz == 0 || iz == n-1 { return; }

    let dx      = params.dx;
    let dt      = params.dt;
    let kappa   = params.kappa_g;
    let inv2dx  = 0.5 / dx;
    let inv_dx2 = 1.0 / (dx * dx);

    // ── ∂C/∂t ≈ (C_cur − C_prev) / dt ────────────────────────────────────────
    // Guard: dt=0 would give NaN (0/0) which corrupts all GEM fields.
    // This should never happen after the dt_s==0 fix in lib.rs, but keep it
    // as a shader-level safety net.
    if dt <= 0.0 { return; }
    let dC_dt = (c_fld[flat] - c_fld_prev[flat]) / dt;

    // ── Laplacian of Φ_g ──────────────────────────────────────────────────────
    let pg_c = pg_at(ix, iy, iz, n);
    let lap_pg = (
        pg_at(ix+1, iy,   iz,   n) + pg_at(ix-1, iy,   iz,   n) +
        pg_at(ix,   iy+1, iz,   n) + pg_at(ix,   iy-1, iz,   n) +
        pg_at(ix,   iy,   iz+1, n) + pg_at(ix,   iy,   iz-1, n)
        - 6.0 * pg_c
    ) * inv_dx2;

    // ── div(A_g_vel) for the GEM Lorenz-type coupling ─────────────────────────
    let div_agv =
        (agv_comp(ix+1,iy,  iz,  n,0u) - agv_comp(ix-1,iy,  iz,  n,0u)) * inv2dx +
        (agv_comp(ix,  iy+1,iz,  n,1u) - agv_comp(ix,  iy-1,iz,  n,1u)) * inv2dx +
        (agv_comp(ix,  iy,  iz+1,n,2u) - agv_comp(ix,  iy,  iz-1,n,2u)) * inv2dx;

    // ── Φ_g acceleration = c²∇²Φ_g − c²·div(A_g_vel) + κ_G·∂C/∂t ───────────
    // (The 4πG·ρ_m source is zero for vacuum; add in Phase 5 for mass distributions.)
    let acc_pg = C2 * (lap_pg - div_agv) + kappa * dC_dt;
    phi_g_vel[flat] += dt * acc_pg;

    // ── Laplacian of A_g components ───────────────────────────────────────────
    let ag_base = flat * 4u;

    let lap_agx = (
        ag_comp(ix+1,iy,  iz,  n,0u) + ag_comp(ix-1,iy,  iz,  n,0u) +
        ag_comp(ix,  iy+1,iz,  n,0u) + ag_comp(ix,  iy-1,iz,  n,0u) +
        ag_comp(ix,  iy,  iz+1,n,0u) + ag_comp(ix,  iy,  iz-1,n,0u)
        - 6.0 * a_g_vec[ag_base]
    ) * inv_dx2;

    let lap_agy = (
        ag_comp(ix+1,iy,  iz,  n,1u) + ag_comp(ix-1,iy,  iz,  n,1u) +
        ag_comp(ix,  iy+1,iz,  n,1u) + ag_comp(ix,  iy-1,iz,  n,1u) +
        ag_comp(ix,  iy,  iz+1,n,1u) + ag_comp(ix,  iy,  iz-1,n,1u)
        - 6.0 * a_g_vec[ag_base+1u]
    ) * inv_dx2;

    let lap_agz = (
        ag_comp(ix+1,iy,  iz,  n,2u) + ag_comp(ix-1,iy,  iz,  n,2u) +
        ag_comp(ix,  iy+1,iz,  n,2u) + ag_comp(ix,  iy-1,iz,  n,2u) +
        ag_comp(ix,  iy,  iz+1,n,2u) + ag_comp(ix,  iy,  iz-1,n,2u)
        - 6.0 * a_g_vec[ag_base+2u]
    ) * inv_dx2;

    // ── ∇(∂Φ_g/∂t) for the EED-type cross-coupling ───────────────────────────
    let gpgv_x = (pgv_at(ix+1,iy,  iz,  n) - pgv_at(ix-1,iy,  iz,  n)) * inv2dx;
    let gpgv_y = (pgv_at(ix,  iy+1,iz,  n) - pgv_at(ix,  iy-1,iz,  n)) * inv2dx;
    let gpgv_z = (pgv_at(ix,  iy,  iz+1,n) - pgv_at(ix,  iy,  iz-1,n)) * inv2dx;

    // ── ∇C for the κ_G source term ────────────────────────────────────────────
    let grad_C_x = (c_at(ix+1,iy,  iz,  n) - c_at(ix-1,iy,  iz,  n)) * inv2dx;
    let grad_C_y = (c_at(ix,  iy+1,iz,  n) - c_at(ix,  iy-1,iz,  n)) * inv2dx;
    let grad_C_z = (c_at(ix,  iy,  iz+1,n) - c_at(ix,  iy,  iz-1,n)) * inv2dx;

    // ── A_g acceleration = c²∇²A_g − c²·∇(∂Φ_g/∂t) + κ_G·∇C ───────────────
    a_g_vel[ag_base]      += dt * (C2*(lap_agx - gpgv_x) + kappa*grad_C_x);
    a_g_vel[ag_base + 1u] += dt * (C2*(lap_agy - gpgv_y) + kappa*grad_C_y);
    a_g_vel[ag_base + 2u] += dt * (C2*(lap_agz - gpgv_z) + kappa*grad_C_z);
}

// ── Pass 2: GEM position update ───────────────────────────────────────────────

@compute @workgroup_size(256)
fn pos_gem(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n    = i32(params.n1);
    let flat = gid.x;
    if flat >= u32(n*n*n) { return; }

    let ix = i32(flat) % n;
    let iy = (i32(flat) / n) % n;
    let iz = i32(flat) / (n*n);

    if ix == 0 || ix == n-1 || iy == 0 || iy == n-1 || iz == 0 || iz == n-1 {
        phi_g[flat] = 0.0;
        let b = flat*4u;
        a_g_vec[b] = 0.0; a_g_vec[b+1u] = 0.0; a_g_vec[b+2u] = 0.0; a_g_vec[b+3u] = 0.0;
        return;
    }

    let dt = params.dt;
    phi_g[flat] += dt * phi_g_vel[flat];
    let b = flat * 4u;
    a_g_vec[b]      += dt * a_g_vel[b];
    a_g_vec[b + 1u] += dt * a_g_vel[b + 1u];
    a_g_vec[b + 2u] += dt * a_g_vel[b + 2u];
}
