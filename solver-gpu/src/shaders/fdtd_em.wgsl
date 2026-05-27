// ─────────────────────────────────────────────────────────────────────────────
// fdtd_em.wgsl — EED potential-primary leapfrog FDTD
//
// Evolves the 4-potential (φ, A) via two explicit passes per time step:
//
//   Pass 1  `vel_step`:   update velocities  (using current positions)
//   Pass 2  `pos_step`:   update positions   (using new velocities)
//
// EED equations of motion (vacuum, γ=1, no gauge fixing):
//
//   ∂²φ/∂t² = c²∇²φ − c²·∂(∇·A)/∂t
//   ∂²A/∂t² = c²∇²A − c²·∇(∂φ/∂t)
//
// Leapfrog (DKD — drift, kick, drift — but here: kick then drift in one step):
//   phi_vel^{n+1} = phi_vel^n  +  dt · [c²∇²φ^n  − c²·div(a_vel^n)]
//   a_vel^{n+1}   = a_vel^n   +  dt · [c²∇²A^n  − c²·∇(phi_vel^n)]
//   phi^{n+1}     = phi^n     +  dt · phi_vel^{n+1}
//   a^{n+1}       = a^n       +  dt · a_vel^{n+1}
//
// Note on atomicity: vel_step and pos_step are dispatched as separate
// GPU passes in a single command buffer (sequential, no CPU round-trip).
// Within each pass, reads and writes to the SAME buffer are isolated
// because each thread only writes its own flat index.
//
// Bindings (same for both entry points):
//   0  phi      storage read_write   n1³ × f32
//   1  a_vec    storage read_write   n1³ × 4·f32    [Ax,Ay,Az,0]
//   2  phi_vel  storage read_write   n1³ × f32
//   3  a_vel    storage read_write   n1³ × 4·f32    [Avx,Avy,Avz,0]
//   4  params   uniform
// ─────────────────────────────────────────────────────────────────────────────

struct FdtdParams {
    dx:          f32,
    dt:          f32,
    n1:          u32,
    gamma:       f32,   // EED coupling: 1.0=full EED, 0.0=Maxwell
    // Sponge layer (absorbing BC): cells nearest boundary are damped.
    // sponge_cells = number of cells in the absorbing sponge layer.
    // sigma_max    = peak damping rate [1/s].  0 = no damping (hard wall).
    sponge_cells: u32,
    sigma_max:    f32,
    _pad:         vec2<u32>,
}

@group(0) @binding(0) var<storage, read_write> phi:     array<f32>;
@group(0) @binding(1) var<storage, read_write> a_vec:   array<f32>;   // stride 4
@group(0) @binding(2) var<storage, read_write> phi_vel: array<f32>;
@group(0) @binding(3) var<storage, read_write> a_vel:   array<f32>;   // stride 4
@group(0) @binding(4) var<uniform>             params:  FdtdParams;

// ── Constants ────────────────────────────────────────────────────────────────

const C2: f32 = 8.9875517873681764e16;  // c² [m²/s²]

// ── Index helpers ─────────────────────────────────────────────────────────────

fn phi_at(ix: i32, iy: i32, iz: i32, n: i32) -> f32 {
    let cx = clamp(ix, 0, n - 1);
    let cy = clamp(iy, 0, n - 1);
    let cz = clamp(iz, 0, n - 1);
    return phi[u32(cx + cy * n + cz * n * n)];
}

fn pv_at(ix: i32, iy: i32, iz: i32, n: i32) -> f32 {
    let cx = clamp(ix, 0, n - 1);
    let cy = clamp(iy, 0, n - 1);
    let cz = clamp(iz, 0, n - 1);
    return phi_vel[u32(cx + cy * n + cz * n * n)];
}

// Vector potential component from a_vec (stride 4): comp ∈ {0,1,2}
fn a_comp(ix: i32, iy: i32, iz: i32, n: i32, comp: u32) -> f32 {
    let cx = clamp(ix, 0, n - 1);
    let cy = clamp(iy, 0, n - 1);
    let cz = clamp(iz, 0, n - 1);
    return a_vec[u32(cx + cy * n + cz * n * n) * 4u + comp];
}

// a_vel component (stride 4): comp ∈ {0,1,2}
fn av_comp(ix: i32, iy: i32, iz: i32, n: i32, comp: u32) -> f32 {
    let cx = clamp(ix, 0, n - 1);
    let cy = clamp(iy, 0, n - 1);
    let cz = clamp(iz, 0, n - 1);
    return a_vel[u32(cx + cy * n + cz * n * n) * 4u + comp];
}

// ── Pass 1: velocity update ───────────────────────────────────────────────────
//
// Reads: phi, a_vec, phi_vel, a_vel (all at time n)
// Writes: phi_vel, a_vel (updated to n+1)
//
// Only interior vertices are updated; boundary vertices keep zero velocity
// (simple homogeneous Dirichlet — Mur ABC added in Phase 4).

@compute @workgroup_size(256)
fn vel_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n    = i32(params.n1);
    let flat = gid.x;
    if flat >= u32(n * n * n) { return; }

    let ix = i32(flat) % n;
    let iy = (i32(flat) / n) % n;
    let iz = i32(flat) / (n * n);

    // Zero velocity at domain boundary (Dirichlet).
    if ix == 0 || ix == n - 1 || iy == 0 || iy == n - 1 || iz == 0 || iz == n - 1 {
        return;
    }

    let dx   = params.dx;
    let dt   = params.dt;
    let inv2dx = 0.5 / dx;
    let inv_dx2 = 1.0 / (dx * dx);

    // ── Laplacian of φ ────────────────────────────────────────────────────────
    let phi_c = phi_at(ix,     iy,     iz,     n);
    let lap_phi = (
        phi_at(ix+1, iy,   iz,   n) + phi_at(ix-1, iy,   iz,   n) +
        phi_at(ix,   iy+1, iz,   n) + phi_at(ix,   iy-1, iz,   n) +
        phi_at(ix,   iy,   iz+1, n) + phi_at(ix,   iy,   iz-1, n)
        - 6.0 * phi_c
    ) * inv_dx2;

    // ── div(A_vel) = ∂Avx/∂x + ∂Avy/∂y + ∂Avz/∂z ───────────────────────────
    let div_av =
        (av_comp(ix+1, iy,   iz,   n, 0u) - av_comp(ix-1, iy,   iz,   n, 0u)) * inv2dx +
        (av_comp(ix,   iy+1, iz,   n, 1u) - av_comp(ix,   iy-1, iz,   n, 1u)) * inv2dx +
        (av_comp(ix,   iy,   iz+1, n, 2u) - av_comp(ix,   iy,   iz-1, n, 2u)) * inv2dx;

    // ── Sponge layer damping ──────────────────────────────────────────────────
    // Minimum distance from any domain face (in cells).
    let sc    = i32(params.sponge_cells);
    let dist  = min(min(ix, n-1-ix), min(min(iy, n-1-iy), min(iz, n-1-iz)));
    var sponge_damp = 0.0f32;
    if dist < sc && params.sigma_max > 0.0 {
        let t   = 1.0 - f32(dist) / f32(sc);  // 1 at wall → 0 at interior
        sponge_damp = params.sigma_max * t * t;  // quadratic profile
    }

    // ── φ velocity update ─────────────────────────────────────────────────────
    let acc_phi = C2 * (lap_phi - params.gamma * div_av);
    phi_vel[flat] = (phi_vel[flat] + dt * acc_phi) * (1.0 - sponge_damp * dt);

    // ── Laplacian of A (each component) + ∇(φ_vel) ───────────────────────────
    // For component k: ∂²Ak/∂t² = c²∇²Ak − c²·∂(φ_vel)/∂xk

    // Precompute Laplacian for each component.
    let a_base = flat * 4u;

    let lap_ax = (
        a_comp(ix+1, iy,   iz,   n, 0u) + a_comp(ix-1, iy,   iz,   n, 0u) +
        a_comp(ix,   iy+1, iz,   n, 0u) + a_comp(ix,   iy-1, iz,   n, 0u) +
        a_comp(ix,   iy,   iz+1, n, 0u) + a_comp(ix,   iy,   iz-1, n, 0u)
        - 6.0 * a_vec[a_base]
    ) * inv_dx2;

    let lap_ay = (
        a_comp(ix+1, iy,   iz,   n, 1u) + a_comp(ix-1, iy,   iz,   n, 1u) +
        a_comp(ix,   iy+1, iz,   n, 1u) + a_comp(ix,   iy-1, iz,   n, 1u) +
        a_comp(ix,   iy,   iz+1, n, 1u) + a_comp(ix,   iy,   iz-1, n, 1u)
        - 6.0 * a_vec[a_base + 1u]
    ) * inv_dx2;

    let lap_az = (
        a_comp(ix+1, iy,   iz,   n, 2u) + a_comp(ix-1, iy,   iz,   n, 2u) +
        a_comp(ix,   iy+1, iz,   n, 2u) + a_comp(ix,   iy-1, iz,   n, 2u) +
        a_comp(ix,   iy,   iz+1, n, 2u) + a_comp(ix,   iy,   iz-1, n, 2u)
        - 6.0 * a_vec[a_base + 2u]
    ) * inv_dx2;

    // Gradient of φ_vel
    let gpv_x = (pv_at(ix+1, iy,   iz,   n) - pv_at(ix-1, iy,   iz,   n)) * inv2dx;
    let gpv_y = (pv_at(ix,   iy+1, iz,   n) - pv_at(ix,   iy-1, iz,   n)) * inv2dx;
    let gpv_z = (pv_at(ix,   iy,   iz+1, n) - pv_at(ix,   iy,   iz-1, n)) * inv2dx;

    // ── A velocity update ─────────────────────────────────────────────────────
    // EED: acc_A = c²∇²A − γ·c²·∇(∂φ/∂t);  also apply sponge damping.
    let damp = 1.0 - sponge_damp * dt;
    a_vel[a_base]      = (a_vel[a_base]      + dt * C2 * (lap_ax - params.gamma * gpv_x)) * damp;
    a_vel[a_base + 1u] = (a_vel[a_base + 1u] + dt * C2 * (lap_ay - params.gamma * gpv_y)) * damp;
    a_vel[a_base + 2u] = (a_vel[a_base + 2u] + dt * C2 * (lap_az - params.gamma * gpv_z)) * damp;
    // a_vel[a_base + 3u] stays 0 (padding component)
}

// ── Pass 2: position update ───────────────────────────────────────────────────
//
// Reads: phi, a_vec, phi_vel (updated), a_vel (updated)
// Writes: phi, a_vec
//
// This pass runs AFTER vel_step in the same command buffer, so it reads the
// freshly-written velocities from pass 1.

@compute @workgroup_size(256)
fn pos_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n    = i32(params.n1);
    let flat = gid.x;
    if flat >= u32(n * n * n) { return; }

    let ix = i32(flat) % n;
    let iy = (i32(flat) / n) % n;
    let iz = i32(flat) / (n * n);

    // Boundary stays fixed at zero (Dirichlet).
    if ix == 0 || ix == n - 1 || iy == 0 || iy == n - 1 || iz == 0 || iz == n - 1 {
        phi[flat] = 0.0;
        let bbase = flat * 4u;
        a_vec[bbase] = 0.0; a_vec[bbase+1u] = 0.0; a_vec[bbase+2u] = 0.0; a_vec[bbase+3u] = 0.0;
        return;
    }

    let dt = params.dt;

    // Drift positions using updated velocities.
    phi[flat] += dt * phi_vel[flat];

    let a_base = flat * 4u;
    a_vec[a_base]      += dt * a_vel[a_base];
    a_vec[a_base + 1u] += dt * a_vel[a_base + 1u];
    a_vec[a_base + 2u] += dt * a_vel[a_base + 2u];
    // a_vec[a_base + 3u] stays 0 (padding)
}
