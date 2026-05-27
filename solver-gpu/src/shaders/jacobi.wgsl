// ─────────────────────────────────────────────────────────────────────────────
// jacobi.wgsl — Jacobi iteration for the static EED elliptic equations.
//
// Solves (∇² − m²) u = rhs with homogeneous Dirichlet BC on the domain boundary.
//
// One dispatch = one Jacobi sweep (ping → pong).  Caller ping-pongs
// `u_in` / `u_out` for N iterations.
//
// Discretisation (central differences, uniform spacing dx):
//
//   ∇²u ≈ (u[i+1]+u[i-1]+u[j+1]+u[j-1]+u[k+1]+u[k-1] − 6·u[i,j,k]) / dx²
//
//   Jacobi update:
//   u_new = (Σ6 neighbors − dx²·rhs) / (6 + m²·dx²)
//
// Uses:
//   1. φ equation   : m = α   , rhs = −∇·J (zero for closed-loop coils)
//   2. Ax/Ay/Az eq  : m = α   , rhs = −μ₀Jx / Jy / Jz (each component)
// ─────────────────────────────────────────────────────────────────────────────

struct JacobiParams {
    dx:   f32,    // cell spacing [m]
    m2:   f32,    // m² = α² in EED,  0 for massless / Maxwell
    n1:   u32,    // vertices per axis
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       u_in:   array<f32>;  // u^k
@group(0) @binding(1) var<storage, read_write> u_out:  array<f32>;  // u^{k+1}
@group(0) @binding(2) var<storage, read>       rhs:    array<f32>;  // source term
@group(0) @binding(3) var<uniform>             params: JacobiParams;

fn idx(ix: u32, iy: u32, iz: u32, n: u32) -> u32 {
    return ix + iy * n + iz * n * n;
}

@compute @workgroup_size(256)
fn jacobi_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n    = params.n1;
    let flat = gid.x;
    if flat >= n * n * n { return; }

    let ix = flat % n;
    let iy = (flat / n) % n;
    let iz = flat / (n * n);

    // Dirichlet BC: zero on boundary.
    if ix == 0u || ix == n - 1u ||
       iy == 0u || iy == n - 1u ||
       iz == 0u || iz == n - 1u
    {
        u_out[flat] = 0.0;
        return;
    }

    // Six-point stencil neighbours.
    let xp = u_in[idx(ix + 1u, iy,       iz,       n)];
    let xm = u_in[idx(ix - 1u, iy,       iz,       n)];
    let yp = u_in[idx(ix,       iy + 1u, iz,       n)];
    let ym = u_in[idx(ix,       iy - 1u, iz,       n)];
    let zp = u_in[idx(ix,       iy,       iz + 1u, n)];
    let zm = u_in[idx(ix,       iy,       iz - 1u, n)];

    let sum_nb = xp + xm + yp + ym + zp + zm;
    let dx2    = params.dx * params.dx;
    let denom  = 6.0 + params.m2 * dx2;      // always > 0 for m² ≥ 0

    // u_new = (Σ_nb − dx²·rhs) / (6 + m²·dx²)
    u_out[flat] = (sum_nb - dx2 * rhs[flat]) / denom;
}
