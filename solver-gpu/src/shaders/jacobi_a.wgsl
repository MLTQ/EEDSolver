// ─────────────────────────────────────────────────────────────────────────────
// jacobi_a.wgsl — Vector Jacobi iteration for the static EED A-field equation
//
// Solves the Yukawa-corrected vector potential equation:
//
//   (∇² − α²) A = −μ₀J + γ∇φ
//
// where:
//   −μ₀J is encoded via the Biot-Savart solution A_BS (satisfies ∇²A_BS = −μ₀J)
//   γ∇φ  is the EED scalar-to-vector coupling (non-zero when φ ≠ 0)
//
// Strategy — decompose A = A_BS + δA, so:
//
//   (∇² − α²)(A_BS + δA) = −μ₀J + γ∇φ
//   ∇²A_BS − α²A_BS − α²δA + ∇²δA = −μ₀J + γ∇φ
//   −μ₀J − α²A_BS − α²δA + ∇²δA = −μ₀J + γ∇φ
//   (∇² − α²)δA = α²A_BS + γ∇φ
//
// The Jacobi update for each component k of δA:
//
//   δA_new[i,k] = (Σ_nb δA[nb,k] − dx² · rhs_k[i]) / (6 + α²·dx²)
//
// where rhs_k = −(α²·A_BS[i,k] + γ·∂φ/∂x_k)
//
// Bindings:
//   0  da_in    storage read        n1³ × 4·f32  — δA at iteration n (stride 4)
//   1  da_out   storage read_write  n1³ × 4·f32  — δA at iteration n+1
//   2  a_bs     storage read        n1³ × 4·f32  — Biot-Savart A (fixed)
//   3  phi      storage read        n1³ × f32    — scalar potential (fixed)
//   4  params   uniform
// ─────────────────────────────────────────────────────────────────────────────

struct JacobiAParams {
    dx:    f32,
    m2:    f32,    // α²  [1/m²]
    gamma: f32,    // EED γ coupling
    n1:    u32,
}

@group(0) @binding(0) var<storage, read>       da_in:  array<f32>;   // stride 4
@group(0) @binding(1) var<storage, read_write> da_out: array<f32>;   // stride 4
@group(0) @binding(2) var<storage, read>       a_bs:   array<f32>;   // stride 4 (Biot-Savart A)
@group(0) @binding(3) var<storage, read>       phi:    array<f32>;
@group(0) @binding(4) var<uniform>             params: JacobiAParams;

// ── Index helpers ─────────────────────────────────────────────────────────────

fn da_comp(ix: i32, iy: i32, iz: i32, n: i32, c: u32) -> f32 {
    let cx = clamp(ix, 0, n - 1);
    let cy = clamp(iy, 0, n - 1);
    let cz = clamp(iz, 0, n - 1);
    return da_in[u32(cx + cy * n + cz * n * n) * 4u + c];
}

fn phi_at(ix: i32, iy: i32, iz: i32, n: i32) -> f32 {
    let cx = clamp(ix, 0, n - 1);
    let cy = clamp(iy, 0, n - 1);
    let cz = clamp(iz, 0, n - 1);
    return phi[u32(cx + cy * n + cz * n * n)];
}

// ── Jacobi sweep ──────────────────────────────────────────────────────────────

@compute @workgroup_size(256)
fn jacobi_a_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n    = i32(params.n1);
    let flat = gid.x;
    if flat >= u32(n * n * n) { return; }

    let ix = i32(flat) % n;
    let iy = (i32(flat) / n) % n;
    let iz = i32(flat) / (n * n);

    // Dirichlet BC: δA = 0 at domain boundary.
    if ix == 0 || ix == n - 1 || iy == 0 || iy == n - 1 || iz == 0 || iz == n - 1 {
        let b = flat * 4u;
        da_out[b] = 0.0; da_out[b+1u] = 0.0; da_out[b+2u] = 0.0; da_out[b+3u] = 0.0;
        return;
    }

    let dx2 = params.dx * params.dx;
    let denom = 6.0 + params.m2 * dx2;
    let inv2dx = 0.5 / params.dx;

    // ── EED γ∇φ source (central difference) ──────────────────────────────────
    let gphi_x = (phi_at(ix+1, iy, iz, n) - phi_at(ix-1, iy, iz, n)) * inv2dx;
    let gphi_y = (phi_at(ix, iy+1, iz, n) - phi_at(ix, iy-1, iz, n)) * inv2dx;
    let gphi_z = (phi_at(ix, iy, iz+1, n) - phi_at(ix, iy, iz-1, n)) * inv2dx;

    let a_base = flat * 4u;

    // ── Update each vector component independently ────────────────────────────
    for (var c = 0u; c < 3u; c++) {
        // Σ_neighbors δA[nb, c]
        let sum_nb =
            da_comp(ix+1, iy,   iz,   n, c) + da_comp(ix-1, iy,   iz,   n, c) +
            da_comp(ix,   iy+1, iz,   n, c) + da_comp(ix,   iy-1, iz,   n, c) +
            da_comp(ix,   iy,   iz+1, n, c) + da_comp(ix,   iy,   iz-1, n, c);

        // RHS for component c: −(α²·A_BS[c] + γ·∂φ/∂x_c)
        // Note sign: rhs = -(source), source = α²·A_BS + γ∇φ
        let a_bs_c = a_bs[a_base + c];
        let gphi_c = select(gphi_z, select(gphi_y, gphi_x, c == 0u), c != 2u);
        let rhs_c  = -(params.m2 * a_bs_c + params.gamma * gphi_c);

        // Jacobi update: δA_new = (Σ_nb − dx²·rhs) / (6 + α²·dx²)
        da_out[a_base + c] = (sum_nb - dx2 * rhs_c) / denom;
    }
    da_out[a_base + 3u] = 0.0;   // padding
}
