// ════════════════════════════════════════════════════════════════════════════
// derive.wgsl — Field derivation kernel
// ════════════════════════════════════════════════════════════════════════════
//
// Given the vector potential A (and scalar φ, zero for static solve) on all
// Yee vertices, compute:
//
//   B = ∇×A            magnetic field [T]
//   C = ∇·A            EED scalar field [1/m]  (static; ∂φ/∂t=0 term absent)
//
// All derivatives use central finite differences.  At boundary vertices the
// index is clamped so the stencil degenerates to a one-sided difference —
// acceptable for display (boundary cells are not physical output targets).
//
// Dispatch: one thread per Yee vertex, same z-major flat index as biot.wgsl.

struct GridParams {
    origin:   vec3<f32>,
    dx:       f32,
    n1:       u32,
    num_segs: u32,   // unused here, same uniform buffer as biot pass
    _pad:     vec2<u32>,
}

@group(0) @binding(0) var<storage, read>       A_buf:  array<f32>;  // stride 4
@group(0) @binding(1) var<storage, read_write> B_buf:  array<f32>;  // stride 4
@group(0) @binding(2) var<storage, read_write> C_buf:  array<f32>;  // stride 1
@group(0) @binding(3) var<uniform>             params: GridParams;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn A_at(ix: i32, iy: i32, iz: i32) -> vec3<f32> {
    let n1   = i32(params.n1);
    let cx   = clamp(ix, 0, n1 - 1);
    let cy   = clamp(iy, 0, n1 - 1);
    let cz   = clamp(iz, 0, n1 - 1);
    let base = u32(cx + cy * n1 + cz * n1 * n1) * 4u;
    return vec3<f32>(A_buf[base], A_buf[base + 1u], A_buf[base + 2u]);
}

// ── Kernel ───────────────────────────────────────────────────────────────────

@compute @workgroup_size(256)
fn derive_fields(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n1  = params.n1;
    if idx >= n1 * n1 * n1 { return; }

    let iz = i32(idx / (n1 * n1));
    let iy = i32((idx / n1) % n1);
    let ix = i32(idx % n1);

    let inv2dx = 0.5 / params.dx;

    // Sample neighbours once.
    let Axp = A_at(ix + 1, iy,     iz    );
    let Axm = A_at(ix - 1, iy,     iz    );
    let Ayp = A_at(ix,     iy + 1, iz    );
    let Aym = A_at(ix,     iy - 1, iz    );
    let Azp = A_at(ix,     iy,     iz + 1);
    let Azm = A_at(ix,     iy,     iz - 1);

    // ── B = curl(A) ──────────────────────────────────────────────────────────
    // Bx = ∂Az/∂y − ∂Ay/∂z
    // By = ∂Ax/∂z − ∂Az/∂x
    // Bz = ∂Ay/∂x − ∂Ax/∂y
    //
    // Example: ∂Az/∂y ≈ (A_at(ix, iy+1, iz).z − A_at(ix, iy-1, iz).z) / 2Δx
    //                  = (Ayp.z − Aym.z) * inv2dx
    let B = vec3<f32>(
        (Ayp.z - Aym.z - Azp.y + Azm.y) * inv2dx,   // Bx = ∂Az/∂y − ∂Ay/∂z
        (Azp.x - Azm.x - Axp.z + Axm.z) * inv2dx,   // By = ∂Ax/∂z − ∂Az/∂x
        (Axp.y - Axm.y - Ayp.x + Aym.x) * inv2dx,   // Bz = ∂Ay/∂x − ∂Ax/∂y
    );

    let bbase = idx * 4u;
    B_buf[bbase]      = B.x;
    B_buf[bbase + 1u] = B.y;
    B_buf[bbase + 2u] = B.z;
    B_buf[bbase + 3u] = 0.0;

    // ── C = div(A) ───────────────────────────────────────────────────────────
    // C = ∂Ax/∂x + ∂Ay/∂y + ∂Az/∂z
    C_buf[idx] = ((Axp.x - Axm.x) + (Ayp.y - Aym.y) + (Azp.z - Azm.z)) * inv2dx;
}
