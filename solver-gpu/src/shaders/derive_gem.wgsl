// ════════════════════════════════════════════════════════════════════════════
// derive_gem.wgsl — Gravitomagnetic field derivation
// ════════════════════════════════════════════════════════════════════════════
//
// Given the gravitomagnetic vector potential A_g on all Yee vertices,
// compute:
//
//   B_g = ∇ × A_g    gravitomagnetic field [1/s]
//
// Same central-difference curl as `derive.wgsl::derive_fields`, but acts on
// the GEM buffers (a_g_vec → b_g_vec).  E_g derivation (= −∇Φ_g − (1/2c)∂A_g/∂t)
// is a separate concern handled by post-processing if/when needed.

struct GridParams {
    origin:   vec3<f32>,
    dx:       f32,
    n1:       u32,
    num_segs: u32,   // unused here; shared uniform layout with biot/derive
    _pad:     vec2<u32>,
}

@group(0) @binding(0) var<storage, read>       Ag_buf: array<f32>;  // stride 4
@group(0) @binding(1) var<storage, read_write> Bg_buf: array<f32>;  // stride 4
@group(0) @binding(2) var<uniform>             params: GridParams;

fn Ag_at(ix: i32, iy: i32, iz: i32) -> vec3<f32> {
    let n1   = i32(params.n1);
    let cx   = clamp(ix, 0, n1 - 1);
    let cy   = clamp(iy, 0, n1 - 1);
    let cz   = clamp(iz, 0, n1 - 1);
    let base = u32(cx + cy * n1 + cz * n1 * n1) * 4u;
    return vec3<f32>(Ag_buf[base], Ag_buf[base + 1u], Ag_buf[base + 2u]);
}

@compute @workgroup_size(256)
fn derive_gem(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n1  = params.n1;
    if idx >= n1 * n1 * n1 { return; }

    let iz = i32(idx / (n1 * n1));
    let iy = i32((idx / n1) % n1);
    let ix = i32(idx % n1);

    let inv2dx = 0.5 / params.dx;

    let Axp = Ag_at(ix + 1, iy,     iz    );
    let Axm = Ag_at(ix - 1, iy,     iz    );
    let Ayp = Ag_at(ix,     iy + 1, iz    );
    let Aym = Ag_at(ix,     iy - 1, iz    );
    let Azp = Ag_at(ix,     iy,     iz + 1);
    let Azm = Ag_at(ix,     iy,     iz - 1);

    // B_g = curl(A_g):  same expansion as derive.wgsl.
    let Bg = vec3<f32>(
        (Ayp.z - Aym.z - Azp.y + Azm.y) * inv2dx,   // Bgx = ∂Agz/∂y − ∂Agy/∂z
        (Azp.x - Azm.x - Axp.z + Axm.z) * inv2dx,   // Bgy = ∂Agx/∂z − ∂Agz/∂x
        (Axp.y - Axm.y - Ayp.x + Aym.x) * inv2dx,   // Bgz = ∂Agy/∂x − ∂Agx/∂y
    );

    let base = idx * 4u;
    Bg_buf[base]      = Bg.x;
    Bg_buf[base + 1u] = Bg.y;
    Bg_buf[base + 2u] = Bg.z;
    Bg_buf[base + 3u] = 0.0;
}
