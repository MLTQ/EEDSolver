// ─────────────────────────────────────────────────────────────────────────────
// li_torr_source.wgsl — Li-Torr gravitomagnetic London moment source.
//
// For a rotating superconductor, the Cooper-pair condensate pins the
// gravitomagnetic vector potential A_g analogously to how the EM London
// equation pins A.  The result (Wilhelm 2026 Eq. 23, Li & Torr 1991) is a
// uniform gravitomagnetic field inside the body:
//
//     B_g = −(2 m_e / e) · ω
//
// This kernel imposes the corresponding A_g pattern algebraically:
//
//     A_g(r) = ½ · B_g × (r − r_c)
//            = −(m_e/e) · (ω × (r − r_c))
//
// where r_c is the entity centre.  Curl gives ∇×A_g = −(2 m_e/e)·ω uniformly
// inside (since ∇×(ω × r) = 2ω for constant ω).
//
// # Volume model
// The entity volume is approximated as a sphere of radius `radius` centred
// at `center`.  Vertices outside that sphere are left untouched (A_g
// preserves whatever the κ_G coupling or other sources produced).
//
// # Overwrite semantics
// Inside the sphere, A_g is overwritten — the condensate physically pins it.
// Overlapping superconducting entities are unphysical; last-write-wins is
// the chosen behaviour for that edge case.
//
// # Bindings
//   0  a_g_vec   storage rw    n1³ × 4·f32   [Agx, Agy, Agz, 0]
//   1  entities  storage read  array<LiTorrEntity>
//   2  params    uniform       GridParams (origin, dx, n1, ...)
// ─────────────────────────────────────────────────────────────────────────────

// Packed as 2 × vec4 (32 bytes) to dodge WGSL's vec3 alignment padding.
//   center_radius = (cx, cy, cz, radius)
//   omega_pad     = (ωx, ωy, ωz, 0)
struct LiTorrEntity {
    center_radius: vec4<f32>,
    omega_pad:     vec4<f32>,
}

struct GridParams {
    origin:   vec3<f32>,
    dx:       f32,
    n1:       u32,
    num_segs: u32,       // unused here; shared uniform layout with biot/derive
    _pad:     vec2<u32>,
}

@group(0) @binding(0) var<storage, read_write> a_g_vec:  array<f32>;
@group(0) @binding(1) var<storage, read>       entities: array<LiTorrEntity>;
@group(0) @binding(2) var<uniform>             params:   GridParams;

// m_e / e in the solver's GEM unit convention (A_g [m/s], B_g [1/s]).
// 2·m_e/e ≈ 1.1374×10⁻¹¹ per Wilhelm 2026 §4.10 and types.rs::GemParams docs.
const M_OVER_E: f32 = 5.6857e-12;

@compute @workgroup_size(256)
fn li_torr(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    let n1   = params.n1;
    if flat >= n1 * n1 * n1 { return; }

    // Vertex world-space position.
    let ix = flat % n1;
    let iy = (flat / n1) % n1;
    let iz = flat / (n1 * n1);
    let r  = params.origin + vec3<f32>(f32(ix), f32(iy), f32(iz)) * params.dx;

    // Accumulate contributions from any SC entity containing this vertex.
    let n_e = arrayLength(&entities);
    var ag = vec3<f32>(0.0, 0.0, 0.0);
    var inside = false;
    for (var i: u32 = 0u; i < n_e; i = i + 1u) {
        let e        = entities[i];
        let center   = e.center_radius.xyz;
        let radius   = e.center_radius.w;
        let omega    = e.omega_pad.xyz;
        let d        = r - center;
        if dot(d, d) <= radius * radius {
            ag = ag - M_OVER_E * cross(omega, d);
            inside = true;
        }
    }

    // Only overwrite vertices that are inside at least one SC volume —
    // leaves the rest of the field untouched (e.g. for κ_G superposition).
    if inside {
        let base = flat * 4u;
        a_g_vec[base]      = ag.x;
        a_g_vec[base + 1u] = ag.y;
        a_g_vec[base + 2u] = ag.z;
        a_g_vec[base + 3u] = 0.0;
    }
}
