// ════════════════════════════════════════════════════════════════════════════
// biot.wgsl — GPU Biot-Savart kernel
// ════════════════════════════════════════════════════════════════════════════
//
// Computes the magnetic vector potential A(r) at every Yee-grid vertex
// due to a collection of straight wire segments carrying steady current.
//
// Physical model (SI units):
//   A(r) = (μ₀/4π) ∑_segs  I ∫ dl / |r − r′|
//
// Numerical integration:
//   Each segment is subdivided into `seg.ndiv` equal pieces.
//   The midpoint of each piece is the source point r′.
//   The integrand dl / |r − r′| is summed over all sub-elements.
//
// Dispatch: one thread per Yee vertex.  Vertices are indexed in z-major order:
//   idx = ix + iy·n1 + iz·n1²
// where n1 = cells_per_axis + 1 (vertices per axis).

const MU0_OVER_4PI: f32 = 1e-7;   // μ₀ / 4π  [T·m/A]

// ── Bindings ─────────────────────────────────────────────────────────────────

struct WireSegment {
    start:   vec3<f32>,  // bytes  0-11
    current: f32,        // bytes 12-15
    end:     vec3<f32>,  // bytes 16-27
    ndiv:    u32,        // bytes 28-31
}

struct GridParams {
    origin:   vec3<f32>,  // bytes  0-11: world position of vertex (0,0,0) [m]
    dx:       f32,        // bytes 12-15: cell (and vertex) spacing [m]
    n1:       u32,        // bytes 16-19: vertices per axis (n_cells + 1)
    num_segs: u32,        // bytes 20-23: number of wire segments
    _pad:     vec2<u32>,  // bytes 24-31: alignment padding
}

@group(0) @binding(0) var<storage, read>       segments: array<WireSegment>;
@group(0) @binding(1) var<storage, read_write> A_buf:    array<f32>;  // stride 4: [Ax, Ay, Az, 0]
@group(0) @binding(2) var<uniform>             params:   GridParams;

// ── Kernel ───────────────────────────────────────────────────────────────────

@compute @workgroup_size(256)
fn biot_savart(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n1  = params.n1;
    let tot = n1 * n1 * n1;
    if idx >= tot { return; }

    // Recover (ix, iy, iz) from flat z-major index.
    let iz = idx / (n1 * n1);
    let iy = (idx / n1) % n1;
    let ix = idx % n1;

    // World position of this vertex.
    let p = params.origin + params.dx * vec3<f32>(f32(ix), f32(iy), f32(iz));

    var A = vec3<f32>(0.0, 0.0, 0.0);

    for (var s = 0u; s < params.num_segs; s++) {
        let seg  = segments[s];
        let I    = seg.current;
        let ndiv = seg.ndiv;

        // dl vector for one sub-element.
        let dl = (seg.end - seg.start) / f32(ndiv);

        for (var k = 0u; k < ndiv; k++) {
            // Midpoint of sub-element k.
            let t  = (f32(k) + 0.5) / f32(ndiv);
            let rp = mix(seg.start, seg.end, t);

            let r    = p - rp;
            let dist = length(r);

            // Skip degenerate points (inside wire — within 10 µm).
            if dist > 1e-5 {
                // dA = (μ₀/4π) I dl / |r − r′|
                A += (MU0_OVER_4PI * I / dist) * dl;
            }
        }
    }

    // Store with stride-4 layout: [Ax, Ay, Az, 0.0].
    let base      = idx * 4u;
    A_buf[base]     = A.x;
    A_buf[base + 1u] = A.y;
    A_buf[base + 2u] = A.z;
    A_buf[base + 3u] = 0.0;
}
