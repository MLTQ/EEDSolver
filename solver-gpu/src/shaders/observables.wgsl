// ─────────────────────────────────────────────────────────────────────────────
// observables.wgsl — EED modified Poynting vector and energy density
//
// Computes both observables in one GPU pass after the FDTD/static solve.
//
// Fields:
//   E = −∇φ − ∂A/∂t        (electric field; ∂A/∂t = a_vel; zero in static mode)
//   B = b_vec               (curl A, pre-computed by derive.wgsl)
//   C = c_fld               (∇·A + (1/c²)∂φ/∂t, EED scalar DOF)
//
// Outputs:
//   |P| = |E×B − C·E|       modified EED Poynting magnitude [W/m² in effective units]
//   u   = ½(|E|² + |B|² + C²)  modified EED energy density  [J/m³ in effective units]
//
// Note on units:
//   The code uses SI: E [V/m], B [T], C [m⁻¹].  The "natural" Poynting vector
//   is (1/μ₀)·E×B, but we store the unnormalised form E×B for visualisation —
//   the absolute scale is calibrated by the field_max/field_min readback.
//
// Bindings:
//   0  phi         storage read         n1³ × f32
//   1  a_vel       storage read         n1³ × 4·f32   [∂Ax/∂t, ∂Ay/∂t, ∂Az/∂t, 0]
//   2  b_vec       storage read         n1³ × 4·f32   [Bx, By, Bz, 0]
//   3  c_fld       storage read         n1³ × f32
//   4  poynting_mag storage read_write  n1³ × f32     |P|
//   5  energy_dens  storage read_write  n1³ × f32     u
//   6  params      uniform
// ─────────────────────────────────────────────────────────────────────────────

struct ObsParams {
    dx:   f32,
    n1:   u32,
    _pad: vec2<u32>,
}

@group(0) @binding(0) var<storage, read>       phi:          array<f32>;
@group(0) @binding(1) var<storage, read>       a_vel:        array<f32>;  // stride 4
@group(0) @binding(2) var<storage, read>       b_vec:        array<f32>;  // stride 4
@group(0) @binding(3) var<storage, read>       c_fld:        array<f32>;
@group(0) @binding(4) var<storage, read_write> poynting_mag: array<f32>;
@group(0) @binding(5) var<storage, read_write> energy_dens:  array<f32>;
@group(0) @binding(6) var<uniform>             params:       ObsParams;

// ── Index helpers ─────────────────────────────────────────────────────────────

fn phi_at(ix: i32, iy: i32, iz: i32, n: i32) -> f32 {
    let cx = clamp(ix, 0, n - 1);
    let cy = clamp(iy, 0, n - 1);
    let cz = clamp(iz, 0, n - 1);
    return phi[u32(cx + cy * n + cz * n * n)];
}

// ── Main kernel ───────────────────────────────────────────────────────────────

@compute @workgroup_size(256)
fn compute_obs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n    = i32(params.n1);
    let flat = gid.x;
    if flat >= u32(n * n * n) { return; }

    let ix = i32(flat) % n;
    let iy = (i32(flat) / n) % n;
    let iz = i32(flat) / (n * n);

    // Boundary: set to zero (Dirichlet — fields are clamped to 0 at walls).
    if ix == 0 || ix == n - 1 || iy == 0 || iy == n - 1 || iz == 0 || iz == n - 1 {
        poynting_mag[flat] = 0.0;
        energy_dens[flat]  = 0.0;
        return;
    }

    let inv2dx = 0.5 / params.dx;

    // ── E field = −∇φ − a_vel ─────────────────────────────────────────────────
    // Central-difference gradient of φ.
    let gphi_x = (phi_at(ix+1, iy, iz, n) - phi_at(ix-1, iy, iz, n)) * inv2dx;
    let gphi_y = (phi_at(ix, iy+1, iz, n) - phi_at(ix, iy-1, iz, n)) * inv2dx;
    let gphi_z = (phi_at(ix, iy, iz+1, n) - phi_at(ix, iy, iz-1, n)) * inv2dx;

    // a_vel = ∂A/∂t (stride-4 buffer); zero in static mode.
    let av_base = flat * 4u;
    let av_x = a_vel[av_base];
    let av_y = a_vel[av_base + 1u];
    let av_z = a_vel[av_base + 2u];

    let ex = -gphi_x - av_x;
    let ey = -gphi_y - av_y;
    let ez = -gphi_z - av_z;

    // ── B field (pre-computed in b_vec) ───────────────────────────────────────
    let bx = b_vec[av_base];
    let by = b_vec[av_base + 1u];
    let bz = b_vec[av_base + 2u];

    // ── C scalar (EED deleted DOF) ────────────────────────────────────────────
    let c = c_fld[flat];

    // ── Modified Poynting: P = E×B − C·E ─────────────────────────────────────
    // E×B cross product
    let exb_x = ey * bz - ez * by;
    let exb_y = ez * bx - ex * bz;
    let exb_z = ex * by - ey * bx;

    // Subtract C·E (scalar C times vector E)
    let px = exb_x - c * ex;
    let py = exb_y - c * ey;
    let pz = exb_z - c * ez;

    poynting_mag[flat] = sqrt(px*px + py*py + pz*pz);

    // ── Modified energy density: u = ½(|E|² + |B|² + C²) ────────────────────
    let e2 = ex*ex + ey*ey + ez*ez;
    let b2 = bx*bx + by*by + bz*bz;
    energy_dens[flat] = 0.5 * (e2 + b2 + c * c);
}
