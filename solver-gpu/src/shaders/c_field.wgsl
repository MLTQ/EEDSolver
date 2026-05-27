// ─────────────────────────────────────────────────────────────────────────────
// c_field.wgsl — Time-domain C-field update
//
// Updates the EED scalar field C after each FDTD step:
//
//   C = ∇·A + (1/c²)·∂φ/∂t
//
// This is Maxwell's "deleted" seventh degree of freedom.  Under Lorenz gauge
// C=0 by definition; in EED (potential-primary, no gauge fixing) C is
// dynamical and satisfies □C = ∂μJμ.
//
// In vacuum (∂μJμ=0), C satisfies the free wave equation □C=0.
// Initial conditions C=0 (static Biot-Savart) means C=0 persists for
// closed-loop coils — until a perturbation or open-circuit source injects it.
//
// Bindings:
//   0  a_vec    storage read   n1³ × 4·f32
//   1  phi_vel  storage read   n1³ × f32
//   2  c_fld    storage rw     n1³ × f32
//   3  params   uniform
// ─────────────────────────────────────────────────────────────────────────────

struct CFieldParams {
    dx:   f32,
    inv_c2: f32,   // 1/c² [s²/m²]
    n1:   u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       a_vec:   array<f32>;
@group(0) @binding(1) var<storage, read>       phi_vel: array<f32>;
@group(0) @binding(2) var<storage, read_write> c_fld:   array<f32>;
@group(0) @binding(3) var<uniform>             params:  CFieldParams;

fn a_comp(ix: i32, iy: i32, iz: i32, n: i32, comp: u32) -> f32 {
    let cx = clamp(ix, 0, n - 1);
    let cy = clamp(iy, 0, n - 1);
    let cz = clamp(iz, 0, n - 1);
    return a_vec[u32(cx + cy * n + cz * n * n) * 4u + comp];
}

@compute @workgroup_size(256)
fn update_c(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n    = i32(params.n1);
    let flat = gid.x;
    if flat >= u32(n * n * n) { return; }

    let ix = i32(flat) % n;
    let iy = (i32(flat) / n) % n;
    let iz = i32(flat) / (n * n);

    let inv2dx = 0.5 / params.dx;

    // div(A) = ∂Ax/∂x + ∂Ay/∂y + ∂Az/∂z  (central differences, clamped at boundary)
    let div_a =
        (a_comp(ix+1, iy,   iz,   n, 0u) - a_comp(ix-1, iy,   iz,   n, 0u)) * inv2dx +
        (a_comp(ix,   iy+1, iz,   n, 1u) - a_comp(ix,   iy-1, iz,   n, 1u)) * inv2dx +
        (a_comp(ix,   iy,   iz+1, n, 2u) - a_comp(ix,   iy,   iz-1, n, 2u)) * inv2dx;

    // C = div(A) + (1/c²)·(∂φ/∂t)
    c_fld[flat] = div_a + params.inv_c2 * phi_vel[flat];
}
