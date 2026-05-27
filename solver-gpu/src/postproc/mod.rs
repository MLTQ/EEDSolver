//! Post-processing: field extraction from GPU buffers to `SliceData` and `VolumeData`.
//!
//! # Phase 1 (implemented)
//! - 2D axis-aligned slices of scalar and vector-magnitude fields
//! - Field min/max computation
//!
//! # Phase 2 (implemented)
//! - 3D volume extraction normalised to [0,1] for ray-marching
//!
//! # TODO (Phase 5)
//! - Modified Poynting vector |P| = |E×B − EC|
//! - Magnetic helicity ∫ A·B d³x
//! - Holonomy path integrals ∮ A·dl

use crate::{
    context::GpuContext,
    error::SolverError,
    grid::{GpuGridState, YeeGrid},
    types::{
        FieldMaximum, FieldName, HolonomyPath, HolonomyResult,
        SliceAxis, SliceData, SliceRequest, VolumeData,
    },
};

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Extract all requested slices from the GPU grid state.
pub fn extract_slices(
    ctx:      &GpuContext,
    gstate:   &GpuGridState,
    grid:     &YeeGrid,
    requests: &[SliceRequest],
) -> Result<Vec<SliceData>, SolverError> {
    requests.iter().map(|req| extract_slice(ctx, gstate, grid, req)).collect()
}

/// Compute field maxima for all populated fields.
/// Phase 1/2/5: |B|, |A|, C, φ, Φ_g, |P|, u.
pub fn compute_maxima(
    ctx:    &GpuContext,
    gstate: &GpuGridState,
    grid:   &YeeGrid,
) -> Result<Vec<FieldMaximum>, SolverError> {
    let n1 = gstate.n1;
    let dx = grid.dx as f32;
    let r  = grid.extent as f32;

    let b_data       = gstate.readback(ctx, &gstate.b_vec,       gstate.vec_len())?;
    let a_data       = gstate.readback(ctx, &gstate.a_vec,       gstate.vec_len())?;
    let c_data       = gstate.readback(ctx, &gstate.c_fld,       gstate.scalar_len())?;
    let phi_data     = gstate.readback(ctx, &gstate.phi,         gstate.scalar_len())?;
    let phi_g_data   = gstate.readback(ctx, &gstate.phi_g,       gstate.scalar_len())?;
    let pmag_data    = gstate.readback(ctx, &gstate.poynting_mag, gstate.scalar_len())?;
    let udens_data   = gstate.readback(ctx, &gstate.energy_dens,  gstate.scalar_len())?;

    let mut maxima = Vec::new();

    // Helper: max of |scalar|
    let scalar_max = |data: &[f32]| -> (f32, usize) {
        data.iter().copied().enumerate()
            .map(|(i, v)| (v.abs(), i))
            .fold((0.0_f32, 0usize), |(m, mi), (v, i)| if v > m { (v, i) } else { (m, mi) })
    };

    // Helper: max of vec3 magnitude (stride-4 buffer)
    let vec_max = |data: &[f32]| -> (f32, usize) {
        data.chunks_exact(4).enumerate()
            .map(|(i, c)| ((c[0]*c[0]+c[1]*c[1]+c[2]*c[2]).sqrt(), i))
            .fold((0.0_f32, 0usize), |(m, mi), (v, i)| if v > m { (v, i) } else { (m, mi) })
    };

    let (b_max, b_idx) = vec_max(&b_data);
    if b_max > 0.0 {
        maxima.push(FieldMaximum {
            field:        FieldName::BMagnitude,
            max_value:    b_max as f64,
            max_location: index_to_world(b_idx, n1, dx, r),
        });
    }

    let (a_max, a_idx) = vec_max(&a_data);
    if a_max > 0.0 {
        maxima.push(FieldMaximum {
            field:        FieldName::AMagnitude,
            max_value:    a_max as f64,
            max_location: index_to_world(a_idx, n1, dx, r),
        });
    }

    let (c_max, c_idx) = scalar_max(&c_data);
    if c_max > 0.0 {
        maxima.push(FieldMaximum {
            field:        FieldName::CField,
            max_value:    c_max as f64,
            max_location: index_to_world(c_idx, n1, dx, r),
        });
    }

    let (phi_max, phi_idx) = scalar_max(&phi_data);
    if phi_max > 0.0 {
        maxima.push(FieldMaximum {
            field:        FieldName::Phi,
            max_value:    phi_max as f64,
            max_location: index_to_world(phi_idx, n1, dx, r),
        });
    }

    let (phi_g_max, phi_g_idx) = scalar_max(&phi_g_data);
    if phi_g_max > 0.0 {
        maxima.push(FieldMaximum {
            field:        FieldName::PhiG,
            max_value:    phi_g_max as f64,
            max_location: index_to_world(phi_g_idx, n1, dx, r),
        });
    }

    let (pmag_max, pmag_idx) = scalar_max(&pmag_data);
    if pmag_max > 0.0 {
        maxima.push(FieldMaximum {
            field:        FieldName::PoyntingMag,
            max_value:    pmag_max as f64,
            max_location: index_to_world(pmag_idx, n1, dx, r),
        });
    }

    let (udens_max, udens_idx) = scalar_max(&udens_data);
    if udens_max > 0.0 {
        maxima.push(FieldMaximum {
            field:        FieldName::EnergyDensity,
            max_value:    udens_max as f64,
            max_location: index_to_world(udens_idx, n1, dx, r),
        });
    }

    Ok(maxima)
}

/// Extract a 3-D scalar volume normalised to [0, 1] for ray-marching.
///
/// The volume covers `[-extent, +extent]³` and has `(n+1)³` voxels.
/// Phase 1 supports `BMagnitude`, `AMagnitude`, and `CField`.
/// Other fields return zeros (will be filled in as phases complete).
pub fn extract_volume(
    ctx:    &GpuContext,
    gstate: &GpuGridState,
    grid:   &YeeGrid,
    field:  &FieldName,
) -> Result<VolumeData, SolverError> {
    let n1    = gstate.n1;
    let total = gstate.scalar_len(); // n1³
    let r     = grid.extent;

    let raw: Vec<f32> = match field {
        FieldName::BMagnitude => {
            let b = gstate.readback(ctx, &gstate.b_vec, gstate.vec_len())?;
            b.chunks_exact(4).map(|c| (c[0]*c[0]+c[1]*c[1]+c[2]*c[2]).sqrt()).collect()
        }
        FieldName::AMagnitude => {
            let a = gstate.readback(ctx, &gstate.a_vec, gstate.vec_len())?;
            a.chunks_exact(4).map(|c| (c[0]*c[0]+c[1]*c[1]+c[2]*c[2]).sqrt()).collect()
        }
        FieldName::CField => {
            gstate.readback(ctx, &gstate.c_fld, total)?
        }
        FieldName::Phi => {
            gstate.readback(ctx, &gstate.phi, total)?
        }
        FieldName::PhiG => {
            gstate.readback(ctx, &gstate.phi_g, total)?
        }
        FieldName::PoyntingMag => {
            gstate.readback(ctx, &gstate.poynting_mag, total)?
        }
        FieldName::EnergyDensity => {
            gstate.readback(ctx, &gstate.energy_dens, total)?
        }
        _ => vec![0.0f32; total],
    };

    // Normalise to [0, 1].  For C field (signed) use absolute value before norm.
    let needs_abs = matches!(field, FieldName::CField);
    let vals: Vec<f32> = if needs_abs { raw.iter().map(|v| v.abs()).collect() } else { raw };

    let max_val = vals.iter().cloned().fold(0.0_f32, f32::max);
    let min_val = vals.iter().cloned().fold(f32::MAX, f32::min);

    let normalised: Vec<f32> = if max_val > 0.0 {
        vals.iter().map(|v| (v - min_val) / (max_val - min_val)).collect()
    } else {
        vec![0.0f32; total]
    };

    let _n = n1 as usize;
    Ok(VolumeData {
        field:     field.clone(),
        shape:     [n1, n1, n1],
        data:      normalised,
        x_range:   [-r, r],
        y_range:   [-r, r],
        z_range:   [-r, r],
        field_min: min_val as f64,
        field_max: max_val as f64,
    })
}

/// Compute holonomy ∮ A·dl for each requested closed path.
///
/// Uses CPU-side trilinear interpolation of the `a_vec` grid.  The path
/// is discretised into `N_SEG` equal-length chords; each chord contributes
/// A(midpoint) · Δr.  This is exact to O(|Δr|²) (midpoint rule).
///
/// Returns value = 0.0 for any path that lies outside the simulation domain.
pub fn compute_holonomies(
    ctx:    &GpuContext,
    gstate: &GpuGridState,
    grid:   &YeeGrid,
    paths:  &[HolonomyPath],
) -> Vec<HolonomyResult> {
    if paths.is_empty() {
        return vec![];
    }

    // Read A once; reuse across all path integrals.
    let a_data = match gstate.readback(ctx, &gstate.a_vec, gstate.vec_len()) {
        Ok(d)  => d,
        Err(e) => {
            log::warn!("Holonomy readback failed: {e}");
            return paths.iter()
                .map(|p| HolonomyResult { path: p.clone(), value: 0.0 })
                .collect();
        }
    };

    let n1     = gstate.n1 as usize;
    let dx     = grid.dx as f32;
    let extent = grid.extent as f32;

    paths.iter()
        .map(|path| {
            let value = path_integral(&a_data, n1, dx, extent, path);
            HolonomyResult { path: path.clone(), value: value as f64 }
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-slice extraction
// ─────────────────────────────────────────────────────────────────────────────

fn extract_slice(
    ctx:    &GpuContext,
    gstate: &GpuGridState,
    grid:   &YeeGrid,
    req:    &SliceRequest,
) -> Result<SliceData, SolverError> {
    let n1 = gstate.n1;
    let r  = grid.extent as f32;

    // Pick the nearest vertex layer for the requested normalised position.
    let layer = (req.position.clamp(0.0, 1.0) * (n1 - 1) as f64).round() as u32;

    // Build the 2D scalar field for this slice.
    let data: Vec<f32> = match req.field {
        FieldName::BMagnitude => {
            let b = gstate.readback(ctx, &gstate.b_vec, gstate.vec_len())?;
            slice_magnitude(&b, n1, req.axis, layer)
        }
        FieldName::AMagnitude => {
            let a = gstate.readback(ctx, &gstate.a_vec, gstate.vec_len())?;
            slice_magnitude(&a, n1, req.axis, layer)
        }
        FieldName::CField => {
            let c = gstate.readback(ctx, &gstate.c_fld, gstate.scalar_len())?;
            slice_scalar(&c, n1, req.axis, layer)
        }
        FieldName::Phi => {
            let phi = gstate.readback(ctx, &gstate.phi, gstate.scalar_len())?;
            slice_scalar(&phi, n1, req.axis, layer)
        }
        FieldName::PhiG => {
            let pg = gstate.readback(ctx, &gstate.phi_g, gstate.scalar_len())?;
            slice_scalar(&pg, n1, req.axis, layer)
        }
        FieldName::PoyntingMag => {
            let pm = gstate.readback(ctx, &gstate.poynting_mag, gstate.scalar_len())?;
            slice_scalar(&pm, n1, req.axis, layer)
        }
        FieldName::EnergyDensity => {
            let ud = gstate.readback(ctx, &gstate.energy_dens, gstate.scalar_len())?;
            slice_scalar(&ud, n1, req.axis, layer)
        }
        // Phase 5+ fields — return zeros.
        _ => vec![0.0f32; (n1 * n1) as usize],
    };

    let (field_min, field_max) = data.iter().copied().fold(
        (f32::MAX, f32::MIN),
        |(mn, mx), v| (mn.min(v), mx.max(v)),
    );
    let field_min = if field_min == f32::MAX { 0.0 } else { field_min };
    let field_max = if field_max == f32::MIN { 0.0 } else { field_max };

    // Physical extent of the slice (the two axes perpendicular to the slice normal).
    let (x_range, y_range) = match req.axis {
        SliceAxis::X => ([-r as f64, r as f64], [-r as f64, r as f64]), // Y × Z
        SliceAxis::Y => ([-r as f64, r as f64], [-r as f64, r as f64]), // X × Z
        SliceAxis::Z => ([-r as f64, r as f64], [-r as f64, r as f64]), // X × Y
    };

    Ok(SliceData {
        axis:      req.axis.clone(),
        position:  req.position,
        field:     req.field.clone(),
        shape:     [n1, n1],
        data,
        x_range,
        y_range,
        field_min: field_min as f64,
        field_max: field_max as f64,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Slice helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Extract a 2D slice of |vec3| magnitude from a stride-4 buffer (Vx,Vy,Vz,0).
fn slice_magnitude(
    buf:   &[f32],
    n1:    u32,
    axis:  SliceAxis,
    layer: u32,
) -> Vec<f32> {
    let n = n1 as usize;
    let l = layer as usize;

    let mag = |base: usize| -> f32 {
        let (x, y, z) = (buf[base], buf[base + 1], buf[base + 2]);
        (x*x + y*y + z*z).sqrt()
    };

    match axis {
        SliceAxis::Z => {
            // Slice at iz = layer: iterate over (ix, iy).
            let iz = l;
            (0..n).flat_map(|iy| {
                (0..n).map(move |ix| {
                    let base = (ix + iy * n + iz * n * n) * 4;
                    mag(base)
                })
            }).collect()
        }
        SliceAxis::Y => {
            let iy = l;
            (0..n).flat_map(|iz| {
                (0..n).map(move |ix| {
                    let base = (ix + iy * n + iz * n * n) * 4;
                    mag(base)
                })
            }).collect()
        }
        SliceAxis::X => {
            let ix = l;
            (0..n).flat_map(|iz| {
                (0..n).map(move |iy| {
                    let base = (ix + iy * n + iz * n * n) * 4;
                    mag(base)
                })
            }).collect()
        }
    }
}

/// Extract a 2D slice of a scalar field (stride 1).
fn slice_scalar(
    buf:   &[f32],
    n1:    u32,
    axis:  SliceAxis,
    layer: u32,
) -> Vec<f32> {
    let n = n1 as usize;
    let l = layer as usize;

    match axis {
        SliceAxis::Z => {
            let iz = l;
            (0..n).flat_map(|iy| {
                (0..n).map(move |ix| buf[ix + iy * n + iz * n * n])
            }).collect()
        }
        SliceAxis::Y => {
            let iy = l;
            (0..n).flat_map(|iz| {
                (0..n).map(move |ix| buf[ix + iy * n + iz * n * n])
            }).collect()
        }
        SliceAxis::X => {
            let ix = l;
            (0..n).flat_map(|iz| {
                (0..n).map(move |iy| buf[ix + iy * n + iz * n * n])
            }).collect()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn index_to_world(flat_idx: usize, n1: u32, dx: f32, extent: f32) -> [f64; 3] {
    let n = n1 as usize;
    let iz = flat_idx / (n * n);
    let iy = (flat_idx / n) % n;
    let ix = flat_idx % n;
    let world = |i: usize| (-extent + dx * i as f32) as f64;
    [world(ix), world(iy), world(iz)]
}

// ─────────────────────────────────────────────────────────────────────────────
// Holonomy helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Number of quadrature points around a closed path.
const N_SEG: usize = 512;

/// Trilinearly interpolate the vector potential A at world position (x, y, z).
///
/// `a_data` is the stride-4 buffer [Ax, Ay, Az, 0] in z-major order.
/// Returns [0, 0, 0] for points outside the domain.
fn interp_a(a_data: &[f32], n1: usize, dx: f32, extent: f32, x: f32, y: f32, z: f32) -> [f32; 3] {
    // Convert world → fractional grid coordinates
    let gx = (x + extent) / dx;
    let gy = (y + extent) / dx;
    let gz = (z + extent) / dx;

    // Bounds check — return zero for out-of-domain queries.
    let n = (n1 - 1) as f32;
    if gx < 0.0 || gx > n || gy < 0.0 || gy > n || gz < 0.0 || gz > n {
        return [0.0; 3];
    }

    let i0x = (gx as usize).min(n1 - 2);
    let i0y = (gy as usize).min(n1 - 2);
    let i0z = (gz as usize).min(n1 - 2);
    let i1x = i0x + 1;
    let i1y = i0y + 1;
    let i1z = i0z + 1;

    let fx = (gx - i0x as f32).clamp(0.0, 1.0);
    let fy = (gy - i0y as f32).clamp(0.0, 1.0);
    let fz = (gz - i0z as f32).clamp(0.0, 1.0);

    let a = |ix: usize, iy: usize, iz: usize| -> [f32; 3] {
        let base = (ix + iy * n1 + iz * n1 * n1) * 4;
        [a_data[base], a_data[base + 1], a_data[base + 2]]
    };

    let w000 = (1.0-fx)*(1.0-fy)*(1.0-fz);
    let w100 = fx      *(1.0-fy)*(1.0-fz);
    let w010 = (1.0-fx)*fy      *(1.0-fz);
    let w110 = fx      *fy      *(1.0-fz);
    let w001 = (1.0-fx)*(1.0-fy)*fz;
    let w101 = fx      *(1.0-fy)*fz;
    let w011 = (1.0-fx)*fy      *fz;
    let w111 = fx      *fy      *fz;

    let c = [a(i0x,i0y,i0z), a(i1x,i0y,i0z), a(i0x,i1y,i0z), a(i1x,i1y,i0z),
             a(i0x,i0y,i1z), a(i1x,i0y,i1z), a(i0x,i1y,i1z), a(i1x,i1y,i1z)];

    let mut out = [0.0f32; 3];
    for k in 0..3 {
        out[k] = c[0][k]*w000 + c[1][k]*w100 + c[2][k]*w010 + c[3][k]*w110
               + c[4][k]*w001 + c[5][k]*w101 + c[6][k]*w011 + c[7][k]*w111;
    }
    out
}

/// Compute ∮ A·dl for a closed path using the midpoint rule.
///
/// `points`: N+1 points where the last equals the first (closed).
/// The tangent at each segment is `points[i+1] - points[i]`.
fn line_integral(
    a_data: &[f32],
    n1:     usize,
    dx:     f32,
    extent: f32,
    points: &[[f32; 3]],
) -> f32 {
    let n = points.len() - 1;
    let mut sum = 0.0f32;
    for i in 0..n {
        let p = points[i];
        let q = points[i + 1];
        // Midpoint of segment
        let mx = 0.5 * (p[0] + q[0]);
        let my = 0.5 * (p[1] + q[1]);
        let mz = 0.5 * (p[2] + q[2]);
        // Tangent vector Δr
        let dr = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
        // A at midpoint
        let a = interp_a(a_data, n1, dx, extent, mx, my, mz);
        // A · Δr
        sum += a[0]*dr[0] + a[1]*dr[1] + a[2]*dr[2];
    }
    sum
}

/// Build path points for a given HolonomyPath and call `line_integral`.
fn path_integral(
    a_data: &[f32],
    n1:     usize,
    dx:     f32,
    extent: f32,
    path:   &HolonomyPath,
) -> f32 {
    use std::f32::consts::TAU;

    // Build N_SEG+1 points (last == first).
    let points: Vec<[f32; 3]> = match path {
        HolonomyPath::ZCircle { z_m, radius_m } => {
            let z = *z_m as f32;
            let r = *radius_m as f32;
            (0..=N_SEG).map(|i| {
                let theta = TAU * i as f32 / N_SEG as f32;
                [r * theta.cos(), r * theta.sin(), z]
            }).collect()
        }
        HolonomyPath::ToroidalLoop { centre_m, major_radius_m } => {
            // Toroidal loop: circle in the XY plane at the torus centre
            // (goes around the hole of the torus).
            let cx = centre_m[0] as f32;
            let cy = centre_m[1] as f32;
            let cz = centre_m[2] as f32;
            let r  = *major_radius_m as f32;
            (0..=N_SEG).map(|i| {
                let theta = TAU * i as f32 / N_SEG as f32;
                [cx + r * theta.cos(), cy + r * theta.sin(), cz]
            }).collect()
        }
        HolonomyPath::PoloidalLoop { centre_m, major_radius_m, minor_radius_m } => {
            // Poloidal loop: small circle in the meridional plane (θ=0)
            // of the torus.  Goes around the tube cross-section.
            let cx = centre_m[0] as f32;
            let cz = centre_m[2] as f32;
            let big_r   = *major_radius_m as f32;
            let small_r = *minor_radius_m as f32;
            let cy = centre_m[1] as f32;
            (0..=N_SEG).map(|i| {
                let phi = TAU * i as f32 / N_SEG as f32;
                // Meridional plane at azimuth=0 → X axis.
                // Point = centre + (big_R + small_r·cos(φ))·x̂ + small_r·sin(φ)·ẑ
                [cx + (big_r + small_r * phi.cos()), cy, cz + small_r * phi.sin()]
            }).collect()
        }
    };

    line_integral(a_data, n1, dx, extent, &points)
}
