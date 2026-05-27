//! Post-processing: field extraction from GPU buffers to `SliceData`.
//!
//! # Phase 1 (implemented)
//! - 2D axis-aligned slices of scalar and vector-magnitude fields
//! - Field min/max computation
//!
//! # TODO (Phase 2+)
//! - 3D volume extraction (ray-marching data)
//! - Modified Poynting vector |P| = |E×B − EC|
//! - Magnetic helicity ∫ A·B d³x
//! - Holonomy path integrals ∮ A·dl

use crate::{
    context::GpuContext,
    error::SolverError,
    grid::{GpuGridState, YeeGrid},
    types::{FieldMaximum, FieldName, HolonomyPath, HolonomyResult, SliceAxis, SliceData, SliceRequest},
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

/// Compute field maxima for the fields we have populated.
/// Phase 1: B magnitude and C field.
pub fn compute_maxima(
    ctx:    &GpuContext,
    gstate: &GpuGridState,
    grid:   &YeeGrid,
) -> Result<Vec<FieldMaximum>, SolverError> {
    let n1 = gstate.n1;
    let dx = grid.dx as f32;
    let r  = grid.extent as f32;

    // Readback B and C
    let b_data = gstate.readback(ctx, &gstate.b_vec, gstate.vec_len())?;
    let c_data = gstate.readback(ctx, &gstate.c_fld, gstate.scalar_len())?;

    let mut maxima = Vec::new();

    // |B| maximum
    let (b_max, b_idx) = b_data
        .chunks_exact(4)
        .enumerate()
        .map(|(i, c)| (((c[0]*c[0] + c[1]*c[1] + c[2]*c[2]).sqrt()), i))
        .fold((0.0_f32, 0usize), |(m, mi), (v, i)| if v > m { (v, i) } else { (m, mi) });

    if b_max > 0.0 {
        maxima.push(FieldMaximum {
            field:        FieldName::BMagnitude,
            max_value:    b_max as f64,
            max_location: index_to_world(b_idx, n1, dx, r),
        });
    }

    // |C| maximum
    let (c_max, c_idx) = c_data
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (v.abs(), i))
        .fold((0.0_f32, 0usize), |(m, mi), (v, i)| if v > m { (v, i) } else { (m, mi) });

    if c_max > 0.0 {
        maxima.push(FieldMaximum {
            field:        FieldName::CField,
            max_value:    c_max as f64,
            max_location: index_to_world(c_idx, n1, dx, r),
        });
    }

    Ok(maxima)
}

/// Stub: holonomy ∮ A·dl (Phase 5).
pub fn compute_holonomies(
    _ctx:    &GpuContext,
    _gstate: &GpuGridState,
    _grid:   &YeeGrid,
    paths:   &[HolonomyPath],
) -> Vec<HolonomyResult> {
    paths.iter()
        .map(|p| HolonomyResult { path: p.clone(), value: 0.0 })
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
        // Phase 2+ fields — return zeros with a note.
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
