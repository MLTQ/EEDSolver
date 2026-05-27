//! Analytical Biot-Savart engine.
//!
//! Each coil entity is described as an ordered list of straight wire segments.
//! The vector potential at grid point r is:
//!
//!   A(r) = (μ₀/4π) Σ_segments  I · dl / |r − r'|
//!
//! This is computed on the GPU: one thread per grid point, loop over segments.
//! The computation is embarrassingly parallel with no data dependencies.
//!
//! # Advantages over FEM volume meshing
//! - Exact for thin-wire limit (no discretisation error)
//! - Motion-compatible: new position → recompute, no remeshing
//! - Multiple entities: linear superposition, no mesh interaction
//!
//! # TODO (Phase 1)
//! - WireEntity data structure
//! - WGSL biot.wgsl kernel
//! - Coil geometry builders (solenoid, toroid, toroid_poloidal, flat_spiral, rodin)
//! - Multi-entity GPU dispatch

use crate::types::CoilEntity;

/// A single straight wire segment carrying current `current_a` [A].
/// Points are in metres.
#[derive(Debug, Clone, bytemuck::Pod, bytemuck::Zeroable, Copy)]
#[repr(C)]
pub struct WireSegment {
    pub start:     [f32; 3],
    pub _pad0:     f32,
    pub end:       [f32; 3],
    pub _pad1:     f32,
    /// Current magnitude [A].
    pub current_a: f32,
    pub _pad2:     [f32; 3],
}

/// Convert a CoilEntity into an ordered list of wire segments.
///
/// # TODO (Phase 1)
/// Currently returns an empty list — implement each coil geometry builder.
pub fn entity_to_segments(_entity: &CoilEntity) -> Vec<WireSegment> {
    // TODO Phase 1: implement solenoid, toroid, toroid_poloidal, flat_spiral, rodin
    vec![]
}
