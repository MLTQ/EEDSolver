//! Biot-Savart sources: wire segment geometry and GPU data layout.
//!
//! # Data flow
//! `CoilEntity` (from the frontend) → `entity_to_segments()` → `Vec<WireSegment>`
//! → uploaded to GPU storage buffer → `biot.wgsl` kernel evaluates A(r) at every
//! Yee-grid vertex.
//!
//! # Wire segment integration
//! The kernel numerically integrates A(r) = (μ₀/4π) ∫ I dl / |r − r′| by
//! subdividing each segment into `ndiv` equal pieces and summing midpoint
//! contributions.  `ndiv` is chosen so each sub-element is ≤ 1 mm long,
//! giving < 0.1% relative error at distances > 1 cm from the wire.

use bytemuck::{Pod, Zeroable};

mod builders;
pub use builders::entity_to_segments;

// ─────────────────────────────────────────────────────────────────────────────
// WireSegment — GPU-ready (Pod, 32 bytes, matches WGSL struct layout)
// ─────────────────────────────────────────────────────────────────────────────
//
// WGSL std430 layout (vec3<f32> align=16, size=12):
//   start:   vec3<f32>  @  0  (bytes 0-11)
//   current: f32        @ 12  (bytes 12-15)
//   end:     vec3<f32>  @ 16  (bytes 16-27)
//   ndiv:    u32        @ 28  (bytes 28-31)
//   struct size: 32 bytes
//
// The Rust repr(C) layout is identical — fields in the same order.

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct WireSegment {
    /// Segment start point [m].
    pub start:   [f32; 3],
    /// Current in the segment [A] (signed; direction = start → end).
    pub current: f32,
    /// Segment end point [m].
    pub end:     [f32; 3],
    /// Sub-divisions for numerical integration (auto-set from length).
    pub ndiv:    u32,
}

/// Target sub-element length for integration accuracy [m].
/// 1 mm gives < 0.1% error at distances > 10 mm from the wire.
const TARGET_ELEM_M: f32 = 1e-3;

impl WireSegment {
    pub fn new(start: [f32; 3], end: [f32; 3], current_a: f32) -> Self {
        let d   = [end[0]-start[0], end[1]-start[1], end[2]-start[2]];
        let len = (d[0]*d[0] + d[1]*d[1] + d[2]*d[2]).sqrt();
        // Clamp: 4 ≤ ndiv ≤ 512 — keeps the inner GPU loop bounded.
        let ndiv = ((len / TARGET_ELEM_M).ceil() as u32).clamp(4, 512);
        Self { start, current: current_a, end, ndiv }
    }
}
