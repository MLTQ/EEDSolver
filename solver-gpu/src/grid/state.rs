//! GPU grid state: buffer allocation, kernel dispatch, and field readback.
//!
//! `GpuGridState` owns all wgpu buffers for one solve and provides methods
//! to run each compute pass and read back results to the CPU.

use std::sync::mpsc;

use bytemuck::{Pod, Zeroable, bytes_of, cast_slice};
use wgpu::util::DeviceExt;

use crate::{biot::WireSegment, context::GpuContext, error::SolverError};
use super::YeeGrid;

// ─────────────────────────────────────────────────────────────────────────────
// GPU uniform struct — must match GridParams in both WGSL shaders
// ─────────────────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GridParamsGpu {
    pub origin:   [f32; 3],  // bytes  0-11
    pub dx:       f32,       // bytes 12-15
    pub n1:       u32,       // bytes 16-19  (vertices per axis = n_cells + 1)
    pub num_segs: u32,       // bytes 20-23
    pub _pad:     [u32; 2],  // bytes 24-31
}

impl GridParamsGpu {
    fn from_grid(grid: &YeeGrid, num_segs: u32) -> Self {
        let r = grid.extent as f32;
        Self {
            origin:   [-r, -r, -r],
            dx:       grid.dx as f32,
            n1:       grid.n + 1,
            num_segs,
            _pad:     [0; 2],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GpuGridState
// ─────────────────────────────────────────────────────────────────────────────

/// All GPU field buffers for one solve.
///
/// Layout per vertex (z-major flat index: `i = ix + iy·n1 + iz·n1²`):
/// - `phi`:   1 × f32  — scalar potential φ [V]
/// - `a_vec`: 4 × f32  — [Ax, Ay, Az, 0] [V·s/m]
/// - `b_vec`: 4 × f32  — [Bx, By, Bz, 0] [T]
/// - `c_fld`: 1 × f32  — EED scalar C [1/m]
pub struct GpuGridState {
    pub n1:    u32,             // vertices per axis = n_cells + 1
    pub phi:   wgpu::Buffer,   // n1³ × f32
    pub a_vec: wgpu::Buffer,   // n1³ × 4×f32
    pub b_vec: wgpu::Buffer,   // n1³ × 4×f32
    pub c_fld: wgpu::Buffer,   // n1³ × f32
}

impl GpuGridState {
    /// Allocate GPU buffers.  Call once per solve before any dispatch.
    pub fn new(ctx: &GpuContext, grid: &YeeGrid) -> Self {
        let n1    = grid.n + 1;
        let total = (n1 * n1 * n1) as u64;
        let dev   = ctx.device();

        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;

        let make = |label: &str, f32_count: u64| {
            dev.create_buffer(&wgpu::BufferDescriptor {
                label:              Some(label),
                size:               f32_count * 4,
                usage:              storage,
                mapped_at_creation: false,
            })
        };

        log::debug!(
            "GpuGridState: n1={}, total_vertices={}, A_buf={:.1} MB",
            n1, total,
            (total * 4 * 4) as f64 / 1e6
        );

        Self {
            n1,
            phi:   make("phi",   total),
            a_vec: make("a_vec", total * 4),
            b_vec: make("b_vec", total * 4),
            c_fld: make("c_fld", total),
        }
    }

    // ── Phase 1a: Biot-Savart ─────────────────────────────────────────────────

    /// Dispatch the Biot-Savart kernel.
    /// Fills `a_vec` with the static vector potential from `segments`.
    pub fn run_biot_savart(
        &self,
        ctx:      &GpuContext,
        grid:     &YeeGrid,
        segments: &[WireSegment],
    ) -> Result<(), SolverError> {
        if segments.is_empty() {
            log::warn!("Biot-Savart: no segments — A_buf stays zero");
            return Ok(());
        }

        let dev = ctx.device();
        let n1  = self.n1;

        // Upload wire segments
        let seg_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("wire_segments"),
            contents: cast_slice(segments),
            usage:    wgpu::BufferUsages::STORAGE,
        });

        let params     = GridParamsGpu::from_grid(grid, segments.len() as u32);
        let params_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("grid_params_biot"),
            contents: bytes_of(&params),
            usage:    wgpu::BufferUsages::UNIFORM,
        });

        let shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("biot_savart"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/biot.wgsl").into()
            ),
        });

        let pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:               Some("biot_savart"),
            layout:              None,
            module:              &shader,
            entry_point:         "biot_savart",
            compilation_options: Default::default(),
            cache:               None,
        });

        let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("biot_bg"),
            layout:  &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: seg_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: self.a_vec.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let total     = n1 * n1 * n1;
        let wg_count  = total.div_ceil(256);

        let mut enc = dev.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(wg_count, 1, 1);
        }
        ctx.queue().submit([enc.finish()]);
        dev.poll(wgpu::MaintainBase::Wait);

        log::info!(
            "Biot-Savart done: {} vertices × {} segments ({} sub-elements avg)",
            total,
            segments.len(),
            segments.iter().map(|s| s.ndiv as u64).sum::<u64>() / segments.len() as u64,
        );
        Ok(())
    }

    // ── Phase 1b: Field derivation ────────────────────────────────────────────

    /// Dispatch the field derivation kernel.
    /// Computes B = curl(A) and C = div(A) from the current `a_vec`.
    pub fn run_derive_fields(
        &self,
        ctx:  &GpuContext,
        grid: &YeeGrid,
    ) -> Result<(), SolverError> {
        let dev = ctx.device();
        let n1  = self.n1;

        let params     = GridParamsGpu::from_grid(grid, 0);
        let params_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("grid_params_derive"),
            contents: bytes_of(&params),
            usage:    wgpu::BufferUsages::UNIFORM,
        });

        let shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("derive_fields"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/derive.wgsl").into()
            ),
        });

        let pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:               Some("derive_fields"),
            layout:              None,
            module:              &shader,
            entry_point:         "derive_fields",
            compilation_options: Default::default(),
            cache:               None,
        });

        let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("derive_bg"),
            layout:  &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: self.a_vec.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: self.b_vec.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  2,
                    resource: self.c_fld.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let total    = n1 * n1 * n1;
        let wg_count = total.div_ceil(256);

        let mut enc = dev.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(wg_count, 1, 1);
        }
        ctx.queue().submit([enc.finish()]);
        dev.poll(wgpu::MaintainBase::Wait);

        log::info!("Derived B and C fields ({} vertices)", total);
        Ok(())
    }

    // ── Readback ─────────────────────────────────────────────────────────────

    /// Copy a GPU buffer back to CPU as a flat `Vec<f32>`.
    ///
    /// `n_f32` must equal the buffer size in bytes ÷ 4.
    pub fn readback(
        &self,
        ctx:   &GpuContext,
        buf:   &wgpu::Buffer,
        n_f32: usize,
    ) -> Result<Vec<f32>, SolverError> {
        let dev       = ctx.device();
        let byte_size = (n_f32 * 4) as u64;

        let staging = dev.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("readback_staging"),
            size:               byte_size,
            usage:              wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        {
            let mut enc = dev.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(buf, 0, &staging, 0, byte_size);
            ctx.queue().submit([enc.finish()]);
        }

        // Map synchronously: poll(Wait) fires the callback before returning.
        let (tx, rx) = mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
        let slice    = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        dev.poll(wgpu::MaintainBase::Wait);
        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(SolverError::BufferMap(e)),
            Err(_)     => return Err(SolverError::BufferMap(
                wgpu::BufferAsyncError
            )),
        }

        let data = {
            let view = slice.get_mapped_range();
            cast_slice::<u8, f32>(&view).to_vec()
        };
        staging.unmap();
        Ok(data)
    }

    /// Total number of f32 values in the scalar field buffers (n1³).
    pub fn scalar_len(&self) -> usize { (self.n1 * self.n1 * self.n1) as usize }

    /// Total number of f32 values in the vector field buffers (n1³ × 4).
    pub fn vec_len(&self) -> usize { self.scalar_len() * 4 }
}
