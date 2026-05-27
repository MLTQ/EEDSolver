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
// GPU uniform structs
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

/// Uniform params for the Jacobi elliptic solver.
/// Must match `JacobiParams` in `jacobi.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct JacobiParamsGpu {
    pub dx:   f32,
    pub m2:   f32,
    pub n1:   u32,
    pub _pad: u32,
}

/// Uniform params for the FDTD leapfrog kernels.
/// Must match `FdtdParams` in `fdtd_em.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FdtdParamsGpu {
    pub dx:           f32,
    pub dt:           f32,
    pub n1:           u32,
    pub gamma:        f32,
    pub sponge_cells: u32,
    pub sigma_max:    f32,
    pub _pad:         [u32; 2],
}

/// Uniform params for the C-field update kernel.
/// Must match `CFieldParams` in `c_field.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CFieldParamsGpu {
    pub dx:     f32,
    pub inv_c2: f32,
    pub n1:     u32,
    pub _pad:   u32,
}

/// Uniform params for the GEM FDTD kernel.
/// Must match `GemParams` in `fdtd_gem.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GemParamsGpu {
    pub dx:      f32,
    pub dt:      f32,
    pub n1:      u32,
    pub kappa_g: f32,
}

/// Uniform params for the observables kernel.
/// Must match `ObsParams` in `observables.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ObsParamsGpu {
    pub dx:   f32,
    pub n1:   u32,
    pub _pad: [u32; 2],
}

/// Uniform params for the vector A Jacobi correction kernel.
/// Must match `JacobiAParams` in `jacobi_a.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct JacobiAParamsGpu {
    pub dx:    f32,
    pub m2:    f32,   // α²
    pub gamma: f32,
    pub n1:    u32,
}

/// Uniform params for the PCG scalar solver.
/// Must match `CgParams` in `cg_scalar.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CgParamsGpu {
    pub dx:    f32,
    pub m2:    f32,    // α²
    pub n1:    u32,
    pub alpha: f32,    // CG α  (updated each iteration)
    pub beta:  f32,    // CG β  (updated each iteration)
    pub _pad:  [u32; 3],
}

/// Uniform params for the J-source injection kernel.
/// Must match `InjectParams` in `inject_j.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct InjectParamsGpu {
    pub source_amp: f32,  // −dt · μ₀c² · I₀ · sin(ωt)
    pub n1:         u32,
    pub _pad:       [u32; 2],
}

// ─────────────────────────────────────────────────────────────────────────────
// GpuGridState
// ─────────────────────────────────────────────────────────────────────────────

/// All GPU field buffers for one solve.
///
/// Layout per vertex (z-major flat index: `i = ix + iy·n1 + iz·n1²`):
///
/// EM sector:
/// - `phi`:     1 × f32  — scalar potential φ [V]
/// - `a_vec`:   4 × f32  — [Ax, Ay, Az, 0] [V·s/m]
/// - `phi_vel`: 1 × f32  — ∂φ/∂t [V/s]
/// - `a_vel`:   4 × f32  — ∂A/∂t [V·s/m / s]
/// - `b_vec`:   4 × f32  — [Bx, By, Bz, 0] [T] (derived)
/// - `c_fld`:   1 × f32  — C = ∇·A + (1/c²)·∂φ/∂t [1/m] (derived)
/// - `c_fld_prev`: 1 × f32 — C from previous step (for ∂C/∂t in GEM)
///
/// AC source:
/// - `j_src`:   4 × f32  — [Jx, Jy, Jz, 0] normalised current density [A/m²]
///              for I₀ = 1A.  Set to zero for DC-only sources.
///
/// GEM gravitational sector (Phase 4):
/// - `phi_g`:      1 × f32  — gravitational scalar Φ_g [m²/s²]
/// - `a_g_vec`:    4 × f32  — [Agx, Agy, Agz, 0] [m/s]
/// - `phi_g_vel`:  1 × f32  — ∂Φ_g/∂t
/// - `a_g_vel`:    4 × f32  — ∂A_g/∂t
///
/// Phase 5 observables (EED energy/momentum):
/// - `poynting_mag`: 1 × f32 — |P| = |E×B − C·E|  [W/m² in effective units]
/// - `energy_dens`:  1 × f32 — u = ½(|E|² + |B|² + C²)  [J/m³ in effective units]
pub struct GpuGridState {
    pub n1:          u32,
    // EM
    pub phi:         wgpu::Buffer,
    pub a_vec:       wgpu::Buffer,
    pub phi_vel:     wgpu::Buffer,
    pub a_vel:       wgpu::Buffer,
    pub b_vec:       wgpu::Buffer,
    pub c_fld:       wgpu::Buffer,
    pub c_fld_prev:  wgpu::Buffer,
    // AC source (4 × f32: Jx, Jy, Jz, pad)
    pub j_src:       wgpu::Buffer,
    // GEM
    pub phi_g:       wgpu::Buffer,
    pub a_g_vec:     wgpu::Buffer,
    pub phi_g_vel:   wgpu::Buffer,
    pub a_g_vel:     wgpu::Buffer,
    // Phase 5 observables
    pub poynting_mag: wgpu::Buffer,
    pub energy_dens:  wgpu::Buffer,
}

impl GpuGridState {
    /// Allocate GPU buffers.  Call once per solve before any dispatch.
    pub fn new(ctx: &GpuContext, grid: &YeeGrid) -> Self {
        let n1    = grid.n + 1;
        let total = (n1 * n1 * n1) as u64;
        let dev   = ctx.device();

        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

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
            // EM sector
            phi:        make("phi",        total),
            a_vec:      make("a_vec",      total * 4),
            phi_vel:    make("phi_vel",    total),
            a_vel:      make("a_vel",      total * 4),
            b_vec:      make("b_vec",      total * 4),
            c_fld:      make("c_fld",      total),
            c_fld_prev: make("c_fld_prev", total),
            // AC source buffer (zero-initialised; filled by upload_j_source)
            j_src:      make("j_src",      total * 4),
            // GEM sector
            phi_g:      make("phi_g",      total),
            a_g_vec:    make("a_g_vec",    total * 4),
            phi_g_vel:  make("phi_g_vel",  total),
            a_g_vel:    make("a_g_vel",    total * 4),
            // Phase 5 observables
            poynting_mag: make("poynting_mag", total),
            energy_dens:  make("energy_dens",  total),
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

    // ── AC source ─────────────────────────────────────────────────────────────

    /// Upload a normalised current-density grid to `j_src`.
    ///
    /// `j_data` must be a flat array of length n1³ × 4 ([Jx, Jy, Jz, 0] per vertex)
    /// with J normalised for I₀ = 1 A.  Obtained from `biot::segments_to_j_grid()`.
    ///
    /// After this call, `run_fdtd_ac` will inject the J source at every FDTD step
    /// scaled by the instantaneous current amplitude I₀ · sin(ωt).
    pub fn upload_j_source(
        &self,
        ctx:    &GpuContext,
        j_data: &[f32],
    ) {
        let dev   = ctx.device();
        let bytes = (j_data.len() as u64) * 4;
        let stg   = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("j_src_upload"),
            contents: cast_slice(j_data),
            usage:    wgpu::BufferUsages::COPY_SRC,
        });
        let mut enc = dev.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&stg, 0, &self.j_src, 0, bytes);
        ctx.queue().submit([enc.finish()]);
        dev.poll(wgpu::MaintainBase::Wait);
        log::debug!("Uploaded j_src: {} f32 values", j_data.len());
    }

    // ── Capacitor φ initialisation ────────────────────────────────────────────

    /// Initialise `phi` analytically for capacitor entity types.
    ///
    /// For `CapacitorSymmetric`: linear φ between the plates, zero outside.
    /// For `CapacitorAsymmetric`: point-dipole approximation (±Q at electrode
    /// centres), scaled to give the requested voltage across the gap.
    ///
    /// # Arguments
    /// * `entity`    — the capacitor entity (position, orientation, params)
    /// * `grid`      — Yee grid for coordinate mapping
    pub fn initialize_phi_capacitor(
        &self,
        ctx:    &GpuContext,
        grid:   &YeeGrid,
        entity: &crate::types::CoilEntity,
    ) {
        use crate::types::CoilType;

        let c    = &entity.coil;
        let n1   = self.n1 as usize;
        let dx   = grid.dx;
        let r    = grid.extent;

        let gap  = if c.plate_gap_m > 0.0 { c.plate_gap_m } else { 2.0 * c.pitch_m };
        let half_gap = gap * 0.5;
        let v    = c.voltage_v;
        let pos  = entity.position_m;

        let mut phi_data = vec![0.0f32; n1 * n1 * n1];

        match c.coil_type {
            CoilType::CapacitorSymmetric => {
                // Linear φ between plates, ±V/2 at the plates.
                // Ignore orientation for simplicity (plates ⊥ Z).
                for iz in 0..n1 {
                    let z = -r + iz as f64 * dx - pos[2];
                    let phi_z = if z >= -half_gap && z <= half_gap {
                        // Linearly interpolate: −V/2 at z=−gap/2, +V/2 at z=+gap/2
                        (v / gap) * z
                    } else if z > half_gap {
                        v * 0.5
                    } else {
                        -v * 0.5
                    };
                    // Plate radius mask (optional — full-width approximation if zero)
                    let plate_r2 = if c.radius_m > 0.0 { c.radius_m * c.radius_m } else { 1e10 };
                    for iy in 0..n1 {
                        let y = -r + iy as f64 * dx - pos[1];
                        for ix in 0..n1 {
                            let x = -r + ix as f64 * dx - pos[0];
                            let rho2 = x*x + y*y;
                            // Apply radial mask only outside plate region
                            let phi_val = if rho2 <= plate_r2 {
                                phi_z
                            } else {
                                // Outside the plate radius: decay as if a disc dipole field
                                // Simplified: cosine-weighted drop-off
                                let decay = (plate_r2 / rho2).min(1.0);
                                phi_z * decay
                            };
                            phi_data[ix + iy * n1 + iz * n1 * n1] = phi_val as f32;
                        }
                    }
                }
            }
            CoilType::CapacitorAsymmetric => {
                // Dipole approximation: +Q at anode (z = +gap/2, small electrode),
                // −Q at cathode (z = −gap/2, large plate).
                // Q is chosen so V(small_elec) − V(large_plate) ≈ voltage_v.
                // Effective: use two-point-charge superposition scaled to give V across gap.
                let small_r = c.radius_m / c.plate_aspect.max(1.0);
                let large_r = c.radius_m;

                // Coulomb constant k = 1/(4πε₀) ≈ 8.988e9 V·m/C
                const K_COULOMB: f64 = 8.9875e9;
                // Charge needed: V ≈ k·Q/small_r − k·Q/gap → Q = V / (k·(1/small_r − 1/gap))
                let denom = K_COULOMB * (1.0 / small_r - 1.0 / gap);
                let q_eff = if denom.abs() > 1e-30 { v / denom } else { 0.0 };

                let anode_z   =  half_gap + pos[2];
                let cathode_z = -half_gap + pos[2];

                for iz in 0..n1 {
                    let z = -r + iz as f64 * dx;
                    for iy in 0..n1 {
                        let y = -r + iy as f64 * dx - pos[1];
                        for ix in 0..n1 {
                            let x = -r + ix as f64 * dx - pos[0];
                            let d_anode2   = x*x + y*y + (z - anode_z)*(z - anode_z);
                            let d_cathode2 = x*x + y*y + (z - cathode_z)*(z - cathode_z);
                            let da = d_anode2.sqrt().max(small_r);
                            let dc = d_cathode2.sqrt().max(large_r * 0.5);
                            let phi_val = K_COULOMB * q_eff * (1.0 / da - 1.0 / dc);
                            phi_data[ix + iy * n1 + iz * n1 * n1] = phi_val as f32;
                        }
                    }
                }
            }
            _ => {
                log::warn!("initialize_phi_capacitor called for non-capacitor type — no-op");
                return;
            }
        }

        // Upload phi_data to GPU phi buffer.
        let dev = ctx.device();
        let stg = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("phi_cap_upload"),
            contents: cast_slice(&phi_data),
            usage:    wgpu::BufferUsages::COPY_SRC,
        });
        let byte_size = (phi_data.len() as u64) * 4;
        let mut enc = dev.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&stg, 0, &self.phi, 0, byte_size);
        ctx.queue().submit([enc.finish()]);
        dev.poll(wgpu::MaintainBase::Wait);

        log::info!(
            "Capacitor φ init: type={:?}, V={:.2}V, gap={:.3}m → peak|φ|={:.3}V",
            c.coil_type,
            v,
            gap,
            phi_data.iter().map(|v| v.abs()).fold(0.0f32, f32::max),
        );
    }

    // ── Phase 3: FDTD time-domain ─────────────────────────────────────────────

    /// Run `n_steps` FDTD leapfrog iterations.
    ///
    /// Each step encodes two GPU passes (vel_step, pos_step) into one command
    /// buffer to avoid CPU round-trips.  After all steps, `c_fld` is updated.
    ///
    /// # Initial conditions
    /// Call after `run_biot_savart()` to start from the static A field.
    /// `phi_vel` and `a_vel` are zero-initialised (static start).
    /// Run one or more FDTD leapfrog steps.
    ///
    /// `sigma_max_override`: peak sponge damping rate [1/s].
    ///   - `None`   → use the default `c/(4·dx)` (absorbing PML-like layer).
    ///   - `Some(0.0)` → disable sponge entirely (useful for conservation tests).
    pub fn run_fdtd(
        &self,
        ctx:     &GpuContext,
        grid:    &YeeGrid,
        dt:      f32,
        n_steps: u32,
        gamma:   f32,   // EED coupling: 1.0 = full EED, 0.0 = Maxwell
    ) -> Result<(), SolverError> {
        self.run_fdtd_sponge(ctx, grid, dt, n_steps, gamma, None)
    }

    /// Like `run_fdtd` but with explicit sponge control.
    /// Pass `sigma_max = Some(0.0)` to disable the absorbing layer.
    pub fn run_fdtd_sponge(
        &self,
        ctx:       &GpuContext,
        grid:      &YeeGrid,
        dt:        f32,
        n_steps:   u32,
        gamma:     f32,
        sigma_max: Option<f32>,
    ) -> Result<(), SolverError> {
        if n_steps == 0 { return Ok(()); }

        const C: f32 = 2.998e8_f32;
        const C2: f32 = C * C;
        const INV_C2: f32 = 1.0 / C2;

        let dev  = ctx.device();
        let n1   = self.n1;

        // Build pipelines once.
        let em_shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("fdtd_em"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fdtd_em.wgsl").into()),
        });

        let vel_pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fdtd_vel"), layout: None, module: &em_shader,
            entry_point: "vel_step",
            compilation_options: Default::default(), cache: None,
        });

        let pos_pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fdtd_pos"), layout: None, module: &em_shader,
            entry_point: "pos_step",
            compilation_options: Default::default(), cache: None,
        });

        let cf_shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("c_field"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/c_field.wgsl").into()),
        });

        let cf_pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("c_field_update"), layout: None, module: &cf_shader,
            entry_point: "update_c",
            compilation_options: Default::default(), cache: None,
        });

        // Build bind groups.
        // Sponge layer: 8 cells deep, σ_max = c/dx ÷ 4  (quarter-wave damping).
        let sponge_cells = (n1 / 8).max(4);
        let sigma_max    = sigma_max.unwrap_or_else(|| (2.998e8f32 / grid.dx as f32) * 0.25);
        let fdtd_params  = FdtdParamsGpu {
            dx: grid.dx as f32, dt, n1, gamma,
            sponge_cells, sigma_max, _pad: [0; 2],
        };
        let fdtd_params_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("fdtd_params"),
            contents: bytes_of(&fdtd_params),
            usage:    wgpu::BufferUsages::UNIFORM,
        });

        let cf_params = CFieldParamsGpu { dx: grid.dx as f32, inv_c2: INV_C2, n1, _pad: 0 };
        let cf_params_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("cf_params"),
            contents: bytes_of(&cf_params),
            usage:    wgpu::BufferUsages::UNIFORM,
        });

        let em_entries = |layout: &wgpu::BindGroupLayout| {
            dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fdtd_em_bg"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.phi.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.a_vec.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.phi_vel.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.a_vel.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: fdtd_params_buf.as_entire_binding() },
                ],
            })
        };

        // vel_step and pos_step share the same bind group layout.
        let vel_bg = em_entries(&vel_pipeline.get_bind_group_layout(0));
        let pos_bg = em_entries(&pos_pipeline.get_bind_group_layout(0));

        let cf_bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cf_bg"),
            layout: &cf_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.a_vec.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.phi_vel.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.c_fld.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cf_params_buf.as_entire_binding() },
            ],
        });

        let wg       = (n1 * n1 * n1).div_ceil(256);
        let c_bytes  = (self.scalar_len() * 4) as u64;

        // Encode all n_steps into ONE command buffer — no CPU round-trips.
        let mut enc = dev.create_command_encoder(&Default::default());
        for _ in 0..n_steps {
            // Snapshot current C into c_fld_prev (used by GEM coupling).
            enc.copy_buffer_to_buffer(&self.c_fld, 0, &self.c_fld_prev, 0, c_bytes);

            // Pass 1: update EM velocities.
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&vel_pipeline);
                pass.set_bind_group(0, &vel_bg, &[]);
                pass.dispatch_workgroups(wg, 1, 1);
            }
            // Pass 2: update EM positions using new velocities.
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pos_pipeline);
                pass.set_bind_group(0, &pos_bg, &[]);
                pass.dispatch_workgroups(wg, 1, 1);
            }
        }
        // Update C = div(A) + (1/c²)·∂φ/∂t after all steps.
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&cf_pipeline);
            pass.set_bind_group(0, &cf_bg, &[]);
            pass.dispatch_workgroups(wg, 1, 1);
        }

        ctx.queue().submit([enc.finish()]);
        dev.poll(wgpu::MaintainBase::Wait);

        log::info!("FDTD: {n_steps} steps × dt={:.3e} s  (sim time {:.3e} s)",
            dt, n_steps as f32 * dt);
        Ok(())
    }

    /// Run AC-driven FDTD: injects time-varying J(t) = J₀ · I₀ · sin(ωt) each step.
    ///
    /// Requires `upload_j_source()` to have been called first.
    /// For DC (frequency_hz = 0) this reduces to a constant-amplitude injection each step.
    ///
    /// The `inject_j` pass runs before each `vel_step` / `pos_step` pair so the
    /// external current drives the leapfrog correctly.
    ///
    /// # Arguments
    /// * `current_a`    — peak current amplitude I₀ [A]
    /// * `frequency_hz` — AC frequency [Hz]; 0 = DC (constant injection)
    /// * `t_start_s`    — simulation time at the start of this call [s]
    pub fn run_fdtd_ac(
        &self,
        ctx:          &GpuContext,
        grid:         &YeeGrid,
        dt:           f32,
        n_steps:      u32,
        gamma:        f32,
        sigma_max:    Option<f32>,
        current_a:    f32,
        frequency_hz: f32,
        t_start_s:    f32,
    ) -> Result<(), SolverError> {
        if n_steps == 0 { return Ok(()); }

        // μ₀c² = 1/ε₀ ≈ 1.1294 × 10¹¹ V·m/A
        const MU0_C2: f32 = 1.1294e11;

        const C: f32   = 2.998e8_f32;
        const C2: f32  = C * C;
        const INV_C2: f32 = 1.0 / C2;

        let dev  = ctx.device();
        let n1   = self.n1;

        // Build FDTD pipelines (same as run_fdtd_sponge).
        let em_shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("fdtd_em_ac"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fdtd_em.wgsl").into()),
        });
        let vel_pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fdtd_vel_ac"), layout: None, module: &em_shader,
            entry_point: "vel_step",
            compilation_options: Default::default(), cache: None,
        });
        let pos_pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fdtd_pos_ac"), layout: None, module: &em_shader,
            entry_point: "pos_step",
            compilation_options: Default::default(), cache: None,
        });
        let cf_shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("c_field_ac"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/c_field.wgsl").into()),
        });
        let cf_pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("c_field_update_ac"), layout: None, module: &cf_shader,
            entry_point: "update_c",
            compilation_options: Default::default(), cache: None,
        });

        // inject_j pipeline.
        let inj_shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("inject_j"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/inject_j.wgsl").into()),
        });
        let inj_pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("inject_j_pl"), layout: None, module: &inj_shader,
            entry_point: "inject_j",
            compilation_options: Default::default(), cache: None,
        });

        // FDTD params (constant across all steps).
        let sponge_cells = (n1 / 8).max(4);
        let sigma_max_v  = sigma_max.unwrap_or_else(|| (2.998e8f32 / grid.dx as f32) * 0.25);
        let fdtd_params  = FdtdParamsGpu {
            dx: grid.dx as f32, dt, n1, gamma,
            sponge_cells, sigma_max: sigma_max_v, _pad: [0; 2],
        };
        let fdtd_params_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fdtd_params_ac"), contents: bytes_of(&fdtd_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let cf_params = CFieldParamsGpu { dx: grid.dx as f32, inv_c2: INV_C2, n1, _pad: 0 };
        let cf_params_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cf_params_ac"), contents: bytes_of(&cf_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Inject params buffer — source_amp updated each step.
        let inj_params_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("inj_params"),
            size:               std::mem::size_of::<InjectParamsGpu>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Build bind groups.
        let em_bg = |layout: &wgpu::BindGroupLayout| {
            dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fdtd_em_bg_ac"), layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.phi.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.a_vec.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.phi_vel.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.a_vel.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: fdtd_params_buf.as_entire_binding() },
                ],
            })
        };
        let vel_bg = em_bg(&vel_pipeline.get_bind_group_layout(0));
        let pos_bg = em_bg(&pos_pipeline.get_bind_group_layout(0));

        let cf_bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cf_bg_ac"), layout: &cf_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.a_vec.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.phi_vel.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.c_fld.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cf_params_buf.as_entire_binding() },
            ],
        });

        let inj_bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("inj_bg"), layout: &inj_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.j_src.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.a_vel.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: inj_params_buf.as_entire_binding() },
            ],
        });

        let wg      = (n1 * n1 * n1).div_ceil(256);
        let c_bytes = (self.scalar_len() * 4) as u64;
        let omega   = 2.0 * std::f32::consts::PI * frequency_hz;

        // Step-by-step FDTD with per-step source amplitude update.
        for step in 0..n_steps {
            let t = t_start_s + step as f32 * dt;
            let amplitude = if frequency_hz > 0.0 { (omega * t).sin() } else { 1.0 };

            // source_amp = −dt · μ₀c² · I₀ · sin(ωt)
            let source_amp = -dt * MU0_C2 * current_a * amplitude;
            let inj_p = InjectParamsGpu { source_amp, n1, _pad: [0; 2] };
            ctx.queue().write_buffer(&inj_params_buf, 0, bytes_of(&inj_p));

            let mut enc = dev.create_command_encoder(&Default::default());

            // Snapshot c_fld → c_fld_prev.
            enc.copy_buffer_to_buffer(&self.c_fld, 0, &self.c_fld_prev, 0, c_bytes);

            // Inject J source into a_vel.
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&inj_pipeline);
                pass.set_bind_group(0, &inj_bg, &[]);
                pass.dispatch_workgroups(wg, 1, 1);
            }

            // vel_step: update EM velocities.
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&vel_pipeline);
                pass.set_bind_group(0, &vel_bg, &[]);
                pass.dispatch_workgroups(wg, 1, 1);
            }

            // pos_step: update EM positions.
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pos_pipeline);
                pass.set_bind_group(0, &pos_bg, &[]);
                pass.dispatch_workgroups(wg, 1, 1);
            }

            ctx.queue().submit([enc.finish()]);
            dev.poll(wgpu::MaintainBase::Wait);
        }

        // Update C-field after all steps.
        {
            let mut enc = dev.create_command_encoder(&Default::default());
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&cf_pipeline);
            pass.set_bind_group(0, &cf_bg, &[]);
            pass.dispatch_workgroups(wg, 1, 1);
            drop(pass);
            ctx.queue().submit([enc.finish()]);
            dev.poll(wgpu::MaintainBase::Wait);
        }

        log::info!(
            "AC FDTD: {n_steps} steps × dt={:.3e}s, f={:.2}Hz, I₀={:.3}A",
            dt, frequency_hz, current_a
        );
        Ok(())
    }

    /// Run GEM FDTD: evolve (Φ_g, A_g) coupled to EED C-field via κ_G.
    ///
    /// Must be called AFTER `run_fdtd()` so that `c_fld` and `c_fld_prev`
    /// are populated from the EM solve.  Runs the same number of steps
    /// as the EM solve so the sim times match.
    pub fn run_gem_fdtd(
        &self,
        ctx:     &GpuContext,
        grid:    &YeeGrid,
        dt:      f32,
        n_steps: u32,
        kappa_g: f32,
    ) -> Result<(), SolverError> {
        if n_steps == 0 || kappa_g == 0.0 { return Ok(()); }

        let dev = ctx.device();
        let n1  = self.n1;

        let gem_shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("fdtd_gem"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fdtd_gem.wgsl").into()),
        });

        let vel_pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gem_vel"), layout: None, module: &gem_shader,
            entry_point: "vel_gem",
            compilation_options: Default::default(), cache: None,
        });

        let pos_pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gem_pos"), layout: None, module: &gem_shader,
            entry_point: "pos_gem",
            compilation_options: Default::default(), cache: None,
        });

        let gem_params = GemParamsGpu { dx: grid.dx as f32, dt, n1, kappa_g };
        let gem_params_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("gem_params"),
            contents: bytes_of(&gem_params),
            usage:    wgpu::BufferUsages::UNIFORM,
        });

        let mk_bg = |pipeline: &wgpu::ComputePipeline| {
            dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label:   Some("gem_bg"),
                layout:  &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.phi_g.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.a_g_vec.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.phi_g_vel.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.a_g_vel.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: self.c_fld.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: self.c_fld_prev.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: gem_params_buf.as_entire_binding() },
                ],
            })
        };

        let vel_bg = mk_bg(&vel_pipeline);
        let pos_bg = mk_bg(&pos_pipeline);

        let wg = (n1 * n1 * n1).div_ceil(256);

        let mut enc = dev.create_command_encoder(&Default::default());
        for _ in 0..n_steps {
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&vel_pipeline);
                pass.set_bind_group(0, &vel_bg, &[]);
                pass.dispatch_workgroups(wg, 1, 1);
            }
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pos_pipeline);
                pass.set_bind_group(0, &pos_bg, &[]);
                pass.dispatch_workgroups(wg, 1, 1);
            }
        }

        ctx.queue().submit([enc.finish()]);
        dev.poll(wgpu::MaintainBase::Wait);

        log::info!("GEM FDTD: {n_steps} steps, κ_G={:.3e}", kappa_g);
        Ok(())
    }

    // ── Phase 2: Static EED — Jacobi elliptic solver ─────────────────────────

    /// Solve ∇²φ − α²φ = rhs_data using `n_iter` Jacobi sweeps.
    ///
    /// Stores the final solution in `self.phi`.
    ///
    /// Source convention:
    ///   EED φ equation: rhs = −∇·J  (zero for closed-loop coils; non-zero at
    ///                                  open-circuit endpoints or charge densities).
    ///   Lorenz gauge enforcer: rhs = −∇·A / dt  (Phase 3).
    ///
    /// `alpha_sq` = α² [1/m²].  Pass 0.0 for standard (massless) Maxwell.
    pub fn run_jacobi_phi(
        &mut self,
        ctx:       &GpuContext,
        grid:      &YeeGrid,
        rhs_data:  &[f32],
        alpha_sq:  f32,
        n_iter:    u32,
    ) -> Result<(), SolverError> {
        if n_iter == 0 { return Ok(()); }

        let dev   = ctx.device();
        let n1    = self.n1;
        let total = self.scalar_len() as u64;
        let bytes = total * 4;

        // Allocate the ping-pong scratch buffer (phi itself is "pong").
        // We start from phi = 0 (already zeroed at allocation).
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let ping = dev.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("phi_ping"),
            size:               bytes,
            usage:              storage,
            mapped_at_creation: false,
        });

        // Upload RHS
        let rhs_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("jacobi_rhs"),
            contents: bytemuck::cast_slice(rhs_data),
            usage:    wgpu::BufferUsages::STORAGE,
        });

        let params     = JacobiParamsGpu { dx: grid.dx as f32, m2: alpha_sq, n1, _pad: 0 };
        let params_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("jacobi_params"),
            contents: bytes_of(&params),
            usage:    wgpu::BufferUsages::UNIFORM,
        });

        let shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("jacobi"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/jacobi.wgsl").into()),
        });

        let pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:               Some("jacobi_step"),
            layout:              None,
            module:              &shader,
            entry_point:         "jacobi_step",
            compilation_options: Default::default(),
            cache:               None,
        });

        let wg_count = (n1 * n1 * n1).div_ceil(256);

        // Helper: build bind group for one Jacobi step (u_in → u_out).
        let make_bg = |u_in: &wgpu::Buffer, u_out: &wgpu::Buffer| {
            dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label:   Some("jacobi_bg"),
                layout:  &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: u_in.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: u_out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: rhs_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
                ],
            })
        };

        // Encode all iterations into one command buffer — avoids repeated
        // CPU round-trips and is much faster on Metal/Vulkan.
        let mut enc = dev.create_command_encoder(&Default::default());
        let mut from_ping = true; // phi starts as zero in self.phi; first ping reads phi

        for _ in 0..n_iter {
            let (u_in, u_out): (&wgpu::Buffer, &wgpu::Buffer) = if from_ping {
                (&ping, &self.phi)     // ping → phi
            } else {
                (&self.phi, &ping)     // phi → ping
            };

            let bg = make_bg(u_in, u_out);
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(wg_count, 1, 1);
            }
            from_ping = !from_ping;
        }

        // If we ended with an even number of iterations, the result is in phi.
        // If odd, result is in ping — copy it to phi.
        if !from_ping {
            // Last write was to ping, so copy ping → phi.
            enc.copy_buffer_to_buffer(&ping, 0, &self.phi, 0, bytes);
        }

        ctx.queue().submit([enc.finish()]);
        dev.poll(wgpu::MaintainBase::Wait);

        log::info!("Jacobi φ: {n_iter} iterations, α²={alpha_sq:.3e}, n1={n1}");
        Ok(())
    }

    // ── Phase 2c: Preconditioned CG scalar solver ─────────────────────────────

    /// Solve (−∇² + α²)φ = −rhs_data using PCG with Jacobi preconditioner.
    ///
    /// Replaces `run_jacobi_phi()` for large grids (better convergence).
    /// Four GPU passes per iteration; two tiny CPU readbacks (~4KB each).
    ///
    /// Stores the solution in `self.phi`.  Initial guess: zero.
    pub fn run_cg_phi(
        &mut self,
        ctx:       &GpuContext,
        grid:      &YeeGrid,
        rhs_data:  &[f32],
        alpha_sq:  f32,
        tol:       f32,   // relative convergence: stop when ‖r‖² < tol²·‖b‖²
        max_iter:  u32,
    ) -> Result<(), SolverError> {
        if max_iter == 0 { return Ok(()); }

        let dev      = ctx.device();
        let n1       = self.n1;
        let bytes    = (self.scalar_len() as u64) * 4;
        let wg_count = (n1 * n1 * n1).div_ceil(256);
        let n_wg     = wg_count as usize;
        let part_bytes = (n_wg as u64) * 4;

        // Helper buffers.
        let make_s = |label: &str, sz: u64| dev.create_buffer(&wgpu::BufferDescriptor {
            label:              Some(label),
            size:               sz,
            usage:              wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let r_buf   = make_s("cg_r",   bytes);
        let p_buf   = make_s("cg_p",   bytes);
        let ap_buf  = make_s("cg_ap",  bytes);
        let _rr_buf = make_s("cg_rr",  bytes);   // same as r for second dot (unused: r_buf reused)
        let par_buf = make_s("cg_par", part_bytes);

        // b = −rhs (A = −∇² + α², so A*u = b = −rhs).
        let dx2   = (grid.dx * grid.dx) as f32;
        let m_inv = dx2 / (6.0 + alpha_sq * dx2);
        let b_data: Vec<f32> = rhs_data.iter().map(|&v| -v).collect();
        let b_norm2: f32 = b_data.iter().map(|v| v * v).sum();
        let abs_tol2 = (tol * tol * b_norm2).max(1e-30_f32);

        // Build r₀ = b and p₀ = M⁻¹b (with boundary zeroing) on CPU, upload.
        let n = n1 as usize;
        let boundary = |i: usize| -> bool {
            let ix = i % n; let iy = (i / n) % n; let iz = i / (n * n);
            ix == 0 || ix == n-1 || iy == 0 || iy == n-1 || iz == 0 || iz == n-1
        };
        let r0: Vec<f32> = b_data.iter().enumerate()
            .map(|(i, &v)| if boundary(i) { 0.0 } else { v }).collect();
        let p0: Vec<f32> = r0.iter().map(|&v| v * m_inv).collect();
        {
            let rb = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cg_r0"), contents: bytemuck::cast_slice(&r0),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let pb = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cg_p0"), contents: bytemuck::cast_slice(&p0),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let mut enc = dev.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&rb, 0, &r_buf, 0, bytes);
            enc.copy_buffer_to_buffer(&pb, 0, &p_buf, 0, bytes);
            ctx.queue().submit([enc.finish()]);
            dev.poll(wgpu::MaintainBase::Wait);
        }

        // ρ_old = r₀ · p₀ = ‖r₀‖² · m_inv  (computed on CPU from r0 and p0).
        let mut rho_old: f32 = r0.iter().zip(p0.iter()).map(|(r, p)| r * p).sum();

        // Params buffer (alpha/beta updated each iteration).
        let mut cg_par = CgParamsGpu {
            dx: grid.dx as f32, m2: alpha_sq, n1,
            alpha: 0.0, beta: 0.0, _pad: [0; 3],
        };
        let params_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("cg_params"),
            contents: bytes_of(&cg_par),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Compile shader + pipelines once.
        let shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("cg_scalar"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/cg_scalar.wgsl").into()),
        });
        let mk_pl = |label: &str, entry: &str| dev.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some(label), layout: None, module: &shader,
                entry_point: entry,
                compilation_options: Default::default(), cache: None,
            }
        );
        let mv_pl  = mk_pl("cg_matvec",    "cg_matvec");
        let dot_pl = mk_pl("cg_dot",       "cg_dot");
        let xr_pl  = mk_pl("cg_update_xr", "cg_update_xr");
        let p_pl   = mk_pl("cg_update_p",  "cg_update_p");

        // Bind group factory.
        let bg = |pl: &wgpu::ComputePipeline, adot: &wgpu::Buffer, bdot: &wgpu::Buffer| {
            dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label:   Some("cg_bg"),
                layout:  &pl.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.phi.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: r_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: p_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: ap_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: adot.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: bdot.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: par_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: params_buf.as_entire_binding() },
                ],
            })
        };

        let mv_bg   = bg(&mv_pl,  &r_buf,  &p_buf);    // a_dot/b_dot unused by matvec
        let pap_bg  = bg(&dot_pl, &p_buf,  &ap_buf);   // dot(p, ap)
        let rr_bg   = bg(&dot_pl, &r_buf,  &r_buf);    // dot(r, r)  ← after update_xr
        let xr_bg   = bg(&xr_pl,  &r_buf,  &p_buf);    // a_dot/b_dot unused by update_xr
        let p_bg    = bg(&p_pl,   &r_buf,  &p_buf);    // a_dot/b_dot unused by update_p

        // Tiny dot-product readback: reads par_buf (≤ 8192 floats), sums on CPU.
        let dot_sum = |enc_pre: Option<wgpu::CommandBuffer>| -> Result<f32, SolverError> {
            // Submit any pending work first.
            if let Some(cb) = enc_pre { ctx.queue().submit([cb]); }
            dev.poll(wgpu::MaintainBase::Wait);

            let sz = part_bytes;
            let stg = dev.create_buffer(&wgpu::BufferDescriptor {
                label:              Some("cg_stg"),
                size:               sz,
                usage:              wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut enc = dev.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&par_buf, 0, &stg, 0, sz);
            ctx.queue().submit([enc.finish()]);

            let (tx, rx) = mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
            stg.slice(..).map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
            dev.poll(wgpu::MaintainBase::Wait);
            match rx.recv() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(SolverError::BufferMap(e)),
                Err(_)     => return Err(SolverError::BufferMap(wgpu::BufferAsyncError)),
            }
            let vals = { let v = stg.slice(..).get_mapped_range(); bytemuck::cast_slice::<u8, f32>(&v).to_vec() };
            stg.unmap();
            Ok(vals.iter().sum())
        };

        let dispatch = |pl: &wgpu::ComputePipeline, bg: &wgpu::BindGroup| {
            let mut enc = dev.create_command_encoder(&Default::default());
            { let mut pass = enc.begin_compute_pass(&Default::default());
              pass.set_pipeline(pl);
              pass.set_bind_group(0, bg, &[]);
              pass.dispatch_workgroups(wg_count, 1, 1); }
            enc.finish()
        };

        // ── PCG iteration loop ────────────────────────────────────────────────
        let mut final_iter = 0u32;
        for iter in 0..max_iter {
            final_iter = iter;

            // 1. ap = A·p
            ctx.queue().submit([dispatch(&mv_pl, &mv_bg)]);
            dev.poll(wgpu::MaintainBase::Wait);

            // 2. dot(p, ap) → par_buf
            ctx.queue().submit([dispatch(&dot_pl, &pap_bg)]);
            let p_ap = dot_sum(None)?;
            if p_ap.abs() < 1e-30 { break; }

            // 3. α = ρ_old / (p·ap)
            cg_par.alpha = rho_old / p_ap;
            cg_par.beta  = 0.0;   // beta unused in update_xr
            ctx.queue().write_buffer(&params_buf, 0, bytes_of(&cg_par));

            // 4. x += α·p;  r -= α·ap
            ctx.queue().submit([dispatch(&xr_pl, &xr_bg)]);
            dev.poll(wgpu::MaintainBase::Wait);

            // 5. dot(r, r) → par_buf  (residual norm squared)
            ctx.queue().submit([dispatch(&dot_pl, &rr_bg)]);
            let r2 = dot_sum(None)?;

            // 6. ρ_new = m_inv · ‖r‖²  (Jacobi preconditioner is scalar)
            let rho_new = r2 * m_inv;

            // 7. Convergence check: ‖r‖² < tol² · ‖b‖²
            if r2 < abs_tol2 { break; }

            // 8. β = ρ_new / ρ_old
            cg_par.beta = rho_new / rho_old;
            ctx.queue().write_buffer(&params_buf, 0, bytes_of(&cg_par));
            rho_old = rho_new;

            // 9. p = M⁻¹r + β·p
            ctx.queue().submit([dispatch(&p_pl, &p_bg)]);
            dev.poll(wgpu::MaintainBase::Wait);
        }

        log::info!(
            "PCG φ: {} iters, α²={alpha_sq:.3e}, ‖r‖/‖b‖={:.2e}, n1={n1}",
            final_iter + 1,
            (b_norm2.max(1e-30).sqrt() / b_norm2.max(1e-30).sqrt()),
        );
        Ok(())
    }

    // ── Phase 5: EED observables ──────────────────────────────────────────────

    /// Compute EED modified Poynting magnitude and energy density.
    ///
    /// Must be called after `run_derive_fields()` (needs `b_vec`, `c_fld`)
    /// and, for time-domain solves, after `run_fdtd()` (needs `phi`, `a_vel`).
    ///
    /// In static mode `a_vel` is zero (buffer initialised to 0), so
    /// E = −∇φ and both observables reflect only the static B/C configuration.
    ///
    /// Fills:
    ///   `poynting_mag` — |P| = |E×B − C·E|  per vertex
    ///   `energy_dens`  — u = ½(|E|² + |B|² + C²)  per vertex
    pub fn run_observables(
        &self,
        ctx:  &GpuContext,
        grid: &YeeGrid,
    ) -> Result<(), SolverError> {
        let dev = ctx.device();
        let n1  = self.n1;

        let params = ObsParamsGpu { dx: grid.dx as f32, n1, _pad: [0; 2] };
        let params_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("obs_params"),
            contents: bytes_of(&params),
            usage:    wgpu::BufferUsages::UNIFORM,
        });

        let shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("observables"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/observables.wgsl").into()
            ),
        });

        let pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:               Some("compute_obs"),
            layout:              None,
            module:              &shader,
            entry_point:         "compute_obs",
            compilation_options: Default::default(),
            cache:               None,
        });

        let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("obs_bg"),
            layout:  &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.phi.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.a_vel.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.b_vec.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.c_fld.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.poynting_mag.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.energy_dens.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: params_buf.as_entire_binding() },
            ],
        });

        let wg = (n1 * n1 * n1).div_ceil(256);
        let mut enc = dev.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(wg, 1, 1);
        }
        ctx.queue().submit([enc.finish()]);
        dev.poll(wgpu::MaintainBase::Wait);

        log::info!("Observables computed: |P| and u ({} vertices)", n1 * n1 * n1);
        Ok(())
    }

    // ── Phase 2b: Static EED vector-potential correction ──────────────────────

    /// Apply the Yukawa + EED γ correction to the static vector potential.
    ///
    /// Solves (∇² − α²) A = −μ₀J + γ∇φ  where −μ₀J is encoded via the
    /// Biot-Savart A already in `self.a_vec`.  The correction δA is computed
    /// by Jacobi iteration and added back to `a_vec`.
    ///
    /// Skip if `alpha_sq == 0.0` and `gamma == 0.0` (Coulomb gauge, pure Biot-Savart).
    ///
    /// # Inputs
    /// - `alpha_sq`: α² [1/m²].  0.0 → no Yukawa correction.
    /// - `gamma`:    EED coupling γ.  0.0 → no φ→A coupling.
    /// - `n_iter`:   Jacobi iteration count (64–128 typical).
    pub fn run_jacobi_a_correction(
        &mut self,
        ctx:      &GpuContext,
        grid:     &YeeGrid,
        alpha_sq: f32,
        gamma:    f32,
        n_iter:   u32,
    ) -> Result<(), SolverError> {
        if n_iter == 0 || (alpha_sq == 0.0 && gamma == 0.0) {
            return Ok(());
        }

        let dev   = ctx.device();
        let n1    = self.n1;
        let total = self.vec_len() as u64;   // n1³ × 4 f32
        let bytes = total * 4;

        // Scratch δA buffer (ping-pong); starts at zero.
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let da_ping = dev.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("da_ping"),
            size:               bytes,
            usage:              storage,
            mapped_at_creation: false,
        });
        let da_pong = dev.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("da_pong"),
            size:               bytes,
            usage:              storage,
            mapped_at_creation: false,
        });

        let params = JacobiAParamsGpu { dx: grid.dx as f32, m2: alpha_sq, gamma, n1 };
        let params_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("jacobi_a_params"),
            contents: bytes_of(&params),
            usage:    wgpu::BufferUsages::UNIFORM,
        });

        let shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("jacobi_a"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/jacobi_a.wgsl").into()),
        });

        let pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:               Some("jacobi_a_step"),
            layout:              None,
            module:              &shader,
            entry_point:         "jacobi_a_step",
            compilation_options: Default::default(),
            cache:               None,
        });

        let wg_count = (n1 * n1 * n1).div_ceil(256);

        // Build bind group for one Jacobi step (da_in → da_out).
        let make_bg = |da_in: &wgpu::Buffer, da_out: &wgpu::Buffer| {
            dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label:   Some("jacobi_a_bg"),
                layout:  &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: da_in.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: da_out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.a_vec.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.phi.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
                ],
            })
        };

        // Encode all iterations into one command buffer.
        let mut enc     = dev.create_command_encoder(&Default::default());
        let mut to_pong = true;   // first write goes to pong (da_ping is input = zero)

        for _ in 0..n_iter {
            let (da_in, da_out): (&wgpu::Buffer, &wgpu::Buffer) = if to_pong {
                (&da_ping, &da_pong)
            } else {
                (&da_pong, &da_ping)
            };
            let bg = make_bg(da_in, da_out);
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(wg_count, 1, 1);
            }
            to_pong = !to_pong;
        }

        ctx.queue().submit([enc.finish()]);
        dev.poll(wgpu::MaintainBase::Wait);

        // Determine which buffer holds the final δA result.
        // After n_iter steps starting from ping=0 → pong:
        //   n_iter odd  → result in pong
        //   n_iter even → result in ping
        let da_final = if n_iter % 2 == 1 { &da_pong } else { &da_ping };

        // A_EED = A_BS + δA.  We need to add da_final to self.a_vec.
        // Use a small add kernel via a copy + compute pipeline.
        // For simplicity: read back δA, add on CPU, re-upload.
        // (A GPU add kernel would be faster for large grids — Phase 2b opt.)
        let (tx, rx) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
        let staging = dev.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("da_staging"),
            size:               bytes,
            usage:              wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        {
            let mut enc2 = dev.create_command_encoder(&Default::default());
            enc2.copy_buffer_to_buffer(da_final, 0, &staging, 0, bytes);
            ctx.queue().submit([enc2.finish()]);
        }
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        dev.poll(wgpu::MaintainBase::Wait);
        if rx.recv().map_or(true, |r| r.is_err()) {
            return Err(SolverError::BufferMap(wgpu::BufferAsyncError));
        }

        let da_vals = {
            let view = staging.slice(..).get_mapped_range();
            bytemuck::cast_slice::<u8, f32>(&view).to_vec()
        };
        staging.unmap();

        // Read back current A_BS, add δA, re-upload.
        let a_bs_vals = self.readback(ctx, &self.a_vec, self.vec_len())?;
        let a_eed: Vec<f32> = a_bs_vals.iter().zip(da_vals.iter())
            .map(|(a, da)| a + da)
            .collect();

        let new_a_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("a_vec_eed"),
            contents: bytemuck::cast_slice(&a_eed),
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        // Copy new A into self.a_vec.
        {
            let mut enc3 = dev.create_command_encoder(&Default::default());
            enc3.copy_buffer_to_buffer(&new_a_buf, 0, &self.a_vec, 0, bytes);
            ctx.queue().submit([enc3.finish()]);
        }
        dev.poll(wgpu::MaintainBase::Wait);

        let da_max: f32 = da_vals.iter().copied().map(f32::abs).fold(0.0, f32::max);
        log::info!(
            "Jacobi A correction: {n_iter} iters, α²={alpha_sq:.3e}, γ={gamma:.3}, |δA|_max={da_max:.3e}"
        );
        Ok(())
    }
}
