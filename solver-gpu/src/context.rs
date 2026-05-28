//! GPU device initialisation and shader cache.
//!
//! `GpuContext` owns the wgpu device, queue, and a `ShaderCache` that holds
//! every compiled `ShaderModule`.  It is created once at app startup (in
//! `OracleSolver::new`) and shared via `Arc` across all solver calls.
//!
//! **Why cache shaders?**
//! Each `device.create_shader_module(WGSL)` call runs the full pipeline:
//!   WGSL → naga IR → validate → MSL → Metal binary
//! This takes ~100–500 ms per shader on M-series Macs.  With 10+ shaders
//! and `fdtd_em`/`c_field` compiled twice per solve (DC + AC paths), the
//! first solve and every subsequent solve paid this cost needlessly.
//!
//! By compiling all modules once at startup, subsequent solves reuse the
//! cached `ShaderModule` objects — `create_compute_pipeline` from an existing
//! module skips the translation step entirely.
//!
//! Backends by platform:
//!   macOS   → Metal   (via wgpu's Metal backend)
//!   Linux   → Vulkan  (or GL fallback)
//!   Windows → DX12 or Vulkan

use std::sync::Arc;
use wgpu::{Adapter, Device, Instance, Queue, ShaderModule};

use crate::error::SolverError;

// ── Shader cache ──────────────────────────────────────────────────────────────

/// All compute `ShaderModule`s compiled once at startup.
///
/// Each field corresponds to one WGSL source file.  Modules are reused across
/// every solve — `create_compute_pipeline` from a pre-compiled module avoids
/// re-running the naga WGSL→MSL translation.
pub struct ShaderCache {
    pub biot:            ShaderModule,
    pub derive:          ShaderModule,
    pub derive_gem:      ShaderModule,
    pub fdtd_em:         ShaderModule,
    pub c_field:         ShaderModule,
    pub inject_j:        ShaderModule,
    pub fdtd_gem:        ShaderModule,
    pub li_torr_source:  ShaderModule,
    pub jacobi:          ShaderModule,
    pub cg_scalar:       ShaderModule,
    pub observables:     ShaderModule,
    pub jacobi_a:        ShaderModule,
}

impl ShaderCache {
    fn compile(device: &Device) -> Self {
        let mk = |label: &str, src: &'static str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label:  Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        };
        log::info!("Compiling GPU shaders…");
        let t0 = std::time::Instant::now();
        let cache = Self {
            biot:           mk("biot",           include_str!("shaders/biot.wgsl")),
            derive:         mk("derive",         include_str!("shaders/derive.wgsl")),
            derive_gem:     mk("derive_gem",     include_str!("shaders/derive_gem.wgsl")),
            fdtd_em:        mk("fdtd_em",        include_str!("shaders/fdtd_em.wgsl")),
            c_field:        mk("c_field",        include_str!("shaders/c_field.wgsl")),
            inject_j:       mk("inject_j",       include_str!("shaders/inject_j.wgsl")),
            fdtd_gem:       mk("fdtd_gem",       include_str!("shaders/fdtd_gem.wgsl")),
            li_torr_source: mk("li_torr_source", include_str!("shaders/li_torr_source.wgsl")),
            jacobi:         mk("jacobi",         include_str!("shaders/jacobi.wgsl")),
            cg_scalar:      mk("cg_scalar",      include_str!("shaders/cg_scalar.wgsl")),
            observables:    mk("observables",    include_str!("shaders/observables.wgsl")),
            jacobi_a:       mk("jacobi_a",       include_str!("shaders/jacobi_a.wgsl")),
        };
        log::info!("Shader compilation done in {:.2}s", t0.elapsed().as_secs_f64());
        cache
    }
}

// ── GpuContext ────────────────────────────────────────────────────────────────

/// Shared GPU resources.  Clone freely — it's `Arc`-backed.
#[derive(Clone)]
pub struct GpuContext {
    inner: Arc<GpuContextInner>,
}

struct GpuContextInner {
    pub device:   Device,
    pub queue:    Queue,
    pub adapter:  Adapter,
    pub shaders:  ShaderCache,
}

impl GpuContext {
    /// Initialise the GPU and compile all shaders.
    /// Call once at app startup — this is the expensive step (~0.5–3 s on
    /// first run due to Metal shader compilation; cached by the OS thereafter).
    pub async fn new() -> Result<Self, SolverError> {
        let instance = Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference:       wgpu::PowerPreference::HighPerformance,
                compatible_surface:     None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(SolverError::NoGpuAdapter)?;

        log::info!(
            "GPU adapter: {} ({:?})",
            adapter.get_info().name,
            adapter.get_info().backend
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label:             Some("oracle"),
                    required_features: wgpu::Features::empty(),
                    required_limits:   wgpu::Limits::default(),
                    memory_hints:      Default::default(),
                },
                None,
            )
            .await
            .map_err(SolverError::DeviceRequest)?;

        // Compile all shaders at startup — paid once, reused forever.
        let shaders = ShaderCache::compile(&device);

        Ok(Self {
            inner: Arc::new(GpuContextInner { device, queue, adapter, shaders }),
        })
    }

    pub fn device(&self)   -> &Device       { &self.inner.device  }
    pub fn queue(&self)    -> &Queue        { &self.inner.queue   }
    pub fn adapter(&self)  -> &Adapter      { &self.inner.adapter }
    pub fn shaders(&self)  -> &ShaderCache  { &self.inner.shaders }

    /// Human-readable GPU name for status display.
    pub fn adapter_name(&self) -> String {
        self.inner.adapter.get_info().name.clone()
    }
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GpuContext({})", self.adapter_name())
    }
}
