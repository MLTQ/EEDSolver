//! GPU device initialisation.
//!
//! `GpuContext` owns the wgpu device and queue.  It is created once at app
//! startup and shared (via `Arc`) across all solver calls.
//!
//! Backends by platform:
//!   macOS   → Metal   (via wgpu's Metal backend)
//!   Linux   → Vulkan  (or GL fallback)
//!   Windows → DX12 or Vulkan

use std::sync::Arc;
use wgpu::{Adapter, Device, Instance, Queue};

use crate::error::SolverError;

/// Shared GPU resources.  Clone freely — it's `Arc`-backed.
#[derive(Clone)]
pub struct GpuContext {
    inner: Arc<GpuContextInner>,
}

struct GpuContextInner {
    pub device:  Device,
    pub queue:   Queue,
    pub adapter: Adapter,
}

impl GpuContext {
    /// Initialise the GPU.  Selects the highest-performance adapter available.
    /// On Apple Silicon this will be the Metal-backed integrated GPU.
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

        Ok(Self {
            inner: Arc::new(GpuContextInner { device, queue, adapter }),
        })
    }

    pub fn device(&self)  -> &Device  { &self.inner.device  }
    pub fn queue(&self)   -> &Queue   { &self.inner.queue   }
    pub fn adapter(&self) -> &Adapter { &self.inner.adapter }

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
