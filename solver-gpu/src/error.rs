//! Solver error types.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SolverError {
    #[error("No GPU adapter found. Is a Metal/Vulkan/DX12 driver available?")]
    NoGpuAdapter,

    #[error("GPU device request failed: {0}")]
    DeviceRequest(#[from] wgpu::RequestDeviceError),

    #[error("WGSL shader compilation failed: {0}")]
    ShaderCompile(String),

    #[error("Buffer map failed: {0:?}")]
    BufferMap(wgpu::BufferAsyncError),

    #[error("Solver not initialised")]
    NotInitialised,

    #[error("Invalid request: {0}")]
    InvalidRequest(String),
}
