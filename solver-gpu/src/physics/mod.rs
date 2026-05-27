//! Physics dispatch: EED and GEM update kernels.
//!
//! # EED potential-primary FDTD (Phase 3)
//!   ∂²A/∂t²  = c²∇²A − ∇(∂φ/∂t) + (1/ε₀)J_e
//!   ∂²φ/∂t²  = c²∇²φ − c²∂(∇·A)/∂t + ρ_e/ε₀
//!   C        = ∇·A + (1/c²)∂φ/∂t   (the deleted scalar field)
//!
//! # GEM coupled sector (Phase 4)
//!   ∂²A_g/∂t² = c²∇²A_g − ∇(∂Φ_g/∂t) − (4πG/c)J_m + κ_G·∇C
//!   ∂²Φ_g/∂t² = c²∇²Φ_g + 4πG·ρ_m                  + κ_G·∂C/∂t
//!
//! Both update equations have the same stencil — GEM reuses EED shaders
//! with different physical constants.
//!
//! # TODO (Phase 3 / Phase 4)
//! - Implement WGSL shader dispatch
//! - CFL time step checker
//! - Absorbing boundary conditions (Mur ABC)

pub mod eed;
pub mod gem;
