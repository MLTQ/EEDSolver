//! Yee grid infrastructure.
//!
//! The Yee grid is a structured Cartesian grid.  In the potential-primary
//! formulation all quantities are stored at vertices for simplicity in
//! Phase 1/2.  The FDTD stagger (Phase 3) will use proper half-cell offsets:
//!
//!   φ        →  cell vertices        (i,   j,   k)
//!   Ax/Ay/Az →  cell edges           (i+½, j,   k) etc.
//!   Ex/Ey/Ez →  same edges as A      (derived)
//!   Bx/By/Bz →  face centres         (i,   j+½, k+½) etc.
//!   C        →  cell centres         (i+½, j+½, k+½) (derived from A, φ)
//!
//! This staggering is the structured-grid realisation of Nédélec H(curl)
//! elements — mathematically equivalent, GPU-friendly.

pub mod state;
pub use state::GpuGridState;

/// Grid configuration.
#[derive(Debug, Clone)]
pub struct YeeGrid {
    /// Number of cells along each axis.
    pub n: u32,
    /// Physical cell size [m].
    pub dx: f64,
    /// Half-extent of the domain [m].  Domain spans [−extent, +extent]³.
    pub extent: f64,
}

impl YeeGrid {
    pub fn new(cells_per_axis: u32, domain_radius_m: f64) -> Self {
        let dx = 2.0 * domain_radius_m / cells_per_axis as f64;
        Self { n: cells_per_axis, dx, extent: domain_radius_m }
    }

    /// Total number of scalar cells (n³).
    pub fn total_cells(&self) -> u64 {
        (self.n as u64).pow(3)
    }

    /// CFL-safe maximum time step [s] for potential-primary FDTD.
    /// dt_max = dx / (c · √3)
    pub fn cfl_dt(&self) -> f64 {
        const C: f64 = 2.998e8;
        self.dx / (C * 3.0_f64.sqrt())
    }
}
