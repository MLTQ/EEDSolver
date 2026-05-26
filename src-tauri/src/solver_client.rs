//! HTTP client for the Oracle Python solver server.
//!
//! The solver runs on localhost:7432 (started manually in dev,
//! as a Tauri sidecar in production). This module handles:
//!   - Health check polling on startup
//!   - Long-timeout solve requests (fine meshes take minutes)
//!   - Clean error messages surfaced to Tauri commands

use std::time::Duration;
use tokio::time::sleep;

use crate::types::{SolveRequest, SolveResult, SolverState, SolverStatus};

const SOLVER_BASE_URL: &str = "http://127.0.0.1:7432";
const HEALTH_POLL_INTERVAL_MS: u64 = 500;
const HEALTH_POLL_TIMEOUT_S: u64 = 30;
/// Fine meshes can take several minutes — use a generous timeout.
const SOLVE_TIMEOUT_S: u64 = 600;

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct SolverClient {
    client: reqwest::Client,
    base_url: String,
}

impl SolverClient {
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            // Note: per-request timeouts override this for the solve endpoint
            .timeout(Duration::from_secs(10))
            .build()
            .expect("Failed to build reqwest client");

        Self {
            client,
            base_url: SOLVER_BASE_URL.to_string(),
        }
    }

    /// Single health check. Returns Ok(true) if solver is ready.
    pub async fn health_check(&self) -> Result<bool, reqwest::Error> {
        let url = format!("{}/health", self.base_url);
        let resp = self
            .client
            .get(&url)
            .timeout(Duration::from_secs(3))
            .send()
            .await?;
        Ok(resp.status().is_success())
    }

    /// Poll /health until the solver responds or we time out.
    /// Used by the Tauri app on startup before enabling the Solve button.
    pub async fn wait_until_ready(&self) -> SolverStatus {
        let deadline = std::time::Instant::now()
            + Duration::from_secs(HEALTH_POLL_TIMEOUT_S);

        loop {
            match self.health_check().await {
                Ok(true) => {
                    log::info!("Solver ready at {}", self.base_url);
                    return SolverStatus {
                        state: SolverState::Ready,
                        message: "Solver ready".into(),
                    };
                }
                Ok(false) => {
                    log::debug!("Solver responded but not healthy — retrying");
                }
                Err(e) => {
                    log::debug!("Solver health check failed: {e}");
                }
            }

            if std::time::Instant::now() >= deadline {
                return SolverStatus {
                    state: SolverState::Error,
                    message: format!(
                        "Solver did not become ready within {HEALTH_POLL_TIMEOUT_S}s. \
                         Run: docker compose up solver"
                    ),
                };
            }

            sleep(Duration::from_millis(HEALTH_POLL_INTERVAL_MS)).await;
        }
    }

    /// POST /solve — sends the solve request, waits for the result.
    /// Uses a long timeout because fine meshes can take several minutes.
    pub async fn solve(&self, request: &SolveRequest) -> Result<SolveResult, SolverError> {
        let url = format!("{}/solve", self.base_url);

        log::info!(
            "POST /solve: {} {:?} mesh={:?}",
            request.coil.coil_type.label(),
            request.formulation.label(),
            request.mesh_resolution.label(),
        );

        // Build a fresh client with the long solve timeout
        let long_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(SOLVE_TIMEOUT_S))
            .build()
            .map_err(|e| SolverError::ClientError(e.to_string()))?;

        let resp = long_client
            .post(&url)
            .json(request)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    SolverError::Timeout(SOLVE_TIMEOUT_S)
                } else if e.is_connect() {
                    SolverError::NotReachable
                } else {
                    SolverError::RequestFailed(e.to_string())
                }
            })?;

        let status = resp.status();
        if status.is_success() {
            let result: SolveResult = resp
                .json()
                .await
                .map_err(|e| SolverError::ParseError(e.to_string()))?;
            log::info!(
                "Solve complete: {:.2}s, {} nodes, {} warnings",
                result.solve_time_s,
                result.mesh_nodes,
                result.warnings.len(),
            );
            Ok(result)
        } else {
            let body = resp.text().await.unwrap_or_else(|_| "(unreadable)".into());
            Err(SolverError::SolverFailed { status: status.as_u16(), body })
        }
    }

    /// GET /fields — list field names the solver can produce.
    #[allow(dead_code)]
    pub async fn list_fields(&self) -> Result<Vec<String>, SolverError> {
        let url = format!("{}/fields", self.base_url);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| SolverError::RequestFailed(e.to_string()))?;
        resp.json()
            .await
            .map_err(|e| SolverError::ParseError(e.to_string()))
    }

    #[allow(dead_code)]
    pub fn status_not_ready() -> SolverStatus {
        SolverStatus {
            state: SolverState::Starting,
            message: "Solver not yet ready. Waiting for startup...".into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Display helpers on type enums (for logging)
// ---------------------------------------------------------------------------

impl crate::types::CoilType {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Solenoid => "solenoid",
            Self::Toroid => "toroid",
            Self::ToroidPoloidal => "toroid_poloidal",
            Self::FlatSpiral => "flat_spiral",
            Self::Rodin => "rodin",
        }
    }
}

impl crate::types::FormulationType {
    pub fn label(&self) -> &'static str {
        match self {
            Self::ScalarOnly => "scalar_only",
            Self::MaxwellOnly => "maxwell_only",
            Self::EedCoupled => "eed_coupled",
        }
    }
}

impl crate::types::MeshResolution {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Coarse => "coarse",
            Self::Medium => "medium",
            Self::Fine => "fine",
        }
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum SolverError {
    NotReachable,
    Timeout(u64),
    RequestFailed(String),
    SolverFailed { status: u16, body: String },
    ParseError(String),
    ClientError(String),
}

impl std::fmt::Display for SolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotReachable => write!(
                f,
                "Cannot reach solver at {SOLVER_BASE_URL}. \
                 Start it: docker compose up solver"
            ),
            Self::Timeout(s) => write!(f, "Solve timed out after {s}s"),
            Self::RequestFailed(msg) => write!(f, "Request failed: {msg}"),
            Self::SolverFailed { status, body } => {
                write!(f, "Solver returned HTTP {status}: {body}")
            }
            Self::ParseError(msg) => write!(f, "Failed to parse solver response: {msg}"),
            Self::ClientError(msg) => write!(f, "HTTP client error: {msg}"),
        }
    }
}

impl From<SolverError> for String {
    fn from(e: SolverError) -> Self {
        e.to_string()
    }
}
