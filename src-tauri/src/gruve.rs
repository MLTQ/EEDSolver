//! Gruve HTTP bridge for remote multiplayer viewers.
//!
//! Tauri IPC only exists inside the desktop webview. Gruve opens the app over
//! HTTP for remote viewers, so this module serves the built frontend and mirrors
//! the solver commands as JSON endpoints on one localhost port.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use solver_gpu::OracleSolver;
use tauri::{AppHandle, Manager};
use tokio::net::TcpListener;
use tower_http::services::{ServeDir, ServeFile};

use crate::commands::{
    delete_hypothesis_entry, load_hypothesis_entries, save_hypothesis_entry, solve_request,
    solver_status,
};
use crate::types::{SolveRequest, SolveResult};

const APP_ID: &str = "oracle";
const APP_NAME: &str = "Oracle";
const ANNOUNCE_URL: &str = "http://127.0.0.1:8088/gruve/announce";
const ANNOUNCE_TTL_SECONDS: u64 = 60;

#[derive(Clone)]
struct GruveState {
    solver: Arc<OracleSolver>,
}

#[derive(Debug)]
struct ApiError(String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (StatusCode::INTERNAL_SERVER_ERROR, self.0).into_response()
    }
}

#[derive(Debug, Deserialize)]
struct SaveHypothesisBody {
    name: String,
    request: SolveRequest,
    result: SolveResult,
    notes: Option<String>,
}

#[derive(Debug, Serialize)]
struct AnnounceBody {
    id: &'static str,
    name: &'static str,
    port: u16,
    ttl: u64,
    hue: u16,
    blurb: &'static str,
    upstreams: AnnounceUpstreams,
}

#[derive(Debug, Serialize)]
struct AnnounceUpstreams {
    api: u16,
}

pub async fn start(app: AppHandle, solver: Arc<OracleSolver>) -> Result<u16, String> {
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .await
        .map_err(|e| format!("Cannot bind Gruve HTTP server: {e}"))?;
    let addr = listener
        .local_addr()
        .map_err(|e| format!("Cannot read Gruve HTTP server address: {e}"))?;
    let port = addr.port();

    let router = Router::new()
        .route("/api/solver-status", get(get_solver_status_http))
        .route("/api/solve", post(solve_http))
        .route("/api/hypotheses", get(load_hypotheses_http))
        .route("/api/hypotheses", post(save_hypothesis_http))
        .route("/api/hypotheses/:id", delete(delete_hypothesis_http));

    let dist_dir = resolve_dist_dir(&app).unwrap_or_else(|| PathBuf::from("../dist"));
    if !dist_dir.join("index.html").is_file() {
        log::warn!(
            "Gruve static assets not found at {}; build the frontend before opening from Gruve",
            dist_dir.display()
        );
    }
    let static_assets =
        ServeDir::new(&dist_dir).not_found_service(ServeFile::new(dist_dir.join("index.html")));

    let router = router
        .fallback_service(static_assets)
        .with_state(GruveState { solver });

    tauri::async_runtime::spawn(async move {
        if let Err(e) = axum::serve(listener, router.into_make_service()).await {
            log::error!("Gruve HTTP server failed on {addr}: {e}");
        }
    });

    tauri::async_runtime::spawn(announce_loop(port));
    log::info!("Gruve HTTP bridge serving Oracle on http://127.0.0.1:{port}");
    Ok(port)
}

async fn get_solver_status_http(
    State(state): State<GruveState>,
) -> Result<Json<crate::types::SolverStatus>, ApiError> {
    Ok(Json(solver_status(&state.solver)))
}

async fn solve_http(
    State(state): State<GruveState>,
    Json(request): Json<SolveRequest>,
) -> Result<Json<SolveResult>, ApiError> {
    solve_request(&state.solver, request)
        .await
        .map(Json)
        .map_err(ApiError)
}

async fn save_hypothesis_http(
    Json(body): Json<SaveHypothesisBody>,
) -> Result<Json<String>, ApiError> {
    save_hypothesis_entry(body.name, body.request, body.result, body.notes)
        .await
        .map(Json)
        .map_err(ApiError)
}

async fn load_hypotheses_http() -> Result<Json<Vec<crate::types::HypothesisEntry>>, ApiError> {
    load_hypothesis_entries().await.map(Json).map_err(ApiError)
}

async fn delete_hypothesis_http(Path(id): Path<String>) -> Result<StatusCode, ApiError> {
    delete_hypothesis_entry(id)
        .await
        .map(|_| StatusCode::NO_CONTENT)
        .map_err(ApiError)
}

fn resolve_dist_dir(app: &AppHandle) -> Option<PathBuf> {
    let mut candidates = Vec::new();

    if let Ok(current_dir) = std::env::current_dir() {
        candidates.push(current_dir.join("dist"));
        candidates.push(current_dir.join("../dist"));
    }

    if let Ok(resource_dir) = app.path().resource_dir() {
        candidates.push(resource_dir.join("dist"));
        candidates.push(resource_dir.join("../dist"));
    }

    candidates
        .into_iter()
        .find(|path| path.join("index.html").is_file())
}

async fn announce_loop(port: u16) {
    let client = reqwest::Client::new();
    let body = AnnounceBody {
        id: APP_ID,
        name: APP_NAME,
        port,
        ttl: ANNOUNCE_TTL_SECONDS,
        hue: 205,
        blurb: "EED field simulator",
        upstreams: AnnounceUpstreams { api: port },
    };
    let mut last_announced = false;

    loop {
        let announced = match client.post(ANNOUNCE_URL).json(&body).send().await {
            Ok(response) => response.status().is_success(),
            Err(_) => false,
        };

        if announced != last_announced {
            if announced {
                log::info!("Oracle announced to Gruve on port {port}");
            } else {
                log::info!("Waiting for Gruve agent at {ANNOUNCE_URL}");
            }
            last_announced = announced;
        }

        tokio::time::sleep(Duration::from_secs(ANNOUNCE_TTL_SECONDS / 3)).await;
    }
}
