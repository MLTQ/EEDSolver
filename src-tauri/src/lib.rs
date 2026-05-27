//! Oracle Tauri application — library entry point.
//! `main.rs` is a thin shim that calls `run()`.

mod commands;
mod types;

use tauri::Manager;
use solver_gpu::OracleSolver;
use commands::{
    delete_hypothesis, get_solver_status, load_hypotheses, save_hypothesis, solve,
};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            // Initialise GPU solver synchronously so state is available before
            // the first invoke call.  Fails fast if no GPU is found.
            let solver = tauri::async_runtime::block_on(OracleSolver::new())
                .map_err(|e| format!("GPU solver init failed: {e}"))?;
            log::info!("GPU solver online: {}", solver.gpu_name());
            app.manage(solver);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            solve,
            get_solver_status,
            save_hypothesis,
            load_hypotheses,
            delete_hypothesis,
        ])
        .run(tauri::generate_context!())
        .expect("Error while running Oracle");
}
