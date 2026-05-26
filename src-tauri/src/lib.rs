//! Oracle Tauri application — library entry point.
//! main.rs is a thin shim that calls run().

mod commands;
mod solver_client;
mod types;

use tauri::Manager;
use solver_client::SolverClient;
use commands::{
    delete_hypothesis, get_solver_status, load_hypotheses, save_hypothesis, solve,
};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(SolverClient::new())
        .invoke_handler(tauri::generate_handler![
            solve,
            get_solver_status,
            save_hypothesis,
            load_hypotheses,
            delete_hypothesis,
        ])
        .setup(|app| {
            // Kick off a background health poll so the frontend's first
            // get_solver_status call comes back quickly.
            let client = app.state::<SolverClient>().inner().clone();
            tauri::async_runtime::spawn(async move {
                let status = client.wait_until_ready().await;
                log::info!("Solver startup status: {:?}", status.state);
            });
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("Error while running Oracle");
}
