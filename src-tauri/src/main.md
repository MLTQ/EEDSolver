# src-tauri/src/main.rs

## Purpose
Thin entry point shim. All logic lives in lib.rs (required by Tauri 2 mobile build pattern).

## Contracts
- `windows_subsystem = "windows"` suppresses the console window in Windows release builds — no effect on macOS/Linux
