# solver/main.py

## Purpose
Entry point for the Oracle solver server. Configures logging, handles SIGTERM for graceful Tauri sidecar shutdown, starts uvicorn.

## Decisions
- PORT=7432 chosen to avoid common conflicts
- SIGTERM handler: Tauri sends SIGTERM on app close — must exit cleanly to avoid zombie processes
- reload=False in production entry; use `uvicorn main:app --reload` for development

## Contracts
- Run via Docker: `docker compose up solver` (recommended — has dolfinx pre-installed)
- Tauri sidecar calls this via shell script in solver/sidecar_entry (also Docker-based)
- dolfinx is NOT available in the UV venv — it lives only in the Docker image
