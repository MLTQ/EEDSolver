# Oracle solver — Docker image
#
# Extends the official FEniCSx image with our FastAPI server deps.
# The dolfinx/dolfinx image provides: dolfinx, mpi4py, gmsh, PETSc, numpy, scipy.
# We add: fastapi, uvicorn, pydantic.
#
# Build:
#   docker build -f solver/Dockerfile -t oracle-solver .
#
# Run (dev, with hot reload):
#   docker run --rm -p 7432:7432 -v $(pwd)/solver:/app/solver oracle-solver \
#     uvicorn solver.main:app --host 0.0.0.0 --port 7432 --reload
#
# Or use: docker compose up solver

FROM dolfinx/dolfinx:stable

WORKDIR /app

# Install server-layer deps on top of the dolfinx base
RUN pip install --no-cache-dir \
    "fastapi>=0.111" \
    "uvicorn[standard]>=0.30" \
    "pydantic>=2.7"

# Copy solver package (overridden by volume mount in dev)
COPY solver/ ./solver/
COPY pyproject.toml .

EXPOSE 7432

CMD ["uvicorn", "solver.main:app", "--host", "0.0.0.0", "--port", "7432"]
