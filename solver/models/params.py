"""
Pydantic models for the Oracle solver API.
Shared type contract between solver, Tauri shell (types.rs), and frontend (fieldTypes.ts).
Any change here must be mirrored in both of those files.
"""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


CoilType = Literal[
    "solenoid",         # Helical winding, standard reference coil
    "toroid",           # Azimuthal (standard) toroid winding
    "toroid_poloidal",  # Poloidal toroid winding — key EED discriminator test
    "flat_spiral",      # Flat Archimedean spiral coil
    "rodin",            # Rodin/Marko coil — figure-8 toroid winding
]

FormulationType = Literal[
    "scalar_only",   # EED scalar φ in isolation (cheapest, start here)
    "maxwell_only",  # Standard magnetostatics — control/baseline
    "eed_coupled",   # Full coupled (φ, A) system — primary research formulation
]

MeshResolution = Literal["coarse", "medium", "fine"]
SliceAxis = Literal["x", "y", "z"]
FieldName = Literal["phi", "A_magnitude", "B_magnitude", "J_magnitude"]


class CoilParams(BaseModel):
    radius_m: float = Field(0.05, gt=0, description="Coil radius (meters)")
    turns: int = Field(10, ge=1, description="Number of turns")
    pitch_m: float = Field(0.005, gt=0, description="Turn-to-turn pitch (meters)")
    wire_radius_m: float = Field(0.001, gt=0, description="Wire cross-section radius (meters)")
    current_A: float = Field(1.0, description="Applied current (amperes)")
    coil_type: CoilType = "solenoid"


class EEDParams(BaseModel):
    """
    EED coupling constants. Free parameters — constrained by experiment, not derived.
    α [1/m]: Yukawa-like scalar mass parameter. λ=1/α is the characteristic decay length.
             α=0 → massless scalar, maximum predicted field extent.
    β [dimensionless]: φ→A coupling (enters A equation as γ∇φ term).
    γ [dimensionless]: A→φ coupling (enters φ equation as β∇·A term).
    When β=γ=0: system decouples to independent Maxwell + scalar.

    TODO: VERIFY AGAINST DDOF PAPER — coupling term sign/structure pending full citation.
    """
    alpha: float = Field(0.0, ge=0.0, description="Scalar mass parameter [1/m]. 0=massless.")
    beta: float = Field(0.1, description="A→φ coupling constant (dimensionless)")
    gamma: float = Field(0.1, description="φ→A coupling constant (dimensionless)")


class SliceRequest(BaseModel):
    axis: SliceAxis = "z"
    position: float = Field(0.5, ge=0.0, le=1.0, description="Position along axis, normalized 0–1")
    field: FieldName = "phi"
    resolution: int = Field(128, ge=16, le=512, description="Grid points per side")


class SliceData(BaseModel):
    axis: SliceAxis
    position: float
    field: FieldName
    shape: list[int]           # [rows, cols]
    data: list[float]          # Flattened 2D array, row-major
    x_range: list[float]       # [min, max] in meters
    y_range: list[float]       # [min, max] in meters
    field_min: float
    field_max: float


class FieldMaximum(BaseModel):
    field: FieldName
    max_value: float
    max_location: list[float]  # [x, y, z] in meters


class VolumeData(BaseModel):
    """
    3D scalar field sampled on a regular grid, normalized to [0, 1].
    Used by the Three.js ray-marching volume viewer (primary 3D view).
    data is flat, row-major (x varies fastest, then y, then z).
    """
    field: FieldName
    shape: list[int]           # [nx, ny, nz]
    data: list[float]          # Normalized to [0, 1], flat row-major
    x_range: list[float]       # [min, max] meters
    y_range: list[float]       # [min, max] meters
    z_range: list[float]       # [min, max] meters
    field_min: float           # Actual pre-normalization field minimum
    field_max: float           # Actual pre-normalization field maximum


class SolveRequest(BaseModel):
    coil: CoilParams = Field(default_factory=CoilParams)
    eed: EEDParams = Field(default_factory=EEDParams)
    domain_radius_m: float = Field(0.2, gt=0, description="Bounding sphere radius (meters)")
    mesh_resolution: MeshResolution = "coarse"
    formulation: FormulationType = "scalar_only"
    slices: list[SliceRequest] = Field(
        default_factory=lambda: [SliceRequest()],
        description="Which 2D slices to extract post-solve",
    )
    request_volume: bool = Field(True, description="Whether to extract 3D volume data")


class SolveResult(BaseModel):
    solve_time_s: float
    mesh_nodes: int
    slices: list[SliceData]
    volume: VolumeData | None = None
    maxima: list[FieldMaximum]
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "ok"
    solver_version: str = "0.1.0"
