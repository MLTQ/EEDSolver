"""Independent vacuum Maxwell-stress integration for point-charge test problems."""

import numpy as np
from model import EPSILON_0
from numpy.polynomial.legendre import leggauss


def sphere_force(charges_C, positions_m, center_m, radius_m, order=32, external_field=None):
    """Integrate T.n over a sphere; charges may lie inside or outside, never on it.

    external_field, when supplied, maps an (N,3) point array to an (N,3) field.
    The caller must use a source-free external field near/on the sphere.
    """
    q = np.asarray(charges_C, dtype=float)
    pos = np.asarray(positions_m, dtype=float)
    center = np.asarray(center_m, dtype=float)
    if radius_m <= 0 or order < 4 or pos.shape != (len(q), 3) or center.shape != (3,):
        raise ValueError("Invalid surface or charge geometry")
    if not all(np.isfinite(x).all() for x in (q, pos, center)) or not np.isfinite(radius_m):
        raise ValueError("Inputs must be finite")
    if np.any(np.isclose(np.linalg.norm(pos - center, axis=1), radius_m, rtol=1e-10, atol=0)):
        raise ValueError("A charge lies on the integration surface")
    mu, weight = leggauss(order)
    phi = np.arange(2 * order) * np.pi / order
    radial = np.sqrt(1 - mu[:, None] ** 2)
    nx = radial * np.cos(phi)
    ny = radial * np.sin(phi)
    nz = np.broadcast_to(mu[:, None], nx.shape)
    normal = np.stack((nx, ny, nz), axis=-1).reshape(-1, 3)
    area = np.broadcast_to(weight[:, None] * np.pi / order * radius_m**2, nx.shape).ravel()
    points = center + radius_m * normal
    field = np.zeros_like(points)
    for charge, position in zip(q, pos):
        delta = points - position
        field += (
            charge / (4 * np.pi * EPSILON_0) * delta / np.linalg.norm(delta, axis=1)[:, None] ** 3
        )
    if external_field is not None:
        field += external_field(points)
    dot = np.einsum("ij,ij->i", field, normal)
    squared = np.einsum("ij,ij->i", field, field)
    traction = EPSILON_0 * (dot[:, None] * field - 0.5 * squared[:, None] * normal)
    return {
        "force_N": np.sum(traction * area[:, None], axis=0),
        "enclosed_charge_from_flux_C": EPSILON_0 * np.sum(dot * area),
    }
