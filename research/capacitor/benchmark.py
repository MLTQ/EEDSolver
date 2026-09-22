"""Run the documented capacitor calculation and save reproducible JSON results."""

import hashlib
import json
from pathlib import Path

import numpy as np
from model import Capacitor, signed_amplitude_interval
from stress import sphere_force

ROOT = Path(__file__).resolve().parent


def run():
    config_bytes = (ROOT / "geometry.json").read_bytes()
    config = json.loads(config_bytes)
    cap = Capacitor(**{key: config[key] for key in Capacitor.__dataclass_fields__})
    endpoint = cap.at_voltage(config["voltage_V"])
    curves = [cap.at_voltage(float(v)) for v in np.linspace(-10000, 10000, 81)]
    bounds = {}
    for name in ("r43_candidate_force_N", "r40_gaussian_converted_candidate_force_N"):
        interval = signed_amplitude_interval(
            config["measured_vertical_force_N"],
            config["reported_vertical_uncertainty_N"],
            endpoint[name],
            config["decision_multiplier"],
        )
        bounds[name] = {
            "signed_amplitude_interval": interval,
            "unknown_sign_absolute_amplitude_limit": max(map(abs, interval)),
            "unit_amplitude_consistent_with_interval": interval[0] <= 1 <= interval[1],
        }
    # Deliberately off-center, non-mirror-symmetric closed source: zero force
    # must emerge from the surface integral, not from pairing mesh points.
    q = np.array([1.0e-9, -0.7e-9, -0.3e-9])
    pos = np.array([[0.005, 0.008, 0.013], [-0.011, 0.002, -0.007], [0.002, -0.009, 0.004]])
    convergence = []
    for radius in (0.025, 0.04, 0.06):
        for order in (8, 16, 32, 64):
            result = sphere_force(q, pos, [0, 0, 0], radius, order)
            convergence.append(
                {
                    "radius_m": radius,
                    "order": order,
                    "force_norm_N": float(np.linalg.norm(result["force_N"])),
                    "charge_flux_C": float(result["enclosed_charge_from_flux_C"]),
                }
            )
    summary = {
        "geometry_id": config["id"],
        "geometry_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "source_pdf_sha256": config["source_pdf_sha256"],
        "scope": "Analytic capacitor baseline plus independent point-charge stress verification",
        "capacitance_F": cap.capacitance_F,
        "dielectric_mass_kg": cap.dielectric_mass_kg,
        "area_m2": cap.area_m2,
        "endpoint": endpoint,
        "conditional_amplitude_bounds": bounds,
        "stray_capacitance_gradient_matching_r43_F_per_m": (
            2 * endpoint["r43_candidate_force_N"] / config["voltage_V"] ** 2
        ),
        "voltage_curve": curves,
        "stress_convergence": convergence,
    }
    (ROOT / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("voltage_curve", "stress_convergence")},
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
