"""SI capacitor estimates and explicitly conditional candidate-force predictions."""

from dataclasses import dataclass
from math import isfinite, pi, sqrt

# CODATA 2022 central values; standard gravity is a scale, not the site's measured g.
EPSILON_0 = 8.8541878188e-12
G = 6.67430e-11
C_LIGHT = 299792458.0
G_STANDARD = 9.80665


@dataclass(frozen=True)
class Capacitor:
    diameter_m: float
    thickness_m: float
    relative_permittivity: float
    density_kg_m3: float

    def __post_init__(self):
        if any(not isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("Geometry, permittivity and density must be finite and positive")

    @property
    def area_m2(self):
        return pi * self.diameter_m**2 / 4

    @property
    def capacitance_F(self):
        return EPSILON_0 * self.relative_permittivity * self.area_m2 / self.thickness_m

    @property
    def dielectric_mass_kg(self):
        return self.density_kg_m3 * self.area_m2 * self.thickness_m

    def at_voltage(self, voltage_V):
        """Return estimates; candidate sign follows the chosen positive electrode axis."""
        if not isfinite(voltage_V):
            raise ValueError("Voltage must be finite")
        field = voltage_V / self.thickness_m
        energy = self.capacitance_F * voltage_V**2 / 2
        pressure = EPSILON_0 * self.relative_permittivity * field**2 / 2
        # R43 Eq. 1, reproduced verbatim as a model; not claimed to follow from Maxwell.
        candidate = sqrt(G * EPSILON_0 * self.relative_permittivity)
        candidate *= self.density_kg_m3 * self.area_m2 * voltage_V
        # Direct Gaussian-unit conversion of R40 Eq. 72, weak-field f ~ 1.
        gaussian_candidate = sqrt(4 * pi) * candidate
        acceleration = gaussian_candidate / self.dielectric_mass_kg
        sheet_density = acceleration / (4 * pi * G)
        return {
            "voltage_V": voltage_V,
            "field_V_per_m": field,
            "free_electrode_charge_C": self.capacitance_F * voltage_V,
            "bound_surface_charge_magnitude_C": (
                EPSILON_0 * (self.relative_permittivity - 1) * abs(field) * self.area_m2
            ),
            "energy_J": energy,
            "internal_pressure_Pa": pressure,
            "internal_attraction_N": pressure * self.area_m2,
            "isolated_static_maxwell_self_force_N": 0.0,
            "external_energy_import_weight_scale_N": energy / C_LIGHT**2 * G_STANDARD,
            "r43_candidate_force_N": candidate,
            "r40_gaussian_converted_candidate_force_N": gaussian_candidate,
            "r40_weak_field_junction_sheet_density_kg_m2": sheet_density,
            "r40_junction_sheet_area_scale_kg": sheet_density * self.area_m2,
        }


def signed_amplitude_interval(mean_N, uncertainty_N, prediction_N, multiplier=3.0):
    """Allowed lambda in mean +/- multiplier*uncertainty for F=lambda*prediction."""
    if not all(isfinite(v) for v in (mean_N, uncertainty_N, prediction_N, multiplier)):
        raise ValueError("Inputs must be finite")
    if uncertainty_N < 0 or prediction_N == 0 or multiplier <= 0:
        raise ValueError("Require nonnegative uncertainty, nonzero prediction, positive multiplier")
    endpoints = [(mean_N + s * multiplier * uncertainty_N) / prediction_N for s in (-1, 1)]
    return sorted(endpoints)


def stray_capacitance_force(voltage_V, capacitance_gradient_F_per_m):
    """Fixed-voltage force on a subsystem relative to an external conductor."""
    return 0.5 * voltage_V**2 * capacitance_gradient_F_per_m
