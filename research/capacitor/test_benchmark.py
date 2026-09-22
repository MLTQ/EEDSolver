"""Scientific invariants and independent checks, not GPU regression tests."""

import unittest
from dataclasses import replace
from math import pi, sqrt

import numpy as np
from model import EPSILON_0, Capacitor, G, signed_amplitude_interval, stray_capacitance_force
from stress import sphere_force


class CapacitorChecks(unittest.TestCase):
    def setUp(self):
        self.cap = Capacitor(0.035, 0.0015, 2.1, 2200)

    def test_virtual_work_at_fixed_charge_and_voltage(self):
        cap = self.cap
        step = cap.thickness_m * 1e-5
        plus = replace(cap, thickness_m=cap.thickness_m + step)
        minus = replace(cap, thickness_m=cap.thickness_m - step)
        voltage = 10000
        charge = cap.capacitance_F * voltage
        force_q = -(
            charge**2 / (2 * plus.capacitance_F) - charge**2 / (2 * minus.capacitance_F)
        ) / (2 * step)
        # At fixed V use U-QV=-CV^2/2, including source work.
        force_v = (plus.capacitance_F - minus.capacitance_F) * voltage**2 / (4 * step)
        expected = -cap.at_voltage(voltage)["internal_attraction_N"]
        self.assertAlmostEqual(force_q / expected, 1, places=8)
        self.assertAlmostEqual(force_v / expected, 1, places=8)

    def test_polarity_and_scaling(self):
        a = self.cap.at_voltage(10000)
        b = self.cap.at_voltage(-10000)
        c = self.cap.at_voltage(20000)
        self.assertEqual(a["energy_J"], b["energy_J"])
        self.assertEqual(a["internal_attraction_N"], b["internal_attraction_N"])
        self.assertEqual(a["r43_candidate_force_N"], -b["r43_candidate_force_N"])
        self.assertAlmostEqual(c["internal_attraction_N"] / a["internal_attraction_N"], 4)
        thicker = replace(self.cap, thickness_m=0.003).at_voltage(10000)
        self.assertAlmostEqual(thicker["r43_candidate_force_N"], a["r43_candidate_force_N"])
        self.assertAlmostEqual(thicker["energy_J"] / a["energy_J"], 0.5)

    def test_published_table_reproduction(self):
        predicted = self.cap.at_voltage(10000)["r43_candidate_force_N"]
        self.assertLess(abs(predicted / 746e-9 - 1), 0.002)

    def test_independent_gaussian_to_si_conversion(self):
        # Convert each original cgs input, rather than reusing the SI sqrt(4pi) formula.
        g_cgs = G * 1000  # cm^3/(g s^2)
        voltage_statvolt = 10000 / 299.792458
        acceleration_cm_s2 = sqrt(g_cgs * 2.1) * voltage_statvolt / 0.15
        force_N = acceleration_cm_s2 * 0.01 * self.cap.dielectric_mass_kg
        from_si = self.cap.at_voltage(10000)["r40_gaussian_converted_candidate_force_N"]
        self.assertAlmostEqual(force_N / from_si, 1, places=8)

    def test_bound_signs_and_conventional_boundary_force(self):
        self.assertEqual(signed_amplitude_interval(2, 1, 10), [-0.1, 0.5])
        self.assertEqual(signed_amplitude_interval(2, 1, -10), [-0.5, 0.1])
        self.assertAlmostEqual(stray_capacitance_force(10000, 2e-14), 1e-6)
        with self.assertRaises(ValueError):
            Capacitor(0.035, 0, 2.1, 2200)


class StressChecks(unittest.TestCase):
    def test_surface_enclosing_one_charge_recovers_pair_force(self):
        q1, q2, separation = 1e-9, -2e-9, 0.04
        result = sphere_force([q1, q2], [[0, 0, 0], [0, 0, separation]], [0, 0, 0], 0.01, 32)
        expected_z = -q1 * q2 / (4 * pi * EPSILON_0 * separation**2)
        np.testing.assert_allclose(result["force_N"], [0, 0, expected_z], atol=1e-15)
        self.assertAlmostEqual(result["enclosed_charge_from_flux_C"] / q1, 1, places=12)

    def test_whole_source_cancellation_and_convergence(self):
        q = [1e-9, -0.7e-9, -0.3e-9]
        pos = [[0.005, 0.008, 0.013], [-0.011, 0.002, -0.007], [0.002, -0.009, 0.004]]
        coarse = sphere_force(q, pos, [0, 0, 0], 0.025, 8)
        fine = sphere_force(q, pos, [0, 0, 0], 0.025, 32)
        self.assertLess(np.linalg.norm(fine["force_N"]), 1e-14)
        self.assertLess(np.linalg.norm(fine["force_N"]), np.linalg.norm(coarse["force_N"]) / 1000)
        for radius in (0.04, 0.06):
            result = sphere_force(q, pos, [0.001, -0.002, 0], radius, 32)
            self.assertLess(np.linalg.norm(result["force_N"]), 1e-14)

    def test_external_field_transfers_momentum(self):
        external = np.array([20.0, -10.0, 30.0])
        q = [1e-9, -0.2e-9]
        result = sphere_force(
            q,
            [[0.002, 0, 0], [-0.003, 0.001, 0]],
            [0, 0, 0],
            0.02,
            32,
            lambda points: np.broadcast_to(external, points.shape),
        )
        np.testing.assert_allclose(result["force_N"], sum(q) * external, atol=1e-15)

    def test_surface_charge_rejected(self):
        with self.assertRaises(ValueError):
            sphere_force([1e-9], [[0, 0, 0.02]], [0, 0, 0], 0.02)


if __name__ == "__main__":
    unittest.main()
