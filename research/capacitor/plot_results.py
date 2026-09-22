"""Render the benchmark with model curves distinct from the single measured datum."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def render():
    result = json.loads((ROOT / "results.json").read_text())
    config = json.loads((ROOT / "geometry.json").read_text())
    curve = [r for r in result["voltage_curve"] if r["voltage_V"] >= 0]
    voltage = [r["voltage_V"] / 1000 for r in curve]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "eed-capacitor-benchmark",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.1), layout="constrained")
    fig.suptitle(
        "Static PTFE capacitor: internal load, proposed thrust, and force accounting",
        fontsize=16,
        fontweight="bold",
    )
    ax = axes[0]
    ax.plot(voltage, [r["internal_attraction_N"] for r in curve], color="#246b8e", lw=2.5)
    ax.set(
        title="Internal attraction between plates",
        xlabel="Applied voltage (kV)",
        ylabel="Internal attraction (N)",
        xlim=(0, 10.5),
        ylim=(0, 0.46),
    )
    ax.text(
        0.05,
        0.92,
        "0.398 N at 10 kV\nOpposed by the assembly's supports",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
    )
    ax.text(
        0.05,
        0.69,
        "Leading parallel-plate estimate\nThis is not whole-apparatus thrust",
        transform=ax.transAxes,
        fontsize=9,
        color="#465360",
    )
    ax = axes[1]
    ax.plot(
        voltage,
        [r["r43_candidate_force_N"] * 1e6 for r in curve],
        color="#c06329",
        lw=2.3,
        label="R43 Eq. 1 candidate",
    )
    ax.plot(
        voltage,
        [r["r40_gaussian_converted_candidate_force_N"] * 1e6 for r in curve],
        color="#86569c",
        lw=2.3,
        label="R40 direct cgs conversion",
    )
    ax.axhline(0, color="#246b8e", lw=1.5, label="Isolated Maxwell self-force")
    ax.errorbar(
        [10],
        [config["measured_vertical_force_N"] * 1e6],
        yerr=[3 * config["reported_vertical_uncertainty_N"] * 1e6],
        fmt="o",
        color="#172c3f",
        capsize=6,
        label="Published datum, +/-3u",
    )
    ax.set(
        title="Candidate net force vs measured result",
        xlabel="Applied voltage (kV)",
        ylabel="Vertical force (micronewtons)",
        xlim=(0, 10.8),
        ylim=(-0.2, 3.0),
    )
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax = axes[2]
    for radius, color in zip((0.025, 0.04, 0.06), ("#246b8e", "#c06329", "#86569c")):
        rows = [r for r in result["stress_convergence"] if r["radius_m"] == radius]
        residual = np.maximum([r["force_norm_N"] for r in rows], 1e-24)
        ax.loglog(
            [r["order"] for r in rows],
            residual,
            "o-",
            color=color,
            label=f"Sphere radius {radius * 1000:g} mm",
        )
    ax.set(
        title="Stress check: synthetic point charges",
        xlabel="Polar quadrature order",
        ylabel="Closed-source force residual (N)",
    )
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    for ax in axes:
        ax.grid(alpha=0.16)
    fig.supxlabel(
        "35 mm diameter | 1.5 mm PTFE | relative permittivity 2.1 | "
        "one experimental point at 10 kV; all curves are calculations",
        fontsize=10,
    )
    for suffix in ("png", "svg"):
        path = ROOT / f"force_comparison.{suffix}"
        metadata = {"Date": None} if suffix == "svg" else None
        fig.savefig(path, dpi=180, metadata=metadata)
        if suffix == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
    plt.close(fig)


if __name__ == "__main__":
    render()
