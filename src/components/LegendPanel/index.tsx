/**
 * LegendPanel — collapsible right-hand reference sidebar.
 *
 * Explains every field chip, EED coupling parameter, solver term,
 * GEM sector concept, and geometry type available in Oracle.
 * Toggled by the ⟩ / ⟨ tab on its left edge.
 */

import { useState } from "react";

interface Props {
  open:     boolean;
  onToggle: () => void;
}

export function LegendPanel({ open, onToggle }: Props) {
  return (
    <aside
      className={`shrink-0 flex flex-row border-l border-rim transition-all duration-200 ${
        open ? "w-72" : "w-7"
      } bg-panel overflow-hidden`}
    >
      {/* ── Toggle tab ───────────────────────────────────────────────── */}
      <button
        onClick={onToggle}
        title={open ? "Close legend" : "Open field legend"}
        className="w-7 shrink-0 flex items-center justify-center border-r border-rim
                   text-slate-600 hover:text-slate-300 hover:bg-white/5 transition-colors
                   cursor-pointer self-stretch"
      >
        <span className="text-xs select-none">{open ? "⟩" : "⟨"}</span>
      </button>

      {/* ── Content ──────────────────────────────────────────────────── */}
      {open && (
        <div className="flex-1 overflow-y-auto px-3 py-4 flex flex-col gap-5 min-w-0">
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-widest">
            Legend
          </div>

          <LegendSection title="Output fields">
            <FieldEntry chip="|B|" unit="T" name="Magnetic field">
              B = ∇×A. The conventional observable field. Closed DC loops concentrate
              B along the axis; toroids confine it inside the torus tube.
            </FieldEntry>
            <FieldEntry chip="|A|" unit="V·s/m" name="Vector potential">
              Primary EED state variable. Unlike in standard Maxwell, A is physically
              meaningful everywhere — the Aharonov-Bohm effect shows a particle
              acquires phase ∮ A·dl even where B = 0.
            </FieldEntry>
            <FieldEntry chip="C" unit="m⁻¹" name="EED scalar (deleted DOF)">
              C = ∇·A + (1/c²) ∂φ/∂t — the "seventh degree of freedom" that Maxwell
              set to zero by gauge choice. In EED it obeys □C = 0 (propagates at c)
              and is sourced by ∇·J ≠ 0. Zero for any closed DC loop; non-zero for
              open helices, capacitors, and AC-driven sources.
            </FieldEntry>
            <FieldEntry chip="φ" unit="V" name="Scalar potential">
              Electric scalar potential. Always zero for closed DC loops (Gauss's law
              + ∇·J = 0 → no free charge). Becomes non-zero for capacitors (charge on
              plates), open helices at wire tips, and any AC-driven circuit where
              charge accumulates.
            </FieldEntry>
            <FieldEntry chip="|P|" unit="W/m²" name="Poynting flux">
              Energy flow per unit area: P = E×B in Maxwell; in EED the C field
              contributes an additional term. High |P| regions show where field energy
              is being transported.
            </FieldEntry>
            <FieldEntry chip="u" unit="J/m³" name="Energy density">
              u = ½ε₀(E² + c²B² + c²C²). The C² term is EED's extra stored energy —
              absent in standard Maxwell. Comparing u with and without γ {'>'} 0 quantifies
              the EED contribution.
            </FieldEntry>
            <FieldEntry chip="Φ_g" unit="m²/s²" name="Gravitomagnetic scalar">
              GEM sector only. Scalar gravitomagnetic potential sourced by the C field
              via the κ_g coupling constant. Analogous to φ but for the gravitational
              sector. Only non-zero when GEM is enabled and C ≠ 0.
            </FieldEntry>
          </LegendSection>

          <LegendSection title="EED coupling">
            <ParamEntry param="γ (gamma)">
              The main EED dial. 1 = full Stueckelberg EED (φ and A fully coupled
              through C); 0 = standard Maxwell (C is suppressed). Increase from 0 to
              watch the C field switch on.
            </ParamEntry>
            <ParamEntry param="α (alpha)">
              Scalar mass of the C field [1/m]. Gives C a Yukawa-style finite range
              ~ 1/α. At α = 0 the C field is massless and propagates to infinity.
              Large α confines C-field effects to the near-source region.
            </ParamEntry>
            <ParamEntry param="β (beta)">
              A→φ back-reaction coupling. Controls how much the vector potential
              feeds back into the scalar potential dynamics. 0 = decoupled.
            </ParamEntry>
          </LegendSection>

          <LegendSection title="Solver">
            <ParamEntry param="Lorenz gauge">
              Forces C ≡ 0 at every step — exact Maxwell behaviour. Use as a null
              hypothesis: if a result disappears when you toggle this on, it is a
              genuine EED effect rather than a numerical artefact.
            </ParamEntry>
            <ParamEntry param="FDTD (time-domain)">
              Leapfrog symplectic Euler scheme that evolves A and φ forward in
              time. Required to see C-field wave propagation, AC source injection,
              and the approach to steady state. The timestep dt is auto-set to the
              CFL stability limit: dx / (c√3).
            </ParamEntry>
            <ParamEntry param="Steps">
              Number of FDTD timesteps to run. Each step covers dt ≈ 0.2–0.5 ps
              for typical domain sizes. More steps reveal wave propagation further
              from the source and approach to equilibrium.
            </ParamEntry>
            <ParamEntry param="3-D volume">
              Extracts a full scalar field volume for the ray-march renderer.
              Disable to speed up solves when you only need the 2-D slices or
              numerical maxima.
            </ParamEntry>
          </LegendSection>

          <LegendSection title="GEM sector">
            <ParamEntry param="κ_g">
              Coupling constant linking the C field to the gravitomagnetic sector.
              Two physical reference values: Kaluza-Klein theory ≈ 7.4×10⁻²⁸
              (unmeasurable); Li-Torr experiment ≈ 1.14×10⁻¹¹ (disputed).
              Both give Φ_g amplitudes ~10⁻²⁰–10⁻²⁵ m²/s² — below any
              colormap resolution. <em>To see GEM effects in the viewer, set
              κ_g ≥ 1e-3 as an exploratory amplification.</em> Set to 0 to
              disable GEM. Requires FDTD + AC open helix + γ {'>'} 0 to drive C ≠ 0.
            </ParamEntry>
            <ParamEntry param="Li-Torr mode">
              Models the Tajmar / Li-Torr result: a rotating superconductor generates
              a gravitomagnetic field via the London moment,
              B_g = −(2mₑ/e) ω. Requires GEM enabled and the coil marked
              superconducting.
            </ParamEntry>
            <ParamEntry param="Superconducting toggle">
              Marks a coil as a superconductor for Li-Torr GEM coupling. Has no
              effect on the EM solve itself — only activates the gravitomagnetic
              London-moment source term.
            </ParamEntry>
          </LegendSection>

          <LegendSection title="Source geometry">
            <ParamEntry param="Solenoid">
              Closed helical coil. Produces uniform axial B inside; A non-zero
              outside. φ = 0 for DC because ∇·J = 0.
            </ParamEntry>
            <ParamEntry param="Toroid">
              Toroidal winding. B is completely confined inside the torus tube.
              A is non-zero in the field-free hole — the canonical Aharonov-Bohm
              geometry. ∮ A·dl ≠ 0 through the hole even though B = 0 there.
            </ParamEntry>
            <ParamEntry param="Open helix">
              Non-closed wire — current enters one tip and exits the other.
              ∇·J ≠ 0 at the tips → charge accumulates → φ ≠ 0. The effect is
              strongest with AC drive (frequency_hz {'>'} 0) where charge sloshes
              back and forth, continuously sourcing C.
            </ParamEntry>
            <ParamEntry param="Capacitor — symmetric">
              Parallel-plate geometry. φ is initialised as a linear ramp between
              the two equal-area plates. Uniform E between plates; C sourced at
              the plate surfaces where ∇·E ≠ 0.
            </ParamEntry>
            <ParamEntry param="Capacitor — TTB asymmetric">
              Townsend-Brown configuration: large plate (anode) + small pointed
              electrode (cathode). The asymmetry creates a strongly non-uniform φ
              and E field — the geometry used in TTB thrust experiments. The
              Asymmetry slider controls the large/small radius ratio.
            </ParamEntry>
          </LegendSection>

          <LegendSection title="Holonomy  ∮ A·dl">
            <p className="text-xs text-slate-500 leading-relaxed">
              Line integral of the vector potential around a closed loop. Equal to
              the magnetic flux through the loop (Stokes' theorem) inside a
              solenoid, but non-zero <em>outside</em> — even where B = 0.
              This is the Aharonov-Bohm phase, measured in V·s (= Wb).
              Add loop paths below the geometry controls to compute it.
            </p>
          </LegendSection>
        </div>
      )}
    </aside>
  );
}

// ── Sub-components ─────────────────────────────────────────────────────────────

function LegendSection({ title, children }: { title: string; children: React.ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);
  return (
    <div>
      <button
        onClick={() => setCollapsed(c => !c)}
        className="w-full flex items-center justify-between mb-2 group"
      >
        <span className="text-[10px] font-semibold uppercase tracking-widest text-slate-500
                         group-hover:text-slate-300 transition-colors">
          {title}
        </span>
        <span className="text-slate-700 group-hover:text-slate-400 text-xs transition-colors">
          {collapsed ? "▸" : "▾"}
        </span>
      </button>
      {!collapsed && (
        <div className="flex flex-col gap-3">
          {children}
        </div>
      )}
    </div>
  );
}

function FieldEntry({
  chip, unit, name, children,
}: {
  chip: string; unit: string; name: string; children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-0.5">
      <div className="flex items-baseline gap-2">
        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-200 shrink-0">
          {chip}
        </span>
        <span className="text-xs text-slate-300 font-medium">{name}</span>
        <span className="text-[10px] text-slate-600 ml-auto shrink-0">{unit}</span>
      </div>
      <p className="text-[11px] text-slate-500 leading-relaxed pl-0.5">{children}</p>
    </div>
  );
}

function ParamEntry({ param, children }: { param: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-xs text-slate-300 font-medium">{param}</span>
      <p className="text-[11px] text-slate-500 leading-relaxed pl-0.5">{children}</p>
    </div>
  );
}

// React import for JSX
import React from "react";
