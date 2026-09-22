# Brown's static solid-dielectric claim

Research focus clarified by Max on 22 September 2026: **Deleted Degrees of Freedom and Thomas Townsend Brown's static dielectric effect**, rather than the fluid/ionic-wind effect used by lifters. The motivating title is confirmed; R18 is the located Wilhelm edition, with R19 its corrections.

The question is whether a charged solid dielectric produces a reproducible force, weight change, or external gravitational field beyond ordinary electromechanics and general relativity. Those are three different observables. A force on a capacitor alone does not identify gravity or demonstrate thrust of an isolated apparatus.

## Read this branch in order

| Source | Why it belongs here | What it establishes or leaves open |
|---|---|---|
| **R37 — [Brown, GB300311A (1928)](https://patents.google.com/patent/GB300311A/en)** | Original claim involving linked conductors and solid insulating slabs. | Defines a historical claim; patent publication is not experimental verification. |
| **R38 — [Talley (1991)](papers/R38.pdf)** | Capacitor tests near 1 microtorr; non-breakdown operation limited to roughly 19 kV. | No detected steady propulsion. The report's separate breakdown-associated ceramic observations must not be relabeled a static result. |
| **R39 — [Buehler (2004)](https://www.space-mixing-theory.com/article2.pdf)** | Positive exploratory report, including a wax-dielectric capacitor. | Apparent upward force reported in atmospheric tests; vacuum operation is an extrapolation, not demonstrated. |
| **R42 — [Tajmar & Schreiber (2020)](https://doi.org/10.1016/j.elstat.2020.103477)** | Explicit solid-dielectric, non-ionizing capacitor-force tests, including ceramic and PTFE. | Null up to 10 kV at approximately 2.9 µN accuracy. Directly relevant, within the tested parameter ranges. |
| **R43 — [Tajmar, Kößling & Neunzig (2024)](https://doi.org/10.1038/s41598-024-70286-w)** | Shielded, high-vacuum tests of steady fields, including high-permittivity, asymmetric and gradient dielectrics. | Stronger, configuration-specific null constraints; inspect Table 1 rather than assuming one sensitivity for every device. |
| **R40 — [Ivanov (2005)](https://arxiv.org/abs/gr-qc/0502047)** | A concrete static-field gravitational proposal using exact Einstein–Maxwell solutions. | The physical source and boundary interpretation needs checking against R42–R43; an exact metric does not establish a realizable propulsion device. |
| **R41 — [Schreiber & Tajmar (2016)](https://doi.org/10.2514/6.2016-4919)** | Related question: persistent polarization of solid electrets. | No confirmed polarization-induced weight change in their wax mixtures. Different chemistry limits its status as a replication of the original electret claim. |

The [2024 arXiv v2 reading copy](https://arxiv.org/abs/2402.15640v2) precedes the journal article. Table 1 includes symmetric PTFE at 10 kV, PZT-5H at 7 kV, and Y5T ceramic at 30 kV vertically. That Y5T result is −9.5 ± 13.5 nN; the authors use a 3σ decision criterion. Operation was usually near 10⁻⁷ mbar. Compare these separate configurations with a specific prediction, not an unrestricted claim about every dielectric.

R38's official DTIC download was unavailable during collection; the local file is a third-party-hosted scan of the public-release report, with its cover and report number checked. R36, the NASA ion-transport study, remains useful control literature but is not the main evidence for this branch.

## Where the proposed EED connection needs a derivation

The flat AB model considered in the [repo audit](REPO_AUDIT.md) has a scalar source proportional to total four-current nonconservation. Ordinary polarization does not automatically provide that source. In macroscopic notation,

\[
\rho_b=-\nabla\cdot\mathbf P,\qquad
\mathbf J_b=\partial_t\mathbf P+\nabla\times\mathbf M,
\]

so

\[
\partial_t\rho_b+\nabla\cdot\mathbf J_b=0.
\]

This identity includes surface bound charge when polarization is treated consistently across interfaces. Free charge and its complete circuit must also satisfy continuity. At static equilibrium, fixed polarization is not a time-dependent source merely because its magnitude or permittivity is large.

Consequently, for that AB model, a conserved complete source with zero scalar initial data and no incoming scalar excitation does not generate the proposed scalar channel. This does **not** prove every alternative EM–gravity model impossible: an alternative action can couple to other invariants, and nonzero homogeneous scalar data is a separate assumption. It does mean we cannot obtain a Brown prediction just by omitting charge density, retaining a nonzero divergence of numerical current, or multiplying a scalar potential by a chosen gravity constant. Read [R19's corrections](https://advanced-rediscovery.com/research/deleted-degrees-of-freedom-errata) and [R15's actual action](https://arxiv.org/abs/2408.05230) before making that connection.

## A concrete next research milestone

**Produce a predicted force curve for one documented capacitor geometry, including the full apparatus, before changing the solver's gravity coupling.** Start with a published configuration so that the result can be compared with an existing sensitivity bound.

1. **Specify the observable.** Distinguish the force on the dielectric, force on the complete capacitor and supply, balance weight, and acceleration of a separate neutral test mass. State the reference frame and the proposed momentum exchange.
2. **Reproduce the ordinary baseline.** Model electrodes, dielectric constitutive response, supports, enclosure, supply and return path. Integrate Maxwell stress over a closed surface in surrounding vacuum for the enclosed system; include mechanical reaction forces and field momentum when switching. Interior stress partition in matter requires a consistent material model.
3. **State the extra theory completely.** Supply an action or closed equations, parameter units, matter coupling and boundary conditions. For Ivanov, check junction/source conditions and the zero-voltage reference. For scalar AB gravity, identify the additional scalar source or initial data explicitly.
4. **Predict discriminants in advance.** Voltage reversal and magnitude, apparatus rotation, material substitution, distance to a separate detector, and steady versus switching response should follow from the model. Dielectrophoresis, electrostriction, piezoelectric strain, leakage, thermal drift and enclosure attraction belong in the ordinary comparison.
5. **Compare like with like.** Match material, geometry, electric field, waveform and measured axis to a published test. If a prediction exceeds its measured bound, identify the conflicting assumption rather than tuning a coupling after seeing the result.

This is a tractable literature-and-calculation task. The collection does not establish a new gravitational effect, but it now identifies experiments that address the actual static claim and the mathematical steps needed to connect it to EED.
