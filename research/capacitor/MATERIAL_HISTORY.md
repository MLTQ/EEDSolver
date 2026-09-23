# Retained charge, asymmetric electrodes, and barium titanate

Max's next target is an asymmetric capacitor with barium titanate dielectric. The accompanying hypothesis is that PTFE electret buildup between runs could confound the earlier null measurements. That hypothesis is worth testing; the available evidence does not establish that it occurred or caused the reported null.

## What the protocol does and does not document

The reviewed **R43 arXiv v2**, section 2.1, describes repeated down/ramp/hold/ramp/down cycles, often more than 80 repetitions, monitored supply voltage/current, and thermal-drift/outlier processing. **R42**, section 3, describes repeated on/off profiles and linear drift fitting through off periods. I did not find an explicit per-run trapped-charge measurement, depolarization verification, or fresh-sample versus conditioned-sample comparison in those descriptions. That is a limit on the documented controls, not proof that the authors ignored charging effects. [R43](https://arxiv.org/abs/2402.15640v2), [R42](https://doi.org/10.1016/j.elstat.2020.103477).

PTFE can hold stable electret charge; electrode material and charging conditions affect retention. R44 studies deliberately corona-charged films, so it supports the mechanism without establishing its magnitude in these millimeter-thick, epoxy-coated capacitor samples. [Rabiee, Sohrabi & Afarideh (2024)](https://doi.org/10.1016/j.apradiso.2024.111187).

The earlier tests also extend beyond PTFE:

| R43 configuration | Applied voltage | Vertical result |
|---|---:|---:|
| Asymmetric PTFE | 10 kV | 6.8 ± 7.7 nN |
| Asymmetric PZT-5H | 2 kV | −1.2 ± 16.7 nN |
| Symmetric commercial Y5T ceramic | 30 kV | −9.5 ± 13.5 nN |

PZT is not barium titanate. Y5T is an electrical classification, not a chemical composition certificate; neither the exact BaTiO₃ formulation nor an asymmetric version of that commercial device is established here. These rows are relevant constraints, not an exact match to the proposed material/geometry. Source: R43, Table 1.

## How material memory could affect the inference

The distinction to make is between the force itself and the **change in force recovered by a measurement protocol**:

\[
F_{\rm raw}(t)=F_{\rm reversible}[V(t)]+F_{\rm history}[P,\rho_{\rm trap},t]
 +F_{\rm environment}(t)+F_{\rm drift}(t).
\]

An on/off subtraction rejects a constant contribution present in both states. Drift removal can also attenuate a slowly evolving contribution. To claim that a real signal was removed, however, we need its predicted time dependence and the actual processing transfer function. Retained charge may instead create an offset, drift or false positive. Its existence alone does not determine which occurs.

There is also a useful constraint from voltage control. In an ideal homogeneous planar slab with uniform frozen remanent polarization, fixing the terminal voltage fixes the field integral:

\[
\int_0^d E_z(z)\,dz=V_1-V_2.
\]

For uniform polarization and no volume free charge, \(E_z=(V_1-V_2)/d\); the electrodes adjust their free charge. Nonuniform trapped charge can redistribute the field but still obeys the voltage integral. Thus, under the earlier candidate's **constant material coefficient** and uniform mass density, a force proportional to \(\int E_z dz\) does not disappear merely because a frozen polarization has accumulated. A different coupling to polarization, a varying coefficient, an asymmetric geometry or an uncontrolled/floating off-state is a different calculation and must be specified.

Shorting terminals is not equivalent to erasing all trapped charge or ferroelectric domains. Conversely, switching a supply off does not by itself establish a controlled zero-voltage state. Both the terminal boundary condition and the material state need verification.

The [first benchmark](README.md) remains a useful reversible-dielectric and force-accounting control. It does not include history-dependent polarization, charge injection, drift-removal filtering, or the proposed asymmetric BaTiO₃ device. Its quoted experimental constraint is conditional on the stated force law and recovered observable.

## Why BaTiO₃ needs a different material model

Poled BaTiO₃ ceramics show rate-dependent remanent polarization, hysteresis and electromechanical strain. Its dielectric response depends on composition, phase, grain structure, temperature, bias and history. R45 directly measures these effects. Replacing PTFE's constant permittivity with a large nominal number would miss the behavior motivating this change. [Kannan, Trassin & Kochmann (2022)](https://doi.org/10.1016/j.mtla.2022.101553).

Charge trapping and conduction do not disappear either. R46 reports high-field, trap-related conduction in thin BaTiO₃ crystals and compares multilayer ceramics. It establishes a mechanism to consider, not the conductivity of an unspecified bulk sample. [Morrison et al. (2005)](https://arxiv.org/abs/cond-mat/0409585).

The material equations should therefore have the form

\[
\mathbf D=\epsilon_0\mathbf E+\mathbf P,\quad
\mathbf P=\mathcal P[\mathbf E,T,\text{stress},\text{history}],\quad
\partial_t\rho_{\rm free}+\nabla\cdot\mathbf J_{\rm free}=0,
\]

with trapping/detrapping exchanges conserved across the mobile and trapped charge populations, plus a measured conduction law and an elastic/electrostrictive response. Polarization current remains part of charge conservation. No extra gravitational coupling follows solely from using a ferroelectric.

## Definition of the next comparison

The initial geometry family is a solid BaTiO₃ body between **unequal-area metal electrodes**, with a matched equal-electrode control. A deliberately semiconducting ceramic is a separate branch: conductivity changes space charge, leakage, heating and charge-relaxation time, and cannot be represented just by an electrode property. The user's intended meaning of “conductive” was asked separately; no specific grade or conductivity is assumed here.

Before calculating a force, specify electrode areas and edge radii, dielectric dimensions, electrode material/contact, composition/doping, temperature, and the actual voltage waveform. Calibrate material parameters to electrical and strain data before examining force. Do not invent an asymmetry multiplier for the old parallel-plate formula.

The decisive comparisons are:

- **State:** fresh or independently verified reset material versus positively and negatively conditioned material. Record post-cycle charge/voltage relaxation; elapsed waiting time alone does not certify a reset.
- **Polarity and geometry:** independently reverse voltage, remanent polarization and the device orientation. These are different operations and can separate geometric force signatures from material bias.
- **Time:** preserve the first cycle and cycle-by-cycle raw force, voltage, current, temperature and strain; compare them with the averaged result. Inject hypothetical slow and steady signals through the same drift-processing method to quantify what it would retain.
- **Boundary:** retain the complete apparatus and its supports/return paths in the momentum balance. A history-dependent conventional material still has no steady isolated Maxwell self-thrust; external interactions and mechanical deformation can change a balance reading.

This changes the research target without assuming the outcome: an explicit BaTiO₃ material history and asymmetric geometry, with a measurable prediction and a way to distinguish retained-charge effects from the proposed force.
