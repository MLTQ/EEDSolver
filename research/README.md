# Electrodynamics and gravity: research collection

Collected **22 September 2026**. Start here, then use the [annotated paper library](PAPERS.md), [BibTeX bibliography](references.bib), and [repository audit](REPO_AUDIT.md).

This is a targeted collection of 43 sources, including papers, reviews, technical reports, a historical patent, and an author-published paper with its errata. It is a starting research map, not a systematic review, an independent replication of every derivation, or a solution to unification. Publication and citation counts are not evidence that a proposed effect exists.

**Your starting point:** Deleted Degrees of Freedom and Brown’s static solid-dielectric claim. Start with the [focused Brown assessment](BROWN_STATIC_DIELECTRICS.md), which distinguishes this question from ionic-wind lifters and compares directly relevant experiments.

## What the literature supports

Electromagnetism and gravity already interact in general relativity: electromagnetic energy, momentum, and stress contribute to spacetime curvature, and curved spacetime affects electromagnetic propagation. A useful next question is whether a proposed extension predicts a reproducible effect beyond that baseline. Wheeler's [geons](https://doi.org/10.1103/PhysRev.97.511) and the [Gertsenshtein calculation](https://jetp.ras.ru/cgi-bin/dn/e_014_01_0084.pdf) make the standard interaction concrete.

Several different research questions are often called “unification”:

| Question | Starting sources | What the connection establishes |
|---|---|---|
| How does EM source gravity in ordinary GR? | R20–R24 | Stress-energy coupling and predicted wave conversion; no new scalar needed. |
| Can a higher-dimensional geometry contain both fields? | R25 | Kaluza–Klein dimensional reduction; the physical extra dimension is unconfirmed. |
| Are gauge theory and gravity mathematically related? | R26–R27 | Double-copy relations for amplitudes and classes of solutions; no automatic laboratory conversion law. |
| Could a physical scalar extend Maxwell theory? | R06–R17 | Different proposed actions, sources, and observables; not one agreed “EED.” |
| Does quantum consistency constrain the strengths of the forces? | R28 | The weak gravity conjecture; a conjectural constraint, not an experimental unification. |
| Do charged solid dielectrics show an additional steady force? | R37–R43 | Original claims, positive exploratory reports, and directly relevant null experiments with stated limits. |
| Does established physics already extend linear Maxwell theory? | R30–R31 | QED nonlinearities and curvature-dependent effective interactions. |

**My assessment:** for the repo’s scalar EED branch, the most useful nearby speculative lead is [Minotti and Modanese's scalar–tensor/AB model](https://arxiv.org/abs/2408.05230) (R15). It supplies an action and a weak-field approximation, whereas Oracle's direct scalar-to-gravity mapping is an additional assumption. Its proposed anomalous source and unknown parameters remain substantive obstacles, not implementation details.

## Read these first

1. **R01 — Aharonov & Bohm (1959):** identify what the interference effect actually measures.
2. **R10 — Modanese (2017):** make the extra-current assumption explicit.
3. **R15 — Minotti & Modanese (2025):** inspect the closest published EM–gravity proposal.
4. **R19 — Wilhelm's errata (2026):** trace corrections relevant to the repo's original motivation.
5. **R25 — Overduin & Wesson (1997):** understand what Kaluza–Klein reduction does and does not imply.
6. **R23 — Domcke & Garcia-Cely (2021):** examine a standard-theory EM–gravity conversion mechanism.
7. **R26 — Monteiro, O'Connell & White (2014):** explore the classical double copy.
8. **R37 → R38/R42/R43 → R40:** follow Brown’s static-dielectric claim through direct experiments and a concrete static gravitational proposal. R33 supplies broader equivalence-principle constraints. R36 is a gas-force control, not the decisive static-dielectric test.

For the EED lineage specifically, follow R02 → R06/R07 → R08 → R10/R12 → R15/R16. These papers overlap but do not all make identical physical assumptions. R05 is the terminology check for “Stueckelberg.”

## The first question worth settling

> With a physically complete, charge-conserving source and specified initial/boundary conditions, does a chosen extension predict a gauge-independent detector response that differs from Maxwell plus general relativity?

For the flat-space AB normalization described in the audit,

\[
\Box C=\mu_0\partial_\mu J^\mu,
\qquad C=\partial_\mu A^\mu.
\]

If the current is conserved **and** the scalar has zero initial data and no incoming boundary excitation, this channel gives \(C=0\). Conserved current alone does not eliminate arbitrary pre-existing homogeneous scalar solutions. This distinction follows from the source equation, not from a preference for either theory. See [R10](https://arxiv.org/abs/1609.00238) and the [audit derivation](REPO_AUDIT.md).

For a radiating-source AB investigation, a good first computational milestone would compare three cases under the same geometry: a conserved circuit, that circuit with a deliberately specified anomalous source, and a Maxwell/GR control. The result should include a detector observable, continuity residuals, boundary sensitivity, and convergence, with parameters fixed before looking at the output. A null result for the ordinary circuit would be useful progress.

For the clarified **static dielectric** target, first reproduce one capacitor configuration and derive its force on the complete apparatus, including polarization charge and the return circuit. The [Brown note](BROWN_STATIC_DIELECTRICS.md) specifies that milestone. A high dielectric constant alone does not supply an anomalous AB scalar source.

This research pass leaves the solver equations unchanged. Follow-up work is tracked in beads: ORC-j6i (static capacitor prediction), ORC-74h (Einstein–Maxwell baseline), ORC-h7j (scalar–tensor AB action), and ORC-xk8 (conserved-source control). The audit gives the scientific acceptance criteria.

## Scale matters

For a compact, isolated system whose total energy increases by \(U\), the far-field mass increase is approximately \(U/c^2\). A scale estimate is

\[
\Delta g\sim\frac{GU}{c^2r^2}.
\]

At \(U=1\,\mathrm{MJ}\) and \(r=1\,\mathrm{m}\), this is about \(7.4\times10^{-22}\,\mathrm{m\,s^{-2}}\). This is an illustrative far-field estimate, not a prediction for the repository's coil geometry. A real calculation includes the apparatus, supports, power supply, and stresses; moving energy around inside a system does not automatically increase its total mass. The point is to compare a claimed enhancement with a dimensional, parameter-free GR baseline. The governing framework is reviewed in [R32](https://arxiv.org/abs/1403.7377).

## Collection and access

- [PAPERS.md](PAPERS.md): grouped annotations, source links, access status, and available local PDFs.
- [references.bib](references.bib): 43 matching citation keys, R01–R43, for Zotero or a paper draft.
- [sources.json](sources.json): citation metadata, source URLs, review depth, pinned arXiv versions where available, retrieval status, page counts, and SHA-256 checksums.
- [papers/](papers/): local reading copies. Downloaded PDFs are excluded from git; the bibliography and retrieval manifest are committed. A fresh clone therefore needs to retrieve reading copies from the recorded URLs.

Primary author, journal, institutional, arXiv, patent and NASA records were used. Talley’s public report was read through an archival scan; that hosting provenance is explicitly recorded. Reviews are explicitly identified. Some older references were verified only at the metadata/abstract level, and no claim is made to have read all papers in full. An accessible abstract, a successful PDF download, and an independently checked derivation are three different levels of review.

Searches covered the repo's Woodside/Hively/Arbab/DDOF lineage, AB source conservation, scalar–tensor couplings, Kaluza–Klein, double copy, standard photon–graviton conversion, QED extensions, static solid-dielectric and electret claims, and experimental controls. The newest included journal article is dated 10 September 2026. This is not a comprehensive survey of string theory, loop quantum gravity, every historical unified-field program, or every experimental anomaly.

Manifest conventions: `year` is the listed publication year when known; `preprint_date` records the arXiv submission separately. `assessment` and `annotation` are this collection's appraisal, not publisher labels. `download.status` distinguishes an obtained file, a failed retrieval, and a reference for which no open PDF was selected. ArXiv PDFs may differ from the journal version of record. Preserve version and checksum when reproducing a calculation.
