# Asymmetric clairvoyance-gap graph families

This folder records a proof-first investigation of structured graph families
for improving the asymmetric stochastic-TSP clairvoyance-gap lower bound of
\(4/3\).  The investigation produced a complete construction whose proved
gap has liminf at least \(3/2\).

## Main result

- `four-star-layered-poset-construction.md`: complete construction and proof.
  A two-lane layered switching/design interface has four independent
  row/column-star selectors per stage.  Its posterior expected width is
  \(2+o(1)\), whereas every causal policy has expected increasing-run count
  at least \(3-o(1)\).  A positive directed poset metric transfers this to
  stochastic TSP.  Its explicit witness has \(120002\) clients and proves
  \(3\operatorname{OPT}_{\rm adapt}
    -4\operatorname{OPT}_{\rm post}>0.24\).
- `four-star-poset-finite-certificate.md`: a fully finite analytic witness
  from an independent audit, with \(336002\) clients, \(q=1/200\),
  \(L=56000\), and
  \(\varepsilon=10^{-10}\).  It proves
  \(3\operatorname{OPT}_{\rm adapt}
    -4\operatorname{OPT}_{\rm post}>0.10826\), hence a strict gap above
  \(4/3\).

## Files

- `problem-brief.md`: shared model, baseline, and proof obligations.
- `composition-principles.md`: cross-family eliminators and route-code
  abstraction.
- One note per candidate family, plus detailed audits of intermediate
  constructions and obstructions.
- `comparison.md`: final comparison and recommended next directions.
- `audit.md`: independent model-level proof audit and corrections.

The investigation deliberately does not use exhaustive search on small
instances.  The proposed families are asymptotic, and a small instance can
miss both the routing structure and the information-theoretic obstruction.
