# Progress

Created: 2026-06-03

## Current Orientation

The working focus is proof development for stochastic TSP. Solver-side work is
out of scope unless explicitly requested. The survival-local packet DP section
of `thesis/main.tex` is also out of scope unless explicitly requested.

The core definitions from `thesis/main.tex` are now captured in `CODEX.md`:
metric space with depot and clients, independent activations, a posteriori
TSP, a priori TSP, adaptive TSP, and the benchmark ordering
`\optpost \le \optadapt \le \optprior`.

## Thesis State Read So Far

- Sections 1 and 2 define the stochastic TSP model and the three benchmarks.
- Section 3 defines the adaptivity, clairvoyance, and obliviousness gaps.
- The thesis currently contains a symmetric cycle-subdivision construction
  claiming an adaptivity gap lower bound tending to `5/4`.
- The thesis currently contains an asymmetric recursive construction claiming a
  clairvoyance gap lower bound tending to `4/3`.
- Earlier notes inside an `\iffalse` block should be treated as scratch unless
  the user asks to revive them.

## Active Theorem Work

No active theorem prompt yet.

Round count for current theorem: 0 of 3.

## Review Log

No external-agent proof review has been run yet.

## Next Step

Wait for the user's next theorem/proof prompt. Then work on the theorem until a
substantive proof, counterexample, corrected statement, or other stable result
is reached; send the TeX proof to a separate reviewing agent; record the
verdict here; and continue for up to three rounds before asking for human
review.

