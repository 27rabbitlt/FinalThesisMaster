# Stochastic TSP thesis defense slides

This folder contains the Beamer source and the compiled 16:9 defense deck for
the thesis *Information Gaps and Algorithms for the Stochastic Traveling
Salesperson Problem*.

## Files

- `main.tex` — the 42-slide talk
- `diagrams.tex` — native TikZ figures used throughout the deck
- `beamerthemestochastic.sty` — custom typography, restrained academic palette,
  section pages, and page-number footer
- `stochastic-tsp-thesis-talk.pdf` — compiled presentation

## Timing map

The title, outline, and final takeaway frame are outside the requested technical
timing.

| Segment | Slides | Target |
|---|---:|---:|
| Problem and three benchmarks | 3–5 | 3 min |
| Gap constructions | 6–18 | 13 min |
| Backbone circulation and rounding | 19–33 | 15 min |
| A priori dynamic programming | 34–41 | 10 min |

The outline is slide 2 (about 30 seconds), and the takeaway is slide 42.

Individual timing cues are kept as comments immediately before each frame in
`main.tex`.

## Build

From this directory, run:

```sh
latexmk -pdf main.tex
```

The source uses `pdflatex`, Source Sans Pro with serif mathematics, Beamer, and
TikZ. To remove auxiliary build products:

```sh
latexmk -c
```

All explanatory graphics are vector TikZ drawings, so no external image assets
are required.
