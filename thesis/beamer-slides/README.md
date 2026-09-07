# Beamer thesis defense deck

This directory contains the 16:9 Beamer deck for the thesis
*Adaptivity Bounds for the Stochastic Traveling Salesperson Problem*.

## Files

- `main.tex` - 24 main-talk frames followed by 9 backup frames
- `diagrams.tex` - native TikZ figures
- `beamerthemethesis.sty` - minimal Beamer theme
- `build/main.pdf` - compiled deck

The stochastic-TSP frame has four Beamer overlay states, so the compiled PDF
contains 36 physical pages while the displayed frame numbering remains unchanged.

## Build

From this directory, run:

```sh
latexmk -pdf main.tex
```

The local `.latexmkrc` forces pdfLaTeX and writes auxiliary files to `build/`.
The root thesis build configuration is not used.
