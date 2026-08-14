# AGENTS.md — Final Thesis Master Project

This is the root of a master's thesis project on stochastic TSP (adaptive vs a posteriori vs a priori).

## Project Structure

- `thesis/` — Thesis LaTeX/markdown sources (`main.tex`, `.bib`, `.latexmkrc`, notes)
- `post-adapt-solver/` — Exact solver for a posteriori, adaptive, and a priori stochastic TSP (see `post-adapt-solver/AGENTS.md` for details)
- `scripts/` — Utility scripts
- `daily-notes/` — Daily research notes (Chinese and English versions). Format: `YYYY-MM-DD-en.md` and `YYYY-MM-DD-zh.md`

## Zotero papers

There are related papers stored in zotero under /Users/liuteng/Zotero/storage. 
However the papers are stored then in each folder with a hased name so its 
difficult to tell which paper is stored where, you have to search for it by yourself.

## Problem definition

Under `thesis` folder, there is a `main.pdf` providing important problem background and definition.
The text is stored in main.tex, main section 1 2 and 3.

## Thesis notation conventions

- Assume every non-root vertex has activation probability in $(0,1]$.
  Vertices with activation probability $0$ are deleted from the instance.
- In asymptotic notation such as $O(\cdot)$, $\Omega(\cdot)$, and
  $\Theta(\cdot)$, write $n$, never $n+1$.  For example, write
  $O(\log^2 n)$ rather than $O(\log^2(n+1))$.  The expression $n+1$ may still
  be used in explicit, non-asymptotic inequalities when needed for small
  values of~$n$.
- In Chapter 6, use 'vertex' and 'vertices' throughout, never 'client' or
  'clients'.
- In Chapter 6, do not write $n+1$ anywhere, including in explicit bounds.
  Handle the cases $n\le1$ separately and use $n$ in logarithmic bounds.
- In Chapter 6, use 'asymmetric' rather than 'directed' for the metric,
  triangle inequality, and shortcutting.  Describe combinatorial orientation
  with arcs, ordered pairs, or the tour order without using 'directed'.
- In Chapter 6, reserve 'chord' for a collapsed outside-vertex attachment or
  an auxiliary rounded copy, 'arc' for a formal flow/multigraph arc, and
  'edge' for a backbone edge.  Do not use 'arc' and 'chord' interchangeably.
