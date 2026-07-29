# A two-lap ceiling for additive tournament-backbone closures

## 1. Local policies that matter for closure

Use the tournament chamber of
`construction-over-4-3-interacting-cycles.md`.  It has permanent open ports
\(a,b\), \(n\) selectors, selector probability \(p=\lambda/n\), and
parameter \(\varepsilon\).  Put
\[
 K\sim\operatorname {Binomial}(n,\lambda/n),\qquad
 Q:=\mathbb E(K-1)_+.
\]
Its exact posterior open value is
\[
                         P=2\varepsilon+Q.                \tag{1.1}
\]

Besides its strict one-piece lower bound, the chamber has the following two
explicit causal **upper** bounds.  They are the obstruction to closing many
copies on an additive backbone.

### One-piece policy

Choose a uniformly random permutation of the selectors, enter at \(a\),
call them in that order, and exit at \(b\).  Conditional on \(K=k\), every
consecutive active pair is in a uniformly random order.  Its expected
tournament distance is therefore
\[
                 \tfrac12\cdot1+\tfrac12\cdot2=\tfrac32.
\]
Thus the expected one-piece cost is
\[
                         U=2\varepsilon+\frac32Q.         \tag{1.2}
\]

### Two-piece policy

Choose a uniformly random selector permutation.

1. Enter at \(a\), call selectors until the first active one, move through
   it to \(b\), and end the piece.  If every selector is inactive, simply
   use \(a\to b\).
2. If uncalled selectors remain, enter again at \(a\) and continue the
   permutation until the second active selector.  From that selector onward,
   call every remaining selector in the same piece and exit at \(b\).

Inactive calls before the first active call of a piece cause no movement.
If \(K\leq1\), the cost is \(2\varepsilon\).  If \(K=k\geq2\), it is, in
expectation,
\[
                         4\varepsilon+\frac32(k-2).
\]
Writing
\[
 q:=\Pr[K\geq2],\qquad R:=\mathbb E(K-2)_+,
\]
the expected two-piece internal cost is therefore
\[
                         F=2\varepsilon+2\varepsilon q
                              +\frac32R.                 \tag{1.3}
\]
The identity
\[
                         Q=q+R                           \tag{1.4}
\]
will be useful.

### Local two-policy inequality

For the sparse parameters used by the tournament construction,
\[
                         2U+F\leq4P.                     \tag{1.5}
\]
Indeed, (1.1)--(1.4) give
\[
 2U+F-4P
 =-2\varepsilon(1-q)-q+\frac12R.                        \tag{1.6}
\]
It is enough that \(R\leq2q\).

For \(K\sim\operatorname {Binomial}(n,\lambda/n)\),
\[
 (K-2)_+\leq\binom K3,
\]
and hence
\[
 R\leq\binom n3(\lambda/n)^3\leq\frac{\lambda^3}{6}.     \tag{1.7}
\]
For \(n\geq3\),
\[
\begin{aligned}
q
&\geq\Pr[K=2]\\
&=\binom n2(\lambda/n)^2(1-\lambda/n)^{n-2}\\
&\geq\frac{\lambda^2}{3}(1-\lambda).                    \tag{1.8}
\end{aligned}
\]
Thus \(R\leq2q\) for \(0<\lambda\leq4/5\).  For \(n=2\), it holds because
\(R=0\).  In particular it holds for the explicit local-gap choice
\(\lambda=0.1\), \(\varepsilon=10^{-4}\).

In the rare-selector limit, the mechanism is transparent:
\[
 P=\frac{\lambda^2}{2}+o(\lambda^2),\quad
 U=\frac{3\lambda^2}{4}+o(\lambda^2),\quad
 F=O(\lambda^3)+O(\varepsilon).
\]
One piece retains the local \(3/2\) obstruction, whereas two pieces erase it
to lower order.

## 2. Additive-backbone theorem

Consider \(m\) copies of this chamber placed along a deterministic closed
backbone tour \(W\) based at the depot.  Let \(B\) be the length of \(W\)
after the chamber-internal movements have been deleted.  Assume the proposed
closure has the additive posterior value
\[
                         P_{\rm cl}=B+mP.                \tag{2.1}
\]
The chamber copies may lie in a high-girth graph, an expander, a tree, or a
lift; only the existence of the claimed backbone tour and the additive
identity (2.1) are used.

There are two closed causal policies.

- Traverse \(W\) once and run the one-piece policy in every chamber.  Its
  expected cost is at most
  \[
                         B+mU.                           \tag{2.2}
  \]
- Traverse \(W\) twice and run the first local piece on the first traversal
  and the second local piece on the second traversal.  Its expected cost is
  at most
  \[
                         2B+mF.                          \tag{2.3}
  \]

The second traversal is legal even though the permanent ports were already
called.  Between two active selector calls, use the metric shortcut of the
comparison walk that follows \(W\) through the next chamber's entry port;
after the chamber piece, use the comparison walk through its exit port.
The directed triangle inequality makes the actual movement no longer than
the charged copy of \(W\).  A chamber needing no second piece is simply
skipped.

Consequently
\[
 A_{\rm cl}\leq\min\{B+mU,\ 2B+mF\}.                    \tag{2.4}
\]

### Theorem

If \(2U+F\leq4P\), every additive-backbone closure (2.1) satisfies
\[
                         \frac{A_{\rm cl}}{P_{\rm cl}}
                         \leq\frac43.                    \tag{2.5}
\]

### Proof

Put \(x=B/(mP)\), \(u=U/P\), and \(f=F/P\).  Equations
(2.1)--(2.4) give
\[
 \frac{A_{\rm cl}}{P_{\rm cl}}
 \leq
 \frac{\min\{x+u,\ 2x+f\}}{x+1}.                        \tag{2.6}
\]
The first numerator gives a ratio at most \(4/3\) whenever
\[
                         x\geq3u-4,                     \tag{2.7}
\]
and the second does so whenever
\[
                         x\leq2-\frac32f.                \tag{2.8}
\]
The two ranges cover every \(x\geq0\) precisely when
\[
 3u-4\leq2-\frac32f
 \quad\Longleftrightarrow\quad
                         2u+f\leq4,                     \tag{2.9}
\]
which is the assumed local inequality. \(\square\)

The same proof allows unequal chamber scales: sum the corresponding
\(P_i,U_i,F_i\), and use
\(2\sum_iU_i+\sum_iF_i\leq4\sum_iP_i\).

## 3. Why high girth and private protection do not evade the theorem

A fixed master tour is repeatable.  High girth may make a shortest
point-to-point repair long, but a second copy of the original closed
backbone walk gives every chamber one additional open piece for the single
aggregate cost \(B\).  It therefore batches all \(K=2\) tournament errors,
which are exactly the first-order source of the local gap.

Trying to forbid this by assigning a private guard of length \(t_i\) to
chamber \(i\) creates the opposite side of the same accounting:

- if the guards are edge-disjoint, the first posterior traversal already
  pays at least \(\sum_i t_i\), diluting the local surplus;
- if guards share backbone arcs, a second traversal crosses the shared arcs
  once and batches the repairs, so charging \(\sum_i t_i\) double-counts
  physical movement;
- on a tree the shared cuts are laminar, making the batching explicit;
- an expander or high-girth lift adds alternate transit routes but does not
  make a used path leave persistent state at a client endpoint.

Thus a protected high-girth/tree/lift wrapper around the tournament chamber
does not yield a strict closed gap when its posterior proof is an additive
backbone calculation.  A successful closure must violate (2.1): its cheap
posterior route must depend on the independent chamber realizations in a
non-repeatable way.  That is a route-code/order-incompatibility construction,
not an additive backbone closure.
