# A \(3/2\) ceiling for rare-pair open gadgets with two-bounded asymmetry

## Setting

Let \(a,b\) be permanent open-service ports and let \(V\) be a fixed finite
selector set.  Use a family indexed by \(t\downarrow0\) in which selector
\(v\) is independently active with probability

\[
       p_v(t)=tq_v+O(t^2).
\]

Assume the port interfaces are uniform,

\[
       d(a,v)=d(v,b)=\varepsilon_t,
\]

with \(\varepsilon_t=o(t^2)\), and all selector distances are bounded
independently of \(t\).  Finally assume pairwise two-bounded asymmetry:

\[
 \max\{d(u,v),d(v,u)\}
 \le 2\min\{d(u,v),d(v,u)\}
 \qquad(u\ne v).
 \tag{1}
\]

This includes every selector metric whose nonzero distances are in
\(\{c,2c\}\), in particular the quasirandom tournament chamber.

### All activation densities for a tournament two-level metric

There is a stronger statement when, for every unordered selector pair, the
two directed distances are \(c,c\) or \(c,2c\).  In particular, at least one
direction of every pair must have length \(c\).  This is the tournament
two-level case; allowing both directions to have length \(2c\) would not
satisfy the argument below.  The statement does not require rare activation
or uniform activation probabilities.

Give every selector an independent uniform random priority and call selectors
in priority order.  Conditional on any realized active set \(A\) of size
\(k\), its active order is a uniform random permutation.  Each ordered
adjacent pair in that permutation is uniform among ordered pairs of distinct
vertices of \(A\).  For every unordered pair, the average of its two
directions is at most \(3c/2\).  Hence the expected internal path cost is at
most

\[
       \frac32c(k-1).
\]

Every path through \(k\) distinct selectors costs at least \(c(k-1)\), so
the realization-wise posterior internal cost has that lower bound.
With zero-cost common ports this proves an exact \(3/2\) adaptive upper
bound; with ports of total cost \(z\ge0\), the ratio only decreases:

\[
 \frac{z+\frac32c(k-1)}{z+c(k-1)}\le\frac32.
\]

Thus changing the activation density or trying to charge three-active and
higher events cannot push a tournament \(\{c,2c\}\) selector metric above
\(3/2\).
Any improvement must change the metric geometry, not merely the binomial
regime used in the tournament note.

## Theorem

In this class,

\[
 \limsup_{t\downarrow0}
 \frac{\operatorname{OPT}^{\rm open}_{\rm adapt}(t)}
      {\operatorname{OPT}^{\rm open}_{\rm post}(t)}
 \le \frac32.
 \tag{2}
\]

### Proof

Use the legal adaptive policy that samples a uniformly random permutation of
all selectors and calls them in that order, with \(a\) first and \(b\) last.
If exactly two selectors \(u,v\) are active, their relative order is a fair
coin.  The expected selector-to-selector transition is

\[
 \frac{d(u,v)+d(v,u)}2
 \le \frac32\min\{d(u,v),d(v,u)\},
 \tag{3}
\]

where (1) gives the inequality.  The two port movements contribute
\(2\varepsilon_t=o(t^2)\).

The probability of the active pair \(\{u,v\}\) is

\[
       t^2q_uq_v+O(t^3).
\]

Realizations with at least three active selectors have probability
\(O(t^3)\), and their service cost is uniformly bounded because \(V\) is
fixed.  Zero- and one-selector realizations contribute only
\(O(\varepsilon_t)=o(t^2)\).  Therefore

\[
\begin{aligned}
 \operatorname{OPT}^{\rm open}_{\rm adapt}(t)
 &\le
 \frac32t^2
 \sum_{\{u,v\}}q_uq_v
 \min\{d(u,v),d(v,u)\}
 o(t^2),\\
 \operatorname{OPT}^{\rm open}_{\rm post}(t)
 &=
 t^2
 \sum_{\{u,v\}}q_uq_v
 \min\{d(u,v),d(v,u)\}
 o(t^2).
\end{aligned}
\]

The second equality holds because, on a two-selector realization, the
clairvoyant chooses the cheaper of the two orders.  Dividing proves (2).
\(\square\)

## Triangle obstruction to high-diameter conflicts

To beat (2) using the exactly-two-active event, some important selector
pairs must have asymmetry strictly larger than two.  Such pairs cannot form
a dense tournament-like system of directed triangles.

Orient a pair \(u,v\) from \(u\) to \(v\) when \(d(u,v)<d(v,u)\).  Consider
a directed triangle

\[
       u\longrightarrow v\longrightarrow w\longrightarrow u
\]

and write

\[
 a=d(u,v),\qquad b=d(v,w),\qquad c=d(w,u)
\]

for its three preferred lengths, and

\[
 A=d(v,u),\qquad B=d(w,v),\qquad C=d(u,w)
\]

for the reverse lengths.  Triangle inequality gives

\[
       A\le b+c,\qquad B\le c+a,\qquad C\le a+b.
 \tag{4}
\]

It is impossible that \(A>2a\), \(B>2b\), and \(C>2c\) simultaneously:
summing those strict inequalities contradicts the sum of (4).  Hence every
directed triangle contains a preferred pair whose reverse distance is at
most twice its forward distance.

Thus the tournament construction is already extremal for the dense
triangle-rich rare-pair mechanism.  A high-diameter improvement must put
its \(>2\)-asymmetric conflicts on directed cycles of length at least four.
It cannot simply replace every reverse tournament arc of length \(2\) by a
longer arc while retaining the same quasirandom triangle structure.

## Scope

The theorem is a genuine ceiling for the rare-pair regime, not a universal
open-service upper bound.  A construction above \(3/2\) would have to use at
least one of:

1. selector pairs with asymmetry greater than two arranged through
   interacting cycles of length at least four;
2. a leading-order realization with at least three active selectors; or
3. permanent internal structure whose posterior route savings are not a
   sum of pair-order savings.
