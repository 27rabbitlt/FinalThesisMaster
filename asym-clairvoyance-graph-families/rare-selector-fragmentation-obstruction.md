# Fragmentation obstruction for rare-selector chambers

This note records a closure obstruction which is independent of the
particular selector metric.  It rules out the rare-pair tournament chamber,
all of its fixed-size higher-cycle variants with cheap common interfaces,
and any parallel repetition for which the aggregate singleton-piece
parameters still satisfy the explicit inequalities below.

## Interrupted value

Let \(P\) be the expected fixed-realization open-service value of a chamber,
let \(A\) be its least expected contiguous adaptive open-service value, and
let
\[
 H(s):=\inf_\pi
 \mathbb E\!\left[Z_\pi+s(N_\pi-1)\right]
\]
be its interrupted adaptive value.  Here \(Z_\pi\) is internal movement
cost and \(N_\pi\geq1\) is the number of service-containing pieces.  A
fixed-order serial closure can amplify a chamber above \(4/3\) only if
\[
                  3H(s)-4P>s
\tag{1}
\]
for some \(s>0\).

Put
\[
                  \Delta:=3A-4P.
\]
The one-piece policy gives \(H(s)\leq A\), so (1) is already impossible
when \(s\geq\Delta\).  It remains only to test tolls \(0<s<\Delta\).

If there is a particular two-piece policy of zero-toll internal value \(F\),
then
\[
                   H(s)\leq\min\{A,F+s\}.
\tag{1a}
\]
The two affine upper bounds alone exclude (1) for every \(s\) exactly when
\[
                   2A+F\leq4P.
\tag{1b}
\]
Indeed, the fragmented line can exceed (1) only for
\[
                   s>\frac{4P-3F}{2},
\]
whereas the one-piece line requires \(s<3A-4P\); these two open intervals
overlap exactly when \(2A+F>4P\).  In particular, whenever \(F\leq P\), a
fragmentation-robust chamber must first have the strict open ratio
\[
                         \frac AP>\frac32.
\tag{1c}
\]
For cheap rare-selector interfaces one has the much stronger
\(F=o(P)\), which is why merely approaching the tournament's \(3/2\)
ceiling cannot suffice.

## A general singleton-piece certificate

Suppose there is a legal causal policy \(\pi_{\rm sing}\) which uses one
piece when at most one selector is active and, in general, satisfies
\[
       \mathbb E Z_{\rm sing}\leq F,\qquad
       \mathbb E(N_{\rm sing}-1)\leq q.
\tag{2}
\]
Then
\[
                     H(s)\leq F+s q.
\tag{3}
\]
Consequently (1) is impossible for every \(s>0\) provided
\[
        \Delta\leq s_0
        \quad\hbox{and}\quad
        3F+3s_0q\leq4P
\tag{4}
\]
for some \(s_0>0\).  Indeed, for \(s\geq\Delta\) use the one-piece
bound.  For \(s<\Delta\leq s_0\), (3)--(4) give
\[
       3H(s)-4P
       \leq 3F+3s q-4P
       \leq 3F+3s_0q-4P
       \leq0<s.
\]

There is also a scale-free version which is often more convenient.  If
\[
             q\leq\frac13
       \quad\hbox{and}\quad
             3F\leq4P,
\tag{5}
\]
then for \(0<s<\Delta\),
\[
 3H(s)-4P-s
 \leq 3F-4P+(3q-1)s
 \leq 3F-4P
 \leq0.
\]
Thus (1) again never holds.  When \(q>1/3\), the corresponding sufficient
condition is
\[
              3F+(3q-1)\Delta\leq4P,
\]
because the affine expression is then maximized as \(s\uparrow\Delta\).

## Rare independent selectors

Let a fixed set of selectors have independent activation probabilities
\[
                    p_v(\tau)=\tau a_v+O(\tau^2)
                    \qquad(\tau\downarrow0),
\tag{6}
\]
and suppose all permanent boundary clients can be assigned to one initial
piece.  Assume that, after that piece, an active selector \(v\) can be
served in its own piece at internal cost \(c_v(\tau)\), where
\[
       \sum_v p_v(\tau)c_v(\tau)=o(P_\tau).
\tag{7}
\]
Call the selectors in any fixed order and give every active selector after
the first its own piece.  This is causal.  It has
\[
\begin{aligned}
 F_\tau
   &\leq\sum_v p_v(\tau)c_v(\tau)=o(P_\tau),\\
 q_\tau
   &\leq\mathbb E[(K_\tau-1)_+]
     =O(\tau^2).
\end{aligned}
\tag{8}
\]

If the chamber has bounded geometry and its gap is supported on events with
at least two active selectors, then
\[
             P_\tau=O(\tau^2),\qquad
             \Delta_\tau=O(\tau^2).
\tag{9}
\]
Equations (8)--(9) imply, for all sufficiently small \(\tau\),
\[
 q_\tau\leq\frac13,\qquad
 3F_\tau\leq4P_\tau.
\]
Condition (5) therefore applies:
\[
             3H_\tau(s)-4P_\tau\leq s
             \qquad\hbox{for every }s>0.
\tag{10}
\]

This conclusion does not use a \(3/2\) ceiling and does not depend on whether
the pair-conflict graph is a tournament, a high-girth orientation, a Cayley
digraph, or an algebraic lift.  It uses only the cheap singleton interfaces
and the fact that two active selectors are needed before the posterior route
obtains a saving.  Increasing the reverse/forward asymmetry or replacing
pairs by fixed-size higher-order rare events cannot repair (10).

## Exact application to the tournament chamber

In the tournament chamber, give the two permanent ports to the first piece.
When \(K\geq1\), serve each active selector in a separate
\(a\to v\to b\) piece.  If \(K=0\), use the single \(a\to b\) piece.  Hence
\[
\begin{aligned}
 \mathbb E Z_{\rm sing}
   &=2\varepsilon\,\Pr[K=0]+2\varepsilon\,\mathbb E K,\\
 \mathbb E(N_{\rm sing}-1)
   &=\mathbb E[(K-1)_+].
\end{aligned}
\tag{11}
\]
In the rare-pair regime \(\varepsilon=o(\tau^2)\), the first expression is
\(o(P_\tau)\), while the second is \(O(\tau^2)\).  Thus (10) applies
directly.  The local open ratio may tend to \(3/2\), but its strict surplus
is necessarily too small to purchase even one fragmentation reset.

## What remains viable

A chamber capable of satisfying (1) must violate the singleton-piece
certificate.  In particular, its uncertainty must interact at first order
with permanent routing structure, or an active selector must itself incur a
non-negligible boundary service cost which is also present in the
a-posteriori value.  Rare interacting selectors with asymptotically free
ports cannot yield a closed construction above \(4/3\).
