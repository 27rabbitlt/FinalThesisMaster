# A directed \(4/3\) ceiling for one selector at a single port

This note rules out the simplest first-order-uncertainty chamber.  Unlike
the auxiliary bound for symmetric metrics, the proof below uses no symmetry
and no insertion estimate.

## Setting

There is one permanent boundary port \(r\), an arbitrary finite set \(D\) of
permanent clients, and one stochastic client \(x\), active with probability
\(p\in[0,1]\).  Distances form an arbitrary directed metric.  A chamber
service starts and ends at \(r\).

Let
\[
 C_0:=\operatorname{OPT}(D\cup\{r\}),\qquad
 C_1:=\operatorname{OPT}(D\cup\{x,r\})
\]
be the two fixed-realization optimum tour costs.  Shortcutting \(x\) from an
active-state tour gives
\[
                         C_0\leq C_1.
\tag{1}
\]
The exact a-posteriori value is
\[
                         P=(1-p)C_0+pC_1.
\tag{2}
\]

## Two causal policies

First, follow an optimal active-state tour and call the clients in its tour
order.  If \(x\) is inactive, shortcutting its call cannot increase cost.
Thus
\[
                         A\leq C_1.
\tag{3}
\]

For the second policy, first execute an optimal \(C_0\)-tour and return to
\(r\).  Then call \(x\); if it is active, make the final return to \(r\).
This costs
\[
                         C_0+p\bigl(d(r,x)+d(x,r)\bigr).
\]
Split an optimal active-state tour at its visit to \(x\).  Directed triangle
inequalities on its two portions give
\[
                         d(r,x)+d(x,r)\leq C_1.
\]
The second policy therefore has expected cost at most
\[
                         C_0+pC_1.
\tag{4}
\]
Combining (3)--(4),
\[
                         A\leq\min\{C_1,C_0+pC_1\}.
\tag{5}
\]

## Optimization

If \(C_0=0\), (5) gives \(A\leq pC_1=P\), so assume \(C_0>0\) and put
\[
                         \rho:=C_1/C_0\geq1.
\]
Equations (2) and (5) give
\[
 \frac AP
 \leq
 \frac{\min\{\rho,1+p\rho\}}
      {1-p+p\rho}.
\tag{6}
\]
For fixed \(p\), the first branch is increasing in \(\rho\), the second is
decreasing, and they meet at
\[
                         \rho=\frac1{1-p}.
\]
Consequently
\[
 \frac AP
 \leq
 \frac1{1-p+p^2}
 \leq\frac43,
\tag{7}
\]
with the last maximum attained at \(p=1/2\).

## Consequence

A single-port chamber with first-order uncertainty can exceed \(4/3\) only
if it contains at least two interacting stochastic clients.  Long directed
permutation paths, high-asymmetry return arcs, and permanent routing padding
cannot evade (7) when their state is controlled by only one activation bit.
The proof already permits arbitrary permanent internal structure.
