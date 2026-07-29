# One selector with freely restartable open service has a \(4/3\) ceiling

## Setting

There is an arbitrary collection \(\mathcal D\) of permanent child blocks
and one stochastic selector \(x\), active with probability \(p\).  Service
is open: a service-containing piece may enter and leave through arbitrary
allowed boundary ports, and entry and exit themselves are free.  A child
block may be interrupted and resumed in any number of pieces.  All charged
movement is measured by a directed metric.

Let
\[
 C_0=\operatorname {OPT}_{\rm open}(\mathcal D),\qquad
 C_1=\operatorname {OPT}_{\rm open}(\mathcal D\cup\{x\})
\]
denote the optimal fixed-realization values in this fully interrupted
service model.  The two optimal traces may use different ports, split the
permanent blocks differently, and interleave their pieces in unrelated
orders.

Shortcutting the visit to \(x\) in an active-state trace gives
\[
                         C_0\leq C_1.                    \tag{1}
\]
The a-posteriori value is
\[
                         P=(1-p)C_0+pC_1.                \tag{2}
\]

## Two legal causal policies

First follow an optimal active-state trace.  If \(x\) is inactive, omit its
movement and shortcut between the preceding and following positions in its
piece.  Directed triangle inequality shows that this cannot increase the
cost.  All permanent blocks are still served, with exactly the same piece
boundaries as before.  Hence
\[
                         A\leq C_1.                      \tag{3}
\]

For the second policy, first execute an optimal inactive-state service of
all permanent blocks, at cost \(C_0\).  It remains only to reveal and, if
necessary, serve \(x\).

Fix an optimal active-state service.  In the piece containing \(x\), write
the charged trace as
\[
       a,\ z_1,\ldots,z_k,\ x,\ z_{k+1},\ldots,z_\ell,\ b,
\]
where \(a,b\) are its free entry and exit ports and the \(z_i\)'s are
permanent service locations or transit locations.  Start one fresh piece
at \(a\) and call \(x\) directly.

- If \(x\) is inactive, the call causes no movement, so this fresh piece
  has zero charged cost and service stops.
- If \(x\) is active, the directed triangle inequality gives
  \[
  d(a,x)\leq
  d(a,z_1)+\cdots+d(z_k,x).
  \]
  After reaching \(x\), leave directly at \(b\); again
  \[
  d(x,b)\leq
  d(x,z_{k+1})+\cdots+d(z_\ell,b).
  \]
  Thus the extra active-state cost is at most the cost of the
  \(x\)-containing piece, hence at most \(C_1\).

The permanent locations on this comparison trace have already been served;
they are used only to certify the two shortcut inequalities.  Therefore
the policy is legal even when child service events may not be repeated.
It proves
\[
                         A\leq C_0+pC_1.                 \tag{4}
\]

This argument explicitly permits arbitrary interleaving and interruption:
the inactive optimum is run with all of its own pieces, and the selector
probe is one additional freely entered and exited piece.

## Optimization

Combining (3) and (4),
\[
                         A\leq\min\{C_1,C_0+pC_1\}.       \tag{5}
\]
If \(C_0=0\), (4) gives \(A\leq pC_1=P\).  Otherwise put
\(\rho=C_1/C_0\geq1\).  Then
\[
 \frac AP\leq
 \frac{\min\{\rho,1+p\rho\}}{1-p+p\rho}.                \tag{6}
\]
The first branch is increasing in \(\rho\), the second is decreasing, and
they meet at \(\rho=1/(1-p)\).  Consequently
\[
 \boxed{\displaystyle
 \frac AP\leq\frac1{1-p+p^2}\leq\frac43.}               \tag{7}
\]

## Consequence

No directed metric quotient with one independent selector and arbitrarily
many permanent child blocks can exceed \(4/3\) when block service may be
split into freely entered and exited open pieces.  Higher arity, opposite
state-optimal orders, and arbitrary piece interleaving do not evade the
bound.

To escape (7), a construction must charge a positive reconnection toll for
an additional selector-containing piece, or use at least two independent
stochastic selectors.  The former returns to the interrupted-service
tradeoff audited in `rare-selector-fragmentation-obstruction.md`; it is no
longer the free-entry/free-exit model of this theorem.
