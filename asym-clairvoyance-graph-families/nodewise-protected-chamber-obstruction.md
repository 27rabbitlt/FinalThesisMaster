# Obstruction to perturbing every Chapter-4 node by a protected chamber

## 1. Canonical nodewise interface

Consider one node of the Chapter-4 directed-triangle recursion.  Its two
children are denoted \(L,R\), its midpoint \(m\) is active with probability
\(1/2\), and every top triangle arc has length \(h\).  After child
reconnection credits are removed, the a-posteriori and adaptive top
contributions are
\[
       \frac32h,\qquad 2h.
\tag{1.1}
\]

Append one copy of the exact two-selector chamber \(D\), scaled by \(s>0\),
at the terminal end of the node.  Its free-boundary values are
\[
       P=\frac{617}{125},\qquad A=\frac{33}{5}.
\tag{1.2}
\]
The chamber can be served terminally without using a return.  If it is
entered before the triangle is complete, its output returns to the node
input (or to the appropriate unfinished child bank) through a protected arc.

The old module return has length \(2h\).  A protected return shorter than
\(2h\) gives a shortcut through the triangle, so write its length as
\[
       c=2h+e,\qquad e\geq0.
\tag{1.3}
\]
The number \(e\) is exactly the increase in the module reset scale caused by
the chamber protection.

The intended one-piece a-posteriori contribution is
\[
       p_{\rm node}=\frac32h+Ps.
\tag{1.4}
\]
For \(s\) sufficiently small compared with \(h\), a nonterminal chamber
visit cannot improve (1.4), because it costs at least \(2h\) and the entire
chamber has \(O(s)\) diameter.  Thus this is also the exact posterior value
in the scale-separated regime for which the perturbation is proposed.

## 2. Two explicit adaptive policies

There are two relevant causal policies.

### Contiguous policy

Run the optimal adaptive triangle policy and then serve the chamber in one
piece.  Its cost is
\[
       2h+As.
\tag{2.1}
\]

### Midpoint-conditioned two-piece policy

Serve \(R\) and call \(m\).

- If \(m\) is active, use \(R\to m\to L\), of cost \(2h\), and then serve
  the chamber contiguously, at expected cost \(As\).
- If \(m\) is inactive, serve the chamber in the following two pieces around
  the unfinished child \(L\).  First enter at \(a\), serve \(a\), and call
  \(x\); if \(x\) is active, use \(a\to x\to b\), and otherwise exit at
  \(a\).  Take the protected return of cost \(c\), serve \(L\), reenter at
  \(b\), serve \(b\), and call \(y\); if \(y\) is active, use
  \(b\to y\to a\), and otherwise exit at \(b\).

The two chamber pieces have expected internal cost
\[
       B:=\frac{13}{25}\,5+\frac15\,8=\frac{21}{5}.
\tag{2.2}
\]
Since the active and inactive midpoint branches have equal probability, the
second policy has expected cost
\[
\begin{aligned}
 a_{\rm split}
 &=\frac12(2h+As)+\frac12(c+Bs)\\
 &=2h+\frac e2+\frac{A+B}{2}s\\
 &=2h+\frac e2+\frac{27}{5}s.
\end{aligned}
\tag{2.3}
\]
This is precisely the policy that invalidates an additive
“triangle baseline plus chamber reconnection toll” inequality: in the
inactive branch, the protected return is also the top movement that changes
the unfinished orientation.

Consequently the true adaptive node value satisfies
\[
       a_{\rm node}\leq
       \min\left\{
           2h+\frac{33}{5}s,\,
           2h+\frac e2+\frac{27}{5}s
       \right\}.
\tag{2.4}
\]

## 3. The exact local-surplus bound

Define the node's possible contribution to
\(3\operatorname {OPT}_{\rm adapt}
-4\operatorname {OPT}_{\rm post}\) by
\[
       M:=3a_{\rm node}-4p_{\rm node}.
\]
Equations (1.2), (1.4), and (2.4) give
\[
\begin{aligned}
 M
 &\leq
 \min\left\{
 3\left(2h+\frac{33}{5}s\right)
 -4\left(\frac32h+\frac{617}{125}s\right),
 \right.\\[-0.2em]
 &\hspace{3.5cm}\left.
 3\left(2h+\frac e2+\frac{27}{5}s\right)
 -4\left(\frac32h+\frac{617}{125}s\right)
 \right\}\\
 &=\min\left\{
       \frac7{125}s,\,
       \frac32e-\frac{443}{125}s
     \right\}.
\end{aligned}
\tag{3.1}
\]

This inequality uses actual legal policies; it does not assume that the
chamber is served contiguously.

## 4. Scale summation and root closure

At a protected node the new module reset is \(2h+e\), while the two children
together account for scale \(2h\).  Thus the branching-to-scale factor is
\[
       q=\frac{2h}{2h+e}
        =1-\frac{e}{2h+e}.
\tag{4.1}
\]
If the same scaled node is iterated recursively, the geometric sum of all
node surpluses, normalized by the root reset scale, is at most
\[
       \frac{M}{1-q}
       =\frac{M(2h+e)}{e}.
\tag{4.2}
\]
Closing the root costs at least one root reset.  That cost is common to the
two benchmarks and hence contributes minus one root scale to
\(3\operatorname {OPT}_{\rm adapt}
-4\operatorname {OPT}_{\rm post}\).  A necessary condition for this
nodewise perturbation to yield a closed ratio above \(4/3\) is therefore
\[
       M>e.
\tag{4.3}
\]

But (4.3) is impossible.  If
\[
       e\leq\frac{886}{125}s,
\]
the second term in (3.1) is at most \(e\):
\[
       \frac32e-\frac{443}{125}s\leq e.
\]
If \(e>886s/125\), the first term in (3.1) gives
\[
       M\leq\frac7{125}s<e.
\]
Thus
\[
       M\leq e
\tag{4.4}
\]
for every \(e\geq0\), with no strict inequality in the direction needed by
(4.3).

### Telescoping form for nonuniform node parameters

The same obstruction does not require a constant scale ratio.  Let \(C_k\)
be the reset scale of a depth-\(k\) module and let its top triangle use
\(h_k=C_{k-1}\).  If its protected return has excess \(e_k\), then
\[
       C_k=2C_{k-1}+e_k.
\tag{4.5}
\]
Let \(S_k=3A_k-4P_k\) for the corresponding open module values.  Recursively
running the better of the two explicit node policies gives
\[
       S_k\leq2S_{k-1}+M_k
       \leq2S_{k-1}+e_k.
\tag{4.6}
\]
Iterating (4.5)--(4.6) yields
\[
\begin{aligned}
 S_l
 &\leq 2^lS_0+\sum_{k=1}^l2^{l-k}e_k,\\
 C_l
 &=2^lC_0+\sum_{k=1}^l2^{l-k}e_k.
\end{aligned}
\tag{4.7}
\]
With the zero-cost base \(S_0=0\) and \(C_0>0\), this gives \(S_l<C_l\).
Closing the root adds the common reset cost \(C_l\), so the closed surplus is
\[
       3\operatorname {OPT}_{\rm adapt}
       -4\operatorname {OPT}_{\rm post}
       \leq S_l-C_l<0.
\tag{4.8}
\]

For (4.8) as an upper bound on the *actual* ratio, rather than only on the
intended recurrence, the natural a-posteriori node service must be exact.
This holds in the scale-separated perturbative regime.  For example,
\(h_k>13s_k\) is a sufficient coarse condition: every nonterminal chamber
piece uses a return of length at least \(2h_k\), the inactive triangle's
natural top cost is only \(h_k\), and every fixed-state one-piece chamber
service costs at most \(13s_k\).  Thus fragmentation cannot improve the
inactive posterior route; in the active state the natural triangle already
uses the minimum \(R\to m\to L\) passage, and an additional protected return
cannot improve it.  Hence the posterior recurrence is exact in the small
chamber regime for which a nodewise perturbation would normally be attempted.

## 5. Scope of the obstruction

This rules out the canonical nodewise protected-chamber perturbation in
which:

1. the chamber is terminal in a normal one-piece service;
2. a nonterminal chamber service returns to the unfinished triangle through
   the module reset or a longer protected version of it;
3. the increased return length determines the scale passed to the parent.

The result includes node-specific returns: no shared global lap is used in
the proof.  The obstruction is instead local.  The midpoint-inactive
execution uses the protected return both as a chamber separator and as the
orientation-changing triangle movement.

There are only two structural ways outside this theorem:

- a return that restores the exact pre-chamber boundary state without
  making progress through the triangle and without adding a common
  one-piece cost; or
- a chamber whose true interrupted-service tradeoff is strong enough that
  its \(3A-4P\) surplus exceeds the reset-scale increase.

A single-output, two-child metric interface does not remember the
pre-chamber child, and duplicating the output state while sharing one
activation bit is exactly the fan-out/correlation problem encountered by
algebraic lifts.  The exact \(D\) chamber fails the second alternative by
(3.1).

### Why a two-sheet reset does not supply the missing memory

Suppose one tries to retain the entry label by providing two chamber outputs
\(o_L,o_R\), with \(o_L\) returning only to \(L\) and \(o_R\) only to \(R\).
If the chamber's stochastic selector is a single vertex \(x\), then after an
active call the physical position is the same vertex \(x\), regardless of
which sheet was used to enter.  Both paths \(x\leadsto o_L\) and
\(x\leadsto o_R\) are present in the fixed graph.  Shortest-path closure
therefore lets the policy choose the favorable output after observing the
midpoint and selector outcomes; the purported sheet label is not a
geometric constraint.

Duplicating the selector as \(x_L,x_R\) keeps the two physical sheets
separate, but then the two activation bits are independent under the model.
Declaring them to be two copies of one hidden bit would impose perfect
correlation and is not a legal product activation law.  Thus ordinary
state-splitting cannot implement a return to the exact pre-chamber child
while preserving the same chamber distribution.
