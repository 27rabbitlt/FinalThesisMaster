# Pairwise-additive obstruction to a higher-order rare chamber

## 1. Outcome

Consider an open chamber with fixed entry \(a\), fixed exit \(b\), and
independently active selector clients.  A selector call that is inactive
causes no movement, so a fixed selector query order induces the same order
on the active subsequence.

The main lemma below is scale-free:

> The expected cost of any fixed query order on all realizations is at most
> \(1/\Pr[K=0]\) times its expected cost on realizations having at most two
> active selectors.

Thus, if one query order is posterior-optimal on every zero-, one-, and
two-selector realization, its cost on the whole distribution is at most
\[
       \frac{1}{\Pr[K=0]}\operatorname {OPT}_{\rm post}.
\]
When the total selector intensity tends to zero, this ratio tends to one.
More generally, a constant local gap in the sparse regime forces a
constant amount of its value to be visible already on two-selector
realizations.

This rules out the proposed mechanism in which the first causal conflict
occurs only when \(q\geq3\) rare groups are active.  It does not use bounded
distances, a common distance scale, or an asymptotic expansion.  The reason
is exactly that metric service cost is pairwise additive along the active
order.

## 2. Fixed-order notation

Let
\[
                  V=\{v_1,\ldots,v_n\}
\]
be written in a fixed query order \(\sigma\).  Selector \(v_i\) is active
with probability \(p_i\), independently of all other selectors.  Put
\[
        q_0:=\Pr[K=0]=\prod_{i=1}^n(1-p_i)>0.
\]

For an active set
\[
       S=\{v_{i_1},\ldots,v_{i_k}\},
       \qquad i_1<\cdots<i_k,
\]
let
\[
\begin{aligned}
 C_\sigma(S)
   &=d(a,v_{i_1})
     +\sum_{\ell=1}^{k-1}d(v_{i_\ell},v_{i_{\ell+1}})
     +d(v_{i_k},b), &&k\geq1,\\
 C_\sigma(\varnothing)&=d(a,b).
\end{aligned}                                                    \tag{2.1}
\]
No triangle inequality is needed for the next lemma; nonnegativity of the
displayed arc costs is enough.

## 3. Sparse-realization domination

### Lemma 1

For every fixed query order \(\sigma\),
\[
 \boxed{\displaystyle
 \mathbb E C_\sigma(X)
 \leq
 \frac1{q_0}\,
 \mathbb E\!\left[
       C_\sigma(X)\mathbf1_{\{|X|\leq2\}}
 \right].}                                                     \tag{3.1}
\]

### Proof

Expanding the left side by the first active selector, last active selector,
and consecutive active selector pairs gives
\[
\begin{aligned}
 \mathbb E C_\sigma(X)
 ={}&
 q_0d(a,b)\\
 &+\sum_i
 p_i\prod_{h<i}(1-p_h)d(a,v_i)\\
 &+\sum_i
 p_i\prod_{h>i}(1-p_h)d(v_i,b)\\
 &+\sum_{i<j}
 p_ip_j\prod_{i<h<j}(1-p_h)d(v_i,v_j).
\end{aligned}                                                   \tag{3.2}
\]

The exact-singleton probability for \(v_i\), divided by \(q_0\), is
\[
 \frac1{q_0}
 p_i\prod_{h\ne i}(1-p_h)
 =\frac{p_i}{1-p_i}
 \geq
 p_i\prod_{h<i}(1-p_h)
 \tag{3.3}
\]
and also dominates the analogous last-active coefficient.  The exact-pair
probability for \(\{v_i,v_j\}\), divided by \(q_0\), is
\[
 \frac1{q_0}
 p_ip_j\prod_{h\ne i,j}(1-p_h)
 =
 \frac{p_ip_j}{(1-p_i)(1-p_j)}
 \geq
 p_ip_j\prod_{i<h<j}(1-p_h).
 \tag{3.4}
\]
Charge the entry and exit terms in (3.2) to the corresponding exact
singleton realization, and charge the \(v_i\to v_j\) term to the
corresponding exact-pair realization.  The sparse realization contains
those charged arcs and possibly additional nonnegative boundary arcs.
Finally,
\[
                    q_0d(a,b)\leq d(a,b),
\]
which is the empty-realization contribution after division by \(q_0\).
Summing the termwise inequalities proves (3.1). \(\square\)

The loss \(1/q_0\) is deliberately coarse but has the useful feature that
it is independent of all metric scales.  Reverse arcs may grow while
forward arcs shrink, and \(n\) may grow with the rarity parameter.

## 4. Quantitative pair-conflict certificate

Let \(P(S)\) be the posterior-optimal \(a\)-to-\(b\) service cost for active
set \(S\), and put
\[
\begin{aligned}
 P_{\leq2}
   &:=\mathbb E[P(X)\mathbf1_{\{|X|\leq2\}}],\\
 \Delta_2(\sigma)
   &:=\mathbb E[
       (C_\sigma(X)-P(X))\mathbf1_{\{|X|\leq2\}}].
\end{aligned}                                                   \tag{4.1}
\]
Both quantities include their true realization probabilities.  Lemma 1
immediately gives a legal causal policy satisfying
\[
\begin{aligned}
 \operatorname {OPT}_{\rm adapt}
 &\leq\min_\sigma\mathbb E C_\sigma(X)\\
 &\leq
 \frac{P_{\leq2}+\min_\sigma\Delta_2(\sigma)}{q_0}\\
 &\leq
 \frac{\operatorname {OPT}_{\rm post}
       +\min_\sigma\Delta_2(\sigma)}{q_0}.
\end{aligned}                                                   \tag{4.2}
\]
Consequently,
\[
 \frac{\operatorname {OPT}_{\rm adapt}}
      {\operatorname {OPT}_{\rm post}}
 \leq
 \frac1{q_0}\left(
       1+
       \frac{\min_\sigma\Delta_2(\sigma)}
            {\operatorname {OPT}_{\rm post}}
       \right).
 \tag{4.3}
\]

If the total selector intensity
\[
                         \mu:=\sum_i p_i
\]
tends to zero, then \(q_0\geq1-\mu\).  Hence
\[
 \min_\sigma\Delta_2(\sigma)
       =o(\operatorname {OPT}_{\rm post})
 \quad\Longrightarrow\quad
 \frac{\operatorname {OPT}_{\rm adapt}}
      {\operatorname {OPT}_{\rm post}}
       =1+o(1).                                               \tag{4.4}
\]
Conversely, a ratio at least \(1+\gamma\), for fixed \(\gamma>0\), forces
\[
 \min_\sigma\Delta_2(\sigma)
 \geq
 \bigl(\gamma-O(\mu)\bigr)
 \operatorname {OPT}_{\rm post}.                              \tag{4.5}
\]
Thus a sparse constant gap necessarily has a leading two-selector
order conflict.

## 5. “First conflict at \(q\geq3\)” is impossible

Suppose a causal policy is posterior-optimal on every realization with at
most two active selectors.  Follow its branch on which every selector call
is inactive.  This branch defines a selector permutation \(\sigma\).

On a realization containing exactly \(\{u,v\}\), the selector that occurs
first in \(\sigma\) is the first active selector reached.  After observing
it, the policy may alter its later query order, but there is only one other
active selector, so the service order of \(u,v\) is still their relative
order in \(\sigma\).  With fixed chamber endpoints, inactive calls do not
affect movement.  Therefore
\[
             C_\sigma(S)=P(S)\qquad(|S|\leq2),
 \tag{5.1}
\]
and \(\Delta_2(\sigma)=0\).  Equation (4.3) becomes
\[
 \boxed{\displaystyle
 \frac{\operatorname {OPT}_{\rm adapt}}
      {\operatorname {OPT}_{\rm post}}
 \leq\frac1{\Pr[K=0]}.}                                      \tag{5.2}
\]

In particular, if \(\mu\to0\), a chamber whose first genuine causal
conflict occurs at three, four, or any fixed larger number of active
selectors has ratio \(1+o(1)\), not a ratio above \(4/3\).

The same conclusion holds in the approximate form: if all two-selector
conflicts have aggregate expected value \(o(P)\), then all higher-order
conflicts together can improve the ratio by only \(o(1)\).

## 6. Relation to multi-lap closure

For comparison, suppose an idealized chamber has posterior value \(P\),
one-piece causal value \(rP\), and a \(q\)-piece policy of negligible
internal cost.  On an additive backbone of cost \(B\), the one-lap and
\(q\)-lap policies give
\[
 A_{\rm cl}\leq
 \min\{B+rP,\ qB\},\qquad
 P_{\rm cl}=B+P.                                             \tag{6.1}
\]
Optimizing over \(B/P\) yields the envelope
\[
 \max_{B\geq0}
 \frac{\min\{B+rP,qB\}}{B+P}
 =
 \frac{qr}{q-1+r}.                                           \tag{6.2}
\]
It exceeds \(4/3\) only if
\[
             r>\frac{4(q-1)}{3q-4}.                          \tag{6.3}
\]
For \(q=3\) this requires \(r>8/5\); for \(q=4\), \(r>3/2\).

Lemma 1 shows that a genuinely \(q\)-first sparse endpoint chamber has
\[
                         r\leq1/q_0=1+o(1),
\]
far below every threshold in (6.3).  If instead \(r\) is bounded away from
one, (4.5) says that the mechanism already has a leading pair conflict.
That is precisely the conflict for which a second service piece or second
backbone lap is available.

## 7. Scope and remaining escape routes

The obstruction applies to:

1. one fixed entry and one fixed exit;
2. independent selector activations;
3. movement cost obtained by summing directed distances between consecutive
   active service events; and
4. sparse total activation intensity.

It permits arbitrary directed metrics, arbitrary nonuniform selector
probabilities, growing selector sets, and parameter-dependent distance
scales.

A higher-order construction must therefore break at least one premise.
The plausible possibilities are:

* a realization-dependent route code whose state persists across active
  service events and is not described by one active-subsequence order;
* a service-once permanent structure whose legal completion depends jointly
  on three or more selectors; or
* a non-sparse product distribution in which \(\Pr[K=0]\) is bounded far
  below one, followed by a new global lower bound that survives adaptive
  successor probing.

Duplicating a client to remember a route-code state normally gives
independent activation bits rather than copies of one hidden bit.  Hence
the first two possibilities require a genuine new geometric mechanism, not
only a larger tournament, lift, or higher-girth selector table.
