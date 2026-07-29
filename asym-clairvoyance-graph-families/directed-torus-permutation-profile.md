# Directed-torus permutation profiles

## 1. The metric

Fix \(m\ge 2\), a selector set \(X\), and \(d\) maps
\[
             \rho_j:X\longrightarrow\{0,\ldots,m-2\}
             \qquad(j\in[d]).
\tag{1.1}
\]
For the intended permutation construction, each \(\rho_j\) is injective
on \(X\); one may take \(|X|\le m-1\).  Add one permanent client
\[
                         z=(m-1,\ldots,m-1).
\tag{1.2}
\]
Represent \(x\in X\) by
\((\rho_1(x),\ldots,\rho_d(x))\in\mathbb Z_m^d\), and define
\[
 d_{\mathbb T}(x,y)
   :=\sum_{j=1}^d [\,\rho_j(y)-\rho_j(x)\,]_m ,
\tag{1.3}
\]
where \([a]_m\in\{0,\ldots,m-1\}\) is the residue of \(a\) modulo \(m\).

Equation (1.3) is a directed metric.  Indeed, in each coordinate
\[
              [c-a]_m\le [b-a]_m+[c-b]_m,
\]
and summing proves the triangle inequality.  Distinct selector profiles
have positive distance when the rank maps jointly separate \(X\).  The
permanent profile \(z\) is distinct from every selector because of
(1.1).

## 2. Exact tour identity

For two selectors put
\[
 c(x,y):=\bigl|\{j:\rho_j(x)>\rho_j(y)\}\bigr|.
\tag{2.1}
\]
Let \(x_1,\ldots,x_k\) be any nonempty selector service order, starting
and ending at \(z\).  In coordinate \(j\), the integer (not reduced
modulo \(m\)) differences telescope to zero.  Every descent contributes
one wrap of size \(m\).  The first transition
\(m-1\to\rho_j(x_1)\) always wraps, the final transition
\(\rho_j(x_k)\to m-1\) never wraps, and an internal transition wraps
exactly when its ranks descend.  Therefore
\[
\boxed{
 d_{\mathbb T}(z,x_1)+
 \sum_{i=1}^{k-1}d_{\mathbb T}(x_i,x_{i+1})+
 d_{\mathbb T}(x_k,z)
 =m\left(d+\sum_{i=1}^{k-1}c(x_i,x_{i+1})\right).
}
\tag{2.2}
\]
For \(k=0\), the cost is zero.

Consequently, for an active set \(A\ne\varnothing\),
\[
 \operatorname{POST}(A)
   =m\left(d+
       \min_{\sigma\in S_A}
       \sum_{i=1}^{|A|-1}c(\sigma_i,\sigma_{i+1})\right).
\tag{2.3}
\]
This is an exact identity for arbitrary tours, not merely an upper bound:
after shortcutting, every closed tour induces a selector service order,
and (2.2) applies to that order.

For a causal policy, the active selectors occur in the order in which
their calls return active.  Inserting \(z\), using inactive selectors as
transit, or expanding a metric move cannot beat (2.2), because (1.3)
already is the shortest-path metric.  Thus the stochastic TSP problem on
this family is exactly the free-order stochastic Hamilton-path problem
with arc costs \(c(x,y)\), plus the \(d\)-wrap charge on every nonempty
realization.

## 3. Pair identities and the majority-vote policy

If every \(\rho_j\) is injective, then
\[
                         c(x,y)+c(y,x)=d.
\tag{3.1}
\]
For a two-selector active set \(\{x,y\}\), put
\[
                  a(x,y):=\frac1d\min\{c(x,y),c(y,x)\}.
\]
The posterior cost divided by \(m\) is
\[
                         d(1+a).
\tag{3.2}
\]

There is a particularly strong legal causal policy for pair
realizations: choose a coordinate \(J\) uniformly in advance and call all
selectors in increasing \(\rho_J\)-order.  Conditional on
\(\{x,y\}\), it uses the majority direction with probability \(1-a\)
and the minority direction with probability \(a\).  Hence its expected
cost divided by \(m\) is
\[
              d+2a(1-a)d,
\tag{3.3}
\]
and its pairwise ratio is
\[
             \frac{1+2a-2a^2}{1+a}.
\tag{3.4}
\]
The maximum of (3.4) on \(0\le a\le1/2\) is
\[
             6-2\sqrt6\approx1.1010<\frac43,
\tag{3.5}
\]
attained at \(a=(\sqrt6-2)/2\).  Thus a construction whose loss is
concentrated on exactly-two-active realizations cannot work in the torus
profile: the profile itself supplies a good randomized query order.

## 4. Cycle-margin obstruction

Suppose an ordered pair \(x\to y\) is declared cheap when
\(c(x,y)\le\alpha d\).  For every directed cycle
\[
                    x_1\to x_2\to\cdots\to x_\ell\to x_1
\]
one has
\[
                  \sum_{i=1}^{\ell}c(x_i,x_{i+1})\ge d.
\tag{4.1}
\]
Indeed, in each coordinate a cyclic sequence of distinct ranks has at
least one descent.  Summing this statement over the \(d\) coordinates
gives (4.1).  In particular, a directed cheap cycle of length \(\ell\)
forces
\[
                         \alpha\ge\frac1\ell.
\tag{4.2}
\]
Every nontransitive tournament contains a directed triangle, so a
tournament orientation cannot have all its winning margins cheaper than
\(d/3\).

This rules out importing the open quasirandom-tournament pair argument
verbatim.  That argument needs a feedback-rich orientation with
minority cost much smaller than \(d/8\), whereas (4.2) forces a
minority cost at least \(d/3\) on some triangle and the randomized
coordinate policy already gives the stronger pairwise ceiling (3.5).

## 5. What a positive profile would have to do

Equations (3.4)--(3.5) show that neither a two-active tournament nor any
other pair-only obstruction can exceed \(4/3\).  A successful profile
must instead use active sets of unbounded size for which:

1. after seeing the whole set, there is an order with few total adjacent
   descents across all coordinates;
2. no distribution over fixed coordinate orders has comparable expected
   cost; and
3. adaptive choice of the next selector after every observed activation
   still incurs many descents.

An order-dimension representation of a poset is a natural special case:
\(x<y\) exactly when \(c(x,y)=0\).  Posterior chains then have no internal
descent cost.  The unresolved issue is a fixed product-activation profile
whose optimal chain order depends on the realized active set; a single
known common linear extension is also a legal causal query order and
therefore gives no gap.
