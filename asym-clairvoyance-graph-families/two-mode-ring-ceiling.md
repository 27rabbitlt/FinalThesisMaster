# A sharp \(4/3\) ceiling for two Hamiltonian modes

## 1. The concrete two-generator ring

Let \(n\) be odd.  The permanent clients are the vertices of
\(\mathbb Z_n\), with depot \(0\).  The generating digraph contains the
unit \(+1\) and \(+2\) arcs, except that the two arcs entering the depot are
subdivided:
\[
 n-1\longrightarrow x_1\longrightarrow0,\qquad
 n-2\longrightarrow x_2\longrightarrow0,
\]
with both displayed arcs of each subdivision having unit length.  The
vertices \(x_1,x_2\) are independently active with probability \(p\).
Distances are directed shortest-path distances.

Put \(L=n+1\).  In each of the states
\[
 \varnothing,\qquad \{x_1\},\qquad \{x_2\},
\]
the posterior optimum is exactly \(L\).

Indeed, the \(+1\) and \(+2\) Hamilton cycles give the matching upper
bounds.  For the lower bound, a depot tour through all \(n\) permanent
vertices has \(n\) client-to-client transitions.  Every nonzero transition
has length at least one.  Its final transition into \(0\) has length at
least two, because the only unit generator arcs that formerly entered
\(0\) have both been subdivided.  Hence the empty-state cost is at least
\(n+1\).  In a singleton state the tour has one additional required client,
so the unit lower bound alone is \(n+1\).

Write
\[
 H:=\operatorname{OPT}(\{x_1,x_2\}),\qquad h:=H/L.
\]
Concatenating the two singleton depot tours is a legal all-active closed
walk of length \(2L\), and shortcutting repeated visits cannot increase its
cost.  Therefore
\[
                         1\le h\le2.                 \tag{1}
\]
The exact expected posterior value is
\[
 P=(1-p^2)L+p^2H
   =L\bigl(1+(h-1)p^2\bigr).                         \tag{2}
\]

## 2. Two universal causal policies

The following argument uses no special property of the mixed \(+1,+2\)
shortest paths.

First, fix an optimal all-active tour and call the two selectors in its
tour order.  On every deletion realization, the induced tour is a shortcut
of the all-active tour.  Thus this policy costs at most \(H\).

Second, execute the \(+1\) singleton tour, calling \(x_1\) at its position.
After returning to the depot, call \(x_2\).  If \(x_2\) is inactive there is
no movement.  If it is active, append the depot roundtrip
\[
 0\longrightarrow x_2\longrightarrow0.
\]
That roundtrip costs at most \(L\), because it is contained, after
shortcutting, in the \(+2\) singleton depot tour.  Hence the second policy
has expected cost at most
\[
                         L(1+p).                     \tag{3}
\]
The same argument works with the roles of the two modes reversed.

Combining (2)--(3),
\[
 \frac{\operatorname{OPT}_{\rm adapt}}
      {\operatorname{OPT}_{\rm post}}
 \le
 \frac{\min\{h,1+p\}}
      {1+(h-1)p^2}.                                  \tag{4}
\]

## 3. Exact optimization of the envelope

For fixed \(p\), the first branch in (4) is increasing in \(h\), while the
second branch is decreasing in \(h\).  Consequently the largest value over
\(1\le h\le2\) occurs at
\[
                         h=1+p.
\]
Substitution gives
\[
 R(p)\le\frac{1+p}{1+p^3}.                           \tag{5}
\]
Finally,
\[
 \frac{1+p}{1+p^3}\le\frac43
 \quad\Longleftrightarrow\quad
 (2p-1)^2(p+1)\ge0.                                  \tag{6}
\]
Equality holds only at \(p=1/2\).

Thus the two-generator ring, including all metric-closure shortcuts and
every possible mixed \(+1/+2\) repair route, has clairvoyance gap at most
\(4/3\).

The same conclusion holds for unequal probabilities \(p_1,p_2\).  Put
\(q=\min\{p_1,p_2\}\), and choose as the fixed singleton mode the selector
with the larger activation probability, so that the missed selector has
probability \(q\).  Then
\[
 \frac{\operatorname{OPT}_{\rm adapt}}
      {\operatorname{OPT}_{\rm post}}
 \le
 \frac{\min\{h,1+q\}}
      {1+(h-1)p_1p_2}
 \le
 \frac{1+q}{1+q^3}
 \le\frac43.                                        \tag{6'}
\]
Here the middle inequality uses \(p_1p_2\ge q^2\), and the last one is
(6).  Thus skewing the two selector probabilities does not help.

## 4. General two-mode theorem

The proof did not use the Cayley presentation.  It establishes the
following reusable statement.

### Theorem

Suppose a closed stochastic ATSP instance has two independent
probability-\(p\) selectors and arbitrary permanent clients.  Assume:

1. the empty and the two singleton posterior costs are all \(L>0\);
2. the all-active posterior cost is \(H\); and
3. each selector has a depot roundtrip of cost at most \(L\).

Then its clairvoyance gap is at most \(4/3\).

Condition 3 is automatic when a singleton optimum of cost \(L\) contains
the corresponding selector: splitting that tour at the selector and
shortcutting its two depot pieces gives the roundtrip bound.  Moreover,
concatenating the two singleton tours gives \(H\le2L\).  Hence every
equal-cost two-mode route code satisfies the theorem.

The obstruction is genuinely closed-depot.  One fixed all-active tour
caps every state by \(H\), while a fixed singleton mode followed by a
depot repair caps the expectation by \(L(1+p)\).  A vertical ladder or a
two-sheet cover does not escape unless it invalidates one of these two
ordinary metric tours, which it cannot do in a fixed directed metric.

## 5. What changes with \(m\ge3\) modes

For \(m\) independent probability-\(p\) selectors, suppose the empty and
all singleton state costs are \(L\).  Let
\[
 C_S:=\operatorname{OPT}(S),\qquad
 P=\sum_{S\subseteq[m]}p^{|S|}(1-p)^{m-|S|}C_S.
\]
Two elementary families of causal policies give
\[
 \operatorname{OPT}_{\rm adapt}
 \le
 \min\left\{
 C_{[m]},\
 \min_{\varnothing\ne T\subseteq[m]}
 \left(C_T+p\sum_{i\notin T}L\right)
 \right\}.                                         \tag{7}
\]
For the policy indexed by \(T\), execute a fixed optimal \(T\)-tour.  After
returning to the depot, call every selector outside \(T\); each active
selector is repaired by its depot roundtrip, of cost at most its singleton
tour length \(L\).

Consequently a necessary condition for a ratio above \(4/3\) is
\[
 \boxed{
 C_T+(m-|T|)pL>\frac43P
 \quad\text{for every nonempty }T\subseteq[m],
 \qquad
 C_{[m]}>\frac43P.}                                \tag{8}
\]
This is a useful first audit for every proposed multi-lane route code.

If one ignores all intermediate-state constraints and writes
\(C_{[m]}=hL\), the full-tour policy and a singleton-plus-repairs policy
only imply
\[
 \frac{\operatorname{OPT}_{\rm adapt}}P
 \le
 \frac{\min\{h,1+(m-1)p\}}
      {1+(h-1)p^m}.                                 \tag{9}
\]
For \(m=2\), (9) is the sharp ceiling above.  For \(m\ge3\), (9) can exceed
\(4/3\); intermediate activation states can no longer be discarded.

## 6. A metric compatibility inequality for three modes

Assume \(m=3\) and all singleton costs equal \(L\).  For each unordered pair
\(\{i,j\}\), orient it \(i\to j\) according to an optimal pair tour that
encounters \(i\) before \(j\).  These three orientations form a tournament,
so they contain a directed Hamilton path, say
\[
                         i\longrightarrow j\longrightarrow k.
\]
Split the optimal \(\{i,j\}\)-tour at \(j\), and likewise split the optimal
\(\{j,k\}\)-tour at \(j\).  Concatenating the first tour's depot-to-\(j\)
piece with the second tour's \(j\)-to-depot piece gives a tour through
\(i,j,k\).  The two discarded complementary pieces together form a
depot-to-\(j\)-to-depot walk and therefore have total length at least the
singleton optimum \(L\).  Hence
\[
                    C_{\{i,j,k\}}
                \le C_{\{i,j\}}+C_{\{j,k\}}-L.       \tag{10}
\]

In the cardinality-symmetric case
\[
 C_\varnothing=C_{\{i\}}=L,\qquad
 C_{\{i,j\}}=cL,\qquad
 C_{\{1,2,3\}}=hL,
\]
equation (10) gives the necessary metric inequality
\[
                         h\le2c-1.                  \tag{11}
\]
Thus the attractive abstract table
\[
 C_S=L\quad(S\subsetneq[3]),\qquad C_{[3]}>L
\]
cannot occur in any directed metric: pairwise cheap mode compatibility
already concatenates to a cheap triple tour.

For three symmetric modes, (8) specializes to
\[
\begin{aligned}
 P/L
 &=(1-p)^3+3p(1-p)^2
   +3c\,p^2(1-p)+h\,p^3,\\
 \operatorname{OPT}_{\rm adapt}/L
 &\le \min\{h,\ c+p,\ 1+2p\},\\
 1\le c\le2,\qquad c\le h\le2c-1.
\end{aligned}                                      \tag{12}
\]
The scalar constraints in (12) alone still leave formal parameter points
above \(4/3\), for example \(p=1/2,c=3/2,h=2\).  They are not sufficient:
at that point the natural realization is a three-cycle preference metric,
and after seeing the first active mode a causal policy probes its cheap
successor first.  Only one of the three two-active states then incurs a
reverse transition.  A genuine three-mode construction must therefore
lower-bound an adaptive decision tree, not merely all fixed mode tours.
