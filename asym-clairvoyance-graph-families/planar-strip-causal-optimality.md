# Planar two-permutation strips have a realization-wise optimal causal policy

## Outcome

The most natural way to keep a two-successor Boolean product sparse for many
layers is the strip
\[
             (t,i)\longrightarrow(t+1,i),\qquad
             (t,i)\longrightarrow(t+1,i+1).
\tag{0.1}
\]
This family has no clairvoyance gap.  After the coordinate change
\[
                         (t,i)\longmapsto(i,t-i),
\tag{0.2}
\]
its reachability order is the ordinary product order on two coordinates.
For every fixed active realization, a causal greedy producer uses exactly
the width of the active product order.  The statement is pathwise and
therefore holds for arbitrary independent, unequal activation
probabilities.

Thus increasing the number of lanes does not rescue the commuting
identity/shift interface.  A surviving union-of-two-permutations lift must
use genuinely noncommuting products; long sparse growth by itself is not
enough.

## 1. The strip is a product order

Let
\[
       V=\{(t,i):0\leq t\leq L,\ 0\leq i\leq q\}
\]
and retain whichever arcs in (0.1) have both endpoints in \(V\).  Write
\(\preceq\) for reachability.

For \(s\geq t\), a path from \((t,i)\) to \((s,j)\) makes exactly \(s-t\)
steps, each of which increases the lane coordinate by zero or one.  Hence
\[
 (t,i)\preceq(s,j)
 \quad\Longleftrightarrow\quad
 0\leq j-i\leq s-t.
\tag{1.1}
\]
Equivalently,
\[
                  i\leq j,\qquad t-i\leq s-j.
\tag{1.2}
\]
Under (0.2), these are precisely the two coordinate inequalities defining
the product order on \(\mathbb Z^2\).

Boundary truncation causes no problem: (1.1) uses a monotone lane sequence
between \(i\) and \(j\), so every intermediate lane remains between two
valid endpoints.  The same identification works for any finite subset of
the infinite strip.

The cylindrical version with lane indices modulo \(q\) is different only
because it adds the seam arc \(q-1\to0\).  Before one full wrap its local
pieces are copies of the strip; after \(q-1\) layers every lane reaches
every lane.  It therefore alternates between the product-order regime
analyzed here and the rapid-mixing regime, rather than supplying a new
long-lived commuting monodromy.

## 2. Causal greedy chain extraction

We prove the following slightly more general statement.

### Theorem 1

Let \(P\) be any finite subset of \(\mathbb R^2\), ordered coordinatewise.
Every point is independently active, with arbitrary known probability.
There is a causal policy which, for every active realization \(A\), produces
exactly
\[
                         \operatorname{width}(P[A])
\tag{2.1}
\]
maximal increasing service runs.

Use lexicographic order by the first and then the second coordinate.  Thus
two points with equal first coordinate occur in increasing second-coordinate
order and may correctly belong to the same chain.  Distinct points cannot
tie in both coordinates.

### Policy

Fix the lexicographic order just described.  Repeat the following round until
every point has been called.

1. Query the first still-uncalled point in lexicographic order, then
   continue in that order until an active point \(z\) is found.  This starts
   the round.  If no active point remains, stop.
2. With current active point \(z\), query, in lexicographic order, every
   still-uncalled point \(y\) satisfying \(y_2\geq z_2\).  Inactive calls
   cause no movement.  At the first active answer, move to \(y\), replace
   \(z\) by \(y\), and continue the same scan.
3. Points with second coordinate below the current \(z_2\) are left
   uncalled for later rounds.

Every active movement within a round is coordinatewise increasing, so one
round is one increasing service run.  The choice of the next query depends
only on already observed answers and the current point, hence the policy is
causal.  Notice in particular that the policy does not reveal a skipped
point: it merely postpones its call.

## 3. Exact optimality

Let \(R(a)\) be the round in which active point \(a\) is served.  We show
inductively that every active point served in round \(r\) is the final
element of an antichain of \(r\) active points whose first coordinates
strictly increase and second coordinates strictly decrease.

The assertion is trivial for \(r=1\).  Suppose \(a\) survives the first
\(r-1\) rounds and is served in round \(r\).  During round \(r-1\), the
lexicographic scan passed the location of \(a\).  The point was not
eligible for that run, so at that moment the current active point \(b\)
satisfied
\[
                         b_1<a_1,\qquad b_2>a_2.
\tag{3.1}
\]
Indeed lexicographic precedence gives \(b_1\leq a_1\); equality would force
\(b_2<a_2\), making \(a\) eligible, so the first inequality is strict.
The current point \(b\) was served in round \(r-1\), so \(R(b)=r-1\).

By induction there are active points
\[
       b_1,\ldots,b_{r-1}=b
\]
with strictly increasing first coordinates and strictly decreasing second
coordinates.  Appending \(a\) gives such a sequence of length \(r\).
It is an antichain in the product order.  Therefore
\[
                R(a)\leq\operatorname{width}(P[A]).
\tag{3.2}
\]

If the policy uses \(K\) nonempty rounds, choose an active point in its last
round.  Equation (3.2) gives
\[
                         K\leq\operatorname{width}(P[A]).
\tag{3.3}
\]
Conversely, every increasing service run is a chain, so any partition of
the active service order into \(K\) runs is a chain cover and Dilworth gives
\[
                         K\geq\operatorname{width}(P[A]).
\tag{3.4}
\]
Equations (3.3)--(3.4) prove Theorem 1.

## 4. Directed-metric consequence

Fix \(0<\varepsilon<1\), put
\[
 d(x,y)=
 \begin{cases}
   \varepsilon,&x<_{P}y,\\
   1,&x\not<_{P}y
 \end{cases}
 \qquad(x\ne y),
\tag{4.1}
\]
and give the depot unit distance to and from every client.  Transitivity
proves the directed triangle inequality.  If a realization has \(N\)
active clients and a service order has \(K\) maximal increasing runs, its
closed cost is
\[
                         1+\varepsilon N+(1-\varepsilon)K.
\tag{4.2}
\]
The posterior minimum has \(K=\operatorname{width}(P[A])\).  The causal
policy of Theorem 1 attains the same \(K\) on every realization, so
\[
             \operatorname{OPT}_{\rm adapt}
             =\operatorname{OPT}_{\rm post}.
\tag{4.3}
\]

This exact equality includes arbitrary inactive lookahead, interleaving,
and shortest-path transit.  It rules out all planar strip truncations,
staircase regions, and non-wrapping identity/shift lifts, with any product
activation law.

## 5. What remains

For interfaces which are unions of two permutations, the commuting case can
often be conjugated to (0.1) on each orbit and is therefore covered above.
To escape the theorem, successive interfaces must generate a genuinely
noncommuting word so that no two scalar coordinates make every allowed step
monotone.

That requirement alone is not positive evidence.  If Boolean products expand
quickly, descendant-bank probing erases local matching failures.  A viable
family would have to combine:

1. noncommuting permutation products;
2. Boolean products that remain sparse for an unbounded number of layers;
3. no coordinatewise or interval-order greedy decomposition; and
4. a whole-run lower bound robust to querying every reachable descendant.

The identity/shift strip satisfies item 2 but fails items 1 and 3 in the
strongest possible way: its causal and posterior values coincide
realization by realization.
