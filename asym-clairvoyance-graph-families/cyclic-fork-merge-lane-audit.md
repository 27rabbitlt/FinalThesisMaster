# Two- and three-lane cyclic fork/merge families

## Outcome

The smallest cyclic fork/merge families do not retain a nontrivial lane
monodromy.

* With two lanes, requiring every middle client to have at least two
  predecessors and two successors makes every consecutive interface
  complete bipartite.  The reachability poset is an ordinal sum of
  antichains and has a realization-wise optimal causal sweep.  Its
  clairvoyance gap is exactly one.
* With three lanes, the sparsest such interface is
  \(K_{3,3}\) minus a perfect matching.  If \(A_t\) is its Boolean
  adjacency matrix, then
  \[
                         A_tA_{t+1}>0                 \tag{0.1}
  \]
  entrywise for every pair of missing matchings.  Thus every lane reaches
  every lane two layers later.  The missing-edge permutations have no
  persistent product or holonomy: after two steps the lane state is
  completely forgotten.
* For the natural \(L\)-layer three-lane family with iid activation
  probability \(p_L\), this transparency gives an asymptotic gap of one for
  **every** probability scale:
  \[
       \frac{\inf_\pi\mathbb E K_\pi}
            {\mathbb E\operatorname {width}(P_L[A])}
                         =1+o(1).                     \tag{0.2}
  \]
  Here \(K_\pi\) is the number of increasing service runs.  Consequently the
  associated positive directed metrics also have adaptive/posterior ratio
  \(1+o(1)\), after repetition amortizes the depot constant.

The only local event that can make a causal two-path assignment choose the
wrong lane is a pair of adjacent layers that each contain at least two
active clients.  Its expected count is at most
\[
                         16L p_L^4.                   \tag{0.3}
\]
If \(Lp_L^3=O(1)\), (0.3) is \(o(1)\).  If \(Lp_L^3\to\infty\), a fully
active layer appears with high probability, the realized width is three,
and a fixed three-path service is asymptotically optimal.

Thus the smallest cyclic switch has the opposite of the desired
whole-cycle capacity obstruction.  A failed local choice is forgotten
within two layers, and one later sweep can batch all such repairs.  A
surviving algebraic lift needs at least four lane states or an interface
whose Boolean products remain sparse for an unbounded number of layers.

## 1. Positive metric and run identity

Let the client set be divided into layers
\[
                         V_1,\ldots,V_L,
             \qquad V_t=\{v_{t,0},\ldots,v_{t,q-1}\}.
\]
Choose a bipartite relation from \(V_t\) to \(V_{t+1}\), and take its
transitive closure to obtain a poset \(P_L\).  The word ``cyclic'' refers
to the repeated lane-switch pattern and its missing-edge permutation; the
service instance is cut at the depot between layers \(L\) and \(1\).
Closing that cut is one common reset.  Making the layer indices themselves
cyclic relations would create a directed cycle and would not define a
reachability poset.

Fix \(0<\varepsilon<1\).  For distinct clients define
\[
 d_\varepsilon(x,y)=
 \begin{cases}
   \varepsilon,&x<_{P_L}y,\\
   1,&\text{otherwise},
 \end{cases}                                          \tag{1.1}
\]
put \(d_\varepsilon(x,x)=0\), and give the depot unit distance in both
directions to every client.  Transitivity proves the directed triangle
inequality exactly as usual: two \(\varepsilon\)-arcs imply that the direct
arc is also an \(\varepsilon\)-arc, while every other nontrivial two-arc
path costs at least \(1+\varepsilon\).

For a realization with \(N\) active clients and \(K\) maximal increasing
runs, the closed tour costs
\[
             C_\varepsilon(N,K)
                =1+\varepsilon N+(1-\varepsilon)K.    \tag{1.2}
\]
The posterior uses
\[
                         K^*(A)=\operatorname {width}(P_L[A]).
                                                               \tag{1.3}
\]
Disjoint repetition and \(\varepsilon\downarrow0\) reduce the metric ratio
to the run/width ratio in (0.2).

## 2. Two lanes collapse to an ordinal sum

Suppose \(q=2\), and every vertex of an internal layer has at least two
predecessors in the preceding layer and at least two successors in the next
layer.  There are only two vertices on either side.  Hence every consecutive
interface is \(K_{2,2}\):
\[
                         V_t<V_{t+1}.                 \tag{2.1}
\]
Therefore
\[
                    P_L=V_1\oplus V_2\oplus\cdots
                                  \oplus V_L.          \tag{2.2}
\]

Let \(X_t\) be the number of active vertices in layer \(t\).  A causal sweep
repeatedly calls one still-uncalled vertex from every layer in increasing
layer order.  Inactive calls cause no movement.  The number of nonempty
sweeps is
\[
                         \max_t X_t.                  \tag{2.3}
\]
An antichain is contained in one layer, so
\[
              \operatorname {width}(P_L[A])
                         =\max_tX_t.                  \tag{2.4}
\]
Equations (2.3)--(2.4) prove realization-wise equality.  There is no
two-lane fork/merge candidate under the stated degree requirement.

## 3. Three lanes: two-step transparency

Now let \(q=3\).  The sparsest interface in which every vertex has two
predecessors and two successors is
\[
                 G_t=K_{3,3}\setminus M_{\sigma_t},   \tag{3.1}
\]
where \(M_{\sigma_t}\) is the perfect matching
\[
                 \{(i,\sigma_t(i)):i\in\{0,1,2\}\}.
\]
Thus
\[
 v_{t,i}<v_{t+1,j}
       \quad\Longleftrightarrow\quad j\ne\sigma_t(i). \tag{3.2}
\]

### Lemma 1 (complete two-step product)

For every \(i,j\) and every two consecutive interfaces, there is a lane
\(h\) such that
\[
                         i\to h\to j.                 \tag{3.3}
\]
Equivalently, the Boolean product of their adjacency matrices is entrywise
positive.

### Proof

The first arc forbids only
\(\sigma_t(i)\), and the second arc forbids only
\(\sigma_{t+1}^{-1}(j)\) as a choice of \(h\).  At most two of the three
lanes are forbidden, so at least one admissible \(h\) remains. \(\square\)

Consequently,
\[
        v_{t,i}<v_{s,j}\qquad\text{for every }s\ge t+2
        \text{ and every }i,j.                        \tag{3.4}
\]
No product
\(\sigma_{L-1}\cdots\sigma_1\) survives in reachability.  The algebraic
``sheet label'' has already disappeared after two interfaces.

### Width

By (3.4), an antichain is contained in two consecutive layers.  Inside one
layer it has size at most three.  Across one interface, the only
incomparable cross pairs are the three missing matching edges.  If an
antichain uses two vertices on one side, no vertex on the other side is
incomparable with both.  Hence
\[
                         \operatorname {width}(P_L)\le3.       \tag{3.5}
\]
For a realization \(A\), its width is characterized locally:

* it is three if some layer has three active vertices;
* in the absence of such a layer, it is two precisely when there is either
  a layer with two active vertices or an active missing-matching pair in
  two consecutive layers;
* otherwise it is one, provided \(A\ne\varnothing\).

This characterization follows directly from (3.4) and the matching form of
the cross-layer incomparabilities.

## 4. A fixed three-path ceiling

Every graph \(G_t\) in (3.1) contains a perfect matching.  Choose one such
matching at every interface.  Their union is three vertex-disjoint paths
through all layers.  Processing these three fixed paths consecutively gives
a legal causal policy with
\[
                         K_{\rm fixed}(A)\le3.         \tag{4.1}
\]

If activations are iid with probability \(p\), a specified layer is fully
active with probability \(p^3\).  Different layers are independent, so
\[
 \Pr[\operatorname {width}(P_L[A])=3]
       \ge1-(1-p^3)^L.                                \tag{4.2}
\]
Therefore
\[
 \frac{\mathbb E K_{\rm fixed}}
      {\mathbb E\operatorname {width}}
 \le
       \frac{3}{3(1-(1-p^3)^L)}
 =
       \frac1{1-(1-p^3)^L}.                           \tag{4.3}
\]
In particular,
\[
             Lp^3\longrightarrow\infty
       \quad\Longrightarrow\quad
 \frac{\mathbb E K_{\rm fixed}}
      {\mathbb E\operatorname {width}}
                         =1+o(1).                     \tag{4.4}
\]

This already covers every fixed positive activation probability and the
entire dense regime.

## 5. Sparse regime: only adjacent double layers are ambiguous

Call a layer **double** if it contains at least two active clients, and
**triple** if all three are active.  Let
\[
 B=\#\{t:V_t\text{ and }V_{t+1}\text{ are both double}\}.       \tag{5.1}
\]

The following deterministic producer is the relevant whole-run statement.

### Lemma 2 (two-path producer with local defects)

There is a causal producer for the three-lane family satisfying
\[
                  K_\pi(A)
       \le \operatorname {width}(P_L[A])+B(A).         \tag{5.2}
\]

### Proof

Process the layers from left to right in increasing runs.  At a current
lane \(i\), query the two allowed clients of the next layer before the one
forbidden client.  Stop querying that layer as soon as an active allowed
client is found; the remaining clients are left for the next run.  If both
allowed calls are inactive, leave the forbidden client for the next run and
continue two layers ahead.  Lemma 1 makes that two-layer continuation
unrestricted.

Induct on the processed-layer prefix while retaining the unserved clients
as the next run's frontier.

* A layer with zero or one active client creates no choice.
* If exactly one of two consecutive layers is double, its two active
  clients can be put on the two current runs.  A singleton in the adjacent
  layer is forbidden from at most one of them and hence continues the
  other.  The state is completely reset at the next layer by Lemma 1.
* A triple layer needs three runs, but the realized width is already three,
  so this creates no excess over the right side of (5.2).
* The only remaining case has two adjacent double layers.  The first active
  allowed call may commit to the wrong one of the two perfect matchings
  between their active pairs.  Starting one additional run repairs the
  choice, and Lemma 1 again resets the lane state after the following
  boundary.

Charge that additional run to the corresponding term of \(B\).  Even when
several double pairs overlap, charging each boundary separately only
overcounts the number of new runs.  This proves (5.2). \(\square\)

The proof explicitly permits future-layer probing and arbitrary inactive
answers.  Its key point is not an offline recoloring: an allowed client is
queried before the forbidden one, and any wrong first-active commitment is
charged immediately to an adjacent double pair.

For iid probability \(p\),
\[
\begin{aligned}
 \Pr[V_t\text{ is double}]
   &=3p^2(1-p)+p^3\\
   &=3p^2-2p^3
    \le4p^2.                                          \tag{5.3}
\end{aligned}
\]
Independence of different layers gives the exact expectation
\[
 \mathbb E B
  =(L-1)(3p^2-2p^3)^2
  \le16Lp^4.                                          \tag{5.4}
\]
Lemma 2 yields
\[
 \inf_\pi\mathbb E K_\pi
 \le
 \mathbb E\operatorname {width}
       +(L-1)(3p^2-2p^3)^2.                           \tag{5.5}
\]

## 6. All activation scales give gap one

Let \(p=p_L\) be arbitrary.

### Dense case

If \(Lp_L^3\to\infty\), (4.4) gives the result.

### Complementary case

Suppose \(Lp_L^3=O(1)\).  Unless \(L\) stays bounded, this implies
\(p_L\to0\), and then
\[
                  Lp_L^4=p_L(Lp_L^3)=o(1).            \tag{6.1}
\]
By (5.5),
\[
 \inf_\pi\mathbb E K_\pi
 \le\mathbb E\operatorname {width}+o(1).              \tag{6.2}
\]
If \(Lp_L\) is bounded away from zero, the expected width is bounded away
from zero, so division gives (0.2).  If \(Lp_L\to0\), then
\[
\begin{aligned}
 \mathbb E\operatorname {width}
   &\ge\Pr[A\ne\varnothing]
     =3Lp_L+O((Lp_L)^2),\\
 \mathbb E B&=O(Lp_L^4),
\end{aligned}                                         \tag{6.3}
\]
and the relative error is \(O(p_L^3)=o(1)\).  The same conclusion follows.

Since every causal run count is at least the posterior width,
\[
 1\le
 \frac{\inf_\pi\mathbb E K_\pi}
      {\mathbb E\operatorname {width}}
 \le1+o(1),                                           \tag{6.4}
\]
proving (0.2).

## 7. Monodromy audit

The intended cyclic obstruction would assign a lane permutation to every
interface and argue that their product forces a global repair.  That
argument tracks the **missing** matching
\(M_{\sigma_t}\), not the actual allowed relation
\(K_{3,3}\setminus M_{\sigma_t}\).

For reachability and shortest-path closure, allowed relations compose by
Boolean matrix multiplication.  Lemma 1 says their product is already the
all-ones matrix after two factors.  Therefore:

1. a policy may skip one problematic layer without retaining a lane label;
2. a repair run can resume in any lane two levels later;
3. separated local failures can be placed on the same repair run; and
4. the formal permutation product of the missing edges is not a metric or
   capacity invariant.

A genuine whole-cycle obstruction must keep at least two lane states
distinguishable under every product of \(o(L)\) allowed-interface matrices.
Neither the two-lane complete interface nor the three-lane
complement-of-matching interface has this property.

## Verdict

**Rigorous ceiling for the smallest cyclic fork/merge families.**  Two
lanes give an ordinal-sum poset and exact gap one.  Three lanes with the
minimal degree-two interfaces have complete two-step reachability and
asymptotic gap one for every iid product-activation scale.  The exact defect
constant is
\[
                 (L-1)(3p^2-2p^3)^2\le16Lp^4.
\]

To seek a construction above \(4/3\), one must move beyond these families:
at least four lanes, non-complement interfaces with sparse long Boolean
products, or a non-poset quotient carrying a conserved directed circulation
that shortest-path closure cannot erase.
