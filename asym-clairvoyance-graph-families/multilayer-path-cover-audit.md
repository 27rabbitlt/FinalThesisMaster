# Multi-layer reachability metrics: exact identities and skip obstruction

## 1. A valid positive metric

Let \(H\) be a graded acyclic digraph with levels \(0,\ldots,L\), all arcs
going from level \(i\) to \(i+1\).  Fix \(0<\varepsilon<1/L\).  Give every
arc of \(H\) length \(\varepsilon\), add a direct arc of length \(1\) between
every ordered pair of distinct vertices, and take shortest-path closure.
Then
\[
 d(x,y)=
 \begin{cases}
  \varepsilon(\operatorname {lev}(y)-\operatorname {lev}(x)),
     &y\text{ is reachable from }x,\\
  1,&\text{otherwise}.
 \end{cases}                                             \tag{1}
\]
This is a genuine directed metric after taking all generating lengths
positive.

For any ordering of an active set, split the ordering into its maximal
reachable runs.  If there are \(K\) runs and their first and last levels are
\((s_j,t_j)\), its internal cost is exactly
\[
       (K-1)+\varepsilon\sum_{j=1}^K(t_j-s_j).             \tag{2}
\]
With unit depot departure and return, add \(2\).  Thus as
\(\varepsilon\downarrow0\), the construction really does approach the
causal chain-cover versus posterior path-cover problem; there is no hidden
adjacency metric error.

Equation (2) also shows that using a path through inactive or uncalled
vertices is legitimate.  The intermediate vertices need not be service
events.

## 2. Transitive skips in a high-girth layered graph

Suppose each consecutive-level graph is \(d\)-regular and is locally
tree-like through depth \(t\).  If \(D_t(x)\) is the set of level-\(i+t\)
descendants of \(x\), then
\[
       |D_t(x)|=(1-o(1))d^t.                              \tag{3}
\]
Activate every vertex independently with probability \(p=c/d\), the natural
scaling that leaves a constant expected number of active immediate
successors.

After an active call at \(x\), a causal policy may call all vertices of
\(D_t(x)\) before leaving the current run.  Inactive calls cause no movement.
The first active answer, if one exists, is reachable from \(x\) at cost
\(t\varepsilon\).  Therefore the probability that this continuation attempt
fails is
\[
\begin{aligned}
 \Pr[D_t(x)\text{ contains no active vertex}]
   &=(1-p)^{|D_t(x)|}\\
   &\le
   \exp\!\left(-(1-o(1))c\,d^{t-1}\right).                 \tag{4}
\end{aligned}
\]
Already for \(t=2\), this is \(\exp(-\Theta(cd))\), not a positive constant.

Consequently an immediate-layer matching deficit cannot be charged at every
layer.  Reachability exposes the entire descendant bank, and a policy can
skip the contested layer while remaining in the same cheap run.  Large
girth strengthens (3) and therefore strengthens this adaptive escape.

## 3. Why permanent intermediate gates do not immediately repair it

One may put a permanent antichain at every intermediate level.  Then a path
cover that skips a gate must cover that gate elsewhere.  This correctly
prevents a skip from being free in the path-cover count.

However, using a stochastic edge-selector \(s_{uv}\) with
\[
       u<s_{uv}<v
\]
turns one stage into a probing-with-commitment matching problem.  An active
call to \(s_{uv}\) from \(u\) irrevocably realizes the local chain
\(u,s_{uv},v\); an inactive call leaves the policy at \(u\).  Selector
clients not used by the matching remain required clients, so they cannot be
discarded as nonexistent edges.

Across several stages, transitivity again makes an unused selector at an
earlier stage comparable with later selectors whenever there is a gate path
between them.  Thus local unmatched-selector deficits can be collected in
the same later chain.  Counting each stage deficit separately double-counts
these transitive repair chains.

## 4. Structural conclusion

The natural multi-layer choices have the following exhaustive failure modes.

* If the reachability poset is series-parallel, the causal round-robin
  producer in `nested-cut-obstruction.md` attains posterior width exactly.
* If consecutive layers branch and merge with high girth, (4) supplies
  exponentially reliable causal skip successors.
* If permanent gates are inserted to make skips consume path capacity, the
  local choice becomes a matching-with-commitment cell, while transitive
  chains can batch unmatched selectors from several stages.

Therefore a valid proof needs a non-series-parallel layered poset together
with a potential that assigns every transitive repair chain to only one
stage.  A sum of immediate-interface matching deficits is not such a
potential.
