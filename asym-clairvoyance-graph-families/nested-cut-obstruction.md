# Nested directed cuts: an exact obstruction

## Scope

This note audits the most literal version of a nested-cut construction.  The
underlying generating graph is a tree, with an arbitrary nonnegative length
in each direction on every tree edge; distances are directed shortest-path
distances.  Clients may occur at arbitrary tree vertices and have arbitrary
activation probabilities (independence is not needed).

For this whole class, a posteriori and adaptive stochastic TSP have the same
value realization by realization.  Thus a lower bound based only on a
laminar family of directed cut crossings cannot yield a clairvoyance gap.
Cycles whose cuts overlap nonlaminarly are indispensable.

## The directed-tree metric

Let \(T=(V,E)\) be a tree rooted at the depot \(r\).  For every undirected
edge \(e=\{u,v\}\), assign lengths
\[
       \ell_{u v}\geq0,\qquad \ell_{v u}\geq0 .
\]
The distance \(d(x,y)\) is the length of the unique directed walk along the
undirected \(x\)-to-\(y\) path.  Zero lengths can be replaced by positive
lengths without changing the argument.

For an edge \(e\), let \(T_e\) be the component below \(e\), away from the
root, and write
\[
       c_e:=\ell_{\text{down},e}+\ell_{\text{up},e}.
\]

## Realization-wise posterior optimum

For an active set \(A\), let
\[
       E(A):=\{e\in E:A\cap T_e\neq\varnothing\}.
\]
Every closed walk from \(r\) visiting \(A\) must cross each edge in \(E(A)\)
at least once downward and once upward.  Consequently,
\[
       \operatorname {OPT}_{\rm post}(A)
       \geq \sum_{e\in E(A)}c_e.                     \tag{1}
\]
The usual depth-first traversal of the minimal subtree connecting
\(\{r\}\cup A\) crosses every such edge exactly once in each direction, so
equality holds:
\[
       \operatorname {OPT}_{\rm post}(A)
       =\sum_{e\in E(A)}c_e.                         \tag{2}
\]

## One fixed call order attains (2)

Fix a planar embedding of \(T\), and list all client vertices in depth-first
preorder.  If several clients occupy one tree vertex, list them consecutively
in any order.  The adaptive policy calls clients in this fixed order and
finally returns to \(r\).  Inactive calls cause no movement, as required by
the remote-call model.

For a realization \(A\), shortcut the full depth-first traversal past all
inactive clients.  Expanding every resulting metric move along its unique
tree path has the following property:

* if \(A\cap T_e=\varnothing\), edge \(e\) is never crossed;
* if \(A\cap T_e\neq\varnothing\), the cyclic depth-first order enters the
  active part below \(e\) once and leaves it once.

Therefore the realized cost of this fixed policy is exactly
\[
       \sum_{e\in E(A)}
       \bigl(\ell_{\text{down},e}+\ell_{\text{up},e}\bigr),
                                                               \tag{3}
\]
which equals the posterior optimum by (2).  Taking expectations gives
\[
       \operatorname {OPT}_{\rm adapt}
       =\operatorname {OPT}_{\rm post}.              \tag{4}
\]

The proof permits deterministic clients, arbitrary correlations, transit
through inactive or uncalled vertices, and zero or positive directed edge
lengths.

## Consequence for recursive proposals

A potential of the form
\[
       \Phi(W)=\sum_{e}c_e\,
       \mathbf 1\{\text{the service has unfinished work below }e\}
\]
correctly charges every repeated crossing of a laminar cut.  However, the
same laminar structure also supplies the realization-wise optimal fixed
depth-first order above.  Hence this potential cannot separate adaptive and
a-posteriori costs.

The same obstruction applies to articulation compositions of local modules:
after contracting each module, a tree scaffold fixes a depth-first order, and
the module contributions add.  Such a composition cannot improve the largest
gap already present in one closed, fixed-interface module.

A successful cut construction must therefore use overlapping, nonlaminar
cuts.  It must also prove that a repair walk cannot cross one common cycle
and thereby repair several local order mistakes at once; laminarity avoids
that batching only by eliminating the information gap altogether.

## A parallel obstruction for series-parallel reachability posets

There is an analogous exact policy for the zero/one reachability metric of
every series-parallel poset.  This covers a large class of proposed
multi-layer path-cover constructions.

Start with singleton posets and use the two operations

* \(P\parallel Q\): no element of \(P\) is comparable with an element of
  \(Q\);
* \(P\oplus Q\): every element of \(P\) is below every element of \(Q\).

For a realized active set, width satisfies
\[
\begin{aligned}
 \operatorname {width}(P\parallel Q)
   &=\operatorname {width}(P)+\operatorname {width}(Q),\\
 \operatorname {width}(P\oplus Q)
   &=\max\{\operatorname {width}(P),\operatorname {width}(Q)\}.
                                                               \tag{5}
\end{aligned}
\]

Inductively, maintain a causal producer which outputs the active vertices as
a sequence of minimum many increasing chains.  For a parallel composition,
run the two producers consecutively; no chain can cross between the
components, so their chain counts add.  For an ordinal composition, run the
two producers in rounds.  In round \(j\), output the \(j\)-th chain produced
by \(P\), if any, followed by the \(j\)-th chain produced by \(Q\), if any.
The concatenation is a chain because \(P<Q\).  Empty rounds are detected by
inactive calls, which cause no movement.  The number of nonempty rounds is
the maximum of the two realized chain counts.

The construction is causal: a component producer is resumed only from its
own previously exposed call state, and outcomes in the other component are
independent side information.  In fact, the argument works under arbitrary
activation correlations if the producer is allowed to retain its complete
call history.

By (5), the producer uses exactly the realized width at every recursive
node.  Consequently, for the metric
\[
 d(x,y)=
 \begin{cases}
 0,&x\le y,\\
 1,&\text{otherwise},
 \end{cases}
\]
with unit depot arcs, the adaptive and posterior values agree for every
series-parallel poset.

In particular, merely adding more levels, parallel branches, ordinal
layers, or a tree of such operations cannot amplify a height-two matching
deficit.  A viable layered poset must contain genuinely non-series-parallel
diamonds whose choices overlap across levels.
