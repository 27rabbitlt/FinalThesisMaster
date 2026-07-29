# Why fully-online matching hardness does not directly give a TSP gap

## 1. The tempting reduction

Fully-online bipartite matching has hard instances on which every online
algorithm obtains less than \(2/3\) of the offline matching.  The strongest
published upper bound known to this note is \(0.6297\), due to Eckl,
Kirschbaum, Leichter, and Schewior.  This is numerically strong enough for
the standard cheap-transition calculation:
\[
       \frac{2N-\alpha T^*}{2N-T^*}>\frac43
       \quad\text{when }T^*\simeq N
       \text{ and }\alpha<\frac23.
\]

Their construction reveals a bipartite graph in chronological order.
Vertices have deadlines, and a hidden designated leaf in a recursively
revealed tree makes an early matching decision irreversible.  A terminal
KVV block supplies the final loss.

There are three separate obstacles to using that theorem in stochastic
TSP.

## 2. A TSP path has capacity two at an internal client

In matching, using edge \(uv\) consumes both \(u\) and \(v\): neither can
participate in another chosen edge.  In a TSP call sequence, an active
client \(v\) may support both
\[
                         u\longrightarrow v
                         \longrightarrow w.
\]
Thus a cheap-transition set has indegree and outdegree at most one, but
not total degree at most one.  On a layered acyclic graph it is a path
forest, not a matching.

This distinction destroys the hidden-leaf tree mechanism.  In a rooted
tree oriented from parent to child, a causal policy at an active parent
queries its uncalled children until the first active answer and then
recurses from that child.  It earns an incoming and an outgoing transition
at the child.  Realization by realization, it obtains every transition in
the posterior path forest.  The offline rule “match the parent to a leaf
so that the nonleaf children remain available” has no analogue: a nonleaf
child remains available as an outgoing endpoint even after it was used as
an incoming endpoint.

Splitting one logical vertex into an in-copy and an out-copy does not
restore matching capacity.  They are distinct TSP clients, so the posterior
may use both copies independently.  Making only one copy active would
require correlation.  If both are permanent, their service is common
baseline and their vertices remain reusable transit.

## 3. Static product activation does not reproduce graph revelation

The fully-online hard instance hides which child is the designated leaf
until the parent's deadline.  This is correlated graph revelation.  In the
stochastic-TSP model the metric, including every incidence, is known in
advance; only vertex activation bits are hidden, and those bits are
mutually independent.

A possible replacement is to define “leaf” as “a child having no active
descendant.”  It does not recover the matching proof.  The policy may call
descendants remotely.  Inactive calls are free, while the first active
descendant moves the policy there and can immediately be extended farther
down the tree.  Because path vertices have separate incoming and outgoing
capacity, this probing does not create the unmatched-leaf deficit used by
the online matching lower bound.

## 4. Chronological gates lose their clock state at a shared target

One can try instead to impose the online arrival/deadline order
\(u_1,\ldots,u_m\) geometrically.  Suppose a shared target \(v\) is eligible
at several times.  A cheap match at time \(i\) has the form
\[
                           u_i\longrightarrow v.
\]
After the call, the physical state is just \(v\).  It does not remember
the source index \(i\).  Therefore a proposed cheap continuation
\[
                           v\longrightarrow u_{i+1}
\]
cannot depend on which \(u_i\) preceded \(v\).  If all continuations
\(v\to u_{i+1}\) are installed, all of them are available after every
match and the clock can jump.  If only one is installed, \(v\) is eligible
at only one time and no longer has shared matching capacity.

Time-specific targets \(v_i\) preserve the clock but eliminate competition:
they are distinct clients.  Correlating their activations so that exactly
one copy represents the logical target is outside the product model.
Nonclient entry or exit ports do not help, because calling the common
active target lands at \(v\) and erases the port state.

This is the same common-client state-erasure phenomenon proved in
`mode-coded-common-payload-obstruction.md`.

## 5. Edge-client probe-commit gadgets split at the middle layer

Another natural translation gives every possible matching edge \(uv\) an
independently active selector client \(e_{uv}\), with cheap arcs
\[
                         u\to e_{uv}\to v.
\]
It looks like stochastic matching with commitment: probing an active edge
selector from \(u\) moves the policy to it.

However, the posterior path-cover objective separates the two boundaries.
It may use
\[
                         u\to e_{uv}
\]
and
\[
                         e_{u'v}\to v
\]
with two different selector clients.  These are two legal path pieces and
have the same total cheap-transition saving as a single length-two piece.
The maximum posterior transition count is therefore the sum of the two
boundary matching numbers, not twice the size of a matching in the original
graph.  The source and target capacities have decoupled.

Weights do not create the missing conjunction.  A static metric assigns an
additive cost to \(u\to e\) and \(e\to v\); it cannot make the second saving
available only when the first arc was used.  Such a rule would again require
hidden path state.

## 6. Precise implication

The fully-online matching hardness is not a black-box construction for
stochastic TSP.  A valid embedding would need a **capacity-one clock
gadget** with all three properties:

1. a logical target is reusable as an option at several chronological
   stages but can be consumed only once;
2. after consuming it at stage \(i\), the route is forced to stage \(i+1\);
3. the gadget uses a fixed directed metric and independent client
   activations.

The state-erasure and middle-layer splitting arguments show that ordinary
shared vertices, time copies, and two-arc edge-selector paths do not have
these properties.  Until such a gadget is supplied, the \(0.6297\)
matching lower bound cannot certify a clairvoyance gap above \(4/3\).

## Reference

A. Eckl, A. Kirschbaum, M. Leichter, and K. Schewior,
*A Stronger Impossibility for Fully Online Matching*,
arXiv:2102.09432.
