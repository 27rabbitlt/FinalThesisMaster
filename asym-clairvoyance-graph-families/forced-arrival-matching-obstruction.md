# Obstruction to a forced-arrival matching reduction

## Intended reduction

The proposed construction has permanent stage gates
\(g_0,g_1,\ldots,g_T\), stochastic type clients revealed at stage \(i\),
and permanent resource clients \(u\in U\).  A cheap service of a type client
followed by \(u\) is intended to mean that the online arrival is matched to
resource \(u\).  Since a client is served once, it is tempting to regard
\(u\) as a unit-capacity resource.  Directed gate tolls are then supposed to
force the stages to be processed in their prescribed order.

There are two distinct implementations, and each has a structural failure.

## 1. Reachability-poset implementation

In a reachability-poset metric, inserting resource \(u\) for free between
stage \(i\) and stage \(i+1\) requires

\[
       x_i < u < x_{i+1},
\]

where \(x_i\) is the active type/state client at stage \(i\).

If the same resource is eligible for a later stage \(k>i\), the analogous
relation includes \(x_k<u\).  Stage order gives
\(x_{i+1}\le x_k\), and hence

\[
       u<x_{i+1}\le x_k<u,
\]

a contradiction.  Therefore a resource that can be inserted at zero cost is
eligible at only one temporal cut.  There is then no cross-stage competition
for resources and no online-matching lower bound.

The height-two representation avoids this cycle: type clients are minimal,
resources are maximal, and a chain cover has value

\[
 |L_{\rm active}|+|U_{\rm active}|-\nu,
\]

where \(\nu\) is a maximum matching.  But all type clients are then an
antichain.  A causal policy is free to choose their probing order.  Making
them comparable to enforce arrival order returns to the cycle above.

## 2. Positive-length directed metric implementation

Suppose instead that a resource supplies a short path

\[
       x_i\longrightarrow u\longrightarrow g_{i+1}.
\]

The metric is the shortest-path closure of the generating graph.  That path
is available whenever the policy moves from \(x_i\) to \(g_{i+1}\),
regardless of whether client \(u\) is unserved, already served, inactive, or
merely used as a transit vertex.  Serving a TSP client does not delete its
vertex or its incident arcs.

Consequently the same resource can be reused as a metric shortcut at every
eligible stage.  Its one-time client status does not impose matching
capacity.  Any distance table that counts a saving only the first time
\(u\) is traversed is state-dependent and is not a stochastic-TSP metric.

Making the resource terminal restores the ordering interpretation: only a
type client immediately followed by the still-unserved resource receives
the cheap transition.  But after a terminal resource the next stage must
pay a restart toll.  A shared resource has no memory of the stage at which
it was used, so its outgoing distances cannot both enforce the next stage
and remain available to several different stages.  Stage-specific resource
copies remove this problem but also remove unit capacity: the copies are
distinct clients and can all be used.

## 3. Activation encoding does not fix resource capacity

Independent marker banks can encode a random type imperfectly (for example,
by the index of the first active marker).  Ordered marker chains can make
later active markers cheap to serve.  This addresses one-hot correlation,
but it does not address the resource obstruction: a fixed resource vertex
is still reusable as transit, while terminal height-two resources still
leave the policy free to choose the order of the minimal type clients.

## Consequence

A valid matching reduction needs a separate, metric-compatible capacity
mechanism.  Permanent stage gates and resource vertices alone do not supply
one.  In particular, a proof cannot identify “resource already served” with
“resource unavailable”: the former changes the remaining client set but
does not change any future distance.

This rules out the direct forced-arrival construction in both the
reachability-poset and ordinary shortest-path formulations.  A viable
higher-layer construction would have to encode allocation by irreversible
client order rather than by reuse of a resource vertex, and simultaneously
prove that the policy cannot reorder the minimal clients.
