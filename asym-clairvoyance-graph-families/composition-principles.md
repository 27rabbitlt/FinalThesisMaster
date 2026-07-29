# Cross-family proof diagnostics

These observations are not a new gap construction.  They are filters for
deciding whether a structured graph is doing mathematical work beyond
replicating the existing directed-triangle gadget.

## 1. Additive replication does not amplify a local ratio

Suppose a construction decomposes into modules whose exact expected
a-posteriori and adaptive contributions are respectively \(P_i>0\) and
\(A_i\), plus a benchmark-independent common cost \(C\ge 0\).  If the
decomposition is genuinely additive for both benchmarks, then
\[
 \frac{C+\sum_i A_i}{C+\sum_i P_i}
 \le
 \max\left\{1,\max_i\frac{A_i}{P_i}\right\}.
\]
This follows by putting
\(\rho=\max\{1,\max_i A_i/P_i\}\) and observing that
\(C+\sum_i A_i\le \rho(C+\sum_i P_i)\).

Consequently, a lift, incidence graph, expander, or building that merely
places many independent copies of the \(4/3\) triangle switch cannot improve
the ratio.  To beat \(4/3\), the outer graph must **couple** local choices:
the route selected for one hidden bit must constrain the affordable choices
for many other bits.

This is only a diagnostic until the asserted additive decomposition is proved.
In particular, arbitrary interleaving can invalidate it in either direction.

## 2. A hidden graph label is not a legal random input

Voltage assignments, edge signs, selected generators, codewords, and apartment
choices belong to the fixed metric unless they are encoded by client
activations.  A construction cannot let the a-posteriori benchmark see a
random lift or random edge orientation while withholding it from the adaptive
policy: both benchmarks know the entire metric before play begins.

The legal source of uncertainty is a product distribution on client-presence
bits.  If a proposed algebraic object uses a correlated random word
\(Y=GX\), only the independent coordinates of \(X\) can directly be
activation bits.  The dependent coordinates of \(Y\) may index deterministic
routes or constraints, but cannot be declared independently present with that
joint law.

## 3. Selector-first audit

Let \(S\) be the stochastic clients whose outcomes select a route, apartment,
sheet, or permutation.  Every proposed lower bound must be tested against the
policy that calls all of \(S\) before serving the permanent backbone.

If \(S\cup\{r\}\) has directed diameter at most \(D\), this preliminary phase
can be implemented in any fixed order with expected movement at most
\[
 D\sum_{v\in S}p_v,
\]
before the final transition to the deterministic service and return are
included.  Thus a selector set of small expected cardinality or small diameter
cannot hide a macroscopic route choice.  A successful construction must make
early active selectors costly in exactly the scale on which full information
saves the a-posteriori tour.

The estimate is intentionally crude; optimizing the order can only strengthen
the audit.

## 4. Route-code formulation

A useful abstraction for the code/design/building candidates is a fixed
directed metric containing a large family of cheap depot tours
\(\{T_x:x\in\{0,1\}^k\}\).  The independent activation vector is
\(X=x\), and \(T_x\) must explicitly serve every permanent client and every
active selector client.  Its expanded paths may transit through other
vertices, including inactive clients.  The desired properties are:

1. \(T_x\) has cost at most \(P\) for every (or almost every) \(x\);
2. tours for well-separated words make genuinely incompatible early directed
   choices;
3. after any causal history leaving many bits unknown, every affordable next
   choice is bad for a constant fraction of the remaining completions;
4. calling the unknown selector bits first already incurs the target lower
   bound.

A code with large Hamming distance addresses only item 2.  Items 1, 3, and 4
are geometric statements in the directed metric and are the central missing
lemmas.

## 5. Shortcut-closure audit

All costs are measured after directed shortest-path closure.  Long penalties
drawn in a generating graph are invalid if a combination of incidence,
expander, inverse-generator, or fiber edges gives a cheaper path.  A robust
proof should use one of:

- a potential function certifying every shortcut has the claimed minimum
  length;
- scale separation, with an induction over levels;
- a quotient lower bound that survives lifting;
- explicit port boundaries and a charge for every extra service piece.

High connectivity makes this audit harder, not easier.

## 6. Stochastic clients are not open/closed switches

Activation affects the service requirement, not the generating graph.  A
shortest path is allowed to pass through an inactive or as-yet-uncalled client,
and such transit neither calls nor serves that client.  Therefore a switching
network cannot encode a random setting by saying that an active switch opens
one edge and an inactive switch opens another.  All edges and all transit
vertices remain available in both realizations.

The legal mechanism is subtler: calling an active selector forces the current
position to that selector, after which directed asymmetry makes some future
orders cheap and others expensive.  Any network proposal that relies on
random edge availability is a different stochastic-routing model.
