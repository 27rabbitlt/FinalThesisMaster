# Shared problem brief

## Model

An instance consists of a depot \(r\), clients \(V\), independent activation
bits \(X_v\sim\mathrm{Bernoulli}(p_v)\), and a directed metric \(d\).  The
metric may be presented as directed shortest-path distances in a finite
strongly connected weighted digraph.

- The **a posteriori** benchmark first sees the entire active set and then
  takes the cheapest depot tour through it.
- An **adaptive** policy may call any uncalled client.  An inactive call causes
  no movement.  If the called client is active, the salesperson must move to
  it immediately.  The next call may depend on all previously seen outcomes
  and the current position.
- The clairvoyance gap is
  \(\operatorname{OPT}_{\mathrm{adapt}}/
    \operatorname{OPT}_{\mathrm{post}}\).

The current asymmetric construction in `thesis/main.tex`, Chapter 4, is a
recursive directed-triangle family.  At level \(l\),
\[
\operatorname{OPT}_{\mathrm{post}}=(3l+4)2^{l-2},\qquad
\operatorname{OPT}_{\mathrm{adapt}}=(l+1)2^l,
\]
so its ratio is \(4(l+1)/(3l+4)\to4/3\).

## Goal

For each proposed graph family, seek one of the following, in decreasing order
of value:

1. a complete construction and proof of a ratio strictly above \(4/3\);
2. a credible asymptotic construction with explicit lemmas still to prove;
3. a rigorous obstruction/no-go statement for the most natural use of the
   family;
4. a precise hybridization with the recursive-triangle mechanism.

“The graph has expansion/high girth/many routes” is not by itself evidence of
a clairvoyance gap.  The hidden activation bits must affect which *directed
route or order* is cheap, while an adaptive policy must irreversibly pay before
learning the relevant bits.

## Required proof obligations

Every positive proposal must specify:

1. **Directed metric:** vertices, weighted arcs, strong connectivity, and why
   shortest-path closure does not introduce destructive shortcuts.
2. **Activation law:** which clients are permanent/stochastic and their
   independent probabilities.  Correlated codewords or random subspaces are
   outside the model unless converted to independent client activations.
3. **A-posteriori upper bound:** an explicit tour for every realization, or a
   distributional bound with all exceptional events controlled.
4. **Adaptive lower bound:** a statement applying to every causal policy.
   Preferred tools include paired-realization arguments, boundary/open-service
   recurrences, Bellman subsolutions, indistinguishability, or a charging
   argument robust to arbitrary interleaving.
5. **Ratio calculation:** leading constants and parameter regime.
6. **Failure audit:** early probing, calling a whole separator first,
   interleaving blocks, shortest-path shortcuts, and using inactive calls as
   free information.

## Key model-specific warnings

- Calls are remote: a policy may call any uncalled vertex from its current
  position.
- Only an **active** call forces movement.  Therefore a hidden “switch bit”
  represented by one client can often be queried early at expected cost only
  \(p\,d(u,v)\).
- Activation bits are mutually independent.  A code/design may index routes
  or constraints, but the realized active set cannot simply be a uniformly
  random codeword.
- Expansion and high girth can lower the cost of routing many terminals for
  both benchmarks.  A gap needs an orientation/order conflict, not merely
  distance.
- A proof based on serving a subgraph in one visit must survive arbitrary
  interleaving.  Use ports and charge every extra service piece, as in the
  current Chapter 4 proof.

## Candidate families

1. Projective-plane incidence graphs.
2. Generalized polygons.
3. Cayley digraphs.
4. High-girth lifts and Ramanujan bigraphs.
5. Linear codes, designs, and locally testable structures.
6. Finite buildings.
7. Switching networks.
8. Algebraic lifts.

