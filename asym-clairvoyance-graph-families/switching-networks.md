# Switching networks

## Executive verdict

**Proof status:** a rigorous ratio-one theorem covers the most direct
butterfly/Beneš use (independently active output terminals).  Standard
switching fabrics are static reusable digraphs, not hidden stateful switches.
They let each metric move choose a fresh route, so rearrangeability,
edge-disjoint routing, and superconcentration do not create an irreversible
choice for the adaptive policy.  A positive construction would need a new
delayed-information switch gadget with a service-piece penalty; at present
that is exactly the missing lemma.

## 1. Explicit butterfly terminal template

Let \(N=2^k\).  The directed butterfly has vertices
\((i,x)\), \(0\le i\le k\), \(x\in\{0,1\}^k\), and unit arcs
\[
 (i,x)\longrightarrow(i+1,x),\qquad
 (i,x)\longrightarrow(i+1,x\oplus e_i)
 \quad(0\le i<k).
\]
Add a depot \(r\), arcs \(r\to(0,x)\) of cost \(\alpha>0\) for every input,
and arcs \((k,x)\to r\) of cost \(\beta>0\) for every output.  Take directed
shortest-path closure.

The generating graph is strongly connected.  Every internal vertex can
continue to an output and then reach \(r\); from \(r\) one can select an input
whose butterfly path reaches any prescribed internal vertex.  Every
input-output path uses exactly \(k\) layered arcs, and an output's only
outgoing arc goes to \(r\).  Hence, writing \(o_x=(k,x)\),
\[
\begin{aligned}
 d(r,o_x)&=\alpha+k,\\
 d(o_x,r)&=\beta,\\
 d(o_x,o_y)&=\beta+\alpha+k \quad (x\ne y).
\end{aligned}
\]

Make only the outputs clients, with arbitrary independent probabilities
\(p_x\).

### Realization-wise a posteriori routing

If \(A\) contains \(q\) active outputs, any order of them has tour cost
\[
 (\alpha+k)+(q-1)(\beta+\alpha+k)+\beta
 =q(\alpha+\beta+k).
\]
This is also a lower bound: the only way to leave an output is its
\(\beta\)-arc to \(r\), and every subsequent output needs an
\(\alpha\)-arc and \(k\) layered arcs.  Thus
\[
 \operatorname{TSP}(A)=qC,\qquad C:=\alpha+\beta+k.
\]

### Universal adaptive lower bound

For **every** adaptive policy and realization, list the active clients in the
order in which the policy calls them.  The first active call costs
\(\alpha+k\), every later one costs \(C\), and the final return costs
\(\beta\).  Thus its realized cost is also exactly \(qC\), regardless of all
inactive calls or the chosen order.  Therefore
\[
 \operatorname{OPT}_{\rm adapt}
 =\operatorname{OPT}_{\rm post}
 =C\sum_x p_x .
\]
The ratio is \(1\) (apart from the vacuous all-zero instance).

### Failure audit

This equality already allows calling every output first, arbitrary
interleaving, and all butterfly routes.  Shortest-path closure is harmless
because the layer potential increases by one on every internal arc and every
output must reset through \(r\).  The network's many input-output paths do no
work for TSP: after an active output is reached, the walk cannot reuse the
prefix to reach another output.

The same proof applies to any layered switching network in which every
terminal-to-terminal move must reset through a common depot interface and all
depot-to-terminal routes have the same cost.  This includes the most direct
output-client use of a Beneš network.

## 2. Why a static switch is not a random switch

A \(2\times2\) switching cell usually offers a straight pairing and a crossed
pairing.  In a switching-network problem a *configuration* chooses one mode
for simultaneous commodities.  In the stochastic-TSP metric there is no
configuration variable.  The graph is fixed and known to both benchmarks.
A shortest path may use the straight route on one traversal and the crossed
route on a later traversal.  Arcs are reusable, and their costs are paid per
occurrence.

Consequently:

* a Beneš network's ability to realize every permutation does not force the
  salesperson to choose one global permutation;
* edge-disjoint paths in a butterfly or superconcentrator do not reduce the
  sum of sequential path lengths; and
* a random switch setting is not legal hidden information unless it is
  encoded by independent client activations.

Independent switch-client bits *can* index a dependent permutation.  The
permutation is then merely a deterministic route label derived from legal
bits.  What remains unproved is that a walk using routes from inconsistent
labels must pay extra.  In a standard reusable fabric that assertion is
false: each traversal can select its own local mode.

## 3. Local revelation is usually causally harmless

Consider a staged fabric in which, immediately before a binary branch, a
selector client says whether the desired continuation is straight or crossed.
An adaptive policy calls that selector upon reaching the stage.  If it is
inactive, no movement occurs and the policy calls the next permanent stage
marker on the straight wire.  If it is active, the forced move can itself
land on the crossed continuation.  The information is revealed before, or at
the same time as, the route choice it controls.

More abstractly, suppose the realization-wise posterior tours
\(\{T_x\}\) form a causal prefix tree: at every common prefix, the next
realization-dependent event is the call of a selector, and each outcome
extends the prefix exactly as \(T_x\) does.  Following that tree is a legal
adaptive policy with realized cost \(\operatorname{cost}(T_x)\).  Hence the
gap is one.  Ordinary layer-by-layer routing has precisely this dangerous
prefix-compatibility.

To produce a gap, selector information must arrive **after** an expensive
choice that it controls, as the midpoint does in Chapter 4.  Merely drawing
the selector after a switch in the generating graph is insufficient: calls
are remote, so the policy can call it before entering the network.

## 4. Delayed-information permutation-network attempt

A plausible but incomplete template is:

1. put permanent payload blocks at \(N\) input and \(N\) output ports of a
   butterfly or Beneš network;
2. represent each intended switch bit by an independent stochastic client;
3. let the activation vector \(x\) index a permutation \(\pi_x\);
4. provide a posterior tour \(T_x\) that serves payload blocks in the order
   induced by \(\pi_x\).

For this to be useful, three additional properties are necessary.

First, all \(T_x\)'s must have a common upper bound \(P\), including their
depot connections and all active switch clients.  Second, any walk that mixes
the early part of \(T_x\) with the late part of \(T_y\) must incur a
quantified switching penalty.  Standard switching cells do not have this
property.  Third, calling all switch clients before entering the payload
network must itself cost at least the proposed adaptive lower bound.  If the
selectors lie in a set of directed diameter \(D\), the crude selector-first
cost is at most
\[
       D\sum_v p_v
\]
before the known-realization payload route is begun.

The independent-bit condition is satisfied in this template, but the second
and third geometric properties are open.  A uniformly random permutation
cannot simply be declared as the hidden state; it must be derived from the
independent selector clients.

### Interleaving and shortcut audit

An adaptive policy may serve an input block, remotely call selectors from
many later layers, traverse part of one nominal permutation route, and then
return through another.  A proof based on serving one input-output connection
at a time is therefore invalid.  One needs ports and service pieces, charging
every re-entry exactly as in Chapter 4.

High connectivity makes shortest-path closure especially dangerous.  Parallel
switching paths may concatenate into an unintended cheap repair.  A valid
gadget needs either a layer potential proving every mixed path expensive or a
quotient count that survives all internal routes.

## 5. Relation to the recursive triangle

A butterfly can replace a complete bundle of arcs between two large port
sets: scale its layer costs so every permitted port-to-port route costs
\(w_k\).  With a suitable layer potential and no reverse shortcuts, this may
give a compact implementation of the Chapter 4 port bundles.

That substitution does not improve the ratio.  It preserves the same
top-level directed-triangle accounting, while any extra fabric cost common to
both benchmarks can only dilute the ratio.  Placing independent triangle
switches at many network cells is likewise additive replication unless the
outer network proves a global incompatibility between their modes.

### Verdict

**Rigorous natural-template no-go.**  The output-terminal butterfly/Beneš
construction has ratio exactly \(1\).  More generally, standard reusable
switching-network functionality supplies alternative paths but no persistent
switch state, so it does not by itself create a clairvoyance gap.  At best it
can compactly realize the existing \(4/3\) recursive port bundles.  No
ratio above \(4/3\) is established.

### Next lemma

The decisive next target is a **delayed reusable-switch penalty lemma**.
Construct a finite strongly connected directed metric with two input ports,
two output ports, downstream independent selector clients, and two cheap
posterior modes, such that:

1. each selector realization has a mode-consistent open service of cost \(P\);
2. every causal service, including one that calls all selectors first and one
   that arbitrarily interleaves the four port blocks, has expected adjusted
   cost at least \((4/3+\varepsilon)P\); and
3. every extra service piece is charged to a distinct boundary crossing, so
   shortest-path mixing of the straight and crossed modes cannot erase the
   penalty.

Without such a cell, composing butterflies, Beneš networks, or
superconcentrators only composes reusable paths and cannot prove the desired
improvement.
