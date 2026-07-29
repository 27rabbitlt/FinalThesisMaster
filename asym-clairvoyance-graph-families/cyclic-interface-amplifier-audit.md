# Audit of a cyclic metric amplifier for weighted interfaces

## 1. The desired transfer

The finite gadget in `finite-weighted-interface-gadget.md` has capacity
baseline \(B=2\), posterior reward \(P\), and optimal causal reward \(C\),
with
\[
       \frac{B-C}{B-P}=\frac{359}{214}>\frac43.
\]
To turn this into the same stochastic-TSP cost ratio, an \(L\)-fold
composition would need
\[
\begin{aligned}
 \mathbb E\operatorname {OPT}_{\rm post}
     &=L(B-P)+o(L),\\
 \operatorname {OPT}_{\rm adapt}
     &\ge L(B-C)-o(L).
\end{aligned}
\tag{1}
\]
In particular, a common service term of order \(L\) cannot simply be
discarded.

## 2. Inactive vertices cannot gate metric paths

Suppose a proposed switch has a cheap entrance \(a\to u\), a cheap matched
arc \(u\to v\), and a cheap exit \(v\to b\).  If these are arcs of an
underlying length graph, then the metric closure gives
\[
                       d(a,v)\le d(a,u)+d(u,v)
\tag{2}
\]
and
\[
                       d(a,b)\le d(a,u)+d(u,v)+d(v,b).
\tag{3}
\]
These inequalities hold whether or not stochastic client \(u\) is active.
A policy may traverse the location of an inactive or unqueried client.
Therefore the cheap route through \(u\) cannot be used as an
activation-controlled gate.

This invalidates the naive optional-detour switch
\[
             a_u\longrightarrow u\longrightarrow v
                    \longrightarrow b_u:
\]
when \(u\) is inactive, (2) still permits the service transition
\(a_u\to v\) to exploit the same metric path.

## 3. Row-separation lemma

There is also a quantitative obstruction to hiding the source-service
baseline inside a cyclic column.

### Lemma

In every directed metric, for all sources \(u,u'\) and every target \(v\),
\[
                     d(u,u')\ge d(u,v)-d(u',v).
\tag{4}
\]

### Proof

The triangle inequality
\[
                     d(u,v)\le d(u,u')+d(u',v)
\]
rearranges to (4). \(\square\)

Thus every target-distance row
\[
                         u\mapsto(d(u,v))_v
\]
is one-sided Lipschitz with respect to the source metric.

For the usual column proposal, let a compatible move \(u'\to v\) cost
\(1\), while an incompatible next-column move \(u\to v\) costs at least
\(K+1\).  Equation (4) forces
\[
                              d(u,u')\ge K.
\tag{5}
\]
Consequently the extra active sources in a layer cannot be grouped at
negligible cost whenever their compatibility rows differ.

In the \(359/214\) gadget a generalist and either specialist disagree on one
target by the full edge/nonedge amount.  A high-penalty column metric
therefore pays an order-\(K\) source transition to move from that specialist
row to the generalist row.  This is a genuine service term, not an endpoint
artifact.

## 4. Why private repetition does not amortize this term

Take \(L\) private copies of an interface.  Even two sources having the same
role in different copies have distinct distance rows: source \(u_i\) is
compatible with its private target \(v_i\), while \(u_j\) is incompatible
with \(v_i\).  Applying (4) with \(v_i\) again gives
\[
                              d(u_j,u_i)\ge K.
\tag{6}
\]
Hence a service order through \(L\) distinct private rows has an
order-\(LK\) term.  This is the same order as the intended sum of local
deficiencies and cannot be put into the \(o(L)\) remainder of (1).

If one instead makes same-role rows identical by connecting them to every
copy's targets, the target capacities are pooled.  The construction is no
longer a product of the four-source gadget, and the finite Bellman
certificate does not apply.  In large symmetric pools the law of large
numbers in fact lets a causal policy choose the correct aggregate
allocation up to boundary fluctuations.

## 5. Consequence for tensor and positional proposals

A valid amplifier must establish all three of the following, rather than
only an interface matching identity:

1. inactive clients used as transit do not open the allegedly gated move;
2. every active source not used by an interface matching is served without
   an order-\(K\) row-transition charge;
3. private target capacities remain private even though the metric
   row-tour cost is \(o(LK)\).

Equations (2)--(6) show why a naive cyclic reset, optional-detour switch,
lexicographic repetition of private copies, or a column potential with
unit edges and \(K+1\) nonedges does not satisfy these requirements.

This does not rule out every algebraic or tensor construction.  It gives
the exact metric inequalities that such a construction must beat, and
prevents treating the finite matching gadget alone as a completed
stochastic-TSP lower bound.
