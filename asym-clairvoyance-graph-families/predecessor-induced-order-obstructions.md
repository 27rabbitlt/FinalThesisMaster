# Obstructions to forcing a bad middle-layer order

## 1. The target

For a three-layer vertex-client path metric
\[
                   L\longrightarrow M\longrightarrow R,
\]
one would like the first interface to reveal the active vertices of \(M\)
in an order that is hard for causal matching into \(R\).  This order must be
intrinsic: the policy chooses all calls, and the fixed graph and all
probabilities are known in advance.

Two natural attempts do not work:

1. use a balanced regular or incidence interface \(L\to M\) to permute the
   middle layer; or
2. put nested neighborhoods on the two interfaces in opposite orders.

The first fails by an arbitrary-order lemma.  The second fails even without
that lemma because laminar deadline neighborhoods admit an order-independent
greedy matching.

## 2. A saturating predecessor matching gives arbitrary order

### Lemma 1

Let \(G=(L,M;E)\) be a fixed bipartite interface.  Vertices of \(L\) are
permanent clients and vertices of \(M\) have arbitrary activations.  Suppose
\(G\) has a matching
\[
                    Q=\{\phi(m)m:m\in M\}
\]
saturating \(M\).  Then, for every prescribed order
\[
                    m_1,m_2,\ldots,m_{|M|},
\]
there is a legal causal policy with the following two properties:

1. the active middle clients occur in the restriction of this prescribed
   order; and
2. every active middle client earns an incoming cheap transition from its
   distinct predecessor in \(L\).

### Proof

For \(j=1,\ldots,|M|\), call the permanent client \(\phi(m_j)\) and then
call \(m_j\).  If \(m_j\) is inactive, the latter call does not move the
salesperson.  If it is active, the consecutive active calls
\(\phi(m_j),m_j\) use the edge of \(Q\), and hence earn a cheap transition.
The predecessors \(\phi(m_j)\) are distinct, so none was previously served.
The active \(m_j\)'s plainly occur in the prescribed order. \(\square\)

The policy may choose the prescribed order to be the best source order for
the second interface \(M\to R\).  Therefore a first interface satisfying
Lemma 1 supplies no chronology at all.

### Consequences

Lemma 1 applies immediately to:

* every balanced regular bipartite graph;
* projective-plane and generalized-polygon incidence bigraphs with equal
  sides;
* every balanced Ramanujan bigraph;
* every balanced lift of a base graph having a perfect matching; and
* every KVV/nested interface that contains the diagonal matching.

Randomly relabeling such an interface does not help.  The relabeling is part
of the fixed known graph, so the policy simply relabels \(Q\) and chooses
the desired middle order afterward.

The lemma is stronger than a high-probability statement: it is
realization-wise and does not use independence.

## 3. Why a potential-middle surplus is necessary

To obtain predecessor-induced order, the potential middle layer must not be
saturated by one fixed predecessor matching.  In particular, a template
with \(|L|\ge|M|\) and a robust Hall condition is immediately suspect: the
same expansion intended to give the posterior a large matching usually
supplies the fixed matching used in Lemma 1.

A credible interface must instead have many more potential middle clients
than predecessor clients, with only a random active subset being matchable.
Then the policy faces a genuine probe-commit allocation problem: inactive
middle probes consume no target, while the first active answer consumes the
current predecessor.

This surplus is not free in stochastic TSP.  Every active surplus middle
client contributes to

* the unavoidable \(N\)-term in the \(1/2\) metric;
* the posterior path-cover count if it cannot be matched at an interface;
  and
* the source population at the second interface, including clients that
  failed to receive an incoming transition.

Thus an online-matching deficit cannot be inserted as an isolated
coefficient.  The full active-client and path-cover accounting is required.

## 4. Opposite nested neighborhoods are also order-robust

Another proposal is to make incoming neighborhoods nested in one direction
and outgoing neighborhoods nested in the opposite direction.  In the
standard form, resources are slots \(1,\ldots,n\), and a type with deadline
\(d\) is adjacent to the prefix
\[
                         \{1,\ldots,d\}.
\]

Nested deadlines do not create online matching loss.  Whenever a type with
deadline \(d\) is processed, assign it the largest still-free slot at most
\(d\), if one exists.

### Lemma 2

For every arrival order and every realized set of deadline types, the
latest-free-slot rule produces a maximum-cardinality matching.

### Proof

Induct over arrivals while maintaining a maximum matching of the processed
types whose occupied slots are lexicographically largest.  For a new
deadline \(d\), if a free slot \(s\le d\) exists, taking the largest such
slot increases the matching size and preserves the invariant.  If no such
slot exists, all \(d\) compatible slots are occupied.  Hall's condition on
the prefix \(\{1,\ldots,d\}\) then certifies that the new type cannot
increase the maximum matching size.  The current matching remains maximum.
\(\square\)

Consequently reversing the deadline order between \(L\to M\) and
\(M\to R\) does not make one interface hard when the other is served well:
the outgoing interface is optimal under every inherited order.

The same argument covers suffix neighborhoods after reversing the slot
labels, and disjoint unions of laminar chains.

## 5. Remaining form of a possible construction

A genuine three-layer construction must use a first interface with no fixed
matching saturating all potential middle clients and a second interface
whose matching problem is genuinely nonlaminar.  It must prove a joint
tradeoff of the form
\[
 \mathbb E(M_{L,M}+M_{M,R})
      <\frac23\,\mathbb E(\nu_{L,M}+\nu_{M,R}),       \tag{5.1}
\]
for every adaptive policy, while including all active surplus middle
clients in both metric costs.

Balanced incidence graphs, alternating KVV suffixes containing a diagonal,
oppositely nested deadline graphs, and known random relabelings cannot
satisfy (5.1) for the reasons above.

The only plausible path-cover version left is a nonlaminar stochastic
matching interface with a large potential-middle surplus, coupled to an
incompatible second nonlaminar interface.  A proof must be a joint
probe-commit lower bound; an externally imposed arrival order is not enough.
