# A metric reduction to free-order stochastic weighted matching

## Outcome

There is an exact directed-metric reduction from the clairvoyance problem to
the following free-order stochastic matching problem.

* \(U\) is a set of independently active online vertices.
* \(V\) is a set of permanent unit-capacity targets.
* The policy chooses which \(u\in U\) to reveal next.
* If \(u\) is active, the policy may irrevocably assign it to an unused
  \(v\in V\), earning \(w_{uv}\), or discard it.
* The prophet sees the active subset and takes a maximum-weight matching.

For every fixed realization \(A\subseteq U\), the posterior TSP cost in the
metric below is exactly
\[
              |A|+|V|+4+\varepsilon
              -\operatorname {MWM}(G[A,V]),
\tag{0.1}
\]
where the harmless constant includes two dummy clients.  For every causal
TSP policy, its cost is at least the same baseline minus the weight of the
free-order matching formed by its consecutive \(U\to V\) service
transitions.

Consequently, an explicit product-activation matching family satisfying
\[
\begin{aligned}
 \mathbb E\operatorname {MWM}(G[A,V])&=(1-o(1))n,\\
 \sup_\pi\mathbb E W_\pi&\le(c+o(1))n
       \quad\text{for some }c<2/3,\\
 |V|+\mathbb E|A|&=(2+o(1))n
\end{aligned}
\tag{0.2}
\]
would immediately give a genuine stochastic-TSP gap
\[
                       2-c-o(1)>\frac43.
\tag{0.3}
\]

The metric and the arbitrary-tour audit are complete.  The missing object
is (0.2).  Fixed-arrival and random-arrival stochastic-matching lower bounds
do not establish it, because the TSP policy chooses the revelation order
adaptively.  Unweighted graphs cannot work: choosing a uniform random source
order and running RANKING gives at least \(0.696\) of the posterior matching
for every realized active set.  A valid construction must therefore be
genuinely edge-weighted and hard in the free-order model.

## 1. The free-order matching benchmark

Fix a bipartite graph \(G=(U,V,E)\) and weights
\[
                         0\leq w_{uv}\leq1-\varepsilon,
 \qquad 0<\varepsilon<1.
\tag{1.1}
\]
Every \(u\in U\) is independently active, with an arbitrary specified
probability.  Every target \(v\in V\) is present.

A free-order policy may query the vertices of \(U\) in an order depending
on all earlier outcomes.  An inactive query only reveals that \(u\) is
absent.  When a query finds \(u\) active, the policy may match \(u\) to one
currently unused target \(v\), or discard \(u\).  Let \(W_\pi(A)\) be the
weight obtained by policy \(\pi\), and put
\[
             W^*(A):=\operatorname {MWM}(G[A,V]).
\tag{1.2}
\]

This benchmark grants the causal policy exactly the information available
in the intended TSP trace: the bit of \(u\) is known before the outgoing
target is selected.  In particular, this is not query-commit matching in
which choosing the target precedes the successful probe.

## 2. Metric construction

Add a permanent dummy source \(u_\circ\) and a permanent dummy target
\(v_\circ\), with zero matching weights.  Write
\[
             \widehat U=U\cup\{u_\circ\},\qquad
             \widehat V=V\cup\{v_\circ\}.
\]
The two dummies ensure that every realization has an unmatched vertex on
each side; they will remove an otherwise irrelevant endpoint loss.

On distinct clients define
\[
d(x,y)=
\begin{cases}
 \varepsilon,
       &x\in\widehat V,\ y\in\widehat U,\\
 1,
       &x,y\in\widehat U\text{ or }x,y\in\widehat V,\\
 2-(w_{xy}+\varepsilon),
       &x\in U,\ y\in V,\\
 2,
       &x\in\widehat U,\ y\in\widehat V
         \text{ and the preceding weighted case does not apply}.
\end{cases}
\tag{2.1}
\]
Put \(d(x,x)=0\).  Add a depot \(r\) with
\[
                         d(r,x)=d(x,r)=2.
\tag{2.2}
\]

### Lemma 1

Equations (2.1)--(2.2) define a strictly positive directed metric.

### Proof

The only distances below one are the arcs
\(\widehat V\to\widehat U\), of length \(\varepsilon\).

* A same-side distance is one.  A two-leg path from \(V\) back to \(V\)
  through \(U\) costs at least
  \(\varepsilon+1\); a path from \(U\) back to \(U\) through \(V\) costs
  at least \(1+\varepsilon\).
* A distance from \(U\) to \(V\) lies in \([1,2]\).  A two-leg path using
  an intermediate client costs at least two: through the same side one
  leg costs one and the other at least one, while through the opposite
  side the same conclusion holds.  Thus it cannot improve a direct
  distance of at most two.
* A distance from \(V\) to \(U\) is already the global positive minimum
  \(\varepsilon\).
* Every depot detour costs at least \(2+\varepsilon\), and a two-leg path
  used to compare a depot arc of length two also costs at least
  \(2+\varepsilon\).

These cases prove all directed triangle inequalities. \(\square\)

Unlike the \([B/2,B]\) construction, this metric uses very short
\(V\to U\) arcs.  The same-side unit distances are essential: they prevent
the short arcs from collapsing the weighted \(U\to V\) distances under
triangle inequality.

## 3. Exact savings identity for an arbitrary tour

Fix a realization \(A\subseteq U\).  Let
\[
             a=|A|+1,\qquad m=|V|+1,\qquad N=a+m
\tag{3.1}
\]
be the numbers of present source-side clients, target-side clients, and
total clients, including the dummies.

Consider any service order of these \(N\) clients.  Let

* \(b\) be the number of consecutive transitions
  \(\widehat V\to\widehat U\);
* \(c\) be the number of consecutive transitions
  \(\widehat U\to\widehat V\); and
* \(W\) be the sum of \(w_{uv}\) over the weighted \(U\to V\)
  transitions in the order.

Relative to an all-two baseline, a same-side transition saves one, a
\(V\to U\) transition saves \(2-\varepsilon\), and a weighted
\(U\to V\) transition saves \(w_{uv}+\varepsilon\).  Hence the total
internal saving is exactly
\[
\begin{aligned}
S_{\rm ord}
 &=N-1-b-c+(2-\varepsilon)b
       +\sum_{U\to V}(w_{uv}+\varepsilon)\\
 &=N-1+(1-\varepsilon)(b-c)+W.
\end{aligned}
\tag{3.2}
\]

The side labels in a binary word imply
\[
                         b-c\in\{-1,0,1\}.
\tag{3.3}
\]
Moreover, the weighted \(U\to V\) adjacencies form a matching: no client
has two predecessors or two successors in a service order.  Therefore
\[
                         W\leq W^*(A)
\tag{3.4}
\]
and (3.2) gives the universal upper bound
\[
                         S_{\rm ord}
                   \leq N-\varepsilon+W^*(A).
\tag{3.5}
\]

This proves the lower bound
\[
             \operatorname {OPT}_{\rm post}(A)
             \geq N+2+\varepsilon-W^*(A),
\tag{3.6}
\]
because the two depot arcs give all-two baseline \(2(N+1)\).

### Lemma 2

Equality holds in (3.6).

### Proof

Take a maximum-weight matching
\[
                     M=\{u_1v_1,\ldots,u_kv_k\}.
\]
List all unmatched target-side clients first, with the dummy target
\(v_\circ\) last in that initial block.  From \(v_\circ\), continue with
\[
 u_1,v_1,\ u_2,v_2,\ldots,u_k,v_k,\ u_\circ,
\tag{3.7}
\]
and then list all remaining unmatched source-side clients.  The dummy
vertices guarantee that the endpoints displayed in (3.7) are available.

The full order starts on the \(V\)-side and ends on the \(U\)-side, so
\(b-c=1\).  Its weighted \(U\to V\) transitions are exactly \(M\).
Equation (3.2) therefore gives saving
\[
                         N-\varepsilon+W^*(A),
\]
which meets (3.5). \(\square\)

Combining (3.6) with Lemma 2 proves the advertised exact formula
\[
\boxed{
 \operatorname {OPT}_{\rm post}(A)
       =N+2+\varepsilon-W^*(A).
}
\tag{3.8}
\]

## 4. Every causal tour projects to a free-order matching

Consider an arbitrary adaptive TSP policy, including randomized policies
and policies whose service order is not alternating.

Inactive calls do not create service transitions.  Shortcut every physical
walk between consecutive service events to the direct metric arc.  Triangle
inequality makes the shortcut no more expensive, so a lower bound for the
shortcut service order is also a lower bound for the original execution.

In the shortcut order, collect every weighted consecutive transition
\[
                           u\longrightarrow v,
 \qquad u\in U,\ v\in V.
\tag{4.1}
\]
These arcs form a matching.  When the transition (4.1) is chosen, the
policy has already queried \(u\) and learned that it is active; \(v\) is an
uncalled permanent client and hence an unused target.  Its choice may
depend on all preceding activation outcomes but not on future ones.
Therefore these arcs are a legal free-order stochastic matching policy.

Let \(W_\pi(A)\) be their total weight.  Repeating (3.2)--(3.5), now without
replacing \(W_\pi\) by the prophet value, gives realization-wise
\[
                       C_\pi(A)
                  \geq N+2+\varepsilon-W_\pi(A).
\tag{4.2}
\]
Randomization causes no issue: condition on the private seed.

Conversely, rejection in the matching policy is physically harmless.  From
an active source \(u\), the TSP policy can call further sources.  Inactive
calls do not move it; the first further active source creates a same-side
transition and discards the previous \(u\).  When the matching policy
accepts a source, it calls the selected unused permanent target.  Remaining
same-side clients can be served consecutively at the end.  Thus the
free-order benchmark captures the only policy-dependent term in (4.2);
nonalternating tours do not expose an additional source of weighted
savings.

## 5. Gap transfer

Put
\[
 P:=\mathbb E\operatorname {OPT}_{\rm post}(A),\qquad
 Q:=\sup_\pi\mathbb E W_\pi(A),\qquad
 S:=\mathbb E W^*(A),
\]
and
\[
                       C:=\mathbb E N+2+\varepsilon.
\tag{5.1}
\]
Equations (3.8) and (4.2) give
\[
                         P=C-S,\qquad
        \operatorname {OPT}_{\rm adapt}\geq C-Q.
\tag{5.2}
\]
Hence if \(Q\leq cS\),
\[
 \frac{\operatorname {OPT}_{\rm adapt}}
      {\operatorname {OPT}_{\rm post}}
 \geq
                         \frac{C-cS}{C-S}.
\tag{5.3}
\]

For a balanced family with
\[
 |V|=(1+o(1))n,\qquad
 \mathbb E|A|=(1+o(1))n,\qquad
 S=(1-o(1))n,
\tag{5.4}
\]
take \(\varepsilon=o(1)\).  Then \(S/C\to1/2\), and (5.3) becomes
\[
                         2-c-o(1).
\tag{5.5}
\]
Thus **any** free-order weighted-matching upper bound \(c<2/3\) with the
density (5.4) completes the requested construction.

## 6. What is and is not already ruled out

### Unweighted graphs cannot supply (0.2)

For any fixed realization \(A\), choose a uniformly random order of the
active-source candidates by first choosing a uniform order of all \(U\),
and independently rank \(V\).  Whenever a queried source is active, match
it to its minimum-ranked unused neighbor.

Conditional on \(A\), this is ordinary RANKING with the online side in
uniform random order.  The Mahdian--Yan random-arrival guarantee gives
\[
                         \mathbb E W_\pi(A)
                         \geq0.696\,W^*(A)
\tag{6.1}
\]
for unit weights.  Averaging over \(A\) preserves (6.1).  Therefore an
unweighted product-activation family cannot have \(c<2/3\).

### A single target cannot supply (0.2)

With one target, the problem is the order-selection prophet inequality for
independent Bernoulli values.  Known free-order prophet algorithms guarantee
strictly more than \(2/3\) (currently at least \(0.7258\) for general
independent values).  Degree-one sources decompose into copies of the same
single-target problem and are likewise insufficient.

### Fixed-order hardness does not transfer

The following results concern different information models and cannot be
inserted into (5.3).

* The \(1/2\) vertex-arrival prophet bound assumes a prescribed adversarial
  order.
* Random-order and known-i.i.d. stochastic-matching hard instances do not
  let the algorithm select the next vertex type after every outcome.
* Fully-online matching hardness relies on exogenous arrivals and deadlines.
* Edge-arrival prophet lower bounds reveal random edge values, whereas here
  one source activation is learned before the policy chooses among all of
  that source's target edges.

The needed theorem is specifically an upper bound against **adaptive
free-order vertex revelation**.

## 7. Remaining target

The metric side of the reduction is now closed.  The research target is a
weighted bipartite family with independent Bernoulli source activations
such that

1. the expected active count and number of permanent targets are both
   \(n+o(n)\);
2. the prophet has a weight-\((1-o(1))n\) matching; and
3. every policy that adaptively chooses the next source to reveal obtains
   at most \((2/3-\delta)n\) for some fixed \(\delta>0\).

The instance must use overlapping target choices and genuinely different
edge weights.  Private-versus-flexible stars are not enough: query the
private/high-specific sources first and use flexible sources only to fill
the revealed holes.  A promising family would need cyclic opportunity
costs with no source class that can safely be exposed first, together with
a policy-independent potential upper bound.

No explicit family meeting these three conditions is presently established
in this audit.
