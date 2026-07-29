# An exogenous poset presentation above \(4/3\), and why it is not yet a
# stochastic-TSP instance

## Outcome

There is a clean finite random-presentation poset for which every causal
online chain partition uses
\[
                    \left(1+\frac1e-o(1)\right)n
\]
chains while the offline width is exactly \(n\).  Thus the abstract
online/offline chain-count ratio is strictly above \(4/3\).

The presentation is the classical random-suffix matching instance, but a
short potential proof is included below.  The final section proves an exact
encoding obstruction: forcing its presentation order by reachability
relations is incompatible with reusing a unit-capacity resource at several
stages.  Hence the construction is not, by itself, a valid fixed-metric
stochastic-TSP instance.

## 1. Random suffix poset

Let
\[
       L=\{u_1,\ldots,u_n\},\qquad
       R=\{v_1,\ldots,v_n\}.
\]
Choose a uniformly random permutation \(\pi\) of \(R\).  The height-two
poset \(P_\pi\) has
\[
       u_i<v
       \quad\Longleftrightarrow\quad
       v\in\{\pi(i),\pi(i+1),\ldots,\pi(n)\}.
\tag{1}
\]
Elements \(u_1,\ldots,u_n\) are presented in this chronological order.
When \(u_i\) is presented, its upper neighborhood in (1) is revealed.  An
online chain-partition policy may either pair \(u_i\) with an unused
comparable element of \(R\), or leave it as a singleton.  Its decision is
irrevocable.  At the end, every unused element of \(R\) is a singleton
chain.

The offline width is exactly \(n\).  Indeed,
\[
                  u_i<\pi(i)
\]
is a perfect matching, so the \(2n\) elements have a chain cover of size
\(n\); the antichain \(L\) supplies the matching lower bound.

If an online policy makes \(M\) pairs, its chain count is
\[
                         K=2n-M.
\tag{2}
\]

## 2. A policy-independent matching upper bound

At stage \(i\), put
\[
       s_i:=n-i+1.
\]
The currently revealed neighborhood is a uniformly random \(s_i\)-element
suffix conditional on the complete history.  Let \(X_i\) be the number of
unused resources in that suffix immediately before processing \(u_i\).
The policy succeeds at stage \(i\) exactly when \(X_i>0\).  Conditional on
success, its chosen resource is one of those \(X_i\) unused resources.
The next deleted permutation element \(\pi(i)\) is uniform in the current
suffix, independently of which unused resource the policy chooses.

We may assume that the policy always matches when \(X_i>0\).  Forcing an
available match cannot reduce final cardinality: if the chosen resource
would later have been used, exchange that later match for the current one;
otherwise the forced match only increases cardinality.  Hence every
non-greedy policy is dominated by a greedy policy, and it is enough to upper
bound the latter.

After the chosen resource is consumed, \(X_i-1\) unused resources remain.
Deleting \(\pi(i)\) removes one of them with probability
\[
                         \frac{X_i-1}{s_i}.
\]
Therefore, conditional on the history and \(X_i>0\),
\[
 \mathbb E[X_{i+1}\mid X_i]
 =X_i-1-\frac{X_i-1}{s_i}.
\tag{3}
\]
Since \(s_{i+1}=s_i-1\), (3) gives the exact normalized drift
\[
 \mathbb E\!\left[
       \frac{X_{i+1}}{s_{i+1}}\ \middle|\ X_i
 \right]
 =
       \frac{X_i}{s_i}-\frac1{s_i}.
\tag{4}
\]
If \(X_i=0\), all later suffix resources have already been consumed, so
the normalized process remains zero.

Let
\[
       a_i:=\Pr[X_i>0].
\]
Taking expectations in (4), including the absorbing zero state, and
summing over the stages yields
\[
       \sum_{i=1}^{n}\frac{a_i}{n-i+1}\leq1.
\tag{5}
\]
Moreover \(a_i\) is nonincreasing, because zero is absorbing, and
\[
                         \mathbb E M=\sum_i a_i.
\tag{6}
\]

Among nonincreasing sequences \(1\ge a_1\ge\cdots\ge a_n\ge0\) satisfying
(5), the unweighted sum is maximized by placing all mass on the earliest
indices, whose weights \(1/(n-i+1)\) are smallest.  If \(k\) is the largest
integer for which
\[
       \sum_{i=1}^{k}\frac1{n-i+1}
       =H_n-H_{n-k}\leq1,
\tag{7}
\]
then (5)--(6) imply
\[
                         \mathbb E M\leq k+1.
\tag{8}
\]
The harmonic-number estimate
\[
       H_n-H_{n-k}
       =\log\frac{n}{n-k}+O\!\left(\frac1{n-k}\right)
\]
shows from (7) that
\[
                         k=(1-e^{-1})n+O(1).
\tag{9}
\]
Equations (2), (8), and (9) prove, for every deterministic causal policy,
\[
       \mathbb E K
       \geq
       \left(1+\frac1e\right)n-O(1).
\tag{10}
\]
Conditioning on a private random seed proves the same statement for every
randomized policy.

Since \(1+1/e>4/3\), this is a genuine abstract chain-partition separation,
not a borderline \(4/3\) example.

## 3. The metric if the poset were fixed

For a fixed \(\pi\), define
\[
 d_\pi(x,y)=
 \begin{cases}
 0,&x\leq_{P_\pi}y,\\
 1,&\text{otherwise},
 \end{cases}
\tag{11}
\]
and give a depot \(r\) unit distance to and from every client.  Transitivity
of the poset proves the directed triangle inequality.  An ordering whose
active subsequence has \(k\) maximal increasing runs has closed-depot cost
\[
                         k+1.
\tag{12}
\]
Taking \(N\) disjoint copies would make the additive depot term negligible.

However, (11) is a different metric for every \(\pi\).  A stochastic-TSP
instance requires one deterministic known metric; only client activations
are random.  Thus randomizing \(\pi\) externally is not a legal
clairvoyance-gap construction.

## 4. Why chronological reachability cannot encode the presentation

One might put every possible type into one fixed poset and add permanent
stage gates
\[
                         g_1<g_2<\cdots<g_{n+1}
\]
to force the types to be processed chronologically.  Suppose a stage-\(i\)
type \(x_i\) can be paired through a shared resource \(v\) before the next
stage.  Reachability then requires
\[
                         x_i<v<g_{i+1}.
\tag{13}
\]
If the same resource is eligible for a later stage \(j>i\), it must also
satisfy
\[
                         x_j<v.
\tag{14}
\]
Chronology gives \(g_{i+1}\leq x_j\), and (13)--(14) yield
\[
                         v<g_{i+1}\leq x_j<v,
\]
a poset cycle.  Therefore a resource insertable at zero cost is eligible
at only one chronological cut.  Stage-specific copies avoid the cycle but
remove unit capacity and hence remove the matching lower bound.

There is a second, simpler failure if one only imposes
\[
                         x_i<x_j\qquad(i<j)
\]
without putting the resource between stages.  All presented types then
belong to one increasing chain.  Their chain-cover cost no longer equals
\(2n-M\), so the matching separation disappears.

Finally, encoding the random suffix by choosing exactly one active type
from a type bank uses one-hot correlations.  Independent activation bits
do not provide that distribution.  A first-active-marker encoding leaves
additional active markers which are themselves clients, and it still does
not solve the chronological shared-resource cycle above.

## 5. Positive chronological arcs and unit resets do not restore capacity

Replace zero reachability by a generating digraph with stage gates \(g_i\),
short forward arcs, and unit backward/reset arcs.  To realize a match at
stage \(i\), the intended cheap trace has the form
\[
                         x_i\longrightarrow v\longrightarrow g_{i+1}.
\tag{15}
\]
This does not implement unit capacity.  After shortest-path closure, the
trace in (15) remains a metric path at every later time, whether resource
client \(v\) is uncalled, already served, inactive, or used only as transit.
Serving a client removes a service obligation; it does not delete the
vertex or its arcs.  Thus one resource can be reused as a shortcut at all
eligible stages, and the cost is no longer \(2n-M\) for a matching \(M\).

Making \(v\) terminal and charging a unit reset after it avoids the immediate
reuse only syntactically.  The same unit reset is available after every
terminal use, so it has no memory of which stage consumed \(v\).  Giving
each stage a separate terminal copy again removes matching capacity.

Probing ahead is a separate failure.  A future type can be called before its
stage gate is served.  If it is inactive there is no movement and its state
is learned for free.  If it is active, the metric move may expand along the
short chronological path through unserved gates; transit does not serve
those gates.  The policy can then pay one reset and resume earlier work.
Because one matching mistake is itself worth only one chain/reset, a unit
backward arc cannot be declared prohibitively expensive.  Any positive-arc
reduction must analyze these ahead-probe-and-reset policies explicitly;
the exogenous lower bound assumes they are forbidden and therefore does not
apply.

If a valid fixed poset were obtained, depot closure would be harmless:
(12) adds exactly one to the chain count, and disjoint repetition would
amortize that constant.  The obstruction occurs before depot closure—at
chronology and resource capacity—not at the final root arcs.

## 6. Sharper surviving class

The exogenous calculation identifies the amount of information loss needed:
a \(1-1/e\) causal matching rate is already sufficient.  But a valid
reachability construction cannot import that loss through an external
arrival order.

The remaining poset target must therefore have all three properties:

1. one fixed deterministic poset and independent product activations;
2. a gap against a policy allowed to choose every next probe adaptively;
3. capacity encoded by irreversible client order without placing a reusable
   resource between two chronological cuts.

In particular, height-two matching plus stage gates is completely excluded.
Any surviving construction must use a genuinely non-series-parallel
height-at-least-three dependency in which allocation is represented by the
order of several stochastic clients, rather than by a shared terminal
resource vertex.
