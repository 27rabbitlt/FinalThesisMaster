# Weighted height-two metrics: an exact committed-assignment reduction

## 1. Construction

Let \(G=(U,V,E)\) be a bipartite graph with edge weights
\[
                         0\le w_{uv}<1.
\]
Every \(u\in U\) is a permanent client.  Every \(v\in V\) is independently
active with probability \(p_v\).  Add a depot \(r\), and define
\[
\begin{aligned}
 d(r,z)=d(z,r)&=\frac12 &&(z\in U\cup V),\\
 d(u,v)&=1-w_{uv} &&(uv\in E),\\
 d(x,y)&=1 &&\text{for every other ordered pair }x\ne y.
\end{aligned}                                           \tag{1}
\]
Weights equal to one may be obtained as a limit, or replaced by
\(1-\varepsilon\).

This is a directed metric.  The only distances below one go from \(U\) to
\(V\).  Consequently two subunit arcs cannot be consecutive.  Every
two-arc client-to-client walk has length at least one, unless its endpoints
already form one of the discounted pairs in (1); every walk through the
depot has length one.  Thus all triangle inequalities hold, and shortest
path closure leaves (1) unchanged.

## 2. Exact posterior identity

Fix an active target set \(A\subseteq V\), and put
\[
                         N(A)=|U|+|A|.
\]
A closed tour through these clients has baseline cost
\[
       \frac12+(N(A)-1)\cdot1+\frac12=N(A).
\]
Its only possible discounts are consecutive calls \(u,v\) with
\(u\in U\), \(v\in A\), and \(uv\in E\).  Such consecutive pairs form a
matching: each source and each target occurs only once in the service
order.  Conversely, every matching can be extended to a client ordering,
placing each matched target immediately after its source.  Hence, if
\(W^*(A)\) is the maximum weight of a matching in \(G[U,A]\),
\[
              \boxed{\operatorname {OPT}_{\rm post}(A)
                     =N(A)-W^*(A).}                     \tag{2}
\]

## 3. Exact causal interpretation

Consider an adaptive TSP policy.  Every discount it earns occurs when it is
located at a just-called permanent source \(u\), calls an uncalled target
\(v\), finds \(v\) active, and moves immediately along \(u\to v\).  The
source and target cannot support another discounted transition.  Therefore
the policy's total discount is the weight of a causal committed matching.

Conversely, every source-first committed-assignment policy is physically
executable as a TSP policy.  When processing \(u\), call \(u\), then query
uncalled target candidates in the prescribed order.  Inactive answers do
not move the salesperson.  The first active answer commits \(u\) to that
target.  After all sources have been processed, call all remaining targets;
those calls use only baseline transitions.  Thus, if \(W_{\rm com}\) is the
largest expected weight attainable by any such causal policy,
\[
 \boxed{\operatorname {OPT}_{\rm adapt}
       =\mathbb E N(A)-W_{\rm com}.}                     \tag{3}
\]
This equality already allows arbitrary source order, adaptive target order,
early calls, interleaving, and private randomization.

Equations (2)--(3) are an exact reduction, not merely upper and lower
bounds.

## 4. The numerical target

Write
\[
             M:=\mathbb E N(A),\qquad
             W:=\mathbb E W^*(A),\qquad
             c:=\frac{W_{\rm com}}W.
\]
Then
\[
 \frac{\operatorname {OPT}_{\rm adapt}}
      {\operatorname {OPT}_{\rm post}}
 =\frac{M-cW}{M-W}.                                     \tag{4}
\]
It exceeds \(4/3\) exactly when
\[
                    \frac WM>\frac1{4-3c}.              \tag{5}
\]
Since \(W\le\min\{|U|,\mathbb E|A|\}\), one always has \(W/M\le1/2\).
Thus (5) requires \(c<2/3\), together with near saturation
\[
             \mathbb E|A|\sim|U|,\qquad W\sim|U|.
\]
In the ideal saturated regime \(M\sim2|U|\), \(W\sim|U|\), equation (4)
becomes
\[
                         2-c.                           \tag{6}
\]

For unweighted graphs this route is closed: the physically executable
source-first random-order `RANKING` policy obtains at least \(0.696\) of
\(W^*(A)\), realization by realization.  The weighted case is different;
ordinary unweighted `RANKING` does not preserve maximum matching weight.
The exact remaining target is therefore:

> Find a fixed weighted bipartite graph and independent target
> activations for which \(\mathbb EW^*(A)=(1-o(1))|U|\), while every
> source-first committed policy has expected weight
> \((2/3-\Omega(1))|U|\).

Such an instance would immediately give a genuine directed-metric
clairvoyance gap above \(4/3\) through (1)--(6), with no further geometric
or interleaving lemma required.

## 5. What does not transfer automatically

The \(1-1/e\) algorithm for weighted edge-independent query--commit
matching does not by itself give either side of the required separation.
Here all edges incident with one target share the same activation bit.
Likewise, hardness for exogenous online vertex arrivals does not apply:
the TSP policy chooses its source order and may query any future target.

The value of the reduction is that these are now the only unresolved
issues.  Capacity, metric closure, inactive lookahead, and arbitrary tour
interleaving are already represented exactly by the committed-assignment
model.
