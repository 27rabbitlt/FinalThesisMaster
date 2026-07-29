# A three-cycle of KVV interfaces cannot fall below \(2/3\)

## 1. Cyclic layered setting

Let \(V_0,V_1,V_2\) be three client layers, with indices modulo three.
Cheap directed transitions go only across
\[
          V_0\longrightarrow V_1,\qquad
          V_1\longrightarrow V_2,\qquad
          V_2\longrightarrow V_0.
\]
Distances are one on these arcs and two on all other ordered pairs of
distinct clients; the depot has distance one in both directions.  This is a
directed metric.

For a realization \(A\), let
\[
 \nu_i(A)=
 \nu\bigl(G_i[A\cap V_i,A\cap V_{i+1}]\bigr)
\]
be the maximum matching size at interface \(i\).

Every set of cheap transitions in a linear service order restricts to a
matching at each interface.  (Unlike the acyclic layered case, the union of
three independently chosen matchings can contain directed cycles and need
not itself be realizable by one linear order.)  The posterior number
\(T^*(A)\) of cheap transitions therefore always satisfies
\[
                 T^*(A)\le \nu_0(A)+\nu_1(A)+\nu_2(A).
\tag{1}
\]

## 2. Order-robust interfaces

Call a bipartite interface \(G=(U,W;E)\) **causally
order-robust** if, for every realized active induced graph and every source
order, a source-first policy can produce a maximum matching without calling
an unmatched target prematurely.

Every KVV prefix/suffix interface is order-robust.  In the prefix form,
slots are \(1,\ldots,n\), a source of type \(d\) is adjacent to
\(\{1,\ldots,d\}\), and the policy assigns the largest free active slot at
most \(d\).  To implement this physically, while the active source is
current, query its still-uncalled slots in decreasing order until the first
active one is found.

For completeness, this greedy matching is maximum after every source
arrival.  If a free compatible slot exists, taking the largest one preserves
a lexicographically largest maximum matching and raises its cardinality by
one.  If no compatible slot is free, all slots in the relevant prefix are
occupied; the prefix Hall constraint certifies that the new source cannot
increase the maximum matching size.  The suffix case follows by reversing
the slot labels.  The proof does not depend on the source order.

Disjoint unions of nested KVV chains have the same property.

## 3. Exact causal service after cutting one interface

Fix a cut \(c\in\{0,1,2\}\), and ignore cheap transitions across
\(V_c\to V_{c+1}\).  The remaining two interfaces form a linear three-level
network
\[
           V_{c+1}\longrightarrow V_{c+2}
                    \longrightarrow V_c.
\]

Use the least-level path scheduler:

1. start uncalled clients in \(V_{c+1}\) in any fixed order;
2. whenever an active source is current, run the order-robust matching rule
   into the next level and immediately continue from the matched target;
3. after a path stops, return to the least of the three linearized levels
   that still has an uncalled client.

An inactive probe causes no movement.  The least-level restart rule implies
that no target is called as a new path start while an earlier-level source
remains uncalled.  Consequently the first remaining interface sees all of
its sources in some order and its rule produces a maximum matching.  The
matched targets induce an arbitrary source order at the second interface,
but order-robustness gives a maximum matching there as well.  Thus,
realization-wise,
\[
             M_c(A)=\nu_{c+1}(A)+\nu_{c+2}(A).
\tag{2}
\]

This is a legal stochastic-TSP policy.  It does not assume an external
arrival order, and it permits all activations in all three layers to be
independent and nonidentical.

## 4. The \(2/3\) guarantee

Choose the cut in advance to minimize expected lost matching value:
\[
             c\in\arg\min_i\mathbb E\nu_i(A).
\]
Equations (1)--(2) give
\[
\begin{aligned}
 \mathbb E M_c
 &=\sum_i\mathbb E\nu_i-\mathbb E\nu_c\\
 &\ge\frac23\sum_i\mathbb E\nu_i\\
 &\ge\frac23\,\mathbb E T^*.
\end{aligned}
\tag{3}
\]

If private randomization is disallowed, none is needed.

Therefore a cyclic composition of three KVV/nested interfaces cannot make
every causal policy capture strictly less than \(2/3\) of the posterior
cheap transitions.  In the favorable case
\(T^*=\nu_0+\nu_1+\nu_2\), the bound is exactly the barrier required to
exclude a strict closed-depot ratio above \(4/3\).

## 5. Generalization and remaining target

The same cut argument on an \(h\)-cycle of order-robust interfaces captures
at least
\[
                       1-\frac1h
\]
of the sum of the interface optima.  Hence increasing the number of cyclic
KVV layers only makes the obstruction stronger.

A positive cyclic construction must use genuinely nonlaminar interfaces
whose realized matching cannot be achieved for every inherited order.
Moreover, its proof must survive the policy's option of choosing the
least-valuable interface as a cut and sacrificing all transitions there.

