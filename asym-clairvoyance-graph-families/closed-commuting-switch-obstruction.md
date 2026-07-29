# Closed commuting switches have no clairvoyance gap

## Purpose

A natural attempt to beat \(4/3\) at a single depot is to put two or more
one-bit route switches on the same permanent tour.  The permanent backbone
and the depot return are then paid only once, while one hopes that the
causal losses of the switches add.  This note gives a complete obstruction:
if the switches commute in the usual path-substitution sense, the
clairvoyance gap is exactly one.

The statement permits arbitrary directed lengths, arbitrarily long switch
paths, and any independent activation probabilities.  It is not a
small-instance calculation.

## 1. Master-tour lemma

Let \(r\) be the depot, \(D\) a set of permanent clients, and \(S\) a set
of independently active stochastic clients.  For \(A\subseteq S\), write
\(C(A)\) for the optimum depot-tour cost on
\(\{r\}\cup D\cup A\).

### Lemma 1

Suppose there is one cyclic order
\[
        T=(r,v_1,\ldots,v_m,r)
\tag{1.1}
\]
on \(\{r\}\cup D\cup S\) such that, for every \(A\subseteq S\), the
shortcut \(T|_{D\cup A}\) has cost \(C(A)\).  Then
\[
       \operatorname {OPT}_{\rm adapt}
       =\operatorname {OPT}_{\rm post}.
\tag{1.2}
\]

### Proof

Use the causal policy that calls every client in the order (1.1).
An inactive stochastic call causes no movement, so on realization \(A\)
the induced tour is exactly \(T|_{D\cup A}\).  Its cost is \(C(A)\).
Thus the policy attains the realization-wise posterior optimum
simultaneously for every activation set.  Averaging proves (1.2).
\(\square\)

Notice that expanded shortest paths are irrelevant to the proof.  They may
pass through inactive or uncalled switch vertices; the metric shortcut
inequality is already built into \(T|_{D\cup A}\).

## 2. Edge-disjoint switch substitutions

Here is the standard shared-backbone construction in full generality.
Start with a directed closed walk
\[
       W_0=(r=u_0,u_1,\ldots,u_k=r)
\tag{2.1}
\]
that serves all permanent clients.  Repeated permanent transit vertices are
allowed.  Choose pairwise disjoint occurrences of arcs
\[
       e_i=(u_{\alpha_i},u_{\alpha_i+1}),
       \qquad i=1,\ldots,h.
\]
For switch \(i\), replace the occurrence \(e_i\) by a directed path
\[
 u_{\alpha_i}=z_{i,0},z_{i,1},\ldots,z_{i,t_i},
 z_{i,t_i+1}=u_{\alpha_i+1}.
\tag{2.2}
\]
Some or all internal \(z_{i,j}\)'s may be stochastic clients.  Different
switch paths have disjoint stochastic-client sets.  Let \(T\) be the cyclic
client order obtained by making all substitutions (2.2).

Assume the proposed posterior proof is additive in the switches: for every
active set \(A\), deleting inactive internal clients from \(T\) gives an
optimal tour.  Equivalently, if \(A_i\) is the active subset in switch \(i\),
the proposed optimum has the form
\[
 C(A)=C_0+\sum_{i=1}^h
 \left(
   \operatorname {cost}\bigl(
     z_{i,0}, A_i\text{ in path order},z_{i,t_i+1}
   \bigr)
   -d(z_{i,0},z_{i,t_i+1})
 \right).
\tag{2.3}
\]

### Theorem 2

Every construction satisfying (2.1)--(2.3) has
\[
       \operatorname {OPT}_{\rm adapt}
       =\operatorname {OPT}_{\rm post}.
\tag{2.4}
\]

### Proof

The all-switches order \(T\) is a master tour in the sense of Lemma 1.
Indeed, shortcutting precisely the inactive switch clients yields the tour
whose cost is displayed in (2.3), and that tour is assumed posterior
optimal.  Lemma 1 proves (2.4). \(\square\)

The conclusion remains true when:

* a switch path has arbitrary length or directed asymmetry;
* a switch is a whole deterministic network with a fixed internal master
  order;
* switch intervals are nested, provided the substitutions admit one common
  recursively expanded order; or
* the base walk and substituted paths are later replaced by their directed
  shortest-path metric.

Thus neither making the graph large nor making each local switch expensive
changes the conclusion.

## 3. Boolean deletion-square formulation

For two stochastic clients \(x,y\), let \(T_{xy}\) be an optimal all-active
tour.  If
\[
\begin{aligned}
 \operatorname {cost}(T_{xy}-y)&=C(\{x\}),\\
 \operatorname {cost}(T_{xy}-x)&=C(\{y\}),\\
 \operatorname {cost}(T_{xy}-\{x,y\})&=C(\varnothing),
\end{aligned}
\tag{3.1}
\]
then \(T_{xy}\) is a master tour and the gap is one.  Consequently every
genuine two-selector closed chamber must violate at least one equality in
(3.1).  In particular, its four optimal state tours cannot form a commuting
Boolean square under deletion.

More generally, for \(h\) selectors a strict gap requires that every
all-active optimum have a subrealization \(A\subsetneq S\) for which its
shortcut is strictly suboptimal.  A construction based on independent
commuting \(2\)-opt moves, edge substitutions, or algebraic voltage switches
therefore cannot work: these mechanisms produce exactly the forbidden master
tour.

## 4. Why this blocks the shared-base amplification

Suppose a one-bit chamber has posterior costs \(C_0,C_1\), and several copies
are put on a common depot tour so that state costs are claimed to add only
their local insertion increments:
\[
       C(A)=C_{\rm base}+\sum_{i\in A}\Delta_i.
\tag{4.1}
\]
If the local insertions occupy compatible positions of a single tour, the
all-active tour shortcuts to cost (4.1) in every state.  Hence the proposed
adaptive losses do **not** add; one fixed call order realizes all local
posterior choices.

This includes the particularly attractive target
\[
 C(\varnothing)=W,\quad
 C(\{x\})=W+s_x,\quad
 C(\{y\})=W+s_y,\quad
 C(\{x,y\})=W+s_x+s_y,
\tag{4.2}
\]
whenever (4.2) is certified by two compatible substitutions of the same
backbone.  Sharing the permanent baseline is harmless only if the two
switches are genuinely noncommuting: the joint optimal tour must shortcut
badly in at least one singleton or empty state.

## 5. Consequence for the remaining search

A viable single-port chamber needs more than two large local detours on a
common route.  It needs a deletion-incompatible route code:

1. every activation set has a cheap depot tour;
2. no all-active cheap tour shortcuts to cheap tours for all subsets; and
3. learning which incompatible order is needed must itself force movement
   on the same scale.

The third item is essential because calls are remote and inactive calls are
free.  The theorem above settles the commuting-switch family completely; it
does not rule out a noncommuting switching network.

