# Ordered stochastic marker banks have a \(1.304\) ceiling

## Verdict

An ordered bank of independent Bernoulli markers does solve the elementary
one-hot problem: the index of its last active marker can have an arbitrary
categorical distribution, and all active markers can be served as one
increasing chain.  It does **not** yield a clairvoyance gap above \(4/3\).

For the natural fixed-poset construction with incomparable marker banks and
permanent maximal targets, there is a legal causal policy whose expected
closed-depot cost is at most
\[
                         2-0.696=1.304
\]
times the a-posteriori optimum.  This statement permits arbitrary bank
lengths, arbitrary independent marker probabilities, and arbitrary
bank--target incidences consistent with the poset.

The policy finishes banks in a private uniformly random order and implements
random-arrival RANKING on their realized last-marker neighborhoods.  A
chain-splitting lemma proves that neither the posterior nor a speculative
tour can obtain more net target savings by cutting one bank into several
pieces.

Trying to restore a KVV adversarial arrival order breaks the reduction.
Putting a shared target between chronological stages creates a poset cycle
as soon as that target is eligible later.  Ordering the banks directly
merges their marker chains and removes the matching baseline.  Thus the
bank idea repairs independent type generation, but not the incompatible
combination of chronology and reusable unit-capacity targets.

## 1. The bank poset

Let \(T\) be a set of \(n\) permanent target clients.  For each
\(i\in I\), make a marker chain
\[
              C_i:\quad b_{i,1}<b_{i,2}<\cdots<b_{i,m_i}.
\tag{1.1}
\]
Every marker is independently active, with arbitrary probability.  Distinct
marker chains are incomparable.  Targets are pairwise incomparable and
maximal.  The only other possible relations are
\[
                              b_{i,j}<t,
              \qquad t\in T.
\tag{1.2}
\]

For a realization \(A\), call bank \(i\) nonempty if
\(A\cap C_i\ne\varnothing\), and let
\[
              z_i=\max(A\cap C_i)
\tag{1.3}
\]
be its last active marker.  Write \(B(A)\) for the number of nonempty banks.
Define the realized bipartite graph
\[
 G_A=(\{i:A\cap C_i\ne\varnothing\},T;E_A),
 \qquad
 it\in E_A\ \Longleftrightarrow\ z_i<t,
\tag{1.4}
\]
and put \(M^*(A)=\nu(G_A)\).

There is already a restriction hidden in transitivity.  If
\[
 U_{i,j}:=\{t:b_{i,j}<t\},
\]
then
\[
                  U_{i,1}\supseteq U_{i,2}\supseteq\cdots
                  \supseteq U_{i,m_i}.
\tag{1.5}
\]
Thus a single chain can encode only a nested family of target
neighborhoods.  The ceiling below is stronger: it continues to hold even
if one grants arbitrary realized neighborhoods in (1.4).

### Independent categorical encoding

Let a desired last-index law be
\[
        \Pr[J=j]=q_j\quad(1\le j\le m),\qquad
        \Pr[J=0]=q_0,
\]
where \(J=0\) means that the bank is empty.  Put
\[
        Q_j=\sum_{k=0}^j q_k,\qquad
        p_j=
        \begin{cases}
          q_j/Q_j,&Q_j>0,\\
          0,&Q_j=0.
        \end{cases}
\tag{1.6}
\]
If marker \(j\) is independently active with probability \(p_j\), then
\(J\) has the desired law.  Indeed, whenever \(q_j>0\), all relevant
cumulative probabilities are positive and
\[
 \Pr[J=j]
   =p_j\prod_{\ell=j+1}^m(1-p_\ell)
   =\frac{q_j}{Q_j}
      \prod_{\ell=j+1}^m\frac{Q_{\ell-1}}{Q_\ell}
   =q_j.
\tag{1.7}
\]
The zero-probability cases follow directly from (1.6).
Therefore the obstruction is not a failure of the marginal type encoding.

## 2. An exact chain-splitting lemma

### Lemma 1

For every realization \(A\), the minimum number of increasing chains
covering all active markers and all permanent targets is
\[
                         K^*(A)=B(A)+n-M^*(A).
\tag{2.1}
\]

### Proof

A maximum matching in \(G_A\) gives a cover with the claimed number of
chains.  Put all active markers of each nonempty bank into their one natural
increasing chain.  For every matched edge \(it\), append \(t\) after \(z_i\).
Leave unmatched targets as singleton chains.  This uses
\(B+n-M^*\) chains.

For the reverse inequality, consider an arbitrary chain cover, including a
cover that splits marker banks.  Let \(s_i\) be the number of chains meeting
the active markers of nonempty bank \(i\), and let \(q_i\) be the number of
those chains that end in a target.  A chain cannot meet two marker banks,
and it contains at most one target, so the total number of chains is
\[
       \sum_i s_i+n-\sum_i q_i
       =
       B+n-\sum_i(q_i-s_i+1).
\tag{2.2}
\]

If \(q_i=s_i\), every marker piece of bank \(i\) ends in a target.  In
particular, the piece containing the final marker \(z_i\) ends in some
target \(t_i\) with \(z_i<t_i\).  The targets \(t_i\) are distinct over all
such banks.  Hence these final pieces extract a matching in \(G_A\).
Moreover,
\[
 q_i-s_i+1=
 \begin{cases}
  1,&q_i=s_i,\\
  \le 0,&q_i<s_i.
 \end{cases}
\]
It follows that
\[
                     \sum_i(q_i-s_i+1)\le M^*(A).
\tag{2.3}
\]
Substituting (2.3) in (2.2) proves the lower bound. \(\square\)

This is the exact accounting for early target speculation.  Splitting a
bank creates one extra marker chain.  An extra target transition can at
best cancel that new chain.  A net saving survives only for a bank whose
piece containing the actual last active marker also receives a target, and
those surviving savings form an ordinary matching of the realized final
types.

## 3. A fully metric version

Fix \(0<\varepsilon<1\).  On distinct clients define
\[
 d_\varepsilon(x,y)=
 \begin{cases}
   \varepsilon,&x<y\text{ in the bank poset},\\
   1,&\text{otherwise}.
 \end{cases}
\tag{3.1}
\]
Give the depot \(r\) unit distance to and from every client.

This is a directed metric.  If \(x\not<z\), a two-arc path of length below
one would require \(x<y<z\), contradicting transitivity.  If \(x<z\), the
direct distance \(\varepsilon\) is no greater than any positive two-arc
path.  The complete directed distance graph is strongly connected.

If a realization has \(N(A)\) active clients and its active-call order has
\(K\) maximal increasing runs, its closed-depot cost is exactly
\[
 \begin{aligned}
 C(A,K)
   &=2+(K-1)+\varepsilon(N-K)\\
   &=1+\varepsilon N+(1-\varepsilon)K.
 \end{aligned}
\tag{3.2}
\]
Since (3.2) is increasing in \(K\), Lemma 1 gives
\[
 \operatorname{OPT}_{\rm post}(A)
   =1+\varepsilon N(A)+(1-\varepsilon)
       \bigl(B(A)+n-M^*(A)\bigr).
\tag{3.3}
\]
Thus there is no zero-distance or shortest-path-closure loophole in the
negative result.

## 4. A causal RANKING tour

The following policy is physically executable.

1. Privately sample a uniformly random permutation of all marker banks and
   a uniformly random ranking of the targets.
2. Process banks in the sampled order.  Within a bank, call every marker in
   increasing chain order.
3. If the bank is nonempty, the policy is now located at its last active
   marker \(z_i\) and knows its complete realized neighborhood.  If it has
   an unused neighboring target, immediately call the minimum-ranked such
   target.
4. After all banks have been processed, call all remaining targets.

Inactive marker calls do not move the salesperson.  The active markers in
one bank occur consecutively and increasingly.  Conditional on the entire
realization \(A\), the relative order of the nonempty banks is uniform.
The target decisions are therefore exactly RANKING on the fixed realized
graph \(G_A\) in the random-arrival model.

The random-arrival analysis of Mahdian and Yan gives a constant
\(\beta\ge0.696\) such that, conditional on every realization,
\[
 \mathbb E_{\rm seed}[M_{\rm rank}(A)]
                         \ge\beta M^*(A).
\tag{4.1}
\]
The active-call order of this policy uses
\[
                   K_{\rm rank}(A)=B(A)+n-M_{\rm rank}(A)
\tag{4.2}
\]
runs.  Since \(M^*(A)\le B(A)\) and \(M^*(A)\le n\),
\[
       M^*(A)\le B(A)+n-M^*(A)=K^*(A).
\tag{4.3}
\]
Equations (4.1)--(4.3) imply
\[
 \begin{aligned}
 \mathbb E_{\rm seed}[K_{\rm rank}(A)]
 &\le K^*(A)+(1-\beta)M^*(A)\\
 &\le(2-\beta)K^*(A).
 \end{aligned}
\tag{4.4}
\]

Take expectation over the independent activations.  With
\[
          C_0=1+\varepsilon\,\mathbb E N(A)\ge0,
\]
(3.2)--(4.4) give
\[
 \begin{aligned}
 \operatorname{OPT}_{\rm adapt}
 &\le C_0+(1-\varepsilon)(2-\beta)\mathbb E K^*(A)\\
 &\le(2-\beta)
       \left(C_0+(1-\varepsilon)\mathbb E K^*(A)\right)\\
 &=(2-\beta)\operatorname{OPT}_{\rm post}.
 \end{aligned}
\tag{4.5}
\]
Consequently
\[
 \boxed{\displaystyle
 \frac{\operatorname{OPT}_{\rm adapt}}
      {\operatorname{OPT}_{\rm post}}
 \le 2-\beta\le1.304<\frac43.}
\tag{4.6}
\]

The use of private randomization is harmless.  Averaging (4.5) over the
private seed shows that at least one fixed seed has no larger expected cost,
so a deterministic causal policy with the same instance-level bound exists.

## 5. Why arbitrary probing does not rescue a lower bound

The proof above is an adaptive **upper** bound, so it does not need to claim
that every optimal policy processes banks forward.  Nevertheless, the main
nonstandard behaviors can be located exactly.

### Descending probes

A policy may query high markers first and find the last active index before
serving lower active markers.  Those later lower calls are backward in the
poset and create additional runs.  Forward processing learns the same final
index after probing the whole bank while keeping all its active markers in
one run.  More importantly, the explicit forward RANKING tour already proves
(4.6), regardless of whether some more complicated descending policy is
even cheaper.

### Interleaving banks

Remote calls allow partial probing of many banks.  An inactive answer is
free, while an active answer may jump to a different incomparable bank and
split the current marker chain.  A lower-bound proof based on exogenous
source arrivals cannot forbid this.  Again (4.6) survives because it only
exhibits one legal policy; the adaptive optimum can be no larger.

### Calling a target before the type is resolved

If a target is called after a provisional active marker and a higher marker
is later found active, the bank has been split.  Lemma 1 gives the exact
charge: every additional targeted piece is canceled by an additional marker
run, except for a target following the piece containing the true last
marker.  Thus speculative target calls do not improve the posterior formula
or create a hidden multi-target saving from one bank.

### Calling targets first

Targets are permanent and their identities reveal no random information.
Calling one before a bank merely consumes it as a singleton run.  It cannot
invalidate the legal RANKING policy or increase the adaptive optimum.

## 6. Why KVV chronology cannot be forced

The preceding ceiling uses the freedom to randomize the bank order.  It is
therefore natural to try to force the random-suffix/KVV order inside the
poset.  The two direct mechanisms both destroy the matching reduction.

### Shared-target cycle

Suppose a stage-\(i\) endpoint \(x_i\) is to be matched to a shared target
\(t\), and using that target is also supposed to continue cheaply into the
next chronological stage.  Then the intended chain contains
\[
                         x_i<t<g_{i+1},
\tag{6.1}
\]
where \(g_{i+1}\) precedes every possible later endpoint \(x_j\),
\(j>i\).  If the same target is eligible at stage \(j\), then
\[
                              x_j<t.
\tag{6.2}
\]
Chronology and (6.1)--(6.2) give
\[
                         t<g_{i+1}\le x_j<t,
\]
a poset cycle.  Hence a target that lies inside one chronological cut cannot
remain an eligible unit-capacity target at a later cut.

If the relation \(t<g_{i+1}\) is omitted, a matched target ends the current
chain.  Starting the next bank costs the same incomparable reset for every
unserved bank, so the policy may choose that next bank arbitrarily.  The
metric has not imposed an arrival order.

### Directly ordering the banks

Suppose instead that marker banks are made an ordinal sum,
\[
                              C_1<C_2<\cdots<C_h.
\tag{6.3}
\]
All active markers across all nonempty banks then fit in one increasing
chain.  Since the \(n\) targets are an antichain, the posterior chain-cover
number is at least \(n\).  Serving the global marker chain separately and
leaving all targets as singletons uses \(n+1\) chains.  Hence the exact
posterior count is always either \(n\) or \(n+1\), regardless of how many
banks are nonempty.  Equivalently, whenever an additional target is appended
by splitting the global marker chain, that split adds the source piece that
cancels the target saving.  The extensive source baseline \(B+n-M\) has
collapsed to a single bit and cannot carry a linear KVV matching deficit.

Permanent gates or shortest-path transit do not repair either issue.
A served gate or target remains available as a metric transit vertex; it is
not a consumable resource.  Stage-specific target copies avoid the cycle
only by eliminating competition for unit-capacity targets.

## 7. Exact scope of the obstruction

The following three ingredients cannot coexist in this construction:

1. incomparable banks, which give the posterior baseline
   \(B+n-M^*\);
2. a forced adversarial order of those banks; and
3. shared target clients with unit matching capacity.

With item 1, the policy is free to randomize the source order and (4.6)
applies.  Adding chronology through shared targets violates antisymmetry;
adding it directly between banks removes item 1.

Therefore ordered independent marker banks are a useful type-encoding
device, but the complete separated-bank reachability family has a certified
clairvoyance-gap ceiling of \(1.304\).  It cannot be the requested
construction above \(4/3\).

## Reference

M. Mahdian and Q. Yan, *Online Bipartite Matching with Random Arrivals: An
Approach Based on Strongly Factor-Revealing LPs*, STOC 2011.  The
random-arrival RANKING guarantee used here is also summarized in
`source-first-ranking-0696-obstruction.md`.
