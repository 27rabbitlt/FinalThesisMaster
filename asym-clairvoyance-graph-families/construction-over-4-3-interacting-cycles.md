# An interacting-cycle open gadget above \(4/3\)

## Scope and status

This note gives a **complete local open-service construction** with ratio
strictly above \(4/3\), in fact tending to \(3/2\).  The construction is a
quasirandom tournament metric: its many directed triangles are interacting
cycles, and it is not reducible to the one-cycle clear-gap model.

The local theorem is fully proved for every causal policy, including policies
that call all selectors first or intersperse the two boundary ports.  The
directed metric and shortest-path closure are exact.

What is **not** claimed here is a completed recursive closure into a standard
depot instance.  The cheap source and sink interfaces that make the local
ratio approach \(3/2\) also make the naive child-piece connector too cheap to
pay for arbitrary re-entry.  A second global lap can batch repairs from many
copies.  Thus this is a genuine winning local quotient, but a
ports-and-pieces recursion still needs a protected-interface lemma.  Stating
otherwise would leave a real gap in the proof.

## 1. A tournament far from every linear order

For a tournament \(T\) on vertex set \([n]\) and a permutation
\(\sigma\), call a pair \(\{u,v\}\) **backward** if \(u\) precedes \(v\) in
\(\sigma\) but the tournament arc is \(v\to u\).

### Lemma 1

For every fixed \(\eta>0\) and all sufficiently large \(n\), there is a
tournament \(T_n\) for which every permutation has at least
\[
       \left(\frac12-\eta\right)\binom n2
 \tag{1.1}
\]
backward pairs.

### Proof

Orient every unordered pair independently and uniformly.  For a fixed
permutation, the number of backward pairs is
\(\operatorname{Binomial}(\binom n2,1/2)\).  A Chernoff bound gives
\[
 \Pr\!\left[
   B_\sigma<\left(\frac12-\eta\right)\binom n2
 \right]
 \le \exp(-\Omega_\eta(n^2)).
\]
There are \(n!=\exp(O(n\log n))\) permutations.  A union bound is smaller
than one for sufficiently large \(n\), proving existence. \(\square\)

Fix one such deterministic tournament.  This outer random choice is only a
probabilistic-method certificate; it is not hidden from either benchmark.

## 2. The directed metric

There are two permanent boundary clients \(a,b\) and stochastic selector
clients \(V=[n]\).  The open-service ports are \(\{a,b\}\).  Fix
\[
       0<\varepsilon<\frac14.
\]
The directed distances, before taking closure, are:
\[
\begin{array}{c|c}
\text{ordered pair}&\text{length}\\ \hline
a\to v&\varepsilon\quad(v\in V),\\
v\to b&\varepsilon\quad(v\in V),\\
a\to b&2\varepsilon,\\
u\to v&1\quad\text{if }u\to v\text{ is an arc of }T_n,\\
u\to v&2\quad\text{if }v\to u\text{ is an arc of }T_n,\\
b\to a&2.
\end{array}
\tag{2.1}
\]
The remaining boundary directions are supplied by their shortest displayed
walks:
\[
       b\to v: b\to a\to v,\qquad
       v\to a:v\to b\to a .
\]
Equivalently, use (2.1) as a finite strongly connected generating digraph and
take shortest-path closure.

### Lemma 2 (closure audit)

In the shortest-path metric:
\[
\begin{aligned}
d(a,v)&=\varepsilon,&d(v,b)&=\varepsilon,&d(a,b)&=2\varepsilon,\\
d(u,v)&=
\begin{cases}
1,&u\to v\text{ in }T_n,\\
2,&v\to u\text{ in }T_n,
\end{cases}\\
d(b,a)&=2,&d(b,v)&=2+\varepsilon,&d(v,a)&=2+\varepsilon.
\end{aligned}
\tag{2.2}
\]

### Proof

All asserted upper bounds are displayed paths.  A reverse tournament arc of
length two cannot be shortened through a third selector: two selector arcs
have total length at least two.  A path through the boundary has length at
least
\[
       u\to b\to a\to v
       =2+2\varepsilon>2.
\]
A path leaving \(b\) must first use \(b\to a\), and a path entering \(a\)
must last use \(b\to a\); this proves the last three equalities.  The other
claims follow directly from positivity and the displayed outgoing/incoming
interfaces. \(\square\)

The selector digraph contains many overlapping directed triangles.  Unlike a
one-way cycle, there is no realization-wise “omit one clear gap”
description.

## 3. Activation law

Every selector is independently active with probability
\[
       p_n=\frac{\lambda}{n},
 \tag{3.1}
\]
where \(\lambda>0\) is a fixed constant.  Both \(a\) and \(b\) are permanent.
Let
\[
       K=|\{v\in V:X_v=1\}|
       \sim\operatorname{Binomial}(n,\lambda/n).
\]

The activation law is a product distribution.  The tournament orientation is
part of the known deterministic metric, not a random realization label.

## 4. Exact a-posteriori open cost

Every finite tournament has a directed Hamilton path.  Hence, for each active
selector set of size \(k\ge1\), order its vertices as
\[
       v_1\to v_2\to\cdots\to v_k
\]
in the induced tournament and use
\[
       a\to v_1\to\cdots\to v_k\to b.
\]
Its cost is
\[
       2\varepsilon+k-1.
\]
For \(k=0\), use \(a\to b\), of cost \(2\varepsilon\).

This is optimal.  An open service through \(k\) distinct selector clients has
at least \(k-1\) selector-to-selector transitions after deleting the two
boundary endpoints, and every such transition costs at least one.  The
cheapest possible entry and exit cost \(\varepsilon\) each.  Any placement of
\(b\) before the end or \(a\) after the start costs at least two and cannot
improve this bound.

Thus realization-wise
\[
       F(K)=2\varepsilon+(K-1)_+,
 \tag{4.1}
\]
and exactly
\[
\begin{aligned}
 P_n
 &:=\operatorname{OPT}_{\rm post}^{\rm open}\\
 &=2\varepsilon+\mathbb E[(K-1)_+]\\
 &=2\varepsilon+np_n-\Pr[K\ge1]\\
 &=2\varepsilon+\lambda-1+
       \left(1-\frac{\lambda}{n}\right)^n.
\end{aligned}
\tag{4.2}
\]

## 5. Universal adaptive lower bound

### Lemma 3 (a backward active pair costs one extra)

Suppose exactly two selectors \(u,v\) are active, \(u\) is called before
\(v\), and the tournament arc is \(v\to u\).  Then every open adaptive
execution has cost at least
\[
       2+2\varepsilon=F(2)+1.
\tag{5.1}
\]

### Proof

The direct transition \(u\to v\) costs two.  Inserting one or both permanent
boundary clients between these active calls cannot reduce it:
\[
\begin{aligned}
d(u,a)+d(a,v)&=2+2\varepsilon,\\
d(u,b)+d(b,v)&=2+2\varepsilon,\\
d(u,b)+d(b,a)+d(a,v)&>2.
\end{aligned}
\]
Any external entry before \(u\) and exit after \(v\) cost at least
\(\varepsilon\) each in the favorable order \(a,\ldots,b\).  Other placements
of \(a,b\) cost more.  Hence the complete service costs at least
\(\varepsilon+2+\varepsilon\). \(\square\)

### Lemma 4 (policy-uniform pair bound)

For every deterministic causal open policy,
\[
 \mathbb E[\operatorname{cost}]
 \ge
 P_n+
 \left(\frac12-\eta\right)\Pr[K=2].
\tag{5.2}
\]
The same bound holds for randomized policies.

### Proof

Follow the policy on the branch on which every selector call made so far is
inactive.  Calls to the permanent clients have deterministic outcomes.
Consequently this branch defines a permutation \(\sigma\) of all selector
clients: their call order before the first active selector is encountered.

Condition on \(K=2\).  The active set is a uniformly random unordered pair.
Whichever member appears first in \(\sigma\) is necessarily the first active
selector called; before that call, all observed selector outcomes are
inactive.  By Lemma 1, for at least a
\((1/2-\eta)\)-fraction of pairs the tournament edge points backward relative
to \(\sigma\).  Lemma 3 charges one extra on every such pair.

On all other realizations the execution costs at least the realization-wise
posterior optimum (4.1).  Averaging gives (5.2).  Conditioning on a private
random seed proves the same statement for randomized policies. \(\square\)

Since
\[
 \Pr[K=2]
 =\binom n2\left(\frac{\lambda}{n}\right)^2
  \left(1-\frac{\lambda}{n}\right)^{n-2},
\tag{5.3}
\]
we obtain
\[
 \liminf_{n\to\infty}
 \frac{\operatorname{OPT}_{\rm adapt}^{\rm open}}{P_n}
 \ge
 1+
 \frac{(1/2-\eta)e^{-\lambda}\lambda^2/2}
      {2\varepsilon+\lambda-1+e^{-\lambda}}.
\tag{5.4}
\]
Letting \(\eta\downarrow0\), the extra term is
\[
       \frac{e^{-\lambda}\lambda^2/4}
            {2\varepsilon+\lambda-1+e^{-\lambda}}.
\tag{5.5}
\]

For example, take
\[
       \lambda=0.1,\qquad \varepsilon=10^{-4}.
\]
Then
\[
\begin{aligned}
\lambda-1+e^{-\lambda}+2\varepsilon
   &\approx0.00503742,\\
e^{-\lambda}\lambda^2/4
   &\approx0.00226209,
\end{aligned}
\]
so the limiting lower bound is approximately
\[
       1.449> \frac43.
\tag{5.6}
\]

More conceptually, take first \(n\to\infty\), then
\(\varepsilon=o(\lambda^2)\), and finally \(\lambda\downarrow0\).  Since
\[
       \lambda-1+e^{-\lambda}
       =\frac{\lambda^2}{2}+O(\lambda^3),
\]
the certified local ratio tends to
\[
       1+\frac{1/4}{1/2}=\frac32.
\tag{5.7}
\]

## 6. Failure audit

### Selector-first probing

Lemma 4 explicitly analyzes the all-inactive selector branch, which is the
branch followed by every selector-first strategy until its first active
answer.  On the exactly-two-active event the policy cannot know the second
active selector when it commits to the first.  Large feedback in every
permutation makes nearly half of those transitions reverse.

### Arbitrary placement of the permanent ports

Lemma 3 permits \(a,b\) before, between, or after the active pair.  Using a
port as a repositioning call costs at least as much as the reverse
tournament transition.

### Inactive transit

All selector vertices remain available as transit vertices.  Lemma 2 already
takes paths through arbitrary third selectors and both ports into account.
No activation deletes an arc.

### Shortest-path closure

Lemma 2 proves the complete relevant distance table after closure.  Two cheap
tournament arcs can tie a reverse arc at cost two, but cannot beat it.

### Exceptional activation counts

The posterior value (4.2) is exact for every \(K\), not only asymptotically.
The adaptive proof uses \(K=2\) for an additional charge and the exact
realization optimum as the lower bound on all other events.

## 7. Why the obvious recursion is not yet valid

Replacing \(a,b\) by child port blocks suggests a recurrence
\[
 P_k=2P_{k-1}+P_n,\qquad
 A_k\stackrel{?}{\ge}2A_{k-1}+A_n.
\]
The question mark is essential.  The cheap arcs
\(a\to v\) and \(v\to b\) have length \(\varepsilon\), while reconnecting two
pieces of a child may require a path on the scale of the reverse tournament
arc, namely one or two.  The Chapter 4 subtraction
\[
 Z+D_{\rm child}(N-1)
\]
therefore cannot be paid by every extra top-level child entry.

Nor does serially placing many copies solve this automatically.  A policy can
defer the second active member of a backward pair in many copies and repair
several copies on an additional global pass.  Unless each re-entry crosses a
separately chargeable toll, one extra lap can be counted repeatedly.

The exact missing statement is:

> **Protected-interface lemma.**  Realize the distance table (2.2) between
> two child port blocks and the selectors so that (i) every extra child
> service piece crosses a distinct boundary arc of length at least the child
> port diameter, while (ii) the first source-to-selector and selector-to-sink
> movements remain \(\varepsilon\) on the tournament scale, and (iii) no
> boundary arc or global lap can pay for violations in two different copies.

This is a nontrivial geometric requirement.  A standard one-way cycle fails
item (iii), precisely by the Bernoulli-max ceiling proved in the companion
cycle note.

## Verdict

**Complete local win, incomplete recursive closure.**  Interacting directed
cycles can beat the \(4/3\) local open-service barrier: the tournament gadget
has a rigorously certified ratio approaching \(3/2\) against every causal
policy.  This proves that the \(4/3\) ceiling is specific to the one-cycle
clear-gap structure, not a universal open-service phenomenon.

The result is not yet a complete lower bound for the original closed-depot
stochastic TSP, because the protected-interface lemma required for recursive
composition remains open.
