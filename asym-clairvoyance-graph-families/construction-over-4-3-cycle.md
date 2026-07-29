# Higher-arity directed-cycle gadgets: a sharp \(4/3\) ceiling

## Outcome

The proposed higher-arity cycle mutation cannot give a gap above \(4/3\).
This note proves a sharp universal upper bound for the whole following class:

* any number of permanent ports on a one-way directed cycle;
* unequal positive gap lengths;
* any number of independent stochastic selectors in each gap;
* unequal selector probabilities and arbitrary selector positions inside a
  gap; and
* recursive composition whenever the child/open-service decomposition is
  exact in the usual ports-and-pieces sense.

For a fixed realization, the exact open cost is the circumference minus the
largest gap containing no active selector.  A simple legal adaptive policy
preselects one gap and achieves expected cost equal to the circumference
minus that gap's expected saving.  The key inequality is
\[
 4\,\mathbb E\!\left[\max_i w_i Z_i\right]
 \le
 \sum_i w_i+3\max_i\mathbb E[w_iZ_i],
 \tag{0.1}
\]
for independent Bernoulli \(Z_i\)'s.  It implies that this adaptive policy is
always within \(4/3\) of the a-posteriori optimum.

The bound is tight.  The Chapter 4 quotient has two gaps of lengths \(2w\)
and \(w\); the long gap is clear with probability \(1/2\), and the short gap
is deterministically clear.  Equality holds in (0.1), producing exactly the
local ratio \(4/3\).

Thus more gaps, unequal weights/probabilities, and nested independent
selectors do not improve the construction.  A successful gadget must leave
the single-cycle/one-omitted-gap paradigm—for example, it must have several
interacting directed cycles so that one selector controls incompatible cuts
in more than one cycle.

## 1. The complete cycle-gap model

Let
\[
       v_0,v_1,\ldots,v_{n-1}
\]
be permanent clients and open-service ports, in this cyclic order.  Between
\(v_i\) and \(v_{i+1}\) (indices modulo \(n\)) put a directed path \(G_i\) of
total length \(w_i>0\).  The internal vertices of \(G_i\) may include an
arbitrary set \(S_i\) of stochastic clients.  Their activation bits are
mutually independent, both within and across gaps, and may have unequal
probabilities.  There are no other arcs.  Thus the generating graph is one
directed cycle and is strongly connected.

Distances are directed shortest-path distances.  Because there is only one
forward route from any vertex until the first complete wrap, shortest-path
closure introduces no alternative shortcut.  Expanded metric moves are
simply clockwise walks, possibly making extra full turns.

Write
\[
       W:=\sum_{i=0}^{n-1}w_i
\]
for the circumference, and let
\[
 Z_i=
 \mathbf 1_{\{\text{every selector in }S_i\text{ is inactive}\}}.
\]
Call \(G_i\) **clear** when \(Z_i=1\).  Since the selector sets in distinct
gaps are disjoint, the \(Z_i\)'s are independent Bernoulli variables, with
\[
 q_i:=\Pr[Z_i=1]
     =\prod_{s\in S_i}(1-p_s).
 \tag{1.1}
\]
An empty gap has \(q_i=1\).  This includes deterministic clear gaps and
arbitrarily many “nested” selectors placed serially in a gap.

An open service starts and ends at freely chosen permanent ports, calls every
permanent and stochastic client exactly once, and serves all active clients.
Initial and terminal internal movement are charged in the same way as in
Chapter 4.

## 2. Exact fixed-realization open cost

### Lemma 1

For every activation realization \(z\),
\[
 F(z)=W-\max_i w_i z_i,
 \tag{2.1}
\]
where the maximum is zero if no gap is clear.

### Proof

If \(G_j\) is clear, start at \(v_{j+1}\), traverse every other gap in cyclic
order, and finish at \(v_j\).  Along a traversed gap, call its stochastic
clients in their geometric order.  Active calls follow the directed path;
inactive calls cause no movement.  The resulting open service has cost
\(W-w_j\).  If no gap is clear, one full turn of cost \(W\) serves everything.
This proves the upper bound in (2.1).

For the lower bound, expand an arbitrary open service into the directed
cycle.  A walk of cost at least \(W\) already satisfies (2.1).  A walk of
cost less than \(W\) cannot make a full turn.  Lift the cycle to its universal
cover, a directed line.  The expanded service is then contained in one
forward interval.  Since it visits all permanent ports, its omitted
complement must be exactly one complete gap \(G_j\): starting or ending at an
internal stochastic vertex is forbidden because open endpoints must be
ports.  No active selector may lie in the omitted gap.  Hence \(G_j\) is
clear, and the saving is at most \(w_j\le\max_i w_i z_i\). \(\square\)

Consequently, with
\[
       M:=\max_i w_iZ_i,
\]
the exact expected a-posteriori top cost is
\[
       P_{\rm cyc}=W-\mathbb E[M].
 \tag{2.2}
\]
There are no exceptional events in this formula.

## 3. A legal adaptive policy

Fix a gap \(G_j\).  Use the following policy.

1. Enter for free at \(v_{j+1}\), which is permanent and hence active.
2. Traverse the other \(n-1\) gaps in cyclic order, ending at \(v_j\).
   Within every traversed gap, call its selectors in geometric order and
   then call the next permanent port.
3. From \(v_j\), call all selectors of \(G_j\), again in geometric order.
   If they are all inactive, terminate at \(v_j\).  If at least one is
   active, the active calls move forward through the gap; append the charged
   terminal movement to port \(v_{j+1}\).

Every traversed gap costs exactly its length regardless of its activation
pattern.  Indeed, inactive calls leave the position fixed, active calls move
forward, and the next permanent call or terminal movement completes the
remaining part of the path.  The selected gap costs zero if it is clear and
\(w_j\) otherwise.  Therefore the policy's expected top cost is
\[
       A_j=W-q_jw_j.
 \tag{3.1}
\]
Choosing a gap with maximum expected saving gives
\[
       \operatorname{OPT}_{\rm adapt}
       \le A_{\rm cyc}:=W-m,
       \qquad
       m:=\max_i q_iw_i.
 \tag{3.2}
\]

This policy is already robust to the model-specific failure modes:

* calls are remote, but it uses them only in the forward order in which an
  active answer is useful;
* inactive selectors are allowed as transit vertices;
* all selector calls are legal and made exactly once;
* terminal movement in the selected active gap is charged; and
* no assumption is made that a gap is served in one uninterrupted metric
  move.

The optimal adaptive value can be lower than (3.2), because a policy may probe
several gaps.  That only strengthens the impossibility result below; an exact
adaptive evaluation is unnecessary for proving the universal ceiling.

## 4. The weighted Bernoulli-max inequality

### Lemma 2

Let \(Z_1,\ldots,Z_n\) be independent Bernoulli variables with
\(\Pr[Z_i=1]=q_i\), and let \(w_i>0\).  Put
\[
       W=\sum_iw_i,\qquad
       M=\max_iw_iZ_i,\qquad
       m=\max_iq_iw_i.
\]
Then
\[
       4\mathbb E[M]\le W+3m.
 \tag{4.1}
\]

### Proof

Order the variables so that \(w_1\ge w_2\ge\cdots\ge w_n\), and induct on
\(n\).  The one-variable case is
\(4q_1w_1\le w_1+3q_1w_1\), which follows from \(q_1\le1\).

For the induction step write
\[
\begin{aligned}
 w&:=w_1,& q&:=q_1,& a&:=qw,\\
 W'&:=\sum_{i\ge2}w_i,&
 M'&:=\max_{i\ge2}w_iZ_i,&
 \mu&:=\max_{i\ge2}q_iw_i .
\end{aligned}
\]
The largest weight wins whenever \(Z_1=1\), so independence gives
\[
       \mathbb E[M]=a+(1-q)\mathbb E[M'].
 \tag{4.2}
\]
By induction,
\[
       4\mathbb E[M']\le W'+3\mu.
 \tag{4.3}
\]
Also \(W'\ge\mu\) when the remainder is nonempty.  Let
\[
 D:=W+3\max\{a,\mu\}-4\mathbb E[M].
\]
Equations (4.2)--(4.3) imply
\[
 D\ge
 w-4a+qW'+3\max\{a,\mu\}-3\mu+3q\mu.
 \tag{4.4}
\]

If \(a\le\mu\), then (4.4) becomes
\[
       D\ge w+q(W'+3\mu-4w).
\]
If the parenthesis is nonnegative, this is nonnegative.  Otherwise use
\(q\le\mu/w\) and \(W'\ge\mu\) to obtain
\[
 D\ge
 w+\frac{\mu}{w}(4\mu-4w)
 =\frac{(w-2\mu)^2}{w}\ge0.
\]

If \(a\ge\mu\), then (4.4) becomes
\[
       D\ge(1-q)(w-3\mu)+qW'.
\]
This is nonnegative when \(w\ge3\mu\).  Otherwise use
\(q\ge\mu/w\), \(W'\ge\mu\), and \(4\mu-w>0\):
\[
\begin{aligned}
 D
 &\ge w-3\mu+q(4\mu-w)\\
 &\ge w-3\mu+\frac{\mu}{w}(4\mu-w)
 =\frac{(w-2\mu)^2}{w}\ge0.
\end{aligned}
\]
Thus \(D\ge0\) in every case, proving (4.1). \(\square\)

This lemma is a prophet-versus-fixed-choice statement tailored to cycle
savings.  The clairvoyant takes the largest realized clear-gap weight
\(M\); the simple adaptive policy commits to a gap with largest mean saving
\(m\).

## 5. Universal \(4/3\) ceiling

### Theorem 3

Every cycle-gap instance from Section 1 satisfies
\[
 \frac{\operatorname{OPT}_{\rm adapt}}
      {\operatorname{OPT}_{\rm post}}
 \le\frac43,
 \tag{5.1}
\]
whenever the denominator is positive.

### Proof

By Lemma 2,
\[
 P_{\rm cyc}
 =W-\mathbb E[M]
 \ge W-\frac{W+3m}{4}
 =\frac34(W-m).
\]
The adaptive policy from Section 3 costs \(A_{\rm cyc}=W-m\).  Therefore
\[
 \operatorname{OPT}_{\rm adapt}
 \le W-m
 \le\frac43P_{\rm cyc}
 =\frac43\operatorname{OPT}_{\rm post}.
\]
\(\square\)

This is an upper bound on the *optimal* adaptive policy, so selector-first
probing, separator-first probing, and every more elaborate interleaving are
automatically covered: they can only lower the numerator.

## 6. Equality and the Chapter 4 gadget

Take two gaps:
\[
\begin{array}{c|c|c}
 &w_i&q_i\\ \hline
\text{selector gap}&2w&1/2\\
\text{direct gap}&w&1 .
\end{array}
\]
Both expected gap savings equal \(w\), so \(m=w\) and \(W=3w\).  The
clairvoyant saving is \(2w\) when the selector is inactive and \(w\) when it
is active, hence
\[
       \mathbb E[M]=\frac32w
                   =\frac{W+3m}{4}.
\]
Thus
\[
       P_{\rm cyc}=3w-\frac32w=\frac32w,
       \qquad
       A_{\rm cyc}=3w-w=2w,
\]
and their ratio is \(4/3\).  The selector gap is precisely
\[
       R\longrightarrow m\longrightarrow L,
\]
while the direct gap is \(L\to R\).  This recovers the open top-level
accounting of Chapter 4.

Equality also explains why simply shortening the two arcs incident with the
midpoint does not improve the construction.  If the selector gap becomes too
short, the policy starts on its upstream side and uses that gap in both
realizations.  If it becomes too long, committing to the direct gap becomes
cheap enough.  The balanced point is exactly the \(4/3\) equality case.

## 7. Recursive composition cannot amplify the cycle nodes

Suppose a recursive family has an exact open-service decomposition
\[
\begin{aligned}
 P_k&=\sum_{C\in\mathcal C_k}P_C+P_{{\rm cyc},k},\\
 A_k&\le\sum_{C\in\mathcal C_k}A_C+A_{{\rm cyc},k},
 \tag{7.1}
\end{aligned}
\]
where \(\mathcal C_k\) are the child blocks.  The first equality must be
proved by ports and extra-piece charging; the second inequality is realized
by concatenating child policies with the selected-gap cycle policy.

If every child satisfies \(A_C\le(4/3)P_C\), Theorem 3 and (7.1) give
\[
       A_k\le\frac43P_k.
\]
The same remains true after adding a common depot normalization \(D_k\ge0\):
\[
 \frac{D_k+A_k}{D_k+P_k}\le\frac43.
\]

Thus scale separation and recursion do not rescue the higher-arity cycle.
They can make the quotient accounting exact, as in Chapter 4, but a weighted
sum of cycle nodes whose local ratios are at most \(4/3\) still has ratio at
most \(4/3\).

If a proposed recursion does **not** satisfy the first line of (7.1), then it
needs a new non-additive interleaving argument.  It is no longer a recursion
of independent directed-cycle open gadgets in the sense analyzed here.

## 8. Failure audit

### Early probing

The no-go proof supplies a concrete policy and therefore does not assume that
early probing is expensive.  It preselects the statistically best gap, calls
all other selectors while moving forward, and delays the selected gap until
the end.  Calling more gaps early can only improve its cost.

### Multiple or nested selectors in one gap

Only the event that the whole gap is clear matters to the fixed open optimum.
Its probability is the product (1.1).  When the gap is traversed, selectors
are called in spatial order and cost exactly the gap length collectively.
Serially nesting more selectors therefore only changes \(q_i\); it does not
escape Lemma 2.

### Unequal weights and probabilities

Both are arbitrary in Lemma 2.  The \(4/3\) ceiling is not an artifact of
equiprobable midpoints or equal arcs.

### Interleaving

Lemma 1 expands the complete realized service and permits repeated turns,
remote calls, and arbitrary call order.  The adaptive side uses an explicit
legal policy, so no assumption about the form of an optimal policy is made.
For recursive blocks, arbitrary interleaving must be handled in the
decomposition hypothesis (7.1); once that is done, the ceiling is automatic.

### Shortest-path closure

The generating graph is one directed cycle.  Lifting it to a directed line
gives the lower bound in Lemma 1 and accounts for every metric shortcut.
Adding chords or a second cycle invalidates the lemma and is exactly the kind
of structural departure now required.

### Inactive transit vertices

The policies and fixed tours freely traverse inactive selector vertices.
No argument treats inactivity as deletion of a vertex or arc.

## Verdict

**Rigorous universal impossibility for the directed-cycle gap class.**  No
choice of arity, gap lengths, independent probabilities, number of selectors
per gap, or exact additive recursion yields a clairvoyance gap above \(4/3\).
The Chapter 4 construction is a sharp equality case of the governing
Bernoulli-max inequality.

## Required redirection

The next explicit topology should be a **multi-cycle route-code cell**, not a
larger cycle.  It must have at least two interacting directed cycles sharing
port blocks, with independent downstream selectors arranged so that:

1. every realization has a cheap choice of a *compatible pair* of cycle cuts;
2. no single preselected gap realizes the expected clairvoyant saving in both
   cycles;
3. traversing an inactive selector remains legal and does not open or close an
   arc;
4. mixed choices incur a circulation defect certified after shortest-path
   closure; and
5. early calls to the shared selectors pay the same defect.

The essential new feature is that one route decision must constrain two or
more cut savings.  As long as the posterior advantage is merely
\(\max_i w_iZ_i\) on one cycle, Lemma 2 caps the ratio at \(4/3\).
