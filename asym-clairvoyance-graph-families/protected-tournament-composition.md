# A protected tournament composition above \(4/3\): candidate and failed
# binary interface

> **Audit status (failed as written).**  Lemma 4.0b is false for the binary
> interface below.  A policy can serve \(R\), call \(m_k\), and then behave
> as follows.  If \(m_k\) is active, use
> \(R\to m_k\to L\) and serve the chamber once.  If it is inactive, enter a
> first chamber piece, use the \(2w_k\) chamber-to-input return to reach
> \(L\), and then finish the chamber.  The two heavy costs are \(2w_k\) and
> \(2w_k\), exactly the ordinary triangle pair lower bound \(4w_k\).
> Therefore the inactive orientation slack cannot also pay the missing
> chamber-piece connector.  Equations (4.0b), (4.2), and the claimed closed
> theorem do not follow.  The material is retained because the graph and the
> counterexample isolate the exact double-charge that a valid protected
> interface must eliminate.

## 1. Statement and parameters

This note gives a candidate closed-depot construction obtained by putting a
small strict-gap tournament chamber behind every node of the recursive
directed-triangle construction.  The chamber has a scale much smaller than
the parent triangle.  A sole output gate prevents a policy from using the
chamber as a free second port: after entering the chamber, every return to
unfinished work crosses a private arc of length \(2w_k\).

Fix a number \(\rho>4/3\).  By the tournament lemma in
`construction-over-4-3-interacting-cycles.md`, there are constants
\[
 n\ge 3,\quad p=\lambda/n,\quad \epsilon>0,
 \tag{1.1}
\]
and a tournament \(T\) on \(n\) vertices such that the following directed
open chamber \(Q\) has
\[
       A_Q>\frac43 P_Q.
 \tag{1.2}
\]
The chamber has permanent input and output gates \(a,b\), independently
active selectors \(S=V(T)\), and generating arcs
\[
\begin{array}{c|c}
\text{arc}&\text{length}\\ \hline
a\to s,\ s\to b&\epsilon,\\
a\to b&2\epsilon,\\
s\to t&1\quad(s\to t\text{ in }T),\\
s\to t&2\quad(t\to s\text{ in }T),\\
b\to a&2 .
\end{array}
\tag{1.3}
\]
Here \(P_Q\) and \(A_Q\) are respectively the a-posteriori and adaptive
input-\(a\), output-\(b\) open-service values.  More explicitly,
\[
 P_Q=2\epsilon+\mathbb E[(K-1)_+],
 \qquad K\sim\operatorname{Binomial}(n,p),
 \tag{1.4}
\]
and the feedback-tournament argument gives
\[
 A_Q\ge P_Q+
   \left(\frac12-\eta\right)\Pr[K=2]>\frac43P_Q
 \tag{1.5}
\]
for suitable \(n,\eta,\lambda,\epsilon\).  For example, the choices in the
companion note certify a ratio larger than \(1.44\).

Choose
\[
 0<\delta<
 \min\left\{\frac1{100n},\frac1{100},
 \frac{3A_Q-4P_Q}{100(A_Q+P_Q+1)}\right\}.
 \tag{1.6}
\]
All copies of \(Q\) below are scaled by \(\delta w_k\).  In particular, a
walk that serves an arbitrary realization of a chamber from \(a\) to \(b\)
has cost at most
\[
       D_Q\,\delta w_k,\qquad D_Q:=2n+4,
 \tag{1.7}
\]
and (1.6) makes this smaller than \(w_k/20\).

## 2. Recursive modules with directed interfaces

A module has a nonempty bank \(I(C)\) of input ports and one output port
\(o(C)\).  All boundary arcs entering the module end in its input bank, and
all boundary arcs leaving it start at its output.  An open service starts at
an input port, ends at the output, and serves all permanent and active
clients.

The module \(C_0\) is one permanent client \(z\), with
\[
       I(C_0)=\{z\},\qquad o(C_0)=z.
\tag{2.1}
\]

For \(k\ge1\), put \(w_k=2^{k-1}\).  Form \(C_k\) from disjoint copies
\(L,R\) of \(C_{k-1}\), a midpoint \(m_k\), and a scaled chamber \(Q_k\)
with gates \(a_k,b_k\).  The midpoint is active with probability \(1/2\);
the chamber selectors use (1.1), independently of every other client.
Add these arcs:
\[
\begin{array}{c|c}
\text{arc bundle}&\text{length}\\ \hline
o(L)\to y,\quad y\in I(R)&w_k,\\
o(R)\to m_k&w_k,\\
m_k\to x,\quad x\in I(L)&w_k,\\
o(L)\to a_k,\quad o(R)\to a_k&\gamma\delta w_k,\\
b_k\to x,\quad x\in I(L)\cup I(R)&2w_k ,
\end{array}
\tag{2.2}
\]
where \(\gamma>0\) is fixed and so small that
\[
       A_Q+\gamma>\frac43(P_Q+\gamma).
\tag{2.3}
\]
Inside \(Q_k\), use the arcs (1.3), multiplied by \(\delta w_k\).  Define
\[
 I(C_k):=I(L)\cup I(R),\qquad o(C_k):=b_k.
\tag{2.4}
\]

The two arcs from the child outputs to \(a_k\) are the only forward entrances
to the chamber.  The arcs of length \(2w_k\) from \(b_k\) back to the input
bank are private return arcs.  They make the generating graph strongly
connected, but a one-piece service does not use them.

For depth \(l\), add a depot \(r\), arcs
\[
       r\to x\quad(x\in I(C_l)),\qquad b_l\to r
\tag{2.5}
\]
of length \(w_l\), and take directed shortest-path distances in the resulting
generating graph.  Every generating arc has positive length.

### Closure audit

The chamber distances in (1.3), after scaling, remain exact in the full
metric.  A reverse tournament distance is \(2\delta w_k\).  A path that
leaves \(Q_k\) for unfinished work uses the private return arc of length
\(2w_k\), and therefore is longer because \(2\delta w_k<2w_k\).  A path
that leaves through an ancestor uses an arc on scale \(w_{k+1}=2w_k\) before
it can re-enter \(C_k\), and is no shorter.  At the root, transit through the
depot costs \(2w_l\).  Two tournament arcs can tie, but cannot beat, a reverse
arc.  Thus the feedback-pair lower bound used in (1.5) survives shortest-path
closure.

The private return \(b_k\to I(C_k)\) also remains a shortest return on its
scale.  Going through a child and the top triangle uses at least two
level-\(k\) arcs, while leaving through a parent uses an arc on the next
scale.  Equality is harmless in all accounting below.

## 3. Interrupted directed-interface services

Let \(F_k(A)\) be the least input-to-output open cost in \(C_k\) for active
set \(A\), and let \(G_k\) be its least expected adaptive open cost.  The
free-entry convention chooses an input \(x\in I(C_k)\), charges the internal
distance from \(x\) to the first active call, and requires terminal movement
to \(o(C_k)\).

If a service of a child \(C_{k-1}\) is split into \(N\) pieces, every piece
enters through its input bank and exits through its sole output.  Consecutive
pieces can be joined by the child's private return, whose length is
\[
       2w_{k-1}=w_k.
\tag{3.1}
\]
Consequently, exactly as in the ports-and-interrupted-services lemma,
\[
\begin{aligned}
 Z+w_k(N-1)&\ge F_{k-1}(A),\\
 \mathbb E Z+w_k\mathbb E(N-1)&\ge G_{k-1}.
\end{aligned}
\tag{3.2}
\]
The adaptive proof exposes all randomness outside the child only as an
independent private seed.  The one-output interface makes the connector in
(3.1) canonical.

The same statement applies to a chamber split into \(N_Q\) pieces, using its
internal return \(b\to a\):
\[
\begin{aligned}
 Z_Q+2\delta w_k(N_Q-1)&\ge
       \delta w_k F_Q(A_Q),\\
 \mathbb E Z_Q+2\delta w_k\mathbb E(N_Q-1)&\ge
       \delta w_k A_Q.
\end{aligned}
\tag{3.3}
\]

## 4. The protected-chamber accounting lemma

The following is the point at which the sole output is used.

### Lemma 4.0 (quotient trace)

Collapse every child piece to \(L\) or \(R\), and collapse every
service-containing chamber piece to \(Q\).  Let \(J\) be the number of
nonterminal \(Q\)-pieces, equivalently the number of chamber pieces followed
by a later child-service event.  Let \(H\) be the cost of all level-\(k\)
arcs of length at least \(w_k\), including the private \(Q\)-to-child returns
of length \(2w_k\), but excluding arcs internal to the children.  Then
\[
 H-w_k\bigl((N_L-1)+(N_R-1)\bigr)
 \ge
 w_k\left(1+\mathbf1_{\{m_k\in A\}}+J\right).
\tag{4.0a}
\]

For an adaptive policy, condition on all randomness except the midpoint and
couple its inactive and active executions.  With the analogous notation,
\[
\begin{aligned}
 &H_0-w_k\bigl((N_{L,0}-1)+(N_{R,0}-1)\bigr)\\
 &\quad+
 H_1-w_k\bigl((N_{L,1}-1)+(N_{R,1}-1)\bigr)\\
 &\hspace{3cm}\ge w_k(4+J_0+J_1).
\end{aligned}
\tag{4.0b}
\]

#### Proof

This is the directed-triangle trace count with one extra symbol.  Every
nonterminal \(Q\) is followed by a private return of cost \(2w_k\).
Discounting one unit for that \(Q\), and discounting one unit whenever the
return creates an additional piece of the same child, leaves the following
four possibilities:
\[
\begin{array}{c|c}
\text{local trace after a child output}&
\text{remaining level-}k\text{ cost}\\ \hline
L,Q,R&w_k,\\
L,Q,L&w_k\text{ after the extra-}L\text{ piece charge},\\
R,Q,R&w_k\text{ after the extra-}R\text{ piece charge},\\
R,Q,L&0\text{ locally, but the reverse }R\to L
       \text{ has one unit of orientation slack}.
\end{array}
\tag{4.0c}
\]
The last line means the following.  With an inactive midpoint, an \(R\)-to-
\(L\) trace costs \(2w_k\), one unit more than the required inactive bound.
With an active midpoint, a \(Q\)-return cannot land at \(m_k\).  The midpoint
must therefore be served in a separate \(R\to m_k\to L\) passage, or one
child is re-entered; after the corresponding piece charge there is again
one spare unit.  Deleting the \(Q\)'s one at a time leaves a trace in the
ordinary triangle.  The ordinary fixed trace has adjusted cost at least one
unit when the midpoint is inactive and two when it is active.  Restoring the
deleted \(Q\)'s and their spare units proves (4.0a).

For (4.0b), follow the same deletion simultaneously in the two midpoint
executions.  Chamber outcomes are already conditioned and hence are merely
fixed side information.  After paying one unit for each deleted
nonterminal \(Q\), the two residual traces agree until the midpoint call.
If the inactive residual trace has adjusted cost at least two, the fixed
bounds \(2+2\) give four.  Otherwise its only zero-slack form is the
canonical \(L\to R\) trace, or the discounted \(R,Q,L\) form in the last row
of (4.0c).  In the canonical form, the three standard call phases
(before \(L\), during \(L\), or during/after \(R\)) force active residual
cost at least three.  In the discounted form, an active midpoint call made
from the chamber costs a \(Q\)-to-\(R\) return followed by
\(R\to m_k\), while a call made before or after the chamber forces a separate
\(R\to m_k\to L\) passage.  In either case the active residual cost is at
least three.  Thus the residual pair sum is at least four.  Restoring the
\(J_0+J_1\) discounted units proves (4.0b). \(\square\)

### Lemma 4.1

For every fixed realization in \(C_k\),
\[
 F_k(A)=F_{k-1}(A_L)+F_{k-1}(A_R)
 +w_k\bigl(1+\mathbf1_{\{m_k\in A\}}\bigr)
 +\delta w_k\bigl(\gamma+F_Q(A_Q)\bigr).
\tag{4.1}
\]
For adaptive services,
\[
 G_k\ge 2G_{k-1}
       +2w_k+\delta w_k(A_Q+\gamma).
\tag{4.2}
\]

### Proof

The upper bound in (4.1) is the natural one-piece service.  If \(m_k\) is
inactive, serve \(L\), use \(o(L)\to I(R)\), serve \(R\), and finish with
\(Q_k\).  If \(m_k\) is active, serve \(R\), use
\(o(R)\to m_k\to I(L)\), serve \(L\), and finish with \(Q_k\).

For the lower bounds, expand every metric move and cut it into child and
chamber pieces.  Use (3.2)-(3.3).  It remains to account for arcs outside
the three submodules.

There are \(J+1\) chamber pieces.  Equation (3.3) needs
\(2\delta w_kJ\) to join them into one causal chamber service.  Lemma 4.0
provides \(w_kJ\), and \(2\delta<1\), so these connectors are fully paid.
The same lemma leaves the fixed triangle contribution
\(w_k(1+\mathbf1_{\{m_k\in A\}})\).  At least one child-output-to-\(a_k\)
arc is used, contributing \(\gamma\delta w_k\); additional entrances are
nonnegative.  Combining these facts with (3.2)-(3.3) proves (4.1).

For (4.2), condition on all randomness except the midpoint bit, including
the chamber activations, child activations, and a policy seed.  Chamber
observations made before completing the triangle are independent side
information and therefore amount only to private randomization for the
projected triangle policy.  Couple the executions with the midpoint inactive
and active.  Lemma 4.0 gives pair sum \(4w_k\), plus one full \(w_k\) for
each nonterminal chamber piece.  Since \(2\delta w_k\) joins two chamber
pieces, this surplus pays every chamber reconnection in both executions.

Conversely, project the calls to \(Q_k\).  All child and midpoint
outcomes are independent exterior randomness, so this projection is a causal
adaptive input-to-output chamber service.  Equation (3.3) therefore
contributes at least \(\delta w_k A_Q\) in expectation.  The terminal
entrance contributes \(\gamma\delta w_k\).  Combining this with (3.2) for
the children and the triangle pair bound proves (4.2). \(\square\)

## 5. Expected recurrences and the closed gap

Taking expectation in (4.1), using the midpoint probability \(1/2\), gives
\[
 P_k=2P_{k-1}+c_Pw_k,\qquad
 c_P:=\frac32+\delta(P_Q+\gamma).
\tag{5.1}
\]
Equation (4.2) gives
\[
 A_k\ge2A_{k-1}+c_Aw_k,\qquad
 c_A:=2+\delta(A_Q+\gamma).
\tag{5.2}
\]
Since \(P_0=A_0=0\) and
\(2^{l-k}w_k=2^{l-1}\), there are \(2^{l-k}\) level-\(k\) nodes and hence
\[
 P_l=c_P\,l\,2^{l-1},\qquad
 A_l\ge c_A\,l\,2^{l-1}.
\tag{5.3}
\]

The depot contributes \(2w_l=2^l\) to the natural a-posteriori tour.
Conversely, the input/output interface and the private return normalize every
closed tour to an input-to-output service plus one depot departure and one
return.  Thus
\[
\begin{aligned}
 \operatorname{OPT}_{\rm post}(I_l)
   &\le 2^l+c_P\,l\,2^{l-1},\\
 \operatorname{OPT}_{\rm adapt}(I_l)
   &\ge 2^l+c_A\,l\,2^{l-1}.
\end{aligned}
\tag{5.4}
\]
Therefore
\[
 \liminf_{l\to\infty}
 \frac{\operatorname{OPT}_{\rm adapt}(I_l)}
      {\operatorname{OPT}_{\rm post}(I_l)}
 \ge \frac{c_A}{c_P}.
\tag{5.5}
\]
Finally,
\[
\begin{aligned}
 3c_A-4c_P
 &=3\bigl(2+\delta(A_Q+\gamma)\bigr)
   -4\bigl(\tfrac32+\delta(P_Q+\gamma)\bigr)\\
 &=\delta\bigl(3A_Q-4P_Q-\gamma\bigr)>0
\end{aligned}
\tag{5.6}
\]
by (2.3).  Hence
\[
       \frac{c_A}{c_P}>\frac43.
\tag{5.7}
\]

## 6. Audit points

- The tournament is deterministic and known to the adaptive policy.  Only
  client activations are random, and they are mutually independent.
- Every metric move is expanded before accounting; inactive clients remain
  available as transit vertices.
- The tournament feedback lower bound is protected from ambient shortcuts
  because leaving its chamber costs on scale \(w_k\), not
  \(\delta w_k\).
- Calling the chamber early is explicitly included: a subsequent return to
  unfinished work pays \(2w_k\), while the entire chamber diameter is less
  than \(w_k/20\).
- Calling a child in several pieces is included through (3.2); the private
  return of a child has exactly the next parent scale.
- A second global lap is handled by the same piece hierarchy as in the
  directed-triangle proof.  It need not literally use every descendant's
  private-return arc: leaving a whole descendant module and re-entering it
  creates an additional module piece at its parent.  Equation (3.2) then
  inserts the descendant private return as a *virtual connector*, and the
  parent's adjusted top-level accounting pays that connector.  Thus no
  physical ancestor arc is charged at two recursive levels.

The only nonstandard feature relative to the original recursive proof is the
directed input-bank/sole-output interface.  It is precisely what makes the
strict local tournament gap composable.
