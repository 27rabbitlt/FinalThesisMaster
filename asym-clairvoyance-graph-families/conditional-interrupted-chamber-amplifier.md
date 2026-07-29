# Conditional amplifier for a fragmentation-robust open chamber

## 0. Status and theorem

This is a complete **transfer theorem**, conditional only on the stated
interrupted-service inequality for one finite chamber.  It does not assert
that such a chamber has already been found.

Let \(Q\) be a finite strongly connected two-terminal directed chamber with
input \(a\), output \(b\), at least one permanent client, and mutually
independent stochastic clients.  In an ambient graph, the only external arcs
entering \(Q\) enter at \(a\), and the only external arcs leaving \(Q\) leave
at \(b\).  Normalize the chamber's generating-arc lengths once and for all.

Write

* \(P\) for its expected a-posteriori \(a\)-to-\(b\) open-service cost;
* \(A\) for its least expected causal **one-piece** \(a\)-to-\(b\)
  open-service cost (entry and exit are accounted as pending generating
  paths, so no voluntary movement is granted); and
* \(H(s)\) for its causal interrupted value
  \[
       H(s):=\inf_\pi
       \mathbb E\bigl[Z_\pi+s(N_\pi-1)\bigr].
  \tag{0.1}
  \]

Here an interrupted policy may serve the chamber in \(N_\pi\geq1\)
service-containing pieces; \(Z_\pi\) is the total cost of generating arcs
internal to \(Q\).  Between pieces it is in an abstract exterior state, and
entry at \(a\) and exit at \(b\) are free.  Calls, observations, and pieces
must still be causal.  Independent side information may be included in the
policy seed.

Assume
\[
                         A>\frac32P
\tag{0.2}
\]
and the fragmentation envelope
\[
               H(s)\geq \min\{A,P+s\}
               \qquad(s\geq0).
\tag{0.3}
\]

### Theorem

Under (0.2)--(0.3), there is a finite standard closed-depot asymmetric
stochastic-TSP instance, with mutually independent client activations and a
strictly positive directed shortest-path metric, for which
\[
       \frac{\operatorname {OPT}_{\rm adapt}}
            {\operatorname {OPT}_{\rm post}}>\frac43.
\tag{0.4}
\]

The construction is explicit once \(Q,P,A\) are given.  The proof permits
arbitrary interleaving of every recursive child and every chamber copy.

## 1. The decisive toll

Put
\[
                         t:=A-P.
\tag{1.1}
\]
By (0.2), \(t>P/2>0\).  A one-piece optimal causal policy gives
\[
                         H(t)\leq A.
\tag{1.2}
\]
The assumed envelope gives the reverse inequality:
\[
 H(t)\geq\min\{A,P+(A-P)\}=A.
\tag{1.3}
\]
Consequently
\[
                         H(t)=A.
\tag{1.4}
\]

The algebra which motivates the construction is
\[
             \frac{t+H(t)}{t+P}
       =\frac{2A-P}{A}
       =2-\frac PA
       >\frac43.
\tag{1.5}
\]
The serial recursion below realizes the two additive \(t\)'s in (1.5) as
one balanced root return, not as a separately charged return for every
copy.

## 2. Parameters

Let \(D\) be the largest finite distance between chamber generating
vertices in the chamber's own directed shortest-path metric.  Choose a
rational
\[
       0<\epsilon<
       \min\left\{\frac1t,\frac1{D+1}\right\}.
\tag{2.1}
\]
Choose any integer \(d\geq2\), and put
\[
       q:=1-\epsilon t,\qquad
       R:=\frac d q,\qquad
       L_k:=R^k\quad(k\geq0).
\tag{2.2}
\]
Then \(0<q<1\), \(R>d\), and
\[
       dL_{k-1}+\epsilon tL_k=L_k.
\tag{2.3}
\]
Equation (2.3) is the exact aggregate reconnection budget for one parent
lap.

## 3. Recursive modules

The depth-zero module \(C_0\) has input \(s_0\), output \(t_0\), and one
permanent client \(z_0\).  Add zero-length forward arcs
\[
                 s_0\to z_0,\qquad z_0\to t_0,
                 \qquad s_0\to t_0
\tag{3.1}
\]
and a reset arc
\[
                         t_0\to s_0
             \quad\hbox{of length }L_0=1.
\tag{3.2}
\]

For \(k\geq1\), take \(d\) disjoint copies
\[
           C_{k-1}^{(1)},\ldots,C_{k-1}^{(d)}
\]
and one disjoint copy \(Q_k\) of \(Q\), with every chamber length multiplied
by \(\epsilon L_k\).  Arrange them in the serial order
\[
       C_{k-1}^{(1)},\ldots,C_{k-1}^{(d)},Q_k.
\tag{3.3}
\]
Add new input \(s_k\), output \(t_k\), and zero-length forward connectors
from \(s_k\) to the first child input, from each child output to the next
child input, from the last child output to \(a(Q_k)\), and from \(b(Q_k)\)
to \(t_k\).  Also add a zero-length \(s_k\)-to-\(t_k\) bypass following
the same serial rail.  Finally add the parent reset
\[
                         t_k\to s_k
             \quad\hbox{of length }L_k.
\tag{3.4}
\]

The chamber copies use fresh independent activation bits, and all module
copies are mutually independent.  The only clients added outside the
chambers are the permanent depth-zero clients.  Inputs, outputs, and rail
vertices are transit vertices.

Every module is strongly connected: its forward rail reaches the output,
and the reset returns to the input.  The only way to leave a serial item and
later re-enter an earlier-ranked item is to use a reset at that item or at
an ancestor.

## 4. Shortest-path audit

All cost arguments are made after expanding each metric movement into a
fixed shortest generating-graph path.

### Children are not shortened

An exterior route which leaves a child \(C_{k-1}\) through its output and
returns to its input must use a reset of length at least \(L_k\).  The child
has its own reset of length \(L_{k-1}<L_k\).  Therefore no exterior path
shortens a child distance used in its local open-service or reconnection
bound.  The same argument applies inductively to every descendant.

### Chambers are not shortened

Every chamber distance is at most
\[
                         \epsilon D L_k<L_k
\tag{4.1}
\]
by (2.1).  Any exterior return from \(b(Q_k)\) to \(a(Q_k)\) uses the parent
reset and costs at least \(L_k\).  Thus the ambient shortest-path closure
does not shorten any internal chamber distance.

### No hidden backward rail

Every zero rail arc goes forward in the order (3.3).  Hence every generating
path that decreases top-level rank contains the parent reset (3.4).
Additional ancestor resets are longer and cannot evade this count.

These statements remain true when a shortest path passes through inactive
or uncalled client locations: transit does not serve a client, and the rank
potential is a property of arcs, not service events.

## 5. Open posterior recurrence

Let \(P_k\) be the expected a-posteriori input-to-output service cost of
\(C_k\).  Let \(A_k\) be the least expected causal one-piece
input-to-output service cost.  The depth-zero values are
\[
                         P_0=A_0=0.
\tag{5.1}
\]

Serve every child by an optimal posterior open service, then serve \(Q_k\)
by its realization-optimal open service.  The rail connectors are free, so
\[
                  P_k\leq dP_{k-1}+\epsilon PL_k.
\tag{5.2}
\]

No equality is needed; an upper bound on the posterior denominator is the
useful direction.

## 6. Arbitrary-interleaving adaptive recurrence

Consider an arbitrary causal open execution of \(C_k\), and expand all
metric moves.  Let \(h\) be the number of occurrences of the parent reset
(3.4).  Cut the service events belonging to each top-level item into maximal
service-containing pieces.  Write

* \(N_i\) and \(Z_i\) for the piece count and internal cost of child \(i\);
* \(N_Q\) and \(Z_Q\) for the corresponding chamber quantities.

Between two parent resets, top-level ranks never decrease.  Therefore every
item has at most one piece in each of the \(h+1\) forward runs.  Since every
item contains a permanent client,
\[
       \sum_{i=1}^d(N_i-1)\leq dh,\qquad
                         N_Q-1\leq h.
\tag{6.1}
\]

Joining consecutive projected child pieces by the child's own reset gives a
legal one-piece causal service of that child.  Exterior activation bits and
the ambient policy's private randomness can be pre-sampled as an independent
seed.  They are revealed to the projected simulation only when the ambient
execution reveals them.  Thus
\[
       \mathbb E Z_i+
       L_{k-1}\mathbb E(N_i-1)\geq A_{k-1}.
\tag{6.2}
\]

The same projection into \(Q_k\), now using the definition (0.1), gives
\[
       \mathbb E Z_Q+
       \epsilon tL_k\,\mathbb E(N_Q-1)
             \geq\epsilon H(t)L_k
             =\epsilon AL_k.
\tag{6.3}
\]
This is precisely where fragmentation robustness is used.  The inequality
already allows the policy to leave the chamber, learn independent exterior
bits, and re-enter later.

The actual parent-reset cost \(hL_k\) pays all local reconnection credits
once and jointly.  Indeed, (2.3) and (6.1) give, pathwise,
\[
\begin{aligned}
 &L_{k-1}\sum_{i=1}^d(N_i-1)
       +\epsilon tL_k(N_Q-1)\\
 &\hspace{2cm}\leq
 h\bigl(dL_{k-1}+\epsilon tL_k\bigr)
 =hL_k.
\end{aligned}
\tag{6.4}
\]
Adding (6.2)--(6.4) proves
\[
                         A_k
       \geq dA_{k-1}+\epsilon AL_k.
\tag{6.5}
\]
No child or chamber was assumed to be served contiguously.  A single parent
lap may split all \(d+1\) items, and (6.4) charges that lap only once.

## 7. Solving the recurrences

Since
\[
                         \frac{dL_{k-1}}{L_k}=q,
\tag{7.1}
\]
induction in (5.2) and (6.5) yields
\[
\begin{aligned}
 \frac{P_l}{L_l}
   &\leq\epsilon P\sum_{j=0}^{l-1}q^j
    =\frac Pt(1-q^l),\\
 \frac{A_l}{L_l}
   &\geq\epsilon A\sum_{j=0}^{l-1}q^j
    =\frac At(1-q^l).
\end{aligned}
\tag{7.2}
\]

The branching factor \(d\) affects the finite size and scale ratio, but not
the limiting constants.  Its role is to let one parent lap carry the
aggregate child reconnection budget in (2.3).

## 8. Closing the root

Add a depot \(r\) with exactly two incident generating arcs,
\[
             r\to s_l,\qquad t_l\to r,
\tag{8.1}
\]
each of length \(L_l/2\).  The depot path from \(t_l\) to \(s_l\) has length
\(L_l\), tying the root reset rather than shortening it.

The posterior service from (5.2) gives
\[
       \operatorname {OPT}_{\rm post}
                         \leq L_l+P_l.
\tag{8.2}
\]

Conversely, expand an arbitrary adaptive depot tour.  Remove its first
depot-to-input half-arc and its final output-to-depot half-arc.  Replace
every intermediate depot excursion by the equal-length root reset.  What
remains is a legal causal input-to-output service of \(C_l\).  Hence
\[
       \operatorname {OPT}_{\rm adapt}
                         \geq L_l+A_l.
\tag{8.3}
\]

Combining (7.2)--(8.3),
\[
\begin{aligned}
 \frac{
 3\operatorname {OPT}_{\rm adapt}
 -4\operatorname {OPT}_{\rm post}}
 {L_l}
 &\geq
 -1+\frac{3A-4P}{t}(1-q^l).
\end{aligned}
\tag{8.4}
\]
The strict chamber condition (0.2) is equivalent to
\[
                3A-4P>A-P=t.
\tag{8.5}
\]
Thus
\[
             0<
       1-\frac{t}{3A-4P}<1.
\tag{8.6}
\]
Choose any finite depth \(l\) satisfying
\[
                         q^l
       <1-\frac{t}{3A-4P}.
\tag{8.7}
\]
Then the right side of (8.4) is strictly positive, proving
\[
       3\operatorname {OPT}_{\rm adapt}
       >4\operatorname {OPT}_{\rm post}.
\tag{8.8}
\]

As \(l\to\infty\), the certified ratio tends to the value in (1.5).

## 9. Strictly positive metric

The zero forward arcs are only an accounting convenience.  At the finite
depth chosen in (8.7), let
\[
 \Delta:=
 3\bigl(L_l+A_l^{\rm lb}\bigr)
 -4\bigl(L_l+P_l^{\rm ub}\bigr)>0
\tag{9.1}
\]
be the explicit positive margin supplied by (7.2)--(8.4).  Let \(n\) be
the number of clients and \(v\) the number of generating vertices in the
closed construction.

Replace every zero generating arc by the same rational \(\tau>0\), chosen
so that
\[
                         4(n+1)(v-1)\tau<\Delta.
\tag{9.2}
\]
Increasing generating-arc lengths cannot decrease the adaptive optimum.  A
simple shortest path uses at most \(v-1\) arcs, and the displayed posterior
tour has at most \(n+1\) metric moves.  Its cost therefore increases by at
most \((n+1)(v-1)\tau\).  Equation (9.2) preserves (8.8).

All generating arcs are now positive, the graph remains finite and strongly
connected, and its shortest-path distances form a genuine directed metric.
If the chamber data are rational, every parameter and length can be chosen
rational as well.

## 10. Exact missing local object

The theorem shows that the closure problem reduces to a local, falsifiable
condition.  It is enough to find one finite two-terminal chamber satisfying
\[
             A>\frac32P,\qquad
             H(A-P)\geq A.
\tag{10.1}
\]
The full envelope (0.3) is stronger than necessary.

The second inequality says that splitting the chamber cannot recover the
posterior advantage for a toll smaller than the chamber's causal excess.
Cheap singleton-piece policies violate it.  This is why the sparse
tournament chamber, whose one-piece ratio approaches \(3/2\) from below and
whose selectors can be served in separate nearly free pieces, cannot be
amplified by this theorem.

Once (10.1) is proved for a genuine chamber, Sections 2--9 give the requested
standard closed-depot construction without any further geometric or
interleaving lemma.
