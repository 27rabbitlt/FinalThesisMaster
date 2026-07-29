# Failed balanced serial-bank composition

> **Audit status: invalid.**  The quotient run-count and aggregate reset
> accounting in Sections 3--5 are useful, but the chamber inequality (2.4)
> is false for separated chamber pieces.  The affine value
> \(G_D(\delta)\) is a contiguous-trace Bellman value with a rebate on a
> repeated port; it is not an interrupted-service value with a small toll
> per new piece.
>
> A two-piece causal counterexample is immediate.  In the first piece, enter
> at \(a\), serve \(a\), and call \(x\); if \(x\) is active, traverse
> \(a\to x\to b\), and otherwise exit at \(a\).  In the second piece, enter
> at \(b\), serve \(b\), and call \(y\); if \(y\) is active, traverse
> \(b\to y\to a\), and otherwise exit at \(b\).  Thus
> \[
>   \mathbb E Z_D
>   =\frac{13}{25}\,5+\frac15\,8
>   =\frac{21}{5},
>   \qquad N_D=2.
> \]
> At \(\delta=1/100\), the left side claimed in (2.4) is \(4.21\), far
> below \(G_D(1/100)=6.5948\).  Consequently (5.5), the adaptive recurrence,
> and the claimed theorem do not follow.  The rest of the note is retained
> only to document the precise shared-lap obstruction.

## 0. Correct obstruction theorem

Although the claimed construction is false, the audit gives a sharp no-go
statement for this entire serial-bank strategy.

Let \(H(t)\) denote the least adaptive interrupted-service value of one
unscaled \(D\) chamber when a toll \(t\) is charged for every piece after the
first.  Two legal policies give
\[
       H(t)\leq
       \min\left\{\frac{33}{5},\,\frac{21}{5}+t\right\}.
\tag{0.1}
\]
The first term is the optimal contiguous adaptive service.  The second is
the explicit two-piece policy in the audit box above.

Consider any recursive serial bank with child scale \(L_{k-1}\), parent reset
length \(L_k\), branching \(b\), and copies of \(D\) of total scale
\(\epsilon L_k\) per parent.  Put \(q=bL_{k-1}/L_k\).  One parent lap can
create one extra piece in every child and in every chamber.  Therefore any
valid aggregate reconnection accounting with chamber toll \(t\) must have
\[
       1-q\geq\epsilon t.
\tag{0.2}
\]
After summing the geometric recurrence and closing the root by a return of
length at least \(L_l\), a certificate above \(4/3\) would require
\[
       \frac{\epsilon\bigl(3H(t)-4P_D\bigr)}{1-q}>1.
\tag{0.3}
\]
By (0.2), a necessary condition is
\[
       3H(t)-4P_D>t.
\tag{0.4}
\]
But (0.4) is impossible for every \(t\geq0\).  Indeed, if
\[
       t\leq\frac{893}{375},
\tag{0.5}
\]
then the second policy in (0.1) gives
\[
       H(t)\leq\frac{21}{5}+t
       \leq\frac43P_D,
\]
so the left side of (0.4) is nonpositive.  If
\(t>893/375\), the first policy in (0.1) gives
\[
       3H(t)-4P_D
       \leq 3\cdot\frac{33}{5}-4\cdot\frac{617}{125}
       =\frac7{125}
       <t.
\tag{0.6}
\]

Thus increasing the branching factor does not rescue a serial bank of
independent copies of \(D\).  A shared lap batches all two-piece repairs, and
the toll needed to suppress that policy is much larger than the chamber's
strict \(3A-4P\) surplus.  A successful high-branching construction must
couple the chamber states so that the two-piece repairs cannot all be
performed in the same lap; additive replication is insufficient.

## 1. Result

This note gives a finite directed-metric instance with
\[
   \frac{\operatorname {OPT}_{\rm adapt}}
        {\operatorname {OPT}_{\rm post}}>\frac43.
\]
The construction uses the exact two-selector chamber
\[
D=\begin{pmatrix}
0&5&4&7\\
3&0&5&7\\
4&1&0&3\\
1&6&5&0
\end{pmatrix},
\qquad
  (p_a,p_b,p_x,p_y)=\left(1,1,\frac{13}{25},\frac15\right),
\tag{1.1}
\]
whose row and column order is \(a,b,x,y\).

The composition point is a **balanced serial bank**.  A lap through a bank
can give every child and the chamber one additional service piece.  We do not
charge the lap once to every item.  Instead, the single lap has length exactly
the *sum* of all item reconnection credits.  This removes the shared-lap
double charge that invalidates a naive parallel composition.

All zero lengths below are only used to make the accounting transparent.
Section 8 replaces them by an explicitly bounded positive number.  Thus the
final instance is a genuine directed metric.

## 2. The chamber lemma

The two stochastic clients in (1.1) are independent.  Their four activation
states \(\varnothing,x,y,xy\) have probabilities
\[
       \frac1{125}(48,52,12,13).
\tag{2.1}
\]
The least fixed-realization service that starts and ends at freely chosen
ports in \(\{a,b\}\), while visiting both permanent ports and all active
selectors, has costs
\[
       (3,5,8,9).
\tag{2.2}
\]
Consequently the expected a-posteriori open cost is
\[
       P_D=\frac{48\cdot3+52\cdot5+12\cdot8+13\cdot9}{125}
          =\frac{617}{125}.
\tag{2.3}
\]

We also need the interrupted adaptive version.  Split an ambient execution
at the boundary of a copy of \(D\).  A chamber piece is a maximal internal
portion containing at least one chamber-service event.  Let \(N_D\) be the
number of pieces and \(Z_D\) the cost of all arcs internal to the chamber.
Exterior activations and policy randomization are independent side
information for the projected chamber policy.

For \(0\leq\delta\leq5/6\), the four-state Bellman recursion gives
\[
 \mathbb E Z_D+\delta\,\mathbb E(N_D-1)
 \ \geq\
 G_D(\delta):=\frac{33}{5}-\frac{13}{25}\delta .
\tag{2.4}
\]
More generally, the exact relaxed Bellman value is
\[
 \min\left\{
       \frac{33}{5}-\frac{13}{25}\delta,\,
       \frac{167}{25}-\frac{77}{125}\delta
     \right\}.
\tag{2.5}
\]
Here a repeated chamber entrance is represented in the Bellman recursion by
moving to an already served port and receiving a rebate \(\delta\).  Joining
the chamber pieces in their original causal order therefore gives (2.4).
This also proves the bound when the exterior behavior depends on earlier
chamber observations: pre-sample all exterior activation bits and a policy
seed, which are independent of \(x,y\), and reveal those coordinates only
when the projected execution calls them.

For the value used below, put
\[
       \eta:=\frac1{100}.
\tag{2.6}
\]
Then
\[
 G:=G_D(\eta)=\frac{16487}{2500},
 \qquad
 3G-4P_D=\frac{101}{2500}>0.
\tag{2.7}
\]

For completeness, (2.3)--(2.5) are finite exact calculations, not
asymptotic or computational assumptions.  One runs the Bellman recursion on
the current vertex, the subset of \(\{a,b\}\) already served, and the subset
of \(\{x,y\}\) not yet called.  The four terminal active-set path values are
(2.2); the two possible first effective port strategies give the two affine
expressions in (2.5).  Their crossover is \(5/6\).

## 3. Parameters

Set
\[
 \epsilon:=\frac1{100},\qquad
 b:=100,\qquad
 R:=\frac{1\,000\,000}{9999},\qquad
 L_k:=R^k \quad(k\geq0).
\tag{3.1}
\]
The balance identity is
\[
       bL_{k-1}+\eta\epsilon L_k=L_k,
\tag{3.2}
\]
because
\[
       \frac bR+\eta\epsilon
       =\frac{9999}{10000}+\frac1{10000}=1.
\tag{3.3}
\]
Write
\[
       q:=\frac bR=\frac{9999}{10000}.
\tag{3.4}
\]

## 4. Recursive two-terminal modules

The module \(C_0\) has an input \(s_0\), an output \(t_0\), and one permanent
client \(z_0\).  Add zero-length arcs
\[
       s_0\to z_0,\quad z_0\to t_0,\quad s_0\to t_0
\tag{4.1}
\]
and a reset arc \(t_0\to s_0\) of length \(L_0=1\).

For \(k\geq1\), take \(b\) disjoint copies
\[
       C_{k-1}^{(1)},\ldots,C_{k-1}^{(b)}
\]
and one copy \(Q_k\) of the chamber (1.1), scaled by
\(\epsilon L_k\).  Add fresh module input and output vertices \(s_k,t_k\)
and fresh chamber boundary vertices \(u_k,v_k\).  Arrange the items in the
serial order
\[
 C_{k-1}^{(1)},\,C_{k-1}^{(2)},\,\ldots,\,
 C_{k-1}^{(b)},\,Q_k.
\tag{4.2}
\]
Add zero-length forward arcs
\[
\begin{aligned}
 s_k&\to s(C_{k-1}^{(1)}),\\
 t(C_{k-1}^{(i)})&\to s(C_{k-1}^{(i+1)})
       &&(1\leq i<b),\\
 t(C_{k-1}^{(b)})&\to u_k,\\
 u_k&\to a_k,\quad u_k\to b_k,\\
 a_k&\to v_k,\quad b_k\to v_k,\quad u_k\to v_k,\\
 v_k&\to t_k .
\end{aligned}
\tag{4.3}
\]
Finally add the sole top-level reset
\[
       t_k\to s_k
       \quad\hbox{of length }L_k.
\tag{4.4}
\]
Each child already has a zero-length input-to-output bypass by induction, so
the entire forward chain is available for transit.  The four chamber
vertices carry all arcs of (1.1), multiplied by \(\epsilon L_k\).

The only clients added at level \(k\) are \(a_k,b_k,x_k,y_k\).
The ports \(a_k,b_k\) are permanent, while \(x_k,y_k\) have probabilities
\(13/25\) and \(1/5\).  All activation bits in all modules are mutually
independent.  Boundary vertices may be regarded as Steiner transit vertices,
or equivalently as probability-zero clients.

The generating graph is strongly connected.  Its top-level quotient is a
directed serial cycle, and all child and chamber subgraphs are strongly
connected.

### Shortest-path audit

Leaving a child and returning to it through the parent costs at least \(L_k\),
whereas its own reset costs \(L_{k-1}<L_k\).  Thus the parent does not shorten
any child distance used in its local-service bound.

An exterior detour from the chamber output back to its input costs \(L_k\).
Every distance in (1.1) is at most
\[
       7\epsilon L_k=\frac7{100}L_k<L_k.
\tag{4.5}
\]
Hence leaving the chamber cannot shorten any chamber distance.  This proves
the required closure statement by induction.

## 5. Open-service recurrences

Let \(P_k\) be the expected a-posteriori input-\(s_k\) to output-\(t_k\)
service cost in \(C_k\).  Let \(A_k\) be its least expected adaptive cost.
The boundary arcs are free at this stage.

Serving every child once in serial order and then taking an optimal
free-boundary chamber service gives
\[
       P_k\leq bP_{k-1}+\epsilon P_D L_k,
       \qquad P_0=0.
\tag{5.1}
\]

We next prove
\[
       A_k\geq bA_{k-1}+\epsilon G L_k,
       \qquad A_0=0.
\tag{5.2}
\]
Expand every metric movement into a fixed shortest generating-graph path.
Let \(h\) be the number of occurrences of the reset arc (4.4).  Cut the
execution into service-containing pieces of the \(b+1\) top-level items.
Between two reset occurrences, the ranks in (4.2) are nondecreasing.
Consequently, each item has at most one piece in each of the \(h+1\)
nondecreasing runs.  If \(N_i\) is the number of pieces of child \(i\) and
\(N_Q\) the number of chamber pieces, then, pathwise,
\[
       \sum_{i=1}^b(N_i-1)\leq bh,
       \qquad
       N_Q-1\leq h.
\tag{5.3}
\]
Every item contains a permanent client, so its number of pieces is at least
one.  Inactive exterior calls cause no movement and no piece, and therefore
do not affect (5.3).

Let \(Z_i\) be the total internal cost in child \(i\).  Consecutive child
pieces can be joined by its own reset \(t_{k-1}\to s_{k-1}\), of length
\(L_{k-1}\).  Projecting the ambient adaptive execution into that child, with
all exterior randomness pre-sampled as an independent seed, gives
\[
       \mathbb E Z_i+
       L_{k-1}\mathbb E(N_i-1)\geq A_{k-1}.
\tag{5.4}
\]
The scaled chamber lemma gives
\[
       \mathbb E Z_Q+
       \eta\epsilon L_k\mathbb E(N_Q-1)
       \geq \epsilon G L_k.
\tag{5.5}
\]

The actual top-level reset cost is \(hL_k\).  By (3.2) and (5.3), it pays all
of the reconnection credits **once and jointly**:
\[
\begin{aligned}
 &L_{k-1}\sum_{i=1}^b(N_i-1)
       +\eta\epsilon L_k(N_Q-1)\\
 &\hspace{2cm}\leq
 h\bigl(bL_{k-1}+\eta\epsilon L_k\bigr)
 =hL_k.
\end{aligned}
\tag{5.6}
\]
Adding (5.4) and (5.5), and using (5.6), proves (5.2).  Notice that a single
global lap really can split all \(b+1\) items.  The proof assigns that lap
one aggregate budget \(L_k\), not \(b+1\) copies of the same cost.

## 6. Solving the recurrences

Divide (5.1)--(5.2) by \(L_k\) and use \(b/R=q\).  Induction gives
\[
\begin{aligned}
 \frac{P_l}{L_l}
   &\leq \epsilon P_D\sum_{j=0}^{l-1}q^j
    =\frac{P_D}{\eta}(1-q^l),\\
 \frac{A_l}{L_l}
   &\geq \epsilon G\sum_{j=0}^{l-1}q^j
    =\frac{G}{\eta}(1-q^l).
\end{aligned}
\tag{6.1}
\]

## 7. Closing the root and exceeding \(4/3\)

Set \(l=10000\).  Add a depot \(r\) and arcs
\[
       r\to s_l,\qquad t_l\to r
\tag{7.1}
\]
of length \(L_l/2\) each.  The path through the depot from \(t_l\) to \(s_l\)
ties the root reset and therefore creates no shortcut.

The natural a-posteriori tour has expected cost at most
\[
       \operatorname {OPT}_{\rm post}
       \leq L_l+P_l.
\tag{7.2}
\]
Conversely, expand any adaptive depot tour.  Remove its first depot-to-input
arc and final output-to-depot arc.  Replace every intermediate depot excursion
by the equal-length root reset.  The result is a legal adaptive
input-to-output service of \(C_l\).  Therefore
\[
       \operatorname {OPT}_{\rm adapt}
       \geq L_l+A_l.
\tag{7.3}
\]

Now
\[
 q^l=
 \left(1-\frac1{10000}\right)^{10000}
 <e^{-1}<\frac12.
\tag{7.4}
\]
Using (2.7) and (6.1)--(7.3),
\[
\begin{aligned}
 \frac{
 3\operatorname {OPT}_{\rm adapt}
 -4\operatorname {OPT}_{\rm post}}
 {L_l}
 &\geq
 \frac{3G-4P_D}{\eta}(1-q^l)-1\\
 &>
 \frac{101}{25}\cdot\frac12-1
 =\frac{51}{50}>0.
\end{aligned}
\tag{7.5}
\]
It follows that the closed instance has ratio strictly larger than \(4/3\).

## 8. Positive-edge perturbation

The zero-length presentation above is a convenient directed semimetric.
Here is a quantitative conversion to a genuine positive directed metric.

Let \(N\) be the number of clients and \(V\) the number of generating-graph
vertices in the depth-\(l\) closed construction.  They are finite; explicitly,
before adding the depot,
\[
\begin{aligned}
 N_0&=1,&N_k&=bN_{k-1}+4,\\
 V_0&=3,&V_k&=bV_{k-1}+8.
\end{aligned}
\tag{8.1}
\]
Replace every zero-length generating arc by the same positive rational
\(\tau\), where
\[
 0<\tau<
 \frac{51L_l}
 {200\,(N+1)(V-1)}.
\tag{8.2}
\]
All other arc lengths and all probabilities remain unchanged.

Increasing arc lengths cannot decrease the adaptive optimum.  On the other
hand, a simple shortest path uses at most \(V-1\) arcs, and a shortcut tour
through at most \(N\) clients uses at most \(N+1\) metric moves.  Hence the
old a-posteriori tour increases by at most
\[
       (N+1)(V-1)\tau.
\tag{8.3}
\]
Combining (7.5), (8.2), and (8.3), the perturbed instance still has
\[
       3\operatorname {OPT}_{\rm adapt}
       -4\operatorname {OPT}_{\rm post}>0.
\tag{8.4}
\]
Every generating arc is now positive and the graph remains strongly
connected, so its directed shortest-path distances form a genuine directed
metric.  All probabilities and all arc lengths may be chosen rational.

## 9. Failure audit

- **One global lap.**  A lap may create one extra piece in every child and in
  the chamber.  Inequality (5.6) charges the lap once, against the sum of all
  credits.  There is no replicated charge.
- **Arbitrary interleaving.**  The proof uses the number of monotone runs of
  the expanded quotient trace, not an assumption that items are served
  contiguously.
- **Remote calls.**  An inactive remote call produces neither movement nor a
  piece.  An active call into an earlier-ranked item necessarily expands
  through the reset arc.
- **Exterior information.**  It is independent of a projected child or
  chamber activation vector and is therefore only an independent randomized
  seed.  It does not invalidate (5.4) or (5.5).
- **Shortest-path closure.**  External chamber detours cost \(L_k\), versus
  chamber diameter at most \(7\epsilon L_k\).  External child detours cost
  \(L_k>L_{k-1}\).
- **Depot batching.**  Every intermediate depot excursion has exactly the
  root reset cost and is included when the closed tour is normalized to an
  open root service.
