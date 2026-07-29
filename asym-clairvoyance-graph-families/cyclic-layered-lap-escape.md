# Lap indices and the descendant escape in cyclic layered lifts

## 1. Setting

Let
\[
                 C_0,C_1,\ldots,C_{k-1}
\]
be client columns, indexed modulo \(k\).  The generating digraph has unit
arcs
\[
             H_j\subseteq C_j\times C_{j+1}
                 \qquad(j\in\mathbb Z_k)
\tag{1.1}
\]
and no shorter arcs.  Assume that the product semigroup is strongly
connected.  Equivalently, fallback arcs may be added at a scale so large
that they do not enter any of the bounds below.  Distances are directed
shortest-path distances.  Clients have independent activation bits.

This includes cyclic high-girth lifts, algebraic lifts, sparse switching
interfaces, and periodic products of bipartite incidence graphs.

## 2. Exact lap identity

Expand every metric move of a tour into a shortest generating walk.  Every
unit arc raises the integer column lift by one.  A closed walk returns to
its original column, so its length is
\[
                              kL
\tag{2.1}
\]
for an integer \(L\), the number of laps.

Every service event at a client in \(C_j\) receives an integer lift
\[
                              j+k\ell.
\tag{2.2}
\]
These lifts are strictly increasing in the service order.  In particular,
one lap contains at most one service event from a given column, and
\[
                    L\ge\max_j |A\cap C_j|.
\tag{2.3}
\]

Fix a cut interface \(H_c\), and split the expanded tour immediately after
each crossing of this cut.  Between consecutive crossings, the serviced
clients form a chain in the acyclic reachability order obtained by
linearizing
\[
                     C_{c+1},C_{c+2},\ldots,C_c .
\]
Thus an \(L\)-lap tour gives an \(L\)-chain cover of every active
realization in this cut-open reachability poset.  The correct cut potential
is therefore a *global transitive path-cover quantity*.  It is not the sum
of maximum-matching deficiencies of the immediate interfaces.

The converse need not hold for a sparse \(H_c\): endpoints of two
cut-open chains may require more than one lap to reconnect.  This is why a
valid positive proof needs connector control in addition to an interface
matching calculation.

## 3. Descendant lookahead is legal and lap-free

For \(u\in C_j\), let
\[
 D_t(u):=\{v\in C_{j+t}:u\leadsto v
                 \text{ by exactly }t\text{ unit arcs}\}.
\tag{3.1}
\]
Suppose the current position is the just-called active client \(u\).
A causal policy may call the still-uncalled clients of \(D_t(u)\) in any
fixed order.

* Every inactive answer reveals one activation bit without movement.
* On the first active answer \(v\), the policy moves from \(u\) to \(v\) in
  at most \(t\) unit steps.
* If the interval from \(C_j\) to \(C_{j+t}\) does not cross the chosen lap
  cut, this continuation creates no new lap.

Hence the exact failure probability of this continuation bank is
\[
             \Pr[D_t(u)\text{ has no uncalled active client}]
             =\prod_{v\in D_t(u)\ {\rm uncalled}}(1-p_v).
\tag{3.2}
\]
For uniform activation probability \(p\),
\[
             \Pr[\text{failure}]
                 \le \exp(-p\,|D_t(u)_{\rm uncalled}|).
\tag{3.3}
\]

This is not an illicit edge-availability argument.  All paths exist in the
fixed metric, inactive intermediate vertices remain usable as transit, and
only the called endpoint \(v\) is a service event.

### High-girth quantitative form

If every interface is \(d\)-regular and the lift is collision-free through
depth \(t\), then
\[
                              |D_t(u)|=d^t.
\tag{3.4}
\]
At the standard immediate-interface scaling \(p=c/d\),
\[
             \Pr[\text{no active \(t\)-step continuation}]
             \le \exp(-c\,d^{t-1}).
\tag{3.5}
\]
Already \(t=2\) makes the failure probability \(e^{-cd}\).  Thus high girth
and expansion make an immediate matching failure *less* likely to force a
new lap: the policy skips the contested column, remains in the same lap,
and serves the skipped active clients on a later pass.

More generally, for constant degree and many columns, choosing
\(t\asymp\log_d(1/p)\) exposes a constant-success descendant bank.  Any
per-interface lower bound must explicitly forbid or charge these longer
same-lap continuations.

## 4. A finite two-lap counterexample to additive deficiency

The failure of additive interface charging already occurs in a six-vertex
strongly connected quotient.  Put
\[
\begin{aligned}
 C_0&=\{a_0,a_1\},&
 C_1&=\{b_0,b_1\},&
 C_2&=\{c_0,c_1\}.
\end{aligned}
\]
Let \(H_0,H_1\) be the identity matchings
\[
                         a_i\to b_i\to c_i,
\]
and let \(H_2\) be complete:
\[
                         c_i\to a_j
                    \qquad(i,j\in\{0,1\}).
\]
The cyclic product is strongly connected.

Consider the active realization
\[
                              A=\{a_0,b_1,c_0\}.
\tag{4.1}
\]
At the three immediate active interfaces, the maximum matching sizes are
\[
                              (0,0,1).
\tag{4.2}
\]
Thus two of the three unit demands are deficient.  Nevertheless the closed
walk
\[
 a_0\to b_0\to c_0\to a_1\to b_1\to c_1\to a_0
\tag{4.3}
\]
has length six, exactly two laps, and serves all of \(A\).  The vertices
\(b_0,a_1,c_1\) are merely transit vertices.

The same escape is causal.  From active \(a_0\), call \(c_0\) directly.
If it is active, the move follows \(a_0,b_0,c_0\) and remains in the first
lap.  From \(c_0\), call \(b_1\); the move follows \(c_0,a_1,b_1\) and uses
the second lap.  Finally return through \(c_1,a_0\).  Inactive lookahead
answers would have caused no movement.

Thus
\[
  \text{one base lap}+\sum_j
     \text{(immediate-interface deficiency at \(j\))}
\tag{4.4}
\]
predicts three laps, while the true value is two.  The two local
deficiencies are repaired by one globally shifted lap.  Parallel copies
only make this batching more pronounced.

## 5. Consequences for a proposed lower bound

A proof based on immediate committed matchings has to establish all of the
following before its deficiencies can be summed.

1. **No transitive skip:** an active source with no usable active neighbor
   in \(C_{j+1}\) must also lack an affordable active descendant in every
   later column of the same lap.
2. **Distinct lap charge:** deficits at different interfaces must require
   distinct additional laps, rather than different phase shifts of one
   common lap.
3. **Lookahead robustness:** calling later-column descendants before the
   skipped clients must be included in the causal lower bound.
4. **Connector control:** a cut-open path-cover lower bound must include the
   cost of reconnecting its paths through the sparse cut interface.

Ordinary high-girth, Ramanujan, incidence, and algebraic lifts violate the
first condition in the natural sparse activation regime by (3.5).  The
finite quotient in Section 4 violates the second even without asymptotics.

## Verdict

**The intended additive per-interface deficiency is refuted.**  Lap indices
are a sound global accounting device, but the quantity they charge is a
cut-open transitive path cover.  Immediate matching failures can be skipped
through inactive transit vertices and several such failures can be batched
into one additional lap.  A positive cyclic lift would need a new global
potential on lap assignments; summing local committed-matching deficiencies
does not provide one.
