# Common-payload mode paths: layer rigidity and the inevitable suffix escape

## 1. Intended mechanism

The proposed chamber has a common permanent payload \(U\), modes
\(j=1,\ldots,m\), and a selector \(s_j\) for each mode.  Mode \(j\) has a
cheap Hamiltonian path
\[
             P_j:\quad r,\ U\text{ in mode-}j\text{ order},\ s_j,
\tag{1}
\]
of length \(L\).  A selector is meant to be cheap only after its matching
payload order.  The desired lower-bound claim is that a causal policy must
either

* query an active selector before serving \(U\), thereby traversing the
  payload once as unserved transit and once again for service; or
* guess a mode, with every wrong guess requiring a second full payload
  traversal.

Both claims conflict with a static layered shortest-path metric.

## 2. Strict layering makes the payload order unique

### Proposition 1

Suppose the generating graph, after deleting depot-reset arcs, has a level
function \(\ell\) which strictly increases on every arc.  If every \(P_j\)
is a directed Hamiltonian path through the same payload \(U\), then all
\(P_j\)'s visit \(U\) in the same level order.

### Proof

Along a directed path the level strictly increases.  Hence two payload
vertices can occur on a Hamiltonian path only in increasing order of their
levels.  Since a Hamiltonian path contains every payload vertex, their
levels are distinct and their relative order is fixed by \(\ell\).
\(\square\)

In particular, all modes have the same last payload client \(u^*\).
If \(s_j\) is cheaply appended after the mode-\(j\) path, then it is cheaply
reachable from \(u^*\) for every \(j\).  A causal policy serves the common
payload in its unique order and only then calls all selectors.  Inactive
calls cause no movement.  Under the exact-one ideal it reaches the unique
active selector through its cheap append edge and attains the posterior
route.  Thus literal global layering gives gap one, not a mode-guessing
problem.

Allowing several payload clients on one level does not help: a strictly
level-increasing path cannot visit two of them.  Within-level arcs or
mode-dependent backward arcs abandon the stated global layering.

## 3. Different payload permutations create suffix shortcuts

Drop strict layering and allow the paths \(P_j\) to use different
permutations.  Let \(t_k\) be the final payload vertex of \(P_k\).  Since
the payload is common, \(t_k\) also occurs somewhere on every \(P_j\).
The suffix of \(P_j\) beginning at that occurrence is a directed walk
\[
                  t_k\leadsto s_j.
\]
Shortest-path closure therefore gives
\[
 d(t_k,s_j)
 \le
 \operatorname{length}\bigl(
      \text{suffix of }P_j\text{ from }t_k\text{ to }s_j
   \bigr).
\tag{2}
\]

This path remains available after all payload clients have been served.
Serving a client removes a service obligation; it does not delete the
vertex or any incident arc.  Consequently a policy may:

1. serve \(U\) once in the order \(P_k\), stopping at \(t_k\);
2. call selectors only afterward; and
3. on the active answer \(s_j\), use the suffix in (2).

Thus a wrong mode does **not** force a fresh \(r\)-to-payload traversal.
The unavoidable shortcut is present even if all other possible
mode-switching walks are somehow excluded.

Adding a private terminal \(q_j\) after the payload does not repair this:
the suffix of \(P_j\) is then a walk \(t_k\leadsto q_j\leadsto s_j\).
Making \(q_j\) a one-way reset state also does not force the policy to enter
the guessed \(q_k\); after the last payload service it can call the
selectors directly from \(t_k\).

## 4. Equal-layer paths have a half-traversal causal policy

The suffix obstruction has an exact average in the canonical equal-length
case.  Suppose:

* \(U=\{t_1,\ldots,t_m\}\), and \(P_j\) is a permutation of \(U\) ending in
  \(t_j\);
* every step of \(P_j\), including the append step to \(s_j\), has the same
  length;
* the total from the first payload vertex through \(s_j\) is \(L\), up to
  a common \(o(L)\) entry/return term.

Choose \(K\) uniformly from \([m]\), serve the payload in the order \(P_K\),
and then query the selectors in an independent uniform order.  In the
exact-one ideal, let \(J\) be the active mode.  Conditional on \(J=j\), the
position of \(t_K\) in the permutation \(P_j\), averaged over uniform \(K\),
is uniform on the \(m\) payload positions.  Hence the average suffix in
(2) has length
\[
                         \frac L2+o(L).
\tag{3}
\]
The policy therefore has expected cost at most
\[
                         \frac32L+o(L)
\tag{4}
\]
against posterior cost \(L+o(L)\).  A claimed two-traversal causal lower
bound is impossible.

## 5. Independent selectors dilute the suffix loss below \(4/3\)

Here is the calculation for the natural independent replacement of the
illegal one-hot state.  Activate the \(m\) selectors independently with
probability \(p\), and let
\[
       Z\sim\operatorname{Binomial}(m,p),\qquad
       q=\Pr[Z=0]=(1-p)^m,\qquad \lambda=mp.
\]
Grant the construction its intended reset separation: after an active
selector is served, the route returns to \(r\), and every further active
selector costs one full mode traversal \(L+o(L)\).  Then the posterior
value is at least, and in the intended canonical chamber equals,
\[
              P=L\,\mathbb E\max\{1,Z\}+o(L)
               =L(q+\lambda)+o(L).
\tag{5}
\]

Use the randomized payload endpoint and selector order from Section 4.
Conditional on \(Z\ge1\), the first active selector in a uniform query order
is uniform over the active set.  Averaging also over \(K\), (3) bounds its
only extra post-payload movement by \(L/2+o(L)\).  Every later active
selector is paid by the same full traversal already present in (5).
Consequently
\[
 A\le
 P+\frac L2(1-q)+o(L),
\]
and
\[
 \frac AP
 \le
 1+\frac{1-q}{2(q+\lambda)}+o(1).
\tag{6}
\]

For \(m\ge2\),
\[
                5(1-p)^m+2mp\ge3.
\tag{7}
\]
Indeed, with \(x=1-p\), the minimum of
\(5x^m+2m(1-x)\) is attained either at an endpoint or at
\(x=(2/5)^{1/(m-1)}\).  At the interior point it equals
\[
 2\bigl[m-(m-1)x\bigr]\ge3,
\]
because
\[
 (2/5)^{1/(m-1)}
 \le1-\frac1{2(m-1)};
\]
raising the latter inequality to the power \(m-1\) reduces it to
\[
 \frac25\le
 \left(1-\frac1{2(m-1)}\right)^{m-1},
\]
whose right-hand side is at least \(1/2\).

Inequality (7) is equivalent to
\[
                 3(1-q)\le2(q+\lambda).
\]
Substitution in (6) gives
\[
                         \boxed{\frac AP\le\frac43+o(1).}
\tag{8}
\]

This calculation is deliberately favorable to the proposed construction:
it assumes that all non-suffix mode switches and all batching between
different active selectors have been eliminated.  Any additional
shortest-path shortcut only improves the causal policy.

## 6. The state-erasure lemma

The same obstruction can be phrased without assuming that a mode is a
Hamiltonian permutation.  It explains why concatenating many clever
multiport gadgets does not by itself accumulate a causal loss.

### Lemma 2 (common-client state erasure)

Let the current physical vertex be a mode-dependent port \(x_j\), and let
\(v\) be the next active client called by the policy.  If \(v\) is the
same client in every mode, then immediately after this call the physical
state is exactly \(v\), independently of \(j\).  Therefore every
continuation cost from that point depends only on \(v\), the unqueried
set, and the revealed activation history, and not on the port \(x_j\).

### Proof

Calling an active client moves the server to that client's vertex.  A
directed shortest-path metric has no hidden edge state: after the move,
all future distances are the fixed numbers \(d(v,\cdot)\).  The mode can
affect the cost already paid, namely \(d(x_j,v)\), but it cannot affect
any future transition.  \(\square\)

Thus a port label can survive across a block only if the first active
client of the next block is itself mode-specific, say \(v_j\).  There are
then only two possibilities.

1. The \(v_j\)'s are copies of one logical payload item but are distinct
   vertices.  In the standard stochastic-TSP model they are distinct
   clients.  If they are permanent, every route must serve all copies, so
   their cost becomes common baseline and their vertices become reusable
   transit.  If only one copy is active, the construction has introduced
   the same correlated one-hot state that the independent-activation
   model forbids.
2. The \(v_j\)'s are merely nonclient ports.  Then the next common active
   client erases the label by the lemma.  A chain of nonclient ports can
   change the single transition cost into that client, but cannot carry a
   mode into later blocks.

Equivalently, a purported noncommuting entry/exit block must expose its
mode in the identity of an active client, not only in the path used to
reach a common client.  Static vertices and shortest-path closure do not
provide a private automaton state.

## 7. Consequence

A common-payload mode code faces a dichotomy:

* strict global layering fixes one payload order and makes the selectors
  safely queryable afterward; or
* distinct mode permutations generate the suffix escape (2).

For equal-layer Hamiltonian paths, the suffix escape plus the unavoidable
zero/multiple-mode mass of independent activations gives the \(4/3\)
ceiling (8).  A successful route code must therefore use non-Hamiltonian
mode services whose state is encoded by mode-specific active clients and
yet is generated by independent activations.  Merely adding noncommuting
entry and exit ports cannot do this: a common active payload call erases
the port state.
