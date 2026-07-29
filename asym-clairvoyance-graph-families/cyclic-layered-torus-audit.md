# Cyclic layered lifts: exact lap assignment and the future-column escape

## Outcome

For the directed two-dimensional torus, the posterior problem has an exact
deterministic formulation as a minimum charged cyclic chain cover.  It is not
the sum of independent matching or cut deficits at consecutive columns.

Let
\[
        \mathbb T_k=(\mathbb Z/k\mathbb Z)^2
\]
have unit generating arcs \(u\to u+e_1\) and \(u\to u+e_2\), and use directed
shortest-path distance.  If a closed service order has coordinate winding
numbers \(D_x,D_y\), its length is exactly
\[
                         k(D_x+D_y).                 \tag{0.1}
\]
For a fixed active set \(A\), the minimum \(D_x+D_y\) is the minimum boundary
charge of a cyclically ordered chain cover of \(A\cup\{0\}\); a boundary is
charged once for each coordinate that wraps.  In particular, if \(w(A)\) is
the width in the ordinary product order on the cut-open square, then
\[
                   k\,w(A)\ \leq\ {\rm POST}(A)\
                   \leq 2k\,w(A)                     \tag{0.2}
\]
for nonempty \(A\), with the harmless depot convention described below.

For iid activation \(p=1/k\), the classical random-product-order estimate is
\[
                      \mathbb E w(A)=(2+o(1))\sqrt{k}.
                                                               \tag{0.3}
\]
Thus (0.2) locates the posterior scale at \(k^{3/2}\), but its factor-two
interval is far too wide to prove a \(4/3\) separation.

There is also a rigorous obstruction to the proposed adaptive lower bound.
After reaching an active point \(u\), a causal policy may probe every
uncalled point in an arbitrary northeast successor bank.  Inactive probes
cause no movement, and the first active answer continues without a
coordinate wrap.  A bank of \(M\) vertices fails with probability
\[
                         (1-1/k)^M\le e^{-M/k}.       \tag{0.4}
\]
Consequently, in a layered lift, pooling \(t\) future columns gives failure
at most \(e^{-\Theta(t)}\), and a two-dimensional descendant rectangle often
gives \(e^{-\Theta(k)}\).  A lower bound that charges a positive loss at
every immediate column is therefore false: the policy can skip the
contested column before it starts a new lap.

What remains unproved is a global lower bound after accounting for
competition between different open runs for the same future vertices.
Neither width alone nor a sum of immediate-column matching deficits gives a
ratio above \(4/3\).

## 1. Directed-torus metric

Represent a torus vertex by
\[
                         u=(u_x,u_y)\in\{0,\ldots,k-1\}^2.
\]
Its directed distance to \(v\) is
\[
 d(u,v)=(v_x-u_x\bmod k)+(v_y-u_y\bmod k).            \tag{1.1}
\]
This is the shortest-path metric of the positive generating arcs, so there
is no metric-closure issue.

Let the depot be \(v_0=v_{N+1}=0\), and let
\[
                         v_1,\ldots,v_N
\]
be a service order of a nonempty active set.  Put
\[
 \delta_j(u,v):=\mathbf 1\{v_j<u_j\},
 \qquad j\in\{x,y\}.                                  \tag{1.2}
\]
For either coordinate,
\[
 (v_j-u_j\bmod k)=v_j-u_j+k\delta_j(u,v).             \tag{1.3}
\]
Summing (1.3) around the closed service order makes the ordinary coordinate
differences telescope.  Hence
\[
\begin{aligned}
 \sum_{i=0}^{N}d(v_i,v_{i+1})
 &=k\sum_{i=0}^{N}
       \bigl(\delta_x(v_i,v_{i+1})
             +\delta_y(v_i,v_{i+1})\bigr)\\
 &=k(D_x+D_y),                                        \tag{1.4}
\end{aligned}
\]
where \(D_j\) is the number of strict descents in coordinate \(j\).
Equation (1.4) proves (0.1) exactly, including moves that skip arbitrary
uncalled or inactive vertices.

## 2. Exact cyclic chain-cover formulation

Cut the torus open at coordinate zero and partially order its vertices by
\[
                  u\preceq v
      \quad\Longleftrightarrow\quad
                  u_x\le v_x\ \hbox{and}\ u_y\le v_y. \tag{2.1}
\]
Define the boundary charge
\[
             c(u,v):=\delta_x(u,v)+\delta_y(u,v)
                     \in\{0,1,2\}.                    \tag{2.2}
\]

### Proposition 1

For every active set \(A\),
\[
 \frac{{\rm POST}(A)}{k}
 =
 \min_{\mathcal C,\tau}
       \sum_{i=1}^{r}c(t_i,s_{\tau(i+1)}).             \tag{2.3}
\]
Here:

* \(\mathcal C=\{C_1,\ldots,C_r\}\) is a chain cover of
  \(A\cup\{0\}\) in (2.1);
* every \(C_i\) is internally listed in nondecreasing product order;
* \(s_i,t_i\) are its first and last vertices;
* \(\tau\) is a cyclic ordering of the chains, with
  \(\tau(r+1)=\tau(1)\).

Zero-charge consecutive chains may of course be merged.  Thus one may
equivalently minimize over covers for which every displayed boundary has
positive charge.

### Proof

Take an arbitrary cyclic service order and cut it immediately after every
transition of positive charge.  Each resulting segment has no descent in
either coordinate and is therefore a chain in (2.1).  Its charged
transitions are precisely the boundaries in (2.3).

Conversely, concatenate the internally increasing chains in any cyclic
order.  Internal transitions have charge zero, and the complete winding
sum is exactly the sum of their boundary charges.  Equation (1.4) then
gives (2.3). \(\square\)

This is the correct ``minimum cyclic path cover'' quotient.  In particular,
two different column deficits may be paid by the same chain boundary; they
cannot be added unless a separate argument proves that their charges use
different boundaries.

### Width bounds

Let \(w\) be the width of \(A\cup\{0\}\) in (2.1).  Cutting any service
order at all positive-charge boundaries gives at most \(D_x+D_y\) chains,
because every boundary has charge at least one.  Dilworth therefore gives
\[
                         w\le D_x+D_y.                 \tag{2.4}
\]
Conversely, take a minimum \(w\)-chain cover and cyclically concatenate its
chains.  Each of its \(w\) boundaries costs at most two, so
\[
                         D_x+D_y\le2w.                 \tag{2.5}
\]
Equations (2.4)--(2.5) prove (0.2).  When \(0\) is the unique minimum of the
cut-open square, adjoining it does not change the width of a nonempty
active set.  It can, however, make the closing boundary cost two even when
\(w=1\); this is why the upper constant two is genuine.

## 3. Equivalent lift/lap assignment

Proposition 1 also has a useful algebraic form.  Given a service order,
replace every torus vertex \(v_i\) by a lift
\[
       \widetilde v_i=(v_{i,x}+ka_i,\ v_{i,y}+kb_i)
                         \in\mathbb Z^2              \tag{3.1}
\]
such that
\[
        \widetilde v_0=(0,0),\qquad
        \widetilde v_0\le\widetilde v_1\le\cdots
                       \le\widetilde v_{N+1}
                                                          \tag{3.2}
\]
coordinatewise.  The least feasible increments are obtained by increasing
\(a_i\) exactly at an \(x\)-descent and \(b_i\) exactly at a \(y\)-descent.
Consequently
\[
                  \widetilde v_{N+1}=(kD_x,kD_y).     \tag{3.3}
\]
The posterior problem is equivalently the minimum \(D_x+D_y\) for which all
active residue classes can be assigned to one monotone lifted service
sequence ending at (3.3).

This formulation makes the batching issue transparent.  One additional
\(x\)-lap changes the lift of an entire suffix of the service, so it may
repair many local column-order conflicts simultaneously.

## 4. Product activation scale

Activate every nondepot vertex independently with probability \(p=1/k\).
Then
\[
             \mathbb E|A|=k+O(1),\qquad
             {\rm Var}(|A|)=k+O(1).                   \tag{4.1}
\]
After rescaling the square to \([0,1]^2\), this is the usual sparse
binomial approximation to \(k\) independent uniform points.  Coordinate
ties contribute only \(o(\sqrt{k})\) to the width.  The classical
Hammersley estimate for the longest decreasing subsequence therefore gives
\[
                     \mathbb E w(A)=(2+o(1))\sqrt{k}.
                                                          \tag{4.2}
\]
Combining with Proposition 1,
\[
 (2+o(1))k^{3/2}
 \ \le\ \mathbb E{\rm POST}
 \ \le\ (4+o(1))k^{3/2}.                              \tag{4.3}
\]

The exact leading constant in the charged cyclic cover (2.3), rather than
the ordinary width constant alone, is needed before a \(4/3\) comparison is
possible.

## 5. Future-column probing lemma

The following elementary fact is the main adaptive escape.

### Lemma 2

Suppose the policy is currently at an active vertex \(u\), and let \(B\) be
any set of uncalled vertices satisfying \(u\preceq v\) for every \(v\in B\).
The policy may call the vertices of \(B\) in any chosen order until the
first active answer.

* Every inactive answer causes no movement.
* If an active \(v\in B\) is found, the move \(u\to v\) creates no
  coordinate descent.
* Under iid activation \(1/k\), the attempt fails with probability
  \[
                  \Pr[A\cap B=\varnothing]
                    =(1-1/k)^{|B|}
                    \le e^{-|B|/k}.                   \tag{5.1}
  \]

### Proof

Only the final active call, if any, causes a metric move.  The assumed
coordinate inequalities make that move nonwrapping.  Independence gives
the displayed product probability. \(\square\)

For example, from \(u=(i,j)\), the no-wrap part of one future column
\(i+t\) contains \(k-j\) successors.  Pooling \(t\) future columns gives a
bank of size \(t(k-j)\), and hence failure at most
\[
                        e^{-t(k-j)/k}.                 \tag{5.2}
\]
Away from the top \(o(k)\) rows, \(t=C\log k\) makes this probability
polynomially small.  A two-dimensional northeast descendant rectangle of
area \(\Theta(k^2)\) makes it \(e^{-\Theta(k)}\).

Lemma 2 does not by itself give a complete low-cost causal tour: after the
first active continuation, the remaining usable successor bank shrinks, and
different runs compete for the same vertices.  It does prove that the
following proposed lower-bound step is invalid:

> an unmatched or empty immediate successor column forces a new winding.

The policy is not restricted to the immediate column.  Transitive torus
distance lets it test all later no-wrap descendants before paying any
winding.

## 6. What a valid adaptive lower bound must prove

A proof above \(4/3\) for this family must establish all three statements
below.

1. **Posterior constant.**  Determine the leading constant of the charged
   cyclic cover (2.3), not just the ordinary width.
2. **Global causal potential.**  Charge a winding only after all usable
   successor banks have been accounted for.  Immediate-layer matching
   deficits do not suffice by Lemma 2.
3. **No batching.**  Show that one extra coordinate lap cannot repair the
   alleged losses of many columns.  In the lift formulation, this means
   proving that the conflicting vertices require distinct increases of the
   integer lift coordinates.

At present there is no such global lower bound.  The rigorous conclusion is
therefore an obstruction, not a \(>4/3\) construction: the torus identity is
valid, but the natural column-by-column adaptive argument is not.
