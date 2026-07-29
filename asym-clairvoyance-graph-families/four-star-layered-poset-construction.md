# A four-star layered-poset construction with gap at least \(3/2-o(1)\)

## Outcome

There is a genuine family of independently activated asymmetric stochastic
TSP instances whose adaptive/a-posteriori ratio has liminf at least
\(3/2\).  In particular, it is strictly larger than \(4/3\) for every
sufficiently large member of the family.

The construction has two permanent gate lanes and \(L\) independent
four-selector stages.  At one stage the four selectors are the two row
stars and two column stars of a \(2\times2\) matrix:
\[
\begin{array}{c|c}
 A_i&\{11,12\}\\
 B_i&\{21,22\}\\
 C_i&\{11,21\}\\
 D_i&\{12,22\}.
\end{array}                                           \tag{0.1}
\]
For an integer \(n\ge2\), take
\[
                         L=n^5,\qquad q=n^{-2}.       \tag{0.2}
\]

Every set of at most two active stars has a system of distinct row and
column representatives.  Consequently the posterior uses two increasing
chains unless a stage has at least three active selectors, an event of
total expectation \(O(Lq^3)=o(1)\).

The causal obstruction occurs one order earlier.  To retain only two
increasing runs, the first run must choose one star and one matrix cell at
each stage before it knows the other active stars.  In every deterministic
local strategy, one exact two-active pattern is fatal.  Its probability is
\[
                         h=q^2(1-q)^2.                \tag{0.3}
\]
A direct-product argument, including arbitrary inactive prequeries into
future stages, gives
\[
 \Pr[K_{\rm causal}\le2]\le(1-h)^L=o(1).              \tag{0.4}
\]
Thus
\[
\begin{aligned}
 \mathbb E\operatorname{width}&=2+o(1),\\
 \inf_\pi\mathbb E K_\pi&\ge3-o(1).                  \tag{0.5}
\end{aligned}
\]

Using the positive directed poset metric
\[
 d_\varepsilon(x,y)=
 \begin{cases}
  \varepsilon,&x<y,\\
  1,&x\not<y,
 \end{cases}
 \qquad
 d(r,x)=d(x,r)=\frac12,                              \tag{0.6}
\]
with \(\varepsilon=n^{-10}=L^{-2}\), converts (0.5) into
\[
 \boxed{\displaystyle
 \liminf_{n\to\infty}
 \frac{\operatorname{OPT}_{\rm adapt}}
      {\operatorname{OPT}_{\rm post}}
             \ge\frac32.}                            \tag{0.7}
\]

The proof is analytic.  No finite-instance enumeration or computational
verification is used.

## 1. The layered poset

For \(i=0,\ldots,L\), introduce two permanent gate clients
\[
                         G_i=\{g_i^1,g_i^2\}.
\]
For every stage \(i=1,\ldots,L\), introduce four stochastic selector
clients
\[
                         A_i,B_i,C_i,D_i.
\]
Their activation bits are mutually independent and each has probability
\(q=n^{-2}\).

The generating comparabilities at stage \(i\) are
\[
\begin{array}{c|c|c}
\text{selector}&\text{allowed predecessor gates}
               &\text{allowed successor gates}\\ \hline
A_i&\{g_{i-1}^1\}&\{g_i^1,g_i^2\}\\
B_i&\{g_{i-1}^2\}&\{g_i^1,g_i^2\}\\
C_i&\{g_{i-1}^1,g_{i-1}^2\}&\{g_i^1\}\\
D_i&\{g_{i-1}^1,g_{i-1}^2\}&\{g_i^2\}.
\end{array}                                          \tag{1.1}
\]
That is, every gate in the middle column of (1.1) is below the selector,
and the selector is below every gate in the last column.  Take the
transitive closure and add no other comparabilities.

The four selectors at one stage are pairwise incomparable.  The two gates
in one layer are incomparable.  Every gate in \(G_{i-1}\) is below every
gate in \(G_i\): the row stars \(A_i,B_i\) already supply the four
corresponding two-step paths in the poset.

The matrix-cell interpretation of (1.1) is exact.  Putting a selector on a
chain that enters stage \(i\) through row \(r\) and leaves through column
\(c\) assigns that selector the cell \(rc\).  The row stars prescribe \(r\)
and allow either \(c\); the column stars allow either \(r\) and prescribe
\(c\).

## 2. Posterior width

### Lemma 1 (two-star matching)

Every subset of at most two of
\(\{A_i,B_i,C_i,D_i\}\) can be assigned distinct matrix cells with distinct
rows and distinct columns.

### Proof

For the two same-type pairs use
\[
 A_i\mapsto11,\ B_i\mapsto22,
 \qquad
 C_i\mapsto11,\ D_i\mapsto22.
\]
For a mixed row-column pair, assign the row star to the column opposite the
column star and assign the column star to the row opposite the row star.
The resulting two cells have different rows and columns.  Singletons and
the empty set are immediate. \(\square\)

Let \(Z_i\sim\operatorname{Binomial}(4,q)\) be the number of active
selectors at stage \(i\).

### Lemma 2 (realization-wise width bound)

For every realization,
\[
 2\le\operatorname{width}
 \le2+\sum_{i=1}^L(Z_i-2)_+.                         \tag{2.1}
\]

### Proof

Every gate layer \(G_i\) is a permanent two-element antichain, proving the
lower bound.

For the upper bound, start two chains at \(g_0^1,g_0^2\).  At stage \(i\),
if at most two selectors are active, use Lemma 1 to put them on distinct
chain transitions from \(G_{i-1}\) to \(G_i\).  The two used cells have
distinct columns, so both chains emerge at distinct gates and the
construction continues at stage \(i+1\).  If more than two selectors are
active, put any two on the two spanning chains and give each remaining
selector a singleton chain.  This is a valid chain cover of the entire
realized poset.  Dilworth gives (2.1). \(\square\)

Since
\[
\begin{aligned}
\mathbb E(Z_i-2)_+
 &=\Pr[Z_i=3]+2\Pr[Z_i=4]\\
 &=4q^3(1-q)+2q^4\\
 &=4q^3-2q^4
 \le4q^3,                                            \tag{2.2}
\end{aligned}
\]
Lemma 2 gives
\[
 2\le\mathbb E\operatorname{width}
      \le2+4Lq^3.                                    \tag{2.3}
\]
For \(L=n^5,q=n^{-2}\),
\[
                         Lq^3=n^{-1}\longrightarrow0,
\]
and hence
\[
                         \mathbb E\operatorname{width}
                                  =2+o(1).            \tag{2.4}
\]

## 3. What a two-run causal execution must do

Fix an arbitrary deterministic causal policy.  Randomized policies will be
handled by conditioning on their private seed.

For an active service order, let \(K\) be the number of maximal increasing
runs in the poset.  On the event \(K\le2\), in fact \(K=2\), because every
permanent gate layer is an antichain of size two.  The two runs form a
two-chain cover of every permanent gate.  Therefore:

1. each run contains exactly one gate from every layer \(G_i\);
2. the first run passes through the stages in increasing order and cannot
   end before layer \(L\);
3. at a stage it contains at most one active selector;
4. a selector in the first run must use an allowed predecessor and an
   allowed successor from (1.1); and
5. the remaining active selector, if any, must fit the complementary row
   and column on the second run.

These facts also dispose of remote prequeries that return active.

* If the first run calls a selector in a future stage before serving its
  intervening permanent gates, an active answer jumps past one gate from
  each intervening layer.  The second run can contain only one of the two
  gates in such a layer, so a two-run cover becomes impossible.
* If it calls a same-stage selector incompatible with its current input
  row, an active answer is not an increasing continuation.
* After one same-stage selector is active, calling a second one active
  creates a break because same-stage selectors are incomparable.

Inactive answers cause no movement and may reveal arbitrary future bits.
Section 5 includes these free negative prequeries in the direct-product
bound.

## 4. The local four-star game

Consider one stage, and suppose the first run enters through a specified
row \(r\in\{1,2\}\).  A local strategy may query selector bits in any
adaptive order.  An inactive query is free.  Its first active selector, if
one occurs on a successful two-run execution, must be assigned immediately
to an allowed cell in row \(r\); querying another same-stage active
selector before leaving the stage already loses the two-run event.  The
unserved active selectors must fit the complementary row and column on the
second run.

### Lemma 3 (one exact-pair atom is always lost)

Every deterministic local strategy fails on at least one realization
having exactly two active selectors.  Consequently its success probability
is at most
\[
                         1-h,\qquad
                         h=q^2(1-q)^2.                \tag{4.1}
\]
The statement holds for either input row and even if the strategy is given
an arbitrary independent advice string.

### Proof

Follow the strategy on the branch on which all selector queries made before
the output gate answer inactive.  If it chooses the output gate without
querying a selector, then every exact two-active realization is fatal:
both incomparable selectors have been left for the single second run.

Otherwise let \(X\) be the first selector queried.  Consider the branch on
which \(X\) is active and every later selector queried before the output
gate is inactive.  If \(X\) is incompatible with the input row, this branch
is already fatal; choosing any \(Y\ne X\), the exact-pair realization
\(\{X,Y\}\) is therefore fatal as well.  Otherwise, to have any chance of
success the strategy eventually chooses an output column \(c\) such that
the cell \((r,c)\) belongs to the star \(X\).

Exactly two of the four stars contain \((r,c)\): its row star and its column
star.  Let \(Y\ne X\) be the other one.  On the realization in which exactly
\(X,Y\) are active, there are two possibilities.  If the strategy queries
\(Y\) before taking the output gate, then \(Y\) is a second active
same-stage selector before either output gate has been put on the first
run; the two-run event is already impossible.  If the strategy does not
query \(Y\), then \(Y\) is left for the second run.  That run must use the
complementary cell \((3-r,3-c)\), but a row or column star through
\((r,c)\) never contains the complementary cell.  Thus the exact
\(\{X,Y\}\) realization is fatal in either case.

Hence at least one specified exact-pair atom is outside the success event.
Every such atom has probability \(q^2(1-q)^2=h\).  Independent advice only
selects a deterministic strategy before the four local bits are exposed,
so the same conclusion holds conditionally on every advice value.
\(\square\)

The lemma deliberately uses only one bad atom.  Realizations with three or
four active selectors also make two runs impossible, and unsafe prequeries
can create further failures.

## 5. Direct product and arbitrary future prequeries

The local games are independent, but a policy may interleave their inactive
queries.  We record explicitly why this does not invalidate multiplication.

### Lemma 4 (interleaved direct product)

For every deterministic causal policy,
\[
                         \Pr[K\le2]\le(1-h)^L.         \tag{5.1}
\]

### Proof

View preservation of two runs as a finite reachability game.  Each stage is
one independent four-bit component.  Its state records:

* the entering row of the first run;
* which local selectors have already answered inactive;
* whether one local selector has been taken and which output column was
  chosen; and
* whether the local two-chain condition has already failed.

For a **ready** component, let \(V(s)\) be the optimal probability of
eventually completing that local component successfully from local state
\(s\).  Its input row is part of \(s\).  The Bellman definition gives, for
every local action,
\[
                 \mathbb E[V(s_{\rm next})\mid s]
                         \le V(s).                    \tag{5.2}
\]
At an untouched ready component, Lemma 3 gives
\[
                         V(s)\le1-h                  \tag{5.3}
\]
for either entering row.

For an **unready** future component, use the following upper potentials.

* If none of its four bits has been queried, assign potential \(1-h\).
* If at least one premature query has answered inactive, assign the coarse
  potential \(1\).
* If a premature query answers active, assign potential zero to the whole
  two-run event.

The first premature query into an untouched component changes its expected
factor by at most
\[
                 q\cdot0+(1-q)\cdot1
                         =1-q\le1-h,                 \tag{5.4}
\]
because \(h=q^2(1-q)^2\le q\).  Further premature queries start from factor
one and have expected next factor at most one.  When the input gate is
eventually reached, an untouched component changes from factor \(1-h\) to
an exact ready value at most \(1-h\) by (5.3).  A prematurely probed
component changes from the coarse factor one to an exact conditional ready
value at most one.  Thus gate advancement also cannot increase the factor.

Multiply these local factors, retaining factor one for a successfully
completed stage and factor zero for a failed stage.  A selector query
changes one factor.  Serving a gate can simultaneously finish one stage
and make the next stage ready; treat this as two deterministic local
updates, each nonincreasing by the Bellman definition and (5.3).
Equation (5.2), the premature-query estimate (5.4), and the gate
observations therefore make the product a nonnegative supermartingale
under arbitrary adaptive interleaving.

Initially its value is at most \((1-h)^L\).  At termination the product is
the indicator that every local component has a valid two-chain routing.
By Section 3, \(K\le2\) implies that all components succeeded.  Taking
expectations proves (5.1). \(\square\)

This is the standard direct-product proof for independent finite decision
problems; it does not assume that the policy literally finishes all queries
of stage \(i\) before probing stage \(i+1\).  Free inactive probes merely
move one factor to a conditional Bellman state.  An active premature probe
moves that factor to zero.

For a randomized policy, condition on its private seed and apply Lemma 4;
averaging preserves (5.1).

Since permanent gate layers force \(K\ge2\),
\[
\begin{aligned}
\mathbb E K
 &\ge2+\Pr[K\ge3]\\
 &=3-\Pr[K\le2]\\
 &\ge3-(1-h)^L.                                      \tag{5.5}
\end{aligned}
\]
With \(L=n^5,q=n^{-2}\),
\[
 Lh
 =n(1-n^{-2})^2
 \longrightarrow\infty,
\]
so
\[
                         (1-h)^L\le e^{-Lh}=o(1).
\]
Taking the infimum over all causal policies gives
\[
                         \inf_\pi\mathbb E K_\pi
                                  \ge3-o(1).          \tag{5.6}
\]

## 6. Positive asymmetric metric

Fix \(0<\varepsilon<1\).  On distinct poset clients define
\[
 d_\varepsilon(x,y)=
 \begin{cases}
  \varepsilon,&x<y,\\
  1,&x\not<y,
 \end{cases}
\qquad d_\varepsilon(x,x)=0.                         \tag{6.1}
\]
Add a depot \(r\) with
\[
                         d(r,x)=d(x,r)=\frac12        \tag{6.2}
\]
for every client.

### Lemma 5

Equations (6.1)--(6.2) define a strictly positive directed metric.

### Proof

Two consecutive \(\varepsilon\)-arcs imply
\(x<y<z\), so transitivity gives \(x<z\) and makes the direct arc
\(\varepsilon\).  Every other two-client path has length at least
\(1+\varepsilon\) whenever its direct distance is one.  A depot detour has
length exactly one and therefore does not shorten an incomparable
client-to-client arc.  Finally, every two-leg path used to compare a depot
arc has length at least \(1/2+\varepsilon>1/2\). \(\square\)

Fix a realization with \(N\) nondepot clients.  If a service order has
\(K\) maximal increasing runs, it has \(N-K\) consecutive comparable
transitions and \(K-1\) breaks.  Including the two depot legs, its cost is
exactly
\[
\begin{aligned}
 C_\varepsilon(N,K)
 &=1+\varepsilon(N-K)+(K-1)\\
 &=\varepsilon N+(1-\varepsilon)K.                   \tag{6.3}
\end{aligned}
\]
Transit through inactive or uncalled vertices cannot improve (6.3),
because (6.1) is already the shortest-path metric.  Conversely, directly
following the displayed service order attains (6.3).

Dilworth and (6.3) give the realization-wise posterior identity
\[
 \operatorname{OPT}_{\rm post}(A)
   =\varepsilon N(A)
       +(1-\varepsilon)\operatorname{width}(P[A]).    \tag{6.4}
\]
For every causal policy the same identity with its realized run count gives
\[
 \mathbb E C_\pi
   =\varepsilon\mathbb EN
       +(1-\varepsilon)\mathbb E K_\pi.               \tag{6.5}
\]

There are \(2(L+1)\) permanent gates and \(4L\) probability-\(q\)
selectors, so
\[
                         \mathbb EN=2(L+1)+4Lq=O(L).
\]
Choose
\[
                         \varepsilon=n^{-10}=L^{-2}.            \tag{6.6}
\]
Then \(\varepsilon\mathbb EN=o(1)\).  Equations (2.4), (5.6),
(6.4), and (6.5) yield
\[
\begin{aligned}
 \operatorname{OPT}_{\rm post}
   &\le2+o(1),\\
 \operatorname{OPT}_{\rm adapt}
   &\ge3-o(1).
\end{aligned}                                         \tag{6.7}
\]
Therefore
\[
 \liminf_{n\to\infty}
 \frac{\operatorname{OPT}_{\rm adapt}}
      {\operatorname{OPT}_{\rm post}}
 \ge\frac32.                                          \tag{6.8}
\]

In particular, (6.8) is strictly larger than \(4/3\) for all sufficiently
large \(L\).  All selector activations are independent, every distance is
strictly positive, the metric is asymmetric, and the lower bound applies to
arbitrary randomized causal policies with arbitrary remote and future
queries.

There is also a completely explicit finite member.  Take
\[
       q=\frac1{100},\qquad L=20{,}000,\qquad
       \varepsilon=10^{-7}.                          \tag{6.9}
\]
Here
\[
 Lh=\frac{9801}{5000}=1.9602>\frac{49}{25}.
\]
The exponential series gives the entirely rational check
\[
 e^{Lh}>
 \sum_{j=0}^{6}\frac{(49/25)^j}{j!}
 =\frac{1242847593301}{175781250000}>7.              \tag{6.10}
\]
Thus \((1-h)^L\le e^{-Lh}<1/7\), and
\[
                 \inf_\pi\mathbb E K_\pi>\frac{20}{7}.          \tag{6.11}
\]
On the other hand, (2.1)--(2.2) give
\[
\begin{aligned}
\mathbb E\operatorname{width}
 &\le2+L(4q^3-2q^4)
   =\frac{5199}{2500},\\
\mathbb EN
 &=2(L+1)+4Lq
   =40{,}802.
\end{aligned}                                                     \tag{6.12}
\]
Using (6.4)--(6.5), three times the adaptive lower bound minus four
times the posterior upper bound is strictly larger than
\[
\begin{aligned}
 &(1-10^{-7})
 \left(3\cdot\frac{20}{7}
       -4\cdot\frac{5199}{2500}\right)
 -10^{-7}(40{,}802)\\
 &=(1-10^{-7})\frac{1107}{4375}-0.0040802
 >0.24.
\end{aligned}                                                     \tag{6.13}
\]
Therefore this single \(120{,}002\)-client instance already has
\(\operatorname{OPT}_{\rm adapt}/\operatorname{OPT}_{\rm post}>4/3\).

## 7. Failure audit

### Remote future probing

An inactive future query is free and is explicitly included in the local
Bellman state and product supermartingale.  An active future query skips a
permanent gate on the first run; with two permanent gates in that layer, the
remaining run cannot cover both.  Thus it loses the event \(K\le2\).

### Calling all selectors first

The first active selector is called before its necessary predecessor gate.
It cannot lie on either of the two gate-spanning chains in the required
order.  If no selector is active, the call bank is free, but that branch
does not compensate for the active branches in the Bellman product.

### Interleaving the two chains

Two increasing runs are contiguous portions of the actual service order.
Once the first run is broken, it cannot be resumed.  Since each future gate
layer needs one gate on each of two chains, a successful first run must reach
layer \(L\) before the second begins.

### Three-active stages

They help the adaptive lower bound and are the only source of posterior
width above two in the estimate.  Their total expected contribution is
\(O(Lq^3)=o(1)\), while exact two-active local conflict has accumulated
intensity
\[
                         Lq^2(1-q)^2\to\infty.
\]
This separation of the second- and third-order scales is the purpose of
\(L=n^5,q=n^{-2}\).

### Shortest-path closure and inactive transit

The distances are specified directly as a metric.  Transitivity prevents a
chain of cheap arcs from shortening an incomparable distance.  Depot
detours tie, but do not beat, a unit break.  Inactive vertices may be used as
transit; doing so is already captured by the metric distance and does not
count as service.
