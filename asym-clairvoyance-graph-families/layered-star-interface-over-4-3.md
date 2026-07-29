# A layered star-interface poset exceeds \(4/3\)

## 0. Result

This note gives a standard closed-depot asymmetric stochastic-TSP
construction with clairvoyance gap strictly larger than \(4/3\).

The local interface has two permanent lanes and four independently active
star types.  Every realization with at most two active types at each stage
has width two.  A causal first run, however, must choose one edge of a
\(K_{2,2}\) interface before learning the other active type.  For every
choice, one exact two-active pattern is incompatible.

The main point is a direct-product lemma robust to arbitrary future probes.
Put
\[
                         h=q^2(1-q)^2.
\tag{0.1}
\]
For an \(L\)-stage block, every causal policy satisfies
\[
       \Pr[\text{its service order has at most two increasing runs}]
                         \leq(1-h)^L.
\tag{0.2}
\]
A premature call into a future stage does not evade the product: if it is
active, two-run feasibility is already destroyed; if it is inactive, that
stage has cost a factor \(1-q\leq1-h\).  If there is no premature call, the
current untouched stage costs the factor \(1-h\) through its incompatible
pair.

Take \(L\asymp q^{-2}\) and then \(q\downarrow0\).  Posterior expected width
tends to two, while every causal policy uses nearly three runs.  Giving
every depot arc length \(1/2\) makes the closed-tour cost exactly a convex
combination of client count and run count.  For suitable finite rational
parameters the resulting ratio is strictly larger than \(4/3\).

## 1. One interface

For every layer \(i=0,\ldots,L\), create two permanent gate clients
\[
                         G_i=\{g_i^1,g_i^2\}.
\]
The two gates in one layer are incomparable.

For stage \(i=1,\ldots,L\), create four stochastic type clients
\[
                         x_i^A,x_i^B,x_i^C,x_i^D,
\]
each independently active with probability \(q\).  All type clients over
all stages are mutually independent.

The four types are the row and column stars of \(K_{2,2}\):
\[
\begin{array}{c|c}
T&E_T\\ \hline
A&\{(1,1),(1,2)\},\\
B&\{(2,1),(2,2)\},\\
C&\{(1,1),(2,1)\},\\
D&\{(1,2),(2,2)\}.
\end{array}
\tag{1.1}
\]
For every \((a,b)\in E_T\), impose
\[
                         g_{i-1}^a<x_i^T<g_i^b,
\tag{1.2}
\]
and take transitive closure.  There are no other generating
comparabilities.

Every pair of gates in consecutive layers is comparable.  Indeed, for
fixed \(a,b\), the row star \(A\) or \(B\) belonging to row \(a\) supplies
a two-step relation from \(g_{i-1}^a\) to \(g_i^b\).  This is a relation in
the fixed poset whether or not that intermediate type client is active.

The four type clients in one stage form an antichain.  A type can lie on a
gate-to-gate chain exactly when that chain uses an edge in \(E_T\).

## 2. Offline width

### Lemma 1 (every pair is compatible)

For every two distinct types \(T,U\), there are disjoint edges
\[
                         (a,b)\in E_T,\qquad
                         (3-a,3-b)\in E_U.
\tag{2.1}
\]

### Proof

The row pair \(A,B\) uses distinct input rows and may choose distinct
columns.  The column pair \(C,D\) uses distinct output columns and may
choose distinct rows.  For a row star and a column star, choose their common
edge for neither type: the row star takes its other column and the column
star takes its other row.  These two edges are complementary. \(\square\)

### Lemma 2 (realization-wise width upper bound)

Let \(Z_i\) be the number of active types at stage \(i\).  For every
realization,
\[
       \operatorname{width}
       \leq2+\sum_{i=1}^L(Z_i-2)_+.
\tag{2.2}
\]
In particular, if \(Z_i\leq2\) for every stage, the width is exactly two.

### Proof

Start two chains at \(g_0^1,g_0^2\).  At each stage with zero, one, or two
active types, Lemma 1 lets us choose a perfect matching from \(G_{i-1}\) to
\(G_i\) which assigns every such active type to a distinct matching edge.
Continue the two chains through that matching.

If a stage has more than two active types, put any two of them on the two
backbone chains and cover each remaining type by a singleton chain.  Doing
this independently at every stage proves (2.2).  When every \(Z_i\leq2\),
the two backbone chains cover the realization.  The permanent antichain
\(G_i\) gives the matching lower bound two. \(\square\)

Since \(Z_i\sim\operatorname{Binomial}(4,q)\),
\[
\begin{aligned}
 \mathbb E(Z_i-2)_+
 &=\Pr[Z_i=3]+2\Pr[Z_i=4]\\
 &=4q^3(1-q)+2q^4\\
 &=4q^3-2q^4.
\end{aligned}
\tag{2.3}
\]
Thus, writing \(W_L\) for the realized width,
\[
                         2\leq\mathbb EW_L
       \leq2+L(4q^3-2q^4).
\tag{2.4}
\]

## 3. Structure of a two-run service

Fix a causal policy and a realization.  Suppose its active service order has
at most two maximal increasing runs.

Every permanent antichain \(G_i\) has one gate in each run.  The first run
therefore visits exactly one gate from every layer, in increasing layer
order, before the second run begins.  In particular:

1. before crossing stage \(i\), the first run is at one input gate
   \(g_{i-1}^a\);
2. it leaves the stage through one output gate \(g_i^b\);
3. it contains at most one active type of that stage; and
4. every other active type of the stage must lie on the complementary edge
   \((3-a,3-b)\) used by the second run.

Calling an active client from a later stage before crossing the current
stage destroys two-run feasibility.  Such a call moves the service order
above at least one unserved permanent gate layer.  Both gates of that
skipped layer would then remain for the second run, but they are
incomparable and cannot both lie in one chain.

This observation is model-specific and decisive: remote future calls are
allowed, but a premature active answer is itself the third-run certificate.

## 4. The local incompatible-pair lemma

### Lemma 3

Fix the input lane \(a\), an arbitrary deterministic causal strategy for an
untouched stage, and arbitrary side information independent of that stage's
four activation bits.  If the strategy never makes a premature query into a
different unresolved stage, its probability of crossing the current stage
while preserving the possibility of a two-run completion is at most
\[
                         1-h,
             \qquad h=q^2(1-q)^2.
\tag{4.1}
\]

### Proof

Follow the strategy on the branch on which every type queried before the
output gate is inactive.

If it calls the output gate without querying a type, then any realization
with exactly two active types is fatal: both types remain for the second
run.  Choose one fixed pair.

Otherwise let \(T\) be the first type it queries.  If an active \(T\) cannot
lie above input lane \(a\), then any exact two-active realization containing
\(T\) is already fatal.  Assume it can.

On the branch where \(T\) is active and every subsequently queried type is
inactive, the strategy must eventually choose an output lane \(b\).  If
\((a,b)\notin E_T\), then any specified exact two-active realization
containing \(T\) is fatal.  Otherwise exactly two star types contain the
edge \((a,b)\): its row star and its column star.  Let \(U\ne T\) be the
other one.

Consider the realization in which exactly \(T,U\) are active.  If the
strategy queries \(U\) before leaving the stage, then the call to \(U\)
starts a new run because same-stage types are incomparable, and the
unfinished first backbone run later forces a third run.  If it leaves \(U\)
for the second run, that run would need the complementary edge
\((3-a,3-b)\).  But \(U\), being the other star through \((a,b)\), does not
contain the complementary edge.  Thus this realization is fatal in either
case.

The probability of one specified exact two-active realization is
\[
                         q^2(1-q)^2=h.
\]
All side information was fixed independently, so the same bound holds after
averaging it. \(\square\)

The proof permits arbitrary inactive probing inside the stage and arbitrary
private randomization, after conditioning on the random seed.

## 5. Direct product despite future probes

### Lemma 4 (causal product lemma)

For every causal policy on the \(L\)-stage block,
\[
       \Pr[K_L\leq2]\leq(1-h)^L,
\tag{5.1}
\]
where \(K_L\) is the number of maximal increasing runs in its active service
order.

### Proof

We use a product Bellman supermartingale.  This formulation is important:
a policy may obtain some local inactive answers and only then decide to
probe a future stage, so a naive induction merely on the number of stages
would lose the partially revealed local state.

For a stage whose input gate has been reached, let \(V(s)\) be the optimal
probability of completing that local two-chain routing from its current
local state \(s\).  The state records the input lane, inactive local
answers, any active type already put on the first run, and the chosen
output lane.  Independent exterior information and private randomization
are allowed.  Bellman's inequality says that every local action satisfies
\[
                  \mathbb E[V(s_{\rm next})\mid s]\leq V(s).
\tag{5.2}
\]
For either input lane, Lemma 3 gives
\[
                  V(s_{\rm untouched})\leq1-h.
\tag{5.3}
\]

For a future stage whose input gate has not yet been reached, assign these
upper potentials:

* an untouched stage has factor \(1-h\);
* a stage with at least one premature inactive answer has the coarse factor
  \(1\);
* a stage with a premature active answer has factor zero.

At the first premature query into an untouched future stage, the expected
new factor is at most
\[
                  q\cdot0+(1-q)\cdot1
                  =1-q\leq1-h,
\tag{5.4}
\]
because \(h=q^2(1-q)^2\leq q\).  Further premature queries start from the
coarse factor one and cannot increase it in expectation.  When the input
gate is eventually reached, an untouched future factor \(1-h\) is replaced
by an exact ready value at most \(1-h\), by (5.3); a prematurely probed
factor one is replaced by an exact conditional value at most one.

Multiply the factors of all stages, using the exact Bellman value for every
ready stage and terminal factors one and zero for successful and failed
stages.  A selector query changes one factor.  Serving a gate may both
finish one stage and make the next ready; regard that as two deterministic
local updates, neither of which increases its factor.  Equations
(5.2)--(5.4) show that the product's conditional expectation never
increases, even when the policy adaptively interleaves stages.  The product
is therefore a nonnegative supermartingale.  Initially it is at most
\((1-h)^L\).
At termination the product is one on the event that every stage has a
valid local two-chain routing and zero otherwise.  Section 3 shows
\[
   \{K_L\leq2\}\subseteq
   \{\text{every local routing succeeds}\}.
\]
Taking expectations proves (5.1).  Randomized policies follow by
conditioning on their private seed. \(\square\)

Since the permanent gate antichains force \(K_L\geq2\) always, Lemma 4 gives
\[
\begin{aligned}
 \inf_\pi\mathbb E K_L
 &\geq
 2\Pr[K_L\leq2]+3\Pr[K_L\geq3]\\
 &=3-\Pr[K_L\leq2]\\
 &\geq3-(1-h)^L.
\end{aligned}
\tag{5.5}
\]

## 6. Choosing finite parameters

Let \(c>\log3\), take \(q>0\) rational and small, and choose an integer
\[
                         L=\left\lceil\frac c{q^2}\right\rceil.
\tag{6.1}
\]
Then
\[
 Lh=Lq^2(1-q)^2\longrightarrow c
 \qquad(q\downarrow0),
\]
so
\[
                         (1-h)^L\longrightarrow e^{-c}<\frac13.
\tag{6.2}
\]
Meanwhile
\[
                         L(4q^3-2q^4)\longrightarrow0.
\tag{6.3}
\]
Combining (2.4) and (5.5),
\[
 \liminf_{q\downarrow0}
 \frac{\inf_\pi\mathbb E K_L}
      {\mathbb EW_L}
 \geq\frac{3-e^{-c}}2>\frac43.
\tag{6.4}
\]
The inequality is strict, so sufficiently small finite rational \(q\) and
the finite integer \(L\) in (6.1) already satisfy
\[
       \inf_\pi\mathbb E K_L
             >\frac43\,\mathbb EW_L.
\tag{6.5}
\]

For example, one may take any fixed \(c>\log3\) and then choose a rational
\(q\) small enough that
\[
       3-(1-q^2(1-q)^2)^L
       >
       \frac43\left[2+L(4q^3-2q^4)\right].
\tag{6.6}
\]
This is an explicit finite inequality and avoids any appeal to computation.

Here is one completely numerical choice:
\[
                         q=\frac1{100},
             \qquad L=20{,}000.
\tag{6.7}
\]
Then
\[
 h=\frac{9801}{100{,}000{,}000},
 \qquad
 Lh=\frac{9801}{5000}=1.9602>\frac{49}{25}.
\]
Moreover,
\[
 e^{Lh}>
 \sum_{j=0}^{6}\frac{(49/25)^j}{j!}
 =\frac{1242847593301}{175781250000}>7.
\]
Consequently
\[
 (1-h)^L\leq e^{-Lh}<\frac17,
 \qquad
 \inf_\pi\mathbb E K_L>\frac{20}{7}.
\tag{6.8}
\]
Also
\[
 L(4q^3-2q^4)=\frac{199}{2500},
 \qquad
 \mathbb EW_L\leq\frac{5199}{2500}.
\tag{6.9}
\]
The strict run-count margin is therefore
\[
 \frac{20}{7}
 -\frac43\frac{5199}{2500}
 =\frac{369}{4375}>0.
\tag{6.10}
\]

## 7. Directed metric

Fix \(0<\varepsilon<1\).  On distinct clients of one block put
\[
 d_\varepsilon(u,v)=
 \begin{cases}
   \varepsilon,&u<v\text{ in the poset},\\
   1,&u\not<v,
 \end{cases}
\tag{7.1}
\]
and set \(d(u,u)=0\).  Transitivity proves the directed triangle inequality:
two \(\varepsilon\)-arcs imply a direct comparability, and every other
nontrivial two-arc path has length at least \(1+\varepsilon\).

For a realization with \(N\) active clients and an active service order
having \(K\) maximal increasing runs, the internal transition cost is
\[
                         \varepsilon(N-K)+(K-1).
\]
Give the depot \(r\) distance \(1/2\) to and from every client.  A detour
through the depot has length one, exactly the distance of an incomparable
client pair, so this extension still satisfies all directed triangle
inequalities.  The exact closed-tour cost identity is
\[
       C_\varepsilon(N,K)
       =1+\varepsilon(N-K)+(K-1)
       =\varepsilon N+(1-\varepsilon)K.
\tag{7.2}
\]
Dilworth gives the posterior value
\[
 \operatorname {OPT}_{\rm post}
 =\varepsilon\,\mathbb EN_L
        +(1-\varepsilon)\mathbb EW_L.
\tag{7.3}
\]
Equation (5.5) gives
\[
 \operatorname {OPT}_{\rm adapt}
 \geq\varepsilon\,\mathbb EN_L
 +(1-\varepsilon)\bigl[3-(1-h)^L\bigr].
\tag{7.4}
\]

Choose the finite \(q,L\) from (6.6), and then choose
\(\varepsilon>0\) sufficiently small.  The common
\(\varepsilon\mathbb EN_L\) term is then small relative to the strict
run/width margin in (6.6).  Equations (7.3)--(7.4) yield
\[
       \operatorname {OPT}_{\rm adapt}
             >\frac43\operatorname {OPT}_{\rm post}.
\tag{7.5}
\]

The parameters in (6.7) can be completed explicitly by taking
\[
                         \varepsilon=10^{-7}.
\tag{7.6}
\]
The block has
\[
 \mathbb EN_L=2(L+1)+4Lq=40{,}802.
\]
Using the lower bound \(a=20/7\) for its causal expected run count and
the upper bound \(w=5199/2500\) for its posterior expected width, three
times the adaptive lower bound minus four times the posterior upper bound
is at least
\[
\begin{aligned}
 (1-\varepsilon)(3a-4w)-\varepsilon\mathbb EN_L
 &=(1-10^{-7})\frac{1107}{4375}-10^{-7}(40{,}802)\\
 &>0.24.
\end{aligned}
\tag{7.7}
\]
Thus this particular finite instance already satisfies (7.5).  It has
\(120{,}002\) potential nondepot clients and entirely rational activation
probabilities and distances.

All off-diagonal distances are strictly positive.  The complete directed
distance graph is strongly connected, so (7.1) together with the
half-length depot distances is already a genuine finite directed metric;
no zero-edge perturbation is required.

## 8. Failure audit

* **Remote future probing.**  It is explicitly charged in the product
  supermartingale.  An active answer skips two permanent gates at some
  layer and makes two runs impossible; an inactive answer changes that
  stage's factor from \(1-h\) to at most one, with expected new factor
  \(1-q\leq1-h\).
* **Arbitrary interleaving.**  Independent exterior histories are granted
  as side information in the local lemma.  The Bellman product evolves
  under the policy's actual query order, not a prescribed stage order.
* **Calling all types first.**  The first active future call immediately
  destroys two-run feasibility.  If every such call is inactive, its
  probability is already charged.
* **Offline exceptional stages.**  They are not ignored.  Equation (2.2)
  covers every realization and (2.3) charges their exact expected excess.
* **Shortest-path transit.**  The metric is defined directly from transitive
  reachability.  Passing through inactive vertices neither serves them nor
  changes the run identity (7.2).
* **Depot normalization.**  The two half-length depot legs contribute one;
  together with the \(K-1\) unit breaks this gives exactly \(K\) in the
  \(\varepsilon\downarrow0\) limit, with no additive dilution.
* **Product activations.**  Every one of the \(4L\) stochastic type clients
  has its own independent Bernoulli(\(q\)) bit.  Gates are permanent.

This completes all six proof obligations for a strict asymmetric
clairvoyance gap above \(4/3\).
