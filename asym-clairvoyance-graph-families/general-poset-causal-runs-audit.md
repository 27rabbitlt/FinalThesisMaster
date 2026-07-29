# Fixed posets with product activations: causal runs versus width

## Outcome

No fixed-poset construction exceeding \(4/3\) is established here.  The
audit does give several exact structural ceilings that remove the most
standard candidate families.

1. Every height-two poset, including projective-plane incidence posets,
   generalized polygons, crowns, and the standard examples \(S_n\), has a
   causal policy with
   \[
                 \frac{\mathbb E K_{\rm causal}}
                      {\mathbb E\operatorname {width}}
                 \leq 2-0.696=1.304<\frac43.
   \tag{0.1}
   \]
   Activations may be independent with unequal probabilities; in fact the
   bound is realization-wise after averaging only the policy's private
   randomization.
2. Series-parallel posets have a realization-wise optimal causal producer,
   so their ratio is one.  This includes arbitrary recursive parallel and
   ordinal sums; ordinal sums of antichains are the basic sweep example.
3. For an arbitrary poset of width \(w\), a fixed minimum chain cover gives
   the exact general estimate
   \[
   \frac{\mathbb E K_{\rm causal}}
        {\mathbb E\operatorname {width}}
   \leq
   \frac{\sum_{i=1}^{w}
          \left(1-\prod_{x\in C_i}(1-p_x)\right)}
        {\sum_{x\in B}p_x},
   \tag{0.2}
   \]
   where \(B\) is a maximum antichain and the minimum chain cover
   \(C_1,\ldots,C_w\) is chosen with one member of \(B\) in each chain.
   In particular, if every element of one maximum antichain has activation
   probability at least \(3/4\), the ratio is at most \(4/3\).  For uniform
   activation \(p\), every poset has ratio at most \(1/p\), so a strict gap
   above \(4/3\) requires \(p<3/4\).
4. Passing from rank-two incidence structures to high-rank buildings does
   not simply repeat a matching loss.  If an element has \(D_t\) descendants
   \(t\) ranks later, a causal policy can probe that entire bank and remain
   in the same increasing run with failure probability
   \[
                         (1-p)^{D_t}\leq e^{-pD_t}.
   \tag{0.3}
   \]
   In a \(q\)-thick building, \(D_t=\Omega(q^t)\).  At the natural
   immediate-neighbor scale \(p=\Theta(1/q)\), already \(t=2\) makes
   (0.3) equal to \(e^{-\Omega(q)}\).  High girth and building expansion
   therefore strengthen the transitive-skip escape.

The surviving class is narrow: height at least three, genuinely
non-series-parallel, activation below the dense \(3/4\) regime, and small
enough transitive descendant banks that skips do not erase local
contention.  A proof must charge whole causal runs, not sum matching
deficits at consecutive rank boundaries.

## 1. Exact metric/run identity

Let \(P\) be a finite poset.  Fix \(0<\varepsilon<1\), and on distinct
clients put
\[
d_\varepsilon(x,y)=
\begin{cases}
 \varepsilon,&x<_{P}y,\\
 1,&x\not<_{P}y.
\end{cases}
\tag{1.1}
\]
Set \(d_\varepsilon(x,x)=0\).
Give the depot \(r\) unit distance to and from every client.

This is a directed metric.  The only potentially dangerous triangle would
have two \(\varepsilon\)-arcs from \(x\) through \(y\) to \(z\); transitivity
then gives \(x<z\), whose direct distance is already \(\varepsilon\).
For three distinct clients, every other two-leg path has length at least
\(1+\varepsilon\), while every direct distance is at most one.  Triangles
with a repeated vertex are immediate, and depot triangles have two-leg
length at least one.

Fix an active realization \(A\), write \(N=|A|\), and let an active-call
order have \(K\) maximal strictly increasing runs.  It has \(N-K\)
comparable consecutive pairs and \(K-1\) breaks.  Therefore its
closed-depot cost is exactly
\[
\begin{aligned}
C_\varepsilon(A,K)
 &=2+\varepsilon(N-K)+(K-1)\\
 &=1+\varepsilon N+(1-\varepsilon)K.
\end{aligned}
\tag{1.2}
\]
By Dilworth,
\[
 \operatorname {OPT}_{\rm post}(A)
 =1+\varepsilon N+(1-\varepsilon)\operatorname {width}(P[A]).
\tag{1.3}
\]

Disjoint repetition amortizes the additive one, and
\(\varepsilon\downarrow0\) removes the common \(\varepsilon N\) term.
Thus the precise combinatorial target is
\[
         \inf_\pi\mathbb E K_\pi
             >\frac43\,\mathbb E\operatorname {width}(P[A]).
\tag{1.4}
\]

At a current active element \(x\), a causal policy may query arbitrary
uncalled elements of the strict upper set \(P_{>x}\).  Inactive answers do
not move it.  The first active answer continues the same run.  This
upper-bank probing power is the main difference from an externally ordered
online chain-partition problem.

## 2. Complete height-two ceiling

Let \(P\) have height at most two, with lower part \(L\) and upper part
\(R\).  Isolated elements may be placed on either side.  For a realization
\(A\), let
\[
 N(A)=|A|,\qquad
 \nu(A)=\nu\bigl(P[A\cap L,A\cap R]\bigr),
\tag{2.1}
\]
where comparability is treated as a bipartite edge.

Every chain contains at most one lower and one upper element.  Hence
Dilworth is exactly the matching identity
\[
                    \operatorname {width}(P[A])
                    =N(A)-\nu(A).
\tag{2.2}
\]

Use the following causal policy.

1. Choose a uniform random order of \(L\) and a uniform random ranking of
   \(R\).
2. Process lower elements in that order.  After an active lower element
   \(x\), query its still-uncalled upper neighbors in ranking order until
   the first active one is found.
3. After all lower elements are called, call the remaining upper elements.

Conditional on every fixed active set \(A\), this is RANKING on the fixed
realized bipartite graph with the online side in uniform random order.
Mahdian and Yan's random-arrival analysis gives
\[
               \mathbb E_{\rm seed}[M_\pi(A)]
                         \geq\beta\,\nu(A),
 \qquad \beta\geq0.696.
\tag{2.3}
\]
The run count is
\[
                         K_\pi(A)=N(A)-M_\pi(A).
\tag{2.4}
\]
Taking expectations and writing
\(\bar W=\mathbb E\operatorname {width}(P[A])\) gives
\[
\begin{aligned}
\mathbb E K_\pi
 &\leq \mathbb E[N-\beta\nu]\\
 &=\bar W+(1-\beta)\mathbb E\nu.
\end{aligned}
\tag{2.5}
\]
Since a matching uses two vertices,
\(\nu(A)\leq N(A)-\nu(A)=\operatorname {width}(P[A])\).
Thus
\[
\boxed{
 \frac{\mathbb E K_\pi}{\bar W}
 \leq2-\beta
 \leq1.304.
}
\tag{2.6}
\]

No independence assumption was used in (2.3)--(2.6).  Consequently all of
the following are excluded, with arbitrary independent unequal
activations:

* standard examples \(S_n\);
* crowns and crown blow-ups;
* projective-plane point--line incidence posets;
* generalized quadrangles, hexagons, and other generalized-polygon
  point--line incidence posets;
* arbitrary high-girth and Ramanujan bipartite incidence graphs viewed as
  height-two posets.

Expansion, girth, and algebraic symmetry cannot bypass (2.6), because the
policy is conditioned on the entire realized induced graph before the
RANKING inequality is applied.

## 3. Exact series-parallel obstruction

For disjoint posets \(P,Q\), write

* \(P\parallel Q\) when every cross pair is incomparable; and
* \(P\oplus Q\) when every element of \(P\) is below every element of
  \(Q\).

Widths obey
\[
\begin{aligned}
\operatorname {width}(P\parallel Q)
 &=\operatorname {width}(P)+\operatorname {width}(Q),\\
\operatorname {width}(P\oplus Q)
 &=\max\{\operatorname {width}(P),\operatorname {width}(Q)\}.
\end{aligned}
\tag{3.1}
\]

Starting from a singleton, these identities have matching causal
operations.

* For \(P\parallel Q\), run the two causal producers consecutively.
* For \(P\oplus Q\), produce chains in rounds: concatenate the \(j\)-th
  produced chain of \(P\) with the \(j\)-th produced chain of \(Q\).
  Empty rounds are detected through inactive calls, which cause no
  movement.

Induction gives a policy using exactly the realized width for every
series-parallel poset.  In particular,
\[
                  K_\pi(A)=\operatorname {width}(P[A])
                  \quad\text{for every }A.
\tag{3.2}
\]

The ordinal sum of antichains
\[
                         A_1\oplus\cdots\oplus A_h
\]
has the particularly transparent sweep policy.  Repeatedly take the first
active uncalled element from each layer in increasing layer order.  If
\(X_i\) is the number active in layer \(i\), the number of nonempty sweeps
is
\[
                         \max_iX_i
                         =\operatorname {width}(P[A]).
\tag{3.3}
\]

Parallel or ordinal recursion therefore cannot amplify a smaller chamber
gap.  A surviving poset must contain the four-element \(N\) obstruction to
series-parallel decomposition, and must use overlapping copies of it rather
than a tree of independent modules.

## 4. A general dense-activation ceiling

Let \(w=\operatorname {width}(P)\), and fix a maximum antichain
\[
                         B=\{b_1,\ldots,b_w\}.
\]
Choose a minimum chain cover
\[
                         P=C_1\cup\cdots\cup C_w
\tag{4.1}
\]
so that \(b_i\in C_i\).  Such a labeling exists because every chain meets
\(B\) at most once and a \(w\)-chain cover must cover all \(w\) members.

Use the fixed causal policy that processes the \(C_i\)'s consecutively and
queries each chain in increasing order.  The active elements of a nonempty
\(C_i\) form one increasing run; transitions between two chain blocks can
only reduce the count.  Therefore
\[
 K_\pi(A)
 \leq \sum_{i=1}^{w}\mathbf 1\{A\cap C_i\ne\varnothing\}.
\tag{4.2}
\]
Independence gives
\[
 \mathbb E K_\pi
 \leq
 \sum_{i=1}^{w}
 \left(1-\prod_{x\in C_i}(1-p_x)\right).
\tag{4.3}
\]
On the other hand, \(A\cap B\) is an antichain, so
\[
 \mathbb E\operatorname {width}(P[A])
 \geq\mathbb E|A\cap B|
 =\sum_{x\in B}p_x.
\tag{4.4}
\]
Equations (4.3)--(4.4) prove (0.2).

Two useful corollaries are immediate.

### Uniform activations

If every element is active with probability \(p\), then
\[
 \frac{\mathbb E K_\pi}
      {\mathbb E\operatorname {width}}
 \leq
 \frac{\sum_i(1-(1-p)^{|C_i|})}{pw}
 \leq\frac1p.
\tag{4.5}
\]
Hence \(p\geq3/4\) excludes a strict \(4/3\) gap for every finite poset.

### Dense maximum antichain

More generally, if \(p_x\geq\rho\) for every \(x\) in some maximum
antichain, then
\[
 \frac{\mathbb E K_\pi}
      {\mathbb E\operatorname {width}}
 \leq\frac1\rho.
\tag{4.6}
\]
Thus a positive construction needs a maximum-antichain bottleneck whose
expected active mass is below \(3w/4\), or it must use very nonuniform
probabilities and show that every maximum antichain has small probability
mass.

## 5. Interval orders: what survives

Represent an interval order by intervals \(I_x=[\ell_x,r_x]\), with
\[
                         x<y\quad\Longleftrightarrow
                         \quad r_x<\ell_y.
\tag{5.1}
\]
Its width is the maximum number of active intervals containing a common
point.

The following subclasses are already excluded.

* Height-two interval orders satisfy the \(1.304\) bound of Section 2.
* Interval orders that are also series-parallel satisfy the exact identity
  (3.2).
* Uniform activation \(p\geq3/4\) satisfies (4.5), without using the
  interval representation.

A tempting stronger claim is false: repeatedly extract the
earliest-finishing active compatible chain.  Consider four intervals
\[
\begin{aligned}
 P&=[1,2],& A&=[0,5],\\
 B&=[4,6],& Q&=[11/2,29/5].
\end{aligned}
\tag{5.2}
\]
The earliest-finish compatible chain is \(P,Q\).  Removing it leaves the
overlapping pair \(A,B\), so this extraction uses three chains, whereas
\[
                       P<B,\qquad A<Q
\]
give a two-chain cover.  The poset in (5.2) is the four-element
non-series-parallel \(N\).

This example is not a clairvoyance lower bound; a policy tailored to the
four intervals can choose the two good chains.  It only shows that the
canonical interval-scheduling greedy rule cannot prove a general gap-one
theorem.

There is nevertheless a local escape analogous to the building case.  From
a current active interval \(I\), every uncalled interval with
\(\ell_y>r_I\) is a legal continuation.  For any chosen bank \(D(I)\) of
such successors,
\[
 \Pr[\text{no active continuation in }D(I)]
     =\prod_{y\in D(I)}(1-p_y).
\tag{5.3}
\]
Dense interval orders expose large suffix banks and are therefore easy to
continue locally.  A positive interval-order construction would need many
overlapping \(N\)-choices but uniformly small usable suffix banks; neither
ordinary nested intervals nor dense endpoint grids have this combination.

No global \(4/3\) ceiling for all interval orders is proved here.

## 6. Buildings and algebraic incidence posets

Rank-two incidence posets of all generalized polygons are covered by
Section 2, regardless of their girth or spectral expansion.

Now consider a ranked incidence poset of rank at least three.  Let
\(D_t(x)\) be a set of descendants of \(x\), all \(t\) ranks above it.
After serving an active \(x\), a causal policy may query every member of
\(D_t(x)\) before leaving the current run.  It need not visit intermediate
faces: comparability is transitive in the metric (1.1).  Under uniform
activation \(p\),
\[
\Pr[D_t(x)\cap A=\varnothing]
                     =(1-p)^{|D_t(x)|}
                     \leq e^{-p|D_t(x)|}.
\tag{6.1}
\]

In the usual \(q\)-thick projective, polar, or building geometries, a
locally nondegenerate \(t\)-step descendant family has
\[
                         |D_t(x)|=\Omega(q^t).
\tag{6.2}
\]
The activation scale that leaves a constant expected number of active
immediate successors is \(p=c/q\).  Substitution in (6.1) gives, for
\(t=2\),
\[
 \Pr[D_2(x)\cap A=\varnothing]
                         \leq e^{-\Omega(cq)}.
\tag{6.3}
\]
Thus an immediate-interface matching failure almost never forces a new
run: the policy skips to a rank-two descendant.

Conversely, suppressing the skip by taking
\[
                         p=O(q^{-2})
\tag{6.4}
\]
makes the expected number of active immediate successors only \(O(1/q)\).
Then the posterior also has too few immediate transitions for a
boundary-by-boundary constant loss to dominate width.

This is the branching-versus-skip tension:
\[
 \boxed{
  pd=\Theta(1)\ \Longrightarrow\ pd^2=\Theta(d),
 }
\tag{6.5}
\]
where \(d\asymp q\) is the cover degree.  High girth validates the
\(\Theta(d^2)\) descendant count and therefore makes (6.3) stronger, not
weaker.

Equation (6.1) alone is not a global optimal-policy theorem—different runs
contend for the same descendants.  It is enough to invalidate a proof that
sums independent matching deficits over building ranks.

## 7. Exact surviving requirements

A candidate not excluded by this audit must satisfy all of the following.

1. **Height at least three.**  Height two has the strict \(1.304\) ceiling.
2. **Non-series-parallel overlap.**  Parallel and ordinal recursion has
   gap one.
3. **Sparse maximum-antichain mass.**  The general estimate (0.2) must
   exceed \(4/3\); uniform \(p\geq3/4\) is impossible.
4. **Controlled transitive closure.**  Useful immediate successors cannot
   generate huge active descendant banks, or causal skipping repairs the
   local loss.
5. **A whole-run potential.**  Posterior savings at different rank
   boundaries share intermediate vertices and transitive shortcuts.
   Summing local matching deficits double-counts the same eventual new run.

Standard examples, crowns, generalized polygons, ordinal-layered posets,
series-parallel interval orders, and ordinary high-degree finite buildings
all fail at least one item above.

The unresolved core is a low-descendant-growth, non-series-parallel,
height-at-least-three poset with product activations and an
algorithm-independent lower bound on the number of causal run starts.
