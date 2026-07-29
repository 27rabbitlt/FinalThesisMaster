# High-asymmetry exact-two selector metrics: cycle audit and a pivot route

## 1. Exact objective

Let \(d\) be a finite directed metric and let the selector weights be
\(p_i\geq 0\).  For an unordered pair \(e=\{i,j\}\), write

\[
 a_e=\min\{d(i,j),d(j,i)\},\qquad
 q_e=|d(i,j)-d(j,i)|,\qquad w_e=p_ip_j,
\]

and orient \(e\) in its cheaper direction.  Conditional on exactly two
active selectors, the posterior and best causal costs (after deleting common
zero-cost ports) are

\[
 M=\sum_e w_ea_e,\qquad
 C^*=M+\min_\pi\sum_{e\text{ backward in }\pi}w_eq_e.
\]

Thus a strict ratio above \(3/2\) is equivalent to

\[
 \operatorname{fas}(T_d;w_eq_e)>\frac12M.                 \tag{1}
\]

The ordinary order/reverse-order argument only gives
\(\operatorname{fas}\leq\frac12\sum_e w_eq_e\).  Consequently a candidate
for (1) needs total relevant asymmetry larger than total baseline, while also
preventing an order from placing the large-asymmetry arcs forward.

## 2. The cyclic-triangle constraint

Suppose the cheap orientations on \(i,j,k\) form

\[
        i\longrightarrow j\longrightarrow k\longrightarrow i .
\]

The three reverse triangle inequalities give

\[
\begin{aligned}
 a_{ij}+q_{ij}&\leq a_{jk}+a_{ki},\\
 a_{jk}+q_{jk}&\leq a_{ki}+a_{ij},\\
 a_{ki}+q_{ki}&\leq a_{ij}+a_{jk}.
\end{aligned}
\]

In particular,

\[
 q_{ij}+q_{jk}+q_{ki}
 \leq a_{ij}+a_{jk}+a_{ki}.                              \tag{2}
\]

This explains why triangle-rich, orbit-symmetric metrics stop at \(3/2\).
An improvement must concentrate \(q_e>a_e\) on longer directed cycles.

## 3. Exact audit of the one-way cyclic metric

Let \(n=2m+1\), let \(V=\mathbb Z_n\), and take the shortest-path metric of
the unit one-way cycle:

\[
                 d(i,j)=(j-i)\pmod n\in\{0,1,\ldots,n-1\}.
\]

Use uniform selector weights.  For every \(k\in\{1,\ldots,m\}\), there are
exactly \(n\) unordered pairs whose cheap direction is clockwise displacement
\(k\).  Their baseline and margin are

\[
                 a_k=k,\qquad q_k=n-2k.
\]

Therefore

\[
 M=n\sum_{k=1}^m k
   =\frac{nm(m+1)}2.                                     \tag{3}
\]

Consider the fixed order \(0,1,\ldots,n-1\).  Among the \(n\) cheap arcs of
displacement \(k\), exactly the \(k\) arcs crossing the cut are backward.
Its feedback penalty is consequently

\[
\begin{aligned}
 B
 &=\sum_{k=1}^m k(n-2k)\\
 &=n\frac{m(m+1)}2
   -2\frac{m(m+1)(2m+1)}6\\
 &=\frac{nm(m+1)}6
  =\frac13M.                                             \tag{4}
\end{aligned}
\]

Thus

\[
             \frac{C^*}{M}\leq 1+\frac BM=\frac43.       \tag{5}
\]

This rules out the most direct high-girth/high-asymmetry Cayley candidate.
The adjacent preferred arcs have asymmetry ratio \(n-1\), but the cheap
longer chords forced by metric closure contribute enough posterior baseline
to reduce the ordering penalty to \(M/3\).

The same calculation also shows why merely increasing the girth of a
preferred cycle is not useful: the reverse length grows linearly with the
girth, but all intermediate-displacement pairs appear at the same leading
order under rank-one exact-two weights.

## 4. A sufficient pivot lemma for a universal \(3/2\) theorem

The following isolates a clean remaining theorem.  For a vertex \(v\), put

\[
 A_v=\{u:u\to v\},\qquad B_v=\{u:v\to u\}.
\]

The pivot order recursively orders \(A_v\), then places \(v\), then
recursively orders \(B_v\).  Its only new backward arcs are

\[
 b\to a,\qquad a\in A_v,\ b\in B_v,
\]

and every such arc is the third side of the cyclic triangle

\[
                 a\to v\to b\to a.
\]

Define

\[
\begin{aligned}
 X_v&=\sum_{\substack{a\in A_v,\ b\in B_v\\b\to a}}
             p_ap_bq_{ab},\\
 N_v&=\sum_{\substack{a\in A_v,\ b\in B_v}}p_ap_ba_{ab}
       +\sum_{u\ne v}p_up_va_{uv}.
\end{aligned}
\]

If every nonempty induced subinstance has a vertex \(v\) satisfying

\[
                       2X_v\leq N_v,                     \tag{6}
\]

then induction proves the desired universal ceiling.  Indeed, the posterior
baseline decomposes as

\[
 M=M(A_v)+M(B_v)+N_v.
\]

Applying induction inside the two sides and then (6) gives

\[
 \operatorname{fas}
 \leq \frac12M(A_v)+\frac12M(B_v)+X_v
 \leq\frac12M.
\]

Hence (6) would imply \(C^*/M\leq3/2\) for every directed metric and every
rank-one weight system.

The triangle inequality gives the useful pointwise estimate

\[
 q_{ab}\leq a_{av}+a_{vb}-a_{ab}
 \quad\text{whenever }a\to v\to b\to a,                  \tag{7}
\]

but summing (7) directly loses the factor \(1/2\).  A fixed pivot can violate
(6), so the missing point is genuinely the existence of a good pivot, not a
per-vertex inequality.  Simple averaging over pivots is also insufficient
for nonuniform \(p_i\).

## 5. Consequence for the construction search

No explicit directed metric with (1) is obtained here.  The audit does give
two rigorous restrictions:

1. the canonical high-asymmetry directed-cycle/Cayley metric has ratio at
   most \(4/3\), not above \(3/2\);
2. any exact-two counterexample must defeat the recursive cyclic-triangle
   pivot inequality (6) on at least one induced subinstance.

In particular, large pairwise asymmetry alone is not enough.  A successful
metric would need several interacting long cycles arranged so that every
possible pivot exposes more than half of the newly exposed posterior
baseline as weighted crossing regret.

## 6. The \(1/2\) feedback bound, if true, is sharp

There cannot be a universal improvement on the coefficient \(1/2\) in
\(\operatorname{fas}\leq M/2\).  Let \(T_n\) be any tournament and define

\[
 d(i,j)=
 \begin{cases}
 1,&i\to j\text{ in }T_n,\\
 2,&j\to i\text{ in }T_n.
 \end{cases}
\]

Every nonzero distance lies in \([1,2]\), so the directed triangle
inequalities hold automatically.  Here \(a_{ij}=q_{ij}=1\), and therefore

\[
 M=\binom n2,\qquad
 \operatorname{fas}(T_d;q)=\operatorname{fas}(T_n).
\]

Tournaments whose minimum feedback arc set has
\((1/2-o(1))\binom n2\) arcs consequently give
\(\operatorname{fas}/M=1/2-o(1)\).  Thus a proof of (6), or any replacement
for it, must use the full constant \(1/2\).

## 7. Hypergraph-cover reformulation and a failed local interpolation

Mastrolilli's hemimetric MinFAS theorem gives a useful exact reformulation.
With variables \(\delta_{ij}\), the covering system is

\[
\begin{aligned}
 \delta_{ij}+\delta_{ji}&\geq1,\\
 \delta_{ij}+\delta_{jk}+\delta_{ki}&\geq1
       \qquad(i,j,k\text{ distinct}),\\
 \delta_{ij}&\in\{0,1\}.
\end{aligned}
\]

For hemimetric objective weights \(d(i,j)\), every integral feasible cover
can be repaired into an actual order without increasing its objective.
Consequently the desired theorem would follow from constructing a cover of
cost at most \(3M/2\).  This is an exact route, but it does not by itself
supply that cover.

A natural fractional attempt illustrates the remaining obstruction.  On a
pair whose cheap direction costs \(a\) and expensive direction costs
\(a+q\), put

\[
 t=\min\left\{\frac12,\frac{a}{2q}\right\},\qquad
 \delta_{\rm cheap}=1-t,\quad \delta_{\rm expensive}=t. \tag{8}
\]

For a cyclic cheap triangle the metric inequalities imply

\[
 q_i\leq a_j+a_k-a_i\quad(i=1,2,3).
\]

Writing \(A=a_1+a_2+a_3\), convexity gives

\[
 \sum_i\min\left\{1,\frac{a_i}{q_i}\right\}\geq2.
\]

Hence the three \(t\)'s in (8) sum to at least \(1\), while the regret paid
by (8) is pointwise

\[
 tq\leq a/2.
\]

So (8) has exactly the desired global objective and satisfies every
constraint coming from a cyclic cheap triangle.  It can nevertheless fail
on a *transitive* cheap triangle.  A concrete directed metric is

\[
\begin{array}{c|ccc}
 &x&y&z\\ \hline
x&0&1&2\\
y&5&0&1\\
z&4&5&0
\end{array}
\]

(all six triangle inequalities are immediate).  Its cheap orientation is
\(x\to y\to z\), \(x\to z\).  Formula (8) gives

\[
 t_{xy}=t_{yz}=\frac18,\qquad t_{xz}=\frac12,
\]

whereas the reverse cyclic constraint requires
\(t_{xz}\leq t_{xy}+t_{yz}\).  Thus local cyclic-triangle charging alone is
not a proof; transitive triangles couple the charges.

## 8. Weight cloning

It is enough to prove the order bound for uniform vertex weights.  For
rational \(p_i\), replace vertex \(i\) by \(m_i\) zero-distance clones, where
\(m_i\) is a common integer scaling of \(p_i\), and retain distance
\(d(i,j)\) between clones of different types.  This is a directed
pseudometric.  If an order of the clones has cost at most \(3/2\) times its
pairwise minimum baseline, choose one clone uniformly from each type.
The expected weighted cost of the induced type order is exactly the clone
order's cross-type cost.  Some induced type order is therefore no worse.
Approximation extends the conclusion from rational to real \(p_i\).

This removes the rank-one weights as a source of difficulty, although a
uniform-weight proof is still missing.
