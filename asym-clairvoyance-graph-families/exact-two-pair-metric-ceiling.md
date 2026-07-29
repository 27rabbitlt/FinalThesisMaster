# Exact-two selectors: ordering reduction and a sharp two-level ceiling

## Scope

This note treats an open selector with a common zero/cheap source and sink
interface, conditional on exactly two selector vertices being active.  It gives
the exact weighted-linear-ordering reduction for an arbitrary finite directed
metric.  It then proves a sharp \(3/2\) ceiling for the important subclass of
two-level tournament metrics.

The argument below does **not** prove the same ceiling for every directed
metric.

## Exact reduction

Let \(V=\{1,\ldots,n\}\), let \(d\) be a directed metric on \(V\), and let the
weight of the active pair \(\{i,j\}\) be
\[
        w_{ij}=p_i p_j,\qquad p_i\geq 0.
\]
(A common normalization factor, for example from conditioning on exactly two
activations, cancels from every ratio.)

Indeed, if the original independent Bernoulli probabilities are \(r_i\), then
the probability that exactly \(\{i,j\}\) is active equals
\[
 \left(\prod_k(1-r_k)\right)
 \frac{r_i}{1-r_i}\frac{r_j}{1-r_j}.
\]
Thus conditioning on exactly two active vertices gives rank-one pair weights
with \(p_i=r_i/(1-r_i)\).

For each unordered pair put
\[
 a_{ij}:=\min\{d(i,j),d(j,i)\},\qquad
 q_{ij}:=|d(i,j)-d(j,i)|.
\]
Orient \(\{i,j\}\) as \(i\to j\) when \(d(i,j)\leq d(j,i)\), breaking ties
arbitrarily.  Denote this tournament by \(T_d\).

The posterior policy sees the active pair and chooses its cheaper orientation,
so its weighted pair cost is
\[
        M:=\sum_{\{i,j\}}p_i p_j a_{ij}.
\]
On the other hand, a causal first-active policy induces a linear order
\(\pi\) of the selector vertices.  For a pair whose cheap arc agrees with
\(\pi\), it pays \(a_{ij}\); for a cheap arc that is backward in \(\pi\), it
pays \(a_{ij}+q_{ij}\).  Consequently
\[
 C(\pi)
 =M+\sum_{\substack{i\to j\text{ in }T_d\\
                     j<_{\pi}i}}p_i p_j q_{ij}.
\]
Thus, exactly,
\[
 \boxed{\min_\pi C(\pi)
 =M+\operatorname{fas}(T_d;\,p_i p_jq_{ij}),}
\]
where \(\operatorname{fas}\) is minimum feedback-arc weight.  In particular,
a universal \(3/2\) ordering theorem would be precisely the inequality
\[
 \operatorname{fas}(T_d;\,p_i p_jq_{ij})\leq \frac12M.
\]

This reduction also identifies why independent pairwise minimization is not
generally causal: the cheap orientations may contain directed cycles.

## Two-level tournament metrics

Fix \(\lambda>0\) and an arbitrary tournament \(T\) on \(V\).  Define
\[
 d_T(i,j)=
 \begin{cases}
 0,&i=j,\\
 \lambda,&i\to j\text{ in }T,\\
 2\lambda,&j\to i\text{ in }T.
 \end{cases}
\]

### Lemma 1: \(d_T\) is a shortest-path directed metric

For three distinct vertices, \(d_T(i,k)\leq2\lambda\), while
\(d_T(i,j)+d_T(j,k)\geq2\lambda\).  The cases with repeated vertices are
immediate, so the directed triangle inequality holds.

Equivalently, start with the complete directed graph having the displayed arc
lengths and take shortest-path closure.  A one-arc cheap distance
\(\lambda\) cannot decrease.  A one-arc expensive distance \(2\lambda\)
cannot decrease either, because every alternative path has at least two
positive arcs and hence length at least \(2\lambda\).  Therefore closure
leaves all displayed distances unchanged.

### Theorem 2: the exact-two causal/posterior ratio is at most \(3/2\)

For \(d_T\), every pair has
\[
 a_{ij}=\lambda,\qquad q_{ij}=\lambda.
\]
Hence
\[
 M=\lambda\sum_{\{i,j\}}p_i p_j.
\]

Take any linear order \(\pi\) and its reverse \(\bar\pi\).  For every
tournament arc, exactly one of \(\pi,\bar\pi\) makes that arc backward.
Therefore their feedback weights add to
\[
 \sum_{\{i,j\}}\lambda p_i p_j=M.
\]
At least one of the two orders consequently has feedback weight at most
\(M/2\).  By the exact reduction,
\[
 \min_\pi C(\pi)\leq M+\frac12M=\frac32M.
\]
This holds for arbitrary nonnegative rank-one weights \(p_i p_j\); no
uniformity assumption is used.

## The constant \(3/2\) is asymptotically sharp in this class

It is enough to use uniform weights \(p_i=1\).  Let
\(N=\binom n2\), and orient every pair independently and uniformly at random.
For any fixed order \(\pi\), the number \(B_\pi\) of backward arcs has
distribution \(\operatorname{Bin}(N,1/2)\).  Hoeffding's inequality gives
\[
 \Pr[B_\pi\leq N/2-t]\leq \exp(-2t^2/N).
\]
Choose
\[
        t=n^{3/2}\sqrt{\log n}.
\]
By a union bound over the at most \(n!\) linear orders,
\[
 \Pr[\exists\pi:\ B_\pi\leq N/2-t]
 \leq n!\exp(-2t^2/N)
 \leq
 \exp\!\left(n\log n-\frac{2n^3\log n}{\binom n2}\right),
\]
which tends to zero.  Hence, for all sufficiently large \(n\), there exists a
tournament \(T_n\) satisfying
\[
        \operatorname{fas}(T_n)\geq N/2-t
        =\left(\frac12-o(1)\right)N.
\]
For its two-level metric \(d_{T_n}\),
\[
 \frac{\min_\pi C(\pi)}{M}
 =1+\frac{\operatorname{fas}(T_n)}{N}
 \geq \frac32-o(1).
\]
Together with Theorem 2, this proves that \(3/2\) is the sharp asymptotic
constant for two-level tournament metrics.

## Consequence for gadget searches

A construction whose pair geometry closes to “cheap tournament arc
\(\lambda\), reverse direction \(2\lambda\)” cannot yield a strict
asymptotic ratio above \(3/2\), even with nonuniform independent activation
parameters.  Quasirandom tournament orientations already approach the ceiling,
so improving the orientation combinatorics inside this metric class cannot
break it.  A \(>3/2\) exact-two selector, if one exists in an arbitrary
directed metric, must exploit genuinely nonuniform pair baselines and
asymmetry margins rather than a two-level tournament closure.
