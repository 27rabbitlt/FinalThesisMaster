# Weighted common-payload mode codes have a \(3/2\) ceiling

## Outcome

This note strengthens the equal-layer suffix calculation in
`mode-coded-common-payload-obstruction.md`.  Equal edge lengths are not
needed.  Even in the deliberately favorable idealization in which every
additional active mode is charged a fresh full posterior traversal, an
arbitrarily weighted common-payload permutation code has open ratio at most
\(3/2\).

The reason is a weighted endpoint averaging identity.  If mode \(j\) serves
the common payload on a path of length at most \(L\), ending at its
distinguished payload endpoint \(t_j\), then a uniformly chosen endpoint
\(t_K\) has mean mode-\(j\) suffix to \(t_j\) at most
\[
                         \frac{m-1}{m}L.             \tag{0.1}
\]
This remains true if nearly all of the path length is concentrated on one
edge.

For \(m\) independent probability-\(p\) selectors, put
\[
 q=(1-p)^m,\qquad \lambda=mp,\qquad
 \alpha=\frac{m-1}{m}.
\]
In the favorable full-reset model the posterior value is
\[
                         P=L(q+\lambda),             \tag{0.2}
\]
where the \(qL\) term serves the payload when no selector is active and each
active selector costs one mode traversal.  The endpoint-randomized causal
policy has
\[
                         A\le P+\alpha L(1-q).        \tag{0.3}
\]
The exact scalar inequality
\[
                         2\alpha(1-q)\le q+\lambda   \tag{0.4}
\]
then gives
\[
                         \boxed{A/P\le3/2}.           \tag{0.5}
\]

Thus neither unequal layer weights nor concentrating the entire payload
length immediately before the mode-specific endpoint can produce the
strict \(>3/2\) chamber needed by the serial-reset amplifier.  A surviving
positive construction must cease to be one common Hamiltonian payload
followed by a mode suffix.  It needs genuinely interacting cuts whose
repair cannot be charged to one endpoint suffix.

## 1. Weighted suffix averaging

Let \(U\) be a common permanent payload containing distinguished vertices
\[
                         t_1,\ldots,t_m.
\]
For each mode \(j\), let \(P_j\) be a directed path that serves all of \(U\)
and ends at \(t_j\).  The path may contain transit vertices and may assign
arbitrary nonnegative lengths to its steps.  Assume
\[
                         \operatorname{len}(P_j)\le L. \tag{1.1}
\]

For \(k\in[m]\), let \(S_j(k)\) be the length of the suffix of \(P_j\)
beginning at the occurrence of \(t_k\) and ending at \(t_j\).  This excludes
any append segment after \(t_j\) leading to the selector or output gate; that
append segment is already present in the matching mode's posterior service.

### Lemma 1

For every mode \(j\),
\[
                 \frac1m\sum_{k=1}^m S_j(k)
                         \le\frac{m-1}{m}L.           \tag{1.2}
\]

### Proof

Orient \(P_j\) from its first payload service toward \(t_j\).  A path edge
is contained in \(S_j(k)\) only when \(t_k\) occurs before that edge.
Because \(t_j\) is the terminal distinguished payload vertex, at most
\(m-1\) of the \(m\) distinguished endpoints can occur before any edge.
Thus every edge length is counted at most \(m-1\) times in
\(\sum_kS_j(k)\).  Summing edge by edge gives
\[
 \sum_{k=1}^mS_j(k)
       \le(m-1)\operatorname{len}(P_j)
       \le(m-1)L.
\]
Division by \(m\) proves (1.2). \(\square\)

This argument is insensitive to the distribution of edge lengths.  In
particular, putting one edge of length \(L-o(L)\) immediately before \(t_j\)
can make almost every wrong endpoint pay almost \(L\), but the one matching
endpoint pays zero; (1.2) is still sharp.

## 2. Independent selectors and the favorable reset model

Let selectors \(s_1,\ldots,s_m\) be independently active with common
probability \(p\), and let
\[
                         Z\sim\operatorname{Binomial}(m,p).
\]
We grant the proposed construction its most favorable intended accounting:

1. if \(Z=0\), the common payload costs \(L\);
2. if \(Z\ge1\), every active selector is served by one full
   mode-consistent traversal of cost \(L\);
3. after the first active selector, resets completely prevent batching
   between distinct active modes; and
4. after serving the payload in mode \(K\), reaching the first active
   selector of mode \(J\) costs at most its posterior append segment plus
   the payload suffix \(S_J(K)\).

Under items 1--3 the realization-wise posterior cost is
\[
                         L\max\{1,Z\},
\]
and therefore
\[
\begin{aligned}
 P
 &=L\,\mathbb E\max\{1,Z\}\\
 &=L\bigl(\Pr[Z=0]+\mathbb EZ\bigr)\\
 &=L(q+\lambda),
\end{aligned}                                         \tag{2.1}
\]
with \(q=(1-p)^m\) and \(\lambda=mp\).

Use this causal policy:

1. choose \(K\) uniformly in \([m]\), independently of all activations;
2. serve the common payload in order \(P_K\);
3. query the selectors in an independent uniform order;
4. if a first active selector has mode \(J\), use the mode-\(J\) suffix
   from the current endpoint \(t_K\), and thereafter use the granted full
   reset service for every further active selector.

Conditional on the nonempty active selector set, the first active mode
\(J\) in a uniform query order is uniform on that active set.  The
independent choice \(K\) is uniform regardless of \(J\).  Lemma 1 hence
gives conditional expected extra suffix at most \(\alpha L\), where
\[
                         \alpha=(m-1)/m.
\]
There is no suffix when \(Z=0\).  All other movement was already granted in
the posterior accounting (2.1).  Consequently
\[
                         A\le P+\alpha L(1-q),        \tag{2.2}
\]
which is (0.3).

Any real metric shortcut, shared reset, or batching between active modes can
only lower this causal upper bound.  Thus the favorable model is the correct
one in which to prove an impossibility statement.

## 3. The exact scalar inequality

It remains to prove
\[
       2\frac{m-1}{m}\bigl(1-(1-p)^m\bigr)
                    \le (1-p)^m+mp.                 \tag{3.1}
\]
After multiplying by \(m\), this is equivalent to
\[
 f_m(p):=(3m-2)(1-p)^m+m^2p-(2m-2)\ge0.              \tag{3.2}
\]

The endpoint values are
\[
                         f_m(0)=m\ge0,\qquad
                         f_m(1)=(m-1)^2+1>0.         \tag{3.3}
\]
There is at most one interior critical point.  If it exists, write
\[
 x=1-p,\qquad
 x^{m-1}=\frac{m}{3m-2}.                             \tag{3.4}
\]
At that point,
\[
\begin{aligned}
 f_m(p)
 &=(3m-2)x^m+m^2(1-x)-(2m-2)\\
 &=m^2-2m+2-m(m-1)x.                                \tag{3.5}
\end{aligned}
\]

We claim
\[
 x\le1-\frac{m-2}{m(m-1)}.                           \tag{3.6}
\]
For \(m=1\) the original assertion is immediate, and for \(m=2\) (3.6) is
trivial.  For \(m\ge3\), Bernoulli's inequality gives
\[
\left(1-\frac{m-2}{m(m-1)}\right)^{m-1}
 \ge1-\frac{m-2}{m}
 =\frac2m.                                           \tag{3.7}
\]
Since
\[
                         \frac2m\ge\frac{m}{3m-2}
\quad\Longleftrightarrow\quad
                         (m-2)^2\ge0,                \tag{3.8}
\]
raising (3.6) to the positive power \(m-1\) and using (3.4) proves the
claim.

Substitution of (3.6) into (3.5) yields
\[
\begin{aligned}
 f_m(p)
 &\ge m^2-2m+2
      -m(m-1)\left(1-\frac{m-2}{m(m-1)}\right)\\
 &=0.
\end{aligned}
\]
Thus (3.1) holds for every integer \(m\ge1\) and every \(p\in[0,1]\).

Finally, (2.2) and (3.1) give
\[
 \frac AP
 \le1+\frac{\alpha(1-q)}{q+\lambda}
 \le1+\frac12
 =\frac32.
\]

## 4. Scope

The theorem covers a broad endpoint-code mechanism, not every possible
multi-cut chamber.  Its hypotheses are exactly the attractive
common-payload proposal:

* every mode is a permutation path through the same permanent payload;
* a mode is identified by an independently active selector;
* after a payload order is chosen, a wrong first mode is repaired through a
  suffix of that mode's payload path; and
* later active modes are separated by full resets.

The proof already grants the strongest reset separation, permits arbitrary
path weights, and charges a wrong endpoint by its complete directed suffix.
Therefore making returns more asymmetric or concentrating length near the
mode endpoint cannot cross \(3/2\).

The remaining positive topology must make one decision constrain at least
two cuts in a way that is not representable by one Hamiltonian payload
suffix.  In particular, its posterior savings must be genuinely
nonlaminar: the repair of a wrong mode must not be a suffix of any one
realization-wise posterior service.
