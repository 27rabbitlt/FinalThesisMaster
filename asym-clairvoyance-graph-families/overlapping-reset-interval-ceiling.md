# Overlapping reset intervals: an exact \(4/3\) ceiling

## Verdict

The natural nonlaminar two-reset quotient does not beat \(4/3\), for any
positive arc lengths or product activation probabilities.  In fact the
argument below works for an arbitrary joint law of the two activation bits.

The reason is structural.  Although the reset intervals overlap on the
permanent chain, every realization has only two relevant open traces:

1. a forward trace which omits every inactive reset path; and
2. a reverse trace which uses both reset paths and omits the two outer chain
   arcs.

A causal policy can commit to either trace.  A bounded-variable prophet
inequality then gives the sharp \(4/3\) ceiling.  Allowing interrupted service
with a nonnegative toll cannot improve the local surplus needed by a
recursive construction.

## 1. The quotient and its metric closure

Let \(A,B,C,D\) be permanent clients, all usable as open-service ports.  The
generating digraph consists of
\[
 A\xrightarrow{\alpha}B\xrightarrow{\beta}C
  \xrightarrow{\gamma}D,
 \qquad
 C\xrightarrow{\mu}x\xrightarrow{\nu}A,
 \qquad
 D\xrightarrow{\sigma}y\xrightarrow{\tau}B,
 \tag{1.1}
\]
where all seven lengths are positive.  The two selector clients \(x,y\) are
stochastic.  Distances are directed shortest-path distances in (1.1).
Inactive selectors remain available as transit vertices, as required by the
stochastic-TSP model.

Put
\[
 h:=\alpha+\gamma,\qquad
 X:=\mu+\nu,\qquad
 Y:=\sigma+\tau,\qquad
 W:=\beta+h+X+Y.
 \tag{1.2}
\]
Write \(I_x,I_y\) for the indicators that the respective selectors are
inactive, and define the realized clear-reset length
\[
 S:=XI_x+YI_y,\qquad 0\le S\le C_0:=X+Y.
 \tag{1.3}
\]

Working with the generating graph loses nothing.  Every metric move can be
expanded into a generating-graph walk of the same length, and passing through
a selector does not serve it unless it was called active.

## 2. Exact realization-wise posterior value

### Lemma 1

For every realization,
\[
 F(I_x,I_y)=W-\max\{h,S\}.
 \tag{2.1}
\]

### Proof

There are two explicit services.

The forward service uses the permanent chain and precisely the reset paths
whose selectors are active.  Its four possible traces are
\[
\begin{array}{c|l}
(x,y)\text{ active?}&\text{trace}\\ \hline
(0,0)&A\to B\to C\to D,\\
(1,0)&C\to x\to A\to B\to C\to D,\\
(0,1)&A\to B\to C\to D\to y\to B,\\
(1,1)&C\to x\to A\to B\to C\to D\to y\to B.
\end{array}
\tag{2.2}
\]
Its cost is \(W-S\).

The reverse service is
\[
 D\to y\to B\to C\to x\to A.
 \tag{2.3}
\]
It is valid in every realization, because inactive selectors may be used as
transit vertices.  Its cost is
\[
 Y+\beta+X=W-h.
\]
Thus \(F\le W-\max\{h,S\}\).

For the reverse inequality, expand an arbitrary open service into (1.1).
The arc \(B\to C\) must occur.  Moreover:

- if both outer chain arcs \(A\to B\) and \(C\to D\) occur, the walk pays
  \(\beta+h\), and it must additionally traverse the full reset path of
  every active selector; hence it costs at least \(W-S\);
- if \(A\to B\) is absent, then \(A\) must be the terminal permanent vertex.
  Reaching it requires the full \(x\)-reset path.  Since \(D\) must also be
  served before that terminal visit, leaving \(D\) requires the full
  \(y\)-reset path.  Together with \(B\to C\), the cost is at least
  \(X+\beta+Y=W-h\);
- if \(C\to D\) is absent, then \(D\) must be the initial permanent vertex.
  Leaving it requires the \(y\)-reset path, and subsequently reaching \(A\)
  requires the \(x\)-reset path.  Again the cost is at least \(W-h\).

All omitted cases contain one of these cases plus additional positive arc
occurrences.  Therefore every service costs at least
\(\min\{W-S,W-h\}=W-\max\{h,S\}\).  \(\square\)

Consequently the exact expected posterior value is
\[
 P=W-\mathbb E[\max\{h,S\}].
 \tag{2.4}
\]

For independent activations with
\(\Pr[x\text{ active}]=p\) and
\(\Pr[y\text{ active}]=q\),
\[
 \mathbb E S=(1-p)X+(1-q)Y,
\]
but independence will not be used below.

## 3. Two legal causal policies

Let
\[
 e:=\mathbb E S,\qquad m:=\max\{h,e\}.
 \tag{3.1}
\]

There is a causal forward policy whose realization-wise cost is exactly
\(W-S\).  While still in the exterior state, call \(x\).

- If \(x\) is active, enter at port \(C\), traverse
  \(C\to x\to A\), and then serve the permanent chain
  \(A\to B\to C\to D\).
- If \(x\) is inactive, no movement occurs; enter at \(A\) and serve the
  permanent chain.
- Upon reaching \(D\), call \(y\).  If it is active, traverse
  \(D\to y\to B\); if it is inactive, no movement occurs.

The final position is a permanent port.  Hence this policy has expected cost
\[
 W-e.
 \tag{3.2}
\]

There is also a causal reverse policy of cost at most \(W-h\): follow the
fixed order
\[
 D,y,B,C,x,A
\]
and shortcut inactive calls.  Metric shortcutting can only decrease the
length of the full reverse trace (2.3).

It follows that
\[
 A:=\operatorname {OPT}_{\rm adapt}^{\rm open}
 \le \min\{W-e,W-h\}=W-m.
 \tag{3.3}
\]

The exterior call to \(x\) is important.  If \(x\) is inactive it reveals
that fact without choosing an entry port; if it is active, the policy may
enter at \(C\).  Thus the apparent realization-dependent starting point in
(2.2) does not create an adaptive obstruction.

## 4. A bounded-variable prophet inequality

### Lemma 2

If \(0\le S\le C_0\), \(e=\mathbb E S\), and
\(m=\max\{h,e\}\), then
\[
 \mathbb E[\max\{h,S\}]
 \le \frac{h+C_0+3m}{4}.
 \tag{4.1}
\]

### Proof

The assertion is immediate if \(h\ge C_0\), so assume \(h<C_0\).  Convexity
of \(s\mapsto(s-h)_+\), or simply the chord from \(h\) to \(C_0\), gives
\[
 (S-h)_+\le \frac{C_0-h}{C_0}S.
\]
Therefore
\[
 \mathbb E[\max\{h,S\}]
 \le h+\frac{C_0-h}{C_0}e.
 \tag{4.2}
\]

Normalize \(z=h/C_0\) and \(u=e/C_0\).

If \(e\le h\), then \(m=h\), \(u\le z\), and
\[
 (1-z)u\le z(1-z)\le\frac14.
\]
This is precisely (4.1) after subtracting \(h\) and dividing by \(C_0\).

If \(e\ge h\), then \(m=e\) and \(z\le u\).  The difference between the
right side of (4.1) and the normalized upper bound (4.2), multiplied by
four, is
\[
 1-u-z(3-4u).
\]
For \(u\le3/4\), this is at least
\[
 1-u-u(3-4u)=(1-2u)^2.
\]
For \(u\ge3/4\), it is at least \(1-u\), because \(3-4u\le0\).
It is nonnegative in both cases.  \(\square\)

Since \(W=\beta+h+C_0\ge h+C_0\), Lemma 2 yields
\[
\begin{aligned}
 P
 &=W-\mathbb E[\max\{h,S\}]\\
 &\ge W-\frac{W+3m}{4}
 =\frac34(W-m).
\end{aligned}
\tag{4.3}
\]
Combining (3.3) and (4.3) proves
\[
 \boxed{\displaystyle \frac{A}{P}\le\frac43}.
 \tag{4.4}
\]

The proof did not use independence, so correlation between the two selector
bits cannot rescue this quotient.

## 5. Interrupted-service accounting

Let \(H(t)\) be the optimum adaptive open value when service may be split
into any number of pieces and each piece after the first incurs a
nonnegative toll \(t\).  A one-piece execution is allowed, so
\[
 H(t)\le A\le\frac43P.
 \tag{5.1}
\]
In particular,
\[
 3H(t)-4P\le0.
 \tag{5.2}
\]
Thus the strict local inequality required by a shared-lap or recursive toll
accounting,
\[
 3H(t)-4P>t,
\]
fails for every \(t\ge0\), before any port-diameter or shortest-path issue is
considered.

## Consequence

Geometric crossing of the reset intervals is insufficient.  A successful
quotient must make the realized reset requirements **order-incompatible**:
there must be no pair of causal traces consisting of

- one trace omitting all currently clear reset length, and
- one realization-independent reverse trace omitting a fixed complementary
  length.

The quotient (1.1) has exactly those two traces, and the bounded-variable
inequality forces the sharp \(4/3\) ceiling.
