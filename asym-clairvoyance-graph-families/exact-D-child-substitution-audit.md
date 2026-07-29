# Exact \(D\)-metric child-substitution audit

## 1. Quotient and rebate

The top quotient is the exact directed metric

\[
D=
\begin{pmatrix}
0&5&4&7\\
3&0&5&7\\
4&1&0&3\\
1&6&5&0
\end{pmatrix},
\qquad
(p_a,p_b,p_x,p_y)=\left(1,1,\frac{13}{25},\frac15\right),
\]

with row and column order \(a,b,x,y\).  The four states
\(\varnothing,x,y,xy\) have probabilities

\[
       \frac1{125}(48,52,12,13).
\]

Replace each permanent port \(a,b\) by a child block of relative scale
\(r\).  Joining two pieces of the same child costs at most \(5r\), so put

\[
       \delta:=5r.
\]

After applying the interrupted-child inequality, the required top
quantity is

\[
 \text{top-walk length}
 -\delta\bigl((N_a-1)+(N_b-1)\bigr),
 \tag{1}
\]

where \(N_a,N_b\) are the numbers of service-containing pieces of the two
children.

The hoped-for statement was that (1) still has fixed-state values
\((3,5,8,9)\) and adaptive value \(33/5\).  It is false.

## 2. Fixed-state failure

In state \(xy\), use the quotient trace

\[
       b\longrightarrow a\longrightarrow x
       \longrightarrow y\longrightarrow a.
\]

Its metric length is

\[
       3+4+3+1=11.
\]

It has two \(a\)-pieces, so its adjusted length is

\[
       11-\delta.
\]

Consequently, as soon as \(\delta>2\), the claimed fixed-state lower bound
\(9\) fails.  At the proposed near-critical value
\(r=1/2\), hence \(\delta=5/2\), the adjusted state cost is \(17/2<9\).

For \(0\le\delta\le3\), the exact relaxed fixed-state values are

\[
\begin{array}{c|c}
\text{state}&F_s(\delta)\\ \hline
\varnothing&3,\\
x&5,\\
y&8,\\
xy&\min\{9,11-\delta\}.
\end{array}
\tag{2}
\]

To verify (2), list the simple port-to-port paths containing the required
selectors.  A repeated \(a\)- or \(b\)-piece contains a cycle before it can
earn another rebate.  For \(\delta\le3\), deleting every cycle except the
displayed \(b,a,x,y,a\) detour cannot increase adjusted cost.  The best
competing repeated-port traces are \(b,a,x,b\), of value \(8-\delta\), and
\(b,a,y,a\), of value \(11-\delta\); they do not beat \(5\) and \(8\),
respectively, in this range.

Thus the relaxed posterior expectation is

\[
P_D(\delta)=
\begin{cases}
\dfrac{617}{125},&0\le\delta\le2,\\[0.6em]
\dfrac{643-13\delta}{125},&2\le\delta\le3.
\end{cases}
\tag{3}
\]

At \(\delta=5/2\),

\[
       P_D(5/2)=\frac{1221}{250}=4.884.
\]

## 3. Adaptive failure and sharp relaxed value

Two causal policies give the complete lower envelope.

### Policy I

Start at \(b\) and call \(y\).  If \(y\) is active, move
\(b\to y\to a\); otherwise move \(b\to a\).  Then call \(x\), and if it is
active move \(a\to x\to b\).

Its adjusted state costs are

\[
       (3,\ 8-\delta,\ 8,\ 13-\delta),
\]

so its expectation is

\[
       A_1(\delta)=\frac{33}{5}-\frac{13}{25}\delta.
\tag{4}
\]

### Policy II

Start with \(b\to a\), call \(x\), and then call \(y\).  If \(x\) is the
only active selector, finish through \(b\).  If \(y\) is active, use the
terminal return to \(a\).  Its adjusted state costs are

\[
       (3,\ 8-\delta,\ 11-\delta,\ 11-\delta),
\]

and hence

\[
       A_2(\delta)=\frac{167}{25}-\frac{77}{125}\delta.
\tag{5}
\]

For \(0\le\delta\le5/2\), the exact relaxed Bellman value is

\[
A_D(\delta)=
\min\{A_1(\delta),A_2(\delta)\}
=
\begin{cases}
\dfrac{33}{5}-\dfrac{13}{25}\delta,
   &0\le\delta\le\dfrac56,\\[0.8em]
\dfrac{167}{25}-\dfrac{77}{125}\delta,
   &\dfrac56\le\delta\le\dfrac52.
\end{cases}
\tag{6}
\]

For completeness, (6) can be checked without numerical optimization.
Use a Bellman state consisting of the current quotient vertex, the set of
called selectors, their revealed active subset, the served ports, and which
port was the most recent service piece.  A return to a previously served
port receives rebate \(\delta\).  Positive adjusted cycle lengths for
\(\delta\le5/2\) allow deletion of every cycle that reveals no new selector
or port.  Substitution in the remaining finite recursion leaves exactly two
nondominated first effective traces: Policy I and Policy II.  Their
crossover is

\[
       A_1(\delta)=A_2(\delta)
       \quad\Longleftrightarrow\quad
       \delta=\frac56.
\]

In particular,

\[
       A_D(5/2)=\frac{257}{50}=5.14,
\tag{7}
\]

not \(33/5=6.6\).  Therefore the desired adaptive top inequality fails by
\(73/50\) at the proposed scale.

## 4. Binary-tree scale audit

At every parent, two children of relative scale \(r\) have total scale

\[
       q=2r=\frac{2\delta}{5}.
\tag{8}
\]

Near-critical substitution means \(q\simeq1\), hence
\(\delta\simeq5/2\).  But the local adaptive-versus-natural-posterior
surplus is

\[
T(\delta):=3A_D(\delta)-4\frac{617}{125}.
\tag{9}
\]

Equations (6) and (9) give

\[
T(\delta)=
\begin{cases}
\dfrac{7-195\delta}{125},
   &0\le\delta\le\dfrac56,\\[0.8em]
\dfrac{37-231\delta}{125},
   &\dfrac56\le\delta\le\dfrac52.
\end{cases}
\tag{10}
\]

It is already negative for every

\[
       \delta>\frac7{195}.
\]

At the near-critical point,

\[
       T(5/2)=-\frac{1081}{250}=-4.324.
\tag{11}
\]

Using the smaller relaxed posterior value from (3) does not help:

\[
3A_D(\delta)-4P_D(\delta)
=-\frac{67+179\delta}{125}<0
\qquad(2\le\delta\le5/2).
\tag{12}
\]

For the only range in which (10) is positive, the binary scale factor is
tiny:

\[
       q=\frac{2\delta}{5}
       \le\frac{14}{975}.
\]

Even before paying the root closure, the entire geometric sum of positive
local surplus is at most

\[
\sup_{0\le\delta\le7/195}
\frac{T(\delta)}{1-2\delta/5}
=\frac7{125}.
\tag{13}
\]

Indeed the quotient decreases from \(7/125\) at \(\delta=0\).  A closed
root must pay a positive port-return cost on the top scale (coefficient at
least \(3\), and coefficient \(5\) in the proposed normalization), which
is vastly larger than (13).

## Verdict

**The exact \(D\)-metric recursive substitution is refuted.**  Repeated
child pieces consume the entire local information advantage:

- the \(xy\) posterior state falls from \(9\) to \(11-\delta\);
- the adaptive value falls from \(33/5\) to the envelope (6);
- at the required near-critical scale \(\delta=5/2\), the local
  \(3A-4P\) surplus is strongly negative; and
- choosing \(\delta\) small enough to retain a positive local surplus makes
  the binary tree far from critical, with total surplus below \(7/125\),
  far short of the root closure.

Thus replacing the two permanent \(D\)-ports directly by child blocks cannot
produce a closed clairvoyance gap above \(4/3\).
