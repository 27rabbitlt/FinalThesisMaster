# Symbolic audit of the four-source weighted interface

## Outcome

Let \(p\) be the common activation probability, put \(q=1-p\), and let the
two specialist edges have weight \(t\in[0,1]\).  The posterior matching
reward and optimal causal query-commit reward are exactly
\[
 \boxed{
 P_{\rm rew}
   =2p+2pqt(1+p-p^2)
 }
 \tag{0.1}
\]
and
\[
 \boxed{
 C_{\rm rew}=
 \begin{cases}
  2p+pqt(1+p)(2-p),
     &t\le t_0(p),\\[0.4em]
  2p-p^3+pt\bigl(1+q^2(1+2p)\bigr),
     &t\ge t_0(p),
 \end{cases}
 }
 \tag{0.2}
\]
where
\[
                         t_0(p)=\frac{p}{1-p+p^2}.     \tag{0.3}
\]
Below \(t_0\), an optimal policy queries a generalist first.  Above \(t_0\),
it queries a specialist first.

Write the posterior and adaptive **deficiencies** from capacity two as
\[
       P_{\rm def}=2-P_{\rm rew},\qquad
       A_{\rm def}=2-C_{\rm rew}.                     \tag{0.4}
\]
The absolute surplus relevant to a \(4/3\) cost comparison is
\[
                         S:=3A_{\rm def}-4P_{\rm def}
                           =4P_{\rm rew}-3C_{\rm rew}-2.       \tag{0.5}
\]
For \(p\ge4/5\), the strict-deficiency-gap region is one interval
\[
                         t_-(p)<t<t_+(p),             \tag{0.6}
\]
with
\[
\begin{aligned}
 t_-(p)
   &=\frac{2}{p(2+5pq)},\\
 t_+(p)
   &=\frac{3p^3-2q}
      {p(3-8q+q^2+2q^3)}.
\end{aligned}                                         \tag{0.7}
\]
The Bellman switch \(t_0\) lies strictly between these endpoints.

The near-certain asymptotics explain both the attraction and the failure of
this gadget.  Put
\[
                         \delta=1-t,\qquad q=1-p.
\]
Then
\[
\begin{aligned}
 1-t_+(p)&=\frac43q^3+O(q^4),\\
 1-t_0(p)&=q^2+q^3+O(q^4),\\
 1-t_-(p)&=\frac32q-\frac{29}{4}q^2+O(q^3).
\end{aligned}                                         \tag{0.8}
\]
Thus the deficiency ratio exceeds \(4/3\) throughout the broad window
\[
                  \frac43q^3+O(q^4)
                         <\delta<
                  \frac32q+O(q^2).                    \tag{0.9}
\]
It can even diverge like \(1/q\).  Nevertheless,
\[
                \sup_{t\in[0,1]} S(p,t)
                         =(3+o(1))q^2.                \tag{0.10}
\]
The absolute surplus, not the ratio of two vanishing deficiencies, is what
must pay a reconnection toll.

This gives an interrupted-service ceiling for the natural metric lift.  A
one-piece policy implies
\[
 H(s)\le A_{\rm def}
 \quad\Longrightarrow\quad
 3H(s)-4P_{\rm def}\le S.                             \tag{0.11}
\]
The row-separation inequality forces a natural specialist/nonedge reset to
have scale at least \(t\) in the same normalization.  For \(p\ge4/5\),
\[
                            S<t.                      \tag{0.12}
\]
Consequently no toll \(s\ge t\) can satisfy the necessary amplification
condition
\[
                         3H(s)-4P_{\rm def}>s.         \tag{0.13}
\]
As \(p,t\to1\), the mismatch is decisive:
\[
                         S=\Theta(q^2),
                  \qquad s_{\rm natural}=\Theta(1).   \tag{0.14}
\]

The symbolic gadget therefore improves the deficiency ratio but does not
provide a theorem-level metric amplifier with a natural protected reset.

## 1. Posterior reward

There are two generalists \(G_1,G_2\), each of unit weight to either target,
and specialists \(S_A,S_B\), each of weight \(t\) only to its named target.
All four sources are independently active with probability \(p\).

Let \(g\) be the number of active generalists.

* If \(g=2\), the posterior reward is \(2\).
* If \(g=1\), it is \(1+t\) when at least one specialist is active and \(1\)
  otherwise.
* If \(g=0\), it is \(t\) times the number of active specialists.

Therefore
\[
\begin{aligned}
P_{\rm rew}
 &=2p^2
   +2pq\bigl[1+t(1-q^2)\bigr]
   +2ptq^2\\
 &=2p+2pqt(1+p-p^2),
\end{aligned}                                         \tag{1.1}
\]
which is (0.1).

Equivalently,
\[
 P_{\rm def}
   =2q\bigl[1-pt(1+q-q^2)\bigr].                      \tag{1.2}
\]

## 2. Symbolic Bellman recursion

The state notation is the same as in
`finite-weighted-interface-gadget.md`.  Only the symbolic values needed at
the root are recorded here.

### One target

For one generalist and the specialist competing for its target, querying
the generalist first is always optimal:
\[
                         U=p(1+qt).                   \tag{2.1}
\]
Indeed, specialist-first is \(p(q+\max\{t,p\})\), which is strictly smaller
for \(p,t\in(0,1)\).

For two generalists and that specialist competing for one target,
generalist-first is again optimal:
\[
                         W=p(1+q+q^2t).               \tag{2.2}
\]
If the specialist would be rejected, its first-query value is
\(p(1+q)\), already below (2.2).  If it would be accepted, subtraction
from (2.2) gives
\[
                         p(1-t)(1-q^2)\ge0.           \tag{2.3}
\]

### One generalist and two specialists

With both targets free, querying a specialist first gives
\[
                         V_1
     =p\bigl[1+t(1+q+pq)\bigr].                       \tag{2.4}
\]
The generalist-first value is \(p[1+t(1+q)]\).  Their difference is
\[
                         p^2qt>0.                    \tag{2.5}
\]

### Two generalists and one specialist

With both targets free, querying a generalist first gives
\[
                         V_2
          =2p+pqt(1+p).                              \tag{2.6}
\]
If the specialist is accepted when active, specialist-first is smaller by
\[
                         p^3(1-t).                   \tag{2.7}
\]
If it is rejected, its value is at most \(2p<V_2\).  Hence (2.6) is exact.

### Root comparison

Querying a generalist first, accepting it when active, and using
(2.1),(2.4) gives
\[
                         C_G
       =2p+pqt(1+p)(1+q).                             \tag{2.8}
\]
Acceptance is optimal because
\[
                 1+U-V_1=1-pt(1+pq)\ge0,             \tag{2.9}
\]
where \(p(1+pq)\le1\).

Querying a specialist first and accepting it when active gives, from
(2.2),(2.6),
\[
                         C_S
       =2p-p^3+pt\bigl[1+q^2(1+2p)\bigr].             \tag{2.10}
\]
If accepting were inferior, rejecting the specialist leaves value \(V_2\),
and (2.8) exceeds \(V_2\) by
\[
                         pq^2t(1+p)>0.                \tag{2.11}
\]
Thus that branch cannot beat \(C_G\).

The two effective root values satisfy
\[
 C_G-C_S
       =p^2\bigl[p-t(1-p+p^2)\bigr].                 \tag{2.12}
\]
This proves the switch point (0.3) and the exact Bellman value (0.2).
When \(t>t_0\), accepting in (2.10) is indeed optimal: its acceptance
threshold is
\[
                  \frac{p^2}{1-2p^2q}\le t_0(p),      \tag{2.13}
\]
because the difference after cross multiplication is
\[
                         q^2(1+p)\ge0.
\]

## 3. Exact \(4/3\) region

### Generalist-first branch

Substituting (0.1) and the first line of (0.2) into (0.5) gives
\[
 \boxed{
 S_G=q\bigl[-2+pt(2+5pq)\bigr].
 }
 \tag{3.1}
\]
Thus \(S_G>0\) exactly when
\[
                         t>t_-(p)
             :=\frac{2}{p(2+5pq)}.                   \tag{3.2}
\]
The interval \(t_-<t_0\) is nonempty exactly when
\[
                         p^3>\frac25.                \tag{3.3}
\]

### Specialist-first branch

The second line of (0.2) gives
\[
 \boxed{
 S_S=3p^3-2q
       +pt(-3+8q-q^2-2q^3).
 }
 \tag{3.4}
\]
For \(p\ge4/5\), the coefficient of \(t\) in (3.4) is negative.  Hence this
branch has \(S_S>0\) exactly for
\[
                         t<t_+(p),                   \tag{3.5}
\]
where \(t_+\) is (0.7).  At \(t=t_0\), (3.1) and (3.4) agree.  Condition
(3.3) makes their common value positive, while at \(t=1\)
\[
                         S=-P_{\rm def}<0.            \tag{3.6}
\]
Therefore
\[
                         t_-<t_0<t_+<1               \tag{3.7}
\]
for \(p\ge4/5\), proving (0.6).

For the original numerical choice \(p=4/5,t=19/20\),
\[
\begin{aligned}
 t_-&=\frac{25}{28},&
 t_0&=\frac{20}{21},&
 t_+&=\frac{355}{364},\\
 S&=\frac{16}{625}.
\end{aligned}                                         \tag{3.8}
\]
The first two deficiencies are
\[
        P_{\rm def}=\frac{148}{3125},\qquad
        A_{\rm def}=\frac{224}{3125},
\]
recovering \(A_{\rm def}/P_{\rm def}=56/37\).

## 4. Near-certain asymptotics

Put \(p=1-q\) and \(t=1-\delta\).

The exact posterior deficiency is
\[
\begin{aligned}
P_{\rm def}
 &=2q\bigl[
       \delta+2q^2-q^3-\delta(2q^2-q^3)
     \bigr]\\
 &=2q\delta+4q^3+O(q^4+\delta q^3).                  \tag{4.1}
\end{aligned}
\]

### Generalist-first region: \(\delta\ge q^2+O(q^3)\)

Here
\[
\begin{aligned}
A_{\rm def}
 &=q^2+2q^3+2q\delta
       +O(q^4+q^2\delta),\\
S_G
 &=3q^2-10q^3-2q\delta
       +O(q^4+q^2\delta).
\end{aligned}                                         \tag{4.2}
\]
If \(\delta=cq\), then
\[
 \frac{A_{\rm def}}{P_{\rm def}}
       \longrightarrow1+\frac1{2c},
 \qquad
 S_G=(3-2c)q^2+o(q^2).                               \tag{4.3}
\]
The \(4/3\) threshold is \(c=3/2\).

If \(\delta=cq^2\) with \(c\ge1\), then
\[
 \frac{A_{\rm def}}{P_{\rm def}}
       \sim\frac{1}{(2c+4)q},
 \qquad
 S_G\sim3q^2.                                        \tag{4.4}
\]

### Specialist-first region: \(\delta\le q^2+O(q^3)\)

At \(t=1\), both deficiencies equal
\[
                         D_0=4q^3-2q^4.              \tag{4.5}
\]
The exact affine forms are
\[
\begin{aligned}
A_{\rm def}
 &=D_0+
   p\delta\bigl[1+q^2(1+2p)\bigr],\\
P_{\rm def}
 &=D_0+
   2pq\delta(1+q-q^2).
\end{aligned}                                         \tag{4.6}
\]
For \(\delta=cq^3\),
\[
 \frac{A_{\rm def}}{P_{\rm def}}
       \longrightarrow1+\frac c4,
 \qquad
 S_S=(3c-4)q^3+o(q^3).                               \tag{4.7}
\]
This gives the lower endpoint \(c=4/3\).

For \(\delta=cq^2\) with \(0<c\le1\),
\[
 \frac{A_{\rm def}}{P_{\rm def}}
       \sim\frac{c}{(2c+4)q},
 \qquad
 S_S\sim3cq^2.                                       \tag{4.8}
\]
The maximum surplus is approached on either side of the Bellman switch and
is \((3+o(1))q^2\), proving (0.10).

## 5. Reset and interrupted-service ceiling

Let \(H(s)\) be any interrupted adaptive cost associated with a metric lift
of the deficiency gadget, with toll \(s\) per extra service piece.  Since a
one-piece execution is allowed,
\[
                         H(s)\le A_{\rm def}.          \tag{5.1}
\]
A necessary local condition for the usual serial or recursive amplifier is
\[
                         3H(s)-4P_{\rm def}>s.         \tag{5.2}
\]
Equations (0.5) and (5.1) imply the still more basic necessity
\[
                              S>s.                    \tag{5.3}
\]

Now realize reward \(w_{uv}\) as a saving in a directed interface metric.
A specialist row and a nonincident specialist row differ at their named
target by \(t\) in the unit normalization.  The directed triangle
inequality gives the row-separation bound
\[
                         d(u,u')\ge t.                \tag{5.4}
\]
Thus a reset which changes between those private source rows has natural
scale at least
\[
                              s_{\rm nat}\ge t.        \tag{5.5}
\]

For \(p\ge4/5\), both Bellman branches satisfy
\[
                              S<t.                    \tag{5.6}
\]
Indeed, on the generalist branch, \(S_G\) is increasing in \(t\) and
\[
 S_G
 \le q[-2+p(2+5pq)]
 \le3q^2\le\frac3{25},
\]
while positive surplus requires \(t>t_-(4/5)=25/28\).  On the
specialist branch, (3.4) is decreasing in \(t\), so its maximum is the
common switch value and obeys the same bound.

Combining (5.3), (5.5), and (5.6) proves that the natural protected-row
reset cannot amplify the gadget.  Scaling all interface weights by a factor
\(\lambda\) does not help: both \(S\) and the forced reset scale by
\(\lambda\), leaving the strict inequality unchanged.

The diverging ratio in (4.4) or (4.8) compares two quantities of orders
\(q^2\) and \(q^3\).  It does not create the order-one absolute credit
needed to cross a private metric row or pay a protected service restart.
