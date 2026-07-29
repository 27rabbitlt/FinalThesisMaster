# An exact weighted interface gadget with deficiency ratio \(359/214\)

## 1. The gadget

There are two unit-capacity targets, \(A\) and \(B\), and four independently
active sources.  Every source is active with probability
\[
                         p=\frac56,\qquad q=1-p=\frac16 .
\]
The two **generalists** \(G_1,G_2\) have weight \(1\) to either target.  The
two **specialists** \(S_A,S_B\) have only their indicated edge, of weight
\[
                              t=\frac{29}{30}.
\]
Thus the weight matrix, with columns \(A,B\), is
\[
 \begin{array}{c|cc}
       &A&B\\ \hline
 G_1   &1&1\\
 G_2   &1&1\\
 S_A   &29/30&0\\
 S_B   &0&29/30
 \end{array}.
\tag{1}
\]

A causal policy chooses the next unqueried source.  If it is active, the
policy may assign it irrevocably to a free target or reject it.  Write \(P\)
for the expected posterior maximum-weight matching and \(C\) for the
optimal expected causal reward.

The relevant baseline is the total target capacity
\[
                                  B=2.
\tag{2}
\]

We prove
\[
       P=\frac{7669}{3888},\qquad
       C=\frac{15193}{7776},
\tag{3}
\]
and hence
\[
\boxed{\quad
   \frac{B-C}{B-P}
       =\frac{359/7776}{214/7776}
       =\frac{359}{214}
       >\frac43 .
 \quad}
\tag{4}
\]

No numerical enumeration is used below.

## 2. Exact posterior value

Let \(g\in\{0,1,2\}\) be the number of active generalists.

* If \(g=2\), the posterior reward is \(2\).
* If \(g=1\), its reward is \(1\) plus \(t\) exactly when at least one
  specialist is active.
* If \(g=0\), its reward is \(t\) times the number of active specialists.

Consequently
\[
\begin{aligned}
P
 &=2p^2
   +2pq\bigl[1+t(1-q^2)\bigr]
   +q^2(2pt)\\
 &=\frac{25}{18}
   +\frac5{18}\left(1+\frac{29}{30}\frac{35}{36}\right)
   +\frac1{36}\frac{29}{18}\\
 &=\frac{7669}{3888}.
\end{aligned}
\tag{5}
\]

## 3. Rational Bellman certificate

For a set \(R\) of unqueried sources and a set \(F\) of free targets, let
\(V(R;F)\) be the optimal expected future reward.  If source \(i\in R\) is
queried next, its Bellman value is
\[
 qV(R-i;F)
 +p\max\left\{
       V(R-i;F),\
       \max_{\substack{j\in F\\w_{ij}>0}}
       \bigl(w_{ij}+V(R-i;F-j)\bigr)
       \right\}.
\tag{6}
\]
Taking the maximum over \(i\in R\) defines \(V(R;F)\).  We now evaluate only
the symmetric substates needed at the root.  Equation (6) also checks every
competing first query, so the computation is an upper as well as a lower
certificate.

### Elementary substates

If \(S\) denotes a specialist whose target is free and \(G\) a generalist,
then
\[
\begin{array}{c|c}
\text{state}&\text{value}\\ \hline
S\text{ with its target free}&pt=29/36\\
G,G\text{ with two free targets}&2p=5/3\\
G,G\text{ with one free target}&1-q^2=35/36\\
G,S\text{ with two free targets}&p+pt=59/36.
\end{array}
\tag{7}
\]

For \(G,S\) competing for the specialist's sole free target, querying \(G\)
first gives
\[
 q(pt)+p=\frac{209}{216}.
\tag{8}
\]
Querying \(S\) first gives only
\[
 qp+pt=\frac{17}{18}<\frac{209}{216}.
\tag{9}
\]
Therefore
\[
 V(G,S;\{\text{target of }S\})=\frac{209}{216}.
\tag{10}
\]

### One generalist and both specialists

With both targets free, query a specialist first.  On an inactive answer
the remaining \(G,S\) can use distinct targets and have value \(59/36\).
On an active answer, take its edge and use (10) on the other target.  This
gives
\[
 q\frac{59}{36}
 +p\left(t+\frac{209}{216}\right)
 =\frac{2443}{1296}.
\tag{11}
\]
If instead \(G\) is queried first, its value is
\[
 q(2pt)+p(1+pt)
 =\frac{383}{216}
 <\frac{2443}{1296}.
\tag{12}
\]
Thus
\[
 V(G,S_A,S_B;\{A,B\})=\frac{2443}{1296}.
\tag{13}
\]
If just one target is free, the specialist for the occupied target is
irrelevant, and (10) gives
\[
 V(G,S_A,S_B;\{\text{one target}\})=\frac{209}{216}.
\tag{14}
\]

### Two generalists and one specialist

With both targets free, query a generalist and, on an active answer, put it
on the target opposite the specialist.  Equations (7) and (10) give
\[
 q\frac{59}{36}
 +p\left(1+\frac{209}{216}\right)
 =\frac{2479}{1296}.
\tag{15}
\]
Querying the specialist first gives
\[
 q\frac53+p\left(t+\frac{35}{36}\right)
 =\frac{409}{216}
 <\frac{2479}{1296}.
\tag{16}
\]
Therefore
\[
 V(G_1,G_2,S;\{A,B\})=\frac{2479}{1296}.
\tag{17}
\]

If only the specialist's target is free, querying a generalist first gives
\[
 q\frac{209}{216}+p
 =\frac{1289}{1296}.
\tag{18}
\]
Querying the specialist first is worth at most \(35/36=1260/1296\): when
the specialist is active it is already better to reject its \(29/30\) edge
and retain the two generalists' success probability \(35/36\).  Hence
\[
 V(G_1,G_2,S;\{\text{target of }S\})=\frac{1289}{1296}.
\tag{19}
\]

### Root

If a generalist is queried first, an inactive answer leaves (13), while an
active answer takes a unit edge and leaves (14).  Its value is
\[
 q\frac{2443}{1296}
 +p\left(1+\frac{209}{216}\right)
 =\frac{15193}{7776}.
\tag{20}
\]

If a specialist is queried first, an inactive answer leaves (17), while an
active answer takes its edge and leaves (19).  Its value is
\[
 q\frac{2479}{1296}
 +p\left(\frac{29}{30}+\frac{1289}{1296}\right)
 =\frac{15188}{7776}
 <\frac{15193}{7776}.
\tag{21}
\]
By symmetry these are all possible first queries.  Equations
(7)--(21), substituted backward into (6), therefore prove
\[
                              C=\frac{15193}{7776}.
\tag{22}
\]

Combining (5) and (22) proves the strict \(359/214\) deficiency ratio in
(4).

## 4. What a transfer must preserve

This gadget supplies a strict \(>4/3\) separation only for **deficiency**
from the capacity baseline:
\[
                 B-P=\frac{214}{7776},\qquad
                 B-C=\frac{359}{7776}.
\tag{23}
\]
Thus a graph-metric lift must make each repeated interface contribute
\(B-\text{matching reward}\), up to an additive term sublinear in the number
of interfaces.  A lift retaining a linear common service baseline does not
inherit (4).  This is the exact accounting condition that must be audited
before claiming a stochastic-TSP construction.
