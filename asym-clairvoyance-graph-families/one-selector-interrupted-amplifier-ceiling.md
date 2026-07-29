# One selector can never satisfy the interrupted amplifier inequality

## Theorem

Consider an arbitrary finite two-terminal directed-metric chamber with
input \(s\), output \(z\), arbitrary permanent clients, and one stochastic
client \(x\), active with probability \(p\).  Let
\[
                         C_0\leq C_1
\tag{1}
\]
be the realization-wise optimal one-piece \(s\)-to-\(z\) open-service
costs when \(x\) is inactive and active, respectively, and put
\[
                         P=(1-p)C_0+pC_1.
\tag{2}
\]
Let
\[
                         H(t)=
       \inf_\pi\mathbb E[Z_\pi+t(N_\pi-1)]
\tag{3}
\]
be the causal interrupted value, where every service-containing piece has
the same free chamber entry \(s\) and exit \(z\).

Then, for every \(t\geq0\),
\[
                         \boxed{3H(t)-4P\leq t.}
\tag{4}
\]
Consequently no one-selector chamber—regardless of the size or
mode-dependence of its permanent payload—can be amplified above \(4/3\) by
the balanced serial transfer theorem.

This strictly strengthens the closed-depot and zero-toll one-selector
ceilings for the purpose of recursive amplification.  A fixed-terminal
one-piece open gap can be large, but fragmentation always restores (4).

## 1. Two causal interrupted policies

### Follow the active-state trace

Fix an optimal active-state \(s\)-to-\(z\) trace of cost \(C_1\), and call
the clients in its service order.  If \(x\) is inactive, omit its service
movement and shortcut the preceding and following legs.  Directed triangle
inequality shows that the resulting permanent service costs at most
\(C_1\).  This is one service-containing piece in both states, so
\[
                         H(t)\leq C_1.
\tag{5}
\]

### Inactive service followed by one selector piece

First execute an optimal inactive-state service, at cost \(C_0\).  Then
start a fresh abstract piece at \(s\) and call \(x\).

If \(x\) is inactive, the call causes no movement and the fresh piece
contains no service event, so it is not counted.  If \(x\) is active, move
to \(x\) and exit at \(z\).  Its internal cost is
\[
                         E:=d(s,x)+d(x,z).
\tag{6}
\]
Split an optimal active-state trace at its service visit to \(x\).  Its
prefix is an \(s\)-to-\(x\) walk and its suffix is an \(x\)-to-\(z\) walk,
so triangle inequality gives
\[
                         E\leq C_1.
\tag{7}
\]
Only the active state has a second service-containing piece.  Therefore
\[
                         H(t)
       \leq C_0+p(E+t)
       \leq C_0+p(C_1+t).
\tag{8}
\]

Combining (5) and (8),
\[
       H(t)\leq
       \min\{C_1,\ C_0+p(C_1+t)\}.
\tag{9}
\]
This permits arbitrary permanent payloads and arbitrary active/inactive
route orders.

## 2. Exact envelope optimization

If \(C_0=0\), (8) gives
\[
 H(t)\leq p(C_1+t),
\]
while (5) gives \(H(t)\leq C_1\).  The two bounds cross at
\[
                         t_0=\frac{1-p}{p}C_1
\]
when \(0<p<1\).  Below the crossover,
\[
 3H(t)-4P-t
 \leq-pC_1+(3p-1)t.
\]
Above the crossover, it is at most
\[
                         (3-4p)C_1-t.
\]
Each affine expression is maximized either at an endpoint where it is
plainly nonpositive or at \(t_0\), where both equal
\[
                     -\frac{(2p-1)^2}{p}C_1\leq0.
\]
The cases \(p=0,1\) are immediate.

Assume \(C_0>0\), and normalize
\[
                         y:=\frac{C_1}{C_0}\geq1,
             \qquad \tau:=\frac t{C_0}.
\tag{10}
\]
It is enough to prove
\[
 3\min\{y,\ 1+py+p\tau\}
 \leq4(1-p+py)+\tau.
\tag{11}
\]

The two terms in the minimum meet at
\[
                         y_0=\frac{1+p\tau}{1-p}
\tag{12}
\]
for \(p<1\).  On the first branch, the difference between the left and
right sides of (11) is affine in \(y\); on the second branch it is affine
and decreasing in \(y\).  If the first branch is increasing, both branches
are maximized at \(y_0\).  If it is decreasing, its maximum over
\(1\leq y\leq y_0\) is at \(y=1\), where it is plainly nonpositive.
Thus the only nontrivial value is the crossover.

Multiplying the crossover difference by \(1-p\) gives exactly
\[
\begin{aligned}
 &(1-p)\left[
 3y_0-4(1-p+py_0)-\tau
 \right]\\
 &\hspace{2cm}
 =-(2p-1)^2(1+\tau)\leq0.
\end{aligned}
\tag{13}
\]
This proves (11).  The endpoint \(p=1\) is deterministic and has
\(H(t)=P=C_1\), so (4) is immediate there as well.

## 3. Consequence for the search

A positive interrupted chamber needs at least two independent stochastic
clients.  The permanent payload must also interact with their joint state:

* one selector is ruled out by (4);
* a common deterministic payload added to every piece only shifts the toll,
  by `uniform-piece-payload-invariance.md`; and
* rare selectors with cheap individual pieces are ruled out by the
  singleton-piece certificate.

The smallest surviving local object is therefore a genuinely two-bit,
fixed-terminal, mode-dependent permanent-routing chamber whose best
fragmented policy still obeys
\[
                         3H(t)-4P>t
\]
at some positive toll.
