# Shared-lap obstruction for high-branch serial banks

## Exact local data

For the two-selector chamber
\[
D=\begin{pmatrix}
0&5&4&7\\
3&0&5&7\\
4&1&0&3\\
1&6&5&0
\end{pmatrix},
\qquad
p_x=\frac{13}{25},\quad p_y=\frac15,
\]
with permanent ports \(a,b\), the expected free-boundary a-posteriori value
and contiguous adaptive value are
\[
       P=\frac{617}{125},\qquad A=\frac{33}{5}.
\]
Thus
\[
       3A-4P=\frac7{125}>0.
\]

Let \(H(t)\) be the least adaptive value when a chamber service may use
arbitrarily many free-boundary pieces, with a toll \(t\) for every piece
after the first.  There are two universal upper bounds:
\[
       H(t)\leq
       \min\left\{\frac{33}{5},\,\frac{21}{5}+t\right\}.
\tag{1}
\]
The first simply uses one piece.  For the second, use two pieces:

1. enter at \(a\), serve \(a\), and call \(x\); if active, traverse
   \(a\to x\to b\), and if inactive, exit at \(a\);
2. enter at \(b\), serve \(b\), and call \(y\); if active, traverse
   \(b\to y\to a\), and if inactive, exit at \(b\).

Their expected internal cost is
\[
       \frac{13}{25}\,5+\frac15\,8=\frac{21}{5}.
\]

## Balanced-bank no-go theorem

Consider a recursive serial bank with:

- \(b\) children at scale \(L_{k-1}\);
- parent reset length \(L_k\);
- independent copies of the chamber whose total scale per parent is
  \(\epsilon L_k\);
- a fixed forward order, so one parent lap can give every child and every
  chamber one additional service piece.

Write
\[
       q:=\frac{bL_{k-1}}{L_k}.
\]
If \(t\) units per chamber scale are reserved to control chamber
fragmentation, a single parent lap has to pay both the \(b\) child
reconnections and all chamber tolls.  Therefore a valid reset budget
requires
\[
       1-q\geq\epsilon t.
\tag{2}
\]

The geometric sum of the strict local \(3A-4P\) surplus, normalized by the
root scale, is at most
\[
       \frac{\epsilon\bigl(3H(t)-4P\bigr)}{1-q}.
\tag{3}
\]
A closed root needs a return of at least the root reset length; because that
cost is common to the two benchmarks, it contributes \(-1\) to normalized
\(3\operatorname {OPT}_{\rm adapt}
-4\operatorname {OPT}_{\rm post}\).
Consequently a certificate above \(4/3\) from this accounting would require
\[
       \frac{\epsilon\bigl(3H(t)-4P\bigr)}{1-q}>1.
\]
By (2), the necessary local condition is
\[
       3H(t)-4P>t.
\tag{4}
\]

Condition (4) never holds.  If
\[
       t\leq\frac{893}{375},
\]
then the two-piece policy in (1) gives
\[
       H(t)\leq\frac{21}{5}+t\leq\frac43P,
\]
so the left side of (4) is nonpositive.  If \(t>893/375\), the one-piece
policy in (1) gives
\[
       3H(t)-4P
       \leq3\cdot\frac{33}{5}
             -4\cdot\frac{617}{125}
       =\frac7{125}<t.
\]

Therefore no choice of branching \(b\), scale ratio, depth, or number of
independent chamber copies rescues this fixed-order serial-bank composition.
The obstruction is not merely a loose port-diameter estimate: it is witnessed
by an explicit causal two-piece service.

## Consequence for other fixed-order banks

The same argument applies to a tree, lift, or switching bank whenever one
fixed global lap gives every local chamber another free-boundary piece.  Such
a lap must be charged once against the aggregate toll, not once per chamber.
To evade the theorem, the cheap global traversal itself must depend on the
independent activation pattern in a way that prevents all required repair
pieces from being placed on one later traversal.  In other words, a valid
next candidate needs a route-code or order incompatibility; high branching
alone is insufficient.

