# Categorical first-active routes: two permanent-routing ceilings

## Verdict

No \(>4/3\) permanent-routing chamber is obtained here.  Two rigorous
obstructions cover the most direct versions of the proposal:

1. an arbitrary **closed** chamber with only one stochastic bit has
   adaptive/a-posteriori ratio at most \(4/3\), no matter how complicated
   its permanent routing structure is;
2. ordered first-active markers give no gap when the proposed mode tours
   reveal every mode before the route decision controlled by that mode.
   This includes ordinary layer-by-layer switching networks.  Pure
   translation families of permanent Cayley tours also fail to create
   mode-specific permanent routing.

Thus a categorical construction must use at least two unresolved stochastic
bits and a genuinely delayed revelation: it must make permanent service
decisions before the marker identifying the cheap mode is called.

## 1. A \(4/3\) theorem for one stochastic bit

Let \(d\) be any finite directed metric, let \(r\) be a depot, let \(U\) be
an arbitrary set of permanent clients, and let \(x\) be one stochastic
client, active with probability \(p\).  Write
\[
 L_0=\operatorname{TSP}_d(U),\qquad
 L_1=\operatorname{TSP}_d(U\cup\{x\}),
\]
where tours start and end at \(r\).  Shortcutting \(x\) in an \(L_1\)-tour
shows that
\[
 L_0\leq L_1.
\]
The a-posteriori value is
\[
        P=(1-p)L_0+pL_1.
\]

### Theorem 1

For every such instance,
\[
        \frac{\operatorname{OPT}_{\rm adapt}}{P}\leq\frac43.
\]

### Proof

There are two legal adaptive policies.

The first executes a fixed optimal \(L_1\)-tour and calls \(x\) when that
tour reaches it.  If \(x\) is inactive, the same fixed walk is still legal:
inactive clients remain usable as transit vertices.  Hence this policy costs
\[
        L_1
\]
in both realizations.

The second executes an optimal \(L_0\)-tour, returns to \(r\), and only then
calls \(x\).  If \(x\) is inactive, it stops.  If \(x\) is active, it moves
from \(r\) to \(x\) and then returns to \(r\).  Its expected cost is
\[
        L_0+p\bigl(d(r,x)+d(x,r)\bigr).
\]
Split an optimal \(L_1\)-tour at any visit to \(x\).  Its \(r\)-to-\(x\)
and \(x\)-to-\(r\) pieces, together, have length \(L_1\).  The directed
triangle inequality therefore gives
\[
        d(r,x)+d(x,r)\leq L_1.
\]
Consequently
\[
 \operatorname{OPT}_{\rm adapt}
 \leq \min\{L_1,L_0+pL_1\}.                     \tag{1}
\]

Assume first that \(L_0>0\) and put \(z=L_1/L_0\geq1\).  Dividing (1) by
\(P\) gives
\[
 \frac{\operatorname{OPT}_{\rm adapt}}P
 \leq
 \frac{\min\{z,1+pz\}}{1-p+pz}.                 \tag{2}
\]
The two numerator terms agree at \(z=1/(1-p)\).  Below that point the first
quotient in (2) is increasing in \(z\); above it the second quotient is
decreasing.  Hence its maximum is
\[
 \frac1{1-p+p^2}
 =
 \frac1{(p-\tfrac12)^2+\tfrac34}
 \leq\frac43.
\]
If \(L_0=0\), the second policy has expected cost at most \(pL_1=P\), so the
claim is immediate. \(\square\)

### Scope of Theorem 1

The common closed depot is essential to this proof.  For an open chamber
with distinct prescribed input \(s\) and output \(t\), an after-the-route
detour \(t\to x\to t\) need not be bounded by the optimal active
\(s\)-to-\(t\) service.  Theorem 1 is therefore a ceiling for closed
permanent-routing chambers, not a claimed theorem for arbitrary fixed-port
open services.

## 2. First-active modes that reveal themselves in time

Let \(m_1,\ldots,m_k\) be independent markers in a fixed order, and define
\[
        J=\min\{j:m_j\text{ is active}\},
\]
with a final value \(J=\infty\) if no marker is active.  A proposed
categorical construction supplies a posterior walk \(W_J\) for each mode.

The following condition makes the construction causally harmless.

### Definition 2 (marker-prefix compatibility)

The family \(\{W_J\}\) is marker-prefix compatible if it has a rooted
decision-tree representation with these properties.

* Every internal node is labeled by the next uncalled marker \(m_i\).
* All realizations reaching that node have incurred the same calls and the
  same movement.
* The marker is called before any movement or permanent call on which its
  outcome has an effect.
* On the inactive branch the call causes no movement.
* On the active branch the forced move to \(m_i\), and every subsequent
  prescribed move and call, are exactly the corresponding segment of
  \(W_i\).
* Later active markers are called at the locations and in the order
  prescribed by \(W_i\); they do not retroactively change an earlier
  segment.

### Theorem 3 (causal-prefix theorem)

If \(\{W_J\}\) is marker-prefix compatible, there is an adaptive policy
whose cost in every realization is exactly \(\operatorname{cost}(W_J)\).
In particular, if the \(W_J\)'s are posterior-optimal, then
\[
        \operatorname{OPT}_{\rm adapt}
        =\operatorname{OPT}_{\rm post}.
\]

### Proof

Execute the decision tree.  At an internal node, call its marker.  An
inactive call leaves the current position unchanged and selects the inactive
child.  An active call moves to the marker, exactly as the active branch
prescribes, and selects that child.  By induction on the depth, the calls,
position, and accumulated movement after every node coincide with the
corresponding prefix of \(W_J\).  At a leaf, complete its fixed suffix.
Thus the realized walk and cost are precisely \(W_J\). \(\square\)

This theorem already accounts for arbitrary transit through uncalled or
inactive clients: such transit is part of the fixed branch walk and reveals
no additional randomness.

### Consequence for switching networks

In an ordinary layered switch construction, a marker located at a switching
stage is called when that stage is reached, before choosing straight versus
crossed continuation.  The active forced movement can itself be the crossed
continuation, while an inactive call leaves the walk on the straight
continuation.  Such mode routes form the decision tree of Definition 2, so
their gap is one.

Moving the marker to a high-diameter remote vertex does not change this
conclusion if the posterior route also uses the same movement before its
first mode-dependent permanent call.  When active, the adaptive call pays
and traverses that posterior segment; when inactive, it costs zero.  Diameter
alone therefore is not delayed information.

The only way out of Theorem 3 is to put a genuinely mode-dependent
**permanent service prefix before revelation**.  Merely putting permanent
vertices on the shortest path to the marker does not serve them during the
remote call: transit is not service.  A positive proof would then have to
show that repairing those unserved prefix clients costs more than \(P/3\)
after all mixed switching paths and shortest-path shortcuts are allowed.

## 3. Translation collapse for Cayley route families

Let \(\Gamma\) be a finite group with a right-Cayley directed metric
\[
        d(g,h)=\ell(g^{-1}h),
\]
and suppose the permanent client set is all of \(\Gamma\).  Let \(C\) be a
closed walk that visits every group element.  For \(a\in\Gamma\), left
translation
\[
        aC
\]
has exactly the same length and again visits every permanent client.

### Proposition 4

If purported mode-specific permanent tours are only left translates
\(\{a_jC\}\) of one closed Cayley walk, then every translated tour can be
made into a depot tour based at \(e\), with the same cost, in every mode.
Thus the categorical label does not select a permanent routing mode.

### Proof

Left invariance gives
\[
        d(ag,ah)=d(g,h),
\]
so translation preserves the walk length.  Because \(aC\) visits all of
\(\Gamma\), it visits the depot \(e\).  Rotate its cyclic list of edges to
start and end at that occurrence of \(e\).  The result is an \(e\)-based
closed tour of the same length serving the same permanent set.  Its
feasibility is independent of marker activations. \(\square\)

Markers attached to selected group elements can still add insertion costs,
but the translated permanent cycles themselves do not supply global
mode-specific incompatibility.  To use noncommuting generators or coset
labels, the marked permanent set must break translation symmetry, and the
proof must quantify the cost of mixing translated word prefixes.  A group
endpoint discrepancy alone is not a service lower bound.

## 4. Design requirement left open

A viable categorical first-active construction must evade all three easy
policies:

1. follow a marker-prefix decision tree;
2. call ordered markers early and repair the permanent prefixes traversed
   but not served;
3. ignore the markers initially and follow a mode-independent permanent
   tour.

For a posterior scale \(P\), the second and third policies must both cost
strictly more than \(4P/3\), after directed shortest-path closure.  Standard
butterfly/Beneš cells and translation-invariant Cayley cycles fail before
this inequality: the former satisfy Theorem 3, and the latter satisfy
Proposition 4.
