# Why a voltage lift does not supply hidden routing memory

## Proposed mechanism

A tempting way to protect many copies of an open clairvoyance gadget is to
give copy \(i\) a voltage \(e_i\in(\mathbb Z/2\mathbb Z)^m\).  A causal
mistake in copy \(i\) is supposed to change the current sheet by \(e_i\);
closing the tour would then require a correction word whose length is the
number of distinct mistakes.  This would prevent one extra global lap from
repairing many copies.

That mechanism is not implementable by fixed client locations in stochastic
TSP.

## Endpoint-sheet lemma

Let \(\widetilde H\) be a regular voltage lift of a directed graph \(H\)
with group \(\Gamma\).  Its vertices are pairs \((v,g)\).  Every client is
a fixed vertex \((v,g)\) of the lifted graph.

After an active client \(x=(v,g)\) is called, the policy is at the fixed
vertex \(x\), hence at the fixed sheet \(g\).  If the next active client is
\(y=(w,h)\), every path used to move from \(x\) to \(y\) starts in sheet
\(g\) and ends in sheet \(h\).  Its total voltage is constrained by these
two endpoint sheets.  After the move, all information about which path was
used is absent from the state of the stochastic-TSP policy: its state is
again just the fixed current vertex \(y\), together with observed activation
bits.

Consequently, sheet displacements do not form an additional syndrome that
can accumulate independently of the served endpoints.  Along a closed tour,
the endpoint displacements telescope back to the depot sheet.  A shortest
path metric makes this loss of path memory explicit: the cost of the next
move depends only on the two fixed endpoints.

## Why fibre copies do not repair the problem

One could instead put a copy of selector \(v\) in every sheet and allow the
policy to use the copy in its current sheet.  But these are distinct clients.
Under the required product activation law, their activation variables are
independent.  They do not represent one hidden selector bit replicated
across the fibre.  Requiring the copies to share one activation bit would
leave the independent-client model.

Thus a voltage lift can still be used as an ordinary large directed metric,
but it cannot by itself make one causal error leave a persistent,
client-independent sheet defect.

## Serial-lap balance for the tournament open gadget

There is also a quantitative obstruction to replacing voltage protection by
a serial lap toll.  In the sparse tournament open gadget, in the regime
used to approach its local \(3/2\) ratio,

\[
P=\frac{\lambda^2}{2}+o(\lambda^2),
\qquad
e=\frac{\lambda^2}{4}+o(\lambda^2),
\]

where \(P\) is the posterior open cost and \(e\) is the causal excess.

Suppose consecutive copies are separated by a toll \(\delta\).  The first
global pass pays \(\delta\) per copy, while repairing all deferred errors on
another pass costs at most another \(\delta\) per copy.  Hence the most that
this accounting can force per copy is

\[
P+\delta+\min\{e,\delta\}.
\]

Ignoring lower-order terms, its ratio to the posterior charge \(P+\delta\)
is maximized at

\[
\delta=e=P/2,
\]

and equals

\[
1+\frac{e}{P+e}
=1+\frac{P/2}{3P/2}
=\frac43.
\]

For \(\delta<e\), the repair lap is cheaper; for \(\delta>e\), paying the
local causal errors is cheaper and the extra first-pass toll only enlarges
the denominator.  Therefore serial toll protection cannot turn this
particular \(3/2\)-open gadget into a closed gap strictly above \(4/3\).

## Consequence

A successful protected-interface construction must store a causal mistake
in an ordinary fixed vertex or cut state that cannot be erased when the
next client is served.  Homological path history in a lift is not such a
state.  Alternatively, one needs a local open gadget with ratio strictly
greater than \(3/2\), so that the serial-toll balance itself leaves room
above \(4/3\).
