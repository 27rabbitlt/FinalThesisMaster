# Why the near-certain \(K_{2,2}\) query-commit gap does not directly lift

## Target being audited

Let the four edges of \(K_{2,2}\) be independently present with probability
\(p\).  The omniscient and optimal query-commit expected matching sizes are
\[
\begin{aligned}
 \mu(p)&=4p-4p^2+4p^3-2p^4,\\
 M(p)&=4p-4p^2+3p^3-p^4,
\end{aligned}
\]
so
\[
 \mu(p)-M(p)=p^3(1-p).
\]
As \(p\uparrow1\), \(2-\mu(p)\) is quadratic but \(2-M(p)\) is
linear.  This looks attractive only if matching deficiency can be made into
TSP cost.  Two metric facts obstruct that conversion.

## 1. All-active-atom ceiling

Consider an arbitrary directed metric instance with permanent clients and
\(m\) stochastic clients.  Write \(C_{\rm all}\) for the posterior optimum
when every stochastic client is active, and let
\[
 q_{\rm all}=\Pr[\text{all stochastic clients are active}].
\]

### Lemma 1 (all-active atom)

For every such instance,
\[
 \frac{\operatorname{OPT}_{\rm adapt}}
      {\operatorname{OPT}_{\rm post}}
 \le \frac1{q_{\rm all}}.
\]

### Proof

Fix an optimal all-active tour and call the stochastic clients in that
fixed order.  On any realization, inactive calls cause no service event.
The active-call tour is a shortcut of the fixed all-active tour, so the
directed triangle inequality gives realization cost at most
\(C_{\rm all}\).  Hence
\[
 \operatorname{OPT}_{\rm adapt}\le C_{\rm all}.
\]
Posterior costs are nonnegative, and on the all-active event the posterior
cost is exactly \(C_{\rm all}\).  Therefore
\[
 \operatorname{OPT}_{\rm post}\ge q_{\rm all}C_{\rm all}.
\]
Divide the two inequalities. \(\square\)

For the four iid \(K_{2,2}\) edge-clients,
\[
 \frac{\operatorname{OPT}_{\rm adapt}}
      {\operatorname{OPT}_{\rm post}}
 \le p^{-4}.
\]
In particular this upper bound is already at most \(4/3\) when
\[
 p\ge(3/4)^{1/4}\approx0.930605,
\]
and it tends to one as \(p\uparrow1\), uniformly over all choices of metric
lengths, including metric families whose lengths depend on \(p\).

Thus the linear-versus-quadratic deficiency asymptotic cannot itself yield
a \(>4/3\) TSP gap.  Any attempted causal cost
\(C_{\rm all}+\Theta(1-p)\) is defeated by the fixed all-active tour, whose
cost is at most \(C_{\rm all}\) on every deletion realization.

## 2. Transit erases endpoint service capacity

Suppose an active edge-client \(z_e\), for \(e=(u,v)\), is intended to have
a cheap complete extraction
\[
                         u\longrightarrow z_e\longrightarrow v.
\]
Metric movement through a location is not a client service event.  Hence,
for every next served client \(y\),
\[
 d(z_e,y)\le d(z_e,v)+d(v,y).
\]
Even if \(v\) was already called, a route that calls \(u,z_e\) may use \(v\)
as an uncharged transit point and leave the extraction corridor at exactly
the same geometric cost.  Symmetrically, for every previous served client
\(x\),
\[
 d(x,z_e)\le d(x,u)+d(u,z_e),
\]
so an already-served \(u\) remains available as a transit entrance.

Consequently service-once does not make \(u\) or \(v\) a consumable
geometric resource.  An incoming half-extraction on one edge-client and an
outgoing half-extraction on another can still use the missing endpoints as
transit vertices.  Changing reset lengths cannot invalidate these two
triangle inequalities.

This is the local reason the literal edge-client DAG reduces to trail
pairing and has realization-wise gap one
(`dag-edge-trail-gap-one.md`): incoming and outgoing savings split at the
edge-client instead of enforcing that the same edge consume both matching
endpoints.

## Consequence

A valid switching reduction would need one of the following features, none
of which is present in the independent-client fixed-metric model:

- deletion of an endpoint or arc after its service event;
- two selector copies with perfectly correlated activation;
- a genuinely higher-order cost depending on the predecessor and successor
  simultaneously, rather than the sum of two metric transitions.

Therefore a default-route/extraction construction must prove a mechanism
other than endpoint service capacity.  Near-certain \(K_{2,2}\)
query-commit asymptotics and nonuniform reset lengths alone do not provide
one.
