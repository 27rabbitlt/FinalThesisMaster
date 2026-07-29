# Uniform per-piece payload does not improve the chamber amplifier condition

## Statement

Let a chamber have expected posterior open value \(P\) and interrupted
causal value
\[
                  H(s)=\inf_\pi
          \mathbb E[Z_\pi+s(N_\pi-1)].
\tag{1}
\]
The serial amplifier can beat \(4/3\) only if
\[
                         3H(s)-4P>s
\tag{2}
\]
for some \(s>0\).

Suppose a proposed protection attaches a deterministic payload traversal of
cost \(e\geq0\) to **every** service-containing chamber piece, without
otherwise changing the chamber decision problem.  Then
\[
             P^{(e)}=P+e,\qquad
             H^{(e)}(t)=e+H(t+e).
\tag{3}
\]
Consequently
\[
       3H^{(e)}(t)-4P^{(e)}>t
 \quad\Longleftrightarrow\quad
       3H(s)-4P>s,
       \qquad s=t+e.
\tag{4}
\]

Thus a uniform permanent block carried by every piece creates no new
amplifiable chamber.  It only moves the point at which the old interrupted
curve is evaluated.

## Proof

For a policy using \(N\geq1\) pieces, the protected internal cost is
\[
                         Z^{(e)}=Z+eN.
\]
Therefore
\[
\begin{aligned}
 Z^{(e)}+t(N-1)
 &=Z+eN+t(N-1)\\
 &=e+Z+(t+e)(N-1).
\end{aligned}
\]
Taking the infimum over causal interrupted policies proves the second
identity in (3).  A posterior one-piece service pays the payload once,
which proves the first identity.

Now put \(s=t+e\).  Direct substitution gives
\[
\begin{aligned}
 3H^{(e)}(t)-4P^{(e)}-t
 &=3(e+H(s))-4(P+e)-(s-e)\\
 &=3H(s)-4P-s.
\end{aligned}
\]
This proves (4).

## Scope

The lemma covers:

* a fixed long source-to-sink rail traversed by every piece;
* a permanent block whose route and cost are independent of the hidden
  selector state;
* a uniform entry/exit toll internalized into the chamber; and
* duplicating the same mandatory payload before every re-entry.

It does not cover a mode-dependent payload whose service order changes with
the selector realization.  Such interaction is exactly what a positive
chamber now needs.  Merely making isolated selector pieces expensive by
adding common deterministic work cannot turn the sparse tournament chamber
or the two-selector \(D\) chamber into an amplifier.
