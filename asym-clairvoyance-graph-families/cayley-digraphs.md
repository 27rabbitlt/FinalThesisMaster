# Cayley digraphs

## Executive verdict

**Proof status:** one completely checked Cayley-metric gadget has gap \(8/7\);
no construction below beats \(4/3\).  Cayley symmetry is not itself an
obstruction, but neither noncommutativity nor a large family of group words
supplies the missing adaptive lower bound.  The promising formulation is a
many-selector **route code** in a directed word metric.  Its missing lemma is
causal prefix incompatibility together with an early-probe lower bound.

Throughout, for a finite group \(\Gamma\) and a weighted generating set \(S\),
the right Cayley digraph has arcs
\[
    g\longrightarrow gs\qquad(g\in\Gamma,\ s\in S)
\]
of weight \(w_s>0\).  Its shortest-path metric is
\[
    d(g,h)=\ell_S(g^{-1}h).
\]
It is strongly connected when \(S\) generates \(\Gamma\) as a semigroup.
The client set and the probabilities need not be translation invariant.

## 1. A fully explicit word-order gadget

This small example is useful because it proves that the homogeneity of a
Cayley metric does **not** force the clairvoyance gap to be one.

Start in the free group \(F=\langle b,h,s\rangle\) and use the unit-cost
generator set
\[
 \mathcal S=
 \{h,\ h^{-1}b,\ b^{-1},\ b,\ b^{-1}s,\ s^{-1}h,\ h^{-1}\}.
\]
Only the four group elements \(e,b,h,s\) will be clients/depot locations.
The following distances are immediate from the displayed words:
\[
\begin{array}{c|cccc}
 &b&h&s&e\\ \hline
e&1&1&2&0\\
b&0&2&1&1\\
h&1&0&2&1\\
s&\ge1&1&0&2 .
\end{array}
\]
Here the four entries equal to \(2\) are exact: none of
\(s,s^{-1},b^{-1}h,h^{-1}s\) belongs to \(\mathcal S\), while the indicated
two-letter paths exist.  All unspecified distances between distinct marked
elements are at least one.

To make the graph finite, use residual finiteness of the free group.  There is
a finite quotient \(\Gamma\) in which the four marked elements remain
distinct and none of the finitely many forbidden equalities between one of
the four length-two differences and a generator occurs.  Thus the displayed
distances are preserved.  Moreover \(\mathcal S\) generates the quotient as a
semigroup: it contains \(b^{\pm1},h^{\pm1}\), and
\[
 s=b(b^{-1}s),\qquad s^{-1}=(s^{-1}h)h^{-1}.
\]
The resulting finite Cayley digraph is therefore strongly connected.  This
argument is an existence construction; a sufficiently large finite quotient
injective on the relevant radius-two ball is enough.

Take depot \(e\), permanent clients \(b,h\), and one stochastic client \(s\)
with probability \(1/2\).

### Realization-wise a posteriori routing

If \(s\) is inactive, the tour
\[
       e\to h\to b\to e
\]
has cost \(3\).  It is optimal because a closed tour through two distinct
clients has three positive-length legs.

If \(s\) is active, the tour
\[
       e\to b\to s\to h\to e
\]
has cost \(4\).  It is optimal because a closed tour through three distinct
clients has four positive-length legs.  Consequently
\[
      \operatorname{OPT}_{\rm post}=\tfrac12(3+4)=\tfrac72 .
\]

### Universal adaptive lower bound

The two realizations agree until \(s\) is called.  It suffices to locate that
call among the two permanent calls.

* If \(s\) is called first, the inactive execution costs at least \(3\).  In
  the active execution the first leg \(e\to s\) costs \(2\), after which
  visiting two distinct permanent clients and returning costs at least \(3\).
  Thus the paired cost is at least \(3+5=8\).
* If \(b\) is called before \(s\), but \(h\) is not, then the inactive
  execution contains \(e\to b\), a \(b\)-to-\(h\) leg of cost \(2\), and a
  return of cost \(1\), so it costs at least \(4\).  The active execution has
  four distinct positive legs and costs at least \(4\).
* If \(h\) is called before \(s\), but \(b\) is not, the inactive execution
  costs at least \(3\).  The active execution contains the common
  \(e\to h\) leg, the length-two leg \(h\to s\), and at least two further
  positive legs, hence costs at least \(5\).
* If both permanent clients precede \(s\), their order is either \(h,b\) or
  \(b,h\).  In the first order the inactive cost is at least \(3\), while the
  active execution pays at least
  \(d(e,h)+d(h,b)+d(b,s)+d(s,e)=5\).  In the second order the common prefix
  alone costs at least \(d(e,b)+d(b,h)=3\), so the pair is even more
  expensive.

Every deterministic causal policy therefore has paired cost at least \(8\).
Conditioning on a private random seed gives the same bound for randomized
policies.  The order \(h,b,s\) costs \(3\) when \(s\) is inactive and \(5\)
when it is active, so
\[
   \operatorname{OPT}_{\rm adapt}=4,\qquad
   \frac{\operatorname{OPT}_{\rm adapt}}
        {\operatorname{OPT}_{\rm post}}=\frac87.
\]

### Failure audit

This proof already includes calling \(s\) first and every possible
interleaving.  Shortest-path closure is handled by the radius-two word audit
before taking the finite quotient.  Passing through an uncalled marked vertex
does not serve it; this is exactly why the policy that calls \(s\) first pays
five in the active realization.

The example is not competitive with Chapter 4.  Additively replicating it
keeps the ratio at most \(8/7\), by the additive-replication diagnostic.

## 2. Noncommuting words and coset switches

A natural scalable template is the Heisenberg group
\[
 H_q=\{(x,y,z):x,y,z\in\mathbb F_q\},\qquad
 (x,y,z)(x',y',z')=(x+x',y+y',z+z'+xy').
\]
With \(a=(1,0,0)\) and \(b=(0,1,0)\),
\[
       ab=(1,1,1)\ne(1,1,0)=ba .
\]
Thus the order of \(a\)- and \(b\)-moves records an inversion count in the
central coordinate.  More generally, for a subgroup \(H\le\Gamma\), right
multiplication by generators permutes the cosets of \(H\); a word can encode a
sequence of quotient/coset switches.

The intended route-code template is:

1. choose permanent marked group elements \(P\), independent selector clients
   \(q_1,\ldots,q_k\), and cheap closed words \(W_x\), one for
   \(x\in\{0,1\}^k\);
2. require \(W_x\) to visit
   \(P\cup\{q_i:x_i=1\}\);
3. arrange that words for separated \(x\)'s have incompatible early
   \(a/b\)-orders or incompatible quotient transversals.

This is legal: the only random variables are the independent activations of
the \(q_i\)'s.  The central coordinate, a coset label, or a chosen group word
is a deterministic function indexing a route, not an additional random
metric label.

### What can currently be proved

For every realization \(x\), the word \(W_x\), followed by shortcutting
unneeded marked vertices, is a valid a posteriori upper bound.  A word metric
also provides a clean shortcut certificate: a vertex potential
\(\phi:\Gamma\to\mathbb R\) with
\(\phi(gs)-\phi(g)\le w_s\) lower-bounds every directed path.

That is not yet an adaptive lower bound.  The fact that \(ab\ne ba\) proves
only that two words end at different group elements.  A policy may:

* call all selectors before touching the permanent backbone;
* move to an active selector along a shortest group word, without calling
  marked vertices traversed by that word;
* interleave prefixes of several candidate words; and
* use a different translated copy of the same local word after each active
  call.

If the selector set has directed diameter \(D\), the selector-first policy has
expected probing movement at most
\[
        D\sum_i p_i
\]
before its final transition to the known-realization route.  Therefore
noncommutativity helps only if early probing is expensive on the same scale as
the posterior saving.

Translation invariance does not by itself neutralize all hidden information:
the marked client set is not invariant under translation.  It does, however,
make local repairs plentiful.  After moving from \(g\) to \(gq_i\), the
policy sees exactly the same outgoing word geometry translated to its new
location.  Any lower bound must charge this repair rather than merely assert
that a previously intended word is no longer available.

## 3. Can Cayley structure improve \(4/3\)?

No improvement is presently proved.  A Cayley realization of many independent
triangle-like word switches would be only additive replication and hence
cannot amplify the local ratio.  A genuine improvement requires one group
word choice to constrain many later selector bits.  The Heisenberg inversion
coordinate is a plausible coupling device, but the following two statements
are both still missing:

* a uniformly cheap closed service word for every independent activation
  vector; and
* a lower bound for every causal call order that survives selector-first
  probing and arbitrary word-prefix interleaving.

### Verdict

**Conditional/no-go verdict.**  Cayley digraphs are expressive enough to
support a strict gap (the checked value \(8/7\)), so vertex transitivity is not
a no-go theorem.  Purely local noncommuting-generator gadgets, translated
copies, and independent coset switches do not improve \(4/3\) without a
global route-code lemma.  No such lemma is established here.

### Next lemma

The next useful target is a **causal inversion lemma**.  Construct finite
quotients \(\Gamma_k\), marked clients, and closed words \(W_x\) of cost
\(P_k\) such that, after any causal history leaving \(m\) independent
selectors unknown, every possible next active move commits a positive
fraction of the remaining Heisenberg inversion/coset constraints incorrectly;
repairing them costs at least \(\delta P_k\).  The same statement must hold
when all remaining selectors are called immediately.  A constant
\(\delta>1/3\) after normalization would be evidence for a ratio above
\(4/3\); without the selector-first clause it would not be a valid lemma for
this model.
