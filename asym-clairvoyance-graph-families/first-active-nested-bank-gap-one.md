# First-active nested banks admit an exact causal sweep

## Candidate

Let \(L=\{\ell_1,\ldots,\ell_n\}\) and
\(R=\{r_1,\ldots,r_n\}\) be permanent antichains.  There are \(n\)
stochastic banks
\[
                 B_i=(x_{i,1}<x_{i,2}<\cdots <x_{i,n}).
\]
The marker activations are independent.  Their probabilities may be chosen
so that every bank is nonempty and its first active index
\[
                 J_i=\min\{k:x_{i,k}\text{ is active}\}
\]
has any desired distribution.  The tempting uniform choice is
\[
                 p_k=\frac1{n-k+1},
\]
for which \(J_i\) is uniform on \([n]\).

Add the nested predecessor relations
\[
                 \ell_j<x_{i,k}\quad\Longleftrightarrow\quad j\le k
\]
and put every bank marker below every vertex of \(R\).  Transitivity also
puts every \(\ell_j\) below every \(r_s\).  Different banks are
incomparable.

For a realization, the offline matching between \(L\) and the banks has
edge
\[
                 \ell_j B_i\quad\Longleftrightarrow\quad j\le J_i.
\]
If its size is \(\nu\), the minimum chain-cover number is
\[
                         K^*=2n-\nu.                 \tag{1}
\]
Indeed, start with the \(3n\) components consisting of \(L\), the \(n\)
nonempty bank chains, and \(R\).  A maximum of \(\nu\) left-to-bank
transitions and exactly \(n\) transitions into distinct right vertices can
be used.  Conversely, every chain contains at most one left vertex, one
bank, and one right vertex, so no further reduction is possible.

At first sight, a causal policy seems forced to choose \(\ell_j\) before
learning \(J_i\).  That intuition is false because it can sweep a common
marker level across all unresolved banks while remaining at \(\ell_j\) on
every inactive answer.

## Exact causal policy

Process levels \(k=1,\ldots,n\).  A bank is *unresolved* if all of its
markers below the current level have been called inactive.

At level \(k\), repeatedly do the following.

1. If an unused left vertex \(\ell_j\) with \(j\le k\) is available, call
   one such vertex.  The largest available \(j\le k\) is the usual greedy
   choice.
2. While staying at that left vertex, call \(x_{i,k}\) for unresolved banks
   \(i\), one by one.
3. An inactive answer causes no movement.  An active answer necessarily
   has \(J_i=k\), because all lower markers of that bank were already
   called inactive.  The transition
   \(\ell_j\to x_{i,k}\) is cheap.  Immediately call all higher markers of
   \(B_i\) in increasing order and then append an unused right vertex.
   This serves the whole bank in one increasing chain.
4. Return to Step 1 with another unused eligible left vertex and continue
   the level-\(k\) scan.

If no eligible left vertex remains, expose and serve the remaining
level-\(k\) banks without a left predecessor, each still as one increasing
bank chain followed by a right vertex.  Then advance to level \(k+1\).

Calls to higher markers after a level-\(k\) success do not affect the
classification of any other bank.  Intervening service may move the
salesperson, but before resuming the scan the policy calls the next unused
eligible left vertex.  Thus every successful scan is physically causal.

## Why the matching is posterior-optimal

For prefix neighborhoods \(\{1,\ldots,J_i\}\), the standard greedy
algorithm that processes banks in nondecreasing \(J_i\) and assigns the
largest available \(j\le J_i\) produces a maximum matching.  The level sweep
does exactly this without knowing the types in advance:

- before level \(k\), it has discovered and processed precisely the banks
  with \(J_i<k\);
- at level \(k\), inactive probes identify no new bank and cause no
  movement;
- an active probe identifies a bank with \(J_i=k\) at the moment at which
  an eligible left predecessor is already in place.

The exchange proof for threshold matching is immediate.  If a matching
assigns a type-\(k\) bank a resource \(j\) while the greedy choice \(j'>j\)
is available, replacing \(j\) by \(j'\le k\), or swapping \(j'\) out of a
later bank whose threshold is at least \(k\), preserves feasibility.
Iterating gives the greedy matching and therefore maximum size \(\nu\).

Consequently the causal policy makes exactly \(\nu\) left-to-bank
transitions and \(n\) bank-to-right transitions.  Its realization-wise run
count is
\[
                         K_{\rm causal}=2n-\nu=K^*.   \tag{2}
\]

For the positive directed metric
\[
 d_\varepsilon(x,y)=
 \begin{cases}
   \varepsilon,&x<y,\\
   1,&\text{otherwise},
 \end{cases}
 \qquad
 d(r,x)=d(x,r)=1,
\]
the closed cost is
\[
                 1+\varepsilon N+(1-\varepsilon)K.
\]
Equation (2) therefore gives equality of the causal and posterior costs
for every realization.

## Consequence

Encoding a hidden predecessor type as the first active marker of a chain
does not create KVV arrival hardness when the eligibility sets are nested.
The policy can sweep the marker levels globally and reveal types in the
same monotone order used by the offline maximum matching.  Any viable bank
construction must use genuinely nonnested eligibility while also surviving
the transitivity constraints imposed by the internal bank order.
