# Maximum-active chain banks reveal their state causally

## Proposed marker distribution

Let
\[
             b_1<b_2<\cdots<b_m
\]
be a fixed chain of stochastic clients.  Take \(b_1\) permanent and, for
\(j\ge2\), activate \(b_j\) independently with probability \(1/j\).  If
\[
             J=\max\{j:b_j\text{ is active}\},
\]
then
\[
\Pr[J=j]
 =\frac1j\prod_{k=j+1}^m\left(1-\frac1k\right)
 =\frac1j\frac jm
 =\frac1m.
\]
Thus \(J\) is an exact uniform random key, while all realized marker clients
belong to one chain.  This looks attractive for replacing the illegal
exogenous random permutation in a layered construction.

It does not create hidden state in the stochastic-TSP model.

## Scan-to-the-maximum lemma

### Lemma

Let \(b_1,\ldots,b_m\) be clients in a directed metric such that
\[
 d(b_i,b_j)=\ell_j-\ell_i
 \qquad(i<j)
\]
for some nondecreasing coordinates \(\ell_1,\ldots,\ell_m\).  The clients
may have arbitrary independent activation probabilities.  There is a legal
causal policy which:

1. calls the markers in the order \(b_1,\ldots,b_m\);
2. incurs exactly
   \[
       \ell_J-\ell_I
   \]
   between the first and last active markers \(b_I,b_J\), with zero cost if
   at most one marker is active; and
3. knows the complete marker realization, and in particular \(J\), while
   its current position is \(b_J\).

### Proof

Call the markers in increasing chain order.  An inactive call causes no
movement.  Whenever \(b_j\) is active, the previous active marker is some
\(b_i\) with \(i<j\), and the forced movement costs
\(\ell_j-\ell_i\).  These movements telescope over all active markers to
\(\ell_J-\ell_I\).  After \(b_m\) has been queried, every marker state is
known and the last active call, hence the current position, is \(b_J\).
\(\square\)

The same statement holds in a zero-reachability poset metric, where the
entire scan has zero internal cost.

## Consequence for a route-key construction

Suppose the intended posterior service scans the active marker chain and
then chooses a continuation from the maximum active type \(b_J\).  The
policy in the lemma implements the same information pattern causally:
before calling any continuation client it already knows \(J\), and it is
located at the same marker \(b_J\) as the posterior service.  Therefore the
uniform law of \(J\) supplies no prophet advantage.

There is a second obstruction in a reachability construction.  If
\(b_i<b_j\) and \(b_j<r\) for a continuation resource \(r\), transitivity
gives \(b_i<r\).  Hence the continuation neighborhoods of the marker types
are nested in the direction of the chain.  Arbitrary route labels or a
random-suffix permutation cannot be attached to the types without either
losing transitivity or introducing positive-scale separating movements.

To make the marker state genuinely unresolved at commitment time, one must
insert an active separator before the rest of the bank is scanned, or make
later marker calls expensive after leaving the current marker.  The former
is a chronological gate and the latter is an interrupted-service toll.
Both costs enter the posterior/adaptive accounting; they cannot be omitted
as a free random-key generator.

