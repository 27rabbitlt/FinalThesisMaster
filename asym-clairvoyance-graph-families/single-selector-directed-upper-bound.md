# A single stochastic selector cannot exceed \(4/3\), even asymmetrically

## Theorem

Let \(r\) be the depot, let \(D\neq\varnothing\) be an arbitrary set of
permanent clients, and let \(s\) be the only stochastic client, active with
probability \(p\in[0,1]\).  In every directed metric,
\[
 \frac{\operatorname {OPT}_{\rm adapt}}
      {\operatorname {OPT}_{\rm post}}
 \leq
 \frac1{1-p+p^2}
 \leq\frac43.
\tag{1}
\]
Thus a single selector controlling two global orders of arbitrarily many
permanent blocks cannot give a strict gap above \(4/3\).

## Proof

Let
\[
\begin{aligned}
 C_0&:=\text{cost of an optimal depot tour through }D,\\
 C_1&:=\text{cost of an optimal depot tour through }D\cup\{s\}.
\end{aligned}
\tag{2}
\]
Shortcutting \(s\) from the second tour gives
\[
       C_1\geq C_0.
\tag{3}
\]
The a-posteriori value is exactly
\[
       P:=(1-p)C_0+pC_1.
\tag{4}
\]

We use two legal causal policies.

### Policy 1: follow the all-active tour

Call the clients in the order of an optimal tour for \(D\cup\{s\}\).
If \(s\) is active, the cost is \(C_1\).  If \(s\) is inactive, its call
causes no movement.  The next permanent client is reached directly from the
preceding permanent client; the directed triangle inequality says that this
shortcut is no more expensive than the two tour legs through \(s\).
Consequently this policy costs at most \(C_1\) in either realization:
\[
       A\leq C_1.
\tag{5}
\]

### Policy 2: query the selector first

Call \(s\) while still at the depot, then follow an optimal permanent tour.
If \(s\) is inactive, the cost is \(C_0\).

Suppose \(s\) is active.  Let \(v\) be the first permanent client on the
chosen \(C_0\)-tour.  Calling \(s\) first and then \(v\) costs
\[
       d(r,s)+d(s,v)
       \leq d(r,s)+d(s,r)+d(r,v).
\tag{6}
\]
Every depot tour through \(s\) contains a prefix from \(r\) to \(s\) and a
suffix from \(s\) back to \(r\).  By the directed triangle inequality,
\[
       d(r,s)+d(s,r)\leq C_1.
\tag{7}
\]
After reaching \(v\), follow the remainder of the \(C_0\)-tour.  Equations
(6)--(7) show that the active-realization cost is at most \(C_0+C_1\).
Therefore
\[
\begin{aligned}
       A
       &\leq(1-p)C_0+p(C_0+C_1)\\
       &=C_0+pC_1.
\end{aligned}
\tag{8}
\]

Combining (5) and (8),
\[
       A\leq\min\{C_1,\ C_0+pC_1\}.
\tag{9}
\]
Put \(x=C_1/C_0\geq1\).  From (4) and (9),
\[
 \frac AP
 \leq
 \frac{\min\{x,\,1+px\}}
      {1-p+px}.
\tag{10}
\]
For fixed \(p\), the branch with numerator \(x\) is increasing in \(x\),
while the branch with numerator \(1+px\) is decreasing.  They meet at
\[
       x=\frac1{1-p}.
\tag{11}
\]
Hence
\[
\begin{aligned}
 \frac AP
 &\leq
 \frac{1/(1-p)}
 {(1-p)+p/(1-p)}\\
 &=\frac1{(1-p)^2+p}
 =\frac1{1-p+p^2}.
\end{aligned}
\tag{12}
\]
The denominator \(1-p+p^2\) is minimized at \(p=1/2\), where it equals
\(3/4\).  This proves (1). \(\square\)

## Audit against the proposed global-order construction

- The proof makes no assumption about the number or internal structure of
  the permanent blocks.
- The two posterior tours may have completely opposite orders.
- Directed asymmetry is fully allowed; only the directed triangle
  inequality is used.
- Selector-first behavior is explicitly included in Policy 2.
- Mixed and interleaved policies cannot invalidate an upper bound obtained
  from two concrete legal policies.
- No symmetry relation such as \(d(u,v)=d(v,u)\) is used.

Therefore adding more permanent blocks, a switching network, or a
topological potential around one activation bit cannot yield a strict
single-selector construction above \(4/3\).

