**Discovering the stochastic online vector discrepancy algorithm**

Research note, 29 September 2026.

The route developed here starts from the problem, rather than from a proposed matrix or martingale: plan a fractional completion, implement one part by randomized rounding, and track how the plan changes. The remaining-input matrix, the sign probabilities, the normalized martingale, and the variance telescope then arise in that order.

This is a reconstruction of a plausible discovery process, not a claim about the historical origin of the supplied proof. Choosing the quadratic planning problem below is a genuine design decision. Once it is chosen, most of the subsequent formulas are consequences that can be worked out systematically.

**The problem and the information we can use.**

We know vectors \(v_i\in\mathbb R^d\), \(\|v_i\|_2\le1\), and independent activations \(A_i\sim\mathrm{Bernoulli}(p_i)\). An adversary chooses the next unprocessed label using previous observations and signs, before learning that label's activation. When the label is active we must immediately assign a sign. All labels are eventually processed. We want a small final signed sum in specified directions, in particular the coordinate directions.

Write \(x\) for the current signed sum and \(U\) for the remaining labels. Conditional on the history and the choice of the next label \(i\), its activation probability is still \(p_i\). Revealing earlier independent activations and signs generated from them does not reveal any of the other activations. This elementary fact lets a plan based on the original probabilities remain meaningful after an adaptive history.

**I1. Measure the difficulty of completing the current state.**

The same current error can be easy or difficult to correct depending on what remains. An error in the first coordinate is relatively manageable if many remaining vectors point along that coordinate; it is harder to correct if no remaining vector has a first-coordinate component.

Consequently, the state we should reason about is \((x,U)\). A potential depending only on the Euclidean size of \(x\) discards useful information. The final-only objective makes it reasonable to ask how well the remaining candidates can complete the current partial signing.

Start with the simplest case: all remaining vectors are the scalar \(1\). If the current sum is \(x\) and the expected number of remaining activations is \(P=\sum_{i\in U}p_i\), distributing the correction evenly suggests a mean sign approximately \(-x/P\). This is already a ratio of error to remaining capacity. It also exposes two problems: \(P\) can vanish, and the requested mean can exceed one in magnitude. We will allow a residual and then prove that the requested means usually remain feasible.

**I2. Design mean signs first, and round them afterward.**

For each remaining candidate introduce a proposed active-sign mean
\[
m_i=\mathbb E[\varepsilon_i\mid A_i=1,\text{current plan}].
\]
Every \(m_i\in[-1,1]\) can be implemented by an actual sign coin:
\[
\Pr(\varepsilon_i=+1\mid A_i=1)=\frac{1+m_i}{2}.
\]
Its expected vector contribution is \(p_i m_i v_i\). Thus a fixed fractional completion plan has mean residual
\[
r=x+\sum_{i\in U}p_i m_i v_i.
\tag{1}
\]
These are provisional recommendations. We execute only the recommendation for the next chosen label, then replan after its actual outcome. No future signs are committed or revealed.

This transformation gives us continuous variables with a direct implementation. The remaining tasks are to choose sensible means, quantify the noise created by rounding, and keep the means inside their allowed interval.

**I3. Spread correction effort instead of demanding exact cancellation.**

A natural first planning problem is to distribute correction using a quadratic effort cost. A candidate uses its bias only when active, so its expected squared bias effort is \(p_i m_i^2\). Allow an imperfect terminal correction, penalized by \(\|r\|_2^2/\lambda\), where \(\lambda>0\) will be chosen after the analysis.

Temporarily relax the constraints \(|m_i|\le1\) and solve
\[
\boxed{
F_U(x)=\min_{m\in\mathbb R^U}
\left\{
\frac12\sum_{i\in U}p_i m_i^2
+\frac1{2\lambda}
\left\|x+\sum_{i\in U}p_i m_i v_i\right\|_2^2
\right\}.}
\tag{2}
\]

The effort term favors distributing a correction over available candidates. The residual term allows for directions that cannot be corrected by those candidates. Increasing \(\lambda\) makes residual error less expensive and the resulting feedback gentler.

Why a quadratic? It is a simple symmetric, smooth cost that penalizes large biases and is tractable. It also appears in minimum-energy linear control. With \(u_i=\sqrt{p_i}m_i\), the actual correction channels are the columns \(\sqrt{p_i}v_i\). Adding virtual residual channels \(\sqrt\lambda I\) turns (2) into a least-norm linear completion problem. The standard geometry of minimum-energy inputs explains why a Gram matrix and its inverse should appear; see Boyd and Lall, [*Controllability and state transfer*, slides 9–11](https://ee263.stanford.edu/lectures/contr.pdf). The application to this discrepancy problem is the derivation here.

A second motivation is the cost of biasing a fair coin. Its relative entropy at mean \(m\) is
\[
\phi(m)=\frac{1+m}{2}\log(1+m)+\frac{1-m}{2}\log(1-m)
=\frac{m^2}{2}+O(m^4)
\]
near zero. Since the coin is used only upon activation, its expected cost is \(p_i\phi(m_i)\). This motivates the quadratic term locally. We do not need an entropy approximation anywhere in the subsequent proof.

Equation (2) is an algorithm-design surrogate. It is not the actual expected squared terminal discrepancy, and choosing it does not prove that the eventual stochastic algorithm is optimal. Its usefulness must be established by analyzing the legal rounded process.

**I4. Solving the plan produces the matrix and the bias together.**

Let \(r\) be the residual in (1), and define \(y=r/\lambda\). For \(p_i>0\), the first-order condition for (2) is
\[
0=p_i m_i+\frac{p_i}{\lambda}v_i^\top r,
\qquad m_i=-v_i^\top y.
\]
Substituting these recommendations into (1) gives
\[
\lambda y=x-\sum_{i\in U}p_i v_iv_i^\top y.
\]
Therefore
\[
\boxed{
Q_U=\lambda I+\sum_{i\in U}p_i v_iv_i^\top,
\qquad y=Q_U^{-1}x,
\qquad m_i=-v_i^\top y.}
\tag{3}
\]
For \(p_i=0\), the choice of \(m_i\) does not affect the plan or any realized contribution.

This explains why the matrix uses \(p_i\), rather than \(p_i^2\) or \(p_i(1-p_i)\): activation probability weights both the planned movement and the bias effort. It also explains the outer product. The candidate measures the scalar correction needed along \(v_i\), and then produces movement along that same vector.

The optimized value is
\[
F_U(x)=\frac12x^\top Q_U^{-1}x,
\qquad \nabla_x F_U(x)=Q_U^{-1}x=y.
\tag{4}
\]
Thus \(v_i^\top y\) is the marginal cost of an extra unit of error in direction \(v_i\), measured by the best available completion. The sign bias is the negative of that marginal cost.

In the scalar example, (3) becomes \(y=x/(\lambda+P)\). If remaining vectors are coordinate vectors with expected counts \(P_j\), it gives \(y_j=x_j/(\lambda+P_j)\). The matrix formula is the corresponding rule when correction directions overlap.

**I5. A completion plan remains consistent when its own recommendation is executed.**

This is the step that discovers the martingale.

Suppose the next label has parameters \(p,v\), and write \(Q'=Q-pvv^\top\) for the capacity after processing it. The current plan recommends expected movement \(-pvv^\top y\). If that movement were executed exactly, the new error would be
\[
x_{\mathrm{mean}}'=x-pvv^\top y=(Q-pvv^\top)y=Q'y.
\]
Hence the replanned marginal-cost vector would be
\[
(Q')^{-1}x_{\mathrm{mean}}'=y.
\tag{5}
\]
The recommendations for all remaining labels would be unchanged. Committing one part of an optimal fractional plan does not force a revision of the rest when it goes exactly as planned.

Now execute the actual sign coin with mean \(-v^\top y\), provided that mean is feasible. Put
\[
W=A_i\varepsilon_i,
\qquad \xi=W+p\,v^\top y.
\]
Conditional on the history before revealing \(A_i\),
\[
\mathbb E\xi=0,
\qquad |\xi|\le2,
\qquad \mathbb E\xi^2=p-p^2(v^\top y)^2\le p.
\tag{6}
\]
The variance identity follows from \(W^2=A_i\). Sign randomness is still present when \(p=1\), so activation variance \(p(1-p)\) would be the wrong noise bound.

The actual error equals the planned error plus a surprise:
\[
x'=Q'y+\xi v.
\]
Replanning therefore gives
\[
\boxed{y'-y=\xi(Q')^{-1}v.}
\tag{7}
\]
Because \(Q'\) and the chosen vector are known before the new activation and sign, this increment has conditional mean zero. The changing plan is a martingale as long as its means are feasible.

The pre-removal matrix in the sign rule is now explained: that rule executes the recommendation of the completion problem that still includes the current candidate. The post-removal matrix in (7) describes how the remaining plan absorbs its surprise.

An inactive candidate still changes the plan: it removes an opportunity that had contributed expected capacity. On an inactive outcome, \(\xi=pv^\top y\), even though \(x\) itself does not move. This is why every processed label is removed from \(Q\), whether active or inactive.

**I6. The same matrix describes consumed capacity and introduced noise.**

The noise in \(x\) has conditional covariance at most \(pvv^\top\). That is precisely the matrix removed from \(Q\). By (7), the conditional noise covariance in \(y\) is at most
\[
p(Q')^{-1}vv^\top(Q')^{-1}.
\tag{8}
\]
How could we discover a quantity that pays for the sum of these terms?

For an infinitesimal decrease in an invertible matrix,
\[
\mathrm d(Q^{-1})=-Q^{-1}(\mathrm dQ)Q^{-1}.
\]
Putting \(\mathrm dQ=-pvv^\top\) produces exactly the form in (8), up to the distinction between a finite and infinitesimal step. Thus the inverse matrix is the natural candidate for a variance account. This differential calculation is a way to discover the account; the next identity proves the required finite-step statement.

Let \(z=(Q')^{-1}v\) and \(a=v^\top(Q')^{-1}v\). Since \(Q=Q'+pvv^\top\), the rank-one inverse formula gives
\[
(Q')^{-1}-Q^{-1}=\frac{p}{1+pa}zz^\top.
\tag{9}
\]
Assume \(\lambda\ge1\). The unit-vector condition implies \(a\le1/\lambda\), so
\[
pzz^\top\preceq(1+1/\lambda)\big((Q')^{-1}-Q^{-1}\big)
\preceq2\big((Q')^{-1}-Q^{-1}\big).
\tag{10}
\]
Each round's variance is charged to the increase in inverse capacity. These increases telescope, and \(Q^{-1}\preceq I/\lambda\) at every state. Thus the total variance budget is at most \(2/\lambda\) in any fixed unit direction, independent of the number and order of the labels.

In one dimension with \(p=v=1\), the same idea is particularly concrete. If capacity falls from \(q\) to \(q-1\), the next variance contribution is at most \(1/(q-1)^2\), while
\[
\frac1{q-1}-\frac1q=\frac1{q(q-1)}.
\]
Their ratio is \(q/(q-1)\le1+1/\lambda\) when \(q-1\ge\lambda\). Early rounds are numerous but have small normalized noise; late rounds have larger normalized noise but only a small remaining supply of candidates.

Adaptive order is handled here without fixing an order in advance. The calculation applies to whichever label was chosen, and the sum telescopes on every realized path.

**I7. Prove the fractional recommendations stay feasible by stopping at the first failure.**

We deliberately postponed \(|m_i|\le1\). We can now resolve it using (7)–(10).

Let \(\mathcal H\) contain all candidate vectors and all output directions we want to control; assume every member has Euclidean norm at most one. For coordinate discrepancy use
\[
\mathcal H=\{v_1,\ldots,v_n,e_1,\ldots,e_d\},
\qquad L=|\mathcal H|\le n+d.
\]
The input directions certify legal sign probabilities; the output directions certify the final guarantee. They serve distinct purposes.

Define the first exit time
\[
\tau=\inf\{t:\max_{h\in\mathcal H}|h^\top y_t|\ge1\}.
\]
Until that time, use the derived sign rule. After an exit, the algorithm may continue with arbitrary legal signs, for example always \(+1\). For analysis freeze the process at the exit: \(\bar y_t=y_{t\wedge\tau}\), \(\bar Q_t=Q_{t\wedge\tau}\).

The crossing step is included. Its sign probabilities were chosen from a safe pre-step state. Only later increments are frozen. In particular,
\[
\Delta\bar y_t=\mathbf1_{\{\tau\ge t\}}\xi_tQ_t^{-1}v_{i_t}
\]
has conditional mean zero, because the indicator is known from the past. This constructs a martingale on the original probability space. Conditioning on the event of eventual success would not give this argument.

For each fixed \(h\in\mathcal H\), put \(M_{h,t}=h^\top\bar y_t\). The increment and variance bounds are
\[
|\Delta M_{h,t}|\le2/\lambda,
\]
\[
\begin{aligned}
\sum_t\mathbb E[(\Delta M_{h,t})^2\mid\mathcal F_{t-1}]
&\le(1+1/\lambda)h^\top(\bar Q_n^{-1}-Q_0^{-1})h\\
&\le\frac{1+1/\lambda}{\lambda}\le\frac2\lambda.
\end{aligned}
\tag{11}
\]
The stopping indicator selects exactly an initial segment, so the matrix increments still telescope. Both bounds hold on failure paths as well as success paths.

**I8. Concentration chooses the regularization scale.**

We have a scalar martingale in each test direction, with total conditional variance \(O(1/\lambda)\) and jump size \(O(1/\lambda)\). A unit deviation should therefore have probability \(\exp(-\Theta(\lambda))\). This predicts \(\lambda\) proportional to \(\log(L/\delta)\), before tracking constants.

The two-sided maximal Freedman inequality says that a zero-start martingale with jump bound \(R\) and total conditional variance at most \(\nu\) satisfies
\[
\Pr\left(\max_t|M_t|\ge b\right)
\le2\exp\left(-\frac{b^2}{2(\nu+Rb/3)}\right).
\tag{12}
\]
The scalar maximal form is stated as Theorem 1.1 in Tropp, [*Freedman's inequality for matrix martingales*](https://arxiv.org/pdf/1101.3039). Only that scalar result is needed here.

Using \(R=\nu=2/\lambda\), \(b=1\), and a union bound over \(\mathcal H\),
\[
\Pr(\tau\le n)\le2L\exp(-3\lambda/16).
\]
Choose
\[
\boxed{\lambda=\frac{16}{3}\log\frac{2L}{\delta}.}
\tag{13}
\]
For \(0<\delta<1\) this is larger than one, as required above. With probability at least \(1-\delta\), all the recommended active-sign means remain feasible and every tested projection of \(y\) stays below one in absolute value. The maximal inequality already handles all times; there is no extra union bound over rounds.

At completion there are no remaining candidates, so \(Q_n=\lambda I\). Therefore
\[
x_n=\lambda y_n,
\qquad \max_{h\in\mathcal H}|h^\top x_n|<\lambda
\]
on the success event. For coordinates we may use
\[
\lambda=\frac{16}{3}\log\frac{2(n+d)}{\delta}.
\]
The same \(\lambda\) has four connected roles: residual tolerance in the fractional plan, invertibility, a bound on accumulated normalized noise, and the final discrepancy allowance.

**The recovered algorithm.**

Start with \(x=0\) and all labels unprocessed. Choose \(\lambda\) as above. Maintain \(Q=\lambda I+\sum_{i\in U}p_i v_iv_i^\top\) and \(y=Q^{-1}x\). When label \(i\) is chosen, use the current value \(r=v_i^\top y\). If active and no safety exit has occurred, give it sign \(+1\) with probability \((1-r)/2\) and sign \(-1\) otherwise. Remove the label regardless of activation, update \(x,Q,y\), and record any exit from the tested region. Continue with legal fallback signs after an exit. The argument above proves that the fallback is needed with probability at most \(\delta\).

This derives the algorithm from a completion problem and then derives its proof from the stability of that completion plan under randomized implementation.

**An independent interpretation: imitate a Gaussian bridge.**

There is another way to arrive at the same formulas, useful for someone familiar with Gaussian conditioning. At a state \((x,U)\), replace future signed activations by independent auxiliary variables
\[
G_i\sim N(0,p_i),\qquad Z\sim N(0,\lambda I).
\]
Imagine conditioning their total to cancel the current error:
\[
x+\sum_{i\in U}G_iv_i+Z=0.
\tag{14}
\]
The covariance before conditioning is \(Q=\lambda I+\sum_i p_i v_iv_i^\top\). The Gaussian conditional-mean formula gives
\[
\mathbb E[G_i\mid(14)]=-p_i v_i^\top Q^{-1}x.
\tag{15}
\]
This is exactly the unconditional mean contribution coefficient requested by our sign rule. Standard Gaussian conditioning formulas appear in Shalizi, [*Multivariate Distributions*, Section 14.2.2](https://stat.cmu.edu/~cshalizi/uADA/12/lectures/ch14.pdf); equations (14)–(16) here are our specialization.

The variance identity becomes even more revealing. With \(Q'=Q-pvv^\top\), \(z=(Q')^{-1}v\), and \(a=v^\top(Q')^{-1}v\),
\[
\operatorname{Var}(G_i\mid(14))
=p-p^2v^\top Q^{-1}v
=\frac{p}{1+pa}.
\]
Thus the normalized Gaussian innovation has covariance exactly
\[
\frac{p}{1+pa}zz^\top=(Q')^{-1}-Q^{-1}.
\tag{16}
\]
The actual activation/sign coin preserves this Gaussian mean drift while having variance at most \(p\), which costs at most the factor \(1+pa\le1+1/\lambda\). The finite-step correction in (10) can therefore be understood as the extra noise from implementing a Gaussian-inspired correction with legal discrete signs.

Equation (14) is an auxiliary design model. Conditioning the real Bernoulli activation process on a small final sum would generally change the activation probabilities, so it would not be a legal implementation. The actual algorithm uses the original probabilities and fresh sign coins throughout.

**Why some tempting explanations lead to a different algorithm.**

It would be incorrect to say that our sign rule greedily minimizes expected next-step quadratic error. For a fixed positive-definite \(B\), conditional active-sign mean \(m\), and current candidate \((p,v)\),
\[
\mathbb E\left[\frac12(x+A\varepsilon v)^\top B(x+A\varepsilon v)\right]
=\frac12x^\top Bx+pm\,v^\top Bx+\frac p2v^\top Bv.
\]
This is linear in \(m\), so its minimizer is generally an endpoint \(m=\pm1\). The planning problem (2) instead uses a quadratic effort cost and the squared mean residual.

Its one-variable elimination identity is
\[
F_U(x)=\min_m\left\{\frac p2m^2+F_{U\setminus\{i\}}(x+pmv)\right\}.
\]
Writing \(B=(Q')^{-1}\), its optimizer is
\[
m=-\frac{v^\top Bx}{1+p\,v^\top Bv}=-v^\top Q^{-1}x.
\]
This is a deterministic recursion for the fractional planning problem, not a Bellman equation for the actual stochastic signing problem.

Nor does the argument give the same bound on every prefix \(x_t\). At intermediate times \(x_t=Q_ty_t\), and \(Q_t\) can be large. The conversion to a small discrepancy uses the terminal identity \(Q_n=\lambda I\). Likewise, the logarithmic upper bound is a guarantee of this method, not an optimality or lower-bound claim.

The order may depend on past signs, but the next activation must remain hidden when its label is chosen. Otherwise \(\mathbb E[A_i\mid\text{history and selected label}]=p_i\) can fail, breaking the match between drift, noise, and consumed capacity. The argument does not require the realized order to be independent of the signs, and it does not union-bound over adversary strategies.

**Where this fits among known ideas.**

Alweiss, Liu, and Sawhney's [*Discrepancy Minimization via a Self-Balancing Walk*, Section 1.2](https://arxiv.org/pdf/2006.14009) motivates sign probabilities with a linear bias opposing the current discrepancy. Their stated online guarantee is for an oblivious input sequence. That supports the use of randomized corrective drift as a design idea, but does not by itself establish the causal adaptive-order result considered here. The remaining-candidate completion plan explains the additional matrix dependence in this problem.

The independent derivation in this note adds three explanatory steps to the supplied proof: a fractional optimization problem that produces its formulas, a consistency principle that produces its martingale, and a capacity/noise correspondence that predicts its variance telescope. Those steps provide a route for rebuilding the method from the problem statement.
