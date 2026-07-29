# Finite certificate for the four-star layered-poset construction

This note records one completely finite member of the four-star
layered-poset family and checks, with slack, that its asymmetric
clairvoyance gap is strictly larger than \(4/3\).  It also states precisely
what the asymptotic argument proves: a liminf of at least \(3/2\), rather
than an equality unless a separate adaptive upper bound is supplied.

The construction and the causal product lemma are proved in
`four-star-layered-poset-construction.md`.  Here we use only the two bounds
proved there.  With \(L\) stages, selector probability \(q\), and
\[
                    h=q^2(1-q)^2,
\]
they are
\[
\begin{aligned}
 \inf_\pi \mathbb E K_\pi
   &\ge 3-(1-h)^L,                                      \tag{1}\\
 \mathbb E W
   &\le 2+L(4q^3-2q^4).                                \tag{2}
\end{aligned}
\]
Here \(K_\pi\) is the number of increasing runs in the service order of a
causal policy and \(W\) is the width of the realized active subposet.

## 1. Concrete parameters

Take
\[
             q=\frac1{200},\qquad L=56000,\qquad
             \varepsilon=10^{-10}.                    \tag{3}
\]
There are \(2(L+1)=112002\) permanent gate clients and \(4L=224000\)
stochastic selector clients, hence \(336002\) clients in total.

For these values,
\[
 h=\frac{39601}{1600000000}
\]
and
\[
 Lh
 =\frac{56000\cdot39601}{1600000000}
 =1.386035.                                           \tag{4}
\]
Therefore
\[
 (1-h)^L\le e^{-Lh}<0.26.                             \tag{5}
\]
The final numerical inequality has ample slack.  For example, the first
five nonzero terms of the exponential series already give
\[
 e^{1.386035}
 >
 1+1.386035+\frac{1.386035^2}{2}
 +\frac{1.386035^3}{6}
 +\frac{1.386035^4}{24}
 >\frac{50}{13},
\]
which is equivalent to the strict part of (5).

It follows from (1) that
\[
                 \inf_\pi\mathbb E K_\pi>2.74.        \tag{6}
\]
For the posterior bound,
\[
 4q^3-2q^4
 =\frac{399}{800000000},
\]
so (2) gives
\[
 \mathbb EW
 \le 2+\frac{56000\cdot399}{800000000}
 =2.02793.                                            \tag{7}
\]

## 2. Conversion to a finite directed metric

For distinct clients \(x,y\), put
\[
 d(x,y)=
 \begin{cases}
   \varepsilon,&x<y\text{ in the layered poset},\\
   1,&x\not<y,
 \end{cases}
\qquad
 d(r,x)=d(x,r)=\frac12.                               \tag{8}
\]
This is a positive directed metric.  Two consecutive
\(\varepsilon\)-arcs imply \(x<y<z\), hence \(x<z\), and every other
two-client path has length at least the direct distance.  A path through
the depot has length one and merely ties an incomparable client arc.

If \(N\) clients are active and their service order has \(K\) maximal
increasing runs, its closed-tour cost is exactly
\[
              \varepsilon N+(1-\varepsilon)K.         \tag{9}
\]
The expected number of active clients is
\[
 \mu:=\mathbb EN
 =2(L+1)+4Lq
 =113122.                                             \tag{10}
\]
Consequently (6)--(10) imply
\[
\begin{aligned}
 \operatorname{OPT}_{\rm adapt}
  &>\varepsilon\mu+(1-\varepsilon)2.74,\\
 \operatorname{OPT}_{\rm post}
  &\le\varepsilon\mu+(1-\varepsilon)2.02793.
\end{aligned}                                         \tag{11}
\]
Multiplying the desired strict inequality by three and using (11),
\[
\begin{aligned}
 3\operatorname{OPT}_{\rm adapt}
 -4\operatorname{OPT}_{\rm post}
 &>
 (1-\varepsilon)
   \bigl(3\cdot2.74-4\cdot2.02793\bigr)
 -\varepsilon\mu\\
 &=(1-\varepsilon)0.10828-0.0000113122\\
 &>0.10826>0.                                        \tag{12}
\end{aligned}
\]
Thus this one finite product-distribution instance satisfies
\[
 \boxed{\displaystyle
 \frac{\operatorname{OPT}_{\rm adapt}}
      {\operatorname{OPT}_{\rm post}}>\frac43.}
\]

No parallel repetition, correlated activation, zero distance, or
small-instance computation is used.

## 3. Exact asymptotic conclusion

The parameter choice
\[
 q_L=L^{-2/5},\qquad \varepsilon_L=L^{-2}
\]
in the main construction gives
\[
 \operatorname{OPT}_{\rm adapt}\ge3-o(1),
 \qquad
 \operatorname{OPT}_{\rm post}=2+o(1).
\]
Hence the proved statement is
\[
 \liminf_{L\to\infty}
 \frac{\operatorname{OPT}_{\rm adapt}}
      {\operatorname{OPT}_{\rm post}}
 \ge\frac32.                                         \tag{13}
\]
Equation (13) is stronger than the required strict \(4/3\) lower bound.
Calling it a limit equal to \(3/2\) would additionally require an adaptive
upper bound of \(3+o(1)\), which is not needed for the construction theorem.
