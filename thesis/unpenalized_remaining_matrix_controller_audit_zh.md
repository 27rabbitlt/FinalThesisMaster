# 删除纠偏惩罚后的 remaining-matrix 控制器：可证上界与反例

日期：2026-09-12。

本笔记分析的是上一轮讨论中的**条件均值平方最小化**算法：删除
\(pz^2\) 惩罚，保留 \(z\in[-1,1]\)，并通过随机选择
\(K\in\{1,3\}\) 实现激活后的期望副本数 \(2+z\)。这与最小化
\(\mathbb E\|X_{t+1}\|^2\) 是不同算法。

以下定理与反例是本次推导所得；它们没有断言特殊 TSP 策略的最坏近似比已经被完全刻画。

## 1. 精确定义与结果范围

给定已知向量 \(u_1,\ldots,u_q\in\mathbb R^d\)，独立激活概率
\(p_i\in[0,1]\)，且 \(\|u_i\|_2\le\rho\)。所有标签均须处理，顺序可依赖过去的激活及随机副本数。

定义
\[
Q_t=\gamma I+\sum_{i\text{ 未处理}}p_i u_i u_i^\top,
\qquad
X_t=\sum_{i\text{ 已处理}}(A_iK_i-2p_i)u_i,
\qquad \gamma>0.
\]
选择下一标签 \(i\) 后，记 \(p=p_i,u=u_i\)，以及
\[
B=Q_{t+1}^{-1}=(Q_t-puu^\top)^{-1}.
\]
算法选择
\[
z_t\in\arg\min_{z\in[-1,1]}\|X_t+pzu\|_B^2.
\]
对 \(p>0,u\ne0\)，最优解为
\[
z_t=\operatorname{clip}_{[-1,1]}
\left(-\frac{u^\top BX_t}{p\,u^\top Bu}\right),
\]
若 \(p=0\) 或 \(u=0\)，取 \(z_t=0\)。
激活后，使用独立的新随机数选择
\[
\Pr(K_i=3\mid A_i=1,\mathcal F_t)=\frac{1+z_t}{2},
\qquad
\Pr(K_i=1\mid A_i=1,\mathcal F_t)=\frac{1-z_t}{2}.
\]

这里的主上界针对始终执行上述 clipped 最优解的版本，不额外触发原文的永久 fallback。第 6 节说明保留该 fallback 时如何选择参数。

令
\[
C_0=\sum_i p_i u_i u_i^\top,
\qquad
S=\operatorname{tr}C_0=\sum_i p_i\|u_i\|_2^2,
\qquad
P=\sum_i p_i.
\]
本次证明的主要结论是：

1. 对任意 \(\gamma>0\)、任意因果自适应顺序，
   \[
   \mathbb E\|X_q\|_2^2
   \le9\gamma\left[\det\left(I+\frac{C_0}{\gamma}\right)-1\right]
   \le9\gamma(e^{S/\gamma}-1).
   \]
2. 若重新选择 \(\gamma\ge S\)，则
   \[
   \mathbb E\|X_q\|_2^2\le9(e-1)S.
   \]
   对任意确定的有限测试族 \(\mathcal V\)，令
   \(R_{\mathcal V}=\max_{v\in\mathcal V}\|v\|_2\)，便有
   \[
   \mathbb E\max_{v\in\mathcal V}|v^\top X_q|
   \le3\sqrt{e-1}\,R_{\mathcal V}\sqrt S
   \le3\sqrt{e-1}\,\rho R_{\mathcal V}\sqrt P.
   \]
3. 对每个 \(\gamma>0\)，都存在固定顺序、所有 \(p_i=1/2\)、单位输入向量和一个固定单位测试向量，使最终投影有 \(\Omega(\sqrt P)\) 的期望和高概率下界。故一般向量问题中的零中心 \(O(\log q)\) 保证不成立。
4. 在论文的 dyadic 编码中，重新选择上述较大 \(\gamma\) 后，可以证明区间误差
   \(O(\sqrt P\log n)\)，并得到路线成本
   \(O(\log^2 n+\sqrt n\log n)\operatorname{OPT}_{\rm post}\)。本笔记没有证明这个 TSP 上界紧，也没有否定该特殊结构存在更强的分析。

## 2. 为什么原来的鞅恒等式不再成立

写
\[
b_t=u^\top Q_t^{-1}X_t,
\qquad
g_t=u^\top Q_t^{-1}u.
\]
Sherman--Morrison 公式给出
\[
z_t=-\operatorname{clip}_{[-1,1]}\left(\frac{b_t}{p g_t}\right).
\]
令 \(Z_t=Q_t^{-1}X_t\)，
\(\varepsilon_t=A_iK_i-p(2+z_t)\)，则
\[
\mathbb E[\varepsilon_t\mid\mathcal F_t]=0,
\qquad
Z_{t+1}=Z_t+B u\,[\varepsilon_t+p(b_t+z_t)].
\]
原文选择 \(z_t=-b_t\) 才能消去这里的可预测漂移。
无惩罚版本中，clipping 的非线性还可能与偏斜的激活噪声共同产生同向偏差。第 4 节给出实际反例，而非仅以鞅证明失效作为否定理由。

## 3. 任意因果顺序下的正向保证

令 \(\Phi_t=X_t^\top Q_t^{-1}X_t\)，并记
\(a=u^\top Bu\ge0\)。因选择下一标签发生在揭示其激活之前，\(B,a\) 在该次激活之前均已知。

由于 \(z=0\) 是可行解，所选最优解满足
\[
\|X_t+pz_tu\|_B^2\le X_t^\top BX_t.
\]
另一方面，\(\mathbb E[A_iK_i\mid\mathcal F_t]=p(2+z_t)\)，且
\[
\operatorname{Var}(A_iK_i\mid\mathcal F_t)
\le\mathbb E[(A_iK_i)^2\mid\mathcal F_t]\le9p.
\]
由二次型的均值与方差分解，
\[
\mathbb E[\Phi_{t+1}\mid\mathcal F_t]
\le X_t^\top BX_t+9pa.
\]

再由 \(Q_t=B^{-1}+puu^\top\) 及 Cauchy--Schwarz，
\[
\begin{aligned}
\Phi_t
&=X_t^\top BX_t-\frac{p(u^\top BX_t)^2}{1+pa}\\
&\ge X_t^\top BX_t-\frac{pa\,X_t^\top BX_t}{1+pa}
=\frac{X_t^\top BX_t}{1+pa}.
\end{aligned}
\]
所以
\[
\boxed{
\mathbb E[\Phi_{t+1}+9\mid\mathcal F_t]
\le(1+pa)(\Phi_t+9).
}
\]

定义
\[
D_t=\frac{\det Q_0}{\det Q_t}.
\]
矩阵行列式引理给出
\[
D_{t+1}=D_t(1+pa).
\]
因此
\[
\frac{\Phi_t+9}{D_t}
\]
是非负超鞅。随机选择下一标签时，可先条件化于选中的标签，再用塔式法则；没有使用顺序独立于过去的假设。

初始值为 \(9\)。所有输入最终处理完，故
\[
Q_q=\gamma I,
\qquad
D_q=\det\left(I+\frac{C_0}{\gamma}\right),
\]
且 \(D_q\) 是确定数。于是
\[
\mathbb E\|X_q\|_2^2
=\gamma\mathbb E\Phi_q
\le9\gamma(D_q-1).
\]
对 \(C_0\) 的非负特征值使用 \(\log(1+x)\le x\)，得
\(D_q\le e^{S/\gamma}\)。若 \(\gamma\ge S>0\)，使用
\(e^x-1\le(e-1)x\) 对 \(0\le x\le1\)，便得到
\[
\mathbb E\|X_q\|_2^2\le9(e-1)S.
\]
若 \(S=0\)，则所有实际可能贡献的向量均为零，结论直接成立。

测试族上界由下式一次性得到，无需对测试做 union bound：
\[
\max_{v\in\mathcal V}|v^\top X_q|
\le R_{\mathcal V}\|X_q\|_2.
\]
相应的直接高概率上界为：对任意 \(\delta\in(0,1)\)，以至少 \(1-\delta\) 的概率，
\[
\max_{v\in\mathcal V}|v^\top X_q|
\le3R_{\mathcal V}\sqrt{(e-1)S/\delta}.
\]
此处使用 Markov，不声称这是最优的尾界。

## 4. 固定顺序下的精确反例

取 \(q=2d\)，按如下成对顺序处理输入：
\[
u_{2j-1}=u_{2j}=e_j,
\qquad p_{2j-1}=p_{2j}=\frac12,
\qquad j=1,\ldots,d.
\]
这里 \(\rho=1\)、\(P=d\)。任意 \(\gamma>0\) 下，剩余矩阵始终对角，因此处理坐标 \(j\) 的规则简化为
\[
z=\operatorname{clip}_{[-1,1]}(-2X_j).
\]

该坐标第一步从零开始，选 \(z=0\)。第一步后的误差 \(W_j\) 满足
\[
\Pr(W_j=-1,0,2)=(1/2,1/4,1/4),
\qquad \mathbb EW_j=0.
\]
第二步的控制分别为 \(1,0,-1\)，所以
\[
\mathbb E z_{2j}
=\frac12-\frac14=\frac14,
\qquad
\mathbb E X_j^{\rm final}
=\mathbb EW_j+\frac12\mathbb E z_{2j}=\frac18.
\]
精确的最终分布是

| \(X_j^{\rm final}\) | \(-2\) | \(-1\) | \(0\) | \(1\) | \(2\) |
|---|---|---|---|---|---|
| 概率 | \(1/4\) | \(1/8\) | \(1/16\) | \(3/8\) | \(3/16\) |

各坐标使用不同激活和随机数，因此独立。固定单位测试向量
\[
v=\frac1{\sqrt d}(1,\ldots,1)
\]
满足
\[
\mathbb E[v^\top X_q]=\frac{\sqrt d}{8}
=\frac{\sqrt P}{8},
\qquad
\mathbb E|v^\top X_q|\ge\frac{\sqrt P}{8}.
\]
因为每个坐标在 \([-2,2]\) 内，Hoeffding 不等式进一步给出
\[
\Pr\left(v^\top X_q\ge\frac{\sqrt d}{16}\right)
\ge1-e^{-d/2048}.
\]

原文一般控制器对 \(\rho=\|v\|=1\)、一个测试、固定失败概率，给出 \(O(\log q)\) 界；上述 \(\Omega(\sqrt q)\) 结果将其否定。这个反例对所有 \(\gamma\) 成立。

它也说明：允许适当选择 \(\gamma\) 时，第 3 节的通用 \(\sqrt P\) 期望量级已经无法整体改进到更小阶。

## 5. 对 dyadic 区间和 TSP 可以推出什么

固定公开的 backbone、chords 和辅助整数流。对 \(n\ge2\) 记
\(H=1+\lceil\log_2m\rceil=O(\log n)\)。原文的编码满足
\[
\|u_i\|_2^2\le2H,
\qquad
\|v_I\|_2^2\le4H.
\]
在这份公开结构上重新选 \(\gamma\ge S\)，则
\[
\begin{aligned}
\mathbb E\max_I|v_I^\top X_q|
&\le2\sqrt H\sqrt{9(e-1)S}\\
&\le6\sqrt{2(e-1)}\,H\sqrt P.
\end{aligned}
\]
这里 \(P\) 是该公开结构中所有外部顶点的激活概率之和，满足 \(P\le n\)。

联合 rounding 的确定性区间误差仍为 \(64H^2\)，故
\[
\mathbb E\Lambda(F)
\le64H^2+6\sqrt{2(e-1)}\,H\sqrt P.
\]
Stack walk 的合法性和路径成本界仅使用 \(1\le K_i\le3\)，不要求原控制器的鞅性质。因此与原文 circulation 成本和 rounding 成本恒等式结合，得到
\[
\boxed{
\mathbb E\operatorname{cost}(\pi)
\le O(\log^2 n+\sqrt n\log n)
\operatorname{OPT}_{\rm post}.
}
\]
对 \(n\le1\) 直接使用最优简单策略。

**这个结论改变了 \(\gamma\) 的选取。** 若保持原文
\(\gamma=\Theta(\rho^2\log n)\)（TSP 应用中的选取），第 3 节仍给出有效的行列式界，但一般不能把它化成上述 \(O(S)\) 均方界。
对固定公开结构，此时仍有
\[
\mathbb E\max_I|v_I^\top X_q|
\le \min\left\{5P,\;6\sqrt{H\gamma(D_q-1)}\right\}.
\]
第一项来自
\(\mathbb E|A_iK_i-2p_i|\le5p_i-4p_i^2\le5p_i\)，且 \(|\chi_I(u_i)|\le1\)。
由此至少得到粗的 \(O(n)\operatorname{OPT}_{\rm post}\) 成本上界；这不是紧性结论。

第 4 节反例中的一般测试向量没有被嵌入单个 cyclic interval，因此它不构成特殊 TSP 策略的 \(\Omega(\sqrt n)\) 近似比下界。本次分析未解决：保持原 \(\gamma\) 时，特殊 dyadic/TSP 结构是否仍能支持 \(O(\log^2 n)\) 的更强证明。

## 6. 若保留原文的全局安全检测和永久 fallback

第 4 节反例在 \(\gamma\ge4\) 时始终满足
\(\max_i|u_i^\top Q_t^{-1}X_t|\le2/\gamma\le1/2\)，因此不会触发 fallback；通用反例仍然成立。

第 3 节的主上界使用每一步都执行所定义的最优解。若另外规定失去安全性后永久取 \(K=1\)，不能直接把这段 fallback 也当作最优解来使用同一递推。

一个保证两种执行一致的参数选择是
\[
\gamma\ge6q\rho^2.
\]
因为对任意实现及任意时刻，\(\|X_t\|\le3q\rho\)，故
\[
|u_j^\top Q_t^{-1}X_t|
\le\frac{3q\rho^2}{\gamma}\le\frac12.
\]
安全检测永远通过，且 \(\gamma\ge S\)，所以此前的 \(O(\sqrt P)\) 通用期望界与 \(O(\sqrt n\log n+\log^2 n)\) 路由界同样适用。

## 7. 本次验证

- 用精确有理数枚举了二步反例：均值 \(1/8\)，二阶矩 \(9/4\)。
- 对 3000 组不同输入方向、概率、误差大小及正则项的数值实例核对了单步能量递推，未发现违反。
- 上界的证明与反例的枚举分别经过独立复核；数值检查仅用于辅助验证，不替代上述解析证明。
