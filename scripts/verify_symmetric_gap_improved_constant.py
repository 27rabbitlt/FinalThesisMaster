"""Exact certificate for the strengthened symmetric stochastic-TSP gap.

This verifies the rational substitutions used with Proposition 7.1 of
``symmetric_gap_one_billionth.pdf``.  Every assertion is performed with
``fractions.Fraction``; floating-point conversions are display-only.
"""

from fractions import Fraction as Q


def exp_bounds(x: Q, degree: int) -> tuple[Q, Q]:
    """Return rational lower and upper bounds for exp(x), for x >= 0."""
    assert x >= 0
    term = Q(1)
    partial = Q(1)
    for k in range(1, degree + 1):
        term *= x / k
        partial += term

    next_term = term * x / (degree + 1)
    tail_ratio = x / (degree + 2)
    assert tail_ratio < 1
    return partial, partial + next_term / (1 - tail_ratio)


# Parameters in Proposition 7.1.
a = Q(129, 125)
b = Q(35, 34)
t = Q(19, 70)
gamma = Q(1, 24_640)
delta_0 = Q(1_023, 1_250_000_000_000)
eta = 2 * delta_0

assert a > 1 and b > 1
assert 0 < t < 1 and 0 < gamma < 1

K = b * (2 * a + 1)
B_0 = 2 * b * (K - a)
c = 10 * (a / (a - 1) + b / (b - 1))
s = t * (1 - gamma)

# Uniform upper bound on z_0 for 0 < delta <= delta_0.
Z = c * delta_0 / ((1 - gamma) * (1 - delta_0))
assert c * delta_0 < 1 - delta_0
assert 0 < Z < 1

# Rational upper bounds on (c delta_0)^t and
# (2a)^(1-t) (2 delta_0)^t.  Raising to the 70th power
# removes all fractional exponents.
U = Q(20_000_264, 1_000_000_000)
V = Q(69_906_229, 10_000_000_000)
assert U**70 > (c * delta_0) ** 19
assert V**70 > (2 * a) ** 51 * (2 * delta_0) ** 19

# Bounds for f_s(Z) = (1 - Z^s)/(s log(1/Z)).
L = Q(720_631, 50_000)
A = Q(3_603_157, 250_000)
B = Q(39_118_403, 10_000_000)
W = Q(400_073, 20_000_000)

_, exp_L_upper = exp_bounds(L, 100)
exp_A_lower, _ = exp_bounds(A, 100)
_, exp_B_upper = exp_bounds(B, 60)

assert exp_L_upper < 1 / Z  # log(1/Z) > L
assert exp_A_lower > 1 / Z  # log(1/Z) < A
assert exp_B_upper < 1 / W  # log(1/W) > B
assert s * A < B            # hence W < Z**s

f_upper = (1 - W) / (s * L)

# For 0 < delta <= delta_0, use r_0 <= 1, h_0 <= c delta_0,
# z_0 <= Z, and monotonicity of f_s.  The two terms linear in
# delta are maximized at delta_0.
upper = (
    2
    + 2 * a * t
    + (2 * a + B_0) * U
    + 2 * V
    + B_0 * t / (1 - gamma) * f_upper
    + B_0 * t * delta_0 / gamma
    + 24 * (a + 1) * delta_0
)
target = 3 - eta
assert upper < target

print("eta =", eta, float(eta))
print("factor =", target, float(target))
print("certified upper =", upper, float(upper))
print("slack =", target - upper, float(target - upper))
print("K =", K, "B_0 =", B_0, "c =", c, "s =", s, "Z =", Z)
