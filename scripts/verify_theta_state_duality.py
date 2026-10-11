#!/usr/bin/env python3
"""Exact checks for theta tree states and two-terminal composition identities."""

from fractions import Fraction as F
from functools import reduce
from itertools import product
from operator import mul

from check_theta_optional_state import (
    connector_states,
    path_statistics,
    posterior_states,
    realization_probability,
)


def fprod(values):
    return reduce(mul, values, F(1))


def theta_state(paths, active_bits):
    lengths = [sum(path, F()) for path in paths]
    total = sum(lengths, F())
    maxima = [
        path_statistics(path, bits)[0]
        for path, bits in zip(paths, active_bits)
    ]
    radii_s = [
        path_statistics(path, bits)[1]
        for path, bits in zip(paths, active_bits)
    ]
    radii_t = [
        path_statistics(tuple(reversed(path)), tuple(reversed(bits)))[1]
        for path, bits in zip(paths, active_bits)
    ]
    retained = [lengths[i] - maxima[i] for i in range(3)]
    separated = 2 * sum(retained, F())
    opened = min(
        [total]
        + [
            lengths[i] + 2 * retained[j] + 2 * retained[k]
            for i in range(3)
            for j, k in [tuple(x for x in range(3) if x != i)]
        ]
    )
    even = min(
        [total + lengths[k] - 2 * maxima[k] for k in range(3)]
        + [
            2 * (lengths[i] + retained[j] + retained[k])
            for i in range(3)
            for j, k in [tuple(x for x in range(3) if x != i)]
        ]
    )
    raw_s = 2 * sum(radii_s, F())
    raw_t = 2 * sum(radii_t, F())
    return (
        separated,
        even,
        opened,
        raw_s,
        raw_t,
        min(raw_s, even),
        min(raw_t, even),
    )


def expected_path_state(lengths, probabilities):
    length = sum(lengths, F())
    separated = rooted_s = rooted_t = F()
    for bits in product((False, True), repeat=len(probabilities)):
        probability = realization_probability(bits, probabilities)
        maximum, radius_s = path_statistics(lengths, bits)
        radius_t = path_statistics(
            tuple(reversed(lengths)), tuple(reversed(bits))
        )[1]
        separated += probability * 2 * (length - maximum)
        rooted_s += probability * 2 * radius_s
        rooted_t += probability * 2 * radius_t
    empty = fprod(1 - p for p in probabilities)
    return separated, 2 * length, length, rooted_s, rooted_t, empty


def audit_tree_state_formula():
    templates = (
        ((F(1), F(2), F(3)), (F(1, 3), F(3, 5))),
        ((F(3), F(1)), (F(2, 7),)),
        ((F(2), F(4), F(1)), (F(1, 2), F(4, 5))),
    )
    backbone = 1
    cuts = (1, 0, 2)
    predicted = connector_states(templates, backbone, cuts)
    totals = [sum(lengths, F()) for lengths, _ in templates]
    actual = [F(), F(), F()]
    bit_spaces = [
        list(product((False, True), repeat=len(probabilities)))
        for _, probabilities in templates
    ]
    for bits_by_path in product(*bit_spaces):
        probability = fprod(
            realization_probability(bits, probabilities)
            for bits, (_, probabilities) in zip(bits_by_path, templates)
        )
        branches = []
        t_trigger = False
        for i, (((lengths, _), bits), cut) in enumerate(
            zip(zip(templates, bits_by_path), cuts)
        ):
            prefix = [F()]
            for length in lengths:
                prefix.append(prefix[-1] + length)
            left = max(
                [j + 1 for j, active in enumerate(bits[:cut]) if active]
                or [0]
            )
            right = min(
                [j + 1 for j, active in enumerate(bits[cut:], start=cut) if active]
                or [len(lengths)]
            )
            branches.append(prefix[left] + totals[i] - prefix[right])
            if i != backbone and any(bits[cut:]):
                t_trigger = True
        _, radius = path_statistics(
            templates[backbone][0], bits_by_path[backbone]
        )
        local_branches = sum(
            branches[i] for i in range(3) if i != backbone
        )
        actual[0] += probability * (
            local_branches + (totals[backbone] if t_trigger else radius)
        )
        actual[1] += probability * (local_branches + totals[backbone])
        actual[2] += probability * sum(branches, F())
    assert tuple(actual) == predicted


def audit_series_recurrence_obstructions():
    # The invalid S/R recurrence can overestimate the true D state by 9/5.
    child_1 = theta_state(
        ((F(1), F(1)), (F(1), F(1)), (F(3), F(1))),
        ((True,), (True,), (True,)),
    )
    child_2 = theta_state(
        ((F(1), F(1)), (F(1), F(3)), (F(1), F(3))),
        ((False,), (True,), (True,)),
    )
    assert child_1 == (F(6), F(6), F(6), F(10), F(6), F(6), F(6))
    assert child_2 == (F(4), F(8), F(6), F(4), F(12), F(4), F(8))
    d1, e1, _, r1s, _, s1s, _ = child_1
    d2, e2, _, _, r2t, _, s2t = child_2
    correct = min(e1 + d2, d1 + e2, r1s + r2t)
    invalid = min(s1s + r2t, r1s + s2t)
    assert correct == 10
    assert invalid == 18
    assert invalid / correct == F(9, 5)

    # Even (D,E,O,Ss,St) is insufficient: raw R cannot be collapsed to S.
    child = theta_state(
        ((F(1), F(1)), (F(1), F(3), F(3)), (F(1), F(5))),
        ((True,), (True, True), (False,)),
    )
    assert child == (F(10), F(9), F(9), F(10), F(14), F(9), F(9))
    d, e, _, rs, rt, ss, st = child
    correct = min(e + d, d + e, rs + rt)
    invalid_collapsed = ss + st
    assert correct == 19
    assert invalid_collapsed == 18


def audit_empty_probability_obstruction():
    block_a = expected_path_state(
        (F(1), F(2), F(3)), (F(3, 7), F(3, 4))
    )
    block_b = expected_path_state(
        (F(2), F(1), F(3)), (F(3, 4), F(4, 7))
    )
    assert block_a[:5] == block_b[:5]
    assert block_a[:5] == (
        F(33, 7),
        F(12),
        F(6),
        F(33, 7),
        F(48, 7),
    )
    assert block_a[5] == F(1, 7)
    assert block_b[5] == F(3, 28)

    left_rooted = F(2)
    left_even = F(4)

    def composed_rooted(block):
        empty = block[5]
        right_rooted = block[3]
        return (
            empty * left_rooted
            + (1 - empty) * left_even
            + right_rooted
        )

    composed_a = composed_rooted(block_a)
    composed_b = composed_rooted(block_b)
    assert composed_a == F(59, 7)
    assert composed_b == F(119, 14)
    assert composed_b - composed_a == F(1, 14)


def audit_minimum_credit_backbone_obstruction():
    # P1 has an always-active vertex at distance 1; P2 and P3 are empty
    # direct paths.  Length-biased cut credits are deterministic here.
    lengths = (F(100), F(95), F(200))
    credits = (F(4901, 50), F(95), F(200))
    base = sum(lengths, F()) - sum(credits, F())
    final_gap_product = F(99, 100)
    assert base == F(99, 50)
    assert min(range(3), key=credits.__getitem__) == 1

    h0_min_credit = base + credits[1] - final_gap_product * lengths[1]
    rooted_posterior = F(2)
    assert h0_min_credit == F(293, 100)
    assert h0_min_credit / rooted_posterior == F(293, 200)
    assert h0_min_credit / rooted_posterior - F(29, 20) == F(3, 200)

    h0_active_path = base + credits[0] - final_gap_product * lengths[0]
    h1_active_path = base + credits[0]
    posterior_d, posterior_e, posterior_o = F(2), F(192), F(97)
    assert h0_active_path == 1
    assert h1_active_path == 100
    assert h0_active_path <= F(29, 20) * rooted_posterior
    assert h1_active_path <= F(29, 20) * posterior_e
    assert h1_active_path <= F(29, 20) * posterior_o
    assert base <= F(29, 20) * posterior_d


def audit_separator_occupancy_obstruction():
    # Equal scalar branch hazards do not imply equal trigger probabilities.
    branch_hazard_a, trigger_a = F(1), F(1)
    branch_hazard_b, trigger_b = F(2) * F(1, 2), F(1, 2)
    assert branch_hazard_a == branch_hazard_b
    connector_gap = F(6)  # H1-H0 of an empty length-6 connector.
    combined_a = branch_hazard_a + trigger_a * connector_gap
    combined_b = branch_hazard_b + trigger_b * connector_gap
    assert combined_a == 7
    assert combined_b == 4


def length_biased_dual(templates):
    """Return the exact maximum in the two-face LP dual (TS9c)."""
    lengths = []
    credits = []
    final_fractions = []
    for edge_lengths, probabilities in templates:
        length = sum(edge_lengths, F())
        lengths.append(length)
        prefix = [F()]
        for edge_length in edge_lengths:
            prefix.append(prefix[-1] + edge_length)
        credit = final_fraction = F()
        for bits in product((False, True), repeat=len(probabilities)):
            probability = realization_probability(bits, probabilities)
            marks = (
                [0]
                + [j + 1 for j, active in enumerate(bits) if active]
                + [len(edge_lengths)]
            )
            gaps = [
                prefix[right] - prefix[left]
                for left, right in zip(marks, marks[1:])
            ]
            credit += probability * sum(gap * gap for gap in gaps) / length
            _, radius = path_statistics(edge_lengths, bits)
            final_fraction += probability * (length - radius) / length
        credits.append(credit)
        final_fractions.append(final_fraction)

    final_product = fprod(final_fractions)
    base = sum(lengths, F()) - sum(credits, F())
    rooted, even, opened, _ = posterior_states(templates)
    required = min(even, opened)
    candidates = {F(0), F(1)}
    for i in range(3):
        for j in range(i):
            denominator = final_product * (lengths[i] - lengths[j])
            if denominator:
                crossing = (credits[i] - credits[j]) / denominator
                if 0 <= crossing <= 1:
                    candidates.add(crossing)

    ratios = []
    for weight in candidates:
        numerator = base + min(
            credits[i] - weight * final_product * lengths[i]
            for i in range(3)
        )
        denominator = weight * rooted + (1 - weight) * required
        if denominator:
            ratios.append((numerator / denominator, weight))
    # On every interval between crossings the ratio is affine-over-affine,
    # hence monotone (or constant); endpoints and crossings are exhaustive.
    return max(ratios), (base, tuple(credits), final_product, tuple(lengths))


def audit_exact_two_face_dual():
    templates = (
        ((F(1), F(99)), (F(1),)),
        ((F(95),), ()),
        ((F(200),), ()),
    )
    maximum, data = length_biased_dual(templates)
    assert data == (
        F(99, 50),
        (F(4901, 50), F(95), F(200)),
        F(99, 100),
        (F(100), F(95), F(200)),
    )
    # Although minimum credit alone fails, the compatible lower envelope is
    # safely below 29/20 (the active path is selected near the optional face).
    assert maximum[0] <= F(29, 20)


def oriented_surplus_moment(edge_lengths, probabilities):
    """Return (L, alpha, z, d) from (TS9f)--(TS9h)."""
    length = sum(edge_lengths, F())
    prefix = [F()]
    for edge_length in edge_lengths:
        prefix.append(prefix[-1] + edge_length)
    alpha = final_fraction = d_moment = F()
    for bits in product((False, True), repeat=len(probabilities)):
        probability = realization_probability(bits, probabilities)
        marks = (
            [0]
            + [j + 1 for j, active in enumerate(bits) if active]
            + [len(edge_lengths)]
        )
        gaps = [
            prefix[right] - prefix[left]
            for left, right in zip(marks, marks[1:])
        ]
        normalized_credit = sum(gap * gap for gap in gaps) / (length * length)
        normalized_final = gaps[-1] / length
        rooted_residual = 1 - 2 * normalized_final + normalized_credit
        alpha += probability * normalized_credit
        final_fraction += probability * normalized_final
        d_moment += probability * normalized_credit * (
            1 - min(rooted_residual, F(1))
        )
    return length, alpha, final_fraction, d_moment


def audit_oriented_rooted_surplus_certificate():
    # This exactly realizable instance refutes the coarse scalar relaxation
    # (TS9j), while the orientation-sensitive product (TS9h) has ample slack.
    templates = (
        ((F(725, 32), F(75, 32)), (F(32, 87),)),
        ((F(12),), ()),
        ((F(30), F(18)), (F(8, 15),)),
    )
    moments = [oriented_surplus_moment(*template) for template in templates]
    lengths = tuple(moment[0] for moment in moments)
    alphas = tuple(moment[1] for moment in moments)
    final_fractions = tuple(moment[2] for moment in moments)
    d_moments = tuple(moment[3] for moment in moments)
    assert lengths == (F(25), F(12), F(48))
    assert alphas == (F(15, 16), F(1), F(3, 4))
    assert final_fractions == (F(2, 3), F(1), F(2, 3))
    assert d_moments == (F(55, 87), F(1), F(203, 384))

    base = sum(
        lengths[i] * (1 - alphas[i]) for i in range(3)
    )
    final_product = fprod(final_fractions)
    square_product = fprod(alphas)
    charge = min(
        lengths[i] * (alphas[i] - final_product) for i in range(3)
    )
    rooted, _, _, _ = posterior_states(templates)
    ell = min(lengths)
    assert base == F(217, 16)
    assert final_product == F(4, 9)
    assert charge == F(20, 3)
    assert rooted == F(16462, 435)
    assert (base + charge) / base == F(971, 651) > F(29, 20)

    oriented_surplus = ell * (square_product - fprod(d_moments))
    assert oriented_surplus == F(425, 96)
    assert rooted - base >= oriented_surplus
    assert (base + charge) / (base + oriented_surplus) == F(1942, 1727)

    # Exact arithmetic behind the sharpened scalar coefficient (TS9l), and
    # the remaining endpoint target (TS9o) on this warning instance.
    assert 4 * 29**3 - 46**3 == 220
    endpoint_residual = fprod(
        alpha - final_product for alpha in alphas
    ) / (1 - final_product) ** 2
    assert endpoint_residual == F(781, 2880)
    assert ell * endpoint_residual == F(781, 240)
    assert (
        ell * endpoint_residual
        <= rooted - base + F(7, 116) * base
    )


def realization_normalized_data(edge_lengths, bits):
    length = sum(edge_lengths, F())
    prefix = [F()]
    for edge_length in edge_lengths:
        prefix.append(prefix[-1] + edge_length)
    marks = (
        [0]
        + [j + 1 for j, active in enumerate(bits) if active]
        + [len(edge_lengths)]
    )
    gaps = [
        prefix[right] - prefix[left]
        for left, right in zip(marks, marks[1:])
    ]
    return (
        sum(gap * gap for gap in gaps) / (length * length),
        gaps[-1] / length,
        max(gaps) / length,
    )


def audit_optional_endpoint_shrinking():
    """Exact audit of every local inequality used in (TS9p)."""
    templates = (
        ((F(1), F(2), F(3)), (F(1, 3), F(3, 5))),
        ((F(3), F(1)), (F(2, 7),)),
        ((F(2), F(4), F(1)), (F(1, 2), F(4, 5))),
    )
    lengths = [sum(edges, F()) for edges, _ in templates]
    ell = min(lengths)
    bit_spaces = [
        list(product((False, True), repeat=len(probabilities)))
        for _, probabilities in templates
    ]
    expected_base = expected_rooted = F()
    alpha = [F(), F(), F()]
    final_fraction = [F(), F(), F()]
    for bits_by_path in product(*bit_spaces):
        probability = fprod(
            realization_probability(bits, probabilities)
            for bits, (_, probabilities) in zip(bits_by_path, templates)
        )
        normalized = [
            realization_normalized_data(edges, bits)
            for (edges, _), bits in zip(templates, bits_by_path)
        ]
        v = [item[0] for item in normalized]
        t = [item[1] for item in normalized]
        largest = [item[2] for item in normalized]
        for i in range(3):
            alpha[i] += probability * v[i]
            final_fraction[i] += probability * t[i]

        baseline_coefficients = [1 - value for value in v]
        direct_coefficients = [2 * (1 - value) for value in t]
        assert all(
            direct_coefficients[i] >= baseline_coefficients[i]
            for i in range(3)
        )
        alternatives = [direct_coefficients]
        for k in range(3):
            alternatives.append([
                2 * (1 - largest[h]) if h == k else F(1)
                for h in range(3)
            ])
        for i in range(3):
            alternatives.append([
                F(2) if h == i else 2 * (1 - largest[h])
                for h in range(3)
            ])
        assert all(
            coefficient >= baseline_coefficients[i]
            for alternative in alternatives
            for i, coefficient in enumerate(alternative)
        )

        baseline = sum(
            lengths[i] * baseline_coefficients[i] for i in range(3)
        )
        rooted = min(
            sum(lengths[i] * alternative[i] for i in range(3))
            for alternative in alternatives
        )
        equal_baseline = ell * sum(baseline_coefficients, F())
        equal_rooted = ell * min(
            sum(alternative, F()) for alternative in alternatives
        )
        # S(A)-A(A) cannot increase when longer paths are shrunk to ell.
        assert rooted - baseline >= equal_rooted - equal_baseline
        expected_base += probability * baseline
        expected_rooted += probability * rooted

    z = fprod(final_fraction)
    assert z < 1
    if all(value >= z for value in alpha):
        endpoint = ell * fprod(value - z for value in alpha) / (1 - z) ** 2
        assert endpoint <= expected_rooted - expected_base + expected_base / 54

    # The equal-length scalar inequality (TS9r), exhaustively on a small
    # rational mesh including its equality point (2/3,2/3,2/3).
    mesh = [F(i, 6) for i in range(7)]
    for x in product(mesh, repeat=3):
        total = sum(x, F())
        product_residual = fprod(1 - value for value in x)
        assert product_residual <= max(F(), 2 - total) + total / 54
    equality = (F(2, 3),) * 3
    assert fprod(1 - value for value in equality) == F(1, 27)
    assert sum(equality, F()) / 54 == F(1, 27)


def audit_optional_dual_sign_case():
    # On P1 the normalized square-gap moment 3/8 is below the final-gap
    # product 1/2, so lambda*=3/4 exercises the negative-factor branch.
    templates = (
        ((F(1), F(1), F(2)), (F(1), F(1))),
        ((F(5),), ()),
        ((F(7),), ()),
    )
    maximum, (base, credits, z, lengths) = length_biased_dual(templates)
    alpha = tuple(credits[i] / lengths[i] for i in range(3))
    assert alpha == (F(3, 8), F(1), F(1))
    assert z == F(1, 2)
    lambda_star = min(alpha) / z
    assert lambda_star == F(3, 4)
    square_product = fprod(alpha)
    for weight in (F(), F(1, 4), F(1, 2)):
        residual = fprod(value - weight * z for value in alpha) / (
            1 - weight * z
        ) ** 2
        assert residual <= (1 - weight) * square_product
    for weight in (lambda_star, F(7, 8), F(1)):
        assert min(
            credits[i] - weight * z * lengths[i] for i in range(3)
        ) <= 0
    assert maximum[0] <= F(29, 20)

    # Exact coefficient bookkeeping from (TS9m), (TS9p), and (TS9c).
    assert F(1, 54) < F(7, 116)
    assert F(29, 4) + F(29, 54) < 9


def main():
    audit_tree_state_formula()
    audit_series_recurrence_obstructions()
    audit_empty_probability_obstruction()
    audit_minimum_credit_backbone_obstruction()
    audit_separator_occupancy_obstruction()
    audit_exact_two_face_dual()
    audit_oriented_rooted_surplus_certificate()
    audit_optional_endpoint_shrinking()
    audit_optional_dual_sign_case()
    print("theta tree state (H0,H1,HD): exact")
    print("series D recurrence and raw-R obstruction: exact")
    print("empty-probability moment match: exact (gap 1/14)")
    print("minimum-credit backbone obstruction: exact (excess 3/200)")
    print("separator occupancy obstruction: exact (7 versus 4)")
    print("two-face LP dual: exact endpoint/crossing audit")
    print("oriented rooted-surplus certificate: exact (coarse relaxation refuted)")
    print("optional endpoint shrinking and all-sign dual reduction: exact")


if __name__ == "__main__":
    main()
