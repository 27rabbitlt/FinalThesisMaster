#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

// Exhaust all {1,2}-metrics on a depot plus five clients.  Every such
// symmetric matrix is automatically metric.  For a fixed common activation
// probability, compute OPT_post and the best rooted open shortcut order.
// This is a finite falsifier for proposed open-order inequalities.

namespace {
constexpr int n = 5;
constexpr int vertices = n + 1;
constexpr int edge_count = vertices * (vertices - 1) / 2;

int edge_index(int u, int v) {
    if (u > v) std::swap(u, v);
    int index = 0;
    for (int i = 0; i < u; ++i) index += vertices - i - 1;
    return index + (v - u - 1);
}

int distance(std::uint32_t metric, int u, int v) {
    if (u == v) return 0;
    return 1 + static_cast<int>((metric >> edge_index(u, v)) & 1U);
}

double subset_probability(int subset, double p) {
    const int active = __builtin_popcount(static_cast<unsigned>(subset));
    return std::pow(p, active) * std::pow(1.0 - p, n - active);
}

double posterior(std::uint32_t metric, double p) {
    std::array<std::array<int, n>, 1 << n> dp{};
    constexpr int infinity = 1'000'000;
    for (auto& row : dp) row.fill(infinity);
    for (int last = 0; last < n; ++last) {
        dp[1 << last][last] = distance(metric, 0, last + 1);
    }
    for (int subset = 1; subset < (1 << n); ++subset) {
        for (int last = 0; last < n; ++last) {
            if (!(subset & (1 << last))) continue;
            const int previous = subset ^ (1 << last);
            if (!previous) continue;
            for (int before = 0; before < n; ++before) {
                if (previous & (1 << before)) {
                    dp[subset][last] = std::min(
                        dp[subset][last],
                        dp[previous][before] + distance(metric, before + 1, last + 1)
                    );
                }
            }
        }
    }
    double value = 0.0;
    for (int subset = 1; subset < (1 << n); ++subset) {
        int best = infinity;
        for (int last = 0; last < n; ++last) {
            if (subset & (1 << last)) {
                best = std::min(best, dp[subset][last] + distance(metric, last + 1, 0));
            }
        }
        value += subset_probability(subset, p) * best;
    }
    return value;
}

double expected_farthest(std::uint32_t metric, double p) {
    double value = 0.0;
    for (int subset = 1; subset < (1 << n); ++subset) {
        int farthest = 0;
        for (int v = 0; v < n; ++v) {
            if (subset & (1 << v)) {
                farthest = std::max(farthest, distance(metric, 0, v + 1));
            }
        }
        value += subset_probability(subset, p) * farthest;
    }
    return value;
}

double open_cost(std::uint32_t metric, double p, const std::array<int, n>& order) {
    const double q = 1.0 - p;
    double value = 0.0;
    double empty_prefix = 1.0;
    for (int i = 0; i < n; ++i) {
        value += p * empty_prefix * distance(metric, 0, order[i]);
        double empty_gap = 1.0;
        for (int j = i + 1; j < n; ++j) {
            value += p * p * empty_gap * distance(metric, order[i], order[j]);
            empty_gap *= q;
        }
        empty_prefix *= q;
    }
    return value;
}

double optimum_open(std::uint32_t metric, double p) {
    std::array<int, n> order{1, 2, 3, 4, 5};
    double best = std::numeric_limits<double>::infinity();
    do {
        best = std::min(best, open_cost(metric, p, order));
    } while (std::next_permutation(order.begin(), order.end()));
    return best;
}

std::array<double, vertices> radii(std::uint32_t metric, double p) {
    const double q = 1.0 - p;
    std::array<double, vertices> rho{};
    for (int v = 1; v < vertices; ++v) {
        if (distance(metric, 0, v) == 1) {
            rho[v] = q;
            continue;
        }
        int short_neighbors = 0;
        for (int u = 1; u < vertices; ++u) {
            if (u != v && distance(metric, u, v) == 1) ++short_neighbors;
        }
        const double mu = 1.0 + std::pow(q, short_neighbors);
        rho[v] = q * mu;
    }
    return rho;
}

std::array<double, vertices> mean_nearest(std::uint32_t metric, double p) {
    const double q = 1.0 - p;
    std::array<double, vertices> mu{};
    for (int v = 1; v < vertices; ++v) {
        if (distance(metric, 0, v) == 1) {
            mu[v] = 1.0;
            continue;
        }
        int short_neighbors = 0;
        for (int u = 1; u < vertices; ++u) {
            if (u != v && distance(metric, u, v) == 1) ++short_neighbors;
        }
        mu[v] = 1.0 + std::pow(q, short_neighbors);
    }
    return mu;
}

double residual_edge(
    std::uint32_t metric,
    const std::array<double, vertices>& rho,
    int u,
    int v
) {
    return std::max(distance(metric, u, v) - rho[u] - rho[v], 0.0);
}

double residual_open_cost(
    std::uint32_t metric,
    double p,
    const std::array<double, vertices>& rho,
    const std::array<int, n>& order
) {
    const double q = 1.0 - p;
    double value = 0.0;
    double empty_prefix = 1.0;
    for (int i = 0; i < n; ++i) {
        value += p * empty_prefix * residual_edge(metric, rho, 0, order[i]);
        double empty_gap = 1.0;
        for (int j = i + 1; j < n; ++j) {
            value += p * p * empty_gap
                * residual_edge(metric, rho, order[i], order[j]);
            empty_gap *= q;
        }
        empty_prefix *= q;
    }
    return value;
}

double optimum_residual_open(
    std::uint32_t metric,
    double p,
    const std::array<double, vertices>& rho
) {
    std::array<int, n> order{1, 2, 3, 4, 5};
    double best = std::numeric_limits<double>::infinity();
    do {
        best = std::min(best, residual_open_cost(metric, p, rho, order));
    } while (std::next_permutation(order.begin(), order.end()));
    return best;
}


double optimum_deep_residual_open(
    std::uint32_t metric,
    double p,
    const std::array<double, vertices>& mu
) {
    const auto deep_edge = [&](int u, int v) {
        return std::max(distance(metric, u, v) - mu[u] - mu[v], 0.0);
    };
    const double q = 1.0 - p;
    std::array<int, n> order{1, 2, 3, 4, 5};
    double best = std::numeric_limits<double>::infinity();
    do {
        double value = 0.0;
        double empty_prefix = 1.0;
        for (int i = 0; i < n; ++i) {
            value += p * empty_prefix * deep_edge(0, order[i]);
            double empty_gap = 1.0;
            for (int j = i + 1; j < n; ++j) {
                value += p * p * empty_gap * deep_edge(order[i], order[j]);
                empty_gap *= q;
            }
            empty_prefix *= q;
        }
        best = std::min(best, value);
    } while (std::next_permutation(order.begin(), order.end()));
    return best;
}
}  // namespace

int main(int argc, char** argv) {
    const double p = argc > 1 ? std::stod(argv[1]) : 0.5;
    double worst_ratio = 0.0;
    std::uint32_t worst_metric = 0;
    double worst_post = 0.0;
    double worst_open = 0.0;
    double worst_residual_ratio = 0.0;
    std::uint32_t worst_residual_metric = 0;
    double worst_residual = 0.0;
    double worst_deficit = 0.0;
    double worst_deep_ratio = 0.0;
    std::uint32_t worst_deep_metric = 0;
    double worst_deep = 0.0;
    double worst_second_budget = 0.0;
    double worst_close_ratio = 0.0;
    std::uint32_t worst_close_metric = 0;
    const std::uint32_t count = 1U << edge_count;
    for (std::uint32_t metric = 0; metric < count; ++metric) {
        const double post = posterior(metric, p);
        const double open = optimum_open(metric, p);
        const double ratio = open / post;
        if (ratio > worst_ratio) {
            worst_ratio = ratio;
            worst_metric = metric;
            worst_post = post;
            worst_open = open;
        }
        const auto rho = radii(metric, p);
        double radial_mass = 0.0;
        for (int v = 1; v < vertices; ++v) radial_mass += p * rho[v];
        const double deficit = post - radial_mass;
        const double close_ratio = expected_farthest(metric, p) / deficit;
        if (close_ratio > worst_close_ratio) {
            worst_close_ratio = close_ratio;
            worst_close_metric = metric;
        }
        const double residual = optimum_residual_open(metric, p, rho);
        const double residual_ratio = residual / deficit;
        if (residual_ratio > worst_residual_ratio) {
            worst_residual_ratio = residual_ratio;
            worst_residual_metric = metric;
            worst_residual = residual;
            worst_deficit = deficit;
        }
        const auto mu = mean_nearest(metric, p);
        double nearest_mass = 0.0;
        for (int v = 1; v < vertices; ++v) nearest_mass += p * mu[v];
        const double second_budget = post - nearest_mass;
        const double deep = optimum_deep_residual_open(metric, p, mu);
        const double deep_ratio = second_budget > 1e-14 ? deep / second_budget : 0.0;
        if (deep_ratio > worst_deep_ratio) {
            worst_deep_ratio = deep_ratio;
            worst_deep_metric = metric;
            worst_deep = deep;
            worst_second_budget = second_budget;
        }
    }
    std::cout << std::setprecision(15)
              << "p=" << p << " metrics=" << count
              << " worst_Q/P=" << worst_ratio
              << " Q=" << worst_open << " P=" << worst_post
              << " mask=" << worst_metric << '\n';
    for (int u = 0; u < vertices; ++u) {
        for (int v = 0; v < vertices; ++v) {
            std::cout << distance(worst_metric, u, v)
                      << (v + 1 == vertices ? '\n' : ' ');
        }
    }
    std::cout << std::setprecision(15)
              << "worst residual/D=" << worst_residual_ratio
              << " residual=" << worst_residual << " D=" << worst_deficit
              << " mask=" << worst_residual_metric << '\n';
    for (int u = 0; u < vertices; ++u) {
        for (int v = 0; v < vertices; ++v) {
            std::cout << distance(worst_residual_metric, u, v)
                      << (v + 1 == vertices ? '\n' : ' ');
        }
    }
    std::cout << std::setprecision(15)
              << "worst deep/(P-L)=" << worst_deep_ratio
              << " deep=" << worst_deep << " P-L=" << worst_second_budget
              << " mask=" << worst_deep_metric << '\n';
    for (int u = 0; u < vertices; ++u) {
        for (int v = 0; v < vertices; ++v) {
            std::cout << distance(worst_deep_metric, u, v)
                      << (v + 1 == vertices ? '\n' : ' ');
        }
    }
    std::cout << std::setprecision(15)
              << "worst farthest/D=" << worst_close_ratio
              << " mask=" << worst_close_metric << '\n';
}
