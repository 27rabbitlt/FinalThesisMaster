#pragma once
// Shared exact solvers for a posteriori TSP, adaptive TSP, and a priori TSP.
// See CLAUDE.md for definitions and algorithm details.

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <iostream>
using namespace std;

static const double INF = 1e18;

// Floyd-Warshall all-pairs shortest paths (in-place)
inline void floyd_warshall(int V, std::vector<std::vector<double>>& d) {
    for (int k = 0; k < V; ++k)
        for (int i = 0; i < V; ++i)
            for (int j = 0; j < V; ++j)
                if (d[i][k] + d[k][j] < d[i][j])
                    d[i][j] = d[i][k] + d[k][j];
}

// Exact a posteriori expected cost via Held-Karp over all subsets.
// d: (n+1)x(n+1) distance matrix, vertex 0 = depot, 1..n = customers.
// p: size n+1, p[0]=0, p[i] = activation probability of customer i.
inline double solve_a_posteriori(int n, const std::vector<std::vector<double>>& d,
                                 const std::vector<double>& p) {
    int full = (1 << n);
    std::vector<std::vector<double>> dp(full, std::vector<double>(n, INF));
    for (int j = 0; j < n; ++j)
        dp[1 << j][j] = d[0][j + 1];
    for (int mask = 1; mask < full; ++mask) {
        for (int j = 0; j < n; ++j) {
            if (!(mask & (1 << j))) continue;
            int prev = mask ^ (1 << j);
            if (prev == 0) continue;
            dp[mask][j] = INF;
            for (int k = 0; k < n; ++k) {
                if (!(prev & (1 << k))) continue;
                dp[mask][j] = std::min(dp[mask][j], dp[prev][k] + d[k + 1][j + 1]);
            }
        }
    }
    std::vector<double> OPT(full, 0.0);
    for (int mask = 1; mask < full; ++mask) {
        OPT[mask] = INF;
        for (int j = 0; j < n; ++j) {
            if (!(mask & (1 << j))) continue;
            OPT[mask] = std::min(OPT[mask], dp[mask][j] + d[j + 1][0]);
        }
    }
    double ans = 0.0;
    for (int mask = 0; mask < full; ++mask) {
        double prob = 1.0;
        for (int i = 0; i < n; ++i)
            prob *= (mask & (1 << i)) ? p[i + 1] : (1.0 - p[i + 1]);
        ans += prob * OPT[mask];
    }
    return ans;
}

// Exact adaptive expected cost via DP over (position, probed_mask).
inline double solve_adaptive(int n, const std::vector<std::vector<double>>& d,
                             const std::vector<double>& p) {
    int full = (1 << n);
    std::vector<std::vector<double>> F(n + 1, std::vector<double>(full, 0.0));
    for (int pos = 0; pos <= n; ++pos)
        F[pos][full - 1] = d[pos][0];
    for (int bits = n - 1; bits >= 0; --bits) {
        for (int mask = 0; mask < full; ++mask) {
            if (__builtin_popcount(mask) != bits) continue;
            for (int pos = 0; pos <= n; ++pos) {
                double best = INF;
                for (int i = 0; i < n; ++i) {
                    if (mask & (1 << i)) continue;
                    int next = mask | (1 << i);
                    double val = (1.0 - p[i + 1]) * F[pos][next]
                               + p[i + 1] * (d[pos][i + 1] + F[i + 1][next]);
                    best = std::min(best, val);
                }
                F[pos][mask] = best;
            }
        }
    }
    return F[0][0];
}

// Helper: expected shortcut cost of one fixed a priori tour.
// perm[k] is the 0-based index of the (k+1)-th customer in the tour.
// Tour visits: depot 0, customers perm[0]+1, ..., perm[n-1]+1, depot 0.
// Inactive customers are skipped (shortcutted).
static inline double a_priori_tour_cost(int n,
                                        const std::vector<std::vector<double>>& d,
                                        const std::vector<double>& p,
                                        const std::vector<int>& perm) {
    // Build tour array: u[0]=depot, u[1..n]=customers, u[n+1]=depot
    std::vector<int> u(n + 2);
    u[0] = 0;
    for (int k = 0; k < n; ++k) u[k + 1] = perm[k] + 1;
    u[n + 1] = 0;

    double expected = 0.0;
    for (int i = 0; i <= n; ++i) {
        // Probability that u[i] is "present" (depot = always; customer = p[u[i]])
        double p_i = (i == 0) ? 1.0 : p[u[i]];
        if (p_i == 0.0) continue;
        // absent_prod = product of (1-p[u[k]]) for k = i+1 .. j-1
        double absent_prod = 1.0;
        for (int j = i + 1; j <= n + 1; ++j) {
            double p_j = (j == n + 1) ? 1.0 : p[u[j]];
            double added = d[u[i]][u[j]] * p_i * absent_prod * p_j;
            expected += added;
            if (j <= n) absent_prod *= (1.0 - p[u[j]]);
        }
    }
    return expected;
}

// Exact a priori expected cost: enumerate all n! master tours, return minimum.
// For each tour, inactive customers are shortcutted; expected cost computed in O(n^2).
// Overall time: O(n! * n^2).  Practical for n <= 11 or so.
inline double solve_a_priori(int n, const std::vector<std::vector<double>>& d,
                             const std::vector<double>& p) {
    if (n == 0) return 0.0;
    std::vector<int> perm(n);
    std::vector<int> bestperm(n);
    std::iota(perm.begin(), perm.end(), 0);
    double best = INF;
    do {
        double cost = a_priori_tour_cost(n, d, p, perm);
        if (cost < best) { best = cost; bestperm = perm; }
    } while (std::next_permutation(perm.begin(), perm.end()));
    return best;
}
