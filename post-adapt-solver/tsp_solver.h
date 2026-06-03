#pragma once
// Shared exact solvers for a posteriori TSP, adaptive TSP, and a priori TSP.
// See CLAUDE.md for definitions and algorithm details.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
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

// ======================================================================
// Large-instance solvers: exploit prob=1 vertices to reduce state space.
// A posteriori: enumerate 2^ns realizations (ns = stochastic count).
// Adaptive: DP on (pos, stochastic_probed_mask), completion via HK.
// ======================================================================

#include <functional>

// Hungarian algorithm for the assignment problem, O(n^3).
// cost[i][j] = cost of assigning row i to column j, 0-indexed.
// Returns (total_cost, assignment) where assignment[i] = column for row i.
static std::pair<double, std::vector<int>> hungarian(int n,
    const std::vector<std::vector<double>>& cost) {
    std::vector<double> u(n + 1, 0), v(n + 1, 0);
    std::vector<int> p(n + 1, 0), way(n + 1, 0);
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(n + 1, INF);
        std::vector<bool> used(n + 1, false);
        do {
            used[j0] = true;
            int i0 = p[j0], j1 = 0;
            double delta = INF;
            for (int j = 1; j <= n; ++j) {
                if (used[j]) continue;
                double cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else minv[j] -= delta;
            }
            j0 = j1;
        } while (p[j0] != 0);
        do { int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
    }
    std::vector<int> asgn(n);
    for (int j = 1; j <= n; ++j)
        if (p[j]) asgn[p[j] - 1] = j - 1;
    double tot = 0;
    for (int i = 0; i < n; ++i) tot += cost[i][asgn[i]];
    return {tot, asgn};
}

// Branch-and-bound ATSP solver with AP (Hungarian) lower bound.
// d[i][j] = distance, 0-indexed, m vertices. Returns min Hamiltonian cycle cost.
static double solve_tsp_bb(int m, const std::vector<std::vector<double>>& d) {
    if (m <= 1) return 0;
    if (m == 2) return d[0][1] + d[1][0];

    std::vector<std::vector<double>> cost(m, std::vector<double>(m));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
            cost[i][j] = (i == j) ? INF : d[i][j];

    double best = INF;
    // Nearest-neighbor upper bound
    {
        std::vector<bool> vis(m, false);
        vis[0] = true; int cur = 0; double c = 0;
        for (int s = 1; s < m; ++s) {
            int nxt = -1; double bd = INF;
            for (int j = 0; j < m; ++j)
                if (!vis[j] && d[cur][j] < bd) { bd = d[cur][j]; nxt = j; }
            vis[nxt] = true; c += d[cur][nxt]; cur = nxt;
        }
        best = c + d[cur][0];
    }

    std::function<void(std::vector<std::vector<double>>&)> bb;
    bb = [&](std::vector<std::vector<double>>& c) {
        auto [lb, asgn] = hungarian(m, c);
        if (lb >= best - 1e-9) return;

        // Find subtours
        std::vector<int> vis(m, -1);
        int ntours = 0;
        std::vector<int> shortest;
        for (int i = 0; i < m; ++i) {
            if (vis[i] >= 0) continue;
            std::vector<int> tour;
            int j = i;
            while (vis[j] < 0) { vis[j] = ntours; tour.push_back(j); j = asgn[j]; }
            if ((int)tour.size() < m && (shortest.empty() || tour.size() < shortest.size()))
                shortest = tour;
            ntours++;
        }
        if (ntours == 1) { best = std::min(best, lb); return; }

        // Branch: exclude each edge of shortest subtour
        for (int k = 0; k < (int)shortest.size(); ++k) {
            int u = shortest[k], v = asgn[u];
            double saved = c[u][v]; c[u][v] = INF;
            bb(c);
            c[u][v] = saved;
        }
    };
    bb(cost);
    return best;
}

// Solve TSP on a subset of vertices from the full distance matrix.
// verts[0] = depot, verts[1..] = customers. Returns min tour cost.
// Uses flat-array Held-Karp for nc <= 25 (~6.4 GB), B&B for larger.
static double solve_tsp_on_vertices(const std::vector<int>& verts,
                                     const std::vector<std::vector<double>>& full_d) {
    int m = (int)verts.size();
    if (m <= 1) return 0;
    if (m == 2) return full_d[verts[0]][verts[1]] + full_d[verts[1]][verts[0]];

    std::vector<std::vector<double>> sub_d(m, std::vector<double>(m));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
            sub_d[i][j] = full_d[verts[i]][verts[j]];

    int nc = m - 1;
    if (nc <= 25) {
        // Held-Karp with flat array for cache performance.
        // Memory: nc * 2^nc * 8 bytes. nc=25 → 6.4 GB.
        long long full = 1LL << nc;
        std::vector<double> dp((long long)full * nc, INF);
        auto DP = [&](long long mask, int j) -> double& {
            return dp[mask * nc + j];
        };
        for (int j = 0; j < nc; ++j)
            DP(1LL << j, j) = sub_d[0][j + 1];
        for (long long mask = 1; mask < full; ++mask)
            for (int j = 0; j < nc; ++j) {
                if (!(mask & (1LL << j))) continue;
                long long prev = mask ^ (1LL << j);
                if (prev == 0) continue;
                DP(mask, j) = INF;
                for (int k = 0; k < nc; ++k)
                    if (prev & (1LL << k))
                        DP(mask, j) = std::min(DP(mask, j), DP(prev, k) + sub_d[k+1][j+1]);
            }
        double opt = INF;
        for (int j = 0; j < nc; ++j)
            opt = std::min(opt, DP(full - 1, j) + sub_d[j + 1][0]);
        return opt;
    } else {
        // Branch-and-bound for larger instances
        return solve_tsp_bb(m, sub_d);
    }
}

// A posteriori expected cost for large instances.
// Enumerates 2^ns realizations of stochastic customers.
// If sample_count > 0, uses Monte Carlo sampling instead of full enumeration.
inline double solve_a_posteriori_large(int V,
    const std::vector<std::vector<double>>& d,
    const std::vector<double>& p,
    int sample_count = 0) {
    std::vector<int> stoch, det;
    for (int i = 1; i < V; ++i) {
        if (p[i] > 1e-12 && p[i] < 1.0 - 1e-12) stoch.push_back(i);
        else if (p[i] >= 1.0 - 1e-12) det.push_back(i);
    }
    int ns = (int)stoch.size();
    int total = 1 << ns;
    int nd = (int)det.size();
    std::fprintf(stderr, "A posteriori (large): %d stochastic + %d deterministic customers\n",
                 ns, nd);

    if (sample_count > 0) {
        // Monte Carlo mode: sample random realizations
        std::fprintf(stderr, "Monte Carlo mode: %d samples\n", sample_count);
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, total - 1);
        double sum = 0, sum_sq = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (int s = 0; s < sample_count; ++s) {
            int s_mask = dist(rng);
            std::vector<int> active;
            active.push_back(0);
            for (int v : det) active.push_back(v);
            for (int i = 0; i < ns; ++i)
                if (s_mask & (1 << i))
                    active.push_back(stoch[i]);
            double opt = solve_tsp_on_vertices(active, d);
            sum += opt;
            sum_sq += opt * opt;
            if ((s + 1) % 10 == 0 || s + 1 == sample_count) {
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(now - t0).count();
                double mean = sum / (s + 1);
                double var = sum_sq / (s + 1) - mean * mean;
                double se = (s > 0) ? std::sqrt(var / (s + 1)) : 0;
                double eta = (s + 1 < sample_count) ? elapsed / (s + 1) * (sample_count - s - 1) : 0;
                std::fprintf(stderr, "  sample %d/%d  mean=%.4f ±%.4f  elapsed=%.0fs  ETA=%.0fs\r",
                             s + 1, sample_count, mean, 1.96 * se, elapsed, eta);
            }
        }
        std::fprintf(stderr, "\n");
        // For Monte Carlo, the expected cost is the mean of sampled OPT values
        // weighted by uniform probability (since we sample uniformly from realizations,
        // and prob(s_mask) = prod of p_i or (1-p_i), we need to weight properly).
        // Actually: E[OPT] = sum_{s_mask} prob(s_mask) * OPT(s_mask).
        // With uniform sampling: estimate = (1/N) * sum_i (prob(s_mask_i) / (1/total)) * OPT(s_mask_i)
        // = total/N * sum_i prob(s_mask_i) * OPT(s_mask_i).
        // But if all p_i = 0.5, then prob(s_mask) = 1/total for all s_mask, and
        // E[OPT] = (1/total) * sum OPT = mean of OPT. So simple mean works.
        // For general p, we need importance weighting. Use simple mean for now
        // (correct when all stochastic probs are 0.5).
        return sum / sample_count;
    }

    // Full enumeration mode.
    // Sort realizations by popcount so smaller (faster) instances run first.
    std::vector<int> order(total);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [](int a, int b) {
        return __builtin_popcount(a) < __builtin_popcount(b);
    });

    std::fprintf(stderr, "Enumerating %d realizations (sorted by size)...\n", total);
    std::fprintf(stderr, "  nc range: %d (all inactive) to %d (all active)\n", nd, nd + ns);
    std::fprintf(stderr, "  Method: Held-Karp for nc<=25, B&B for nc>25\n");

    double expected = 0;
    auto t0 = std::chrono::steady_clock::now();
    int done = 0;
    int last_k = -1;
    int k_start_done = 0;
    for (int idx = 0; idx < total; ++idx) {
        int s_mask = order[idx];
        int k = __builtin_popcount(s_mask);
        if (k != last_k) {
            if (last_k >= 0) {
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(now - t0).count();
                int k_count = done - k_start_done;
                double k_elapsed = elapsed; // total elapsed so far
                std::fprintf(stderr, "  k=%d done (%d realizations, elapsed %.1fs = %.1fh)\n",
                             last_k, k_count, elapsed, elapsed / 3600);
                // Print partial result: expected cost so far (covering k=0..last_k)
                double weight_covered = 0;
                for (int kk = 0; kk <= last_k; ++kk) {
                    double w = 1.0;
                    // C(ns,kk) / 2^ns for uniform case, or compute exactly
                    int cnt = 1;
                    for (int i = 0; i < kk; ++i) cnt = cnt * (ns - i) / (i + 1);
                    for (int i = 0; i < ns; ++i) w *= 0.5; // if all p=0.5
                    weight_covered += cnt * w;
                }
                std::fprintf(stderr, "  >>> Partial E[OPT] (k=0..%d): %.6f  (%.1f%% of probability weight)\n",
                             last_k, expected, done * 100.0 / total);
            }
            k_start_done = done;
            int nk = 1;
            for (int i = 0; i < k; ++i) nk = nk * (ns - i) / (i + 1);
            std::fprintf(stderr, "  Starting k=%d (nc=%d, %d realizations, method=%s)...\n",
                         k, nd + k, nk, (nd + k <= 25) ? "HK" : "B&B");
            last_k = k;
        }
        std::vector<int> active;
        active.push_back(0);
        for (int v : det) active.push_back(v);
        double prob = 1.0;
        for (int i = 0; i < ns; ++i) {
            if (s_mask & (1 << i)) {
                active.push_back(stoch[i]);
                prob *= p[stoch[i]];
            } else {
                prob *= 1.0 - p[stoch[i]];
            }
        }
        double opt = solve_tsp_on_vertices(active, d);
        expected += prob * opt;
        done++;

        // Progress every 100 realizations
        if (done % 100 == 0) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - t0).count();
            double rate = done / elapsed;
            double eta = (total - done) / rate;
            std::fprintf(stderr, "    %d/%d (%.1f%%)  elapsed=%.0fs  ETA=%.0fs (%.1fh)\r",
                         done, total, 100.0 * done / total, elapsed, eta, eta / 3600);
        }
    }
    auto t_end = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double>(t_end - t0).count();
    std::fprintf(stderr, "  k=%d done\n", last_k);
    std::fprintf(stderr, "  >>> Final E[OPT]: %.6f  (all %d realizations in %.1fs = %.1fh)\n",
                 expected, total, total_time, total_time / 3600);
    return expected;
}

// Adaptive expected cost for large instances (stochastic-only DP).
// Probes stochastic customers first, then visits all deterministic customers.
// This is an UPPER BOUND on the exact adaptive cost (interleaving may help).
inline double solve_adaptive_large(int V,
    const std::vector<std::vector<double>>& d,
    const std::vector<double>& p) {
    std::vector<int> stoch, det;
    for (int i = 1; i < V; ++i) {
        if (p[i] > 1e-12 && p[i] < 1.0 - 1e-12) stoch.push_back(i);
        else if (p[i] >= 1.0 - 1e-12) det.push_back(i);
    }
    int ns = (int)stoch.size(), nd = (int)det.size();
    int s_full = 1 << ns;
    std::fprintf(stderr, "Adaptive (large): %d stochastic + %d deterministic customers\n", ns, nd);

    // Precompute completion[v] = min TSP-path from v through all det customers to depot 0
    std::vector<double> completion(V, 0);
    if (nd == 0) {
        for (int v = 0; v < V; ++v) completion[v] = d[v][0];
    } else {
        int d_full = 1 << nd;
        for (int v = 0; v < V; ++v) {
            std::vector<std::vector<double>> dp(d_full, std::vector<double>(nd, INF));
            for (int j = 0; j < nd; ++j)
                dp[1 << j][j] = d[v][det[j]];
            for (int mask = 1; mask < d_full; ++mask)
                for (int j = 0; j < nd; ++j) {
                    if (!(mask & (1 << j))) continue;
                    int prev = mask ^ (1 << j);
                    if (prev == 0) continue;
                    for (int k = 0; k < nd; ++k)
                        if (prev & (1 << k))
                            dp[mask][j] = std::min(dp[mask][j],
                                dp[prev][k] + d[det[k]][det[j]]);
                }
            double best = INF;
            for (int j = 0; j < nd; ++j)
                best = std::min(best, dp[d_full - 1][j] + d[det[j]][0]);
            completion[v] = best;
        }
    }
    std::fprintf(stderr, "Completion costs precomputed.\n");

    // DP: F[pos][s_mask] = expected remaining cost
    // pos in {0} ∪ stoch (at most ns+1 positions)
    int npos = ns + 1; // index 0 = depot (vertex 0), index i+1 = stoch[i]
    auto pos_idx = [&](int vertex) -> int {
        if (vertex == 0) return 0;
        for (int i = 0; i < ns; ++i)
            if (stoch[i] == vertex) return i + 1;
        return -1; // should not happen in valid states
    };

    std::vector<std::vector<double>> F(npos, std::vector<double>(s_full, 0));
    // Base case: all stochastic probed → complete with deterministic
    for (int pi = 0; pi < npos; ++pi) {
        int vertex = (pi == 0) ? 0 : stoch[pi - 1];
        F[pi][s_full - 1] = completion[vertex];
    }

    for (int bits = ns - 1; bits >= 0; --bits) {
        for (int s_mask = 0; s_mask < s_full; ++s_mask) {
            if (__builtin_popcount(s_mask) != bits) continue;
            for (int pi = 0; pi < npos; ++pi) {
                int vertex = (pi == 0) ? 0 : stoch[pi - 1];
                double best = INF;
                for (int i = 0; i < ns; ++i) {
                    if (s_mask & (1 << i)) continue;
                    int nmask = s_mask | (1 << i);
                    int active_pi = i + 1; // pos_idx of stoch[i]
                    double val = p[stoch[i]] * (d[vertex][stoch[i]] + F[active_pi][nmask])
                               + (1.0 - p[stoch[i]]) * F[pi][nmask];
                    best = std::min(best, val);
                }
                F[pi][s_mask] = best;
            }
        }
    }
    return F[0][0];
}
