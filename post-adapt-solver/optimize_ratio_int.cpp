// SA optimizer for adapt/apost ratio over integer-weight sparse symmetric graphs.
//
// Search space: sparse symmetric graph with integer edge weights + continuous probabilities.
// Perturbations: change edge weight ±1, add/remove edges, perturb probability.
// Floyd-Warshall enforces metric.
//
// Usage:
//   ./optimize_ratio_int -n 9 -w 10 -i 3000000 --restarts 20
//   ./optimize_ratio_int -n 10 -w 8 -i 2000000 --restarts 30 --nstoch 2

#include "tsp_solver.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

struct IntGraph {
    int V;
    vector<vector<int>> weight;   // 0 = no edge, >0 = edge weight
    vector<vector<double>> dist;  // metric (after Floyd-Warshall)
    vector<double> prob;
};

static void compute_metric(IntGraph& g) {
    g.dist.assign(g.V, vector<double>(g.V, INF));
    for (int i = 0; i < g.V; ++i) {
        g.dist[i][i] = 0;
        for (int j = i + 1; j < g.V; ++j) {
            if (g.weight[i][j] > 0) {
                g.dist[i][j] = g.dist[j][i] = g.weight[i][j];
            }
        }
    }
    floyd_warshall(g.V, g.dist);
}

static double eval_ratio(const IntGraph& g) {
    int nc = g.V - 1;
    double apost = solve_a_posteriori(nc, g.dist, g.prob);
    if (apost < 1e-12) return 1.0;
    double adapt = solve_adaptive(nc, g.dist, g.prob);
    return adapt / apost;
}

static string to_json(const IntGraph& g, double ratio) {
    int V = g.V;
    ostringstream oss;
    oss << "{\n  \"_comment\": \"SA-int-optimized adapt/apost = " << ratio << "\",\n";
    oss << "  \"n\": " << V << ",\n";
    oss << "  \"sym\": true,\n";
    oss << "  \"edges\": [\n";
    bool first = true;
    for (int i = 0; i < V; ++i) {
        for (int j = i + 1; j < V; ++j) {
            if (g.weight[i][j] > 0) {
                if (!first) oss << ",\n";
                oss << "    {\"s\": " << i << ", \"t\": " << j
                    << ", \"w\": " << g.weight[i][j] << "}";
                first = false;
            }
        }
    }
    oss << "\n  ],\n";
    oss << "  \"prob\": [";
    for (int i = 0; i < V; ++i) {
        if (i) oss << ", ";
        oss << g.prob[i];
    }
    oss << "]\n}\n";
    return oss.str();
}

int main(int argc, char* argv[]) {
    int V = 9;
    int w_max = 10;
    int sa_iters = 3000000;
    int seed = 42;
    int restarts = 20;
    int n_stoch = 0;  // 0 = unconstrained number of stochastic vertices
    string outfile = "examples/sa_int_optimized.json";

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "-n" && i+1 < argc) V = atoi(argv[++i]);
        else if (arg == "-w" && i+1 < argc) w_max = atoi(argv[++i]);
        else if (arg == "-i" && i+1 < argc) sa_iters = atoi(argv[++i]);
        else if (arg == "-s" && i+1 < argc) seed = atoi(argv[++i]);
        else if (arg == "--restarts" && i+1 < argc) restarts = atoi(argv[++i]);
        else if (arg == "--nstoch" && i+1 < argc) n_stoch = atoi(argv[++i]);
        else if (arg == "-o" && i+1 < argc) outfile = argv[++i];
    }

    int nc = V - 1;
    int n_pairs = V * (V - 1) / 2;

    double T0 = 0.02;
    double T_final = 1e-6;
    double cool_rate = pow(T_final / T0, 1.0 / sa_iters);

    printf("=== SA Integer-Weight Optimizer for adapt/apost ===\n");
    printf("  V=%d (nc=%d), w_max=%d\n", V, nc, w_max);
    printf("  SA iters=%d, T0=%.4f, T_final=%.2e, cool=%.8f\n",
           sa_iters, T0, T_final, cool_rate);
    printf("  restarts=%d, n_stoch=%d%s\n", restarts, n_stoch,
           n_stoch == 0 ? " (unconstrained)" : "");
    printf("  %d possible edges + %d prob vars\n\n", n_pairs, nc);

    mt19937 rng(seed);
    uniform_real_distribution<double> unif(0.0, 1.0);
    uniform_int_distribution<int> pair_idx(0, n_pairs - 1);

    // Enumerate pairs
    vector<pair<int,int>> pairs;
    for (int i = 0; i < V; ++i)
        for (int j = i + 1; j < V; ++j)
            pairs.push_back({i, j});

    double global_best_ratio = 1.0;
    IntGraph global_best;
    string global_best_json;

    for (int r = 0; r < restarts; ++r) {
        // Random initial sparse graph
        IntGraph cur;
        cur.V = V;
        cur.weight.assign(V, vector<int>(V, 0));

        // Start with a random spanning tree + some extra edges
        // Spanning tree: random permutation, connect consecutive
        vector<int> perm(V);
        iota(perm.begin(), perm.end(), 0);
        shuffle(perm.begin(), perm.end(), rng);
        for (int i = 0; i + 1 < V; ++i) {
            int a = perm[i], b = perm[i+1];
            int w = 1 + (int)(unif(rng) * w_max);
            if (w > w_max) w = w_max;
            cur.weight[a][b] = cur.weight[b][a] = w;
        }
        // Add a few random extra edges (targeting ~1.5*V edges total)
        int extra = V/2 + (int)(unif(rng) * V/2);
        for (int e = 0; e < extra; ++e) {
            int idx = pair_idx(rng);
            auto [a, b] = pairs[idx];
            if (cur.weight[a][b] == 0) {
                int w = 1 + (int)(unif(rng) * w_max);
                if (w > w_max) w = w_max;
                cur.weight[a][b] = cur.weight[b][a] = w;
            }
        }
        compute_metric(cur);

        cur.prob.assign(V, 1.0);
        if (n_stoch > 0) {
            // Randomly choose n_stoch vertices (not depot) to be stochastic
            vector<int> cands;
            for (int i = 1; i < V; ++i) cands.push_back(i);
            shuffle(cands.begin(), cands.end(), rng);
            for (int i = 0; i < min(n_stoch, (int)cands.size()); ++i)
                cur.prob[cands[i]] = 0.1 + unif(rng) * 0.9;
        } else {
            for (int i = 1; i < V; ++i)
                cur.prob[i] = 0.05 + unif(rng) * 0.95;
        }

        double cur_ratio = eval_ratio(cur);
        double best_ratio = cur_ratio;
        IntGraph best = cur;

        double T = T0;

        for (int it = 0; it < sa_iters; ++it) {
            IntGraph next = cur;

            // Choose perturbation type
            // 50% edge weight change, 15% add edge, 20% remove edge, 15% prob
            double roll = unif(rng);

            if (roll < 0.50) {
                // Change an existing edge weight by ±1 or ±2
                // Pick a random pair
                int idx = pair_idx(rng);
                auto [a, b] = pairs[idx];
                if (next.weight[a][b] > 0) {
                    int delta = (unif(rng) < 0.5) ? -1 : 1;
                    if (unif(rng) < 0.2) delta *= 2;  // occasionally ±2
                    int new_w = next.weight[a][b] + delta;
                    if (new_w >= 1 && new_w <= w_max) {
                        next.weight[a][b] = next.weight[b][a] = new_w;
                        compute_metric(next);
                    } else {
                        continue;  // skip invalid move
                    }
                } else {
                    // No edge here, try adding one
                    int w = 1 + (int)(unif(rng) * w_max);
                    if (w > w_max) w = w_max;
                    next.weight[a][b] = next.weight[b][a] = w;
                    compute_metric(next);
                }
            } else if (roll < 0.65) {
                // Add a random edge
                int idx = pair_idx(rng);
                auto [a, b] = pairs[idx];
                int w = 1 + (int)(unif(rng) * w_max);
                if (w > w_max) w = w_max;
                next.weight[a][b] = next.weight[b][a] = w;
                compute_metric(next);
            } else if (roll < 0.85) {
                // Remove a random edge (but keep graph connected)
                int idx = pair_idx(rng);
                auto [a, b] = pairs[idx];
                if (next.weight[a][b] > 0) {
                    next.weight[a][b] = next.weight[b][a] = 0;
                    compute_metric(next);
                    // Check connectivity (all dist < INF/2)
                    bool connected = true;
                    for (int i = 0; i < V && connected; ++i)
                        for (int j = i+1; j < V && connected; ++j)
                            if (next.dist[i][j] >= INF/2) connected = false;
                    if (!connected) continue;  // reject
                } else {
                    continue;  // nothing to remove
                }
            } else {
                // Perturb a probability
                int ci = 1 + (int)(unif(rng) * nc);
                if (ci >= V) ci = V - 1;

                if (n_stoch > 0 && cur.prob[ci] >= 0.999) {
                    // This vertex is deterministic and we have a stochastic constraint
                    // Don't change it
                    continue;
                }

                double scale = max(0.01, sqrt(T / T0));
                double delta = (unif(rng) - 0.5) * 2.0 * scale;
                double new_p = cur.prob[ci] + delta;
                new_p = max(0.01, min(1.0, new_p));
                next.prob[ci] = new_p;
            }

            double next_ratio = eval_ratio(next);

            double diff = next_ratio - cur_ratio;
            if (diff > 0 || unif(rng) < exp(diff / T)) {
                cur = next;
                cur_ratio = next_ratio;
                if (cur_ratio > best_ratio) {
                    best_ratio = cur_ratio;
                    best = cur;
                }
            }

            T *= cool_rate;

            if (it % 500000 == 0) {
                fprintf(stderr, "  r%d it=%d/%d: cur=%.6f best=%.6f T=%.2e\r",
                        r, it, sa_iters, cur_ratio, best_ratio, T);
            }
        }

        fprintf(stderr, "\n");
        printf("Restart %d: best=%.6f\n", r, best_ratio);

        // Print edge summary
        int edge_count = 0;
        for (auto& [a,b] : pairs) if (best.weight[a][b] > 0) edge_count++;
        printf("  edges=%d, stochastic:", edge_count);
        for (int i = 1; i < V; ++i)
            if (best.prob[i] < 0.999) printf(" v%d(%.3f)", i, best.prob[i]);
        printf("\n");
        fflush(stdout);

        if (best_ratio > global_best_ratio + 1e-9) {
            global_best_ratio = best_ratio;
            global_best = best;
            global_best_json = to_json(best, best_ratio);

            printf("  >>> NEW GLOBAL BEST: %.6f <<<\n", global_best_ratio);
            printf("  Edges: ");
            for (auto& [a,b] : pairs)
                if (best.weight[a][b] > 0)
                    printf("%d-%d(%d) ", a, b, best.weight[a][b]);
            printf("\n");
            printf("  Prob: ");
            for (int i = 0; i < V; ++i) printf("%.4f ", best.prob[i]);
            printf("\n");
            fflush(stdout);
        }
    }

    printf("\n=== FINAL BEST ===\n");
    printf("ratio = %.6f\n", global_best_ratio);

    if (!global_best_json.empty()) {
        double apost = solve_a_posteriori(nc, global_best.dist, global_best.prob);
        double adapt = solve_adaptive(nc, global_best.dist, global_best.prob);
        printf("adapt = %.6f, apost = %.6f, ratio = %.6f\n", adapt, apost, adapt/apost);

        ofstream f(outfile);
        if (f) {
            f << global_best_json;
            printf("Saved to %s\n", outfile.c_str());
        }
    }

    return 0;
}
