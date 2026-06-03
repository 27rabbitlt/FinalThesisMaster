// Simulated annealing optimizer for adapt/apost ratio.
//
// Optimizes over symmetric distance matrices and probabilities.
// Uses Floyd-Warshall to enforce metric (triangle inequality).
//
// Usage:
//   ./optimize_ratio_sa -n 7 -d 10 -i 3000000 --restarts 20
//   ./optimize_ratio_sa -n 8 -d 5 -i 2000000 --restarts 30

#include "tsp_solver.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

struct Instance {
    int V;
    vector<vector<double>> raw;   // raw edge weights (before Floyd-Warshall)
    vector<vector<double>> dist;  // metric (after Floyd-Warshall)
    vector<double> prob;
};

static double eval_ratio(const Instance& inst) {
    int nc = inst.V - 1;
    double apost = solve_a_posteriori(nc, inst.dist, inst.prob);
    if (apost < 1e-12) return 1.0;
    double adapt = solve_adaptive(nc, inst.dist, inst.prob);
    return adapt / apost;
}

static void apply_floyd(Instance& inst) {
    inst.dist = inst.raw;
    floyd_warshall(inst.V, inst.dist);
}

static string to_json(const Instance& inst, double ratio) {
    int V = inst.V;
    ostringstream oss;
    oss << "{\n  \"_comment\": \"SA-optimized adapt/apost = " << ratio << "\",\n";
    oss << "  \"n\": " << V << ",\n";
    oss << "  \"dist\": [\n";
    for (int i = 0; i < V; ++i) {
        oss << "    [";
        for (int j = 0; j < V; ++j) {
            oss << inst.dist[i][j];
            if (j + 1 < V) oss << ", ";
        }
        oss << "]";
        if (i + 1 < V) oss << ",";
        oss << "\n";
    }
    oss << "  ],\n";
    oss << "  \"prob\": [";
    for (int i = 0; i < V; ++i) {
        if (i) oss << ", ";
        oss << inst.prob[i];
    }
    oss << "]\n}\n";
    return oss.str();
}

int main(int argc, char* argv[]) {
    int V = 7;
    double d_max = 10.0;
    int sa_iters = 3000000;
    int seed = 42;
    int restarts = 20;
    string outfile = "examples/sa_optimized.json";

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "-n" && i+1 < argc) V = atoi(argv[++i]);
        else if (arg == "-d" && i+1 < argc) d_max = atof(argv[++i]);
        else if (arg == "-i" && i+1 < argc) sa_iters = atoi(argv[++i]);
        else if (arg == "-s" && i+1 < argc) seed = atoi(argv[++i]);
        else if (arg == "--restarts" && i+1 < argc) restarts = atoi(argv[++i]);
        else if (arg == "-o" && i+1 < argc) outfile = argv[++i];
    }

    int nc = V - 1;
    int n_pairs = V * (V - 1) / 2;

    // SA temperature schedule: T goes from T0 to T_final over sa_iters
    // T0 chosen so initial acceptance of bad moves ~50% for diff=-0.01
    // T_final chosen so only improvements accepted at the end
    double T0 = 0.02;
    double T_final = 1e-6;
    double cool_rate = pow(T_final / T0, 1.0 / sa_iters);

    printf("=== SA Optimizer for adapt/apost ===\n");
    printf("  V=%d (nc=%d), d_max=%.1f\n", V, nc, d_max);
    printf("  SA iters=%d, T0=%.4f, T_final=%.2e, cool=%.8f\n",
           sa_iters, T0, T_final, cool_rate);
    printf("  restarts=%d\n", restarts);
    printf("  %d distance vars + %d prob vars = %d total\n\n",
           n_pairs, nc, n_pairs + nc);

    mt19937 rng(seed);
    uniform_real_distribution<double> unif(0.0, 1.0);
    uniform_int_distribution<int> perturb_type(0, n_pairs + nc - 1);

    // Enumerate pairs
    vector<pair<int,int>> pairs;
    for (int i = 0; i < V; ++i)
        for (int j = i + 1; j < V; ++j)
            pairs.push_back({i, j});

    double global_best_ratio = 1.0;
    Instance global_best;
    string global_best_json;

    for (int r = 0; r < restarts; ++r) {
        // Random initial instance
        Instance cur;
        cur.V = V;
        cur.raw.assign(V, vector<double>(V, INF));
        for (int i = 0; i < V; ++i) cur.raw[i][i] = 0;
        for (auto& [a, b] : pairs) {
            double w = 0.1 + unif(rng) * (d_max - 0.1);
            cur.raw[a][b] = cur.raw[b][a] = w;
        }
        apply_floyd(cur);
        cur.prob.resize(V);
        cur.prob[0] = 1.0;
        for (int i = 1; i < V; ++i)
            cur.prob[i] = 0.05 + unif(rng) * 0.95;

        double cur_ratio = eval_ratio(cur);
        double best_ratio = cur_ratio;
        Instance best = cur;

        double T = T0;

        for (int it = 0; it < sa_iters; ++it) {
            Instance next = cur;
            int which = perturb_type(rng);

            // Perturbation scale: decreases with T but always has minimum magnitude
            double scale = max(0.01, sqrt(T / T0));  // 1.0 at start, ~0.007 at end

            if (which < n_pairs) {
                // Perturb a distance
                auto [a, b] = pairs[which];
                double delta = (unif(rng) - 0.5) * 2.0 * d_max * scale;
                double new_w = max(0.01, next.raw[a][b] + delta);
                new_w = min(new_w, d_max);
                next.raw[a][b] = next.raw[b][a] = new_w;
                apply_floyd(next);
            } else {
                // Perturb a probability
                int ci = which - n_pairs;
                double delta = (unif(rng) - 0.5) * 2.0 * scale;
                next.prob[ci + 1] = max(0.01, min(1.0, next.prob[ci + 1] + delta));
            }

            double next_ratio = eval_ratio(next);

            // SA acceptance
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
        fflush(stdout);

        if (best_ratio > global_best_ratio + 1e-9) {
            global_best_ratio = best_ratio;
            global_best = best;
            global_best_json = to_json(best, best_ratio);

            printf("  >>> NEW GLOBAL BEST: %.6f <<<\n", global_best_ratio);
            printf("  Prob: ");
            for (int i = 0; i < V; ++i) printf("%.4f ", best.prob[i]);
            printf("\n");
            printf("  Dist matrix:\n");
            for (int i = 0; i < V; ++i) {
                printf("    ");
                for (int j = 0; j < V; ++j)
                    printf("%7.3f", best.dist[i][j]);
                printf("\n");
            }
            fflush(stdout);
        }
    }

    printf("\n=== FINAL BEST ===\n");
    printf("ratio = %.6f\n", global_best_ratio);

    if (!global_best_json.empty()) {
        // Verify
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
