// Brute-force search for the smallest symmetric unit-weight graph
// where OPT_adapt / OPT_apost > 1.
//
// Enumerates all connected undirected graphs on n vertices (edges = weight 1),
// computes shortest-path metric via Floyd-Warshall, then solves both problems.
// Tries probability assignments: all-0.5, and (for small n) subsets stochastic.

#include "tsp_solver.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <queue>
#include <string>

// BFS connectivity check from vertex 0
static bool is_connected(int n, long long edge_mask, int num_pairs,
                          const std::vector<std::pair<int,int>>& pairs) {
    std::vector<std::vector<int>> adj(n);
    for (int b = 0; b < num_pairs; ++b) {
        if (edge_mask & (1LL << b)) {
            adj[pairs[b].first].push_back(pairs[b].second);
            adj[pairs[b].second].push_back(pairs[b].first);
        }
    }
    std::vector<bool> vis(n, false);
    std::queue<int> q;
    q.push(0); vis[0] = true;
    int cnt = 1;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int v : adj[u]) {
            if (!vis[v]) { vis[v] = true; cnt++; q.push(v); }
        }
    }
    return cnt == n;
}

struct Result {
    double ratio = 1.0;
    int n = 0;
    long long edge_mask = 0;
    int prob_mask = 0;  // which customers are stochastic
    double ap = 0, ad = 0;
    std::vector<std::vector<double>> d;
    std::vector<double> p;
};

int main() {
    Result global_best;

    int max_n = 7; // n=8 takes hours; increase if needed
    for (int n = 3; n <= max_n; ++n) {
        int nc = n - 1;

        // Enumerate all pairs (i,j) with i < j
        std::vector<std::pair<int,int>> pairs;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                pairs.push_back({i, j});
        int num_pairs = (int)pairs.size();
        long long total_graphs = 1LL << num_pairs;

        // For n <= 7: try all stochastic/deterministic assignments
        // For n = 8: only all-0.5
        bool try_all_prob = (n <= 7);

        Result best;
        best.n = n;

        long long connected_count = 0;
        long long checked = 0;

        std::fprintf(stderr, "n=%d: %lld graphs (%d edges), try_all_prob=%s\n",
                     n, total_graphs, num_pairs, try_all_prob ? "yes" : "no");

        for (long long g = 0; g < total_graphs; ++g) {
            // Progress
            if ((g & 0xFFFFF) == 0 && g > 0)
                std::fprintf(stderr, "  n=%d: %lld/%lld (%.1f%%), connected=%lld, best=%.6f\r",
                             n, g, total_graphs, 100.0*g/total_graphs,
                             connected_count, best.ratio);

            // Quick connectivity check
            if (!is_connected(n, g, num_pairs, pairs)) continue;
            connected_count++;

            // Build distance matrix
            std::vector<std::vector<double>> d(n, std::vector<double>(n, INF));
            for (int i = 0; i < n; ++i) d[i][i] = 0;
            for (int b = 0; b < num_pairs; ++b) {
                if (g & (1LL << b)) {
                    d[pairs[b].first][pairs[b].second] = 1;
                    d[pairs[b].second][pairs[b].first] = 1;
                }
            }
            floyd_warshall(n, d);

            // Try probability assignments
            int prob_limit = try_all_prob ? (1 << nc) : ((1 << nc)); // always try all-stochastic
            for (int pmask = 1; pmask < (1 << nc); ++pmask) {
                // pmask bit i set => customer i+1 is stochastic (p=0.5)
                if (!try_all_prob && pmask != (1 << nc) - 1) continue;

                // Need at least 1 stochastic customer
                if (__builtin_popcount(pmask) < 1) continue;

                std::vector<double> p(n, 1.0);
                for (int i = 0; i < nc; ++i)
                    p[i + 1] = (pmask & (1 << i)) ? 0.5 : 1.0;

                double ap = solve_a_posteriori(nc, d, p);
                double ad = solve_adaptive(nc, d, p);
                checked++;

                if (ap > 1e-12 && ad / ap > best.ratio + 1e-9) {
                    best.ratio = ad / ap;
                    best.edge_mask = g;
                    best.prob_mask = pmask;
                    best.ap = ap;
                    best.ad = ad;
                    best.d = d;
                    best.p = p;
                }
            }
        }

        std::fprintf(stderr, "\n");
        std::printf("n=%d: best ratio = %.6f  (checked %lld cases, %lld connected graphs)\n",
                    n, best.ratio, checked, connected_count);
        std::fflush(stdout);

        if (best.ratio > 1.0 + 1e-9) {
            std::printf("  adapt=%.6f  apost=%.6f\n", best.ad, best.ap);
            std::printf("  Edges: ");
            for (int b = 0; b < num_pairs; ++b)
                if (best.edge_mask & (1LL << b))
                    std::printf("(%d,%d) ", pairs[b].first, pairs[b].second);
            std::printf("\n");
            std::printf("  Probabilities: ");
            for (int i = 0; i < n; ++i)
                std::printf("p[%d]=%.1f ", i, best.p[i]);
            std::printf("\n");
            std::printf("  Distance matrix:\n");
            for (int i = 0; i < n; ++i) {
                std::printf("    ");
                for (int j = 0; j < n; ++j)
                    std::printf("%4.0f", best.d[i][j]);
                std::printf("\n");
            }

            std::fflush(stdout);
            if (best.ratio > global_best.ratio + 1e-9) {
                global_best = best;
            }
        }
    }

    // Save the best example as JSON
    if (global_best.ratio > 1.0 + 1e-9) {
        int n = global_best.n;
        std::printf("\n=== BEST OVERALL: n=%d, ratio=%.6f ===\n", n, global_best.ratio);

        // Write JSON
        std::string fname = "examples/sym_unit_gap_n" + std::to_string(n) + ".json";
        std::ofstream fout(fname);
        fout << "{\n";
        fout << "  \"_comment\": \"Smallest symmetric unit-weight graph with adapt/apost > 1"
             << ", ratio=" << global_best.ratio << "\",\n";
        fout << "  \"n\": " << n << ",\n";
        fout << "  \"sym\": true,\n";
        fout << "  \"edges\": [";
        bool first = true;
        std::vector<std::pair<int,int>> pairs;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                pairs.push_back({i, j});
        for (int b = 0; b < (int)pairs.size(); ++b) {
            if (global_best.edge_mask & (1LL << b)) {
                if (!first) fout << ", ";
                fout << "{\"s\": " << pairs[b].first
                     << ", \"t\": " << pairs[b].second << ", \"w\": 1}";
                first = false;
            }
        }
        fout << "],\n";
        fout << "  \"prob\": [";
        for (int i = 0; i < n; ++i) {
            if (i) fout << ", ";
            fout << global_best.p[i];
        }
        fout << "]\n}\n";
        fout.close();
        std::printf("Saved to %s\n", fname.c_str());
    } else {
        std::printf("\nNo graph with ratio > 1 found up to n=8.\n");
    }

    return 0;
}
