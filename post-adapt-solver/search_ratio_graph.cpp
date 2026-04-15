// search_ratio_graph.cpp — Search for large ratio instances using sparse graph representation.
//
// Like search_ratio.cpp but generates random *connected sparse graphs* (edge lists) rather than
// full random distance matrices. The shortest-path metric is still used for the solvers, but the
// saved JSON uses the "edges" key so visualize.py shows the actual graph structure, making the
// examples much easier to interpret visually.
//
// Graph generation:
//   Symmetric  : random spanning tree + extra undirected edges  → connected undirected graph
//   Asymmetric : random directed Hamiltonian cycle + extra directed edges → strongly connected
//
// JSON convention (matches solver.cpp / visualize.py):
//   "n"     : total vertices INCLUDING depot (depot = vertex 0)
//   "edges" : list of {s, t, w} directed edges; if sym=true they are treated as bidirectional
//   "sym"   : true / false
//   "prob"  : length-n array; prob[0]=1.0, prob[i] for customers
//
// Usage:
//   ./search_ratio_graph [options]
//
// Options:
//   --sym              Symmetric (undirected) distances  (default: asymmetric)
//   --ratio TYPE       adapt/apost (default) | apriori/adapt | apriori/apost
//   -n N               Max total vertices including depot (default: 7)
//   -d D               Max integer edge weight (default: 20)
//   -e E               Extra edges beyond spanning structure (default: n-1)
//   -i ITERS           Number of random trials (default: 2000000)
//   -s SEED            RNG seed (default: 42)
//   --probs P          Comma-separated probs to sample for customers (default: 0.5,1.0)
//   -o FILE            Output file (default: examples/best_ratio_graph.json)
//
// Examples:
//   ./search_ratio_graph --sym --ratio adapt/apost -n 7 -d 10
//   ./search_ratio_graph --ratio apriori/apost -n 5 --probs 0.3,0.5,0.7 -i 500000

#include "tsp_solver.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <vector>
using namespace std;

// ---------- helpers ----------

struct Edge { int s, t; double w; };

static string to_json(int V, const vector<Edge>& edges, const vector<double>& p, bool sym) {
    ostringstream oss;
    oss << "{\n  \"n\": " << V << ",\n";
    oss << "  \"sym\": " << (sym ? "true" : "false") << ",\n";
    oss << "  \"edges\": [\n";
    for (int i = 0; i < (int)edges.size(); ++i) {
        const auto& e = edges[i];
        oss << "    {\"s\": " << e.s << ", \"t\": " << e.t << ", \"w\": " << (long long)e.w << "}";
        if (i + 1 < (int)edges.size()) oss << ",";
        oss << "\n";
    }
    oss << "  ],\n";
    oss << "  \"prob\": [";
    for (int i = 0; i < V; ++i) {
        if (p[i] == (int)p[i]) oss << (int)p[i];
        else oss << p[i];
        if (i + 1 < V) oss << ", ";
    }
    oss << "]\n}\n";
    return oss.str();
}

static vector<double> parse_probs(const string& s) {
    vector<double> res;
    stringstream ss(s);
    string tok;
    while (getline(ss, tok, ','))
        if (!tok.empty()) res.push_back(stod(tok));
    return res;
}

// Build distance matrix from edge list (INF where unreachable), then Floyd-Warshall.
static vector<vector<double>> build_dist(int V, const vector<Edge>& edges, bool sym) {
    vector<vector<double>> d(V, vector<double>(V, INF));
    for (int i = 0; i < V; ++i) d[i][i] = 0;
    for (const auto& e : edges) {
        if (e.w < d[e.s][e.t]) d[e.s][e.t] = e.w;
        if (sym && e.w < d[e.t][e.s]) d[e.t][e.s] = e.w;
    }
    floyd_warshall(V, d);
    return d;
}

// Check strong connectivity: all pairs reachable.
static bool is_strongly_connected(int V, const vector<vector<double>>& d) {
    for (int i = 0; i < V; ++i)
        for (int j = 0; j < V; ++j)
            if (i != j && d[i][j] >= INF / 2) return false;
    return true;
}

// Generate a random connected graph as an edge list.
//   Symmetric : spanning tree  + extra undirected edges
//   Asymmetric: directed cycle + extra directed edges   (strongly connected)
//
// Returns the edge list; caller builds the distance matrix.
static vector<Edge> gen_graph(int V, bool sym, int extra_edges,
                               int d_max, mt19937& rng) {
    uniform_int_distribution<int> rd(1, d_max);
    vector<Edge> edges;

    // Shuffle vertices to randomise structure.
    vector<int> perm(V);
    iota(perm.begin(), perm.end(), 0);
    shuffle(perm.begin(), perm.end(), rng);

    // ---- connectivity backbone ----
    if (sym) {
        // Random spanning tree: connect perm[i-1] -- perm[i]
        for (int i = 1; i < V; ++i)
            edges.push_back({perm[i-1], perm[i], (double)rd(rng)});
    } else {
        // Directed Hamiltonian cycle: perm[0]->perm[1]->...->perm[V-1]->perm[0]
        for (int i = 0; i < V; ++i)
            edges.push_back({perm[i], perm[(i+1) % V], (double)rd(rng)});
    }

    // ---- extra edges ----
    // Track existing directed edges (for sym, both directions count as one).
    set<pair<int,int>> existing;
    for (const auto& e : edges) {
        existing.insert({e.s, e.t});
        if (sym) existing.insert({e.t, e.s});
    }

    int attempts = extra_edges * 20;  // bounded rejection sampling
    int added = 0;
    uniform_int_distribution<int> rv(0, V - 1);
    while (added < extra_edges && attempts-- > 0) {
        int u = rv(rng), v = rv(rng);
        if (u == v) continue;
        if (sym && u > v) swap(u, v);  // canonical form for undirected
        if (existing.count({u, v})) continue;
        double w = rd(rng);
        edges.push_back({u, v, w});
        existing.insert({u, v});
        if (sym) existing.insert({v, u});
        ++added;
    }

    return edges;
}

// ---------- main ----------

int main(int argc, char* argv[]) {
    // --- defaults ---
    bool symmetric    = false;
    string ratio_type = "adapt/apost";
    int n_max         = 7;
    int d_max         = 20;
    int extra_edges   = -1;  // -1 = auto (set to nc after V is known)
    int iters         = 2000000;
    int seed          = 42;
    string outfile    = "examples/best_ratio_graph.json";
    vector<double> prob_choices = {0.5, 1.0};

    // --- parse args ---
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if      (arg == "--sym")                      symmetric = true;
        else if (arg == "--asym")                     symmetric = false;
        else if (arg == "--ratio"  && i+1 < argc)    ratio_type  = argv[++i];
        else if (arg == "-n"       && i+1 < argc)    n_max       = atoi(argv[++i]);
        else if (arg == "-d"       && i+1 < argc)    d_max       = atoi(argv[++i]);
        else if (arg == "-e"       && i+1 < argc)    extra_edges = atoi(argv[++i]);
        else if (arg == "-i"       && i+1 < argc)    iters       = atoi(argv[++i]);
        else if (arg == "-s"       && i+1 < argc)    seed        = atoi(argv[++i]);
        else if (arg == "-o"       && i+1 < argc)    outfile     = argv[++i];
        else if (arg == "--probs"  && i+1 < argc)    prob_choices = parse_probs(argv[++i]);
        else { cerr << "Unknown argument: " << arg << "\n"; return 1; }
    }

    if (ratio_type != "adapt/apost" && ratio_type != "apriori/adapt" && ratio_type != "apriori/apost") {
        cerr << "Unknown ratio type: " << ratio_type
             << "\n  Valid: adapt/apost, apriori/adapt, apriori/apost\n";
        return 1;
    }
    bool need_apriori = (ratio_type == "apriori/adapt" || ratio_type == "apriori/apost");
    if (need_apriori && n_max > 9) {
        cerr << "Warning: a_priori is O(n! n^2). Capping n_max at 9 for safety.\n";
        n_max = 9;
    }
    if (n_max < 3) n_max = 3;

    printf("=== stochastic TSP ratio search (graph-based) ===\n");
    printf("  ratio:      %s\n", ratio_type.c_str());
    printf("  graph:      %s\n", symmetric ? "symmetric (undirected)" : "asymmetric (directed)");
    printf("  n_max:      %d total vertices\n", n_max);
    printf("  d_max:      %d\n", d_max);
    printf("  extra edges:%s\n", extra_edges < 0 ? " auto (= nc per trial)" : to_string(extra_edges).c_str());
    printf("  probs:      {");
    for (int i = 0; i < (int)prob_choices.size(); ++i) {
        printf("%.4g", prob_choices[i]);
        if (i+1 < (int)prob_choices.size()) printf(", ");
    }
    printf("}\n");
    printf("  iterations: %d,  seed: %d\n", iters, seed);
    printf("  output:     %s\n\n", outfile.c_str());

    mt19937 rng(seed);
    uniform_int_distribution<int> rn_int(3, n_max);
    uniform_int_distribution<int> rp(0, (int)prob_choices.size() - 1);

    double best_ratio = 1.0;
    string best_json;
    double best_num = 0, best_den = 0;
    int best_V = 0;

    for (int it = 0; it < iters; ++it) {
        int V  = rn_int(rng);
        int nc = V - 1;  // customers
        int ex = (extra_edges < 0) ? nc : extra_edges;

        // Generate graph and derive shortest-path distance matrix.
        vector<Edge> edges = gen_graph(V, symmetric, ex, d_max, rng);
        vector<vector<double>> d = build_dist(V, edges, symmetric);

        // Skip disconnected instances (shouldn't happen for our backbone, but be safe).
        if (!is_strongly_connected(V, d)) continue;

        // Build probabilities.
        vector<double> p(V, 0.0);
        p[0] = 1.0;
        for (int i = 1; i < V; ++i) p[i] = prob_choices[rp(rng)];

        // Compute ratio.
        double num, den;
        if (ratio_type == "adapt/apost") {
            den = solve_a_posteriori(nc, d, p);
            if (den < 1e-12) continue;
            num = solve_adaptive(nc, d, p);
        } else if (ratio_type == "apriori/adapt") {
            den = solve_adaptive(nc, d, p);
            if (den < 1e-12) continue;
            num = solve_a_priori(nc, d, p);
        } else {
            den = solve_a_posteriori(nc, d, p);
            if (den < 1e-12) continue;
            num = solve_a_priori(nc, d, p);
        }

        double ratio = num / den;
        if (ratio > best_ratio) {
            best_ratio = ratio;
            best_num   = num;
            best_den   = den;
            best_V     = V;
            best_json  = to_json(V, edges, p, symmetric);

            printf("it=%d  V=%d (nc=%d)  edges=%d  ratio=%.6f  num=%.4f  den=%.4f\n",
                   it, V, nc, (int)edges.size(), ratio, num, den);
            fflush(stdout);
        }

        if (it % 500000 == 0 && it > 0)
            printf("  [%d/%d] best_ratio=%.6f (V=%d)\n", it, iters, best_ratio, best_V);
    }

    printf("\n=== FINAL BEST ===\n");
    printf("V=%d  ratio=%.6f  %s=%.6f  %s=%.6f\n",
           best_V, best_ratio,
           ratio_type.substr(0, ratio_type.find('/')).c_str(), best_num,
           ratio_type.substr(ratio_type.find('/')+1).c_str(), best_den);
    printf("%s\n", best_json.c_str());

    if (!best_json.empty()) {
        ofstream f(outfile);
        if (f) { f << best_json; printf("Saved to %s\n", outfile.c_str()); }
        else   { cerr << "Could not write to " << outfile << "\n"; }
    }
    return 0;
}
