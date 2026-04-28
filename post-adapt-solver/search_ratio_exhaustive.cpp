// search_ratio_exhaustive.cpp — Exhaustive search for max adapt/apost ratio.
//
// Enumerates ALL directed graphs on V vertices with edge weight 1,
// filters for strong connectivity, then tries all probability combos
// from a configurable set. Reports the maximum ratio found.
//
// Usage:
//   ./search_ratio_exhaustive [options]
//
// Options:
//   -n N              Total vertices including depot (default: 5)
//   --ratio TYPE      adapt/apost (default) | apriori/adapt | apriori/apost
//   --probs P         Comma-separated prob values for customers (default: 0.5,1.0)
//   -o FILE           Output file (default: examples/best_ratio_exhaustive.json)
//
// For V=5 asymmetric: 2^20 graphs × prob combos. Fast with weight-1 edges.

#include "tsp_solver.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>
using namespace std;

struct Edge { int s, t; };

// All possible directed edges (i != j) for V vertices.
static vector<Edge> all_possible_edges(int V) {
    vector<Edge> edges;
    for (int i = 0; i < V; ++i)
        for (int j = 0; j < V; ++j)
            if (i != j) edges.push_back({i, j});
    return edges;
}

// Build distance matrix from edge subset (all weight 1), run Floyd-Warshall.
static vector<vector<double>> build_dist(int V, const vector<Edge>& all_edges, long long edge_mask) {
    vector<vector<double>> d(V, vector<double>(V, INF));
    for (int i = 0; i < V; ++i) d[i][i] = 0;
    for (int e = 0; e < (int)all_edges.size(); ++e) {
        if (edge_mask & (1LL << e)) {
            d[all_edges[e].s][all_edges[e].t] = 1.0;
        }
    }
    floyd_warshall(V, d);
    return d;
}

static bool is_strongly_connected(int V, const vector<vector<double>>& d) {
    for (int i = 0; i < V; ++i)
        for (int j = 0; j < V; ++j)
            if (i != j && d[i][j] >= INF / 2) return false;
    return true;
}

static string to_json(int V, const vector<Edge>& all_edges, long long edge_mask,
                       const vector<double>& p) {
    ostringstream oss;
    oss << "{\n  \"n\": " << V << ",\n";
    oss << "  \"edges\": [\n";
    bool first = true;
    for (int e = 0; e < (int)all_edges.size(); ++e) {
        if (!(edge_mask & (1LL << e))) continue;
        if (!first) oss << ",\n";
        oss << "    {\"s\": " << all_edges[e].s << ", \"t\": " << all_edges[e].t << ", \"w\": 1}";
        first = false;
    }
    oss << "\n  ],\n";
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

int main(int argc, char* argv[]) {
    int V = 5;
    string ratio_type = "adapt/apost";
    vector<double> prob_choices = {0.5, 1.0};
    string outfile = "examples/best_ratio_exhaustive.json";

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if      (arg == "-n"      && i+1 < argc) V          = atoi(argv[++i]);
        else if (arg == "--ratio"  && i+1 < argc) ratio_type  = argv[++i];
        else if (arg == "--probs"  && i+1 < argc) prob_choices = parse_probs(argv[++i]);
        else if (arg == "-o"       && i+1 < argc) outfile     = argv[++i];
        else { cerr << "Unknown argument: " << arg << "\n"; return 1; }
    }

    if (ratio_type != "adapt/apost" && ratio_type != "apriori/adapt" && ratio_type != "apriori/apost") {
        cerr << "Unknown ratio type: " << ratio_type << "\n";
        return 1;
    }
    bool need_apriori = (ratio_type == "apriori/adapt" || ratio_type == "apriori/apost");
    int nc = V - 1;

    vector<Edge> all_edges = all_possible_edges(V);
    int E = (int)all_edges.size();  // V*(V-1)
    long long total_graphs = 1LL << E;

    // Enumerate all prob combos for nc customers.
    int npc = (int)prob_choices.size();
    int total_probs = 1;
    for (int i = 0; i < nc; ++i) total_probs *= npc;

    printf("=== Exhaustive ratio search ===\n");
    printf("  V=%d (nc=%d), directed edges=%d\n", V, nc, E);
    printf("  Total graph subsets: %lld\n", total_graphs);
    printf("  Prob choices: {");
    for (int i = 0; i < npc; ++i) {
        printf("%.4g", prob_choices[i]);
        if (i+1 < npc) printf(", ");
    }
    printf("}  -> %d combos per graph\n", total_probs);
    printf("  Ratio: %s\n", ratio_type.c_str());
    printf("  Output: %s\n\n", outfile.c_str());

    double best_ratio = 1.0;
    string best_json;
    double best_num = 0, best_den = 0;
    long long best_mask = 0;
    long long graphs_tested = 0;
    long long connected_count = 0;

    for (long long mask = 0; mask < total_graphs; ++mask) {
        if (mask % 100000 == 0 && mask > 0) {
            printf("  [%lld/%lld graphs, %lld connected] best=%.6f\n",
                   mask, total_graphs, connected_count, best_ratio);
            fflush(stdout);
        }

        auto d = build_dist(V, all_edges, mask);
        if (!is_strongly_connected(V, d)) continue;
        connected_count++;

        // Try all probability combos.
        for (int pc = 0; pc < total_probs; ++pc) {
            vector<double> p(V);
            p[0] = 1.0;
            int tmp = pc;
            for (int i = 0; i < nc; ++i) {
                p[i + 1] = prob_choices[tmp % npc];
                tmp /= npc;
            }

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
                best_num = num;
                best_den = den;
                best_mask = mask;
                best_json = to_json(V, all_edges, mask, p);
                printf("  NEW BEST: ratio=%.6f  num=%.4f  den=%.4f  edges=%d  probs=[",
                       ratio, num, den, __builtin_popcountll(mask));
                for (int i = 1; i < V; ++i) {
                    printf("%.2f", p[i]);
                    if (i+1 < V) printf(",");
                }
                printf("]\n");
                fflush(stdout);
            }
        }
        graphs_tested++;
    }

    printf("\n=== FINAL RESULT ===\n");
    printf("Tested %lld connected graphs (out of %lld total)\n", connected_count, total_graphs);
    printf("Best ratio (%.6f): %s=%.6f / %s=%.6f\n",
           best_ratio,
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
