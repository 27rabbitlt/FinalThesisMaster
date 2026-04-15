// search_ratio.cpp — General search for stochastic TSP ratio examples.
//
// Randomly samples instances and maximises a chosen ratio between
// a_posteriori, adaptive, and a_priori expected costs.
//
// JSON convention (matches solver.cpp / visualize.py):
//   "n"    : total number of vertices INCLUDING depot (depot = vertex 0)
//   "dist" : n×n distance matrix (or "edges" list)
//   "prob" : length-n array; prob[0] = 1.0 (depot always present), prob[i] for customers
//
// Usage:
//   ./search_ratio [options]
//
// Options:
//   --sym              Symmetric distances (default: asymmetric)
//   --ratio TYPE       Ratio to maximise — one of:
//                        adapt/apost    (default)  adaptive / a_posteriori
//                        apriori/adapt             a_priori / adaptive
//                        apriori/apost             a_priori / a_posteriori
//   -n N               Max total vertices including depot (default: 7)
//   -d D               Max integer edge distance (default: 20)
//   -i ITERS           Number of random trials (default: 2000000)
//   -s SEED            Random seed (default: 42)
//   --probs P          Comma-separated probs to sample for customers (default: 0.5,1.0)
//   -o FILE            Save best instance to FILE (default: examples/best_ratio_found.json)
//
// Examples:
//   ./search_ratio --sym --ratio adapt/apost -n 7 -d 10 -o examples/sym_gap.json
//   ./search_ratio --ratio apriori/apost -n 5 --probs 0.3,0.5,0.7 -i 500000

#include "tsp_solver.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>
using namespace std;

// ---------- helpers ----------

static string to_json(int V, const vector<vector<double>>& d, const vector<double>& p) {
    // V = total vertices, p[0] = 1.0 (depot)
    ostringstream oss;
    oss << "{\n  \"n\": " << V << ",\n";
    oss << "  \"dist\": [\n";
    for (int i = 0; i < V; ++i) {
        oss << "    [";
        for (int j = 0; j < V; ++j) {
            oss << (long long)d[i][j];
            if (j + 1 < V) oss << ", ";
        }
        oss << "]";
        if (i + 1 < V) oss << ",";
        oss << "\n";
    }
    oss << "  ],\n";
    oss << "  \"prob\": [";
    for (int i = 0; i < V; ++i) {
        // Print as integer if it is one, otherwise as float
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
    while (getline(ss, tok, ',')) {
        if (!tok.empty()) res.push_back(stod(tok));
    }
    return res;
}

// ---------- main ----------

int main(int argc, char* argv[]) {
    // --- defaults ---
    bool symmetric   = false;
    string ratio_type = "adapt/apost";  // adapt/apost | apriori/adapt | apriori/apost
    int n_max        = 7;               // max total vertices (including depot)
    int d_max        = 20;
    int iters        = 2000000;
    int seed         = 42;
    string outfile   = "examples/best_ratio_found.json";
    vector<double> prob_choices = {0.5, 1.0};

    // --- parse args ---
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--sym")                      symmetric = true;
        else if (arg == "--asym")                symmetric = false;
        else if (arg == "--ratio" && i+1 < argc) ratio_type = argv[++i];
        else if (arg == "-n"     && i+1 < argc)  n_max  = atoi(argv[++i]);
        else if (arg == "-d"     && i+1 < argc)  d_max  = atoi(argv[++i]);
        else if (arg == "-i"     && i+1 < argc)  iters  = atoi(argv[++i]);
        else if (arg == "-s"     && i+1 < argc)  seed   = atoi(argv[++i]);
        else if (arg == "-o"     && i+1 < argc)  outfile = argv[++i];
        else if (arg == "--probs" && i+1 < argc) prob_choices = parse_probs(argv[++i]);
        else { cerr << "Unknown argument: " << arg << "\n"; return 1; }
    }

    // Validate ratio type
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

    // --- print config ---
    printf("=== stochastic TSP ratio search ===\n");
    printf("  ratio:      %s\n", ratio_type.c_str());
    printf("  distances:  %s, max %d\n", symmetric ? "symmetric" : "asymmetric", d_max);
    printf("  n_max:      %d total vertices\n", n_max);
    printf("  probs:      {");
    for (int i = 0; i < (int)prob_choices.size(); ++i) {
        printf("%.4g", prob_choices[i]);
        if (i+1 < (int)prob_choices.size()) printf(", ");
    }
    printf("}\n");
    printf("  iterations: %d,  seed: %d\n", iters, seed);
    printf("  output:     %s\n\n", outfile.c_str());

    mt19937 rng(seed);
    // n_max >= 3 total vertices (depot + 2 customers minimum)
    if (n_max < 3) n_max = 3;
    uniform_int_distribution<int> rn_int(3, n_max);  // total vertices
    uniform_int_distribution<int> rd(1, d_max);
    uniform_int_distribution<int> rp(0, (int)prob_choices.size() - 1);

    double best_ratio = 1.0;
    string best_json;
    double best_num = 0, best_den = 0;
    int best_V = 0;

    for (int it = 0; it < iters; ++it) {
        int V = rn_int(rng);            // total vertices including depot
        int nc = V - 1;                 // number of customers (1..V-1)

        // Build distance matrix
        vector<vector<double>> d(V, vector<double>(V, INF));
        for (int i = 0; i < V; ++i) d[i][i] = 0;
        if (symmetric) {
            for (int i = 0; i < V; ++i)
                for (int j = i+1; j < V; ++j)
                    d[i][j] = d[j][i] = rd(rng);
        } else {
            for (int i = 0; i < V; ++i)
                for (int j = 0; j < V; ++j)
                    if (i != j) d[i][j] = rd(rng);
        }
        floyd_warshall(V, d);

        // Build probabilities: depot = 1.0, customers sampled from prob_choices
        vector<double> p(V, 0.0);
        p[0] = 1.0;
        for (int i = 1; i < V; ++i)
            p[i] = prob_choices[rp(rng)];

        // Compute requested ratio
        double num, den;
        if (ratio_type == "adapt/apost") {
            den = solve_a_posteriori(nc, d, p);
            if (den < 1e-12) continue;
            num = solve_adaptive(nc, d, p);
        } else if (ratio_type == "apriori/adapt") {
            den = solve_adaptive(nc, d, p);
            if (den < 1e-12) continue;
            num = solve_a_priori(nc, d, p);
        } else {  // apriori/apost
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
            best_json  = to_json(V, d, p);

            printf("it=%d  V=%d (nc=%d)  ratio=%.6f  num=%.4f  den=%.4f\n",
                   it, V, nc, ratio, num, den);
            if (ratio > 1.06)
                printf("  >>> ratio > 1.06 reached <<<\n");
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
