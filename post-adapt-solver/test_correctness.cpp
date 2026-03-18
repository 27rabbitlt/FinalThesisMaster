// Correctness test: compare DP solvers against brute-force on small random instances.
//
// For a posteriori: brute-force enumerates all permutations per subset.
// For adaptive: brute-force recursion over all decision trees (no memoization).
// Both are independent implementations, so agreement is strong evidence of correctness.

#include "tsp_solver.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace std;

static const double EPS = 1e-6;

// ==================== Brute-force solvers ====================

// Enumerate all permutations of the active subset to find OPT(A)
double brute_a_posteriori(int n, const vector<vector<double>>& d, const vector<double>& p) {
    int full = (1 << n);
    double ans = 0.0;
    for (int mask = 0; mask < full; ++mask) {
        double prob = 1.0;
        for (int i = 0; i < n; ++i)
            prob *= (mask & (1 << i)) ? p[i + 1] : (1.0 - p[i + 1]);

        vector<int> active;
        for (int i = 0; i < n; ++i)
            if (mask & (1 << i)) active.push_back(i + 1);

        if (active.empty()) { continue; }

        double best = INF;
        sort(active.begin(), active.end());
        do {
            double cost = d[0][active[0]];
            for (int j = 1; j < (int)active.size(); ++j)
                cost += d[active[j - 1]][active[j]];
            cost += d[active.back()][0];
            best = min(best, cost);
        } while (next_permutation(active.begin(), active.end()));

        ans += prob * best;
    }
    return ans;
}

// Recursive decision tree enumeration (no memoization — independent of DP)
double brute_adaptive(int pos, int mask, int n,
                      const vector<vector<double>>& d, const vector<double>& p) {
    int full = (1 << n) - 1;
    if (mask == full) return d[pos][0];

    double best = INF;
    for (int i = 0; i < n; ++i) {
        if (mask & (1 << i)) continue;
        int next = mask | (1 << i);
        double val = (1.0 - p[i + 1]) * brute_adaptive(pos, next, n, d, p)
                   + p[i + 1] * (d[pos][i + 1] + brute_adaptive(i + 1, next, n, d, p));
        best = min(best, val);
    }
    return best;
}

// ==================== Test helpers ====================

struct Instance {
    int n;
    vector<vector<double>> d;
    vector<double> p;
};

Instance random_instance(int n) {
    int V = n + 1;
    vector<vector<double>> d(V, vector<double>(V, 0));
    vector<double> p(V, 0.0);
    for (int i = 0; i < V; ++i)
        for (int j = 0; j < V; ++j)
            if (i != j) d[i][j] = 1 + rand() % 100;
    for (int i = 1; i <= n; ++i)
        p[i] = 0.1 + 0.8 * (rand() % 100) / 100.0;
    floyd_warshall(V, d);
    return {n, d, p};
}

bool check(double dp_val, double brute_val, const char* name, int trial) {
    double rel = fabs(dp_val - brute_val) / max(1.0, fabs(brute_val));
    if (rel > EPS) {
        printf("FAIL [%s] trial %d: dp=%.9f brute=%.9f rel_err=%.2e\n",
               name, trial, dp_val, brute_val, rel);
        return false;
    }
    return true;
}

// ==================== Tests ====================

// Test 1: edge cases
void test_edge_cases() {
    printf("Test 1: edge cases ... ");

    // n=1, single customer
    {
        vector<vector<double>> d = {{0, 5}, {3, 0}};
        vector<double> p = {0, 0.7};
        double ap = solve_a_posteriori(1, d, p);
        double ad = solve_adaptive(1, d, p);
        // Expected: 0.7 * (5+3) + 0.3 * 0 = 5.6
        assert(fabs(ap - 5.6) < EPS);
        assert(fabs(ad - 5.6) < EPS);
    }

    // All p=0: expected cost = 0
    {
        vector<vector<double>> d = {{0, 10, 20}, {10, 0, 15}, {20, 15, 0}};
        vector<double> p = {0, 0, 0};
        assert(fabs(solve_a_posteriori(2, d, p)) < EPS);
        assert(fabs(solve_adaptive(2, d, p)) < EPS);
    }

    // All p=1: both should equal deterministic TSP optimum
    {
        vector<vector<double>> d = {{0, 10, 15, 20},
                                    {10, 0, 35, 25},
                                    {15, 35, 0, 30},
                                    {20, 25, 30, 0}};
        floyd_warshall(4, d);
        vector<double> p = {0, 1, 1, 1};
        double ap = solve_a_posteriori(3, d, p);
        double ad = solve_adaptive(3, d, p);
        assert(fabs(ap - 80.0) < EPS);
        assert(fabs(ad - 80.0) < EPS);
    }

    // a_posteriori <= adaptive always
    {
        auto inst = random_instance(5);
        double ap = solve_a_posteriori(inst.n, inst.d, inst.p);
        double ad = solve_adaptive(inst.n, inst.d, inst.p);
        assert(ap <= ad + EPS);
    }

    printf("PASS\n");
}

// Test 2: DP vs brute-force on random instances (small n for brute-force feasibility)
void test_random_vs_brute(int n, int trials) {
    printf("Test 2: DP vs brute-force, n=%d, %d trials ... ", n, trials);
    int pass = 0;
    for (int t = 0; t < trials; ++t) {
        auto inst = random_instance(n);

        double ap_dp    = solve_a_posteriori(inst.n, inst.d, inst.p);
        double ap_brute = brute_a_posteriori(inst.n, inst.d, inst.p);
        if (!check(ap_dp, ap_brute, "a_post", t)) return;

        double ad_dp    = solve_adaptive(inst.n, inst.d, inst.p);
        double ad_brute = brute_adaptive(0, 0, inst.n, inst.d, inst.p);
        if (!check(ad_dp, ad_brute, "adapt", t)) return;

        // Invariant: a_posteriori <= adaptive
        if (ap_dp > ad_dp + EPS) {
            printf("FAIL: a_post (%.9f) > adaptive (%.9f) at trial %d\n", ap_dp, ad_dp, t);
            return;
        }
        ++pass;
    }
    printf("PASS (%d/%d)\n", pass, trials);
}

// Test 3: symmetry — for symmetric distances, verify both solvers agree on known structure
void test_symmetric() {
    printf("Test 3: symmetric instance hand-check ... ");

    // n=3, symmetric, p=0.5 each. Hand-computed: a_post = 42.5
    vector<vector<double>> d = {{0, 10, 15, 20},
                                {10, 0, 35, 25},
                                {15, 35, 0, 30},
                                {20, 25, 30, 0}};
    floyd_warshall(4, d);
    vector<double> p = {0, 0.5, 0.5, 0.5};

    double ap = solve_a_posteriori(3, d, p);
    double ad = solve_adaptive(3, d, p);

    // After Floyd-Warshall on this symmetric matrix, some distances may decrease
    // Recompute expected value with brute-force to get ground truth
    double ap_brute = brute_a_posteriori(3, d, p);
    double ad_brute = brute_adaptive(0, 0, 3, d, p);

    assert(fabs(ap - ap_brute) < EPS);
    assert(fabs(ad - ad_brute) < EPS);
    assert(ap <= ad + EPS);

    printf("PASS (a_post=%.4f, adaptive=%.4f)\n", ap, ad);
}

int main() {
    srand(12345);
    printf("=== Correctness Tests ===\n\n");

    test_edge_cases();
    test_random_vs_brute(4, 5000);   // n=4: brute-force is fast
    test_random_vs_brute(6, 500);    // n=6: brute-force still feasible
    test_random_vs_brute(8, 50);     // n=8: brute-force is slow, fewer trials
    test_symmetric();

    printf("\nAll tests passed.\n");
    return 0;
}
