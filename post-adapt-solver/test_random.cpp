// Random instance tester: generates random asymmetric instances and checks for gap
#include "tsp_solver.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

using namespace std;

int main() {
    srand(42);
    int trials = 100000;
    int n = 4;
    double max_ratio = 1.0;

    for (int t = 0; t < trials; ++t) {
        int V = n + 1;
        vector<vector<double>> d(V, vector<double>(V, 0));
        vector<double> p(V, 0.0);

        for (int i = 0; i < V; ++i)
            for (int j = 0; j < V; ++j)
                if (i != j) d[i][j] = 1 + rand() % 100;

        for (int i = 1; i <= n; ++i)
            p[i] = 0.1 + 0.8 * (rand() % 100) / 100.0;

        // Compute shortest-path distances (input is a metric generator)
        floyd_warshall(V, d);

        double ap = solve_a_posteriori(n, d, p);
        double ad = solve_adaptive(n, d, p);

        double ratio = ad / ap;
        if (ratio > max_ratio + 1e-9) {
            max_ratio = ratio;
            printf("Trial %d: ap=%.4f ad=%.4f ratio=%.6f\n", t, ap, ad, ratio);
            printf("  d = ");
            for (int i = 0; i < V; ++i) {
                printf("[");
                for (int j = 0; j < V; ++j) printf("%.0f%s", d[i][j], j<V-1?",":"");
                printf("] ");
            }
            printf("\n  p = ");
            for (int i = 0; i <= n; ++i) printf("%.2f ", p[i]);
            printf("\n");
        }
    }
    printf("Max ratio found: %.6f over %d trials with n=%d\n", max_ratio, trials, n);
    return 0;
}
