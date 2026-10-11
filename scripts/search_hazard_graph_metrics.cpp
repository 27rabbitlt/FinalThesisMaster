#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <tuple>
#include <vector>

// Exploratory exact-objective search for the rooted hazard-tree ratio on
// small unweighted graph metrics.  The arithmetic is floating point, but both
// optimization problems are solved exactly for each generated finite metric.

using Matrix = std::vector<std::vector<double>>;

static std::vector<double> subset_tours(const Matrix& d) {
    const int n = static_cast<int>(d.size()) - 1;
    const int root = n, z = 1 << n;
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> path(static_cast<size_t>(z) * n, inf), tour(z, 0.0);
    auto at = [&](int m, int v) -> double& { return path[static_cast<size_t>(m) * n + v]; };
    for (int v = 0; v < n; ++v) at(1 << v, v) = d[root][v];
    for (int m = 1; m < z; ++m) {
        for (int v = 0; v < n; ++v) if (m & (1 << v)) {
            const int pm = m ^ (1 << v);
            if (pm) for (int u = 0; u < n; ++u) if (pm & (1 << u))
                at(m, v) = std::min(at(m, v), at(pm, u) + d[u][v]);
            tour[m] = tour[m] == 0.0 ? at(m, v) + d[v][root]
                                      : std::min(tour[m], at(m, v) + d[v][root]);
        }
    }
    return tour;
}

static double posterior(const std::vector<double>& tours, const std::vector<double>& p) {
    const int n = static_cast<int>(p.size()), z = 1 << n;
    double ans = 0.0;
    for (int m = 0; m < z; ++m) {
        double pr = 1.0;
        for (int v = 0; v < n; ++v) pr *= (m & (1 << v)) ? p[v] : 1.0 - p[v];
        ans += pr * tours[m];
    }
    return ans;
}

static double hazard(const Matrix& d, const std::vector<double>& p) {
    const int n = static_cast<int>(p.size()), root = n, z = 1 << n;
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> q(z), inactive(z, 1.0);
    for (int m = 1; m < z; ++m) {
        int b = m & -m, v = std::countr_zero(static_cast<unsigned>(b));
        inactive[m] = inactive[m ^ b] * (1.0 - p[v]);
        q[m] = 1.0 - inactive[m];
    }
    std::vector<double> f(static_cast<size_t>(n + 1) * z, inf);
    auto at = [&](int r, int m) -> double& { return f[static_cast<size_t>(r) * z + m]; };
    for (int r = 0; r <= n; ++r) at(r, 0) = 0.0;
    for (int card = 1; card <= n; ++card) for (int m = 1; m < z; ++m) {
        if (std::popcount(static_cast<unsigned>(m)) != card) continue;
        const int anchor = m & -m, rem = m ^ anchor;
        for (int r = 0; r <= n; ++r) {
            if (r < n && (m & (1 << r))) continue;
            double best = inf;
            for (int sub = rem;; sub = (sub - 1) & rem) {
                const int branch = sub | anchor, rest = m ^ branch;
                for (int bits = branch; bits; bits &= bits - 1) {
                    const int b = bits & -bits;
                    const int u = std::countr_zero(static_cast<unsigned>(b));
                    best = std::min(best, q[branch] * d[r][u] + at(u, branch ^ b) + at(r, rest));
                }
                if (!sub) break;
            }
            at(r, m) = best;
        }
    }
    return at(root, z - 1);
}

static Matrix graph_metric(const std::vector<std::vector<int>>& a, int root0) {
    const int N = static_cast<int>(a.size());
    std::vector<int> order;
    for (int v = 0; v < N; ++v) if (v != root0) order.push_back(v);
    order.push_back(root0);
    Matrix d(N, std::vector<double>(N, N + 1));
    for (int s = 0; s < N; ++s) {
        std::queue<int> qu; std::vector<int> dist(N, N + 1);
        dist[order[s]] = 0; qu.push(order[s]);
        while (!qu.empty()) { int u = qu.front(); qu.pop();
            for (int v = 0; v < N; ++v) if (a[u][v] && dist[v] > dist[u] + 1)
                dist[v] = dist[u] + 1, qu.push(v);
        }
        for (int t = 0; t < N; ++t) d[s][t] = dist[order[t]];
    }
    return d;
}

static bool connected(const std::vector<std::vector<int>>& a) {
    const int N = static_cast<int>(a.size()); std::vector<int> seen(N); std::queue<int> q;
    seen[0] = 1; q.push(0); int k = 0;
    while (!q.empty()) { int u = q.front(); q.pop(); ++k;
        for (int v = 0; v < N; ++v) if (a[u][v] && !seen[v]) seen[v] = 1, q.push(v);
    }
    return k == N;
}

int main(int argc, char** argv) {
    const int N = argc > 1 ? std::stoi(argv[1]) : 11; // includes depot
    const int trials = argc > 2 ? std::stoi(argv[2]) : 2000;
    const unsigned seed = argc > 3 ? static_cast<unsigned>(std::stoul(argv[3])) : 20260907;
    std::mt19937 rng(seed); std::uniform_real_distribution<double> U(0.0, 1.0);
    double record = -1.0;
    for (int power = 1; power <= std::min(5, (N - 1) / 2); ++power) {
        std::vector<std::vector<int>> a(N, std::vector<int>(N));
        for (int u = 0; u < N; ++u) for (int step = 1; step <= power; ++step) {
            int v = (u + step) % N; a[u][v] = a[v][u] = 1;
        }
        Matrix d = graph_metric(a, 0); auto tours = subset_tours(d);
        for (double pv : {0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.97}) {
            std::vector<double> p(N - 1, pv);
            const double P = posterior(tours, p), H = hazard(d, p), ratio = H / P;
            if (ratio > record + 1e-12) {
                record = ratio;
                std::cout << std::setprecision(15) << "best " << ratio << " N " << N
                          << " ring-power " << power << " p " << pv << " P " << P
                          << " H " << H << '\n' << std::flush;
            }
        }
    }
    for (int it = 0; it < trials; ++it) {
        // Mixture of sparse Erdos-Renyi graphs and ring lattices with random chords.
        std::vector<std::vector<int>> a(N, std::vector<int>(N));
        if (it & 1) {
            for (int u = 0; u < N; ++u) for (int step = 1; step <= 1 + int(rng() % 3); ++step) {
                int v = (u + step) % N; a[u][v] = a[v][u] = 1;
            }
            for (int u = 0; u < N; ++u) for (int v = 0; v < u; ++v)
                if (!a[u][v] && U(rng) < 0.08) a[u][v] = a[v][u] = 1;
        } else {
            const double density = 0.12 + 0.55 * U(rng);
            for (int u = 1; u < N; ++u) { int v = rng() % u; a[u][v] = a[v][u] = 1; }
            for (int u = 0; u < N; ++u) for (int v = 0; v < u; ++v)
                if (!a[u][v] && U(rng) < density) a[u][v] = a[v][u] = 1;
        }
        if (!connected(a)) continue;
        const int root = rng() % N;
        Matrix d = graph_metric(a, root);
        auto tours = subset_tours(d);
        for (double pv : {0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.97}) {
            std::vector<double> p(N - 1, pv);
            const double P = posterior(tours, p), H = hazard(d, p), ratio = H / P;
            if (ratio > record + 1e-12) {
                record = ratio;
                std::cout << std::setprecision(15) << "best " << ratio << " N " << N
                          << " iter " << it << " root " << root << " p " << pv
                          << " P " << P << " H " << H << " edges";
                for (int u = 0; u < N; ++u) for (int v = 0; v < u; ++v) if (a[u][v])
                    std::cout << ' ' << v << '-' << u;
                std::cout << '\n' << std::flush;
            }
        }
    }
}
