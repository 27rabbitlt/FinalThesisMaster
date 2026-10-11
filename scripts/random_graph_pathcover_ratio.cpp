// Exact finite-size stress test for the sparse-{1,2}-metric limit.
//
// For G(n,c/n), Held--Karp computes the minimum number rho_n(c)*n of
// nonedges in a Hamiltonian cycle.  Combining independent estimates at c
// and alpha*c gives the rare-client prediction for rootless compression.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

static int completion_defect(const std::vector<std::vector<unsigned char>>& edge) {
  const int n = static_cast<int>(edge.size());
  const int states = 1 << n;
  const unsigned char infinity = 127;
  std::vector<unsigned char> dp(static_cast<size_t>(states) * n, infinity);
  dp[n] = 0;  // mask {0}, last 0
  for (int mask = 1; mask < states; ++mask) {
    if (!(mask & 1)) continue;
    for (int last = 0; last < n; ++last) {
      const auto value = dp[static_cast<size_t>(mask) * n + last];
      if (value == infinity) continue;
      int remaining = (states - 1) ^ mask;
      while (remaining) {
        const int bit = remaining & -remaining;
        const int next = __builtin_ctz(static_cast<unsigned>(bit));
        auto& target = dp[static_cast<size_t>(mask | bit) * n + next];
        target = std::min<unsigned char>(target, value + !edge[last][next]);
        remaining ^= bit;
      }
    }
  }
  int answer = n;
  const int full = states - 1;
  for (int last = 1; last < n; ++last) {
    answer = std::min(answer,
                      static_cast<int>(dp[static_cast<size_t>(full) * n + last])
                          + !edge[last][0]);
  }
  return answer;
}

static double estimate(int n, double c, int trials, std::mt19937_64& rng) {
  std::bernoulli_distribution present(std::min(1.0, c / n));
  long long total = 0;
  for (int trial = 0; trial < trials; ++trial) {
    std::vector<std::vector<unsigned char>> edge(n, std::vector<unsigned char>(n));
    for (int u = 0; u < n; ++u) {
      for (int v = 0; v < u; ++v) {
        edge[u][v] = edge[v][u] = present(rng);
      }
    }
    total += completion_defect(edge);
  }
  return static_cast<double>(total) / (trials * n);
}

int main(int argc, char** argv) {
  const int n = argc > 1 ? std::stoi(argv[1]) : 15;
  const int trials = argc > 2 ? std::stoi(argv[2]) : 200;
  const uint64_t seed = argc > 3 ? std::stoull(argv[3]) : 20260907;
  std::mt19937_64 rng(seed);
  const double alpha = 1.0 - 1.0 / std::exp(1.0);
  double best = 0.0;
  for (double c = 0.25; c <= 10.000001; c += 0.25) {
    const double rho = estimate(n, c, trials, rng);
    const double rho_alpha = estimate(n, alpha * c, trials, rng);
    const double factor =
        (alpha * (1.0 + rho_alpha) + 2.0 * (1.0 + std::exp(-alpha * c)))
        / (1.0 + rho);
    best = std::max(best, factor);
    std::cout << std::fixed << std::setprecision(8) << "c=" << c
              << " rho=" << rho << " rho_alpha=" << rho_alpha
              << " factor=" << factor << '\n';
  }
  std::cout << "best=" << std::fixed << std::setprecision(8) << best << '\n';
}
