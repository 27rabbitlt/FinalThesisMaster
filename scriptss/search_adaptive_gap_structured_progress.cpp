#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using std::size_t;

static constexpr double INF = std::numeric_limits<double>::infinity();
static constexpr double EPS = 1e-12;

struct ProgressReporter {
    using Clock = std::chrono::steady_clock;

    size_t total_work;
    double min_seconds_between_updates;
    double min_fraction_between_updates;
    Clock::time_point start;
    Clock::time_point last_print;
    double last_fraction_printed = -1.0;

    ProgressReporter(size_t total_work_, double min_seconds = 5.0, double min_fraction = 0.01)
        : total_work(total_work_),
          min_seconds_between_updates(min_seconds),
          min_fraction_between_updates(min_fraction),
          start(Clock::now()),
          last_print(start) {}

    static double seconds_since(const Clock::time_point &a, const Clock::time_point &b) {
        return std::chrono::duration_cast<std::chrono::duration<double>>(b - a).count();
    }

    static std::string format_seconds(double seconds) {
        if (!std::isfinite(seconds)) return "?";
        long long total = static_cast<long long>(std::llround(seconds));
        long long h = total / 3600;
        long long m = (total % 3600) / 60;
        long long s = total % 60;
        std::ostringstream oss;
        if (h > 0) oss << h << "h";
        if (h > 0 || m > 0) oss << m << "m";
        oss << s << "s";
        return oss.str();
    }

    static double estimate_eta(double elapsed, size_t done_work) {
        if (done_work == 0) return INF;
        return elapsed * (static_cast<double>(1.0) / static_cast<double>(done_work) * 1.0 - 1.0) +
               elapsed * (static_cast<double>(0));
    }

    void update(size_t done_work, double best_ratio, const std::string &best_desc = "", bool force = false) {
        if (done_work > total_work) done_work = total_work;
        const auto now = Clock::now();
        const double elapsed = seconds_since(start, now);
        const double since_last = seconds_since(last_print, now);
        const double fraction_done = (total_work == 0) ? 1.0 : static_cast<double>(done_work) / static_cast<double>(total_work);
        const double fraction_advance = fraction_done - last_fraction_printed;

        if (!force && since_last < min_seconds_between_updates && fraction_advance < min_fraction_between_updates) {
            return;
        }

        double eta = INF;
        if (done_work > 0 && done_work < total_work) {
            eta = elapsed * (static_cast<double>(total_work - done_work) / static_cast<double>(done_work));
        } else if (done_work >= total_work) {
            eta = 0.0;
        }

        std::ostringstream oss;
        oss << "[progress] "
            << done_work << "/" << total_work
            << " (" << std::fixed << std::setprecision(1) << (100.0 * fraction_done) << "%)"
            << ", elapsed=" << format_seconds(elapsed)
            << ", eta=" << format_seconds(eta)
            << ", best_ratio=" << std::setprecision(6) << best_ratio;
        if (!best_desc.empty()) oss << ", best=" << best_desc;
        std::cerr << oss.str() << '\n';

        last_print = now;
        last_fraction_printed = fraction_done;
    }

    void improvement(size_t done_work, double best_ratio, const std::string &best_desc = "") {
        std::ostringstream oss;
        oss << "[best] step=" << done_work
            << ", best_ratio=" << std::fixed << std::setprecision(6) << best_ratio;
        if (!best_desc.empty()) oss << ", " << best_desc;
        std::cerr << oss.str() << '\n';
        update(done_work, best_ratio, best_desc, true);
    }
};

struct Instance {
    int n = 0;
    int root = 0;
    std::string family = "random";
    std::vector<std::vector<double>> dist;
    std::vector<double> probs;
    std::vector<double> target_probs;
    std::vector<int> roles;
};

struct Summary {
    double posteriori = 0.0;
    double adaptive = 0.0;
    double ratio = 1.0;
    std::vector<int> policy; // size n * 2^(n-1)
};

static inline int popcount_u64(uint64_t x) {
    return __builtin_popcountll(x);
}

static inline int ctz_u64(uint64_t x) {
    return __builtin_ctzll(x);
}

std::vector<std::vector<double>> floyd_warshall(std::vector<std::vector<double>> d) {
    int n = static_cast<int>(d.size());
    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            const double dik = d[i][k];
            for (int j = 0; j < n; ++j) {
                double cand = dik + d[k][j];
                if (cand < d[i][j]) d[i][j] = cand;
            }
        }
    }
    for (int i = 0; i < n; ++i) d[i][i] = 0.0;
    return d;
}

std::vector<std::vector<double>> complete_metric_from_raw(const std::vector<std::vector<double>> &raw) {
    return floyd_warshall(raw);
}

std::vector<std::vector<double>> random_shortest_path_metric(int n, std::mt19937_64 &rng, int lo = 1, int hi = 100) {
    std::uniform_int_distribution<int> dist(lo, hi);
    std::vector<std::vector<double>> raw(n, std::vector<double>(n, INF));
    for (int i = 0; i < n; ++i) {
        raw[i][i] = 0.0;
        for (int j = 0; j < n; ++j) {
            if (i != j) raw[i][j] = static_cast<double>(dist(rng));
        }
    }
    return floyd_warshall(raw);
}

std::vector<double> random_probability_vector(int n, std::mt19937_64 &rng, const std::vector<double> &palette) {
    std::uniform_int_distribution<int> pick(0, static_cast<int>(palette.size()) - 1);
    std::vector<double> probs(n, 1.0);
    probs[0] = 1.0;
    for (int i = 1; i < n; ++i) probs[i] = palette[pick(rng)];
    return probs;
}

Instance make_random_instance(int n, std::mt19937_64 &rng) {
    Instance inst;
    inst.n = n;
    inst.root = 0;
    inst.family = "random";
    inst.roles.assign(n, 0);
    inst.dist = random_shortest_path_metric(n, rng);
    inst.target_probs.assign(n, 0.5);
    inst.target_probs[0] = 1.0;
    return inst;
}

Instance make_branch_gadget(int n, std::mt19937_64 &rng) {
    Instance inst;
    inst.n = n;
    inst.root = 0;
    inst.family = "branch";
    inst.roles.assign(n, 0);
    inst.target_probs.assign(n, 0.0);

    int b = std::max(2, static_cast<int>(std::sqrt(std::max(1, n - 1))));
    int branch_len = std::max(1, (n - 1 + b - 1) / b);
    std::vector<std::vector<double>> raw(n, std::vector<double>(n, INF));
    for (int i = 0; i < n; ++i) raw[i][i] = 0.0;

    std::vector<std::vector<int>> branches(b);
    int cur = 1;
    for (int bi = 0; bi < b && cur < n; ++bi) {
        for (int t = 0; t < branch_len && cur < n; ++t) {
            branches[bi].push_back(cur);
            inst.roles[cur] = bi + 1;
            ++cur;
        }
    }

    std::uniform_int_distribution<int> jitter(0, 3);
    for (int bi = 0; bi < b; ++bi) {
        if (branches[bi].empty()) continue;
        int first = branches[bi][0];
        raw[0][first] = 2.0 + jitter(rng);
        raw[first][0] = 8.0 + jitter(rng);
        inst.target_probs[first] = 0.75;
        for (size_t t = 0; t + 1 < branches[bi].size(); ++t) {
            int u = branches[bi][t], v = branches[bi][t + 1];
            raw[u][v] = 1.0;
            raw[v][u] = 7.0;
            inst.target_probs[v] = 0.35 + 0.1 * ((t + bi) % 3);
        }
    }

    for (int i = 1; i < n; ++i) {
        for (int j = 1; j < n; ++j) {
            if (i == j) continue;
            raw[i][j] = std::min(raw[i][j], 24.0 + static_cast<double>(jitter(rng)));
        }
    }

    inst.dist = floyd_warshall(raw);
    inst.target_probs[0] = 1.0;
    for (int i = 1; i < n; ++i) if (inst.target_probs[i] == 0.0) inst.target_probs[i] = 0.4;
    return inst;
}

Instance make_layered_gadget(int n, std::mt19937_64 &rng) {
    Instance inst;
    inst.n = n;
    inst.root = 0;
    inst.family = "layered";
    inst.roles.assign(n, 0);
    inst.target_probs.assign(n, 0.0);

    int layers = std::max(2, static_cast<int>(std::sqrt(std::max(1, n - 1))));
    std::vector<std::vector<int>> layer_vertices(layers);
    int v = 1;
    for (int L = 0; L < layers && v < n; ++L) {
        int cnt = std::max(1, (n - 1 - v + 1 + (layers - L) - 1) / (layers - L));
        for (int t = 0; t < cnt && v < n; ++t, ++v) {
            layer_vertices[L].push_back(v);
            inst.roles[v] = L + 1;
        }
    }

    std::uniform_int_distribution<int> jitter(0, 2);
    std::vector<std::vector<double>> raw(n, std::vector<double>(n, INF));
    for (int i = 0; i < n; ++i) raw[i][i] = 0.0;

    for (int L = 0; L < layers; ++L) {
        for (int x : layer_vertices[L]) {
            raw[0][x] = 4.0 + 3.0 * L + jitter(rng);
            raw[x][0] = 6.0 + 2.0 * L + jitter(rng);
            inst.target_probs[x] = std::max(0.15, 0.85 - 0.18 * L);
            for (int M = 0; M < layers; ++M) {
                for (int y : layer_vertices[M]) {
                    if (x == y) continue;
                    if (M == L) {
                        raw[x][y] = std::min(raw[x][y], 3.0 + jitter(rng));
                    } else if (M == L + 1) {
                        raw[x][y] = std::min(raw[x][y], 2.0 + jitter(rng));
                    } else if (M > L + 1) {
                        raw[x][y] = std::min(raw[x][y], 5.0 + 2.0 * (M - L) + jitter(rng));
                    } else {
                        raw[x][y] = std::min(raw[x][y], 10.0 + 4.0 * (L - M) + jitter(rng));
                    }
                }
            }
        }
    }

    inst.dist = floyd_warshall(raw);
    inst.target_probs[0] = 1.0;
    return inst;
}

Instance make_gateway_gadget(int n, std::mt19937_64 &rng) {
    Instance inst;
    inst.n = n;
    inst.root = 0;
    inst.family = "gateway";
    inst.roles.assign(n, 0);
    inst.target_probs.assign(n, 0.0);

    std::vector<int> left, right;
    for (int i = 1; i < n; ++i) {
        if (i % 2 == 1) left.push_back(i);
        else right.push_back(i);
    }
    int left_gate = left.empty() ? -1 : left.front();
    int right_gate = right.empty() ? -1 : right.front();

    std::uniform_int_distribution<int> jitter(0, 3);
    std::vector<std::vector<double>> raw(n, std::vector<double>(n, INF));
    for (int i = 0; i < n; ++i) raw[i][i] = 0.0;

    for (int x : left) {
        inst.roles[x] = 1;
        raw[0][x] = (x == left_gate ? 3.0 : 7.0) + jitter(rng);
        raw[x][0] = 7.0 + jitter(rng);
        inst.target_probs[x] = (x == left_gate ? 0.90 : 0.70);
        for (int y : left) if (x != y) raw[x][y] = std::min(raw[x][y], 2.0 + jitter(rng));
    }
    for (int x : right) {
        inst.roles[x] = 2;
        raw[0][x] = (x == right_gate ? 4.0 : 10.0) + jitter(rng);
        raw[x][0] = 10.0 + jitter(rng);
        inst.target_probs[x] = (x == right_gate ? 0.55 : 0.25);
        for (int y : right) if (x != y) raw[x][y] = std::min(raw[x][y], 2.0 + jitter(rng));
    }
    if (left_gate != -1 && right_gate != -1) {
        raw[left_gate][right_gate] = 3.0 + jitter(rng);
        raw[right_gate][left_gate] = 16.0 + jitter(rng);
    }
    for (int x : left) {
        for (int y : right) {
            raw[x][y] = std::min(raw[x][y], 24.0 + jitter(rng));
            raw[y][x] = std::min(raw[y][x], 24.0 + jitter(rng));
        }
    }

    inst.dist = floyd_warshall(raw);
    inst.target_probs[0] = 1.0;
    return inst;
}

Instance make_gadget(const std::string &family, int n, std::mt19937_64 &rng) {
    if (family == "random") return make_random_instance(n, rng);
    if (family == "branch") return make_branch_gadget(n, rng);
    if (family == "layered") return make_layered_gadget(n, rng);
    if (family == "gateway") return make_gateway_gadget(n, rng);
    throw std::runtime_error("unknown gadget family: " + family);
}

std::vector<double> sample_probs_from_targets(const Instance &templ, std::mt19937_64 &rng) {
    std::vector<double> probs = templ.target_probs;
    std::vector<double> palette{0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95};
    std::uniform_int_distribution<int> pick(0, static_cast<int>(palette.size()) - 1);
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    probs[0] = 1.0;
    for (int i = 1; i < templ.n; ++i) {
        double base = templ.target_probs[i] > 0.0 ? templ.target_probs[i] : palette[pick(rng)];
        if (coin(rng) < 0.35) {
            probs[i] = palette[pick(rng)];
        } else {
            double delta = (coin(rng) - 0.5) * 0.20;
            probs[i] = std::clamp(base + delta, 0.01, 0.99);
        }
    }
    return probs;
}

std::vector<double> all_subset_tsp_costs(const Instance &inst) {
    const int n = inst.n;
    const int m = n - 1;
    if (m >= 63) throw std::runtime_error("too many non-root vertices for uint64_t masks");
    const uint64_t states = 1ULL << m;
    const uint64_t full_mask = states - 1ULL;
    std::vector<double> hk(states * static_cast<uint64_t>(n), INF);

    auto idx = [n](uint64_t mask, int j) -> uint64_t {
        return mask * static_cast<uint64_t>(n) + static_cast<uint64_t>(j);
    };

    for (int j = 1; j < n; ++j) {
        uint64_t bit = 1ULL << (j - 1);
        hk[idx(bit, j)] = inst.dist[inst.root][j];
    }

    for (uint64_t mask = 1; mask < states; ++mask) {
        uint64_t rem_j = mask;
        while (rem_j) {
            uint64_t bitj = rem_j & -rem_j;
            int j = ctz_u64(bitj) + 1;
            double cur = hk[idx(mask, j)];
            if (cur < INF / 2.0) {
                uint64_t remaining = full_mask ^ mask;
                uint64_t add = remaining;
                while (add) {
                    uint64_t bitk = add & -add;
                    int k = ctz_u64(bitk) + 1;
                    uint64_t new_mask = mask | bitk;
                    double cand = cur + inst.dist[j][k];
                    double &slot = hk[idx(new_mask, k)];
                    if (cand < slot) slot = cand;
                    add ^= bitk;
                }
            }
            rem_j ^= bitj;
        }
    }

    std::vector<double> cycle_cost(states, 0.0);
    for (uint64_t mask = 1; mask < states; ++mask) {
        double best = INF;
        uint64_t rem = mask;
        while (rem) {
            uint64_t bit = rem & -rem;
            int j = ctz_u64(bit) + 1;
            best = std::min(best, hk[idx(mask, j)] + inst.dist[j][inst.root]);
            rem ^= bit;
        }
        cycle_cost[mask] = best;
    }
    return cycle_cost;
}

double expected_posteriori(const Instance &inst, const std::vector<double> &subset_costs) {
    const int n = inst.n;
    const int m = n - 1;
    const uint64_t states = 1ULL << m;
    double ans = 0.0;
    for (uint64_t mask = 0; mask < states; ++mask) {
        double prob = 1.0;
        for (int i = 1; i < n; ++i) {
            bool active = ((mask >> (i - 1)) & 1ULL) != 0ULL;
            prob *= active ? inst.probs[i] : (1.0 - inst.probs[i]);
        }
        ans += prob * subset_costs[mask];
    }
    return ans;
}

Summary adaptive_optimal_value(const Instance &inst, bool store_policy) {
    const int n = inst.n;
    const int m = n - 1;
    const uint64_t states = 1ULL << m;
    const uint64_t full_mask = states - 1ULL;

    std::vector<double> dp(static_cast<uint64_t>(n) * states, INF);
    std::vector<int> decision;
    if (store_policy) decision.assign(static_cast<uint64_t>(n) * states, -1);

    auto idx = [states](int cur, uint64_t mask) -> uint64_t {
        return static_cast<uint64_t>(cur) * states + mask;
    };

    for (int cur = 0; cur < n; ++cur) {
        dp[idx(cur, full_mask)] = inst.dist[cur][inst.root];
        if (store_policy) decision[idx(cur, full_mask)] = inst.root;
    }

    for (int size = m - 1; size >= 0; --size) {
        for (uint64_t mask = 0; mask < states; ++mask) {
            if (popcount_u64(mask) != size) continue;
            uint64_t unrevealed = full_mask ^ mask;
            if (unrevealed == 0ULL) continue;

            std::vector<int> current_vertices;
            current_vertices.push_back(inst.root);
            uint64_t rem_cur = mask;
            while (rem_cur) {
                uint64_t bit = rem_cur & -rem_cur;
                current_vertices.push_back(ctz_u64(bit) + 1);
                rem_cur ^= bit;
            }

            for (int cur : current_vertices) {
                double best = INF;
                int best_probe = -1;
                uint64_t rem = unrevealed;
                while (rem) {
                    uint64_t bit = rem & -rem;
                    int v = ctz_u64(bit) + 1;
                    uint64_t next_mask = mask | bit;
                    double p = inst.probs[v];
                    double cand = (1.0 - p) * dp[idx(cur, next_mask)] +
                                  p * (inst.dist[cur][v] + dp[idx(v, next_mask)]);
                    if (cand < best) {
                        best = cand;
                        best_probe = v;
                    }
                    rem ^= bit;
                }
                dp[idx(cur, mask)] = best;
                if (store_policy) decision[idx(cur, mask)] = best_probe;
            }
        }
    }

    Summary s;
    s.adaptive = dp[idx(inst.root, 0ULL)];
    if (store_policy) s.policy = std::move(decision);
    return s;
}

Summary ratio_summary(const Instance &inst, bool store_policy) {
    std::vector<double> subset_costs = all_subset_tsp_costs(inst);
    double post = expected_posteriori(inst, subset_costs);
    Summary s = adaptive_optimal_value(inst, store_policy);
    s.posteriori = post;
    s.ratio = (post <= EPS) ? INF : (s.adaptive / post);
    return s;
}

std::string json_escape(const std::string &s) {
    std::ostringstream oss;
    for (char c : s) {
        switch (c) {
            case '\\': oss << "\\\\"; break;
            case '"': oss << "\\\""; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default: oss << c; break;
        }
    }
    return oss.str();
}

void print_matrix(const std::vector<std::vector<double>> &mat) {
    for (const auto &row : mat) {
        std::cout << "[";
        for (size_t j = 0; j < row.size(); ++j) {
            if (j) std::cout << ' ';
            double x = row[j];
            if (std::fabs(x - std::round(x)) < 1e-9) {
                std::cout << std::setw(4) << static_cast<long long>(std::llround(x));
            } else {
                std::cout << std::fixed << std::setprecision(3) << std::setw(8) << x;
            }
        }
        std::cout << "]\n";
    }
}

void print_summary(const Instance &inst, const Summary &s) {
    std::cout << "family           = " << inst.family << "\n";
    std::cout << "n                = " << inst.n << "\n";
    std::cout << "root             = " << inst.root << "\n";
    std::cout << "a posteriori     = " << std::fixed << std::setprecision(6) << s.posteriori << "\n";
    std::cout << "adaptive         = " << std::fixed << std::setprecision(6) << s.adaptive << "\n";
    std::cout << "adaptive/post    = " << std::fixed << std::setprecision(6) << s.ratio << "\n";
    std::cout << "probs            = [";
    for (int i = 0; i < inst.n; ++i) {
        if (i) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(4) << inst.probs[i];
    }
    std::cout << "]\n";
    std::cout << "target_probs     = [";
    for (int i = 0; i < inst.n; ++i) {
        if (i) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(4) << inst.target_probs[i];
    }
    std::cout << "]\n";
    std::cout << "roles            = [";
    for (int i = 0; i < inst.n; ++i) {
        if (i) std::cout << ", ";
        std::cout << inst.roles[i];
    }
    std::cout << "]\n";
    std::cout << "distance matrix:\n";
    print_matrix(inst.dist);
}

void save_checkpoint(const std::string &path, const Instance &inst, const Summary &s,
                     int trial, int sample, size_t done, size_t total_work) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("failed to open checkpoint file: " + path);
    out << "{\n";
    out << "  \"family\": \"" << json_escape(inst.family) << "\",\n";
    out << "  \"n\": " << inst.n << ",\n";
    out << "  \"root\": " << inst.root << ",\n";
    out << "  \"trial\": " << trial << ",\n";
    out << "  \"sample\": " << sample << ",\n";
    out << "  \"done_work\": " << done << ",\n";
    out << "  \"total_work\": " << total_work << ",\n";
    out << "  \"posteriori\": " << std::setprecision(17) << s.posteriori << ",\n";
    out << "  \"adaptive\": " << std::setprecision(17) << s.adaptive << ",\n";
    out << "  \"ratio\": " << std::setprecision(17) << s.ratio << ",\n";

    out << "  \"probs\": [";
    for (int i = 0; i < inst.n; ++i) {
        if (i) out << ", ";
        out << std::setprecision(17) << inst.probs[i];
    }
    out << "],\n";

    out << "  \"target_probs\": [";
    for (int i = 0; i < inst.n; ++i) {
        if (i) out << ", ";
        out << std::setprecision(17) << inst.target_probs[i];
    }
    out << "],\n";

    out << "  \"roles\": [";
    for (int i = 0; i < inst.n; ++i) {
        if (i) out << ", ";
        out << inst.roles[i];
    }
    out << "],\n";

    out << "  \"dist\": [\n";
    for (int i = 0; i < inst.n; ++i) {
        out << "    [";
        for (int j = 0; j < inst.n; ++j) {
            if (j) out << ", ";
            out << std::setprecision(17) << inst.dist[i][j];
        }
        out << "]" << (i + 1 == inst.n ? "\n" : ",\n");
    }
    out << "  ]";

    if (!s.policy.empty()) {
        uint64_t states = 1ULL << (inst.n - 1);
        out << ",\n  \"policy\": [\n";
        for (int cur = 0; cur < inst.n; ++cur) {
            out << "    [";
            for (uint64_t mask = 0; mask < states; ++mask) {
                if (mask) out << ", ";
                out << s.policy[static_cast<uint64_t>(cur) * states + mask];
            }
            out << "]" << (cur + 1 == inst.n ? "\n" : ",\n");
        }
        out << "  ]\n";
    } else {
        out << "\n";
    }
    out << "}\n";
}

struct SearchResult {
    Instance inst;
    Summary summary;
    int best_trial = -1;
    int best_sample = -1;
};

SearchResult search_family(const std::string &family, int n, int trials, uint64_t seed,
                           int prob_samples, const std::string &save_best_path,
                           double progress_seconds = 5.0, double progress_fraction = 0.01) {
    if (n < 2) throw std::runtime_error("n must be at least 2");
    if (n > 22) {
        std::cerr << "warning: n=" << n << " will likely be very slow / memory-heavy in exact mode\n";
    }

    std::mt19937_64 rng(seed);
    const size_t total_work = static_cast<size_t>(trials) * static_cast<size_t>(prob_samples);
    ProgressReporter progress(total_work, progress_seconds, progress_fraction);

    double best_ratio = -INF;
    std::string best_desc;
    size_t done = 0;
    std::optional<SearchResult> best;

    for (int t = 0; t < trials; ++t) {
        Instance templ = make_gadget(family, n, rng);
        std::vector<double> subset_costs = all_subset_tsp_costs(templ);

        for (int s = 0; s < prob_samples; ++s) {
            Instance inst = templ;
            inst.probs = sample_probs_from_targets(templ, rng);
            Summary cur = adaptive_optimal_value(inst, true);
            cur.posteriori = expected_posteriori(inst, subset_costs);
            cur.ratio = (cur.posteriori <= EPS) ? INF : (cur.adaptive / cur.posteriori);
            ++done;

            if (cur.ratio > best_ratio) {
                best_ratio = cur.ratio;
                std::ostringstream oss;
                oss << "trial=" << t << ", sample=" << s << ", family=" << family;
                best_desc = oss.str();
                best = SearchResult{inst, cur, t, s};
                if (!save_best_path.empty()) {
                    save_checkpoint(save_best_path, inst, cur, t, s, done, total_work);
                }
                progress.improvement(done, best_ratio, best_desc);
            } else {
                progress.update(done, best_ratio, best_desc);
            }
        }
    }

    if (!best.has_value()) throw std::runtime_error("search did not evaluate any instance");
    progress.update(done, best_ratio, best_desc, true);
    return *best;
}

void usage(const char *prog) {
    std::cout << "Usage:\n"
              << "  " << prog << " list-gadgets\n"
              << "  " << prog << " demo <family> <seed>\n"
              << "  " << prog << " search <family> <n> <trials> <seed> <prob_samples> [save_best.json]\n"
              << "\nFamilies: random, branch, layered, gateway\n";
}

int main(int argc, char **argv) {
    try {
        if (argc < 2) {
            usage(argv[0]);
            return 1;
        }
        std::string cmd = argv[1];
        if (cmd == "list-gadgets") {
            std::cout << "random\nbranch\nlayered\ngateway\n";
            return 0;
        }
        if (cmd == "demo") {
            if (argc < 4) {
                usage(argv[0]);
                return 1;
            }
            std::string family = argv[2];
            uint64_t seed = std::stoull(argv[3]);
            SearchResult res = search_family(family, 8, 80, seed, 4, "", 1.0, 0.05);
            print_summary(res.inst, res.summary);
            std::cout << "best_trial       = " << res.best_trial << "\n";
            std::cout << "best_sample      = " << res.best_sample << "\n";
            return 0;
        }
        if (cmd == "search") {
            if (argc < 7) {
                usage(argv[0]);
                return 1;
            }
            std::string family = argv[2];
            int n = std::stoi(argv[3]);
            int trials = std::stoi(argv[4]);
            uint64_t seed = std::stoull(argv[5]);
            int prob_samples = std::stoi(argv[6]);
            std::string save_best_path = (argc >= 8) ? argv[7] : "best_current.json";
            SearchResult res = search_family(family, n, trials, seed, prob_samples, save_best_path);
            print_summary(res.inst, res.summary);
            std::cout << "best_trial       = " << res.best_trial << "\n";
            std::cout << "best_sample      = " << res.best_sample << "\n";
            if (!save_best_path.empty()) {
                std::cout << "saved_best       = " << save_best_path << "\n";
            }
            return 0;
        }

        usage(argv[0]);
        return 1;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
