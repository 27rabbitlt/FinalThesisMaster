// Topology-decomposed search for maximum adapt/apost ratio.
//
// Key idea: separate TOPOLOGY (which edges exist) from WEIGHTS (how heavy).
// For a sparse graph with m edges, the search space is m-dimensional
// (vs n*(n-1)/2 for the full SA). Floyd-Warshall turns edge weights into a metric.
//
// Strategy:
//   1. Generate candidate topologies (random trees + extra edges, known structures)
//   2. For each topology, SA/optimize over edge weights only
//   3. Try each non-depot vertex as the single stochastic vertex (p=0.5)
//   4. Report the best instance found
//
// Usage:
//   ./topology_search -n 10 --topologies 200 --iters 500000
//   ./topology_search -n 13 --topologies 100 --iters 1000000 --3chain

#include "tsp_solver.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

struct Topology {
    int V;
    vector<pair<int,int>> edges;  // undirected edges
    string name;
};

struct Result {
    double ratio;
    int stoch_vertex;
    vector<double> weights;
    Topology topo;
    vector<vector<double>> dist;
};

// Build metric from topology + weights
static vector<vector<double>> build_metric(const Topology& topo, const vector<double>& weights) {
    int V = topo.V;
    vector<vector<double>> d(V, vector<double>(V, INF));
    for (int i = 0; i < V; ++i) d[i][i] = 0;
    for (int e = 0; e < (int)topo.edges.size(); ++e) {
        auto [a, b] = topo.edges[e];
        d[a][b] = d[b][a] = weights[e];
    }
    floyd_warshall(V, d);
    return d;
}

// Evaluate adapt/apost ratio for given metric with vertex sv stochastic at p=0.5
static double eval(int V, const vector<vector<double>>& dist, int sv) {
    vector<double> prob(V, 1.0);
    prob[sv] = 0.5;
    int nc = V - 1;
    double apost = solve_a_posteriori(nc, dist, prob);
    if (apost < 1e-12) return 1.0;
    double adapt = solve_adaptive(nc, dist, prob);
    return adapt / apost;
}

// ============================================================
// Topology generators
// ============================================================

// Random spanning tree via random Prüfer sequence
static Topology random_tree(int V, mt19937& rng, int extra_edges = 0) {
    Topology t;
    t.V = V;

    // Prüfer sequence → tree
    uniform_int_distribution<int> vert(0, V - 1);
    vector<int> prufer(V - 2);
    for (auto& x : prufer) x = vert(rng);

    vector<int> degree(V, 1);
    for (int x : prufer) degree[x]++;

    set<pair<int,int>> edge_set;
    for (int x : prufer) {
        for (int i = 0; i < V; ++i) {
            if (degree[i] == 1) {
                int a = min(i, x), b = max(i, x);
                edge_set.insert({a, b});
                degree[i]--;
                degree[x]--;
                break;
            }
        }
    }
    // Last edge
    vector<int> leaves;
    for (int i = 0; i < V; ++i)
        if (degree[i] == 1) leaves.push_back(i);
    if (leaves.size() == 2)
        edge_set.insert({min(leaves[0], leaves[1]), max(leaves[0], leaves[1])});

    // Add extra random edges
    uniform_int_distribution<int> vpick(0, V - 1);
    for (int i = 0; i < extra_edges; ++i) {
        for (int tries = 0; tries < 100; ++tries) {
            int a = vpick(rng), b = vpick(rng);
            if (a == b) continue;
            if (a > b) swap(a, b);
            if (edge_set.count({a, b})) continue;
            edge_set.insert({a, b});
            break;
        }
    }

    for (auto& [a, b] : edge_set)
        t.edges.push_back({a, b});

    t.name = "tree+" + to_string(extra_edges);
    return t;
}

// Complete graph topology
static Topology complete_graph(int V) {
    Topology t;
    t.V = V;
    for (int i = 0; i < V; ++i)
        for (int j = i + 1; j < V; ++j)
            t.edges.push_back({i, j});
    t.name = "complete";
    return t;
}

// Three-chain topology G_k: 3 chains of length k + 1 stochastic vertex
// Vertices: 0=depot=a_0, a_1..a_{k-1}, b_0..b_{k-1}, c_0..c_{k-1}, s
static Topology three_chain_topo(int k) {
    int V = 3 * k + 1;
    Topology t;
    t.V = V;

    // a_i = i (0..k-1), b_i = k+i, c_i = 2k+i, s = 3k
    int s = 3 * k;

    // Internal chain edges
    for (int c = 0; c < 3; ++c) {
        int base = c * k;
        for (int i = 0; i < k - 1; ++i)
            t.edges.push_back({base + i, base + i + 1});
    }

    // Inter-chain edges (as in the paper)
    int a0 = 0, ak1 = k - 1;
    int b0 = k, bk1 = 2 * k - 1;
    int c0 = 2 * k, ck1 = 3 * k - 1;

    t.edges.push_back({a0, bk1});
    t.edges.push_back({a0, ck1});
    t.edges.push_back({b0, ak1});
    t.edges.push_back({c0, ak1});
    t.edges.push_back({c0, bk1});
    t.edges.push_back({b0, s});
    t.edges.push_back({ck1, s});

    t.name = "3chain_k" + to_string(k);
    return t;
}

// 3-chain variant: 3 chains with ARBITRARY inter-chain edges and stochastic vertex placement
// chain_lens[3] = lengths of chains A, B, C (can be different!)
// inter_mask = bitmask selecting which of the possible inter-chain edges to include
// s_conn = bitmask selecting which chain endpoints the stochastic vertex connects to
static Topology three_chain_variant(const int chain_lens[3], int inter_mask, int s_conn, int variant_id) {
    int ka = chain_lens[0], kb = chain_lens[1], kc = chain_lens[2];
    int V = 1 + (ka - 1) + kb + kc + 1;  // depot(=a0) + rest of A + B + C + s
    // a_i = i for i=0..ka-1, b_i = ka+i for i=0..kb-1, c_i = ka+kb+i for i=0..kc-1, s = ka+kb+kc
    Topology t;
    t.V = V;

    int a0 = 0, ak = ka - 1;
    int b0 = ka, bk = ka + kb - 1;
    int c0 = ka + kb, ck = ka + kb + kc - 1;
    int s = ka + kb + kc;

    // Internal chain edges
    for (int i = 0; i < ka - 1; ++i) t.edges.push_back({i, i + 1});
    for (int i = b0; i < bk; ++i) t.edges.push_back({i, i + 1});
    for (int i = c0; i < ck; ++i) t.edges.push_back({i, i + 1});

    // All possible inter-chain edges between the 6 chain endpoints:
    // a0, ak, b0, bk, c0, ck
    int endpoints[6] = {a0, ak, b0, bk, c0, ck};
    // All 15 pairs of distinct endpoints (excluding same-chain pairs: (a0,ak), (b0,bk), (c0,ck))
    // That leaves 15 - 3 = 12 cross-chain pairs
    vector<pair<int,int>> cross_pairs;
    for (int i = 0; i < 6; ++i)
        for (int j = i + 1; j < 6; ++j) {
            // Skip same-chain pairs
            if (i / 2 == j / 2) continue;
            cross_pairs.push_back({endpoints[i], endpoints[j]});
        }
    // inter_mask selects which cross-chain edges to include (up to 12 bits)
    for (int b = 0; b < (int)cross_pairs.size(); ++b) {
        if (inter_mask & (1 << b))
            t.edges.push_back(cross_pairs[b]);
    }

    // s_conn selects which endpoints the stochastic vertex connects to (6 bits)
    for (int b = 0; b < 6; ++b) {
        if (s_conn & (1 << b))
            t.edges.push_back({s, endpoints[b]});
    }

    // Deduplicate
    set<pair<int,int>> eset;
    for (auto& [a, b] : t.edges) {
        int lo = min(a, b), hi = max(a, b);
        if (lo != hi) eset.insert({lo, hi});
    }
    t.edges.clear();
    for (auto& e : eset) t.edges.push_back(e);

    t.name = "3cv_" + to_string(ka) + to_string(kb) + to_string(kc) + "_" + to_string(variant_id);
    return t;
}

// Enumerate promising 3-chain variants for a given V
static vector<Topology> enumerate_3chain_variants(int V, mt19937& rng, int max_topos = 200) {
    vector<Topology> topos;
    int vid = 0;

    // Try different chain length partitions: ka + kb + kc + 1(stoch) = V, ka >= 1
    // Since a0 = depot, chain A has ka vertices (including depot)
    for (int ka = 2; ka <= V - 3; ++ka) {
        for (int kb = 1; kb <= V - ka - 2; ++kb) {
            int kc = V - 1 - ka - kb;
            if (kc < 1) continue;

            int chain_lens[3] = {ka, kb, kc};

            // For the inter-chain connectivity:
            // The paper's pattern uses 5 of 12 possible cross-chain edges.
            // Enumerate subsets of size 3-7 (need enough for Hamiltonian cycle, not too many)
            // But 2^12 = 4096 subsets × 2^6 = 64 s-connections = too many.
            // Instead: sample random subsets.

            uniform_int_distribution<int> inter_dist(0, (1 << 12) - 1);
            uniform_int_distribution<int> sconn_dist(1, (1 << 6) - 1);  // at least 1 connection

            int trials_per_partition = max(1, max_topos / 10);
            for (int trial = 0; trial < trials_per_partition; ++trial) {
                int inter_mask = inter_dist(rng);
                int s_conn = sconn_dist(rng);

                // Require: stochastic vertex has exactly 2 connections (like the paper)
                int sc = __builtin_popcount(s_conn);
                if (sc != 2) {
                    // Force exactly 2 connections
                    s_conn = 0;
                    vector<int> bits = {0,1,2,3,4,5};
                    shuffle(bits.begin(), bits.end(), rng);
                    s_conn = (1 << bits[0]) | (1 << bits[1]);
                }

                // Require: 4-7 inter-chain edges (enough for cycle, not too dense)
                int ic = __builtin_popcount(inter_mask);
                if (ic < 4 || ic > 7) continue;

                auto topo = three_chain_variant(chain_lens, inter_mask, s_conn, vid++);
                // Sanity: need at least V-1 edges for connectivity
                if ((int)topo.edges.size() >= V - 1)
                    topos.push_back(topo);
            }
        }
    }

    // Also add the exact paper topology if it fits
    if ((V - 1) % 3 == 0) {
        int k = (V - 1) / 3;
        if (k >= 2) topos.push_back(three_chain_topo(k));
    }

    // Shuffle and limit
    shuffle(topos.begin(), topos.end(), rng);
    if ((int)topos.size() > max_topos)
        topos.resize(max_topos);

    return topos;
}

// Star topology: depot connected to all, plus a ring among customers
static Topology star_ring(int V) {
    Topology t;
    t.V = V;
    for (int i = 1; i < V; ++i)
        t.edges.push_back({0, i});
    for (int i = 1; i < V - 1; ++i)
        t.edges.push_back({i, i + 1});
    t.edges.push_back({1, V - 1});
    t.name = "star_ring";
    return t;
}

// Hub-and-spokes: depot + hub vertex, spokes of length L
static Topology hub_spokes(int n_spokes, int spoke_len) {
    int V = 2 + n_spokes * spoke_len;
    Topology t;
    t.V = V;
    t.edges.push_back({0, 1});
    int vid = 2;
    for (int s = 0; s < n_spokes; ++s) {
        t.edges.push_back({1, vid});
        for (int i = 0; i < spoke_len - 1; ++i)
            t.edges.push_back({vid + i, vid + i + 1});
        vid += spoke_len;
    }
    t.name = "hub_" + to_string(n_spokes) + "x" + to_string(spoke_len);
    return t;
}

// ============================================================
// NEW: Topologies designed for tour restructuring
// ============================================================

// Ladder: two parallel paths connected by rungs
// 0-1-2-..-(k-1) and k-(k+1)-..-(2k-1), plus rungs i-(k+i)
// Stochastic vertex = 2k connected to specific ladder vertices
static Topology ladder(int k) {
    int V = 2 * k + 1;  // +1 for stochastic vertex
    Topology t;
    t.V = V;
    // Top path (includes depot=0)
    for (int i = 0; i < k - 1; ++i) t.edges.push_back({i, i + 1});
    // Bottom path
    for (int i = k; i < 2 * k - 1; ++i) t.edges.push_back({i, i + 1});
    // Rungs
    for (int i = 0; i < k; ++i) t.edges.push_back({i, k + i});
    // Close into a cycle: connect ends
    t.edges.push_back({k - 1, 2 * k - 1});
    // Stochastic vertex connects to two non-adjacent ladder vertices
    int s = 2 * k;
    t.edges.push_back({s, k});            // bottom-left
    t.edges.push_back({s, 2 * k - 1});    // bottom-right
    t.name = "ladder_k" + to_string(k);
    return t;
}

// Prism graph: two cycles connected by edges, + stochastic vertex
// Cycle 1: 0-1-..(m-1)-0, Cycle 2: m-(m+1)-..(2m-1)-m, spokes i-(m+i)
static Topology prism(int m) {
    int V = 2 * m + 1;
    Topology t;
    t.V = V;
    // Cycle 1 (includes depot=0)
    for (int i = 0; i < m; ++i) t.edges.push_back({i, (i + 1) % m});
    // Cycle 2
    for (int i = 0; i < m; ++i) t.edges.push_back({m + i, m + (i + 1) % m});
    // Spokes
    for (int i = 0; i < m; ++i) t.edges.push_back({i, m + i});
    // Stochastic vertex
    int s = 2 * m;
    t.edges.push_back({s, m});              // connect to cycle 2 start
    t.edges.push_back({s, m + m / 2});      // connect to opposite side of cycle 2
    t.name = "prism_m" + to_string(m);
    return t;
}

// N-chain generalization: N chains of length k emanating from depot,
// with inter-chain edges at endpoints + 1 stochastic vertex
static Topology n_chain(int n_chains, int k) {
    // Vertices: 0=depot, chains of k vertices each, then stochastic
    int V = 1 + n_chains * k + 1;
    Topology t;
    t.V = V;
    int s = V - 1;

    vector<int> chain_start(n_chains), chain_end(n_chains);
    for (int c = 0; c < n_chains; ++c) {
        int base = 1 + c * k;
        chain_start[c] = base;
        chain_end[c] = base + k - 1;
        // Depot to chain start
        t.edges.push_back({0, base});
        // Internal edges
        for (int i = 0; i < k - 1; ++i)
            t.edges.push_back({base + i, base + i + 1});
    }

    // Inter-chain: connect each chain_end[i] to chain_start[(i+1)%N]
    for (int c = 0; c < n_chains; ++c)
        t.edges.push_back({chain_end[c], chain_start[(c + 1) % n_chains]});

    // Also connect alternating ends for tour restructuring potential
    for (int c = 0; c < n_chains; ++c)
        t.edges.push_back({chain_end[c], chain_start[(c + 2) % n_chains]});

    // Stochastic vertex connects to two chain endpoints
    t.edges.push_back({s, chain_start[1]});
    t.edges.push_back({s, chain_end[n_chains - 1]});

    // Deduplicate edges
    set<pair<int,int>> eset;
    for (auto& [a, b] : t.edges) {
        int lo = min(a, b), hi = max(a, b);
        if (lo != hi) eset.insert({lo, hi});
    }
    t.edges.clear();
    for (auto& e : eset) t.edges.push_back(e);

    t.name = to_string(n_chains) + "chain_k" + to_string(k);
    return t;
}

// Wheel: central vertex (depot=0) + outer cycle 1..V-2, stochastic = V-1
// with extra spokes from depot to every other cycle vertex
static Topology wheel(int V) {
    Topology t;
    t.V = V;
    int cyc = V - 2;  // cycle size (excluding depot and stoch)
    int s = V - 1;
    // Depot to some cycle vertices
    for (int i = 1; i <= cyc; i += 2)
        t.edges.push_back({0, i});
    // Cycle
    for (int i = 1; i < cyc; ++i) t.edges.push_back({i, i + 1});
    t.edges.push_back({1, cyc});
    // Stochastic connects to two opposite cycle vertices
    t.edges.push_back({s, 1});
    t.edges.push_back({s, cyc / 2 + 1});
    t.name = "wheel";
    return t;
}

// Theta graph: three internally disjoint paths between two vertices
// Good for forcing different tour structures
static Topology theta(int len1, int len2, int len3) {
    // Paths from vertex 0 to vertex "meet"
    // Path 1: 0 - a1 - a2 - ... - a_{len1} (=meet)
    // Path 2: 0 - b1 - ... - b_{len2} (=meet)  [shares endpoints]
    // Path 3: 0 - c1 - ... - c_{len3} (=meet)
    // Then stochastic vertex connects to interior of two paths
    int meet = 1;  // vertex 1 is the other endpoint
    int vid = 2;
    Topology t;

    vector<vector<int>> paths(3);
    int lens[3] = {len1, len2, len3};
    for (int p = 0; p < 3; ++p) {
        paths[p].push_back(0);
        for (int i = 0; i < lens[p] - 1; ++i)
            paths[p].push_back(vid++);
        paths[p].push_back(meet);
    }
    int s = vid++;  // stochastic vertex
    t.V = vid;

    for (int p = 0; p < 3; ++p)
        for (int i = 0; i + 1 < (int)paths[p].size(); ++i)
            t.edges.push_back({paths[p][i], paths[p][i + 1]});

    // Stochastic connects to midpoints of path 1 and path 2
    int mid1 = paths[0][len1 / 2];
    int mid2 = paths[1][len2 / 2];
    t.edges.push_back({s, mid1});
    t.edges.push_back({s, mid2});

    t.name = "theta_" + to_string(len1) + "_" + to_string(len2) + "_" + to_string(len3);
    return t;
}

// Double-triangle: two triangles sharing an edge, with chains inside
// Designed for maximum tour restructuring
static Topology double_triangle(int chain_len) {
    // Triangle 1: 0 - A_chain - pivot1
    // Triangle 2: 0 - B_chain - pivot2
    // Shared edge: pivot1 - pivot2
    // Cross edges: pivot1 to B_chain start, pivot2 to A_chain start
    // Stochastic vertex on the shared edge
    int k = chain_len;
    int V = 1 + 2 * k + 1;  // depot + 2 chains + stochastic
    Topology t;
    t.V = V;

    int a_start = 1, a_end = k;
    int b_start = k + 1, b_end = 2 * k;
    int s = 2 * k + 1;

    // Chain A: depot(0) -> 1 -> 2 -> ... -> k
    for (int i = 0; i < k; ++i) t.edges.push_back({i, i + 1});
    // Chain B: depot(0) -> k+1 -> k+2 -> ... -> 2k
    t.edges.push_back({0, b_start});
    for (int i = b_start; i < b_end; ++i) t.edges.push_back({i, i + 1});

    // Cross connections (create two triangles)
    t.edges.push_back({a_end, b_end});       // shared edge between chain ends
    t.edges.push_back({a_end, b_start});     // cross: A end to B start
    t.edges.push_back({b_end, a_start});     // cross: B end to A start

    // Stochastic vertex
    t.edges.push_back({s, a_end});
    t.edges.push_back({s, b_end});

    t.name = "dbl_tri_k" + to_string(k);
    return t;
}

// Möbius ladder: cycle of 2m vertices with antipodal connections
static Topology mobius_ladder(int m) {
    int V = 2 * m + 1;  // +1 for stochastic
    Topology t;
    t.V = V;
    // Cycle: 0, 1, ..., 2m-1
    for (int i = 0; i < 2 * m; ++i)
        t.edges.push_back({i, (i + 1) % (2 * m)});
    // Antipodal edges
    for (int i = 0; i < m; ++i)
        t.edges.push_back({i, i + m});
    // Stochastic vertex
    int s = 2 * m;
    t.edges.push_back({s, 1});
    t.edges.push_back({s, m});
    t.name = "mobius_m" + to_string(m);
    return t;
}

// Generalized Petersen graph GP(n,k): outer cycle + inner star
static Topology gen_petersen(int n, int k) {
    int V = 2 * n + 1;  // outer + inner + stochastic
    Topology t;
    t.V = V;
    // Outer cycle: 0..n-1
    for (int i = 0; i < n; ++i)
        t.edges.push_back({i, (i + 1) % n});
    // Inner star: n..2n-1, connected as i -> i+k (mod n)
    for (int i = 0; i < n; ++i)
        t.edges.push_back({n + i, n + (i + k) % n});
    // Spokes: outer i -> inner i
    for (int i = 0; i < n; ++i)
        t.edges.push_back({i, n + i});
    // Stochastic vertex
    int s = 2 * n;
    t.edges.push_back({s, n});          // connect to inner vertex
    t.edges.push_back({s, n + n / 2});  // connect to opposite inner
    t.name = "petersen_" + to_string(n) + "_" + to_string(k);
    return t;
}

// ============================================================
// Weight optimizer (SA over edge weights only)
// ============================================================

static Result optimize_topology(const Topology& topo, int sa_iters, double d_max,
                                mt19937& rng, bool try_all_sv = true) {
    int V = topo.V;
    int m = (int)topo.edges.size();

    uniform_real_distribution<double> unif(0.0, 1.0);
    uniform_int_distribution<int> edge_pick(0, m - 1);

    double T0 = 0.02;
    double T_final = 1e-6;
    double cool_rate = pow(T_final / T0, 1.0 / sa_iters);

    Result global_best;
    global_best.ratio = 1.0;

    // Quick screening: random weights, eval all sv, keep top candidates
    vector<int> sv_candidates;
    {
        // Screen with a few random weight vectors
        vector<pair<double,int>> sv_scores(V - 1, {1.0, 0});
        for (int i = 0; i < V - 1; ++i) sv_scores[i].second = i + 1;
        for (int trial = 0; trial < 3; ++trial) {
            vector<double> init_w(m);
            for (int e = 0; e < m; ++e)
                init_w[e] = 0.1 + unif(rng) * (d_max - 0.1);
            auto init_dist = build_metric(topo, init_w);
            for (int i = 0; i < V - 1; ++i)
                sv_scores[i].first = max(sv_scores[i].first, eval(V, init_dist, i + 1));
        }
        sort(sv_scores.rbegin(), sv_scores.rend());
        int keep = try_all_sv ? V - 1 : min(3, V - 1);
        for (int i = 0; i < keep; ++i)
            sv_candidates.push_back(sv_scores[i].second);
    }

    for (int sv : sv_candidates) {
        // Random initial weights
        vector<double> cur_w(m);
        for (int e = 0; e < m; ++e)
            cur_w[e] = 0.1 + unif(rng) * (d_max - 0.1);

        auto cur_dist = build_metric(topo, cur_w);
        double cur_ratio = eval(V, cur_dist, sv);
        double best_ratio = cur_ratio;
        vector<double> best_w = cur_w;

        double T = T0;
        for (int it = 0; it < sa_iters; ++it) {
            vector<double> next_w = cur_w;
            int e = edge_pick(rng);

            double scale = max(0.01, sqrt(T / T0));
            double delta = (unif(rng) - 0.5) * 2.0 * d_max * scale;
            next_w[e] = max(0.01, min(d_max, next_w[e] + delta));

            auto next_dist = build_metric(topo, next_w);
            double next_ratio = eval(V, next_dist, sv);

            double diff = next_ratio - cur_ratio;
            if (diff > 0 || unif(rng) < exp(diff / T)) {
                cur_w = next_w;
                cur_dist = next_dist;
                cur_ratio = next_ratio;
                if (cur_ratio > best_ratio) {
                    best_ratio = cur_ratio;
                    best_w = cur_w;
                }
            }
            T *= cool_rate;
        }

        if (best_ratio > global_best.ratio) {
            global_best.ratio = best_ratio;
            global_best.stoch_vertex = sv;
            global_best.weights = best_w;
            global_best.topo = topo;
            global_best.dist = build_metric(topo, best_w);
        }
    }

    return global_best;
}

// ============================================================
// JSON output
// ============================================================

static string to_json(const Result& res) {
    int V = res.topo.V;
    ostringstream oss;
    oss << "{\n  \"_comment\": \"topo_search: " << res.topo.name
        << ", ratio=" << res.ratio << ", sv=" << res.stoch_vertex << "\",\n";
    oss << "  \"n\": " << V << ",\n";
    oss << "  \"dist\": [\n";
    for (int i = 0; i < V; ++i) {
        oss << "    [";
        for (int j = 0; j < V; ++j) {
            oss << res.dist[i][j];
            if (j + 1 < V) oss << ", ";
        }
        oss << "]" << (i + 1 < V ? "," : "") << "\n";
    }
    oss << "  ],\n";
    oss << "  \"prob\": [";
    for (int i = 0; i < V; ++i) {
        if (i) oss << ", ";
        oss << (i == res.stoch_vertex ? 0.5 : 1.0);
    }
    oss << "],\n";
    oss << "  \"edges\": [";
    for (int e = 0; e < (int)res.topo.edges.size(); ++e) {
        if (e) oss << ", ";
        auto [a, b] = res.topo.edges[e];
        oss << "{\"s\":" << a << ",\"t\":" << b << ",\"w\":" << res.weights[e] << "}";
    }
    oss << "],\n  \"sym\": true\n}\n";
    return oss.str();
}

// ============================================================
// Main
// ============================================================

int main(int argc, char* argv[]) {
    int V = 10;
    int n_topos = 200;
    int sa_iters = 500000;
    double d_max = 10.0;
    int seed = 42;
    string outfile = "";
    bool include_3chain = false;
    bool verbose = false;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "-n" && i+1 < argc) V = atoi(argv[++i]);
        else if (arg == "--topologies" && i+1 < argc) n_topos = atoi(argv[++i]);
        else if (arg == "--iters" && i+1 < argc) sa_iters = atoi(argv[++i]);
        else if (arg == "-d" && i+1 < argc) d_max = atof(argv[++i]);
        else if (arg == "-s" && i+1 < argc) seed = atoi(argv[++i]);
        else if (arg == "-o" && i+1 < argc) outfile = argv[++i];
        else if (arg == "--3chain") include_3chain = true;
        else if (arg == "-v") verbose = true;
    }

    mt19937 rng(seed);
    int nc = V - 1;

    printf("=== Topology-Decomposed Search ===\n");
    printf("  V=%d, topologies=%d, SA iters/topo=%d, d_max=%.1f\n\n", V, n_topos, sa_iters, d_max);

    Result global_best;
    global_best.ratio = 1.0;
    int topo_count = 0;

    auto try_topo = [&](const Topology& topo, int iters_override = 0) {
        topo_count++;
        int iters = iters_override > 0 ? iters_override : sa_iters;
        auto res = optimize_topology(topo, iters, d_max, rng, false);

        if (verbose || res.ratio > global_best.ratio) {
            printf("  [%3d] %-20s  |E|=%2d  sv=%d  ratio=%.6f",
                   topo_count, topo.name.c_str(), (int)topo.edges.size(),
                   res.stoch_vertex, res.ratio);
            if (res.ratio > global_best.ratio + 1e-9)
                printf("  >>> NEW BEST <<<");
            printf("\n");
            fflush(stdout);
        }

        if (res.ratio > global_best.ratio)
            global_best = res;
    };

    // Phase 1: Known good topologies
    printf("--- Phase 1: Structured topologies ---\n");

    if (include_3chain && (V - 1) % 3 == 0) {
        int k = (V - 1) / 3;
        if (k >= 2) try_topo(three_chain_topo(k));
    }

    // Star+ring
    try_topo(star_ring(V));

    // Hub-and-spokes (various configs)
    for (int ns = 2; ns <= 4; ++ns) {
        int sl = (V - 2) / ns;
        if (sl >= 1 && 2 + ns * sl == V)
            try_topo(hub_spokes(ns, sl));
    }

    // Complete graph
    if (V <= 8) try_topo(complete_graph(V));

    // NEW: Ladder graphs
    for (int k = 3; k <= 8; ++k) {
        if (2 * k + 1 == V) try_topo(ladder(k));
    }

    // NEW: Prism graphs
    for (int m = 3; m <= 8; ++m) {
        if (2 * m + 1 == V) try_topo(prism(m));
    }

    // NEW: N-chain generalizations (2,4,5 chains)
    for (int nc : {2, 4, 5}) {
        for (int k = 1; k <= 6; ++k) {
            if (1 + nc * k + 1 == V) try_topo(n_chain(nc, k));
        }
    }

    // NEW: Wheel
    if (V >= 6) try_topo(wheel(V));

    // NEW: Theta graphs (3 paths between two vertices)
    for (int l1 = 2; l1 <= 5; ++l1)
        for (int l2 = l1; l2 <= 5; ++l2)
            for (int l3 = l2; l3 <= 5; ++l3) {
                int vcount = 2 + (l1 - 1) + (l2 - 1) + (l3 - 1) + 1; // 2 endpoints + internal + stochastic
                if (vcount == V) try_topo(theta(l1, l2, l3));
            }

    // NEW: Double-triangle
    for (int k = 2; k <= 8; ++k) {
        if (2 * k + 2 == V) try_topo(double_triangle(k));
    }

    // NEW: Möbius ladder
    for (int m = 3; m <= 8; ++m) {
        if (2 * m + 1 == V) try_topo(mobius_ladder(m));
    }

    // NEW: Generalized Petersen graphs
    for (int n = 3; n <= 8; ++n) {
        if (2 * n + 1 != V) continue;
        for (int k = 1; k < n / 2; ++k)
            try_topo(gen_petersen(n, k));
    }

    printf("  Phase 1 done: %d structured topologies, best=%.6f\n\n", topo_count, global_best.ratio);

    // Phase 2: 3-chain variants with randomized connectivity
    auto variants = enumerate_3chain_variants(V, rng, n_topos);
    printf("\n--- Phase 2: 3-chain variants (%d topologies) ---\n", (int)variants.size());

    for (int t = 0; t < (int)variants.size(); ++t) {
        try_topo(variants[t]);

        if ((t + 1) % 20 == 0) {
            fprintf(stderr, "  %d/%d variants done, best=%.6f\r", t + 1, (int)variants.size(), global_best.ratio);
        }
    }

    // Phase 3: Refine best topology with more iterations
    printf("\n--- Phase 3: Refine best topology (%s) ---\n", global_best.topo.name.c_str());
    for (int r = 0; r < 5; ++r) {
        auto res = optimize_topology(global_best.topo, sa_iters * 4, d_max, rng, true);
        if (res.ratio > global_best.ratio) {
            global_best = res;
            printf("  Refinement %d: ratio=%.6f >>> NEW BEST <<<\n", r, res.ratio);
        } else {
            printf("  Refinement %d: ratio=%.6f\n", r, res.ratio);
        }
    }

    printf("\n=== FINAL BEST ===\n");
    printf("  Topology: %s (|E|=%d)\n", global_best.topo.name.c_str(),
           (int)global_best.topo.edges.size());
    printf("  Stochastic vertex: %d\n", global_best.stoch_vertex);
    printf("  Ratio: %.6f\n", global_best.ratio);
    printf("  Edge weights:");
    for (int e = 0; e < (int)global_best.weights.size(); ++e) {
        auto [a, b] = global_best.topo.edges[e];
        printf(" (%d-%d):%.2f", a, b, global_best.weights[e]);
    }
    printf("\n");

    if (!outfile.empty()) {
        ofstream f(outfile);
        f << to_json(global_best);
        printf("  Saved to %s\n", outfile.c_str());
    }

    return 0;
}
