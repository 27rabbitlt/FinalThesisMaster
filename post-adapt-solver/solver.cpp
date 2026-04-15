// Exact solver for a posteriori TSP and adaptive TSP
// See CLAUDE.md for definitions and algorithm details.
//
// Usage: ./solver input.json
//
// JSON format:
// {
//   "n": <number of customers>,
//   "dist": [ [d00,d01,...], [d10,d11,...], ... ],   // (n+1)x(n+1) matrix, vertex 0 = depot
//   "prob": [0.0, p1, p2, ..., pn]                   // p[0]=0 (depot), p[i] for customer i
// }
//
// If "dist" is omitted but "edges" is provided, we build the distance matrix from edges:
// "edges": [ {"from": u, "to": v, "weight": w}, ... ]
// Missing edges get weight INF; then Floyd-Warshall computes all-pairs shortest paths.

#include "tsp_solver.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

// ---------- Minimal JSON parser (no external dependency) ----------
// Supports the subset we need: objects, arrays, numbers, strings.

struct JsonValue;
using JsonObject = vector<pair<string, JsonValue>>;
using JsonArray = vector<JsonValue>;

struct JsonValue {
    enum Type { NUL, NUM, STR, ARR, OBJ } type = NUL;
    double num = 0;
    string str;
    JsonArray arr;
    JsonObject obj;

    const JsonValue& operator[](const string& key) const {
        for (auto& kv : obj) if (kv.first == key) return kv.second;
        static JsonValue null_val;
        return null_val;
    }
    const JsonValue& operator[](size_t i) const { return arr[i]; }
    bool has(const string& key) const {
        for (auto& kv : obj) if (kv.first == key) return true;
        return false;
    }
};

static void skip_ws(const string& s, size_t& i) {
    while (i < s.size() && isspace((unsigned char)s[i])) ++i;
}

static JsonValue parse_value(const string& s, size_t& i);

static string parse_string(const string& s, size_t& i) {
    ++i; // skip "
    string res;
    while (i < s.size() && s[i] != '"') {
        if (s[i] == '\\') { ++i; res += s[i++]; }
        else res += s[i++];
    }
    ++i; // skip closing "
    return res;
}

static JsonValue parse_value(const string& s, size_t& i) {
    skip_ws(s, i);
    JsonValue v;
    if (s[i] == '"') {
        v.type = JsonValue::STR;
        v.str = parse_string(s, i);
    } else if (s[i] == '[') {
        v.type = JsonValue::ARR;
        ++i; skip_ws(s, i);
        if (s[i] != ']') {
            v.arr.push_back(parse_value(s, i));
            skip_ws(s, i);
            while (s[i] == ',') { ++i; v.arr.push_back(parse_value(s, i)); skip_ws(s, i); }
        }
        ++i; // ]
    } else if (s[i] == '{') {
        v.type = JsonValue::OBJ;
        ++i; skip_ws(s, i);
        if (s[i] != '}') {
            skip_ws(s, i);
            string key = parse_string(s, i);
            skip_ws(s, i); ++i; // :
            v.obj.push_back({key, parse_value(s, i)});
            skip_ws(s, i);
            while (s[i] == ',') {
                ++i; skip_ws(s, i);
                key = parse_string(s, i);
                skip_ws(s, i); ++i; // :
                v.obj.push_back({key, parse_value(s, i)});
                skip_ws(s, i);
            }
        }
        ++i; // }
    } else if (s[i] == 'n') {
        i += 4; // null
    } else if (s[i] == 't') {
        v.type = JsonValue::NUM; v.num = 1; i += 4; // true
    } else if (s[i] == 'f') {
        v.type = JsonValue::NUM; v.num = 0; i += 5; // false
    } else {
        // number
        v.type = JsonValue::NUM;
        size_t start = i;
        if (s[i] == '-') ++i;
        while (i < s.size() && (isdigit((unsigned char)s[i]) || s[i] == '.' || s[i] == 'e' || s[i] == 'E' || s[i] == '+' || s[i] == '-')) ++i;
        // careful: avoid double-consuming a minus that's part of a negative number vs exponent
        v.num = stod(s.substr(start, i - start));
    }
    return v;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " input.json [--no-apriori]\n"
             << "  --no-apriori  skip a priori computation (fast for large n)\n";
        return 1;
    }

    bool skip_apriori = false;
    string json_path;
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--no-apriori") skip_apriori = true;
        else json_path = arg;
    }
    if (json_path.empty()) { cerr << "No input file given.\n"; return 1; }

    // Auto-skip a priori for large instances (nc > 10 => 11! ops = too slow)
    ifstream fin(json_path);
    if (!fin) { cerr << "Cannot open " << json_path << endl; return 1; }
    string content((istreambuf_iterator<char>(fin)), istreambuf_iterator<char>());
    fin.close();

    size_t pos = 0;
    JsonValue root = parse_value(content, pos);

    int n = (int)root["n"].num;  // total vertices including depot (depot = vertex 0)
    int V = n;

    // Read probabilities: prob[0] = 1.0 (depot always present), prob[i] for customers
    vector<double> p(V, 0.0);
    if (root.has("prob")) {
        for (int i = 0; i < V && i < (int)root["prob"].arr.size(); ++i)
            p[i] = root["prob"].arr[i].num;
    }

    // Read distance matrix (V x V)
    vector<vector<double>> d(V, vector<double>(V, INF));
    for (int i = 0; i < V; ++i) d[i][i] = 0;

    bool sym = root.has("sym") && root["sym"].num != 0;

    if (root.has("dist")) {
        for (int i = 0; i < V; ++i)
            for (int j = 0; j < V; ++j)
                d[i][j] = root["dist"].arr[i].arr[j].num;
    } else if (root.has("edges")) {
        for (auto& e : root["edges"].arr) {
            int u = (int)e["s"].num;
            int v = (int)e["t"].num;
            double w = 1;
            if (e.has("w"))
                w = e["w"].num;
            else 
                w = 1;
            d[u][v] = w;
            if (sym) {
                d[v][u] = w;
            }
        }
        floyd_warshall(V, d);
    } else {
        cerr << "Input must have either 'dist' or 'edges'." << endl;
        return 1;
    }

    // Always compute shortest-path distances: the input graph is a metric generator,
    // so d[u][v] should be the shortest-path distance between u and v.
    floyd_warshall(V, d);

    int nc = V - 1;  // number of customers
    printf("n = %d vertices (depot=0, customers=1..%d)\n", V, nc);
    printf("Probabilities: ");
    for (int i = 0; i < V; ++i) printf("p[%d]=%.4g ", i, p[i]);
    printf("\n\n");

    double ap = solve_a_posteriori(nc, d, p);
    printf("A posteriori expected cost: %.6f\n", ap);

    double ad = solve_adaptive(nc, d, p);
    printf("Adaptive expected cost:     %.6f\n", ad);

    if (nc > 10) skip_apriori = true;  // auto-skip: 11! is already ~40M permutations
    double apr = -1;
    if (!skip_apriori) {
        apr = solve_a_priori(nc, d, p);
        printf("A priori expected cost:     %.6f\n", apr);
    } else {
        printf("A priori expected cost:     (skipped, nc=%d)\n", nc);
    }

    if (ap > 1e-12) {
        printf("\nRatios (a_posteriori as baseline):\n");
        printf("  adaptive / a_posteriori = %.6f\n", ad / ap);
        if (!skip_apriori && apr >= 0) {
            printf("  a_priori / a_posteriori = %.6f\n", apr / ap);
            printf("  a_priori / adaptive     = %.6f\n", apr / ad);
        }
    }

    return 0;
}
