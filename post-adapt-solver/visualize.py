#!/usr/bin/env python3
"""
Interactive visualization of stochastic TSP graphs.

Generates a standalone HTML file using vis.js — drag nodes, zoom, pan.
Optionally runs the C++ solver and displays costs/ratios.

Usage:
  python visualize.py examples/small_symmetric.json
  python visualize.py examples/two_circles_k3.json --run-solver
  python visualize.py examples/*.json --run-solver
  python visualize.py examples/gap_example.json -o graph.html
"""

import argparse
import html as html_mod
import json
import math
import os
import subprocess
import sys
import tempfile
import webbrowser


def load_instance(path):
    """Load a JSON instance and return (n, dist, prob, edges_raw, meta)."""
    with open(path) as f:
        data = json.load(f)

    n = data["n"]
    V = n + 1
    prob = data["prob"]

    edges_raw = None
    if "dist" in data:
        dist = [row[:] for row in data["dist"]]
    elif "edges" in data:
        INF = 1e18
        dist = [[INF] * V for _ in range(V)]
        for i in range(V):
            dist[i][i] = 0
        edges_raw = data["edges"]
        for e in edges_raw:
            u, v, w = e["from"], e["to"], e["weight"]
            if w < dist[u][v]:
                dist[u][v] = w
        # Floyd-Warshall
        for k in range(V):
            for i in range(V):
                for j in range(V):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
    else:
        raise ValueError("JSON must have 'dist' or 'edges'")

    meta = data.get("_comment", "")
    return n, dist, prob, edges_raw, meta


def check_symmetric(n, dist, edges_raw):
    """Check if the graph is symmetric."""
    V = n + 1
    if edges_raw is not None:
        edge_set = {}
        for e in edges_raw:
            edge_set[(e["from"], e["to"])] = e["weight"]
        for (u, v), w in list(edge_set.items()):
            if (v, u) not in edge_set or abs(edge_set[(v, u)] - w) > 1e-9:
                return False
        return True
    else:
        for i in range(V):
            for j in range(V):
                if abs(dist[i][j] - dist[j][i]) > 1e-9:
                    return False
        return True


def get_display_edges(n, dist, edges_raw, is_symmetric):
    """Get edges for display. For edge-list inputs use explicit edges; for matrix use all pairs."""
    V = n + 1
    INF = 1e15
    edges = []

    if edges_raw is not None:
        seen = set()
        for e in edges_raw:
            u, v, w = e["from"], e["to"], e["weight"]
            if is_symmetric:
                key = (min(u, v), max(u, v))
                if key in seen:
                    continue
                seen.add(key)
            edges.append((u, v, w))
    else:
        if is_symmetric:
            for i in range(V):
                for j in range(i + 1, V):
                    if dist[i][j] < INF:
                        edges.append((i, j, dist[i][j]))
        else:
            for i in range(V):
                for j in range(V):
                    if i != j and dist[i][j] < INF:
                        edges.append((i, j, dist[i][j]))

    return edges


def auto_positions(n, edges_raw, meta):
    """Compute initial node positions. Returns dict {id: {x, y}} or None for physics layout."""
    V = n + 1

    # Two-circle graph
    if "Two-circle" in meta or "two_circle" in meta.lower():
        k = (n + 1) // 2
        pos = {}
        r = 150
        cx_a, cx_b = -250, 250
        for i in range(k):
            angle = -math.pi / 2 + 2 * math.pi * i / k
            pos[i] = {"x": cx_a + r * math.cos(angle), "y": r * math.sin(angle)}
        for i in range(k):
            vi = k + i
            if vi < V:
                angle = -math.pi / 2 + 2 * math.pi * i / k
                pos[vi] = {"x": cx_b + r * math.cos(angle), "y": r * math.sin(angle)}
        return pos

    # Default circular layout
    pos = {}
    r = 50 * V
    for i in range(V):
        angle = -math.pi / 2 + 2 * math.pi * i / V
        pos[i] = {"x": r * math.cos(angle), "y": r * math.sin(angle)}
    return pos


def prob_to_color(p):
    """Map probability [0,1] to a blue color hex string."""
    # Light blue (low p) to dark blue (high p)
    r = int(220 - 180 * p)
    g = int(230 - 180 * p)
    b = int(255 - 55 * p)
    return f"#{r:02x}{g:02x}{b:02x}"


def run_solver(json_path):
    """Run the C++ solver and return stdout."""
    solver_dir = os.path.dirname(os.path.abspath(json_path))
    for solver_path in [
        os.path.join(solver_dir, "build", "solver"),
        os.path.join(solver_dir, "solver"),
        "./build/solver",
        "./solver",
    ]:
        if os.path.isfile(solver_path) and os.access(solver_path, os.X_OK):
            try:
                result = subprocess.run(
                    [solver_path, json_path],
                    capture_output=True, text=True, timeout=60
                )
                return result.stdout.strip()
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
    return None


def build_html(instances):
    """Build a standalone HTML page with vis.js network for each instance."""
    # Build tab structure if multiple instances
    multi = len(instances) > 1

    tabs_html = ""
    panels_html = ""

    for idx, inst in enumerate(instances):
        name = inst["name"]
        n = inst["n"]
        prob = inst["prob"]
        display_edges = inst["edges"]
        is_symmetric = inst["is_symmetric"]
        positions = inst["positions"]
        solver_output = inst.get("solver_output", "")
        meta = inst.get("meta", "")
        V = n + 1

        # Build vis.js nodes
        vis_nodes = []
        for i in range(V):
            if i == 0:
                vis_nodes.append({
                    "id": i,
                    "label": "Depot (0)",
                    "color": {"background": "#ff4444", "border": "#333333",
                              "highlight": {"background": "#ff6666", "border": "#000"}},
                    "shape": "square",
                    "size": 25,
                    "font": {"size": 14, "face": "monospace", "bold": True, "color": "#fff"},
                })
            else:
                color = prob_to_color(prob[i])
                vis_nodes.append({
                    "id": i,
                    "label": f"{i}  (p={prob[i]:.2g})",
                    "color": {"background": color, "border": "#333333",
                              "highlight": {"background": color, "border": "#000"}},
                    "shape": "circle",
                    "size": 20,
                    "font": {"size": 13, "face": "monospace", "color": "#222"},
                })

        # Apply positions
        if positions:
            for node in vis_nodes:
                nid = node["id"]
                if nid in positions:
                    node["x"] = positions[nid]["x"]
                    node["y"] = positions[nid]["y"]

        # Build vis.js edges
        vis_edges = []
        for u, v, w in display_edges:
            label = f"{w:.4g}" if w != int(w) else str(int(w))
            edge = {
                "from": u, "to": v,
                "label": label,
                "font": {"size": 10, "color": "#666", "strokeWidth": 0, "align": "middle"},
                "color": {"color": "#999", "highlight": "#333"},
                "width": 1.5,
            }
            if not is_symmetric:
                edge["arrows"] = "to"
            vis_edges.append(edge)

        active_class = "active" if idx == 0 else ""

        # Info panel content
        info_parts = [f"<b>{html_mod.escape(name)}</b> &mdash; n={n}, {'symmetric' if is_symmetric else 'asymmetric'}"]
        if meta:
            info_parts.append(f"<span style='color:#666'>{html_mod.escape(meta[:120])}</span>")
        if solver_output:
            info_parts.append(f"<pre style='margin:6px 0 0 0;font-size:12px;line-height:1.4'>{html_mod.escape(solver_output)}</pre>")

        info_html = "<br>".join(info_parts)

        nodes_json = json.dumps(vis_nodes)
        edges_json = json.dumps(vis_edges)
        use_physics = "true" if positions is None else "false"

        tab_id = f"tab_{idx}"
        if multi:
            tabs_html += f'<button class="tab-btn {active_class}" onclick="switchTab({idx})" id="btn_{idx}">{html_mod.escape(name)}</button>\n'

        panels_html += f"""
<div class="tab-panel {active_class}" id="{tab_id}">
  <div class="info-bar">{info_html}</div>
  <div class="net-container" id="net_{idx}"></div>
</div>
<script>
(function() {{
  var nodes = new vis.DataSet({nodes_json});
  var edges = new vis.DataSet({edges_json});
  var container = document.getElementById('net_{idx}');
  var data = {{ nodes: nodes, edges: edges }};
  var options = {{
    physics: {{
      enabled: {use_physics},
      solver: 'forceAtlas2Based',
      forceAtlas2Based: {{ gravitationalConstant: -100, springLength: 150 }},
      stabilization: {{ iterations: 200 }}
    }},
    interaction: {{
      dragNodes: true,
      dragView: true,
      zoomView: true,
      hover: true,
      tooltipDelay: 100,
    }},
    edges: {{
      smooth: {{ type: 'continuous' }},
    }},
    layout: {{
      improvedLayout: true,
    }},
  }};
  var network = new vis.Network(container, data, options);
  window._networks = window._networks || [];
  window._networks.push(network);
}})();
</script>
"""

    tabs_section = ""
    if multi:
        tabs_section = f'<div class="tab-bar">{tabs_html}</div>'

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Stochastic TSP Graph Visualizer</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{ height: 100%; overflow: hidden; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
         background: #f5f5f5; color: #333; display: flex; flex-direction: column; }}
  .header {{ background: #2c3e50; color: white; padding: 12px 20px; font-size: 16px; flex-shrink: 0; }}
  .tab-bar {{ background: #34495e; padding: 0 10px; display: flex; gap: 2px; overflow-x: auto; flex-shrink: 0; }}
  .tab-btn {{ background: #3d566e; color: #ccc; border: none; padding: 10px 18px;
              cursor: pointer; font-size: 13px; font-family: monospace; white-space: nowrap;
              border-radius: 4px 4px 0 0; transition: background 0.15s; }}
  .tab-btn:hover {{ background: #4a6a8a; color: #fff; }}
  .tab-btn.active {{ background: #f5f5f5; color: #333; }}
  .tab-panel {{ display: none; flex: 1; min-height: 0; }}
  .tab-panel.active {{ display: flex; flex-direction: column; }}
  .info-bar {{ padding: 12px 16px; background: #fff; border-bottom: 1px solid #ddd;
               font-size: 13px; line-height: 1.5; flex-shrink: 0; }}
  .info-bar pre {{ background: #f8f8f8; padding: 8px 12px; border-radius: 4px;
                   border: 1px solid #e0e0e0; overflow-x: auto; }}
  .net-container {{ flex: 1; min-height: 0; background: #fff; }}
  .legend {{ position: fixed; bottom: 12px; right: 12px; background: rgba(255,255,255,0.95);
             border: 1px solid #ddd; border-radius: 6px; padding: 10px 14px; font-size: 12px;
             box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  .legend-item {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; }}
  .legend-swatch {{ width: 18px; height: 18px; border: 1px solid #999; border-radius: 3px; }}
</style>
</head>
<body>
<div class="header">Stochastic TSP Graph Visualizer</div>
{tabs_section}
{panels_html}

<div class="legend">
  <div style="font-weight:bold; margin-bottom:4px;">Legend</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#ff4444;border-radius:2px;"></div> Depot (vertex 0)</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#dce6ff;"></div> Low probability</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#2832c8;"></div> High probability</div>
  <div style="margin-top:6px;color:#888;font-size:11px;">Drag nodes to rearrange<br>Scroll to zoom</div>
</div>

<script>
function switchTab(idx) {{
  document.querySelectorAll('.tab-panel').forEach(function(el) {{ el.classList.remove('active'); }});
  document.querySelectorAll('.tab-btn').forEach(function(el) {{ el.classList.remove('active'); }});
  document.getElementById('tab_' + idx).classList.add('active');
  document.getElementById('btn_' + idx).classList.add('active');
  // Redraw network after tab switch
  if (window._networks && window._networks[idx]) {{
    setTimeout(function() {{ window._networks[idx].fit(); }}, 50);
  }}
}}
</script>
</body>
</html>
"""
    return page


def main():
    parser = argparse.ArgumentParser(
        description="Interactive visualization of stochastic TSP graphs (HTML + vis.js).",
    )
    parser.add_argument("files", nargs="+", help="JSON input file(s)")
    parser.add_argument("--run-solver", action="store_true",
                        help="Run the C++ solver and show results")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output HTML file path (default: open in browser)")

    args = parser.parse_args()

    instances = []
    for path in args.files:
        n, dist, prob, edges_raw, meta = load_instance(path)
        is_sym = check_symmetric(n, dist, edges_raw)
        display_edges = get_display_edges(n, dist, edges_raw, is_sym)
        positions = auto_positions(n, edges_raw, meta)

        solver_output = ""
        if args.run_solver:
            out = run_solver(path)
            if out:
                solver_output = out
                print(f"--- {path} ---")
                print(out)
                print()

        instances.append({
            "name": os.path.basename(path),
            "n": n,
            "prob": prob,
            "edges": display_edges,
            "is_symmetric": is_sym,
            "positions": positions,
            "solver_output": solver_output,
            "meta": meta,
        })

    html_content = build_html(instances)

    if args.output:
        with open(args.output, "w") as f:
            f.write(html_content)
        print(f"Saved to {args.output}")
    else:
        # Write to temp file and open in browser
        fd, tmp_path = tempfile.mkstemp(suffix=".html", prefix="tsp_viz_")
        with os.fdopen(fd, "w") as f:
            f.write(html_content)
        webbrowser.open(f"file://{tmp_path}")
        print(f"Opened in browser ({tmp_path})")


if __name__ == "__main__":
    main()
