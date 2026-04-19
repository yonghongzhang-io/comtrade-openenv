# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Comtrade Env Environment.

This module creates an HTTP server that exposes the ComtradeEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

# Import from local models.py (PYTHONPATH includes /app/env in Docker)
from models import ComtradeAction, ComtradeObservation
from .comtrade_env_environment import ComtradeEnvironment


# Create the app with web interface and README integration
app = create_app(
    ComtradeEnvironment,
    ComtradeAction,
    ComtradeObservation,
    env_name="comtrade_env",
    max_concurrent_envs=8,  # supports concurrent GRPO rollouts (SUPPORTS_CONCURRENT_SESSIONS=True)
)

# ---------------------------------------------------------------------------
# Custom landing page — replaces the raw Swagger UI at /
# ---------------------------------------------------------------------------
from fastapi.responses import HTMLResponse

LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ComtradeBench — An OpenEnv Benchmark for Reliable LLM Tool-Use</title>
<style>
  :root {
    --bg: #0f172a; --surface: #1e293b; --border: #334155;
    --text: #e2e8f0; --muted: #94a3b8; --accent: #6366f1;
    --accent2: #818cf8; --green: #22c55e; --red: #ef4444;
    --orange: #f59e0b; --radius: 12px;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: var(--bg); color: var(--text); line-height: 1.6; }

  .container { max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem; }

  /* Hero */
  .hero { text-align: center; padding: 3rem 0 2rem; }
  .hero-badge { display: inline-block; background: rgba(99,102,241,.15); color: var(--accent2);
                padding: .35rem 1rem; border-radius: 20px; font-size: .8rem; font-weight: 600;
                letter-spacing: .5px; margin-bottom: 1rem; border: 1px solid rgba(99,102,241,.3); }
  .hero h1 { font-size: 2.5rem; font-weight: 800; letter-spacing: -.02em;
             background: linear-gradient(135deg, #e2e8f0 30%, #6366f1 100%);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .hero p { color: var(--muted); font-size: 1.1rem; max-width: 650px; margin: .75rem auto 0; }

  /* Status bar */
  .status-bar { display: flex; justify-content: center; gap: 2rem; margin: 2rem 0;
                flex-wrap: wrap; }
  .status-item { display: flex; align-items: center; gap: .5rem; font-size: .9rem; }
  .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
  .dot.green { background: var(--green); box-shadow: 0 0 8px rgba(34,197,94,.5); }
  .dot.orange { background: var(--orange); }

  /* Cards grid */
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
          gap: 1.25rem; margin: 2rem 0; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
          padding: 1.5rem; transition: border-color .2s, transform .2s; }
  .card:hover { border-color: var(--accent); transform: translateY(-2px); }
  .card h3 { font-size: 1rem; font-weight: 700; margin-bottom: .75rem;
             display: flex; align-items: center; gap: .5rem; }
  .card-icon { font-size: 1.2rem; }

  /* Task table */
  .task-table { width: 100%; border-collapse: separate; border-spacing: 0;
                background: var(--surface); border: 1px solid var(--border);
                border-radius: var(--radius); overflow: hidden; margin: 2rem 0; }
  .task-table th { background: rgba(99,102,241,.1); color: var(--accent2); font-size: .75rem;
                   font-weight: 700; letter-spacing: .05em; text-transform: uppercase;
                   padding: .75rem 1rem; text-align: left; }
  .task-table td { padding: .65rem 1rem; font-size: .88rem; border-top: 1px solid var(--border); }
  .task-table tr:hover td { background: rgba(99,102,241,.04); }
  .task-id { font-family: 'JetBrains Mono', 'Fira Code', monospace; font-weight: 600;
             color: var(--accent2); font-size: .82rem; }
  .difficulty { display: inline-block; padding: .15rem .55rem; border-radius: 6px;
                font-size: .72rem; font-weight: 600; }
  .difficulty.easy { background: rgba(34,197,94,.15); color: #4ade80; }
  .difficulty.medium { background: rgba(245,158,11,.15); color: #fbbf24; }
  .difficulty.hard { background: rgba(239,68,68,.15); color: #f87171; }

  /* Scoring */
  .score-bar { display: flex; align-items: center; gap: .5rem; margin: .4rem 0; }
  .score-track { flex: 1; height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }
  .score-fill { height: 100%; border-radius: 3px; background: linear-gradient(90deg, var(--accent), var(--accent2)); }
  .score-label { font-size: .78rem; color: var(--muted); min-width: 90px; }
  .score-val { font-size: .78rem; font-weight: 600; min-width: 28px; text-align: right; }

  /* API section */
  .endpoint { display: flex; align-items: center; gap: .75rem; padding: .6rem 0;
              border-bottom: 1px solid var(--border); }
  .endpoint:last-child { border-bottom: none; }
  .method { font-family: monospace; font-size: .72rem; font-weight: 700; padding: .2rem .5rem;
            border-radius: 4px; min-width: 48px; text-align: center; }
  .method.get { background: rgba(34,197,94,.15); color: #4ade80; }
  .method.post { background: rgba(99,102,241,.15); color: var(--accent2); }
  .path { font-family: monospace; font-size: .88rem; font-weight: 500; }
  .endpoint .desc { font-size: .8rem; color: var(--muted); margin-left: auto; }

  /* Tool cards */
  .tool { background: rgba(99,102,241,.06); border: 1px solid rgba(99,102,241,.2);
          border-radius: 8px; padding: 1rem; margin-bottom: .75rem; }
  .tool-name { font-family: monospace; font-weight: 700; color: var(--accent2); font-size: .92rem; }
  .tool-sig { font-family: monospace; font-size: .78rem; color: var(--muted); margin-top: .25rem; }
  .tool-desc { font-size: .82rem; color: var(--text); margin-top: .4rem; }

  /* Footer */
  .footer { text-align: center; padding: 2rem 0; color: var(--muted); font-size: .82rem;
            border-top: 1px solid var(--border); margin-top: 2rem; }
  .footer a { color: var(--accent2); text-decoration: none; }
  .footer a:hover { text-decoration: underline; }

  /* Links */
  .btn { display: inline-flex; align-items: center; gap: .4rem; padding: .5rem 1.2rem;
         background: var(--accent); color: white; border-radius: 8px; font-size: .85rem;
         font-weight: 600; text-decoration: none; transition: background .2s; }
  .btn:hover { background: var(--accent2); }
  .btn-outline { background: transparent; border: 1px solid var(--border); color: var(--text); }
  .btn-outline:hover { border-color: var(--accent); color: var(--accent2); background: rgba(99,102,241,.08); }
  .btn-group { display: flex; gap: .75rem; justify-content: center; margin: 1.5rem 0; flex-wrap: wrap; }

  @media (max-width: 640px) {
    .hero h1 { font-size: 1.8rem; }
    .grid { grid-template-columns: 1fr; }
    .endpoint .desc { display: none; }
  }
</style>
</head>
<body>
<div class="container">

  <!-- Hero -->
  <div class="hero">
    <div class="hero-badge">OPENENV &middot; AGENTBEATS PHASE 2</div>
    <h1>ComtradeBench</h1>
    <p>An OpenEnv Benchmark for Reliable LLM Tool-Use Under Adversarial API Conditions.</p>
  </div>

  <!-- Status -->
  <div class="status-bar">
    <div class="status-item"><span class="dot green"></span> Environment Online</div>
    <div class="status-item"><span class="dot green"></span> Mock Service Active</div>
    <div class="status-item">10 Tasks &middot; 3 MCP Tools &middot; 6 Scoring Dimensions</div>
  </div>

  <!-- Action buttons -->
  <div class="btn-group">
    <a class="btn" href="/docs">API Documentation</a>
    <a class="btn btn-outline" href="https://github.com/yonghongzhang-io/comtrade-openenv" target="_blank">GitHub Repository</a>
    <a class="btn btn-outline" href="https://huggingface.co/spaces/yonghongzhang/comtrade-bench-blog" target="_blank">Read the Blog →</a>
    <a class="btn btn-outline" href="/health">Health Check</a>
  </div>

  <!-- Tasks -->
  <h2 style="font-size:1.2rem; margin:2.5rem 0 .5rem; font-weight:700;">Benchmark Tasks</h2>
  <table class="task-table">
    <thead>
      <tr><th>Task</th><th>Name</th><th>Challenge</th><th>Difficulty</th></tr>
    </thead>
    <tbody>
      <tr><td class="task-id">T1</td><td>Single Page</td><td>Fetch one page, submit correctly</td><td><span class="difficulty easy">Easy</span></td></tr>
      <tr><td class="task-id">T2</td><td>Multi-Page</td><td>Paginate until <code>has_more=false</code></td><td><span class="difficulty easy">Easy</span></td></tr>
      <tr><td class="task-id">T3</td><td>Deduplication</td><td>Overlapping pages; dedup by primary key</td><td><span class="difficulty medium">Medium</span></td></tr>
      <tr><td class="task-id">T4</td><td>Rate Limit (429)</td><td>Retry on HTTP 429 without data loss</td><td><span class="difficulty medium">Medium</span></td></tr>
      <tr><td class="task-id">T5</td><td>Server Error (500)</td><td>Retry transient 500 failures</td><td><span class="difficulty medium">Medium</span></td></tr>
      <tr><td class="task-id">T6</td><td>Page Drift</td><td>Non-deterministic page order</td><td><span class="difficulty hard">Hard</span></td></tr>
      <tr><td class="task-id">T7</td><td>Totals Trap</td><td>Drop summary rows (<code>is_total=true</code>)</td><td><span class="difficulty hard">Hard</span></td></tr>
      <tr><td class="task-id">T8</td><td>Mixed Faults</td><td>429 rate-limit AND cross-page duplicates simultaneously</td><td><span class="difficulty hard">Hard</span></td></tr>
      <tr><td class="task-id">T9</td><td>Adaptive Adversary</td><td>Faults escalate mid-episode based on agent progress</td><td><span class="difficulty hard" style="background:rgba(239,68,68,.25);color:#fca5a5;">Novel</span></td></tr>
      <tr><td class="task-id">T10</td><td>Constrained Budget</td><td>Single agent under halved budget, avoid redundant fetches</td><td><span class="difficulty hard" style="background:rgba(239,68,68,.25);color:#fca5a5;">Novel</span></td></tr>
    </tbody>
  </table>

  <!-- Info cards -->
  <div class="grid">
    <!-- MCP Tools -->
    <div class="card">
      <h3><span class="card-icon">🔧</span> MCP Tools</h3>
      <div class="tool">
        <div class="tool-name">get_task_info()</div>
        <div class="tool-desc">Returns task description, query params, request budget remaining</div>
      </div>
      <div class="tool">
        <div class="tool-name">fetch_page(page, page_size)</div>
        <div class="tool-sig">→ {rows, page, total_pages, has_more}</div>
        <div class="tool-desc">Fetches one page; may return 429/500 faults</div>
      </div>
      <div class="tool">
        <div class="tool-name">submit_results(data, meta, log)</div>
        <div class="tool-sig">→ {reward, score, breakdown}</div>
        <div class="tool-desc">Submit deduplicated records for scoring</div>
      </div>
    </div>

    <!-- Scoring -->
    <div class="card">
      <h3><span class="card-icon">📊</span> Scoring (0–100)</h3>
      <div class="score-bar">
        <span class="score-label">Correctness</span>
        <div class="score-track"><div class="score-fill" style="width:30%"></div></div>
        <span class="score-val">30</span>
      </div>
      <div class="score-bar">
        <span class="score-label">Completeness</span>
        <div class="score-track"><div class="score-fill" style="width:15%"></div></div>
        <span class="score-val">15</span>
      </div>
      <div class="score-bar">
        <span class="score-label">Robustness</span>
        <div class="score-track"><div class="score-fill" style="width:15%"></div></div>
        <span class="score-val">15</span>
      </div>
      <div class="score-bar">
        <span class="score-label">Efficiency</span>
        <div class="score-track"><div class="score-fill" style="width:15%"></div></div>
        <span class="score-val">15</span>
      </div>
      <div class="score-bar">
        <span class="score-label">Data Quality</span>
        <div class="score-track"><div class="score-fill" style="width:15%"></div></div>
        <span class="score-val">15</span>
      </div>
      <div class="score-bar">
        <span class="score-label">Observability</span>
        <div class="score-track"><div class="score-fill" style="width:10%"></div></div>
        <span class="score-val">10</span>
      </div>
    </div>

    <div class="card">
      <h3><span class="card-icon">📈</span> Results Snapshot</h3>
      <p style="font-size:.85rem; color:var(--text); margin-bottom:.6rem;">
        Rule-based baseline: <strong>96.8 / 100</strong> across T1-T10.
      </p>
      <p style="font-size:.82rem; color:var(--muted); line-height:1.7;">
        Kimi V1-128k &amp; Claude Sonnet 4.6: <strong style="color:var(--text)">97.5 / 100</strong>
        each (+0.7), identical per-task scores on all 10 tasks &mdash; frontier saturation. Llama 3.3 70B
        collapses on T9 to <strong style="color:#f87171">18.7</strong>: <strong style="color:var(--text)">T9
        produces a 78.8-pt frontier vs. sub-frontier gap</strong>.
      </p>
    </div>

    <div class="card">
      <h3><span class="card-icon">🧭</span> Why It Matters</h3>
      <p style="font-size:.82rem; color:var(--muted); line-height:1.7;">
        Final answers are not enough. ComtradeBench rewards agents that recover from 429/500
        faults, deduplicate correctly, filter totals rows, stay within budget, and leave an
        auditable run log.
      </p>
    </div>

    <!-- API Endpoints -->
    <div class="card">
      <h3><span class="card-icon">⚡</span> API Endpoints</h3>
      <div class="endpoint">
        <span class="method post">POST</span>
        <span class="path">/reset</span>
        <span class="desc">Start new episode</span>
      </div>
      <div class="endpoint">
        <span class="method post">POST</span>
        <span class="path">/step</span>
        <span class="desc">Execute MCP action</span>
      </div>
      <div class="endpoint">
        <span class="method get">GET</span>
        <span class="path">/state</span>
        <span class="desc">Current env state</span>
      </div>
      <div class="endpoint">
        <span class="method get">GET</span>
        <span class="path">/schema</span>
        <span class="desc">Action/Obs schemas</span>
      </div>
      <div class="endpoint">
        <span class="method get">GET</span>
        <span class="path">/health</span>
        <span class="desc">Health check</span>
      </div>
      <div class="endpoint">
        <span class="method get">GET</span>
        <span class="path">/docs</span>
        <span class="desc">Swagger UI</span>
      </div>
    </div>

    <!-- Architecture -->
    <div class="card">
      <h3><span class="card-icon">🏗️</span> Architecture</h3>
      <p style="font-size:.85rem; color:var(--muted); margin-bottom:.75rem;">
        Built on <strong style="color:var(--text)">OpenEnv</strong> — the open RL environment framework by Meta.
      </p>
      <p style="font-size:.82rem; color:var(--muted); line-height:1.7;">
        <strong style="color:var(--text)">Environment:</strong> MCPEnvironment with FastMCP tools<br>
        <strong style="color:var(--text)">Mock Service:</strong> FastAPI with seeded RNG data generation<br>
        <strong style="color:var(--text)">Fault Injection:</strong> Configurable 429/500 errors per task<br>
        <strong style="color:var(--text)">Scoring:</strong> 6-dimension weighted judge (0–100)<br>
        <strong style="color:var(--text)">Training:</strong> GRPO with parallel rollouts<br>
        <strong style="color:var(--text)">Concurrency:</strong> Episode-isolated, thread-safe
      </p>
    </div>
  </div>

  <!-- Live Demo -->
  <h2 style="font-size:1.2rem; margin:2.5rem 0 .5rem; font-weight:700;">Try It Live</h2>
  <div class="card" style="margin-bottom:2rem;">
    <h3><span class="card-icon">🚀</span> Interactive API Demo</h3>
    <p style="font-size:.85rem; color:var(--muted); margin-bottom:1rem;">
      Click the buttons below to interact with the live environment. Each call hits the real API running in this Space.
    </p>
    <div style="display:flex; gap:.5rem; flex-wrap:wrap; margin-bottom:1rem;">
      <button onclick="demoCall('/health','GET')" class="btn" style="cursor:pointer;border:none;font-size:.8rem;padding:.4rem .8rem;">Health Check</button>
      <button onclick="demoCall('/reset','POST',{task_id:'T1_single_page'})" class="btn" style="cursor:pointer;border:none;font-size:.8rem;padding:.4rem .8rem;">Reset T1</button>
      <button onclick="demoCall('/state','GET')" class="btn btn-outline" style="cursor:pointer;font-size:.8rem;padding:.4rem .8rem;">Get State</button>
      <button onclick="demoCall('/schema','GET')" class="btn btn-outline" style="cursor:pointer;font-size:.8rem;padding:.4rem .8rem;">View Schema</button>
    </div>
    <pre id="demo-output" style="background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:1rem;font-size:.78rem;max-height:300px;overflow:auto;color:var(--green);font-family:'JetBrains Mono','Fira Code',monospace;">Click a button above to see the API response...</pre>
  </div>
  <script>
  async function demoCall(path, method, body) {
    const out = document.getElementById('demo-output');
    out.textContent = `${method} ${path} ...`;
    try {
      const opts = {method, headers:{'Content-Type':'application/json'}};
      if (body) opts.body = JSON.stringify(body);
      const r = await fetch(path, opts);
      const data = await r.json();
      out.textContent = `${method} ${path}  →  ${r.status}\\n\\n` + JSON.stringify(data, null, 2);
    } catch(e) {
      out.textContent = `Error: ${e.message}`;
    }
  }
  </script>

  <!-- Footer -->
  <div class="footer">
    <p>ComtradeBench &middot; AgentBeats Phase 2 &middot; OpenEnv Challenge</p>
    <p style="margin-top:.4rem;">
      <a href="https://github.com/yonghongzhang-io/comtrade-openenv">GitHub</a> &middot;
      <a href="https://huggingface.co/spaces/yonghongzhang/comtrade-bench-blog">Blog</a> &middot;
      <a href="/docs">API Docs</a> &middot;
      <a href="https://github.com/meta-pytorch/OpenEnv">OpenEnv Framework</a>
    </p>
  </div>

</div>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def landing_page():
    return LANDING_HTML


@app.get("/web", response_class=HTMLResponse, include_in_schema=False)
async def landing_page_web():
    return LANDING_HTML


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m comtrade_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn comtrade_env.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
