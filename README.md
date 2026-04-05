---
title: Comtrade Env Environment Server
emoji: 📊
colorFrom: indigo
colorTo: gray
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Comtrade Env — UN Trade Data Fetching Benchmark

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) MCP environment that benchmarks LLM agents on paginated data fetching against a simulated UN Comtrade API.

The agent must fetch trade records across multiple pages, handle faults (rate limits, server errors, duplicates, page drift), deduplicate data, and submit clean output — all via three MCP tools.

## Tasks (T1–T7)

| ID | Name | Description |
|----|------|-------------|
| T1 | Single page | Fetch one page, submit. Baseline correctness. |
| T2 | Multi-page pagination | Iterate pages until `has_more=False`. |
| T3 | Deduplication | Pages overlap; agent must dedup by primary key. |
| T4 | HTTP 429 retry | Rate-limit fault; agent must retry without data loss. |
| T5 | HTTP 500 retry | Server error fault; agent must retry transient failures. |
| T6 | Page drift | Non-deterministic page ordering; agent must handle instability. |
| T7 | Totals trap | Summary rows mixed in; agent must drop `is_total=true` rows. |

## MCP Tools

```
get_task_info()
  → Returns task_id, description, query params (reporter, partner, flow, hs, year),
    constraints, mock_service_url, and request budget remaining.

fetch_page(page: int = 1, page_size: int = 500)
  → Fetches one page of trade records from the mock service.
    Returns: {rows, page, total_pages, has_more}
    On fault: {status: 429|500, retry: true}

submit_results(data_jsonl, metadata_json, run_log)
  → Submits final deduplicated records for scoring.
    Returns: {reward (0.0–1.0), score (0–100), breakdown, errors}
```

## Scoring (0–100 → reward 0.0–1.0)

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| Correctness | 30 | All expected rows present and correct |
| Completeness | 15 | No missing records |
| Robustness | 15 | Correct handling of 429/500 faults |
| Efficiency | 15 | Request count relative to minimum needed |
| Data Quality | 15 | No duplicates, no totals rows leaked |
| Observability | 10 | `run.log` contains `task_id=`, `page=`, `request=`, `complete=` |

## Quick Start

```python
from comtrade_env import ComtradeAction, ComtradeEnv

with ComtradeEnv(base_url="http://localhost:8000") as env:
    result = env.reset(task_id="T2_multi_page")
    task_info = env.call_tool("get_task_info", {})

    collected = {}
    page = 1
    while True:
        data = env.call_tool("fetch_page", {"page": page, "page_size": 500})
        if data.get("status") in (429, 500):
            data = env.call_tool("fetch_page", {"page": page, "page_size": 500})
        for row in data.get("rows", []):
            if row.get("is_total"):
                continue
            pk = "|".join(str(row.get(k, "")) for k in
                         ("year", "reporter", "partner", "flow", "hs", "record_id"))
            collected[pk] = row
        if not data.get("has_more"):
            break
        page += 1

    import json
    data_jsonl = "\n".join(json.dumps(r) for r in collected.values())
    metadata = json.dumps({
        "task_id": task_info["task_id"],
        "query": task_info["query"],
        "row_count": len(collected),
        "schema": list(next(iter(collected.values())).keys()) if collected else [],
        "dedup_key": ["year", "reporter", "partner", "flow", "hs", "record_id"],
        "totals_handling": {"enabled": True, "rows_dropped": 0},
    })
    result = env.call_tool("submit_results", {
        "data_jsonl": data_jsonl,
        "metadata_json": metadata,
        "run_log": f"task_id={task_info['task_id']}\npage=1\nrequest=1\ncomplete=true",
    })
    print(f"reward={result['reward']:.4f}  score={result['score']:.1f}")
    print(result['breakdown'])
```

## Running the Server

```bash
# From the comtrade_env directory
uvicorn server.app:app --port 8000

# Or with Docker
docker build -t comtrade-env:latest -f server/Dockerfile .
docker run -p 8000:8000 comtrade-env:latest
```

## Smoke Test (rule-based agent, no LLM required)

```bash
# Start the server first, then:
cd llm_agent
python smoke_test.py --env-url http://localhost:8000 --task T1_single_page
```

## GRPO Training

Train an LLM agent with Group Relative Policy Optimization:

```bash
cd llm_agent

# Using a local Ollama/vLLM endpoint (rollout-only, no gradient updates)
python train_grpo.py \
    --env-url http://localhost:8000 \
    --api-url http://localhost:11434/v1 \
    --api-model qwen2.5:7b \
    --iterations 200

# Using a HuggingFace model (full training with gradients)
python train_grpo.py \
    --env-url http://localhost:8000 \
    --hf-model Qwen/Qwen2.5-7B-Instruct \
    --iterations 200 \
    --save-dir ./checkpoints
```

## Project Structure

```
comtrade_env/
├── __init__.py                  # Module exports
├── README.md                    # This file
├── openenv.yaml                 # OpenEnv manifest
├── pyproject.toml               # Project dependencies
├── client.py                    # ComtradeEnv HTTP/WebSocket client
├── models.py                    # ComtradeAction / ComtradeObservation
└── server/
    ├── app.py                   # FastAPI app (HTTP + WebSocket)
    ├── comtrade_env_environment.py  # Core MCP environment logic
    ├── tasks.py                 # Task definitions (T1–T7)
    ├── judge.py                 # Scoring engine
    ├── mock_service/            # Embedded mock Comtrade API
    │   ├── app.py               # FastAPI mock with fault injection
    │   └── fixtures/            # Ground-truth data per task
    └── Dockerfile               # Container image
```

## Deploying to Hugging Face Spaces

```bash
# From the environment directory
openenv push

# Or specify a target
openenv push --repo-id my-org/comtrade-env --private
```

After deployment the space exposes:
- **`/web`** — Interactive UI
- **`/docs`** — OpenAPI / Swagger
- **`/health`** — Health check
- **`/ws`** — WebSocket endpoint for persistent sessions
