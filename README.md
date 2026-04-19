---
title: "ComtradeBench: An OpenEnv Benchmark for Reliable LLM Tool-Use Under Adversarial API Conditions"
emoji: 📊
colorFrom: indigo
colorTo: gray
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - rl-environment
  - adversarial
  - tool-use
  - benchmark
  - grpo
---

<p align="center">
  <img src="banner.png" width="100%" alt="ComtradeBench — An OpenEnv Benchmark for Reliable LLM Tool-Use Under Adversarial API Conditions"/>
</p>

<p align="center">
  <a href="https://github.com/yonghongzhang-io/comtrade-openenv"><img src="https://img.shields.io/badge/GitHub-Repository-181717?logo=github" alt="GitHub"/></a>
  &nbsp;
  <a href="https://huggingface.co/spaces/yonghongzhang/comtrade-env"><img src="https://img.shields.io/badge/HF%20Space-Live%20Demo-FFD21E?logo=huggingface&logoColor=black" alt="HF Space"/></a>
  &nbsp;
  <a href="https://huggingface.co/spaces/yonghongzhang/comtrade-bench-blog"><img src="https://img.shields.io/badge/Blog-Technical%20Write--up-6366f1" alt="Blog"/></a>
  &nbsp;
  <img src="https://img.shields.io/badge/OpenEnv-Native-4B8BBE" alt="OpenEnv"/>
  &nbsp;
  <img src="https://img.shields.io/badge/Tasks-10-brightgreen" alt="10 Tasks"/>
  &nbsp;
  <img src="https://img.shields.io/badge/Training-GRPO-orange" alt="GRPO"/>
</p>

# ComtradeBench

### An OpenEnv Benchmark for Reliable LLM Tool-Use Under Adversarial API Conditions

ComtradeBench is a ten-task [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment that
measures **execution reliability** of LLM agents in a realistic API workflow. The domain is trade
data retrieval; the benchmark is about whether an agent can handle the failure modes that appear in
every production API at scale — pagination drift, duplicate records, transient errors, misleading
summary rows, and constrained request budgets.

The environment is **adversarial by design**: fault injection, non-stationary dynamics, and
multi-dimensional scoring reward correct execution, not fluent output.

**AgentBeats Phase 2 — OpenEnv Challenge** | Author: MateFin
[GitHub](https://github.com/yonghongzhang-io/comtrade-openenv) ·
[Env Space](https://huggingface.co/spaces/yonghongzhang/comtrade-env) ·
[Blog](https://huggingface.co/spaces/yonghongzhang/comtrade-bench-blog)

---

## Results at a glance

| Agent | Avg (T1-T10) | Beats baseline on | T9 score | Notes |
|---|---:|:---:|---:|---|
| Rule-based baseline | **96.8 / 100** | reference | 96.9 | deterministic, no LLM |
| **Kimi / Moonshot V1 (128k)** | **97.5 / 100** | **10 of 10** | **97.5** | same model on all 10 tasks, apples-to-apples |
| Llama 3.3 70B (Groq) | 89.3 / 100 | 0 of 10 | **18.7** | matches Kimi on T1-T8 but collapses on T9 |
| GRPO (reward-only, Qwen2.5-7B) | see curve | — | — | beat rule-based baseline mean in 6 of 8 training iterations |

**T9 (adaptive adversary)** produces the sharpest discriminative signal in the benchmark:
Kimi 97.5 vs Llama 18.7 — a **78.8-point gap** on the same task with the same prompt and agent loop.
This validates T9 as a genuine stress test rather than a cosmetic hard task.

The same environment code runs in-process during GRPO rollouts and as a deployed Docker service
during eval, with zero divergence. Context-vs-prompt ablation on T4/T5 is in the Results section below.

---

## What makes this benchmark different

Most API-task benchmarks evaluate whether an agent retrieves the correct answer from a clean API.
ComtradeBench evaluates whether the agent executes correctly when the API actively resists correct
execution:

- `T3`, `T8`: cross-page duplicate records can overcount rows and inflate trade totals.
- `T4`, `T8`: HTTP 429 rate limits can create missing pages if the agent advances too early.
- `T5`: HTTP 500 transient failures can leave silent data gaps when retry is skipped.
- `T6`: non-deterministic page ordering breaks agents that assume stable row position.
- `T7`: synthetic totals rows (`is_total=true`) contaminate aggregates unless filtered.
- `T9`: adaptive fault escalation tests whether policy still holds under mid-episode shift.
- `T10`: a halved request budget exposes redundant fetches and incomplete retrieval plans.

The agent has three MCP tools and 100 requests. The six-dimensional judge scores correctness,
completeness, robustness, efficiency, data quality, and observability. There is no partial credit
for correct-sounding output from an incorrect execution.

## Project Structure

```
comtrade_env/
├── README.md                    # This file
├── blog_post.md                 # Submission blog post
├── openenv.yaml                 # OpenEnv manifest
├── pyproject.toml               # Environment dependencies
├── Dockerfile                   # Container image
├── __init__.py                  # Module exports
├── client.py                    # ComtradeEnv HTTP/WebSocket client
├── models.py                    # ComtradeAction / ComtradeObservation
├── server/                      # Environment + mock service
│   ├── app.py                   # FastAPI app (HTTP + WebSocket)
│   ├── comtrade_env_environment.py  # Core MCP environment logic
│   ├── tasks.py                 # Task definitions (T1–T10)
│   ├── judge.py                 # Scoring engine (6 dimensions)
│   ├── mock_service/            # Embedded mock Comtrade API
│   │   ├── app.py               # FastAPI mock with fault injection
│   │   └── fixtures/            # Ground-truth data (seeded RNG)
│   ├── Dockerfile               # Server container image
│   └── requirements.txt
├── green/                       # Green Agent (A2A evaluator for AgentBeats)
│   ├── agent_a2a.py             # A2A server (JSON-RPC 2.0)
│   ├── judge_green.py           # Scoring engine
│   ├── tasks_green.py           # Task definitions
│   └── Dockerfile               # Green agent container
└── agent/                       # LLM training agent
    ├── agent.py                 # LLM-powered agentic loop
    ├── env_client.py            # InProcessEnvClient (no HTTP needed)
    ├── train_grpo.py            # GRPO training pipeline
    ├── smoke_test.py            # Rule-based smoke test (no LLM)
    ├── direct_test.py           # Direct environment test
    ├── inference.py             # Inference script
    ├── plot_training.py         # Training curve visualisation
    └── tests/
        └── test_comtrade.py     # Unit + integration tests
```

## Tasks (T1–T10)

| ID | Name | Challenge |
|----|------|-----------|
| T1 | Single page | Fetch one page, submit. Baseline correctness. |
| T2 | Multi-page pagination | Iterate pages until `has_more=False`. |
| T3 | Deduplication | Pages overlap; agent must dedup by primary key. |
| T4 | HTTP 429 retry | Rate-limit fault injection; retry without data loss. |
| T5 | HTTP 500 retry | Server error fault; retry transient failures. |
| T6 | Page drift | Non-deterministic page ordering; handle instability. |
| T7 | Totals trap | Summary rows mixed in; drop `is_total=true` rows. |
| T8 | Mixed faults | 429 rate-limit + cross-page duplicates simultaneously. |
| **T9** | **Adaptive adversary** | **Faults escalate mid-episode based on agent progress.** |
| **T10** | **Constrained budget** | **Single agent runs under halved request budget.** |

## MCP Tools

```
get_task_info()       → task description, query params, request budget
fetch_page(page, page_size)  → {rows, page, total_pages, has_more}
submit_results(data_jsonl, metadata_json, run_log)  → {reward, score, breakdown}
```

## Scoring (0–100 → reward 0.0–1.0)

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| Correctness | 30 | All expected rows present and correct |
| Completeness | 15 | No missing records |
| Robustness | 15 | Correct handling of 429/500 faults |
| Efficiency | 15 | Request count relative to minimum needed |
| Data Quality | 15 | No duplicates, no totals rows leaked |
| Observability | 10 | `run.log` contains required fields |

## Quick Start

### 1. Smoke Test (no LLM required)

```bash
cd comtrade_env

# Install OpenEnv framework (if not already)
pip install openenv-core[core]

# Run rule-based agent on one task
python agent/smoke_test.py --task T1_single_page

# Run all tasks
for t in T1_single_page T2_multi_page T3_duplicates \
         T4_rate_limit_429 T5_server_error_500 T6_page_drift T7_totals_trap \
         T8_mixed_faults T9_adaptive_adversary T10_constrained_budget; do
    python agent/smoke_test.py --task $t
done
```

### 2. Run Tests

```bash
cd comtrade_env
pip install pytest
python -m pytest agent/tests/ -v
```

### 3. GRPO Training

```bash
cd comtrade_env

# Install agent dependencies
pip install torch transformers accelerate peft trl openai requests fastmcp fastapi uvicorn

# Using a local Ollama/vLLM endpoint (rollout-only, no gradient updates)
python agent/train_grpo.py \
    --api-url http://localhost:11434/v1 \
    --api-model qwen2.5:7b \
    --num-iterations 200 \
    --batch-size 4 \
    --group-size 4

# Using a HuggingFace model (full GRPO training with gradients)
python agent/train_grpo.py \
    --hf-model Qwen/Qwen2.5-7B-Instruct \
    --num-iterations 200
```

No external OpenEnv server is needed — `InProcessEnvClient` runs the environment in-process.

### 4. Run the OpenEnv Server (Docker)

```bash
cd comtrade_env
docker build -t comtrade-env:latest -f server/Dockerfile .
docker run -p 8000:8000 comtrade-env:latest
```

### 5. Deploy to Hugging Face Spaces

```bash
# Auto-uploads README, Dockerfile, server/, green/, blog, images, results JSONs.
# Uses `hf upload` so LFS is handled without a local git-lfs install.
bash deploy_hf.sh
```

Or, from scratch with the OpenEnv CLI:

```bash
openenv push --repo-id <your-hf-org>/comtrade-env
```

## Key Design Decisions

- **Same env code in training and eval.** Rollouts use `InProcessEnvClient`, eval uses the Docker Space. Both construct the identical `ComtradeEnvironment` instance, so training conditions and judged conditions never diverge.
- **Episode isolation across concurrent rollouts.** The embedded mock service keys state by `(task_id, episode_id)`, so parallel GRPO workers never corrupt each other's data even though they share one service.
- **Procedural fixtures, not recorded data.** All 10 tasks are generated from a seeded PRNG. No external API dependency, no fixture drift, full reproducibility from a task ID plus seed.
- **Scoring aligned to training signal.** The six-dimensional judge emits a scalar reward that matches the same breakdown used for eval, so GRPO optimises directly against the evaluation metric rather than a proxy.

## Results

### Rule-Based Baseline (no LLM)

| Task | Score | Reward |
|------|-------|--------|
| T1 Single page | 98.0 | 0.980 |
| T2 Multi-page | 98.0 | 0.980 |
| T3 Duplicates | 98.0 | 0.980 |
| T4 Rate limit | 95.0 | 0.950 |
| T5 Server error | 95.7 | 0.957 |
| T6 Page drift | 94.0 | 0.940 |
| T7 Totals trap | 98.0 | 0.980 |
| T8 Mixed faults | 96.4 | 0.964 |
| T9 Adaptive adversary | 96.9 | 0.969 |
| T10 Constrained budget | 98.0 | 0.980 |
| **Average** | **96.8** | **0.968** |

![Benchmark Results](benchmark_results.png)
*Rule-based baseline vs. Kimi LLM agent across the 10-task suite.*

### GRPO Training Curve (8 iterations, real LLM)

![Training Curve](training_curve.png)

### LLM Agent — Kimi / Moonshot V1-128k (apples-to-apples across all 10 tasks)

All 10 tasks run under the same `moonshot-v1-128k` variant, `temperature=0.0`, `seed=42`. See
`llm_results_kimi.json` for the full breakdown including per-dimension sub-scores.

| Task | Score | Reward | Delta vs baseline (pts) |
|------|-------|--------|-------------------------|
| T1 Single page | 98.7 | 0.987 | +0.7 |
| T2 Multi-page | 98.7 | 0.987 | +0.7 |
| T3 Duplicates | 98.7 | 0.987 | +0.7 |
| T4 Rate limit (429) | 95.7 | 0.957 | +0.7 |
| T5 Server error (500) | 96.3 | 0.963 | +0.6 |
| T6 Page drift | 94.7 | 0.947 | +0.7 |
| T7 Totals trap | 98.7 | 0.987 | +0.7 |
| T8 Mixed faults | 97.3 | 0.973 | +0.9 |
| T9 Adaptive adversary | 97.5 | 0.975 | +0.6 |
| T10 Constrained budget | 98.7 | 0.987 | +0.7 |
| **Average (T1-T10)** | **97.5** | **0.975** | **+0.7** |

Kimi-128k matches or slightly exceeds the rule-based baseline on **all 10 tasks**. The remaining
gap on T4/T5 Robustness (12/15, not 15/15) is a scoring sub-criterion explored in the ablation
below, not a silent-retry failure.

### Cross-model comparison — T9 is genuinely discriminative

| Model | Avg (T1-T10) | T1-T8 avg | T9 score | T10 score |
|---|---:|---:|---:|---:|
| Rule-based baseline | 96.8 | 96.5 | 96.9 | 98.0 |
| Kimi Moonshot V1-128k | **97.5** | **97.4** | **97.5** | **98.7** |
| Llama 3.3 70B (Groq) | 89.3 | 97.4 | **18.7** | 95.7 |

Two frontier-class LLMs perform nearly identically on T1-T8 (97.4 each) but diverge catastrophically
on T9: Kimi scores 97.5, Llama scores 18.7 — a **78.8-point gap on the same task**. Llama handles
static faults (429/500/duplicates/drift) but fails on within-episode fault escalation. This is the
first empirical confirmation that **T9 produces meaningful separation between models**, not just
harder numbers on the same axis. Full per-task breakdown in `llm_results_llama.json`.

### Ablation — does context or prompt engineering drive T4/T5 Robustness?

We originally claimed the T4/T5 (HTTP 429 / 500) Robustness gap could be closed with an EVENTS
scratchpad prompt pattern. The data says otherwise. Three conditions on Kimi (same model family,
same agent loop, same seed):

| Condition | Context | Prompt | T4 Robustness | T5 Robustness |
|---|---|---|---:|---:|
| A | 8k | default | 0 / 15 | 0 / 15 |
| B | 128k | default | 12 / 15 | 12 / 15 |
| C | 128k | EVENTS scratchpad (enhanced) | 12 / 15 | 12 / 15 |

**A → B (context effect):** +12 Robustness on both tasks just from enlarging the context window.
**B → C (prompt effect):** zero additional gain from explicit EVENTS instructions.

The original T4/T5 = 0 Robustness result was not a narration failure — it was a context-truncation
failure. At 8k, the retry narration fell off the back of the buffer before it could land in
`run_log`. At 128k, the same prompt captures everything. Adding explicit EVENTS scaffolding on top
changes nothing, because the model already logs adequately when it has room to.

**Takeaway for agent builders:** on tool-use benchmarks with long trajectories, **size the context
to the episode length before reaching for prompt engineering**. A prompt cannot recover narration
that was never written because the buffer filled up. Full data in `ablation_context_vs_prompt.json`.

### Scoring weight rationale

The six-dimensional rubric is weighted 30/15/15/15/15/10. The design principle is that **correctness
is necessary but not sufficient** — so Correctness gets the largest single weight (30), but the
combined weight of "execution quality under adversity" dimensions (Completeness + Robustness +
Efficiency + Data Quality = 60) exceeds Correctness. This forces scoring to reward agents that do
the job right, not just return something plausible. Observability at 10 is intentionally lower
than the execution dimensions: it's an audit requirement rather than a core task, but it's not
zero because an un-auditable pipeline is not a production-ready pipeline.

### How ComtradeBench compares to existing tool-use benchmarks

| Benchmark | Adversarial faults in env | Within-episode non-stationarity | Multi-dim execution scoring | Budget constraints |
|---|:---:|:---:|:---:|:---:|
| ToolBench (Qin et al., 2023) | — | — | — | — |
| τ-bench (Sierra / Anthropic) | partial (policy violations) | — | ✓ (pass@k on policies) | — |
| BFCL (Berkeley) | — | — | — | — |
| API-Bank | — | — | — | — |
| **ComtradeBench** | **✓** (429/500/drift/dupes/totals) | **✓** (T9) | **✓** (6 dimensions) | **✓** (T10) |

Closest relative is τ-bench — it also scores beyond "did the final answer match" and injects
policy-level adversarial conditions. ComtradeBench's unique combination is **environment-level
fault injection + within-episode escalation (T9) + budget-aware rollouts (T10)**. The adversarial
bits are not in the prompts or the labels — they are in the environment, so an agent cannot route
around them by rephrasing.

### Limitations and next steps

These are the specific things this release does not yet do:

- **T9 calibration is one-sided.** T9 sharply separates Llama from Kimi, but does not yet produce
  fine-grained separation among frontier models (Kimi and the rule-based baseline both clear ~97).
  A harder T9 variant with steeper mid-episode escalation would differentiate strong models further.
- **T4/T5 Robustness ceiling at 12/15.** Neither a larger context nor an explicit EVENTS prompt
  pushes past 12 on retry-heavy tasks. The remaining 3 points correspond to a retry-count or
  retry-timing fidelity sub-check we have not yet fully diagnosed; future work is to make that
  sub-criterion explicit in the judge.
- **Two LLMs evaluated.** Kimi (Moonshot V1-128k) and Llama 3.3 70B. Adding GPT-4o, Claude 4.6
  Sonnet, and Qwen2.5-72B would strengthen the cross-model story.
- **GRPO training at PoC scale only.** 8 iterations is a sanity-check run, not a full training
  study. Extending to 50-200 iterations on a held-out task split (e.g. T1-T8 train, T9-T10 test)
  would convert the pipeline from "plumbing" to "experiment."
- **Benchmark comparison is qualitative.** We describe the feature matrix vs. τ-bench / BFCL /
  ToolBench but have not yet run the same LLM across all four benchmarks side-by-side.
- **Single-seed evaluation.** All LLM runs use `seed=42`. Multi-seed robustness intervals would
  quantify variance.

## License

Environment code follows the OpenEnv BSD-style license.
Agent training code is provided as-is for the AgentBeats competition.
