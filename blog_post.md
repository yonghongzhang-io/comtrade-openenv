---
tags:
- benchmark
- tool-use
- openenv
- rl-environment
- adversarial
- grpo
language:
- en
---

<p align="center">
  <img src="banner.png" width="100%" alt="ComtradeBench — An OpenEnv Benchmark for Reliable LLM Tool-Use"/>
</p>

<p align="center">
  <a href="https://github.com/yonghongzhang-io/comtrade-openenv">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?logo=github" alt="GitHub"/>
  </a>
  &nbsp;
  <a href="https://huggingface.co/spaces/yonghongzhang/comtrade-env">
    <img src="https://img.shields.io/badge/HF%20Space-Live%20Demo-FFD21E?logo=huggingface&logoColor=black" alt="HF Space"/>
  </a>
  &nbsp;
  <img src="https://img.shields.io/badge/OpenEnv-Native-4B8BBE" alt="OpenEnv"/>
  &nbsp;
  <img src="https://img.shields.io/badge/Tasks-10-brightgreen" alt="10 Tasks"/>
  &nbsp;
  <img src="https://img.shields.io/badge/Training-GRPO-orange" alt="GRPO"/>
</p>

<p align="center"><em>AgentBeats Phase 2 — OpenEnv Challenge Submission &nbsp;|&nbsp; Author: MateFin</em></p>

---

## Agents should be judged by whether they finish the job

Large language models are often evaluated on what they can **say**.  
Real agents, however, are judged by whether they can **finish the job** when tools fail.

In practical API workflows, failure rarely comes from language alone. Pages drift. Duplicate rows appear across requests. Rate limits interrupt execution. Transient server errors force retries. Summary rows contaminate aggregates. Budgets make brute-force strategies impossible.

These are not unusual edge cases. **They are normal operating conditions for production systems.**

ComtradeBench is an OpenEnv benchmark designed to measure exactly this problem: can an LLM agent execute a multi-step API workflow reliably under realistic failure modes?

---

## Why this benchmark matters

Many current evaluations still focus on final answers, clean tool calls, or static environments. But deployed agents fail for more operational reasons:

| Failure | What goes wrong |
|---------|----------------|
| Miss pages | Incomplete data submitted as complete |
| Retry incorrectly | Page skipped after error — silent data gap |
| Double-count duplicates | Overcounted rows, inflated aggregates |
| Leak summary rows | Contaminated totals corrupt downstream analysis |
| Waste budget | Redundant fetches exhaust request limit |
| Recover silently | No auditable trace — failure invisible in production |

These are **execution failures**, not just reasoning failures.

If we want useful agents, we need benchmarks that measure reliable task completion under imperfect conditions — not only answer quality in idealized settings.

---

## What ComtradeBench is

> ComtradeBench is an OpenEnv-native benchmark and training environment for reliable tool-use. The domain is trade-data retrieval; the problem is broader: robust multi-step API execution under shifting, imperfect, and partially adversarial conditions.

The environment asks an agent to retrieve, clean, and submit records from a paginated API while handling:

- **Pagination drift** — page ordering randomized between calls
- **Duplicate records** — within-page (8%) and cross-page (3%) overlap
- **Transient errors** — HTTP 429 rate-limits and HTTP 500 server faults
- **Totals trap** — synthetic summary rows mixed into real data
- **Mixed faults** — rate-limit retry + dedup simultaneously
- **Constrained budget** — halved request limit, no room for waste

The goal is not to test whether the agent can *describe* the workflow.  
The goal is to test whether it can *execute* it — correctly, completely, efficiently, and robustly.

---

## Environment design

Each episode gives the agent a parameterized retrieval task and a limited request budget. The agent interacts through **three MCP tools only**:

```
get_task_info()         →  task parameters + request budget
fetch_page(page, size)  →  {rows, has_more}  or  {status: 429|500, retry: true}
submit_results(...)     →  {reward, score, breakdown}
```

The benchmark is structured as a **curriculum of ten tasks**:

| # | Task | Core challenge |
|---|------|----------------|
| T1 | Single page | Baseline correctness |
| T2 | Multi-page pagination | Merge 2,345+ rows across pages |
| T3 | Duplicates | Primary-key deduplication |
| T4 | HTTP 429 | Backoff + retry without data loss |
| T5 | HTTP 500 | Transient error recovery |
| T6 | Page drift | Canonicalize under non-deterministic ordering |
| T7 | Totals trap | Filter `is_total=true` rows |
| T8 | Mixed faults | Retry AND dedup simultaneously |
| **T9** | **Adaptive adversary** | **Fault intensity escalates mid-episode** |
| **T10** | **Constrained budget** | **50 requests instead of 100** |

T9 is, to our knowledge, among the earliest OpenEnv-style tasks to model **within-episode fault escalation** — where the environment becomes harder as the agent makes progress.

---

## Why OpenEnv

We built ComtradeBench on OpenEnv because this benchmark is meant to be more than a one-off simulator.

OpenEnv gives us a standard environment interface, reproducible execution, and clean integration with evaluation and post-training workflows. The same environment code runs both in-process during GRPO training and as a deployed Docker service during evaluation — with no divergence.

Our goal is not only to score agents, but to provide a **reusable environment where robustness can be studied and trained systematically**.

---

## Scoring what actually matters

ComtradeBench uses structured evaluation across **six dimensions** — not a binary pass/fail:

| Dimension | Weight | What it measures |
|-----------|:------:|-----------------|
| Correctness | **30%** | All expected rows present with correct field values |
| Completeness | 15% | Zero missing records |
| Robustness | 15% | Correct fault handling with logged evidence |
| Efficiency | 15% | Request count vs. task-optimal minimum |
| Data Quality | 15% | No duplicates or leaked totals rows |
| Observability | 10% | Structured execution trace in the run log |

**Why multi-dimensional scoring matters:**  
An agent that retrieves correct data but skips retry logging loses 15 points on Robustness. An agent that skips pages to save budget loses Completeness and all Efficiency credit. These behaviors are not equivalent — the benchmark does not treat them as equivalent.

The **Observability** dimension deserves special note: requiring structured log entries incentivizes the agent to maintain explicit execution state. This is not artificial — structured logs are how production ETL pipelines are monitored and debugged.

---

## Baselines and results

### Rule-based baseline (no LLM)

A deterministic rule-based agent achieves **96.8 / 100** average across all ten tasks, confirming the environment is well-calibrated and solvable.

| Task | Score | Reward |
|------|------:|-------:|
| T1 Single page | 98.0 | 0.980 |
| T2 Multi-page | 98.0 | 0.980 |
| T3 Duplicates | 98.0 | 0.980 |
| T4 Rate limit (429) | 95.0 | 0.950 |
| T5 Server error (500) | 95.7 | 0.957 |
| T6 Page drift | 94.0 | 0.940 |
| T7 Totals trap | 98.0 | 0.980 |
| T8 Mixed faults | 96.4 | 0.964 |
| T9 Adaptive adversary | 96.9 | 0.969 |
| T10 Constrained budget | 98.0 | 0.980 |
| **Average** | **96.8** | **0.968** |

### LLM agent — Kimi / Moonshot V1 (full T1-T10 coverage)

T1-T8 were evaluated under `moonshot-v1-8k`. T9 and T10 required a larger context window
due to longer episodes (mid-episode fault escalation and budget-aware rollouts) and were
evaluated under `moonshot-v1-128k`. Both are the same Kimi / Moonshot V1 family at
`temperature=0.0`.

| Task | Score | Reward |
|------|------:|-------:|
| T1 Single page | 98.7 | 0.987 |
| T2 Multi-page | 98.7 | 0.987 |
| T3 Duplicates | 98.7 | 0.987 |
| T4 Rate limit | 83.7 | 0.837 |
| T5 Server error | 84.3 | 0.843 |
| T6 Page drift | 94.7 | 0.947 |
| T7 Totals trap | 98.7 | 0.987 |
| T8 Mixed faults | 97.3 | 0.973 |
| T9 Adaptive adversary | 97.5 | 0.975 |
| T10 Constrained budget | 98.7 | 0.987 |
| **Average (T1-T10)** | **95.1** | **0.951** |

The LLM matches or slightly exceeds the rule-based baseline on **8 of 10 tasks**, including the
two novel hard tasks T9 (adaptive adversary) and T10 (constrained budget). Relative to the
baseline (96.8 avg across the same 10 tasks), the LLM lands at **95.1 (−1.7 pts)** — the
remaining gap is concentrated in T4/T5, where the Robustness dimension penalizes silent retries
rather than correctness gaps. The next section explains how to close that gap with prompt design.

### Why prompt design matters for T4/T5

T4 (HTTP 429) and T5 (HTTP 500) are the tasks where prompt design has the largest effect, and they expose a subtle gap between *doing the right thing* and *being scored for it*. The agent loop already retries faults mechanically, so Correctness stays perfect — but if the model treats `<tool_result>` as transient context and never echoes the fault into its own narration, the recovery happens silently and the judge sees no proof. Up to 15 Robustness points evaporate.

Two prompt-level changes closed most of the gap:

- **Persistent `EVENTS:` scratchpad in the system prompt.** The model is instructed to maintain a running event log in every assistant turn (`page=N status=429 retry=1 wait=2s`). Because the scratchpad is regenerated each turn, it survives context truncation and lands verbatim in `submit_results.run_log` — exactly what the Observability and Robustness scorers grep for.
- **"Log before you retry" framing.** Earlier prompts said *"retry on 429/500"* and the model would silently retry and forget. Reframing as *"first record the fault in EVENTS, then the loop will retry"* turns the fault into a first-class observation rather than an exception to swallow.

The deeper point: T4/T5 are not really testing whether the agent can retry — the loop already does that. They are testing whether the agent's *narration of its own behavior* is faithful enough to be auditable. In production ETL, this is the difference between a pipeline that "worked" and one you can defend in a postmortem.

### GRPO training curve

We ran 8 iterations of GRPO-style rollouts with group-relative advantage normalization. Training signal is reward-only — no human labels, no reward model. Mean reward exceeded the rule-based baseline in **6 of 8 iterations**.

<p align="center">
  <img src="training_curve.png" width="80%" alt="GRPO Training Curve"/>
</p>

---

## What this benchmark reveals

ComtradeBench is designed to expose a gap that clean evaluations often miss: agents can appear capable in idealized settings while remaining brittle under operational noise.

The hardest problems are not "knowing what the API is." They are:

- continuing correctly **after an interruption**
- maintaining data integrity **across many pages**
- adapting when **conditions shift mid-episode**
- balancing **coverage against cost**

This is where reliable agents differ from merely fluent ones.

---

## Benchmark and training substrate

ComtradeBench is not just an evaluation harness — it is built to support agent improvement.

The environment ships with a full **GRPO training pipeline**: reproducible rollouts, group-relative advantage normalization, and reward-only optimization. No human labels needed. No separate reward model.

This is an intentional design choice: if robust tool-use is a real bottleneck for agentic AI, we need environments that can **both measure and train** that capability — with identical conditions in evaluation and training.

---

## Quick start

```bash
# No LLM, no GPU, no API key required
git clone https://github.com/yonghongzhang-io/comtrade-openenv
pip install openenv-core[core]
python agent/smoke_test.py --task T1_single_page
python agent/smoke_test.py --task T9_adaptive_adversary

# GRPO training via local Ollama (CPU-capable)
python agent/train_grpo.py \
    --api-url http://localhost:11434/v1 \
    --api-model qwen2.5:7b \
    --num-iterations 200 --group-size 4
```

All benchmark data is generated procedurally from a seeded PRNG — no external fixtures, no live API dependency. Every result is fully reproducible from a task ID and a random seed.

---

## Conclusion

<p align="center">

---

### 💬 *Can an agent still finish the job when the API fights back?*

---

</p>

That question matters far beyond trade data. It applies to any agent expected to operate against real interfaces with pagination, retries, noisy outputs, and resource limits.

If we want more reliable agents, we need environments that reward reliability directly.  
That is the role ComtradeBench is designed to play.

---

<p align="center">
  <a href="https://github.com/yonghongzhang-io/comtrade-openenv">GitHub</a>
  &nbsp;·&nbsp;
  <a href="https://huggingface.co/spaces/yonghongzhang/comtrade-env">HF Space</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/meta-pytorch/OpenEnv">OpenEnv Framework</a>
</p>
