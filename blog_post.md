# Green Comtrade Bench: Teaching LLM Agents to Fetch Trade Data Reliably

**AgentBeats Phase 2 — OpenEnv Challenge Submission**  
Author: Yonghong Zhang | [GitHub](https://github.com/yonghongzhang-io/comtrade-openenv) | [HF Space](https://huggingface.co/spaces/yonghongzhang/comtrade-env)

---

## Motivation

Real-world data pipelines are messy. They paginate. They rate-limit you. They return duplicates across page boundaries. They inject summary rows into data feeds. They reorder results non-deterministically between calls.

Most LLM benchmarks evaluate reasoning in clean, single-turn settings. We asked: **can an LLM agent reliably fetch and clean real-world paginated API data under adversarial conditions?**

To answer this, we built **Green Comtrade Bench** — a seven-task OpenEnv environment where an LLM agent must interact with a simulated UN Comtrade trade statistics API, handle faults gracefully, and submit clean deduplicated output.

---

## Environment Design

### The Task

The agent is given a trade data query (reporter country, partner country, trade flow, HS product code, year). It must:

1. Discover pagination bounds via the API
2. Fetch all pages until `has_more=False`
3. Deduplicate records by primary key `(year, reporter, partner, flow, hs, record_id)`
4. Drop summary rows (`is_total=true`)
5. Submit a JSONL file with clean data + metadata + execution log

The agent has a budget of 100 requests per episode.

### Three MCP Tools

The environment exposes exactly three tools via the Model Context Protocol (MCP):

```
get_task_info()
  → Returns task parameters, mock service URL, and request budget.

fetch_page(page: int, page_size: int = 500)
  → Fetches one page. Returns {rows, page, total_pages, has_more}.
    On fault: {status: 429|500, retry: true}

submit_results(data_jsonl, metadata_json, run_log)
  → Scores the submission. Returns {reward, score, breakdown, errors}.
```

This minimal interface mirrors how real API agents are constrained: the agent cannot inspect internal state, cannot bypass pagination, and cannot retry with a fresh session.

### Seven Tasks — Progressive Difficulty

| Task | Fault Injected | Key Challenge |
|------|---------------|---------------|
| T1 | None | Schema validation, baseline correctness |
| T2 | Pagination only | Multi-page merge (2,345 rows across 5+ pages) |
| T3 | 8% within-page + 3% cross-page duplicates | Primary-key deduplication |
| T4 | HTTP 429 on page 2 | Backoff + retry without data loss |
| T5 | HTTP 500 on page 2 | Transient error handling |
| T6 | Non-deterministic page ordering | Canonicalization + dedup under drift |
| T7 | `is_total=true` summary rows mixed in | Totals-trap filtering |

Tasks are drawn from real UN Comtrade API behaviors: the pagination drift, duplicate records, and totals rows are documented failure modes that production ETL pipelines routinely encounter.

### Mock Service Architecture

The embedded mock service is a FastAPI application with per-task fault injection:

```
comtrade_env/
├── server/
│   ├── comtrade_env_environment.py  ← MCPEnvironment (3 MCP tools)
│   ├── tasks.py                     ← Task definitions T1-T7
│   ├── judge.py                     ← Scoring engine
│   └── mock_service/
│       └── app.py                   ← Stateless /api/data with fault injection
```

The mock service is **stateless**: each request reconstructs the response from task configuration + request parameters. This makes the environment reproducible and concurrent-safe — multiple agents can run simultaneously without shared state corruption.

### Scoring (0–100 → reward 0.0–1.0)

The judge evaluates six dimensions:

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| Correctness | 30 | Row-level accuracy (content + count) |
| Completeness | 15 | Zero missing records |
| Robustness | 15 | Correct fault handling (429/500 retry) |
| Efficiency | 15 | Request count vs. task baseline |
| Data Quality | 15 | No duplicates leaked, no totals rows |
| Observability | 10 | Log contains `task_id=`, `page=`, `request=`, `complete=` |

**Governance rules prevent gaming:**
- Efficiency and Observability points are capped at 50% if Correctness < 70%
- Efficiency points require 100% Completeness — you cannot skip pages and claim efficiency
- Execution time > 45s incurs a penalty (max 3 points)

---

## LLM Agent Design

### Agentic Loop

The agent (`llm_agent/agent.py`) runs a standard tool-use loop:

```
SYSTEM_PROMPT + task description
        ↓
  LLM generates <tool_call>{...}</tool_call>
        ↓
  Environment executes tool
        ↓
  <tool_result>{...}</tool_result> appended to context
        ↓
  repeat until submit_results called
```

Tool calls use a lightweight XML format that works with any instruction-tuned model:

```xml
<tool_call>{"name": "fetch_page", "arguments": {"page": 1}}</tool_call>
```

The agent handles the protocol details (deduplication, retry on 429/500, totals filtering) in its loop logic, not by prompting the model to implement them. This keeps the model focused on **sequencing decisions** (which page to fetch next, when to submit) while the infrastructure handles correctness invariants.

### Fault Handling

```python
# Retry on transient faults
if tool_result.get("status") in (429, 500) or tool_result.get("retry"):
    wait = 2 * (retry_count + 1)
    time.sleep(wait)
    tool_result = self.env.call_tool(tool_name, tool_args)

# Dedup + totals filter on every fetch_page
for row in tool_result["rows"]:
    if row.get("is_total"):
        continue
    pk = "|".join(str(row.get(k, "")) for k in
                 ("year", "reporter", "partner", "flow", "hs", "record_id"))
    collected_rows[pk] = row  # dict assignment = automatic dedup
```

### Backend Flexibility

The `LLMBackend` class supports two modes:

```python
# Local HuggingFace model
backend = LLMBackend.from_hf("Qwen/Qwen2.5-7B-Instruct")

# OpenAI-compatible API (vLLM, Ollama, Together, etc.)
backend = LLMBackend.from_api("http://localhost:11434/v1", "qwen2.5:7b")
```

---

## GRPO Training

We implement **Group Relative Policy Optimization** (GRPO, from DeepSeekMath) to train the agent purely from environment reward signals — no human-labeled data needed.

### Why GRPO for Agentic Tasks

Standard RLHF requires a separate reward model. GRPO replaces it with **group-relative normalization**: run `G` episodes per task, compute each episode's advantage as `(reward - group_mean) / group_std`. This:

- Eliminates reward model training overhead
- Naturally handles sparse rewards (most steps get reward only at episode end)
- Scales to long multi-turn trajectories without value function estimation

### Implementation (`llm_agent/train_grpo.py`)

```python
def grpo_loss(log_probs, advantages, old_log_probs, ref_log_probs,
              clip_eps=0.2, kl_coeff=0.04):
    """Clipped surrogate + KL penalty."""
    ratio = torch.exp(log_probs - old_log_probs)
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    pg_loss = -torch.min(ratio * advantages, clipped * advantages).mean()

    kl = (log_probs - ref_log_probs).mean()
    return pg_loss + kl_coeff * kl
```

Training loop:
1. **Rollout phase**: run `G=4` episodes per task using current policy
2. **Advantage computation**: `A_i = (r_i - mean_group) / (std_group + 1e-8)`
3. **Policy update**: minimize GRPO loss over all trajectory tokens
4. **Checkpoint**: save every 50 iterations; monitor per-task reward

### Key Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `clip_eps` | 0.2 | Standard PPO clip; prevents large policy jumps |
| `kl_coeff` | 0.04 | Light KL penalty; allows exploration |
| `group_size` | 4 | 4 rollouts per task per iteration |
| `lr` | 1e-5 | Conservative for fine-tuning |
| `max_steps` | 30 | Sufficient for all T1-T7 tasks |

---

## Results

### Rule-Based Baseline (no LLM)

The deterministic baseline agent in `smoke_test.py` achieves high scores on all tasks, validating the environment and scoring machinery end-to-end:

| Task | Score | Reward | Breakdown |
|------|-------|--------|-----------|
| T1 single page | 95.0 | 0.9500 | corr=30 comp=15 robu=12 effi=15 data=15 obs=8 |
| T2 multi-page | 98.0 | 0.9800 | corr=30 comp=15 robu=15 effi=15 data=15 obs=8 |
| T3 duplicates | 98.0 | 0.9800 | corr=30 comp=15 robu=15 effi=15 data=15 obs=8 |
| T4 rate-limit 429 | 83.0 | 0.8300 | corr=30 comp=15 robu=0 effi=15 data=15 obs=8 |
| T5 server error 500 | 83.7 | 0.8370 | corr=30 comp=15 robu=0 effi=15 data=15 obs=8.7 |
| T6 page drift | 94.3 | 0.9430 | corr=26.3 comp=15 robu=15 effi=15 data=15 obs=8 |
| T7 totals trap | 96.0 | 0.9600 | corr=28 comp=15 robu=15 effi=15 data=15 obs=8 |
| **Average** | **92.6** | **0.9257** | |

All scores from `inference.py --mode rule-based` (deterministic, no LLM, reproducible). Full breakdown available in `inference_results_baseline.json`.

### LLM Agent Results (Moonshot V1-8K via GRPO Rollouts)

We ran 8 iterations of GRPO rollouts using Moonshot V1-8K (Kimi) as the LLM backend. The agent uses the agentic loop described above: LLM decides tool sequencing, while the infrastructure handles dedup, retry, and submission.

| Iteration | Mean Reward | Max Reward | Tasks Evaluated |
|-----------|-------------|------------|-----------------|
| 1 | 0.987 | 0.987 | T3, T1 |
| 2 | 0.967 | 0.987 | T6, T2 |
| 3 | 0.902 | 0.967 | T4, T7 |
| 4-8 | 0.912-0.987 | 0.987 | Mixed |

Key findings:
- **LLM agent achieves 0.987 reward on simple tasks** (T1, T2, T3) — matching or exceeding the rule-based baseline on Observability (the LLM naturally generates structured logs)
- **Fault tasks score lower** (T4: 0.837, T6: 0.947) — the LLM sometimes wastes request budget on unnecessary retries
- **Average reward across all iterations: 0.964** — strong performance with zero task-specific fine-tuning

### What the Scoring Reveals

The rule-based baseline loses points on two dimensions:

- **Observability**: the run log requires specific structured entries (`task_id=`, `page=N`, `request=N`, `complete=true`); a naive agent that omits these loses up to 10 points
- **Efficiency**: fault-injection tasks (T4/T5/T6) require one or more retries, consuming extra request budget against the task baseline

The LLM agent improves on Observability (naturally verbose logs) but sometimes regresses on Efficiency (unnecessary fetches). This trade-off is exactly what GRPO gradient training would optimize: with a local HuggingFace model, the clipped surrogate loss would push the policy toward efficient tool sequences while the KL penalty prevents forgetting correct pagination behavior.

---

## OpenEnv Integration

The environment follows the OpenEnv contract exactly:

```python
class ComtradeEnvironment(MCPEnvironment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True  # parallel training episodes

    def reset(self, task_id=None, seed=None, **kwargs) -> Observation: ...
    def _step_impl(self, action: Action, **kwargs) -> Observation: ...
```

Agents interact via MCP tools, never via direct method calls. The reward is computed entirely inside the environment — the agent cannot inspect or manipulate the judge. This aligns with OpenEnv's core invariant: *rewards inside environment, not external*.

The mock service starts as an embedded subprocess on `reset()` and is torn down with the environment, making each Docker container self-contained.

---

## Running the Environment

```bash
# Clone both repos side-by-side
git clone https://github.com/meta-pytorch/OpenEnv
git clone https://github.com/yonghongzhang-io/comtrade-openenv

# Install OpenEnv dependencies
cd OpenEnv && uv sync && cd ..

# Rule-based smoke test — no LLM, no external server needed
# (auto-discovers OpenEnv, spins up mock service in-process)
cd comtrade-openenv/llm_agent
python smoke_test.py --task T1_single_page
python smoke_test.py --task T2_multi_page
python smoke_test.py --task T7_totals_trap

# In-process integration test (same approach, more verbose output)
python direct_test.py --task T3_duplicates

# Start the OpenEnv HTTP server (needed for LLM agent / GRPO training)
cd ../..
cd OpenEnv && uvicorn envs.comtrade_env.server.app:app --port 8000 &

# Train with GRPO via local Ollama/vLLM (rollout-only mode, no GPU required)
cd ../comtrade-openenv/llm_agent
python train_grpo.py \
    --env-url http://localhost:8000 \
    --api-url http://localhost:11434/v1 \
    --api-model qwen2.5:7b \
    --num-iterations 200

# Train with gradient updates (requires GPU + HuggingFace model)
python train_grpo.py \
    --env-url http://localhost:8000 \
    --hf-model Qwen/Qwen2.5-7B-Instruct \
    --num-iterations 200 \
    --output-dir ./checkpoints
```

---

## Design Decisions and Lessons Learned

**Stateless mock service is essential.** The first implementation used per-session state in the mock service, which caused race conditions when multiple agents ran concurrently during GRPO rollouts. Switching to stateless `/api/data` with per-task `_API_STATE` dictionaries eliminated the issue entirely.

**Three tools is the right abstraction.** Early prototypes had separate tools for setting query parameters and for pagination. Collapsing to `get_task_info` + `fetch_page` + `submit_results` reduced token overhead and made the tool-use pattern easier for the model to learn.

**Protocol-level dedup beats prompt-level dedup.** Telling the model "deduplicate records" in the system prompt is fragile — the model may not track state correctly across long contexts. Instead, the agent loop handles dedup mechanically using a Python dict keyed by primary key. The model only needs to decide *when* to call which tool.

**Observability scoring drives good agent habits.** The 10-point observability dimension, which requires structured log entries (`task_id=`, `page=N`, `request=N`, `complete=true`), incentivizes the agent to maintain explicit execution state. This is valuable beyond scoring: structured logs are how real ETL pipelines are debugged.

---

## Links

- **Environment**: [github.com/yonghongzhang-io/comtrade-openenv](https://github.com/yonghongzhang-io/comtrade-openenv)
- **HF Space**: [huggingface.co/spaces/yonghongzhang/comtrade-env](https://huggingface.co/spaces/yonghongzhang/comtrade-env)
- **Full competition repo**: [github.com/yonghongzhang-io/AIAgentCompetition-phase2](https://github.com/yonghongzhang-io/AIAgentCompetition-phase2)
- **OpenEnv framework**: [github.com/meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv)
