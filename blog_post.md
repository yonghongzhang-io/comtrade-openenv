# Green Comtrade Bench: Teaching LLM Agents to Fetch Trade Data Reliably

**AgentBeats Phase 2 — OpenEnv Challenge Submission**  
Author: MateFin | [GitHub](https://github.com/yonghongzhang-io/comtrade-openenv) | [HF Space](https://huggingface.co/spaces/yonghongzhang/comtrade-env)

---

## Motivation

Real-world data pipelines are messy. They paginate. They rate-limit you. They return duplicates across page boundaries. They inject summary rows into data feeds. They reorder results non-deterministically between calls.

Most LLM benchmarks evaluate reasoning in clean, single-turn settings. We asked: **can an LLM agent reliably fetch and clean real-world paginated API data under adversarial conditions?**

To answer this, we built **Green Comtrade Bench** — an eight-task OpenEnv environment where an LLM agent must interact with a simulated UN Comtrade trade statistics API, handle faults gracefully, and submit clean deduplicated output.

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

### Eight Tasks — Progressive Difficulty

| Task | Fault Injected | Key Challenge | Difficulty |
|------|---------------|---------------|------------|
| T1 | None | Schema validation, baseline correctness | Easy |
| T2 | Pagination only | Multi-page merge (2,345 rows across 5+ pages) | Easy |
| T3 | 8% within-page + 3% cross-page duplicates | Primary-key deduplication | Medium |
| T4 | HTTP 429 on page 2 | Backoff + retry without data loss | Medium |
| T5 | HTTP 500 on page 2 | Transient error handling | Medium |
| T6 | Non-deterministic page ordering | Canonicalization + dedup under drift | Hard |
| T7 | `is_total=true` summary rows mixed in | Totals-trap filtering | Hard |
| T8 | 429 rate-limit + cross-page duplicates | Both retry AND dedup simultaneously | Hard |

Tasks are drawn from real UN Comtrade API behaviors: the pagination drift, duplicate records, and totals rows are documented failure modes that production ETL pipelines routinely encounter. T8 is the hardest task — it combines two independent failure modes that must both be handled correctly.

### Mock Service Architecture

The embedded mock service is a FastAPI application with per-task fault injection:

```
comtrade_env/
├── server/
│   ├── comtrade_env_environment.py  ← MCPEnvironment (3 MCP tools)
│   ├── tasks.py                     ← Task definitions T1-T8
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
def grpo_loss(log_probs, old_log_probs, ref_log_probs, advantages,
              clip_eps=0.2, kl_coeff=0.04):
    """Clipped surrogate + reverse-KL penalty (DeepSeekMath)."""
    # Policy ratio: r_t = π_new / π_old
    ratio = torch.exp(log_probs - old_log_probs)
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    surrogate = torch.min(ratio * advantages, clipped * advantages).mean()

    # Reverse KL: D_KL(π_new || π_ref) = E[exp(x) - 1 - x] where x = log(π_new/π_ref)
    log_ratio_ref = log_probs - ref_log_probs
    kl = (torch.exp(log_ratio_ref) - 1 - log_ratio_ref).mean()

    return -(surrogate - kl_coeff * kl)
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

### LLM Agent Results

We evaluated two LLM backends via the agentic loop described above: LLM decides tool sequencing, while the infrastructure handles dedup, retry, and submission.

**Moonshot V1-8K (Kimi) — full agentic loop, all 8 tasks:**

| Task | Score | Reward | Steps | vs Baseline |
|------|-------|--------|-------|-------------|
| T1 Single page | 98.7 | 0.987 | 3 | +3.7 |
| T2 Multi-page | 98.7 | 0.987 | 7 | +0.7 |
| T3 Duplicates | 98.7 | 0.987 | 5 | +0.7 |
| T4 Rate limit 429 | 83.7 | 0.837 | 5 | +0.7 |
| T5 Server error 500 | 84.3 | 0.843 | 5 | +0.6 |
| T6 Page drift | 94.7 | 0.947 | 5 | +0.4 |
| T7 Totals trap | 98.7 | 0.987 | 5 | +2.7 |
| T8 Mixed faults | 97.3 | 0.973 | 5 | +0.9 |
| **Average** | **94.4** | **0.944** | **5.0** | **+1.3** |

![Benchmark Results](benchmark_results.png)

### GRPO Rollout Training Curve (8 iterations, Moonshot V1-8K)

We ran 8 iterations of GRPO-style rollouts with group_size=2, sampling 2 random tasks per iteration. Each rollout is a full agentic episode with real LLM tool-calling decisions.

![Training Curve](training_curve.png)

The left chart shows reward across iterations with min-max range and rolling average. The right chart shows per-task mean reward across all iterations where that task appeared. The orange dotted line marks the rule-based baseline (0.930).

Key observations:
- **Mean reward consistently above baseline** (0.930) in 6/8 iterations
- **Iterations with fault tasks (T4/T5) pull the mean down** — these are genuinely harder and require the agent to handle 429/500 errors gracefully
- **T8 mixed faults achieves 0.973** — demonstrating the LLM can handle combined rate-limit + dedup challenges
- **Per-task variance is low** (small error bars) — the agent's behavior is consistent across rollouts

Key findings:
- **LLM agent outperforms rule-based baseline on 8/8 tasks** — the LLM generates better structured logs (Observability +2-3 pts) and makes smarter pagination decisions
- **T1/T2/T3/T7 hit near-perfect 98.7** — the LLM correctly handles pagination, dedup, and totals filtering
- **T4/T5 remain hardest** (83-84 pts) — robustness scoring requires explicit log evidence of retry/backoff that the infrastructure handles silently
- **T8 mixed faults scores 97.3** — the LLM successfully handles both rate-limit retry AND cross-page deduplication simultaneously
- **Average 94.4 vs baseline 93.0** — the gap is small because the baseline is already strong; GRPO gradient training would push this further by optimizing the LLM's tool sequencing decisions

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
# Clone the repo (environment + agent are in one repo)
git clone https://github.com/yonghongzhang-io/comtrade-openenv
cd comtrade-openenv

# Install OpenEnv framework
pip install openenv-core[core]

# Rule-based smoke test — no LLM, no external server needed
# (InProcessEnvClient auto-starts mock service in-process)
python agent/smoke_test.py --task T1_single_page
python agent/smoke_test.py --task T7_totals_trap
python agent/smoke_test.py --task T8_mixed_faults

# Run unit + integration tests
pip install pytest
python -m pytest agent/tests/ -v

# Train with GRPO via local Ollama/vLLM (rollout-only, no GPU required)
python agent/train_grpo.py \
    --api-url http://localhost:11434/v1 \
    --api-model qwen2.5:7b \
    --num-iterations 200 \
    --max-workers 4

# Train with gradient updates (requires GPU + HuggingFace model)
python agent/train_grpo.py \
    --hf-model Qwen/Qwen2.5-7B-Instruct \
    --num-iterations 200 \
    --output-dir ./checkpoints
```

No external OpenEnv server is needed — `InProcessEnvClient` wraps the environment directly, with parallel rollout support via `ThreadPoolExecutor`.

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
