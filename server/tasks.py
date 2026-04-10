"""Task definitions for the UN Comtrade Benchmark (Green Comtrade Bench).

Tasks T1–T10 cover the full spectrum of real-world data-fetching challenges:

  T1  Single page — baseline correctness + schema validation
  T2  Multi-page pagination — merge multiple pages correctly
  T3  Deduplication — detect and remove duplicate records by primary key
  T4  HTTP 429 retry — exponential backoff on rate-limit errors
  T5  HTTP 500 retry — retry transient server errors
  T6  Page drift — non-deterministic page ordering; canonicalize + dedup
  T7  Totals trap — drop summary rows (is_total=True marker)
  T8  Mixed faults — 429 rate-limit AND cross-page duplicates simultaneously

  Novel tasks (unique to ComtradeBench):
  T9  Adaptive adversary — fault intensity escalates mid-episode based on
                           agent's success; tests robustness under distribution shift
  T10 Constrained budget — single agent must finish under a halved
                           request budget while avoiding redundant fetches

Each Task is a frozen dataclass with:
  task_id       str  — unique identifier (e.g. "T1_single_page")
  description   str  — human-readable task description
  query         dict — Comtrade query params: reporter, partner, flow, hs, year
  constraints   dict — operational limits: page_size, max_requests, total_rows,
                       rate_limit_qps, paging_mode
  fault_injection dict — fault mode and parameters for the mock service
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Task:
    """A single benchmark task for the Comtrade environment.

    Attributes:
        task_id:        Unique task identifier, e.g. "T1_single_page".
        description:    Human-readable description of the challenge.
        query:          Comtrade API query parameters (reporter, partner,
                        flow, hs, year).  Passed automatically to fetch_page().
        constraints:    Operational constraints for the task:
                        page_size, max_requests, total_rows, rate_limit_qps,
                        paging_mode ("page" or "offset").
        fault_injection: Fault mode configuration for the mock service.
                        Keys: mode (str), and optional mode-specific params
                        such as fail_on, duplicate_rate, etc.
    """

    task_id: str
    description: str
    query: Dict[str, Any]
    constraints: Dict[str, Any]
    fault_injection: Dict[str, Any]


def get_tasks() -> List[Task]:
    """Return all benchmark tasks in canonical order (T1–T10)."""
    return [
        Task(
            task_id="T1_single_page",
            description=(
                "Single query, single page. "
                "Validate schema + row count."
            ),
            query={"reporter": "840", "partner": "156", "flow": "M", "hs": "85", "year": 2021},
            constraints={
                "paging_mode": "page", "page_size": 1000,
                "max_requests": 5, "rate_limit_qps": 5, "total_rows": 800,
            },
            fault_injection={"mode": "none"},
        ),
        Task(
            task_id="T2_multi_page",
            description=(
                "Multi-page fetch (page+maxRecords). "
                "Must fetch all pages and merge into a single dataset."
            ),
            query={"reporter": "276", "partner": "250", "flow": "X", "hs": "84", "year": 2022},
            constraints={
                "paging_mode": "page", "page_size": 500,
                "max_requests": 50, "rate_limit_qps": 5, "total_rows": 2345,
            },
            fault_injection={"mode": "pagination"},
        ),
        Task(
            task_id="T3_duplicates",
            description=(
                "Duplicates within and across pages. "
                "Must deduplicate by primary key before submitting."
            ),
            query={"reporter": "392", "partner": "410", "flow": "M", "hs": "87", "year": 2020},
            constraints={
                "paging_mode": "offset", "page_size": 10,
                "max_requests": 50, "rate_limit_qps": 5, "total_rows": 25,
            },
            fault_injection={
                "mode": "duplicates",
                "duplicate_rate": 0.08,
                "cross_page_duplicate_rate": 0.03,
            },
        ),
        Task(
            task_id="T4_rate_limit_429",
            description=(
                "Occasional HTTP 429 responses. "
                "Must implement backoff + retry and still retrieve all records."
            ),
            query={"reporter": "724", "partner": "826", "flow": "X", "hs": "30", "year": 2019},
            constraints={
                "paging_mode": "page", "page_size": 10,
                "max_requests": 60, "rate_limit_qps": 3, "total_rows": 30,
            },
            fault_injection={"mode": "rate_limit", "fail_on": [2]},
        ),
        Task(
            task_id="T5_server_error_500",
            description=(
                "Occasional HTTP 500 responses. "
                "Must retry transient server errors and still finish."
            ),
            query={"reporter": "124", "partner": "36", "flow": "M", "hs": "12", "year": 2023},
            constraints={
                "paging_mode": "page", "page_size": 10,
                "max_requests": 60, "rate_limit_qps": 3, "total_rows": 30,
            },
            fault_injection={"mode": "server_error", "fail_on": [2]},
        ),
        Task(
            task_id="T6_page_drift",
            description=(
                "Non-deterministic page ordering: the same page number may return "
                "different rows on repeated requests. Must canonicalize and dedup."
            ),
            query={"reporter": "356", "partner": "704", "flow": "X", "hs": "09", "year": 2018},
            constraints={
                "paging_mode": "page", "page_size": 12,
                "max_requests": 60, "rate_limit_qps": 5, "total_rows": 36,
            },
            fault_injection={"mode": "page_drift"},
        ),
        Task(
            task_id="T7_totals_trap",
            description=(
                "Summary/totals rows are mixed in with individual records "
                "(marked with is_total=True). Must drop totals rows before submitting."
            ),
            query={"reporter": "826", "partner": "372", "flow": "M", "hs": "27", "year": 2017},
            constraints={
                "paging_mode": "offset", "page_size": 250,
                "max_requests": 60, "rate_limit_qps": 5, "total_rows": 750,
            },
            fault_injection={"mode": "totals_trap"},
        ),
        Task(
            task_id="T8_mixed_faults",
            description=(
                "Hardest task: combines HTTP 429 rate-limit faults AND "
                "cross-page duplicate records simultaneously. "
                "Must handle both retry logic and deduplication correctly."
            ),
            query={"reporter": "484", "partner": "528", "flow": "M", "hs": "72", "year": 2016},
            constraints={
                "paging_mode": "page", "page_size": 20,
                "max_requests": 80, "rate_limit_qps": 2, "total_rows": 60,
            },
            fault_injection={
                "mode": "mixed",
                "rate_limit_fail_on": [2, 5],          # 429 on 2nd and 5th requests
                "duplicate_rate": 0.10,                 # 10% within-page duplicates
                "cross_page_duplicate_rate": 0.05,      # 5% cross-page duplicates
            },
        ),
        # ----------------------------------------------------------------
        # Novel tasks — unique to ComtradeBench
        # ----------------------------------------------------------------
        Task(
            task_id="T9_adaptive_adversary",
            description=(
                "Adaptive adversarial fault injection: the environment observes "
                "the agent's success rate and dynamically escalates fault intensity. "
                "Initial faults are mild (10% dup rate), but each successful page fetch "
                "causes the environment to increase duplicate rate, inject additional "
                "429 errors, and introduce totals rows. The agent must stay robust as "
                "difficulty ramps up mid-episode. Tests generalization under distribution shift."
            ),
            query={"reporter": "276", "partner": "156", "flow": "X", "hs": "84", "year": 2020},
            constraints={
                "paging_mode": "page", "page_size": 15,
                "max_requests": 100, "rate_limit_qps": 2, "total_rows": 75,
            },
            fault_injection={
                "mode": "adaptive",
                "initial_duplicate_rate": 0.05,
                "escalation_per_page": 0.03,     # +3% dup rate per successful fetch
                "adaptive_429_after_page": 3,     # start injecting 429s after page 3
                "adaptive_totals_after_page": 4,  # start injecting totals rows after page 4
            },
        ),
        Task(
            task_id="T10_multi_agent_coop",
            description=(
                "Constrained-budget data fetching: single-agent challenge with "
                "a halved request budget (50 requests). The agent must avoid "
                "redundant page fetches while still covering all pages and "
                "deduplicating correctly. This keeps T10 stable for the current "
                "single-agent training stack while preserving pressure from a "
                "tight budget."
            ),
            query={"reporter": "392", "partner": "410", "flow": "M", "hs": "90", "year": 2021},
            constraints={
                "paging_mode": "page", "page_size": 10,
                "max_requests": 50, "rate_limit_qps": 5, "total_rows": 80,
            },
            fault_injection={
                "mode": "duplicates",
                "duplicate_rate": 0.05,
                "cross_page_duplicate_rate": 0.02,
            },
        ),
    ]


def get_task(task_id: str) -> Optional[Task]:
    """Return the Task with the given task_id, or None if not found.

    Args:
        task_id: Task identifier, e.g. "T1_single_page" or "T8_mixed_faults".

    Returns:
        Task instance or None.
    """
    for t in get_tasks():
        if t.task_id == task_id:
            return t
    return None
