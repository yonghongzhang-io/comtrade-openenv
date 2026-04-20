"""Comtrade RL Environment.

An OpenEnv MCP environment that wraps the UN Comtrade mock benchmark.
The agent must fetch paginated trade data from a mock HTTP service,
handle faults (rate limits, server errors, duplicates, page drift),
and produce clean, deduplicated output.

Tasks (T1-T10):
  T1 - Single page fetch
  T2 - Multi-page pagination
  T3 - Deduplication across pages
  T4 - Retry on HTTP 429
  T5 - Retry on HTTP 500
  T6 - Page drift (non-deterministic ordering)
  T7 - Totals trap (drop summary rows)
  T8 - Mixed faults
  T9 - Adaptive adversary
  T10 - Constrained-budget stress task

Reward: normalized judge score (0.0 - 1.0), max raw score = 100.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import random
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from fastmcp import FastMCP

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State

logger = logging.getLogger(__name__)

MOCK_SERVICE_PORT = int(os.environ.get("MOCK_SERVICE_PORT", "7654"))

# Module-level coordination so parallel ComtradeEnvironment inits don't
# all try to start the mock service on the same port simultaneously.
_mock_service_lock = threading.Lock()
_mock_service_started = threading.Event()
MAX_REQUESTS_PER_EPISODE = 100
MAX_SCORE = 100.0


def _probe_mock_service_running(timeout_s: float = 1.0) -> bool:
    """Cheap outside-the-lock probe: is the mock service already listening?

    Uses a raw TCP connect instead of urllib.request.urlopen so that a
    not-yet-fully-ready uvicorn (accepting connections but not yet serving
    /docs) doesn't mislead us. TCP-level accept is what matters — if we can
    connect, the port is claimed by *something* that resembles our service.

    Returns True if port is reachable, False otherwise. Never raises.
    """
    try:
        with socket.create_connection(("127.0.0.1", MOCK_SERVICE_PORT), timeout=timeout_s):
            return True
    except (OSError, socket.timeout):
        return False


def _load_tasks():
    tasks_path = Path(__file__).parent / "tasks.py"
    spec = importlib.util.spec_from_file_location("tasks", tasks_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tasks"] = mod  # register before exec so @dataclass can resolve module
    spec.loader.exec_module(mod)
    return mod


def _load_judge():
    judge_path = Path(__file__).parent / "judge.py"
    spec = importlib.util.spec_from_file_location("judge", judge_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["judge"] = mod
    spec.loader.exec_module(mod)
    return mod


class ComtradeEnvironment(MCPEnvironment):
    """OpenEnv MCP environment for the UN Comtrade benchmark.

    The agent interacts via MCP tools:
      - get_task_info(): see current task details
      - fetch_page(...): fetch one page of trade data from mock service
      - submit_results(...): submit final data.jsonl + metadata for scoring

    The environment runs an embedded mock HTTP service that simulates
    the Comtrade API with configurable fault injection per task.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._lock = threading.Lock()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task = None
        self._mock_proc = None
        self._request_count = 0
        self._request_budget = MAX_REQUESTS_PER_EPISODE
        self._submitted = False
        self._output_dir = None
        self._tasks_mod = _load_tasks()
        self._judge_mod = _load_judge()

        mcp = FastMCP("comtrade_env")

        @mcp.tool
        def get_task_info() -> dict:
            """Get information about the current task.

            Returns the task ID, description, query parameters (reporter, partner,
            flow, hs, year), constraints (page_size, max_requests), and the
            mock service URL to call.
            """
            if self._current_task is None:
                return {"error": "No active episode. Call reset() first."}
            t = self._current_task
            return {
                "task_id": t.task_id,
                "description": t.description,
                "query": t.query,
                "constraints": t.constraints,
                "mock_service_url": f"http://localhost:{MOCK_SERVICE_PORT}/api/data",
                "requests_used": self._request_count,
                "requests_remaining": max(self._request_budget - self._request_count, 0),
            }

        @mcp.tool
        def fetch_page(page: int = 1, page_size: int = 500) -> dict:
            """Fetch one page of trade data from the mock Comtrade service.

            Uses the current task's query parameters (reporter, partner, flow,
            hs, year) automatically. You only need to specify pagination.

            Args:
                page: Page number (1-indexed).
                page_size: Number of records per page (default 500).

            Returns:
                Dict with 'rows' (list of records), 'page', 'total_pages',
                'has_more', and optionally 'status' on HTTP errors.
            """
            with self._lock:
                if self._current_task is None:
                    return {"error": "No active episode. Call reset() first."}
                if self._request_count >= self._request_budget:
                    return {"error": "Request limit exceeded for this episode."}
                self._request_count += 1
                self._state.step_count += 1
                task = self._current_task

            params = urllib.parse.urlencode({
                "task_id": task.task_id,
                "page": page,
                "page_size": page_size,
                "episode_id": self._state.episode_id,  # isolates concurrent agents
                **task.query,
            })
            url = f"http://localhost:{MOCK_SERVICE_PORT}/api/data?{params}"
            try:
                with urllib.request.urlopen(url, timeout=10) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                # Surface HTTP status distinctly so the agent can retry on 429/500.
                return {"status": e.code, "error": str(e), "retry": True}
            except urllib.error.URLError as e:
                # Network-layer failure (DNS, connection refused, timeout).
                logger.warning(f"fetch_page URLError: {e}")
                return {"error": f"URLError: {e.reason}", "retry": True}
            except json.JSONDecodeError as e:
                # Mock service returned invalid JSON — infrastructure issue, not agent error.
                logger.error(f"fetch_page mock returned non-JSON: {e}")
                return {"error": f"invalid JSON from mock: {e}"}
            except socket.timeout as e:
                logger.warning(f"fetch_page timed out: {e}")
                return {"error": "timeout", "retry": True}

        @mcp.tool
        def submit_results(
            data_jsonl: str,
            metadata_json: str,
            run_log: str = "",
        ) -> dict:
            """Submit final results for scoring.

            Call this once you have fetched all pages and cleaned the data.

            Args:
                data_jsonl: All trade records as newline-delimited JSON
                    (one JSON object per line, deduplicated).
                metadata_json: JSON string with keys: task_id, query,
                    row_count, schema (list of field names), totals_handling.
                run_log: Execution log. Must contain 'page=', 'request=',
                    'complete=', and 'task_id=' entries for full observability score.

            Returns:
                Dict with 'reward' (0.0-1.0), 'score' (raw 0-100),
                'breakdown' (per-category scores), and 'errors'.
            """
            with self._lock:
                if self._current_task is None:
                    return {"error": "No active episode."}
                if self._submitted:
                    return {"error": "Already submitted. Call reset() to start a new episode."}
                self._submitted = True
                task = self._current_task
                out_base = self._output_dir

            try:
                out_dir = Path(out_base) / task.task_id
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "data.jsonl").write_text(data_jsonl)
                try:
                    metadata = json.loads(metadata_json) if metadata_json else {}
                except json.JSONDecodeError as e:
                    # Agent submitted malformed metadata — log but don't crash scoring.
                    # The judge will penalise via reduced Correctness / Observability.
                    logger.info(f"submit_results: invalid metadata_json ({e}); using empty dict")
                    metadata = {}
                if not isinstance(metadata, dict):
                    metadata = {}
                metadata.setdefault("request_count", self._request_count)
                metadata.setdefault("request_budget", self._request_budget)
                request_stats = metadata.setdefault("request_stats", {})
                if not isinstance(request_stats, dict):
                    request_stats = {}
                    metadata["request_stats"] = request_stats
                request_stats.setdefault("requests_used", self._request_count)
                request_stats.setdefault("request_budget", self._request_budget)
                (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False))
                log_content = run_log or (
                    f"task_id={task.task_id}\npage=1\nrequest=1\ncomplete=true\n"
                )
                (out_dir / "run.log").write_text(log_content)

                result = self._judge_mod.score_task(
                    task_id=task.task_id,
                    output_dir=out_base,
                    task=task,
                )
                reward = round(min(result.total / MAX_SCORE, 1.0), 4)
                logger.info(f"[{task.task_id}] score={result.total:.1f} reward={reward}")
                return {
                    "reward": reward,
                    "score": result.total,
                    "breakdown": result.breakdown,
                    "errors": result.errors,
                    "done": True,
                }
            except Exception as e:
                logger.exception("Judge error")
                return {"reward": 0.0, "score": 0.0, "errors": [str(e)], "done": True}

        super().__init__(mcp)
        self._start_mock_service()

    def _start_mock_service(self):
        mock_dir = Path(__file__).parent / "mock_service"
        if not mock_dir.exists():
            logger.warning("mock_service not found")
            return

        # --- Fast path (no lock): Event already set ---
        # Under parallel GRPO rollouts, dozens of ComtradeEnvironment() ctors
        # can fire in under a second. Serializing an HTTP probe through the
        # lock turns that into a stampede. This fast-path lets everyone after
        # the first init skip the lock entirely.
        if _mock_service_started.is_set():
            logger.debug(f"Mock service already running on :{MOCK_SERVICE_PORT}, reusing (fast path).")
            return

        # --- Probe BEFORE lock ---
        # Handles the external-process case (e.g. the user ran the mock service
        # manually, or we're in a fresh Python process that inherited an
        # already-bound port). The probe may take up to 1 s on a cold connection;
        # doing it outside the lock means N processes probe in parallel instead
        # of being serialised into N × 1 s.
        if _probe_mock_service_running():
            _mock_service_started.set()
            logger.info(f"Mock service already running externally on :{MOCK_SERVICE_PORT} (probed, no lock).")
            return

        # --- Slow path: acquire lock and start a subprocess ---
        # We held off the lock until here to avoid the probe-under-lock stampede,
        # but inside the lock we must re-check Event (another thread may have
        # started the service while we were probing). This is double-checked
        # locking; it's correct with threading.Event (which has proper memory
        # synchronisation on set/is_set).
        with _mock_service_lock:
            if _mock_service_started.is_set():
                logger.debug(f"Mock service started by another thread while we probed; reusing.")
                return

            try:
                self._mock_proc = subprocess.Popen(
                    [sys.executable, "-m", "uvicorn", "app:app",
                     "--host", "0.0.0.0", "--port", str(MOCK_SERVICE_PORT),
                     "--log-level", "warning"],
                    cwd=str(mock_dir),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                time.sleep(2.0)
                _mock_service_started.set()
                logger.info(f"Mock service started on :{MOCK_SERVICE_PORT}")
            except Exception as e:
                logger.warning(f"Mock service failed to start: {e}")

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """Start a new episode with a Comtrade data-fetching task.

        Args:
            seed: Optional random seed for task selection.
            episode_id: Optional episode ID.
            task_id: Pin to a specific task ID (T1-T10). Random if not set.

        Returns:
            Observation with task metadata and instructions.
        """
        if seed is not None:
            random.seed(seed)

        with self._lock:
            self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
            self._request_count = 0
            self._submitted = False
            self._output_dir = tempfile.mkdtemp(prefix="comtrade_out_")

            if task_id:
                self._current_task = self._tasks_mod.get_task(task_id)
            else:
                tasks = self._tasks_mod.get_tasks()
                self._current_task = random.choice(tasks)
            if self._current_task is not None:
                self._request_budget = int(
                    self._current_task.constraints.get("max_requests", MAX_REQUESTS_PER_EPISODE)
                )
            else:
                self._request_budget = MAX_REQUESTS_PER_EPISODE

        t = self._current_task
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "task_id": t.task_id,
                "description": t.description,
                "query": t.query,
                "constraints": t.constraints,
                "instructions": (
                    "1. Call get_task_info() to see query parameters.\n"
                    "2. Call fetch_page(page=1) and subsequent pages until has_more=False.\n"
                    "3. Deduplicate records by primary key (year, reporter, partner, flow, hs, record_id).\n"
                    "4. Call submit_results(data_jsonl=..., metadata_json=..., run_log=...) when done."
                ),
            },
        )

    def _step_impl(self, action: Action, **kwargs: Any) -> Observation:
        return Observation(
            done=False,
            reward=0.0,
            metadata={"error": f"Unknown action: {type(action).__name__}. Use MCP tools."},
        )

    @property
    def state(self) -> State:
        return self._state

    def close(self) -> None:
        """Explicit cleanup — terminates the mock service subprocess if this
        environment owns one. Safe to call multiple times.

        Prefer this over relying on __del__, which has unpredictable GC
        timing and can leave zombie processes under CPython reference-cycle
        collection.

        Usage:
            env = ComtradeEnvironment()
            try:
                # ... use env ...
            finally:
                env.close()

            # Or with contextlib.closing:
            from contextlib import closing
            with closing(ComtradeEnvironment()) as env:
                # ...
        """
        proc = getattr(self, "_mock_proc", None)
        if proc is None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=1.0)
        except (OSError, ProcessLookupError) as e:
            logger.debug(f"close: subprocess already gone: {e}")
        finally:
            self._mock_proc = None
            # NOTE: we do NOT clear _mock_service_started — other env instances
            # may still be using the same service, and the module-level Event
            # outlives any single instance.

    def __del__(self):
        # Fallback only; prefer explicit close(). See close() docstring.
        self.close()
