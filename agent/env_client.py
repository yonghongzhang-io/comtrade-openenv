"""
env_client.py — Two client backends for comtrade_env.

  ComtradeEnvClient   — HTTP REST client for a running OpenEnv HTTP server.
                        Kept for compatibility, but NOTE: the /step endpoint
                        expects a typed ComtradeAction payload; arbitrary tool
                        calls via POST /step may return 422.

  InProcessEnvClient  — Preferred for training. Wraps ComtradeEnvironment
                        directly (no HTTP server required). Auto-discovers
                        OpenEnv relative to this file. Each instance creates
                        its own ComtradeEnvironment, which is episode-isolated
                        and safe for concurrent use.

ComtradeAgent accepts either backend via its `env_client` parameter.
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8000"
_TOOL_ACTION_TYPE = "mcp_tool"

MOCK_PORT = 7654  # default; overridden by MOCK_SERVICE_PORT env var


# ---------------------------------------------------------------------------
# Path helper (shared by InProcessEnvClient and direct_test.py)
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    """Return the comtrade_env repo root (parent of agent/)."""
    return Path(__file__).resolve().parent.parent


def _setup_openenv_path() -> None:
    """Add OpenEnv source paths so ComtradeEnvironment can be imported.

    New layout:
        comtrade_env/          ← repo root
          agent/env_client.py  ← this file
          server/              ← environment code
        OpenEnv/src/           ← framework (two levels up from repo root)
    """
    import os
    global MOCK_PORT
    MOCK_PORT = int(os.environ.get("MOCK_SERVICE_PORT", "7654"))

    repo_root = _repo_root()  # comtrade_env/

    # Find OpenEnv framework root (contains src/)
    candidates = [
        repo_root.parent.parent,               # OpenEnv/envs/comtrade_env → OpenEnv
        repo_root.parent / "OpenEnv",           # sibling layout
        repo_root.parent.parent / "OpenEnv",    # grandparent layout
    ]
    openenv_root = next((p for p in candidates if (p / "src").exists()), None)
    if openenv_root is None:
        raise RuntimeError(
            "Cannot find OpenEnv/src/. Searched:\n"
            + "\n".join(f"  {c}" for c in candidates)
            + "\nInstall openenv-core or clone OpenEnv adjacent to this repo."
        )
    for sub_path in [
        openenv_root / "src",
        openenv_root / "envs",
        repo_root,  # so 'server.comtrade_env_environment' resolves
    ]:
        s = str(sub_path)
        if s not in sys.path:
            sys.path.insert(0, s)


# ---------------------------------------------------------------------------
# HTTP client (for running OpenEnv server)
# ---------------------------------------------------------------------------

class ComtradeEnvClient:
    """Thin synchronous HTTP client for one comtrade_env session.

    NOTE: The OpenEnv /step endpoint expects a typed ComtradeAction payload.
    Prefer InProcessEnvClient for training; this client is kept for cases
    where an external OpenEnv HTTP server is explicitly required.
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session_id: Optional[str] = None
        self._task_info: Optional[dict] = None

    def reset(self, task_id: Optional[str] = None, seed: Optional[int] = None) -> dict:
        """Reset the environment. Returns initial observation as dict."""
        payload: dict[str, Any] = {}
        if task_id:
            payload["task_id"] = task_id
        if seed is not None:
            payload["seed"] = seed
        resp = requests.post(
            f"{self.base_url}/reset", json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        obs = resp.json()
        self._session_id = obs.get("episode_id")
        self._task_info = obs.get("metadata", {})
        return obs

    @property
    def task_info(self) -> dict:
        if self._task_info is None:
            raise RuntimeError("Call reset() before accessing task_info")
        return self._task_info

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        payload = {"type": _TOOL_ACTION_TYPE, "tool": tool_name, "arguments": arguments}
        resp = requests.post(
            f"{self.base_url}/step", json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        result = resp.json()
        return result.get("metadata", result)

    def get_task_info(self) -> dict:
        return self.call_tool("get_task_info", {})

    def fetch_page(self, page: int = 1, page_size: int = 500) -> dict:
        return self.call_tool("fetch_page", {"page": page, "page_size": page_size})

    def submit_results(self, data_jsonl: str, metadata_json: str, run_log: str = "") -> dict:
        return self.call_tool("submit_results", {
            "data_jsonl": data_jsonl,
            "metadata_json": metadata_json,
            "run_log": run_log,
        })

    def state(self) -> dict:
        resp = requests.get(f"{self.base_url}/state", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def is_alive(self) -> bool:
        try:
            requests.get(f"{self.base_url}/state", timeout=5)
            return True
        except Exception:
            return False

    def wait_until_ready(self, max_wait: int = 30) -> None:
        deadline = time.time() + max_wait
        while time.time() < deadline:
            if self.is_alive():
                return
            time.sleep(1)
        raise TimeoutError(f"comtrade_env server not reachable at {self.base_url}")


# ---------------------------------------------------------------------------
# In-process client (preferred for training — no HTTP server required)
# ---------------------------------------------------------------------------

class InProcessEnvClient:
    """
    Wraps ComtradeEnvironment directly — no OpenEnv HTTP server required.

    Each instance creates its own ComtradeEnvironment (episode-isolated).
    Multiple instances share one mock service subprocess (first-start wins;
    subsequent inits detect the running service via HTTP health check).

    Implements the same interface as ComtradeEnvClient so ComtradeAgent
    and train_grpo.py can use either backend interchangeably.

    Usage:
        env = InProcessEnvClient()
        obs = env.reset(task_id="T1_single_page")
        result = env.call_tool("fetch_page", {"page": 1})
        result = env.call_tool("submit_results", {...})
    """

    _path_setup_done: bool = False
    _path_lock = threading.Lock()

    def __init__(self):
        self._env = None   # ComtradeEnvironment, lazy-initialized
        self._mock_port: int = MOCK_PORT

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _ensure_env(self) -> None:
        if self._env is not None:
            return
        # Set up sys.path once per process (thread-safe)
        with InProcessEnvClient._path_lock:
            if not InProcessEnvClient._path_setup_done:
                _setup_openenv_path()
                InProcessEnvClient._path_setup_done = True
                import os
                self._mock_port = int(os.environ.get("MOCK_SERVICE_PORT", "7654"))

        # Import ComtradeEnvironment directly from its file to avoid triggering
        # comtrade_env/__init__.py which imports gradio via the OpenEnv client stack.
        import importlib.util as _ilu
        _env_file = _repo_root() / "server" / "comtrade_env_environment.py"
        if not _env_file.exists():
            raise RuntimeError(f"Cannot find environment at {_env_file}")
        _spec = _ilu.spec_from_file_location("comtrade_env_environment", _env_file)
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        self._env = _mod.ComtradeEnvironment()

    # ------------------------------------------------------------------
    # Public interface (matches ComtradeEnvClient)
    # ------------------------------------------------------------------

    def reset(self, task_id: Optional[str] = None, seed: Optional[int] = None) -> dict:
        """Reset the environment. Returns initial observation as dict."""
        self._ensure_env()
        obs = self._env.reset(task_id=task_id, seed=seed)
        return {
            "metadata": dict(obs.metadata),
            "done": obs.done,
            "reward": obs.reward,
            "episode_id": self._env._state.episode_id,
        }

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Dispatch to the appropriate tool handler."""
        self._ensure_env()
        if tool_name == "get_task_info":
            return self._get_task_info()
        elif tool_name == "fetch_page":
            return self._fetch_page(**arguments)
        elif tool_name == "submit_results":
            return self._submit_results(**arguments)
        return {"error": f"Unknown tool: {tool_name}. Available: get_task_info, fetch_page, submit_results"}

    # Convenience wrappers (same as ComtradeEnvClient)
    def get_task_info(self) -> dict:
        return self.call_tool("get_task_info", {})

    def fetch_page(self, page: int = 1, page_size: int = 500) -> dict:
        return self.call_tool("fetch_page", {"page": page, "page_size": page_size})

    def submit_results(self, data_jsonl: str, metadata_json: str, run_log: str = "") -> dict:
        return self.call_tool("submit_results", {
            "data_jsonl": data_jsonl,
            "metadata_json": metadata_json,
            "run_log": run_log,
        })

    def is_alive(self) -> bool:
        return True  # always alive for in-process

    def wait_until_ready(self, max_wait: int = 30) -> None:
        """Trigger environment initialization (starts mock service if needed)."""
        self._ensure_env()

    def state(self) -> dict:
        if self._env is None:
            return {}
        return {
            "episode_id": self._env._state.episode_id,
            "step_count": self._env._state.step_count,
        }

    # ------------------------------------------------------------------
    # Tool implementations (mirror the MCP tool closures in the environment)
    # ------------------------------------------------------------------

    def _get_task_info(self) -> dict:
        env = self._env
        if env._current_task is None:
            return {"error": "No active episode. Call reset() first."}
        t = env._current_task
        import os
        port = int(os.environ.get("MOCK_SERVICE_PORT", "7654"))
        return {
            "task_id": t.task_id,
            "description": t.description,
            "query": t.query,
            "constraints": t.constraints,
            "mock_service_url": f"http://localhost:{port}/api/data",
            "requests_used": env._request_count,
            "requests_remaining": 100 - env._request_count,
        }

    def _fetch_page(self, page: int = 1, page_size: int = 500) -> dict:
        """Fetch one page via the embedded mock service HTTP endpoint."""
        env = self._env
        with env._lock:
            if env._current_task is None:
                return {"error": "No active episode. Call reset() first."}
            if env._request_count >= 100:
                return {"error": "Request limit exceeded for this episode."}
            env._request_count += 1
            env._state.step_count += 1
            task = env._current_task
            episode_id = env._state.episode_id

        import os
        port = int(os.environ.get("MOCK_SERVICE_PORT", "7654"))
        params = urllib.parse.urlencode({
            "task_id": task.task_id,
            "page": page,
            "page_size": page_size,
            "episode_id": episode_id,
            **task.query,
        })
        url = f"http://localhost:{port}/api/data?{params}"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            return {"status": e.code, "error": str(e), "retry": True}
        except Exception as e:
            return {"error": str(e)}

    def _submit_results(
        self,
        data_jsonl: str,
        metadata_json: str,
        run_log: str = "",
    ) -> dict:
        """Write output files and score via the judge."""
        env = self._env
        with env._lock:
            if env._current_task is None:
                return {"error": "No active episode."}
            if env._submitted:
                return {"error": "Already submitted. Call reset() to start a new episode."}
            env._submitted = True
            task = env._current_task
            out_base = env._output_dir

        try:
            out_dir = Path(out_base) / task.task_id
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "data.jsonl").write_text(data_jsonl, encoding="utf-8")
            (out_dir / "metadata.json").write_text(metadata_json, encoding="utf-8")
            log_content = run_log or (
                f"task_id={task.task_id}\npage=1\nrequest=1\ncomplete=true\n"
            )
            (out_dir / "run.log").write_text(log_content, encoding="utf-8")

            result = env._judge_mod.score_task(
                task_id=task.task_id,
                output_dir=out_base,
                task=task,
            )
            reward = round(min(result.total / 100.0, 1.0), 4)
            logger.info(f"[{task.task_id}] score={result.total:.1f} reward={reward}")
            return {
                "reward": reward,
                "score": result.total,
                "breakdown": result.breakdown,
                "errors": result.errors,
                "done": True,
            }
        except Exception as e:
            logger.exception("Judge error in InProcessEnvClient")
            return {"reward": 0.0, "score": 0.0, "errors": [str(e)], "done": True}
