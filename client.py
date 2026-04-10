# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Comtrade Env Environment Client.

Connects to a running ComtradeEnvironment server and provides a typed interface
for RL training loops and manual testing.

The Comtrade environment is an MCP-first environment — the agent interacts via
MCP tools (get_task_info, fetch_page, submit_results) rather than through typed
actions. ComtradeAction is therefore an empty placeholder; use
``env.call_tool(name, **kwargs)`` from MCPToolClient for tool interactions, or
use the WebSocket step interface for orchestration.

Example (async, recommended for RL loops):
    >>> import asyncio
    >>> from comtrade_env import ComtradeAction, ComtradeEnv
    >>>
    >>> async def run():
    ...     async with ComtradeEnv(base_url="http://localhost:8000") as env:
    ...         result = await env.reset(task_id="T1_single_page")
    ...         print(result.observation.task_id)
    ...         print(result.observation.description)
    ...         # Use MCP tools for actual interaction — see MCPToolClient
    >>>
    >>> asyncio.run(run())

Example (sync wrapper):
    >>> from comtrade_env import ComtradeAction, ComtradeEnv
    >>>
    >>> with ComtradeEnv(base_url="http://localhost:8000").sync() as env:
    ...     result = env.reset(task_id="T2_multi_page")
    ...     task = result.observation
    ...     print(f"Task: {task.task_id} — {task.description}")
    ...     print(f"Query: {task.query}")

Example with Docker (auto-lifecycle management):
    >>> import asyncio
    >>> from comtrade_env import ComtradeEnv
    >>>
    >>> async def run():
    ...     env = await ComtradeEnv.from_docker_image("comtrade-env:latest")
    ...     try:
    ...         result = await env.reset()
    ...         state = await env.state()
    ...         print(f"Episode: {state.episode_id}")
    ...     finally:
    ...         await env.close()
    >>>
    >>> asyncio.run(run())
"""

from typing import Any, Dict, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ComtradeAction, ComtradeObservation


class ComtradeEnv(EnvClient[ComtradeAction, ComtradeObservation, State]):
    """Client for the UN Comtrade RL environment.

    Wraps a running ComtradeEnvironment server over WebSocket, providing
    typed reset() and step() calls alongside MCP tool access.

    ComtradeEnvironment is MCP-first: the agent interacts with the environment
    via three MCP tools exposed at ``/mcp``:

    * ``get_task_info()`` — returns the current task details
    * ``fetch_page(page, page_size)`` — fetches one page of trade data
    * ``submit_results(data_jsonl, metadata_json, run_log)`` — submits for scoring

    The ``reset()`` method selects a task (T1–T7) and returns an observation
    with task metadata. The ``step()`` method is available for orchestration
    but is not the primary interaction path.

    Tasks:
        T1  Single page fetch, validate schema and row count.
        T2  Multi-page pagination across multiple pages.
        T3  Deduplication across overlapping pages.
        T4  HTTP 429 rate-limit fault; must retry with backoff.
        T5  HTTP 500 server error fault; must retry.
        T6  Page drift (non-deterministic ordering); must canonicalize.
        T7  Totals trap (summary rows mixed in); must drop totals.

    Scoring dimensions (0–100 → reward 0.0–1.0):
        correctness (30) + completeness (15) + robustness (15)
        + efficiency (15) + data_quality (15) + observability (10)
    """

    def _step_payload(self, action: ComtradeAction) -> Dict[str, Any]:
        """Convert ComtradeAction to the JSON payload expected by the server.

        ComtradeAction is an intentionally empty action class — real agent
        interactions happen through MCP tools, not typed step() actions.
        This method returns an empty dict so the WebSocket step message is
        valid but carries no action data.

        Args:
            action: ComtradeAction instance (fields ignored; it carries no data).

        Returns:
            Empty dictionary — MCP tools are the primary interaction path.
        """
        return {}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ComtradeObservation]:
        """Parse a server response into StepResult[ComtradeObservation].

        The Comtrade environment encodes task metadata inside the observation's
        ``metadata`` dict. This method extracts those keys into typed fields on
        ``ComtradeObservation`` for convenient access.

        Metadata keys surfaced as typed fields:
            task_id       — e.g. "T1_single_page"
            description   — human-readable task description
            query         — dict of reporter, partner, flow, hs, year
            constraints   — dict of page_size, max_requests, total_rows, etc.
            instructions  — step-by-step usage guidance

        Args:
            payload: Raw JSON response dict from the server.

        Returns:
            StepResult with a populated ComtradeObservation.
        """
        obs_data = payload.get("observation", {})
        # The environment packs task info into metadata when returning Observation
        meta: Dict[str, Any] = obs_data.get("metadata", {})

        observation = ComtradeObservation(
            task_id=meta.get("task_id", ""),
            description=meta.get("description", ""),
            query=meta.get("query", {}),
            constraints=meta.get("constraints", {}),
            instructions=meta.get("instructions", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=meta,
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """Parse a server response into a State object.

        Args:
            payload: Raw JSON dict from the ``/state`` WebSocket message.

        Returns:
            State with episode_id and step_count.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
