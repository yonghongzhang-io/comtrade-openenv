# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the Comtrade RL Environment.

The comtrade_env environment presents UN Comtrade-style trade data fetching
tasks to an LLM agent.  Agents interact via three MCP tools
(get_task_info, fetch_page, submit_results) rather than through typed actions.

``ComtradeAction`` is therefore an intentionally empty placeholder — it
satisfies the EnvClient generic typing but carries no data.

``ComtradeObservation`` surfaces the task metadata returned by ``reset()``
as typed, named fields so that client code can access task details without
digging through an untyped ``metadata`` dict.

Typing philosophy
-----------------
- All fields that the server guarantees to populate on ``reset()`` have
  concrete types (str, dict).
- Optional/nullable fields that may be absent (e.g. reward after reset)
  use ``Optional[...]``.
- The raw ``metadata`` field is still present for forward compatibility so
  new server fields are accessible without updating the model.
"""

from typing import Any, Dict, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class ComtradeAction(Action):
    """Generic action placeholder for the MCP-first Comtrade environment.

    Real agent interactions happen through MCP tools, not through typed
    actions.  This class exists only to satisfy the EnvClient generic
    contract; its instances carry no meaningful data.

    Usage:
        The server ignores the payload of ComtradeAction completely. Use
        ``MCPToolClient.call_tool()`` or the ``/mcp`` endpoint directly
        to interact with the environment's MCP tools.
    """

    pass


class ComtradeObservation(Observation):
    """Observation from the Comtrade environment.

    Returned by both ``reset()`` and ``step()``.

    On ``reset()``:
        All task-metadata fields (``task_id``, ``description``, ``query``,
        ``constraints``, ``instructions``) are populated using the selected
        task (T1–T7).  ``reward`` is 0.0 and ``done`` is False.

    On ``step()`` / MCP ``submit_results()`` response:
        ``reward`` is populated (0.0–1.0).  ``done`` is True.
        Other fields may be empty for non-terminal steps.

    Task metadata fields
    --------------------
    task_id : str
        Task identifier, e.g. ``"T1_single_page"`` or ``"T6_page_drift"``.
    description : str
        Human-readable description of what the agent must do.
    query : dict
        Comtrade query parameters: reporter (ISO 3166-1 numeric), partner,
        flow (M=import / X=export), hs (HS commodity code), year.
        Example: ``{"reporter": "840", "partner": "156", "flow": "M",
        "hs": "85", "year": 2021}``
    constraints : dict
        Operational constraints for the task:
        page_size, max_requests, rate_limit_qps, total_rows, paging_mode.
    instructions : str
        Step-by-step guidance for the agent (also available via
        ``get_task_info()`` MCP tool).

    Scoring fields
    --------------
    After ``submit_results()`` returns, the environment injects the final
    score into the step result.  The breakdown is available via the MCP
    tool response rather than this Observation.
    """

    # Task identification
    task_id: str = Field(
        default="",
        description="Task identifier, e.g. 'T1_single_page'.",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the task.",
    )

    # Query + operational parameters
    query: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Comtrade query parameters: reporter, partner, flow, hs, year. "
            "These are automatically used by fetch_page() — do not override."
        ),
    )
    constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Task operational constraints: page_size, max_requests, "
            "rate_limit_qps, total_rows, paging_mode."
        ),
    )

    # Agent guidance
    instructions: str = Field(
        default="",
        description=(
            "Step-by-step guidance on how to complete the task. "
            "Also available via get_task_info() MCP tool."
        ),
    )
