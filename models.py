# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the Comtrade RL Environment.

The comtrade_env environment presents UN Comtrade-style trade data fetching tasks.
Agents interact via MCP tools (fetch_page, submit_results) rather than typed actions.
"""

from openenv.core.env_server.types import Action, Observation


class ComtradeAction(Action):
    """Generic MCP action placeholder (actual interaction via MCP tools)."""
    pass


class ComtradeObservation(Observation):
    """Observation from the Comtrade environment.

    The metadata field contains:
      - task_id: current task identifier
      - description: what the agent must do
      - query: dict of reporter, partner, flow, hs, year
      - constraints: page_size, max_requests, etc.
      - instructions: step-by-step guidance
    """
    pass
