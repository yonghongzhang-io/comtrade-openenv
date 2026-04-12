"""
Green-agent task definitions for ComtradeBench.

The Green evaluator and the main environment must score the exact same task set.
To keep that contract tight, this module re-exports the canonical task
definitions from `server.tasks` instead of maintaining a second copy that can
drift.
"""

from __future__ import annotations

try:
    from server.tasks import Task, get_task, get_tasks
except ImportError:  # pragma: no cover
    from comtrade_env.server.tasks import Task, get_task, get_tasks

__all__ = ["Task", "get_task", "get_tasks"]
