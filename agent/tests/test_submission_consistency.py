from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def test_green_tasks_match_server_tasks():
    from green.tasks_green import get_tasks as get_green_tasks
    from server.tasks import get_tasks as get_server_tasks

    assert [task.task_id for task in get_green_tasks()] == [
        task.task_id for task in get_server_tasks()
    ]


def test_green_and_server_efficiency_baselines_stay_in_sync():
    from green.judge_green import TASK_EFFICIENCY_BASELINES as green_baselines
    from server.judge import TASK_EFFICIENCY_BASELINES as server_baselines
    from server.tasks import get_tasks

    task_ids = [task.task_id for task in get_tasks()]

    for task_id in task_ids:
        assert task_id in server_baselines
        assert task_id in green_baselines
        assert green_baselines[task_id] == server_baselines[task_id]
