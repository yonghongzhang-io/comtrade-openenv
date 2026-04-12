# conftest.py — pytest configuration for agent tests.
# Prevents pytest from walking up to comtrade_env/__init__.py
# which imports openenv (may not be installed in the test venv).
import sys
from pathlib import Path

# Ensure agent/ is on sys.path so 'from agent import ...' works
sys.path.insert(0, str(Path(__file__).resolve().parent))
