"""Shared pytest configuration and environment checks."""

import functools
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import psutil
import pytest


def pytest_configure(config):
    """Verify test dependencies are importable, with a clear error message."""
    missing = []
    for mod in ("numpy", "pydantic"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)

    if missing:
        print(
            f"\nERROR: Missing required packages: {', '.join(missing)}\n"
            f"The pytest binary may be running in a different environment than\n"
            f"the project's installed dependencies. Try:\n"
            f"  python -m pytest tests/ -v\n"
            f"Or reinstall with:  pip install -e '.[dev]'\n",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # git exports GIT_DIR / GIT_INDEX_FILE / GIT_WORK_TREE to hooks. If the suite
    # runs from a hook (pre-push), every test that `git init`s a tmp_path and
    # shells out inherits them and operates on the REAL repository instead:
    # on 2026-09-06 this flipped the shared .git/config to core.bare=true and
    # wrote a test identity into [user]. Strip them before any test runs.
    for var in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_PREFIX",
        "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY",
    ):
        os.environ.pop(var, None)


@pytest.fixture(autouse=True, scope="session")
def _isolate_default_api_db(tmp_path_factory):
    """Redirect the agent-API persistence default DB path to a temp location for
    the whole test session.

    The API stores resolve their path as ``db_path or _DEFAULT_DB_PATH``. Per-test
    fixtures pass an explicit temp ``db_path``, but a background run thread can
    outlive its test, and any lazy ``get_store()`` call once the per-test store has
    been reset to ``None`` constructs a store with *no* path — which would otherwise
    fall back to the real ``runs/agent_api.db``, leaking test writes to the real
    database and contaminating later tests. Pointing the default at a throwaway temp
    file makes that fallback harmless. Guarded so environments that can't import the
    persistence module (e.g. missing optional deps) still run.
    """
    try:
        import swarm.api.persistence as persistence
    except Exception:
        yield
        return

    original = persistence._DEFAULT_DB_PATH
    persistence._DEFAULT_DB_PATH = (
        tmp_path_factory.mktemp("api_default_db") / "agent_api.db"
    )
    try:
        yield
    finally:
        persistence._DEFAULT_DB_PATH = original


# ---------------------------------------------------------------------------
# Memory Profiling and Limiting
# ---------------------------------------------------------------------------


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB (RSS)."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def get_peak_memory_mb() -> float:
    """Get peak memory usage in MB using resource module."""
    if platform.system() == "Linux":
        # On Linux, ru_maxrss is in KB
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    else:
        # On macOS/BSD, ru_maxrss is in bytes
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def load_memory_baselines() -> Dict[str, Any]:
    """Load memory baselines from tests/memory_baselines.json."""
    baseline_path = Path(__file__).parent / "memory_baselines.json"
    if not baseline_path.exists():
        return {}
    with open(baseline_path) as f:
        return json.load(f)


def memory_limit(max_mb: int):
    """Decorator to enforce process-level memory limit on a test.

    Args:
        max_mb: Maximum memory in megabytes

    Usage:
        @memory_limit(max_mb=500)
        def test_memory_bounded():
            # test code
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Set memory limit (only works on Unix-like systems)
            if platform.system() != "Windows":
                try:
                    # Set both soft and hard limits
                    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                    limit_bytes = max_mb * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, hard))
                except (ValueError, OSError) as e:
                    pytest.skip(f"Cannot set memory limit: {e}")

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Reset limit
                if platform.system() != "Windows":
                    try:
                        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
                    except (ValueError, OSError):
                        # Reset failure is non-fatal for test cleanup on some platforms.
                        pass

        return wrapper
    return decorator


@pytest.fixture
def memory_tracker():
    """Fixture to track memory usage during a test.

    Usage:
        def test_memory(memory_tracker):
            # do work
            memory_tracker.checkpoint("after_work")
            assert memory_tracker.peak_mb < 100
    """
    class MemoryTracker:
        def __init__(self):
            self.start_mb = get_memory_usage_mb()
            self.peak_mb = self.start_mb
            self.checkpoints = {}
            self.start_time = time.time()

        def checkpoint(self, name: str):
            """Record a memory checkpoint."""
            current_mb = get_memory_usage_mb()
            self.peak_mb = max(self.peak_mb, current_mb)
            self.checkpoints[name] = {
                "memory_mb": current_mb,
                "delta_mb": current_mb - self.start_mb,
                "elapsed_seconds": time.time() - self.start_time,
            }

        def get_delta_mb(self) -> float:
            """Get memory increase since test start."""
            return get_memory_usage_mb() - self.start_mb

        def assert_bounded(self, max_mb: float, message: Optional[str] = None):
            """Assert that memory usage is below threshold."""
            current_mb = get_memory_usage_mb()
            self.peak_mb = max(self.peak_mb, current_mb)
            if current_mb > max_mb:
                msg = message or f"Memory exceeded: {current_mb:.1f} MB > {max_mb} MB"
                raise AssertionError(msg)

        def assert_stable(self, tolerance_mb: float = 5.0):
            """Assert that memory has stabilized (not growing)."""
            if len(self.checkpoints) < 2:
                raise ValueError("Need at least 2 checkpoints to check stability")

            values = [cp["memory_mb"] for cp in self.checkpoints.values()]
            # Check if last few values are within tolerance
            recent = values[-3:] if len(values) >= 3 else values
            if max(recent) - min(recent) > tolerance_mb:
                raise AssertionError(
                    f"Memory not stable: recent values {recent} MB (tolerance {tolerance_mb} MB)"
                )

    tracker = MemoryTracker()
    yield tracker

    # Final checkpoint
    tracker.checkpoint("test_end")


def assert_memory_bounded(max_mb: float, message: Optional[str] = None):
    """Helper to assert current memory is below threshold.

    Args:
        max_mb: Maximum allowed memory in MB
        message: Optional custom error message
    """
    current_mb = get_memory_usage_mb()
    if current_mb > max_mb:
        msg = message or f"Memory exceeded: {current_mb:.1f} MB > {max_mb} MB"
        raise AssertionError(msg)


def assert_memory_stable(
    checkpoints: list[float],
    tolerance_mb: float = 5.0,
    message: Optional[str] = None
):
    """Assert that memory measurements show stability.

    Args:
        checkpoints: List of memory measurements in MB
        tolerance_mb: Allowed variation in MB
        message: Optional custom error message
    """
    if len(checkpoints) < 2:
        raise ValueError("Need at least 2 checkpoints to check stability")

    if max(checkpoints) - min(checkpoints) > tolerance_mb:
        msg = message or (
            f"Memory not stable: range {min(checkpoints):.1f}-{max(checkpoints):.1f} MB "
            f"(tolerance {tolerance_mb} MB)"
        )
        raise AssertionError(msg)
