# Memory Testing Guide

This document describes the memory management testing infrastructure for the SWARM project, designed to prevent memory leaks and unbounded growth in long-running simulations.

## Overview

The memory testing infrastructure includes:
- **Memory-focused test markers** for pytest
- **Memory limiting decorators** for resource-bounded tests
- **Memory tracking fixtures** for detailed profiling
- **Baseline expectations** for each agent type
- **CI workflows** for automated testing

## Quick Start

### Running Memory Tests Locally

```bash
# Run all memory-focused tests
python -m pytest tests/test_memory_bounds.py -v -m memory

# Run with memory profiling
python -m memory_profiler tests/test_memory_bounds.py

# Run only quick tests (exclude intensive ones)
python -m pytest tests/ -v -m "memory and not memory_intensive"

# Run with detailed memory tracking
python -m pytest tests/test_memory_bounds.py -v --durations=10
```

### Writing Memory Tests

```python
import pytest
from tests.conftest import memory_limit, assert_memory_bounded

@pytest.mark.memory
def test_agent_memory_bounded(memory_tracker):
    """Test that agent memory stays within limits."""
    agent = MyAgent(agent_id="test")
    
    # Initial checkpoint
    memory_tracker.checkpoint("start")
    
    # Simulate workload
    for i in range(1000):
        agent.process_interaction(...)
    
    # Check memory
    memory_tracker.checkpoint("after_1000")
    memory_tracker.assert_bounded(max_mb=100)
    
    # Continue workload
    for i in range(1000, 10000):
        agent.process_interaction(...)
    
    # Verify memory stabilized
    memory_tracker.checkpoint("after_10000")
    memory_tracker.assert_stable(tolerance_mb=5.0)

@pytest.mark.memory
@memory_limit(max_mb=500)
def test_with_hard_limit():
    """Test with process-level memory limit."""
    # If this test exceeds 500 MB, the process will be killed
    pass
```

## Test Markers

### `@pytest.mark.memory`
- Marks tests that focus on memory behavior
- Runs in short CI builds (100 epochs, <5 minutes)
- Use for standard memory bounds tests

### `@pytest.mark.memory_intensive`
- Marks long-running memory stress tests
- Runs in weekly scheduled CI (1000+ epochs, <30 minutes)
- Use for extended stability tests

## Memory Baselines

The file `tests/memory_baselines.json` contains expected memory usage for each agent type:

| Agent Type | Max Memory | Test Duration |
|------------|-----------|---------------|
| CrewBackedAgent | 200 MB | 1000 epochs |
| LLMAgent | 50 MB | 1000 exchanges |
| LDTAgent | 100 MB | 100+ peers |
| RLMAgent | 150 MB | memory_budget=100 |
| BaseAgent | 50 MB | standard workload |
| AdaptiveAdversary | 75 MB | full tracking sets |

Tests fail if actual usage exceeds baseline by >20%.

## Memory Tracking Utilities

### `memory_tracker` fixture

Provides detailed memory tracking throughout a test:

```python
def test_with_tracking(memory_tracker):
    # Automatic start checkpoint
    
    # Record checkpoints
    memory_tracker.checkpoint("phase1")
    
    # Get current delta
    delta = memory_tracker.get_delta_mb()
    
    # Assert bounded
    memory_tracker.assert_bounded(max_mb=100)
    
    # Assert stable
    memory_tracker.assert_stable(tolerance_mb=5.0)
```

### `@memory_limit(max_mb=N)` decorator

Enforces hard process-level memory limit:

```python
@memory_limit(max_mb=500)
def test_hard_limit():
    # Process will be killed if it exceeds 500 MB
    pass
```

### Helper functions

```python
from tests.conftest import (
    assert_memory_bounded,
    assert_memory_stable,
    get_memory_usage_mb,
    get_peak_memory_mb,
)

# Direct assertions
assert_memory_bounded(max_mb=100)
assert_memory_stable(checkpoints=[95.2, 96.1, 95.8], tolerance_mb=2.0)

# Get current usage
current = get_memory_usage_mb()
peak = get_peak_memory_mb()
```

## CI Workflows

### Short Run (Every PR)
- **Trigger:** Every pull request and push to main
- **Duration:** ~5 minutes per Python version
- **Tests:** Standard memory bounds (100 epochs)
- **Python:** 3.10, 3.11, 3.12

### Extended Run (Weekly)
- **Trigger:** Scheduled (Sundays 2 AM UTC) or manual
- **Duration:** ~30 minutes per Python version
- **Tests:** Memory-intensive stress tests (1000+ epochs)
- **Python:** 3.10, 3.11, 3.12

### Manual Runs

Trigger manually via GitHub Actions UI:
1. Go to Actions → "Memory Management Tests"
2. Click "Run workflow"
3. Select test mode: `short` or `long`

## Memory Leak Detection

The test suite validates fixes for known unbounded growth issues:

### ✅ Fixed Issues
1. **BaseAgent deques**: `_memory` and `_interaction_history` capped at 1000 entries
2. **RLMAgent counterparty models**: LRU eviction at 100 models
3. **RLMAgent per-model history**: Capped at 50 entries per model
4. **AdaptiveAdversary tracking sets**: All sets capped at 200 entries
5. **PromptAuditLog**: Entries capped at 50,000 (default)

### Test Coverage

| Component | Test | What It Validates |
|-----------|------|-------------------|
| BaseAgent._memory | `test_memory_deque_cap` | Deque bounded at MAX_MEMORY_SIZE |
| BaseAgent._interaction_history | `test_interaction_history_deque_cap` | Deque bounded at MAX_INTERACTION_HISTORY |
| RLMAgent.counterparty_models | `test_counterparty_model_eviction` | LRU eviction when max reached |
| RLMAgent per-model history | `test_per_model_history_cap` | History capped per model |
| AdaptiveAdversary tracking | `test_vulnerable_targets_capped` | Set size limited |
| PromptAuditLog | `test_entries_capped` | Stops writing after max_entries |

## Troubleshooting

### Tests fail with MemoryError

If tests are killed due to excessive memory:
1. Check if the memory growth is expected for the test
2. Review baseline thresholds in `memory_baselines.json`
3. Verify deque/collection caps are correctly configured
4. Use `memory_tracker` to identify growth phases

### Memory profiling shows unexpected usage

```bash
# Profile a specific test
python -m memory_profiler tests/test_memory_bounds.py::TestClass::test_name

# Get detailed memory line-by-line
mprof run python -m pytest tests/test_memory_bounds.py::test_name
mprof plot
```

### Can't set memory limits on Windows

The `@memory_limit` decorator only works on Unix-like systems. Windows tests will skip limit enforcement but still track memory usage.

## Best Practices

1. **Always use bounded collections**: Use `deque(maxlen=N)` instead of list
2. **Implement LRU eviction**: For caches and model storage
3. **Track memory in long runs**: Use `memory_tracker` fixture
4. **Set realistic baselines**: Based on production workloads
5. **Test at multiple scales**: Short (100) and long (1000+) epochs
6. **Monitor CI trends**: Watch for gradual increases

## References

- Test Suite: `tests/test_memory_bounds.py`
- Baselines: `tests/memory_baselines.json`
- CI Workflow: `.github/workflows/test-memory-limits.yml`
- Memory Helpers: `tests/conftest.py`

## Contact

For questions or issues with memory testing:
- File an issue: https://github.com/swarm-ai-research/swarm/issues
- Tag: `memory`, `testing`, `ci`
