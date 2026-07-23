---
description: "The provenance trace schema extends the existing Event model with deterministic IDs and parent-child relationships, enabling:"
---

# Unified Provenance Trace Schema

This document describes the unified provenance trace schema and exporters added to SWARM's event logging system.

## Overview

The provenance trace schema extends the existing `Event` model with deterministic IDs and parent-child relationships, enabling:

- **Complete audit trails** for tool calls, artifacts, audits, and interventions
- **Deterministic IDs** that remain stable across replays
- **CSV export** for analysis with standard data tools
- **Parent-child linking** to reconstruct causal chains

## Key Features

### 1. Provenance Fields on Events

All events now support the following optional fields:

| Field | Type | Description |
|-------|------|-------------|
| `provenance_id` | `str` | Deterministic 12-char hex ID for this event in provenance chain |
| `parent_event_id` | `str` | Links to parent event for causal chains |
| `tool_call_id` | `str` | ID for tool call if this is a tool execution |
| `artifact_id` | `str` | ID for artifact if this event produces/references one |
| `audit_id` | `str` | ID for audit if this is an audit event |
| `intervention_id` | `str` | ID for governance intervention |

### 2. Deterministic ID Generation

```python
from swarm.models.events import generate_deterministic_id

# Generate a stable ID based on event attributes
prov_id = generate_deterministic_id(
    event_type="tool_call",
    agent_id="agent_1",
    timestamp=timestamp,
    tool_name="query_db",
)
```

Deterministic IDs are:
- **Stable**: Same inputs always generate the same ID
- **Compact**: 12-character hex strings (first 12 chars of SHA-256)
- **Collision-resistant**: Based on cryptographic hash

### 3. Provenance-Aware Event Factories

Convenience functions for creating events with provenance tracking:

```python
from swarm.models.events import (
    tool_call_executed_event,
    artifact_created_event,
    audit_event,
    intervention_event,
    agent_message_event,
)

# Tool call with automatic ID generation
tool_evt = tool_call_executed_event(
    agent_id="agent_1",
    tool_name="query_database",
    arguments={"query": "SELECT 1"},
    result={"rows": 1},
    success=True,
    parent_event_id=parent_id,  # Optional: link to parent
)

# Artifact creation
artifact_evt = artifact_created_event(
    agent_id="agent_1",
    artifact_id="artifact_123",
    artifact_type="memory",
    artifact_data={"title": "Test", "summary": "..."},
    parent_event_id=tool_evt.provenance_id,
)

# Audit event
audit_evt = audit_event(
    agent_id="auditor",
    audit_type="safety",
    audit_result={"decision": "approve", "risk": 0.1},
    audited_event_id=tool_evt.event_id,
    parent_event_id=tool_evt.provenance_id,
)

# Intervention event
intervention_evt = intervention_event(
    agent_id="governance",
    intervention_type="freeze",
    intervention_action="freeze_agent",
    intervention_reason="suspicious_behavior",
    affected_agents=["agent_2"],
    parent_event_id=audit_evt.provenance_id,
)
```

### 4. CSV Export

Two export methods are available:

#### Full CSV Export

Exports all events with all fields:

```python
from swarm.logging.event_log import EventLog

log = EventLog("events.jsonl")

# Export with standard columns
log.to_csv("events.csv")

# Include payload as JSON string
log.to_csv("events.csv", include_payload=True)
```

Columns: `event_id`, `timestamp`, `event_type`, `agent_id`, `provenance_id`, `parent_event_id`, `tool_call_id`, `artifact_id`, `audit_id`, `intervention_id`, etc.

#### Provenance-Focused CSV Export

Exports only events with provenance tracking, with extracted key fields:

```python
log.to_provenance_csv("provenance.csv")
```

Columns: `provenance_id`, `event_type`, `timestamp`, `agent_id`, `parent_event_id`, `tool_call_id`, `artifact_id`, `audit_id`, `intervention_id`, `tool_name`, `artifact_type`, `audit_type`, `intervention_type`, `success`, `payload_summary`

This format is optimized for:
- Reconstructing provenance chains
- Analyzing tool usage patterns
- Auditing governance interventions
- Tracking artifact lineage

## Usage Examples

### Basic Provenance Chain

```python
from swarm.logging.event_log import EventLog
from swarm.models.events import tool_call_executed_event, audit_event

log = EventLog("log.jsonl")

# 1. Tool call
tool_evt = tool_call_executed_event(
    agent_id="agent_1",
    tool_name="risky_op",
    arguments={"danger": "high"},
    result={"status": "executed"},
    success=True,
)
log.append(tool_evt)

# 2. Audit the tool call
audit_evt = audit_event(
    agent_id="auditor",
    audit_type="safety",
    audit_result={"flagged": True, "risk": 0.9},
    audited_event_id=tool_evt.event_id,
    parent_event_id=tool_evt.provenance_id,  # Link to tool call
)
log.append(audit_evt)

# 3. Export to CSV
log.to_provenance_csv("provenance.csv")
```

### Querying Provenance Chains

```python
# Read back from JSONL
for event in log.replay():
    if event.provenance_id:
        print(f"Event: {event.event_type}")
        print(f"  Provenance ID: {event.provenance_id}")
        print(f"  Parent: {event.parent_event_id or 'None'}")
        
        # Check specific IDs
        if event.tool_call_id:
            print(f"  Tool Call: {event.tool_call_id}")
        if event.audit_id:
            print(f"  Audit: {event.audit_id}")
```

### Analyzing CSV Output

The CSV exports are designed to work with standard data tools:

```python
import pandas as pd

# Load provenance CSV
df = pd.read_csv("provenance.csv")

# Count tool calls by agent
tool_calls = df[df["tool_call_id"] != ""]
print(tool_calls.groupby("agent_id").size())

# Find audit chains
audits = df[df["audit_id"] != ""]
print(audits[["provenance_id", "parent_event_id", "audit_type"]])

# Trace interventions back to their triggers
interventions = df[df["intervention_id"] != ""]
for _, row in interventions.iterrows():
    parent_id = row["parent_event_id"]
    parent = df[df["provenance_id"] == parent_id]
    if not parent.empty:
        print(f"Intervention {row['provenance_id']} triggered by {parent.iloc[0]['event_type']}")
```

## Backward Compatibility

The provenance fields are **optional** and **backward compatible**:

- Old events without provenance fields deserialize correctly
- Existing code continues to work without changes
- JSONL logs can mix old and new event formats
- CSV export handles missing fields gracefully

## Testing

Comprehensive tests are in `tests/test_provenance_trace.py`:

```bash
pytest tests/test_provenance_trace.py -v
```

Test coverage includes:
- Deterministic ID generation
- Provenance-aware event factories
- Event serialization/deserialization
- CSV export formats
- Provenance chain reconstruction
- Backward compatibility

## Example Script

Run the full example demonstrating all features:

```bash
python examples/provenance_tracking.py
```

This creates a complete provenance chain with tool calls, artifacts, audits, and interventions, then exports to both JSONL and CSV formats.

## Design Rationale

### Why Deterministic IDs?

- **Reproducibility**: Same simulation with same seed generates same IDs
- **Stable references**: IDs remain valid across log replays
- **Efficient**: No need for centralized ID allocation
- **Verifiable**: Anyone can recompute IDs from event data

### Why Parent-Child Linking?

- **Causal chains**: Trace root causes of interventions
- **Audit trails**: Answer "what triggered this?"
- **Debugging**: Understand event sequences
- **Governance**: Justify decisions with evidence chain

### Why Two CSV Formats?

1. **Full CSV**: For comprehensive analysis and data warehousing
2. **Provenance CSV**: For focused investigation of specific chains

Different use cases need different views of the same data.

## Future Enhancements

Potential future additions:

- [ ] Graph export for visualization (e.g., DOT/Graphviz)
- [ ] Provenance query DSL for complex filtering
- [ ] Automatic chain validation (detect cycles, orphans)
- [ ] Integration with existing metrics exporters
- [ ] Provenance-aware event replay with filtering

## Related Documentation

- [Event Schema](https://github.com/swarm-ai-research/swarm/blob/main/swarm/models/events.py) — Full Event model
- [Event Log](https://github.com/swarm-ai-research/swarm/blob/main/swarm/logging/event_log.py) — JSONL logger
- [Export Utilities](https://github.com/swarm-ai-research/swarm/blob/main/swarm/analysis/export.py) — Simulation result exporters
- [Simulated API Schema](https://github.com/swarm-ai-research/swarm/blob/main/swarm/env/simulated_apis/SCHEMA.md) — Alternative provenance approach
