"""YAML schema validation and resource estimation for scenario submissions."""

from __future__ import annotations

import yaml

# Known optional keys and their expected types / constraints.
_INT_GE1_KEYS = {"agents", "n_agents", "epochs", "n_epochs", "steps_per_epoch", "steps"}
_INT_KEYS = {"seed"}


def _validate_int_ge1(key: str, value: object) -> str | None:
    if not isinstance(value, int) or isinstance(value, bool):
        return f"'{key}' must be an integer, got {type(value).__name__}"
    if value < 1:
        return f"'{key}' must be >= 1, got {value}"
    return None


def _validate_int(key: str, value: object) -> str | None:
    if not isinstance(value, int) or isinstance(value, bool):
        return f"'{key}' must be an integer, got {type(value).__name__}"
    return None


def validate_scenario_yaml(yaml_content: str) -> tuple[list[str], dict | None]:
    """Validate YAML content and return (errors, parsed_config_or_None).

    Validation rules:
    - Must parse successfully with yaml.safe_load.
    - Must parse to a dict (not None, not a string, not a list).
    - Known optional keys are type-checked when present.

    Returns:
        A 2-tuple of (error_list, parsed_dict_or_None).  If YAML cannot be
        parsed, parsed_dict is None.
    """
    errors: list[str] = []

    # Empty / whitespace-only content.
    if not yaml_content.strip():
        errors.append("YAML content cannot be empty")
        return errors, None

    # Attempt to parse.
    try:
        parsed = yaml.safe_load(yaml_content)
    except yaml.YAMLError as exc:
        errors.append(f"YAML parse error: {exc}")
        return errors, None

    # Must be a dict.
    if not isinstance(parsed, dict):
        type_name = type(parsed).__name__ if parsed is not None else "NoneType"
        errors.append(
            f"YAML must parse to a mapping (dict), got {type_name}"
        )
        return errors, None

    # Type-check known optional keys.
    for key, value in parsed.items():
        if key in _INT_GE1_KEYS:
            err = _validate_int_ge1(key, value)
            if err:
                errors.append(err)
        elif key in _INT_KEYS:
            err = _validate_int(key, value)
            if err:
                errors.append(err)

    return errors, parsed


def estimate_resources(parsed_config: dict) -> dict:
    """Estimate runtime and memory from a parsed scenario config.

    Extracts agent count, epoch count, and step count (with defaults) and
    produces rough resource estimates.

    Returns:
        A dict with keys: n_agents, n_epochs, steps, estimated_interactions,
        estimated_runtime_seconds, estimated_memory_mb, within_limits.
    """
    n_agents = parsed_config.get("agents", parsed_config.get("n_agents", 10))
    n_epochs = parsed_config.get("epochs", parsed_config.get("n_epochs", 10))
    steps = parsed_config.get("steps_per_epoch", parsed_config.get("steps", 10))

    estimated_interactions = n_agents * n_epochs * steps
    estimated_runtime_seconds = estimated_interactions * 0.01
    estimated_memory_mb = n_agents * 2 + estimated_interactions * 0.001

    return {
        "n_agents": n_agents,
        "n_epochs": n_epochs,
        "steps": steps,
        "estimated_interactions": estimated_interactions,
        "estimated_runtime_seconds": estimated_runtime_seconds,
        "estimated_memory_mb": estimated_memory_mb,
        "within_limits": estimated_interactions < 1_000_000,
    }
