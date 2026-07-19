"""Scenario generator for creating YAML scenarios from text descriptions.

Generates scenario dictionaries from free-text input using rule-based
entity extraction. Output is valid YAML compatible with swarm.scenarios.loader.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from swarm.scenarios.text_processor import EntityExtractor, TextChunker

# Domain-specific default configurations
DOMAIN_DEFAULTS = {
    'market': {
        'governance': {
            'transaction_tax_rate': 0.05,
            'transaction_tax_split': 0.5,
            'reputation_decay_rate': 0.95,
            'bandwidth_cap': 10,
            'staking_enabled': False,
            'circuit_breaker_enabled': True,
            'freeze_threshold_toxicity': 0.7,
            'audit_enabled': True,
            'audit_probability': 0.1,
        },
        'payoff': {
            's_plus': 2.0,
            's_minus': 1.0,
            'h': 2.0,
            'theta': 0.5,
            'rho_a': 0.1,
            'rho_b': 0.1,
            'w_rep': 1.0,
        },
        'simulation': {
            'n_epochs': 15,
            'steps_per_epoch': 10,
        },
    },
    'security': {
        'governance': {
            'transaction_tax_rate': 0.02,
            'reputation_decay_rate': 0.90,
            'circuit_breaker_enabled': True,
            'freeze_threshold_toxicity': 0.6,
            'freeze_threshold_violations': 3,
            'audit_enabled': True,
            'audit_probability': 0.2,
            'audit_penalty_multiplier': 2.0,
        },
        'payoff': {
            's_plus': 3.0,
            's_minus': 1.5,
            'h': 2.5,
            'theta': 0.5,
            'rho_a': 0.2,
            'rho_b': 0.2,
            'w_rep': 2.0,
        },
        'simulation': {
            'n_epochs': 20,
            'steps_per_epoch': 12,
        },
    },
    'social': {
        'governance': {
            'transaction_tax_rate': 0.01,
            'reputation_decay_rate': 0.98,
            'circuit_breaker_enabled': False,
            'audit_enabled': False,
        },
        'payoff': {
            's_plus': 1.5,
            's_minus': 0.5,
            'h': 1.5,
            'theta': 0.5,
            'rho_a': 0.05,
            'rho_b': 0.05,
            'w_rep': 1.5,
        },
        'simulation': {
            'n_epochs': 25,
            'steps_per_epoch': 8,
        },
    },
    'governance': {
        'governance': {
            'transaction_tax_rate': 0.03,
            'reputation_decay_rate': 0.92,
            'circuit_breaker_enabled': True,
            'audit_enabled': True,
            'audit_probability': 0.15,
        },
        'payoff': {
            's_plus': 2.5,
            's_minus': 1.0,
            'h': 2.0,
            'theta': 0.5,
            'rho_a': 0.15,
            'rho_b': 0.15,
            'w_rep': 1.5,
        },
        'simulation': {
            'n_epochs': 18,
            'steps_per_epoch': 10,
        },
    },
    'general': {
        'governance': {
            'transaction_tax_rate': 0.0,
            'transaction_tax_split': 0.5,
            'reputation_decay_rate': 1.0,
            'bandwidth_cap': 10,
            'staking_enabled': False,
            'circuit_breaker_enabled': False,
            'audit_enabled': False,
        },
        'payoff': {
            's_plus': 2.0,
            's_minus': 1.0,
            'h': 2.0,
            'theta': 0.5,
            'rho_a': 0.0,
            'rho_b': 0.0,
            'w_rep': 1.0,
        },
        'simulation': {
            'n_epochs': 10,
            'steps_per_epoch': 10,
        },
    },
}

# Default rate limits
DEFAULT_RATE_LIMITS = {
    'posts_per_epoch': 10,
    'interactions_per_step': 5,
    'votes_per_epoch': 50,
    'tasks_per_epoch': 3,
}

# Default success criteria
DEFAULT_SUCCESS_CRITERIA = {
    'min_epochs': 10,
    'min_agents': 3,
    'toxicity_threshold': 0.5,
}


class ScenarioGenerator:
    """Generate SWARM scenario YAML from text descriptions."""

    # Map agent type keywords to canonical SWARM agent types
    AGENT_TYPE_MAP = {
        'adversarial': 'adversarial',
        'deceptive': 'deceptive',
        'opportunistic': 'opportunistic',
        'cooperative': 'honest',
        'honest': 'honest',
    }

    @staticmethod
    def from_text(text: str, name: Optional[str] = None) -> Dict[str, Any]:
        """Generate a scenario dict from free-text description.

        Args:
            text: Free-text scenario description
            name: Optional scenario ID (generated if not provided)

        Returns:
            Dictionary conforming to SWARM scenario format
        """
        # Preprocess text
        text = TextChunker.preprocess(text)

        # Classify domain
        domain = EntityExtractor.classify_domain(text)

        # Extract entities
        entities = EntityExtractor.extract_entities(text)
        EntityExtractor.extract_relationships(text)

        # Infer agent composition from entities
        agent_types = ScenarioGenerator._infer_agent_types(entities, text)

        # Generate scenario dict
        scenario: Dict[str, Any] = {
            'scenario_id': name or f'generated_{domain}',
            'description': f'Generated scenario from text: {text[:100]}...',
            'agents': agent_types,
        }

        # Add domain defaults
        domain_defaults = DOMAIN_DEFAULTS.get(domain, DOMAIN_DEFAULTS['general'])
        governance = domain_defaults['governance']
        payoff = domain_defaults['payoff']
        simulation = domain_defaults['simulation']
        scenario['governance'] = dict(governance) if isinstance(governance, dict) else governance
        scenario['payoff'] = dict(payoff) if isinstance(payoff, dict) else payoff
        scenario['simulation'] = dict(simulation) if isinstance(simulation, dict) else simulation

        # Add standard sections
        scenario['rate_limits'] = DEFAULT_RATE_LIMITS.copy()
        scenario['success_criteria'] = DEFAULT_SUCCESS_CRITERIA.copy()

        # Update success criteria based on inferred agent count
        scenario['success_criteria']['min_agents'] = max(3, len(agent_types))

        return scenario

    @staticmethod
    def from_file(path: str, name: Optional[str] = None) -> Dict[str, Any]:
        """Generate scenario from a text file.

        Args:
            path: Path to .txt or .md file
            name: Optional scenario ID

        Returns:
            Scenario dictionary
        """
        text = TextChunker.extract_from_file(path)
        return ScenarioGenerator.from_text(text, name=name)

    @staticmethod
    def to_yaml(scenario_dict: Dict[str, Any], output_path: str) -> None:
        """Write scenario YAML to file.

        Args:
            scenario_dict: Scenario dictionary
            output_path: Path to write YAML file
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, 'w') as f:
            yaml.dump(scenario_dict, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def _infer_agent_types(
        entities: List[Dict],
        text: str,
    ) -> List[Dict[str, Any]]:
        """Infer agent composition from extracted entities.

        Args:
            entities: Extracted entities
            text: Original text

        Returns:
            List of agent type specs
        """
        agent_type_counts: Dict[str, int] = {}

        # Count agent type mentions
        for entity in entities:
            if entity.get('type') == 'agent_type':
                subtype = entity.get('subtype', 'honest')
                canonical_type = ScenarioGenerator.AGENT_TYPE_MAP.get(
                    subtype, 'honest'
                )
                agent_type_counts[canonical_type] = (
                    agent_type_counts.get(canonical_type, 0) + 1
                )

        # If no explicit agent types found, use heuristics
        if not agent_type_counts:
            # Default: mix of honest and one adversarial
            if any(word in text.lower() for word in ['attack', 'threat', 'malicious']):
                agent_type_counts = {
                    'honest': 3,
                    'adversarial': 1,
                }
            elif any(word in text.lower() for word in ['cheat', 'game', 'exploit']):
                agent_type_counts = {
                    'honest': 2,
                    'opportunistic': 1,
                    'deceptive': 1,
                }
            else:
                # Safe default
                agent_type_counts = {'honest': 3}

        # Convert to agent specs
        agent_specs = [
            {'type': agent_type, 'count': count}
            for agent_type, count in sorted(agent_type_counts.items())
        ]

        return agent_specs

    @staticmethod
    def _infer_agent_count(text: str, agent_types: List[Dict]) -> int:
        """Infer total number of agents.

        Args:
            text: Original text
            agent_types: Agent type specs

        Returns:
            Total agent count
        """
        # Check for explicit numbers (e.g., "5 agents", "10 participants")
        pattern = r'(\d+)\s*(?:agents?|participants?|actors?|nodes?)'
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return int(matches[0])

        # Fall back to sum of counts in agent specs
        return sum(spec.get('count', 1) for spec in agent_types)
