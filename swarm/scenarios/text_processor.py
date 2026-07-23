"""Text processing utilities for scenario generation from documents.

This module provides tools to extract entities, relationships, and domain
information from free-text descriptions, enabling rule-based scenario generation.
No LLM required - uses heuristic-based extraction.
"""

import re
from pathlib import Path
from typing import Dict, List

try:
    import chardet
except ImportError:
    chardet = None


class TextChunker:
    """Split text into overlapping chunks with preprocessing."""

    @staticmethod
    def preprocess(text: str) -> str:
        """Normalize whitespace, strip empty lines.

        Args:
            text: Raw text input

        Returns:
            Cleaned text with normalized whitespace
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text

    @staticmethod
    def extract_from_file(path: str) -> str:
        """Read text from .txt or .md files with encoding detection.

        Args:
            path: Path to text or markdown file

        Returns:
            File contents as string

        Raises:
            ValueError: If file is not .txt or .md
            IOError: If file cannot be read
        """
        file_path = Path(path)

        if file_path.suffix.lower() not in ['.txt', '.md']:
            raise ValueError(f"File must be .txt or .md, got {file_path.suffix}")

        raw_data = file_path.read_bytes()

        # Try chardet for encoding detection
        if chardet:
            detected = chardet.detect(raw_data)
            encoding = detected.get('encoding', 'utf-8') if detected else 'utf-8'
        else:
            encoding = 'utf-8'

        try:
            return raw_data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            # Fall back to utf-8 with error handling
            return raw_data.decode('utf-8', errors='replace')

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks.

        Args:
            text: Text to chunk
            chunk_size: Size of each chunk (in characters)
            overlap: Character overlap between consecutive chunks

        Returns:
            List of text chunks
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        text = TextChunker.preprocess(text)
        chunks = []

        # For text shorter than chunk_size, return as single chunk
        if len(text) <= chunk_size:
            return [text] if text else []

        # Validate overlap now that we know text is large enough
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be in [0, chunk_size)")

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)

            # Stop if we reached the end
            if end >= len(text):
                break

            # Move start by (chunk_size - overlap)
            start = end - overlap

        return chunks


class EntityExtractor:
    """Rule-based entity and relationship extraction."""

    # Keywords that indicate agent type/behavior
    AGENT_TYPE_KEYWORDS = {
        'adversarial': [
            'adversarial', 'attack', 'malicious', 'attacker', 'exploit',
            'hostile', 'harmful', 'destructive', 'sabotage',
        ],
        'deceptive': [
            'deceptive', 'deceive', 'fraud', 'lie', 'mislead', 'mask',
            'hide', 'conceal', 'obfuscate', 'camouflage',
        ],
        'opportunistic': [
            'opportunistic', 'selfish', 'self-interest', 'greedy', 'free-ride',
            'cheat', 'game', 'exploit', 'cut corners',
        ],
        'cooperative': [
            'cooperative', 'collaborate', 'cooperate', 'honest', 'sincere',
            'trustworthy', 'reliable', 'fair',
        ],
        'honest': [
            'honest', 'truthful', 'integrity', 'benign', 'good', 'well-intentioned',
        ],
    }

    # Domain classification keywords
    DOMAIN_KEYWORDS = {
        'market': [
            'market', 'trading', 'transaction', 'buy', 'sell', 'price',
            'buyer', 'seller', 'merchant', 'commerce', 'exchange', 'bounty',
        ],
        'social': [
            'social', 'network', 'community', 'relationship', 'trust',
            'friendship', 'interaction', 'vote', 'peer',
        ],
        'security': [
            'security', 'attack', 'threat', 'defense', 'protect', 'breach',
            'vulnerability', 'intrusion', 'malware', 'audit',
        ],
        'governance': [
            'governance', 'policy', 'rule', 'regulation', 'decision',
            'voting', 'committee', 'council', 'mechanism',
        ],
    }

    @staticmethod
    def extract_entities(text: str) -> List[Dict]:
        """Extract named entities using heuristics.

        Identifies:
        - Capitalized multi-word phrases (agents/organizations)
        - Numbers with surrounding context (parameters)
        - Agent type keywords

        Args:
            text: Text to extract entities from

        Returns:
            List of dicts with keys: name, type, context
        """
        entities = []

        # Extract capitalized multi-word phrases (potential agent names)
        # Match sequences of capitalized words
        cap_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
        for match in re.finditer(cap_pattern, text):
            phrase = match.group(1)
            # Ignore common words
            if phrase not in ['The', 'This', 'A', 'An', 'In', 'On', 'At']:
                entities.append({
                    'name': phrase,
                    'type': 'entity',
                    'context': text[max(0, match.start()-30):match.end()+30],
                })

        # Extract numbers with context
        num_pattern = r'(\d+(?:\.\d+)?)\s*([a-z\s]*(?:agent|actor|node|participant))'
        for match in re.finditer(num_pattern, text, re.IGNORECASE):
            entities.append({
                'name': f"{match.group(1)} {match.group(2)}",
                'type': 'parameter',
                'context': text[max(0, match.start()-30):match.end()+30],
                'value': match.group(1),
            })

        # Extract agent type mentions
        for agent_type, keywords in EntityExtractor.AGENT_TYPE_KEYWORDS.items():
            for keyword in keywords:
                pattern = rf'\b{re.escape(keyword)}\b'
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entities.append({
                        'name': keyword,
                        'type': 'agent_type',
                        'subtype': agent_type,
                        'context': text[max(0, match.start()-30):match.end()+30],
                    })

        return entities

    @staticmethod
    def extract_relationships(text: str) -> List[Dict]:
        """Extract relationships like "A attacks B", "A cooperates with B".

        Args:
            text: Text to extract relationships from

        Returns:
            List of dicts with keys: subject, predicate, object
        """
        relationships = []

        # Pattern: [Entity] [verb] [Entity]
        # Common relationship verbs
        verbs = [
            'attacks', 'attack', 'cooperates', 'cooperate', 'betrays', 'betray',
            'trusts', 'trust', 'opposes', 'oppose', 'helps', 'help',
            'communicates', 'communicate', 'coordinates', 'coordinate',
            'collude', 'colludes', 'sabotage', 'sabotages',
        ]

        cap_entity = r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*'

        for verb in verbs:
            pattern = rf'({cap_entity})\s+{verb}s?\s+(?:with\s+)?({cap_entity})'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                relationships.append({
                    'subject': match.group(1),
                    'predicate': verb,
                    'object': match.group(2),
                    'context': text[max(0, match.start()-50):match.end()+50],
                })

        return relationships

    @staticmethod
    def classify_domain(text: str) -> str:
        """Classify text into domain categories.

        Args:
            text: Text to classify

        Returns:
            Domain name: 'market', 'social', 'security', 'governance', or 'general'
        """
        text_lower = text.lower()
        domain_scores = {}

        for domain, keywords in EntityExtractor.DOMAIN_KEYWORDS.items():
            score = sum(text_lower.count(kw) for kw in keywords)
            domain_scores[domain] = score

        # Return highest-scoring domain, or 'general' if no clear match
        if domain_scores and max(domain_scores.values()) > 0:
            return max(domain_scores, key=lambda d: domain_scores[d])
        return 'general'
