"""
Robust Parsing - A-Team Approved Generic Parsing
================================================

A-Team Decision: NO REGEX for LLM output parsing.

This module provides robust parsing that works with any LLM output format:
1. Tries JSON parsing first
2. Falls back to structured extraction
3. Returns None instead of magic numbers on failure
4. Never uses regex for value extraction

Generic enough for any agentic system.
"""

import json
import logging
from typing import Optional, Any, Dict, Tuple, Union

logger = logging.getLogger(__name__)


def parse_float_robust(value: Any, default: Optional[float] = None) -> Optional[float]:
    """
    Robustly parse a float from any input.
    
    A-Team Approved: No regex, handles all edge cases.
    
    Works with:
    - float/int: 0.7, 1
    - string: "0.7", "0.7%", "70%", "approximately 0.7"
    - dict: {"value": 0.7}
    - None/empty: returns default
    
    Returns None on failure (not magic number).
    """
    if value is None:
        return default
    
    # Already a number
    if isinstance(value, (int, float)):
        return float(value)
    
    # Dict with value key
    if isinstance(value, dict):
        for key in ['value', 'score', 'q_value', 'confidence', 'result']:
            if key in value:
                return parse_float_robust(value[key], default)
        return default
    
    # String parsing
    if isinstance(value, str):
        value = value.strip()
        
        if not value:
            return default
        
        # Try direct float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # Handle percentage
        if value.endswith('%'):
            try:
                return float(value[:-1]) / 100.0
            except ValueError:
                pass
        
        # Try JSON parsing
        try:
            parsed = json.loads(value)
            return parse_float_robust(parsed, default)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Extract first number-like substring (no regex!)
        # Walk through string finding digits and decimal points
        num_str = ""
        in_number = False
        has_decimal = False
        
        for char in value:
            if char.isdigit():
                num_str += char
                in_number = True
            elif char == '.' and in_number and not has_decimal:
                num_str += char
                has_decimal = True
            elif in_number:
                # End of number
                break
        
        if num_str:
            try:
                result = float(num_str)
                # Normalize if looks like percentage
                if result > 1.0 and 'percent' in value.lower():
                    result /= 100.0
                return result
            except ValueError:
                pass
    
    return default


def parse_bool_robust(value: Any, default: bool = False) -> bool:
    """
    Robustly parse a boolean from any input.
    
    Works with:
    - bool: True, False
    - string: "true", "yes", "1", "proceed", "valid"
    - int: 0, 1
    - None: returns default
    """
    if value is None:
        return default
    
    if isinstance(value, bool):
        return value
    
    if isinstance(value, (int, float)):
        return value > 0
    
    if isinstance(value, str):
        value_lower = value.lower().strip()
        
        positive = {'true', 'yes', '1', 'proceed', 'valid', 'accept', 'approved', 'pass'}
        negative = {'false', 'no', '0', 'block', 'invalid', 'reject', 'denied', 'fail'}
        
        if value_lower in positive:
            return True
        if value_lower in negative:
            return False
    
    return default


def parse_json_robust(value: Any) -> Optional[Dict]:
    """
    Robustly parse JSON from any input.
    
    Works with:
    - dict: returns as-is
    - string: tries JSON parsing
    - string with markdown code blocks: extracts JSON
    """
    if value is None:
        return None
    
    if isinstance(value, dict):
        return value
    
    if isinstance(value, str):
        value = value.strip()
        
        # Try direct parsing
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # Try extracting from markdown code blocks (no regex!)
        if '```' in value:
            # Find content between ``` markers
            parts = value.split('```')
            for i, part in enumerate(parts):
                if i % 2 == 1:  # Odd indices are inside code blocks
                    # Remove language tag if present
                    lines = part.strip().split('\n')
                    if lines and lines[0] in ['json', 'JSON', '']:
                        content = '\n'.join(lines[1:])
                    else:
                        content = part.strip()
                    
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        continue
        
        # Try finding JSON object in string (no regex!)
        # Look for {...} pattern
        start = value.find('{')
        if start >= 0:
            # Find matching closing brace
            depth = 0
            end = start
            for i, char in enumerate(value[start:], start):
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            
            if end > start:
                try:
                    return json.loads(value[start:end])
                except json.JSONDecodeError:
                    pass
    
    return None


class AdaptiveThreshold:
    """
    Adaptive threshold that learns from data.
    
    A-Team Approved: No hardcoded thresholds like 0.8 or 0.2.
    Uses running statistics to determine what's "high" or "low".
    """
    
    def __init__(self, initial_mean: float = 0.5, initial_std: float = 0.2):
        self.mean = initial_mean
        self.std = initial_std
        self.count = 0
        self.m2 = 0  # For Welford's algorithm
    
    def update(self, value: float):
        """Update running statistics (Welford's algorithm)."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        
        if self.count > 1:
            self.std = (self.m2 / (self.count - 1)) ** 0.5
    
    def is_high(self, value: float, sigma: float = 1.5) -> bool:
        """Check if value is significantly above mean."""
        return value > self.mean + sigma * max(self.std, 0.1)
    
    def is_low(self, value: float, sigma: float = 1.5) -> bool:
        """Check if value is significantly below mean."""
        return value < self.mean - sigma * max(self.std, 0.1)
    
    def is_extreme(self, value: float, sigma: float = 2.0) -> bool:
        """Check if value is extreme (either direction)."""
        deviation = abs(value - self.mean)
        return deviation > sigma * max(self.std, 0.1)


class EpsilonGreedy:
    """
    Deterministic epsilon-greedy decision making.
    
    A-Team Approved: No random.random() > 0.5 fallbacks.
    """
    
    def __init__(self, initial_epsilon: float = 0.3, decay: float = 0.99, min_epsilon: float = 0.05):
        self.epsilon = initial_epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon
        self.decision_count = 0
    
    def should_explore(self) -> bool:
        """
        Deterministic exploration based on decision count.
        
        Returns True for exploration with frequency epsilon.
        NOT random - uses hash of decision count for determinism.
        """
        self.decision_count += 1
        
        # Use hash for deterministic pseudo-randomness
        hash_val = hash(self.decision_count) % 1000 / 1000.0
        
        should_explore = hash_val < self.epsilon
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
        
        return should_explore
    
    def decide(self, exploit_decision: bool) -> bool:
        """
        Make decision: explore (True) or exploit (use exploit_decision).
        """
        if self.should_explore():
            return True  # Explore: try new thing
        return exploit_decision  # Exploit: use learned strategy


def safe_hash(content: Any, max_length: Optional[int] = None) -> int:
    """
    Safe hash that handles any input.
    
    A-Team Approved: No hash(content) assumptions.
    """
    if content is None:
        return 0
    
    # Convert to string if needed
    if not isinstance(content, str):
        content = str(content)
    
    # Truncate if needed, but handle edge cases
    if max_length and len(content) > max_length:
        content = content[:max_length]
    
    return hash(content)


def content_similarity(content1: Any, content2: Any, threshold: float = 0.8) -> bool:
    """
    Simple content similarity check without external dependencies.
    
    Uses character overlap, not regex.
    """
    if content1 is None or content2 is None:
        return content1 is content2
    
    s1 = str(content1).lower()
    s2 = str(content2).lower()
    
    if not s1 or not s2:
        return s1 == s2
    
    # Simple word overlap
    words1 = set(s1.split())
    words2 = set(s2.split())
    
    if not words1 or not words2:
        return s1 == s2
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    jaccard = intersection / union if union > 0 else 0
    return jaccard >= threshold


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'parse_float_robust',
    'parse_bool_robust',
    'parse_json_robust',
    'AdaptiveThreshold',
    'EpsilonGreedy',
    'safe_hash',
    'content_similarity'
]

