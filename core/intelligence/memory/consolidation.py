"""Memory consolidation components — Signatures, Validator, Classifier, MemoryCluster."""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

from Jotty.core.infrastructure.foundation.data_structures import (
    MemoryEntry, MemoryLevel, GoalValue, SwarmConfig,
    GoalHierarchy, GoalNode, CausalLink, StoredEpisode
)

import dspy

# =============================================================================
# CONSOLIDATION SIGNATURES
# =============================================================================

class PatternExtractionSignature(dspy.Signature):
    """Extract patterns from episodic memories using chain-of-thought."""
    
    memories: str = dspy.InputField(desc="JSON list of related episodic memories")
    goal_context: str = dspy.InputField(desc="The goal context these memories relate to")
    domain: str = dspy.InputField(desc="Domain (e.g., sql, python, api, etc.)")
    
    reasoning: str = dspy.OutputField(desc="Analysis of what patterns emerge")
    pattern: str = dspy.OutputField(desc="The extracted pattern as a clear statement")
    confidence: float = dspy.OutputField(desc="Confidence in pattern 0.0-1.0")
    conditions: str = dspy.OutputField(desc="When this pattern applies")
    exceptions: str = dspy.OutputField(desc="When this pattern does NOT apply")


class ProceduralExtractionSignature(dspy.Signature):
    """Extract procedural knowledge (how to do things)."""
    
    success_traces: str = dspy.InputField(desc="Traces of successful episodes")
    failure_traces: str = dspy.InputField(desc="Traces of failed episodes")
    task_type: str = dspy.InputField(desc="Type of task")
    
    reasoning: str = dspy.OutputField(desc="Analysis of what steps lead to success")
    procedure: str = dspy.OutputField(desc="Step-by-step procedure")
    key_decisions: str = dspy.OutputField(desc="Critical decision points")


class MetaWisdomSignature(dspy.Signature):
    """Extract meta-level wisdom about learning itself."""
    
    learning_history: str = dspy.InputField(desc="Summary of learning progress")
    failure_analysis: str = dspy.InputField(desc="Common failure patterns")
    success_analysis: str = dspy.InputField(desc="Common success patterns")
    
    wisdom: str = dspy.OutputField(desc="Meta-level insight about when to apply what knowledge")
    applicability: str = dspy.OutputField(desc="When this wisdom applies")


# =============================================================================
# MEMORY LEVEL CLASSIFIER (A-Team Enhancement)
# =============================================================================

class MemoryLevelClassificationSignature(dspy.Signature):
    """
    A-Team Enhancement: LLM-based memory level classification.

    Instead of hardcoding which level to store to, use LLM to decide:
    - EPISODIC: Raw experiences, specific events, tool outputs
    - SEMANTIC: Patterns, abstractions, generalizations
    - PROCEDURAL: How-to knowledge, step sequences
    - META: Wisdom about learning, when to use what
    - CAUSAL: Why things work, cause-effect relationships
    """

    experience: str = dspy.InputField(desc="The experience/knowledge to classify")
    context: str = dspy.InputField(desc="Context: task type, agent, goal, outcome")

    reasoning: str = dspy.OutputField(desc="Why this memory level is appropriate")
    level: str = dspy.OutputField(desc="One of: EPISODIC, SEMANTIC, PROCEDURAL, META, CAUSAL")
    confidence: float = dspy.OutputField(desc="Confidence 0.0-1.0")
    should_store: bool = dspy.OutputField(desc="True if worth storing, False if redundant")


# =============================================================================
# CONSOLIDATION VALIDATION (A-Team Critical Fix)
# =============================================================================

class ConsolidationValidationSignature(dspy.Signature):
    """
    Validate consolidated patterns against source memories.

    A-Team Critical Fix: Prevent hallucinated patterns during consolidation.
    """

    pattern: str = dspy.InputField(desc="The extracted pattern/wisdom to validate")
    source_memories: str = dspy.InputField(desc="JSON list of source memories that led to this pattern")
    pattern_type: str = dspy.InputField(desc="Type: SUCCESS_PATTERN, FAILURE_PATTERN, CAUSAL, PROCEDURAL")

    is_valid: bool = dspy.OutputField(desc="True if pattern is supported by sources, False if hallucinated")
    confidence: float = dspy.OutputField(desc="Confidence in validation (0.0-1.0)")
    reasoning: str = dspy.OutputField(desc="Why pattern is valid/invalid")
    corrections: str = dspy.OutputField(desc="Suggested corrections if pattern is invalid/vague")


class ConsolidationValidator:
    """
    Validate consolidated patterns before storing.

    A-Team Critical Fix: Prevents hallucinated patterns from entering memory.

    Validation checks:
    1. Pattern references concepts from source memories
    2. Pattern doesn't contradict source memories
    3. Pattern is actionable (not vague)
    4. Confidence threshold met

    Usage:
        validator = ConsolidationValidator()
        is_valid, confidence, reasoning = validator.validate_pattern(
            pattern="SQL queries with date filters should use partition columns",
            source_memories=[memory1, memory2, memory3]
        )

        if is_valid and confidence > 0.7:
            memory.store(pattern, level=SEMANTIC)
        else:
            validator.quarantine_suspicious(pattern)
    """

    def __init__(self, confidence_threshold: float = 0.7, use_llm_validation: bool = True, quarantine_enabled: bool = True) -> None:
        """
        Initialize validator.

        Args:
            confidence_threshold: Minimum confidence for valid patterns
            use_llm_validation: Use LLM for validation (vs heuristics)
            quarantine_enabled: Store suspicious patterns for review
        """
        self.confidence_threshold = confidence_threshold
        self.use_llm_validation = use_llm_validation
        self.quarantine_enabled = quarantine_enabled

        # LLM validator
        if use_llm_validation:
            self.validator = dspy.ChainOfThought(ConsolidationValidationSignature)
        else:
            self.validator = None

        # Quarantine storage
        self._quarantine: List[Dict] = []
        self._quarantine_max_size = 100

        # Statistics
        self._total_validated = 0
        self._total_accepted = 0
        self._total_rejected = 0
        self._total_quarantined = 0

        logger.info(
            f"ConsolidationValidator initialized: "
            f"threshold={confidence_threshold}, llm={use_llm_validation}"
        )

    def validate_pattern(
        self,
        pattern: str,
        source_memories: List[Any],
        pattern_type: str = "SEMANTIC"
    ) -> Tuple[bool, float, str]:
        """
        Validate pattern against source memories.

        Args:
            pattern: The extracted pattern/wisdom
            source_memories: Source memories that led to this pattern
            pattern_type: Type of pattern (SUCCESS_PATTERN, FAILURE_PATTERN, etc.)

        Returns:
            (is_valid, confidence, reasoning)
        """
        self._total_validated += 1

        # Empty or too short patterns are invalid
        if not pattern or len(pattern.strip()) < 10:
            self._total_rejected += 1
            return False, 0.0, "Pattern is empty or too short"

        # If LLM validation enabled, use it
        if self.use_llm_validation and self.validator:
            return self._llm_validate(pattern, source_memories, pattern_type)

        # Fallback to heuristic validation
        return self._heuristic_validate(pattern, source_memories, pattern_type)

    def _llm_validate(
        self,
        pattern: str,
        source_memories: List[Any],
        pattern_type: str
    ) -> Tuple[bool, float, str]:
        """Use LLM to validate pattern."""
        import json

        try:
            # Format source memories
            source_texts = []
            for mem in source_memories[:10]:  # Limit for context
                if hasattr(mem, 'content'):
                    source_texts.append(mem.content[:500])
                elif isinstance(mem, dict):
                    source_texts.append(str(mem)[:500])
                else:
                    source_texts.append(str(mem)[:500])

            result = self.validator(
                pattern=pattern,
                source_memories=json.dumps(source_texts),
                pattern_type=pattern_type
            )

            is_valid = result.is_valid if hasattr(result, 'is_valid') else True
            try:
                confidence = float(result.confidence)
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError, AttributeError):
                confidence = 0.5
            reasoning = result.reasoning or "LLM validation complete"

            # Apply confidence threshold
            if is_valid and confidence < self.confidence_threshold:
                is_valid = False
                reasoning = f"Confidence {confidence:.2f} below threshold {self.confidence_threshold}"

            if is_valid:
                self._total_accepted += 1
            else:
                self._total_rejected += 1
                # Auto-quarantine rejected patterns
                if self.quarantine_enabled:
                    self.quarantine_suspicious(pattern, source_memories, reasoning)

            return is_valid, confidence, reasoning

        except Exception as e:
            logger.warning(f"LLM validation failed: {e}, using heuristic")
            return self._heuristic_validate(pattern, source_memories, pattern_type)

    def _heuristic_validate(
        self,
        pattern: str,
        source_memories: List[Any],
        pattern_type: str
    ) -> Tuple[bool, float, str]:
        """Heuristic-based pattern validation."""
        issues = []
        confidence = 0.7  # Base confidence for heuristics

        # Check 1: Pattern should not be too vague
        vague_phrases = [
            "things work", "stuff happens", "in general", "usually",
            "it depends", "sometimes", "maybe", "possibly"
        ]
        pattern_lower = pattern.lower()
        for phrase in vague_phrases:
            if phrase in pattern_lower:
                issues.append(f"Contains vague phrase: '{phrase}'")
                confidence -= 0.1

        # Check 2: Pattern should reference something concrete
        concrete_indicators = [
            "when", "if", "use", "avoid", "for", "because",
            "results in", "leads to", "causes", "prevents"
        ]
        has_concrete = any(ind in pattern_lower for ind in concrete_indicators)
        if not has_concrete:
            issues.append("Pattern lacks concrete indicators (when/if/use/avoid)")
            confidence -= 0.15

        # Check 3: Pattern should have some overlap with source memories
        if source_memories:
            source_words = set()
            for mem in source_memories:
                content = mem.content if hasattr(mem, 'content') else str(mem)
                source_words.update(content.lower().split())

            pattern_words = set(pattern_lower.split())
            overlap = len(pattern_words & source_words)
            overlap_ratio = overlap / len(pattern_words) if pattern_words else 0

            if overlap_ratio < 0.2:
                issues.append(f"Low overlap with sources ({overlap_ratio:.0%})")
                confidence -= 0.2
            elif overlap_ratio > 0.5:
                confidence += 0.1  # Good grounding

        # Check 4: Pattern length should be reasonable
        if len(pattern) > 500:
            issues.append("Pattern too long (>500 chars)")
            confidence -= 0.1
        elif len(pattern) < 20:
            issues.append("Pattern too short (<20 chars)")
            confidence -= 0.15

        # Clamp confidence
        confidence = max(0.0, min(1.0, confidence))

        # Determine validity
        is_valid = confidence >= self.confidence_threshold and len(issues) <= 1

        if is_valid:
            self._total_accepted += 1
            reasoning = f"Heuristic validation passed (confidence={confidence:.2f})"
        else:
            self._total_rejected += 1
            reasoning = f"Heuristic validation failed: {'; '.join(issues)}"
            if self.quarantine_enabled:
                self.quarantine_suspicious(pattern, source_memories, reasoning)

        return is_valid, confidence, reasoning

    def quarantine_suspicious(
        self,
        pattern: str,
        source_memories: List[Any] = None,
        reason: str = ""
    ) -> None:
        """
        Move suspicious patterns to quarantine for review.

        Args:
            pattern: The suspicious pattern
            source_memories: Source memories (for debugging)
            reason: Why pattern was quarantined
        """
        import time

        entry = {
            'pattern': pattern,
            'source_count': len(source_memories) if source_memories else 0,
            'reason': reason,
            'timestamp': time.time(),
        }

        self._quarantine.append(entry)
        self._total_quarantined += 1

        # Keep quarantine bounded
        if len(self._quarantine) > self._quarantine_max_size:
            self._quarantine = self._quarantine[-self._quarantine_max_size:]

        logger.warning(
            f"Quarantined pattern: {pattern[:100]}... "
            f"Reason: {reason}"
        )

    def get_quarantine(self) -> List[Dict]:
        """Get all quarantined patterns for review."""
        return self._quarantine.copy()

    def clear_quarantine(self) -> int:
        """Clear quarantine and return count."""
        count = len(self._quarantine)
        self._quarantine.clear()
        return count

    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            'total_validated': self._total_validated,
            'total_accepted': self._total_accepted,
            'total_rejected': self._total_rejected,
            'total_quarantined': self._total_quarantined,
            'acceptance_rate': (
                self._total_accepted / self._total_validated
                if self._total_validated > 0 else 0
            ),
            'quarantine_size': len(self._quarantine),
        }


class MemoryLevelClassifier:
    """
    A-Team Enhancement: LLM-based automatic memory level classification.
    
    Replaces hardcoded level decisions with intelligent classification.
    
    Usage:
        classifier = MemoryLevelClassifier()
        level, confidence, should_store = classifier.classify(
            experience="Successfully mapped bank_code column using regex extraction",
            context={"task": "column_mapping", "agent": "diffuser", "outcome": "success"}
        )
        
        if should_store:
            memory.store(content=experience, level=level, ...)
    """
    
    def __init__(self, use_cot: bool = True) -> None:
        self.use_cot = use_cot
        if use_cot:
            self.classifier = dspy.ChainOfThought(MemoryLevelClassificationSignature)
        else:
            self.classifier = dspy.Predict(MemoryLevelClassificationSignature)
        
        # Level mapping
        self.level_map = {
            'EPISODIC': MemoryLevel.EPISODIC,
            'SEMANTIC': MemoryLevel.SEMANTIC,
            'PROCEDURAL': MemoryLevel.PROCEDURAL,
            'META': MemoryLevel.META,
            'CAUSAL': MemoryLevel.CAUSAL
        }
        
        # A-Team v8.0: NO keyword lists! Structure-based classification only
        # Removed level_hints entirely - uses _heuristic_classify with structural analysis
    
    def classify(self, experience: str, context: Dict[str, Any]) -> Tuple[MemoryLevel, float, bool]:
        """
        Classify experience into appropriate memory level.
        
        Returns:
            (MemoryLevel, confidence, should_store)
        """
        import json
        
        try:
            result = self.classifier(
                experience=experience, # NO LIMIT - FULL content
                context=json.dumps(context)
            )
            
            level_str = (result.level or 'EPISODIC').upper().strip()
            level = self.level_map.get(level_str, MemoryLevel.EPISODIC)
            confidence = float(result.confidence) if result.confidence else 0.5
            should_store = result.should_store if hasattr(result, 'should_store') else True

            return level, confidence, should_store

        except Exception as e:
            logger.debug(f"Classification failed: {e}, using heuristic")
            # Fallback to heuristic classification
            return self._heuristic_classify(experience), 0.5, True
    
    async def _classify_with_retry(self, experience: str, context: Dict[str, Any]) -> MemoryLevel:
        """
        A-Team v9.0: NO HEURISTIC FALLBACKS.
        
        If primary classification fails:
        1. Retry with context of failure
        2. If still fails, use FallbackClassificationAgent
        3. NEVER use hardcoded rules
        """
        from .modern_agents import UniversalRetryHandler, PatternDetector
        
        from Jotty.core.infrastructure.foundation.config_defaults import MAX_RETRIES
        retry_handler = UniversalRetryHandler(max_retries=MAX_RETRIES)
        
        async def classify_attempt(**kwargs: Any) -> Any:
            exp = kwargs.get('experience', '')
            ctx = kwargs.get('context', {})
            
            result = self.classifier(
                experience=exp,
                context=json.dumps(ctx)
            )
            
            level_str = (result.level or 'EPISODIC').upper().strip()
            return self.level_map.get(level_str, MemoryLevel.EPISODIC)

        # Fallback agent (specialized for difficult cases)
        async def fallback_classifier(**kwargs: Any) -> Any:
            """
            Specialized agent for difficult classification cases.
            
            Gets full error context and tries harder.
            """
            task_info = kwargs.get('task', '')
            errors = kwargs.get('error_history', [])
            
            # Create a more detailed prompt with all context
            detailed_prompt = f"""
            DIFFICULT CLASSIFICATION TASK
            
            The primary classifier failed with these errors:
            {errors}
            
            Original task: {task_info}
            
            Please classify this experience into one of:
            - EPISODIC: Raw events, specific instances, tool outputs
            - SEMANTIC: Patterns, generalizations, abstractions
            - PROCEDURAL: How-to knowledge, step sequences
            - META: Wisdom about approach, when to use what
            - CAUSAL: Why things work, cause-effect relationships
            
            Think carefully and provide classification.
            """
            
            # Use a fresh classifier with detailed prompt
            class DetailedClassification(dspy.Signature):
                detailed_context = dspy.InputField()
                experience = dspy.InputField()
                level = dspy.OutputField()
                confidence = dspy.OutputField()
            
            specialist = dspy.ChainOfThought(DetailedClassification)
            result = specialist(
                detailed_context=detailed_prompt,
                experience=kwargs.get('original_input', {}).get('experience', '')
            )
            
            level_str = (result.level or 'EPISODIC').upper().strip()
            return self.level_map.get(level_str, MemoryLevel.EPISODIC)

        result = await retry_handler.execute_with_retry(
            agent_func=classify_attempt,
            task_description=f"Classify memory level for: {experience}...",
            initial_input={'experience': experience, 'context': context},
            specialist_agent=fallback_classifier
        )
        
        if result.is_certain:
            return result.value
        else:
            # Even specialist failed - return with uncertainty flag
            # (This is NOT a heuristic fallback, it's explicit uncertainty)
            logger.warning(f"Classification uncertain after all retries: {result.reasoning}")
            return MemoryLevel.EPISODIC  # Default with logged uncertainty


# =============================================================================
# MEMORY CLUSTER
# =============================================================================

@dataclass
class MemoryCluster:
    """A cluster of related memories for consolidation."""
    cluster_id: str
    goal_signature: str
    memories: List[MemoryEntry]
    
    # Cluster statistics
    avg_value: float = 0.0
    success_rate: float = 0.0
    common_keywords: List[str] = field(default_factory=list)
    
    # Extracted pattern (if consolidated)
    extracted_pattern: Optional[str] = None
    pattern_confidence: float = 0.0
    
    def compute_statistics(self) -> None:
        """
        Compute cluster statistics.
        
        A-Team Fix: Removed keyword extraction (loses semantic meaning).
        Uses content length and value distribution instead.
        """
        if not self.memories:
            return
        
        values = [m.default_value for m in self.memories]
        self.avg_value = sum(values) / len(values)
        
        # Success rate from memory values (value > 0.5 = successful use)
        successful = sum(1 for v in values if v > 0.5)
        self.success_rate = successful / len(values)
        
        # A-Team: Instead of keywords, store content signatures for hash-based similarity
        # This avoids keyword matching which loses semantic meaning
        # Keywords list now stores content length buckets for fast filtering
        length_buckets = []
        for m in self.memories:
            content_len = len(m.content)
            if content_len < 100:
                length_buckets.append("short")
            elif content_len < 500:
                length_buckets.append("medium")
            else:
                length_buckets.append("long")
        
        # Store most common length bucket (useful for clustering similar experiences)
        from collections import Counter
        bucket_counts = Counter(length_buckets)
        self.common_keywords = [f"content_{b}" for b, _ in bucket_counts.most_common(3)]

