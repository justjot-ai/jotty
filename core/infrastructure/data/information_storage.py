"""
Jotty v7.6 - Information-Theoretic Storage
============================================

A-Team Approved: Shannon-inspired memory storage.

NO HARDCODED "if failure store more".

Uses proper information theory:
- I(x) = -log₂ P(x) = information content
- Estimate P(x) using frequency + LLM surprise
- Store detail proportional to information content

High surprise events are RARE and VALUABLE for learning.
Low surprise events are COMMON and should be stored compactly.
"""

import asyncio
import hashlib
import json
import math
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class InformationWeightedMemory:
    """A memory with information-theoretic weighting."""
    key: str
    content: str
    information_content: float  # -log₂ P(event)
    frequency_estimate: float  # P(event) from frequency
    llm_surprise: float  # P(event) from LLM
    detail_level: str  # "maximum", "high", "normal", "minimal"
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self) -> None:
        # Ensure information content is valid
        if self.information_content < 0:
            self.information_content = 0


# =============================================================================
# SURPRISE ESTIMATOR
# =============================================================================

class SurpriseSignature(dspy.Signature):
    """
    Estimate how surprising/unexpected an event is.
    
    Think about:
    - Have you seen similar events before?
    - How typical is this outcome?
    - Would an expert be surprised by this?
    
    More surprising = higher score (0.0-1.0)
    """
    
    event_description = dspy.InputField(desc="Description of the event")
    context = dspy.InputField(desc="Context in which event occurred")
    historical_patterns = dspy.InputField(desc="Summary of similar past events")
    
    surprise_score = dspy.OutputField(
        desc="How surprising is this? 0.0 = expected, 1.0 = very surprising"
    )
    reasoning = dspy.OutputField(desc="Why is this surprising/expected?")


class SurpriseEstimator:
    """
    Estimate event surprise using LLM.
    
    Combines:
    1. LLM judgment of how surprising the event is
    2. Comparison to historical patterns
    """
    
    def __init__(self) -> None:
        if DSPY_AVAILABLE:
            self.estimator = dspy.ChainOfThought(SurpriseSignature)
        else:
            self.estimator = None
    
    async def estimate_surprise(
        self,
        event: Dict[str, Any],
        context: Dict[str, Any],
        historical_patterns: List[str]
    ) -> Tuple[float, str]:
        """
        Estimate how surprising an event is.
        
        Returns: (surprise_score: 0-1, reasoning: str)
        Higher score = more surprising = more information
        """
        if not self.estimator:
            # Fallback: use outcome as proxy
            outcome = str(event.get('outcome', event.get('success', 'unknown'))).lower()
            if 'fail' in outcome or 'error' in outcome:
                return (0.7, "Failures are typically less common")
            return (0.3, "Success is typically more common")
        
        try:
            result = self.estimator(
                event_description=json.dumps(event, default=str),
                context=json.dumps(context, default=str),
                historical_patterns=json.dumps(historical_patterns)
            )
            
            score = self._parse_score(result.surprise_score)
            return (score, result.reasoning or "")
            
        except Exception as e:
            logger.debug(f"Surprise estimation failed: {e}")
            return (0.5, "Could not estimate surprise")
    
    def _parse_score(self, score_str: str) -> float:
        """Parse score from LLM output."""
        if isinstance(score_str, (int, float)):
            return float(score_str)

        try:
            return float(score_str)
        except (ValueError, TypeError) as e:
            logger.debug(f"Score parsing failed: {e}")
            import re
            numbers = re.findall(r'[\d.]+', str(score_str))
            if numbers:
                return max(0.0, min(1.0, float(numbers[0])))
            return 0.5


# =============================================================================
# INFORMATION-THEORETIC STORAGE
# =============================================================================

class InformationTheoreticStorage:
    """
    Store memories with detail proportional to information content.
    
    Shannon's Information Theory:
    I(x) = -log₂ P(x)
    
    High information = rare event = store MORE detail
    Low information = common event = store LESS detail
    
    We estimate P(x) using:
    1. Frequency tracking: P_freq = count(x) / total
    2. LLM surprise: P_llm = 1 - surprise_score
    3. Combined: P = α * P_freq + (1-α) * P_llm
    """
    
    def __init__(self, alpha: float = 0.5, min_info_threshold: float = 0.5, max_info_threshold: float = 3.0) -> None:
        self.alpha = alpha
        self.min_info_threshold = min_info_threshold
        self.max_info_threshold = max_info_threshold
        
        # Frequency tracking
        self.event_counts: Dict[str, int] = defaultdict(int)
        self.total_events = 0
        
        # Historical patterns for LLM context
        self.recent_patterns: List[str] = []
        self.max_patterns = 50
        
        # Surprise estimator
        self.surprise_estimator = SurpriseEstimator()
        
        # Stored memories
        self.memories: Dict[str, InformationWeightedMemory] = {}
        
        logger.info(" InformationTheoreticStorage initialized")
    
    async def compute_information_content(
        self,
        event: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[float, str, float, float]:
        """
        Compute information content of an event.
        
        Returns: (info_content, detail_level, freq_estimate, llm_surprise)
        """
        # 1. Compute event signature (hash of key features)
        event_sig = self._compute_event_signature(event)
        
        # 2. Update frequency count
        self.event_counts[event_sig] += 1
        self.total_events += 1
        
        # 3. Frequency-based probability estimate
        p_freq = self.event_counts[event_sig] / self.total_events
        
        # 4. LLM surprise-based probability estimate
        surprise, _ = await self.surprise_estimator.estimate_surprise(
            event, context, self.recent_patterns
        )
        p_llm = 1.0 - surprise  # Surprise of 1.0 means very rare (low P)
        
        # 5. Combined probability
        p_combined = self.alpha * p_freq + (1 - self.alpha) * p_llm
        p_combined = max(0.001, min(0.999, p_combined))  # Avoid log(0)
        
        # 6. Information content: I = -log₂(P)
        info_content = -math.log2(p_combined)
        
        # 7. Determine detail level
        if info_content >= self.max_info_threshold:
            detail_level = "maximum"
        elif info_content >= 2.0:
            detail_level = "high"
        elif info_content >= 1.0:
            detail_level = "normal"
        else:
            detail_level = "minimal"
        
        # 8. Update recent patterns
        pattern = self._summarize_event(event)
        self.recent_patterns.append(pattern)
        if len(self.recent_patterns) > self.max_patterns:
            self.recent_patterns = self.recent_patterns[-self.max_patterns:]
        
        return (info_content, detail_level, p_freq, surprise)
    
    async def store(
        self,
        event: Dict[str, Any],
        context: Dict[str, Any],
        raw_content: str
    ) -> InformationWeightedMemory:
        """
        Store event with information-weighted detail level.
        
        High information events get full trace.
        Low information events get brief summary.
        """
        # Compute information content
        info_content, detail_level, freq_est, surprise = \
            await self.compute_information_content(event, context)
        
        # Generate content based on detail level
        if detail_level == "maximum":
            content = self._maximum_detail(event, context, raw_content)
        elif detail_level == "high":
            content = self._high_detail(event, context, raw_content)
        elif detail_level == "normal":
            content = self._normal_detail(event, raw_content)
        else:  # minimal
            content = self._minimal_detail(event)
        
        # Create memory
        key = self._generate_key(event)
        memory = InformationWeightedMemory(
            key=key,
            content=content,
            information_content=info_content,
            frequency_estimate=freq_est,
            llm_surprise=surprise,
            detail_level=detail_level
        )
        
        self.memories[key] = memory
        
        logger.debug(
            f" Stored memory: info={info_content:.2f}, "
            f"detail={detail_level}, freq={freq_est:.3f}"
        )
        
        return memory
    
    def _compute_event_signature(self, event: Dict[str, Any]) -> str:
        """Compute a signature for event categorization."""
        # Extract key features
        features = {
            'type': event.get('type', event.get('action', 'unknown')),
            'outcome': str(event.get('outcome', event.get('success', 'unknown'))),
            'agent': event.get('agent', 'unknown')
        }
        
        sig_str = json.dumps(features, sort_keys=True)
        return hashlib.md5(sig_str.encode()).hexdigest()
    
    def _summarize_event(self, event: Dict[str, Any]) -> str:
        """Create a brief summary of an event for pattern tracking."""
        return f"{event.get('agent', '?')}: {event.get('action', '?')} -> {event.get('outcome', '?')}"
    
    def _generate_key(self, event: Dict[str, Any]) -> str:
        """Generate unique key for memory."""
        timestamp = datetime.now().isoformat()
        content = json.dumps(event, default=str, sort_keys=True)
        return hashlib.md5(f"{content}{timestamp}".encode()).hexdigest()
    
    def _maximum_detail(
        self,
        event: Dict[str, Any],
        context: Dict[str, Any],
        raw_content: str
    ) -> str:
        """Maximum detail for high-information events."""
        return f""" HIGH INFORMATION EVENT (Rare/Surprising)
══════════════════════════════════════════════════════════════

EVENT DETAILS:
{json.dumps(event, indent=2, default=str)}

FULL CONTEXT:
{json.dumps(context, indent=2, default=str)}

RAW CONTENT:
{raw_content}

LEARNING NOTE:
This is a rare event with high information content.
Store full details to prevent similar issues.
══════════════════════════════════════════════════════════════"""
    
    def _high_detail(
        self,
        event: Dict[str, Any],
        context: Dict[str, Any],
        raw_content: str
    ) -> str:
        """High detail for somewhat rare events."""
        return f""" NOTABLE EVENT
Event: {json.dumps(event, default=str)}
Context Summary: {json.dumps(context, default=str)}
Key Content: {raw_content}"""
    
    def _normal_detail(
        self,
        event: Dict[str, Any],
        raw_content: str
    ) -> str:
        """Normal detail for typical events."""
        return f"Event: {event.get('type', event.get('action', 'unknown'))} | " \
               f"Outcome: {event.get('outcome', 'unknown')} | " \
               f"Summary: {raw_content}"
    
    def _minimal_detail(self, event: Dict[str, Any]) -> str:
        """Minimal detail for very common events."""
        return f"{event.get('agent', '?')}: {event.get('outcome', 'ok')}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        detail_counts = defaultdict(int)
        for mem in self.memories.values():
            detail_counts[mem.detail_level] += 1
        
        return {
            'total_events': self.total_events,
            'unique_signatures': len(self.event_counts),
            'memories_stored': len(self.memories),
            'detail_distribution': dict(detail_counts),
            'average_info_content': sum(
                m.information_content for m in self.memories.values()
            ) / max(len(self.memories), 1)
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'InformationWeightedMemory',
    'SurpriseEstimator',
    'InformationTheoreticStorage'
]

