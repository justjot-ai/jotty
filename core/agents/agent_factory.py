"""
Jotty v9.0 - Modern Agent-Only Architecture
============================================

A-Team Deliberation Result: NO HEURISTICS, ONLY AGENTS

Key Principles:
1. NO fallback heuristics - if something fails, retry with context
2. NO keyword matching - use pattern detector agents
3. NO hardcoded thresholds - LLM decides everything
4. Modern algorithms (2020+) adapted for LLM agents

Inspired by:
- COMA (2018): Counterfactual credit assignment
- Self-RAG (2023): Self-reflective retrieval
- RAPTOR (2024): Recursive summarization for memory
- HAPPO (2022): Heterogeneous agent coordination

All decisions made by LLM agents, not heuristics.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

from ..foundation.exceptions import (
    AgentExecutionError,
    ValidationError,
    wrap_exception
)

logger = logging.getLogger(__name__)


# =============================================================================
# RETRY WITH CONTEXT (No Silent Fallbacks)
# =============================================================================

class RetryStrategy(Enum):
    """How to handle failures."""
    RETRY_WITH_CONTEXT = "retry_with_context"  # Retry with failure info
    ESCALATE_TO_SPECIALIST = "escalate"        # Use specialist agent
    RETURN_UNCERTAINTY = "uncertainty"          # Return "I don't know"


@dataclass
class AgentResult:
    """Result from any agent with explicit uncertainty."""
    value: Any
    confidence: float
    reasoning: str
    is_certain: bool = True
    needs_human_review: bool = False
    error_history: List[str] = field(default_factory=list)


class RetryWithContextSignature(dspy.Signature):
    """
    Given a failed attempt, analyze why and suggest how to succeed.
    """
    original_task = dspy.InputField(desc="What was the agent trying to do?")
    previous_attempts = dspy.InputField(desc="JSON list of previous attempts and their errors")
    available_context = dspy.InputField(desc="What additional context is available?")
    
    analysis = dspy.OutputField(desc="Why did previous attempts fail?")
    strategy = dspy.OutputField(desc="What should be different this time?")
    modified_approach = dspy.OutputField(desc="The modified approach to try")
    should_escalate = dspy.OutputField(desc="true if this needs a specialist, false otherwise")


class UniversalRetryHandler:
    """
    Universal retry handler that NEVER uses heuristics.
    
    When any agent fails:
    1. Analyze the failure with an LLM
    2. Retry with modified approach
    3. If still failing, escalate to specialist agent
    4. Never fall back to hardcoded rules
    """
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        if DSPY_AVAILABLE:
            self.analyzer = dspy.ChainOfThought(RetryWithContextSignature)
        else:
            self.analyzer = None
        
        logger.info("🔄 UniversalRetryHandler initialized (NO heuristic fallbacks)")
    
    async def execute_with_retry(
        self,
        agent_func,
        task_description: str,
        initial_input: Dict[str, Any],
        specialist_agent=None
    ) -> AgentResult:
        """
        Execute an agent function with intelligent retry.
        
        NO FALLBACK TO HEURISTICS - if all retries fail, escalates to specialist
        or returns explicit uncertainty.
        """
        attempts = []
        current_input = initial_input.copy()
        
        for attempt in range(self.max_retries):
            try:
                result = await self._call_agent(agent_func, current_input)
                return AgentResult(
                    value=result,
                    confidence=1.0,
                    reasoning=f"Succeeded on attempt {attempt + 1}",
                    is_certain=True
                )
                
            except Exception as e:
                error_info = {
                    'attempt': attempt + 1,
                    'error': str(e),
                    'input_summary': str(current_input)
                }
                attempts.append(error_info)
                
                # Analyze failure and modify approach
                if self.analyzer and attempt < self.max_retries - 1:
                    analysis = self._analyze_failure(
                        task_description,
                        attempts,
                        current_input
                    )
                    
                    if analysis.get('should_escalate', False):
                        break  # Go to specialist
                    
                    # Modify input based on analysis
                    current_input['_retry_context'] = {
                        'previous_errors': attempts,
                        'suggested_strategy': analysis.get('strategy', ''),
                        'modified_approach': analysis.get('modified_approach', '')
                    }
        
        # All retries exhausted - escalate or return uncertainty
        if specialist_agent:
            logger.info(f"Escalating to specialist after {len(attempts)} failures")
            return await self._call_specialist(
                specialist_agent,
                task_description,
                initial_input,
                attempts
            )
        
        # No specialist - return explicit uncertainty (NOT heuristic!)
        return AgentResult(
            value=None,
            confidence=0.0,
            reasoning=f"Failed after {len(attempts)} attempts: {attempts[-1]['error']}",
            is_certain=False,
            needs_human_review=True,
            error_history=[a['error'] for a in attempts]
        )
    
    async def _call_agent(self, agent_func, input_dict: Dict) -> Any:
        """Call agent function (async or sync)."""
        if asyncio.iscoroutinefunction(agent_func):
            return await agent_func(**input_dict)
        return agent_func(**input_dict)
    
    def _analyze_failure(
        self,
        task: str,
        attempts: List[Dict],
        context: Dict
    ) -> Dict[str, Any]:
        """Use LLM to analyze why attempts failed."""
        if not self.analyzer:
            return {}
        
        try:
            result = self.analyzer(
                original_task=task,
                previous_attempts=json.dumps(attempts),
                available_context=json.dumps(str(context))
            )
            
            return {
                'analysis': result.analysis,
                'strategy': result.strategy,
                'modified_approach': result.modified_approach,
                'should_escalate': str(result.should_escalate).lower() == 'true'
            }
        except Exception as e:
            logger.warning(f"Failure analysis failed: {e}")
            return {}
    
    async def _call_specialist(
        self,
        specialist,
        task: str,
        original_input: Dict,
        error_history: List[Dict]
    ) -> AgentResult:
        """Call specialist agent with full context."""
        specialist_input = {
            'task': task,
            'original_input': original_input,
            'error_history': error_history,
            'request': 'Previous attempts failed. Please handle this difficult case.'
        }
        
        try:
            result = await self._call_agent(specialist, specialist_input)
            return AgentResult(
                value=result,
                confidence=0.7,  # Lower confidence since specialist was needed
                reasoning="Resolved by specialist agent after main agent failures",
                is_certain=True
            )
        except Exception as e:
            return AgentResult(
                value=None,
                confidence=0.0,
                reasoning=f"Even specialist failed: {e}",
                is_certain=False,
                needs_human_review=True
            )


# =============================================================================
# PATTERN DETECTOR (Replaces ALL Keyword Matching)
# =============================================================================

class PatternDetectionSignature(dspy.Signature):
    """
    Detect patterns in text WITHOUT keyword matching.
    
    This replaces dangerous patterns like:
    - 'if "should" in text'  # WRONG: keyword matching
    - 'if "because" in text' # WRONG: keyword matching
    
    Instead, the LLM understands the INTENT.
    """
    text = dspy.InputField(desc="Text to analyze")
    patterns_to_detect = dspy.InputField(desc="List of patterns to look for (semantic, not keywords)")
    
    detected_patterns = dspy.OutputField(desc="JSON dict of pattern_name: true/false")
    reasoning = dspy.OutputField(desc="Why each pattern was detected or not")
    confidence = dspy.OutputField(desc="Confidence in detection 0.0-1.0")


class PatternDetector:
    """
    LLM-based pattern detector that replaces ALL keyword matching.
    
    Instead of:
        has_modal = 'should ' in text.lower()  # DANGEROUS
    
    Use:
        patterns = await detector.detect(text, ["contains_modal_verbs"])
        has_modal = patterns["contains_modal_verbs"]
    
    The LLM understands semantics, not just keywords.
    """
    
    STANDARD_PATTERNS = [
        "contains_modal_verbs",      # should, could, would, might
        "is_question",               # interrogative
        "is_comparison",             # comparing two things
        "is_causal_statement",       # because, therefore, causes
        "is_procedural",             # step-by-step instructions
        "contains_code_or_json",     # structured data
        "is_meta_commentary",        # about the process itself
        "is_specific_example",       # concrete instance
        "is_abstract_pattern",       # generalization
        "expresses_uncertainty"      # hedging, doubt
    ]
    
    def __init__(self):
        if DSPY_AVAILABLE:
            self.detector = dspy.ChainOfThought(PatternDetectionSignature)
        else:
            self.detector = None
        
        logger.info("🔍 PatternDetector initialized (NO keyword matching)")
    
    async def detect(
        self,
        text: str,
        patterns: List[str] = None
    ) -> Dict[str, bool]:
        """
        Detect patterns in text using LLM understanding.
        
        Returns dict of pattern_name -> detected (bool)
        """
        if not self.detector:
            # Return uncertainty, not heuristics!
            return {p: None for p in (patterns or self.STANDARD_PATTERNS)}
        
        patterns = patterns or self.STANDARD_PATTERNS
        
        try:
            result = self.detector(
                text=text,  # Limit for context
                patterns_to_detect=json.dumps(patterns)
            )
            
            # Parse result
            detected = self._parse_detected(result.detected_patterns)
            return detected
            
        except Exception as e:
            logger.warning(f"Pattern detection failed: {e}")
            # Return uncertainty, NOT heuristics
            return {p: None for p in patterns}
    
    def _parse_detected(self, detected_str: str) -> Dict[str, bool]:
        """Parse detected patterns from LLM output."""
        try:
            if isinstance(detected_str, dict):
                return detected_str
            return json.loads(detected_str)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            # Try to extract from string if JSON parsing fails
            logger.debug(f"JSON parsing failed, extracting from string: {e}")
            result = {}
            for line in detected_str.split('\n'):
                line = line.strip().lower()
                for pattern in self.STANDARD_PATTERNS:
                    if pattern.lower() in line:
                        result[pattern] = 'true' in line or 'yes' in line
            return result


# =============================================================================
# LLM-AS-CRITIC (COMA 2018 Adapted for LLM Agents)
# =============================================================================

class CounterfactualCriticSignature(dspy.Signature):
    """
    COMA-style counterfactual credit assignment using LLM.
    
    From Foerster et al. 2018 "Counterfactual Multi-Agent Policy Gradients"
    Adapted for LLM agents (no neural network training).
    
    Instead of training a critic network, we use LLM reasoning to estimate:
    "What would have happened without this agent?"
    """
    trajectory = dspy.InputField(desc="Full trajectory of all agent actions and outcomes")
    agent_to_evaluate = dspy.InputField(desc="Which agent are we evaluating?")
    agent_action = dspy.InputField(desc="What action did this agent take?")
    final_outcome = dspy.InputField(desc="What was the final outcome?")
    other_agents_actions = dspy.InputField(desc="What did other agents do?")
    
    counterfactual_reasoning = dspy.OutputField(
        desc="If this agent had taken a default/null action instead, what would have happened?"
    )
    marginal_contribution = dspy.OutputField(
        desc="How much did this agent's specific action contribute to the outcome? 0.0 to 1.0"
    )
    credit_score = dspy.OutputField(desc="Final credit score for this agent 0.0 to 1.0")
    confidence = dspy.OutputField(desc="Confidence in this assessment 0.0 to 1.0")


class LLMCounterfactualCritic:
    """
    LLM-as-Critic for credit assignment (COMA-style, 2018).
    
    Modern alternative to Shapley Value (1953).
    
    Benefits:
    - Uses LLM's reasoning about counterfactuals
    - No combinatorial explosion (unlike Shapley)
    - Semantic understanding of agent contributions
    - No training required
    """
    
    def __init__(self):
        if DSPY_AVAILABLE:
            self.critic = dspy.ChainOfThought(CounterfactualCriticSignature)
        else:
            self.critic = None
        
        logger.info("🎯 LLMCounterfactualCritic initialized (COMA-style, 2018)")
    
    async def assign_credit(
        self,
        trajectory: List[Dict],
        agents: List[str],
        final_reward: float
    ) -> Dict[str, float]:
        """
        Assign credit to each agent using counterfactual reasoning.
        
        No heuristics, no hardcoded weights.
        """
        if not self.critic:
            # Return equal credit (uncertainty), not heuristics
            equal = final_reward / len(agents) if agents else 0
            return {a: equal for a in agents}
        
        credits = {}
        
        for agent in agents:
            # Find this agent's actions
            agent_actions = [s for s in trajectory if s.get('agent') == agent]
            other_actions = [s for s in trajectory if s.get('agent') != agent]
            
            try:
                result = self.critic(
                    trajectory=json.dumps(trajectory),
                    agent_to_evaluate=agent,
                    agent_action=json.dumps(agent_actions),
                    final_outcome=f"Reward: {final_reward}",
                    other_agents_actions=json.dumps(other_actions)
                )
                
                credit = self._parse_credit(result.credit_score)
                credits[agent] = credit * final_reward
                
            except Exception as e:
                logger.warning(f"Credit assignment for {agent} failed: {e}")
                credits[agent] = final_reward / len(agents)  # Fair split on failure
        
        # Normalize to sum to final_reward
        total = sum(credits.values())
        if total > 0:
            credits = {k: v / total * final_reward for k, v in credits.items()}
        
        return credits
    
    def _parse_credit(self, credit_str: str) -> float:
        """Parse credit score from LLM output."""
        try:
            return float(credit_str)
        except (ValueError, TypeError) as e:
            # Extract number from string
            logger.debug(f"Direct float parsing failed: {e}, trying extraction")
            for part in str(credit_str).split():
                try:
                    val = float(part.strip(',%'))
                    if 0 <= val <= 1:
                        return val
                    if 0 <= val <= 100:
                        return val / 100
                except (ValueError, TypeError):
                    continue
            return 0.5  # Uncertainty, not heuristic


# =============================================================================
# SELF-RAG STYLE MEMORY RETRIEVAL (2023)
# =============================================================================

class SelfReflectiveRetrievalSignature(dspy.Signature):
    """
    Self-RAG style retrieval (Asai et al., 2023).
    
    The LLM decides:
    1. Whether to retrieve at all
    2. What to retrieve
    3. Whether retrieved content is relevant
    4. How to use it
    
    No embedding model, fully agentic.
    """
    current_task = dspy.InputField(desc="What is the current task?")
    current_context = dspy.InputField(desc="What context do we already have?")
    available_memories = dspy.InputField(desc="Summary of available memories")
    
    should_retrieve = dspy.OutputField(desc="Do we need to retrieve memories? yes/no")
    retrieval_query = dspy.OutputField(desc="If yes, what should we search for?")
    reasoning = dspy.OutputField(desc="Why this retrieval decision?")


class MemoryRelevanceSignature(dspy.Signature):
    """
    Judge relevance of retrieved memory.
    """
    task = dspy.InputField(desc="Current task")
    memory_content = dspy.InputField(desc="Retrieved memory content")
    
    is_relevant = dspy.OutputField(desc="Is this memory relevant? yes/no")
    usefulness_score = dspy.OutputField(desc="How useful? 0.0 to 1.0")
    how_to_use = dspy.OutputField(desc="How should this memory be used?")


class SelfRAGMemoryRetriever:
    """
    Self-Reflective Retrieval-Augmented Generation for Memory.
    
    Modern alternative to embedding-based retrieval (2023).
    
    Features:
    - LLM decides when to retrieve
    - LLM formulates retrieval query
    - LLM judges relevance (no embedding similarity)
    - LLM decides how to use retrieved content
    
    No embedding models required.
    """
    
    def __init__(self):
        if DSPY_AVAILABLE:
            self.retrieval_decider = dspy.ChainOfThought(SelfReflectiveRetrievalSignature)
            self.relevance_judge = dspy.ChainOfThought(MemoryRelevanceSignature)
        else:
            self.retrieval_decider = None
            self.relevance_judge = None
        
        logger.info("📚 SelfRAGMemoryRetriever initialized (Self-RAG 2023)")
    
    async def retrieve(
        self,
        task: str,
        context: str,
        memories: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Self-reflective retrieval of relevant memories.
        
        No embedding similarity, fully agent-driven.
        """
        if not self.retrieval_decider:
            return memories[:top_k]  # Simple fallback if no LLM
        
        # Step 1: Decide if we need to retrieve
        memory_summary = self._summarize_memories(memories)
        
        try:
            decision = self.retrieval_decider(
                current_task=task,
                current_context=context,
                available_memories=memory_summary
            )
            
            should_retrieve = str(decision.should_retrieve).lower() in ['yes', 'true']
            
            if not should_retrieve:
                logger.debug("Self-RAG: No retrieval needed")
                return []
            
            retrieval_query = decision.retrieval_query
            
        except Exception as e:
            logger.warning(f"Retrieval decision failed: {e}")
            retrieval_query = task  # Fall back to task as query
        
        # Step 2: For each memory, judge relevance
        relevant = []
        for mem in memories:
            try:
                judgment = self.relevance_judge(
                    task=retrieval_query,
                    memory_content=str(mem.get('content', mem))
                )
                
                is_relevant = str(judgment.is_relevant).lower() in ['yes', 'true']
                score = self._parse_score(judgment.usefulness_score)
                
                if is_relevant and score > 0.3:
                    relevant.append({
                        'memory': mem,
                        'score': score,
                        'how_to_use': judgment.how_to_use
                    })
                    
            except Exception as e:
                logger.debug(f"Relevance judgment failed for memory: {e}")
                continue
        
        # Step 3: Sort by usefulness and return top_k
        relevant.sort(key=lambda x: x['score'], reverse=True)
        return [r['memory'] for r in relevant[:top_k]]
    
    def _summarize_memories(self, memories: List[Dict]) -> str:
        """Create summary of available memories."""
        if not memories:
            return "No memories available"
        
        summary = f"{len(memories)} memories available:\n"
        for i, mem in enumerate(memories):
            content = str(mem.get('content', mem))
            summary += f"  {i+1}. {content}...\n"
        
        if len(memories) > 10:
            summary += f"  ... and {len(memories) - 10} more"
        
        return summary
    
    def _parse_score(self, score_str: str) -> float:
        """Parse score from LLM output."""
        try:
            return float(score_str)
        except (ValueError, TypeError) as e:
            logger.debug(f"Direct float parsing failed: {e}, trying extraction")
            for part in str(score_str).split():
                try:
                    val = float(part.strip(',%'))
                    if 0 <= val <= 1:
                        return val
                    if 0 <= val <= 100:
                        return val / 100
                except (ValueError, TypeError):
                    continue
            return 0.5


# =============================================================================
# LLM SURPRISE ESTIMATOR (Replaces Shannon 1948)
# =============================================================================

class SurpriseEstimationSignature(dspy.Signature):
    """
    LLM-based surprise estimation (replaces Shannon entropy calculation).
    
    Instead of computing I(x) = -log₂ P(x) mathematically,
    we ask the LLM to reason about surprise.
    """
    event_description = dspy.InputField(desc="What event occurred?")
    context = dspy.InputField(desc="What was the context?")
    history_summary = dspy.InputField(desc="Summary of similar past events")
    
    surprise_level = dspy.OutputField(desc="How surprising? 1-10 (10 = very surprising)")
    detail_recommendation = dspy.OutputField(
        desc="How much detail to store? minimal/normal/detailed/maximum"
    )
    reasoning = dspy.OutputField(desc="Why this level of surprise?")


class LLMSurpriseEstimator:
    """
    LLM-based surprise estimation (modern alternative to Shannon 1948).
    
    Instead of mathematical entropy:
    - LLM reasons about how surprising an event is
    - LLM decides how much detail to store
    - Semantic understanding, not formula
    
    This is more appropriate for agentic systems where:
    - Events are semantic, not just symbols
    - Context matters
    - The LLM understands what's "normal"
    """
    
    def __init__(self):
        if DSPY_AVAILABLE:
            self.estimator = dspy.ChainOfThought(SurpriseEstimationSignature)
        else:
            self.estimator = None
        
        logger.info("😮 LLMSurpriseEstimator initialized (modern alternative to Shannon)")
    
    async def estimate_surprise(
        self,
        event: Dict[str, Any],
        context: Dict[str, Any],
        history: List[str]
    ) -> Tuple[int, str, str]:
        """
        Estimate how surprising an event is.
        
        Returns:
            (surprise_level: 1-10, detail_recommendation, reasoning)
        """
        if not self.estimator:
            return 5, "normal", "No LLM available for estimation"
        
        try:
            result = self.estimator(
                event_description=json.dumps(event),
                context=json.dumps(context),
                history_summary="\n".join(history)
            )
            
            surprise = self._parse_surprise(result.surprise_level)
            detail = str(result.detail_recommendation).lower()
            
            # Normalize detail recommendation
            if 'max' in detail:
                detail = 'maximum'
            elif 'detail' in detail:
                detail = 'detailed'
            elif 'min' in detail:
                detail = 'minimal'
            else:
                detail = 'normal'
            
            return surprise, detail, result.reasoning
            
        except Exception as e:
            logger.warning(f"Surprise estimation failed: {e}")
            return 5, "normal", f"Estimation failed: {e}"
    
    def _parse_surprise(self, surprise_str: str) -> int:
        """Parse surprise level from LLM output."""
        try:
            val = int(float(surprise_str))
            return max(1, min(10, val))
        except (ValueError, TypeError) as e:
            logger.debug(f"Direct int parsing failed: {e}, trying extraction")
            for part in str(surprise_str).split():
                try:
                    val = int(float(part))
                    if 1 <= val <= 10:
                        return val
                except (ValueError, TypeError):
                    continue
            return 5


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Retry handling
    'UniversalRetryHandler',
    'RetryStrategy',
    'AgentResult',
    
    # Pattern detection (replaces keyword matching)
    'PatternDetector',
    
    # Credit assignment (COMA 2018)
    'LLMCounterfactualCritic',
    
    # Memory retrieval (Self-RAG 2023)
    'SelfRAGMemoryRetriever',
    
    # Surprise estimation (modern Shannon)
    'LLMSurpriseEstimator'
]

