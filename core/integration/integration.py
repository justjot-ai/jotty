"""
Jotty v8.0 - Integration Module
================================

A-Team Approved: Ties all components together.

This module ensures:
1. GlobalContextGuard patches ALL DSPy calls
2. AlgorithmicCreditAssigner is used for credit
3. ContentGate handles all file reads
4. All components are properly initialized

Usage:
    from Jotty.core.integration import initialize_jotty
    
    # Initialize with all protections
    jotty_system = initialize_jotty(config)
    
    # Now all DSPy calls are protected, credit is algorithmic, etc.
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

from ..foundation.data_structures import SwarmConfig, SwarmLearningConfig
from ..context.global_context_guard import (
    GlobalContextGuard,
    patch_dspy_with_guard,
    unpatch_dspy
)
from ..utils.algorithmic_foundations import (
    ShapleyValueEstimator,
    DifferenceRewardEstimator,
    SurpriseEstimator,
    MutualInformationRetriever,
    GlobalContextGuard,
    ContentGate
)

logger = logging.getLogger(__name__)


class JottyIntegration:
    """
    Central integration point for all Jotty v8.0 components.
    
    Ensures:
    1. Context overflow protection on ALL LLM calls
    2. Algorithmic credit assignment (Shapley, Difference Rewards)
    3. Information-theoretic memory storage
    4. Universal document processing
    """
    
    _instance: Optional['JottyIntegration'] = None
    _initialized: bool = False
    
    def __init__(self, config: SwarmConfig = None) -> None:
        self.config = config or SwarmConfig()
        
        # Context management
        self.context_guard = GlobalContextGuard(
            max_tokens=self.config.max_context_tokens
        )
        
        # Document processing
        self.doc_processor = ContentGate(
            max_tokens=self.config.max_context_tokens
        )
        
        # Algorithmic components
        self.shapley_estimator = ShapleyValueEstimator()
        self.difference_estimator = DifferenceRewardEstimator()
        self.surprise_estimator = SurpriseEstimator()
        self.mi_retriever = MutualInformationRetriever()
        
        # Statistics
        self.stats = {
            'dspy_calls': 0,
            'context_overflows_caught': 0,
            'documents_processed': 0,
            'credit_assignments': 0
        }
        
        logger.info(" JottyIntegration initialized (v8.0 Algorithmic Edition)")
    
    @classmethod
    def get_instance(cls, config: SwarmConfig = None) -> 'JottyIntegration':
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None
    
    def patch_dspy(self) -> None:
        """
        Patch DSPy to protect all LLM calls from context overflow.
        
        MUST be called before any DSPy operations.
        """
        if not DSPY_AVAILABLE:
            logger.warning("DSPy not available, skipping patch")
            return
        
        patch_dspy_with_guard(self.context_guard)
        logger.info(" DSPy patched with GlobalContextGuard")
    
    def unpatch_dspy(self) -> None:
        """Remove DSPy patch."""
        unpatch_dspy()
        logger.info(" DSPy unpatched")
    
    def read_document(self, 
                      path: str, 
                      extraction_goal: str = "Extract key information",
                      future_tasks: list = None) -> str:
        """
        Read a document with automatic chunking if too large.
        
        ALL file reads should go through this.
        """
        self.stats['documents_processed'] += 1
        
        content = Path(path).read_text()
        return self.doc_processor.process(
            content, 
            extraction_goal=extraction_goal,
            future_tasks=future_tasks
        )
    
    async def assign_credit(self,
                            agents: list,
                            agent_capabilities: dict,
                            actions: dict,
                            states: dict,
                            trajectory: list,
                            task: str,
                            global_reward: float) -> dict:
        """
        Assign credit using algorithmic methods (Shapley + Difference).
        
        Returns dict of agent_name -> credit_score.
        """
        self.stats['credit_assignments'] += 1
        
        # Use Shapley Value for marginal contribution
        shapley_credits = await self.shapley_estimator.estimate_shapley_values(
            agents=agents,
            agent_capabilities=agent_capabilities,
            task=task,
            trajectory=trajectory,
            actual_reward=global_reward
        )
        
        # Combine with difference rewards for counterfactual impact
        difference_rewards = await self.difference_estimator.compute_difference_rewards(
            agents=agents,
            actions=actions,
            states=states,
            global_reward=global_reward
        )
        
        # Combine: 60% Shapley, 40% Difference (but this is adaptive)
        final_credits = {}
        for agent in agents:
            shapley = shapley_credits.get(agent).combined_credit if agent in shapley_credits else 0.0
            diff = difference_rewards.get(agent, 0.0)
            
            # Adaptive weighting based on confidence
            confidence = shapley_credits.get(agent).confidence if agent in shapley_credits else 0.5
            shapley_weight = 0.4 + 0.3 * confidence  # 0.4 to 0.7
            
            final_credits[agent] = shapley_weight * shapley + (1 - shapley_weight) * diff
        
        # Normalize to sum to global_reward
        total = sum(final_credits.values())
        if total > 0:
            final_credits = {k: v / total * global_reward for k, v in final_credits.items()}
        
        return final_credits
    
    async def should_store_memory(self, 
                                   event: dict, 
                                   expected: str,
                                   history: list) -> tuple:
        """
        Use information theory to decide storage level.
        
        Returns (bits_of_information, storage_level, reasoning).
        """
        bits, level, reason = self.surprise_estimator.estimate_information_content(
            event=event,
            expected=expected,
            history=history
        )
        
        detail_config = self.surprise_estimator.get_storage_detail_level(bits)
        
        return bits, detail_config, reason
    
    async def retrieve_memories(self,
                                 task: str,
                                 uncertainties: list,
                                 candidates: list,
                                 top_k: int = 5) -> list:
        """
        Retrieve memories using mutual information maximization.
        """
        selected_ids, reduction, reason = self.mi_retriever.retrieve(
            task=task,
            uncertainties=uncertainties,
            candidates=candidates,
            top_k=top_k
        )
        
        logger.debug(f"Retrieved {len(selected_ids)} memories (uncertainty reduction: {reduction:.2f})")
        
        return [candidates[i] for i in selected_ids if i < len(candidates)]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get integration statistics."""
        return {
            **self.stats,
            'context_guard_stats': self.context_guard.get_statistics(),
            'doc_processor_stats': {
                'processed': self.doc_processor.documents_processed,
                'chunked': self.doc_processor.documents_chunked
            }
        }


def initialize_jotty(config: SwarmConfig = None) -> JottyIntegration:
    """
    Initialize JOTTY with all v8.0 protections.
    
    This is the recommended entry point for using JOTTY.
    
    Returns:
        JottyIntegration instance with all components ready.
    """
    integration = JottyIntegration.get_instance(config)
    
    # Patch DSPy to protect all LLM calls
    integration.patch_dspy()
    
    logger.info(" Jotty v8.0 initialized with full algorithmic protection")
    
    return integration


# Singleton accessor
def get_jotty() -> Optional[JottyIntegration]:
    """Get existing JOTTY integration or None if not initialized."""
    return JottyIntegration._instance


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'JottyIntegration',
    'initialize_jotty',
    'get_jotty',
]

