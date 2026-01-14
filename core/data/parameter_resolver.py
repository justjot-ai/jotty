"""
🎯 SOTA Agentic Parameter Resolver

Replaces ALL fuzzy/regex/rule-based matching with intelligent LLM-based resolution.

Design Principles:
1. LLM-Based Matching: Semantic understanding, not string similarity
2. Self-Describing Data: Every artifact has description + tags
3. Message Passing Protocol: Actors can REQUEST data explicitly
4. Confidence Scoring: Never force matches
5. Fully Generic: Works for any actor, any parameter

Author: A-TEAM
Date: 2025-12-26
"""

import dspy
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Type

logger = logging.getLogger(__name__)


class ParameterMatchingSignature(dspy.Signature):
    """
    Expert at matching actor parameter requirements to available data.
    
    Match by MEANING, not string similarity.
    Consider synonyms, context, and semantic relationships.
    """
    
    actor_name: str = dspy.InputField(desc="Name of the actor requesting data")
    parameter_name: str = dspy.InputField(desc="Name of the missing parameter")
    parameter_type: str = dspy.InputField(desc="Expected Python type (e.g., 'list', 'dict', 'str')")
    parameter_purpose: str = dspy.InputField(desc="What the actor will use this parameter for")
    
    available_data: str = dspy.InputField(
        desc="JSON of available data: {key: {value_preview, type, description, tags, source}}"
    )
    
    reasoning: str = dspy.OutputField(desc="Step-by-step reasoning for the match")
    best_match_key: str = dspy.OutputField(desc="The key of the best matching data, or 'NO_MATCH' if no good match")
    confidence: float = dspy.OutputField(desc="Confidence in match (0.0-1.0). Use NO_MATCH if confidence < 0.7")
    semantic_explanation: str = dspy.OutputField(desc="Why this data semantically matches the parameter need")


class AgenticParameterResolver(dspy.Module):
    """
    SOTA Agentic Parameter Resolver using LLM reasoning.
    
    Replaces ALL fuzzy/regex/rule-based matching with intelligent LLM-based resolution.
    
    Features:
    - Semantic matching (meaning, not strings)
    - Self-describing data protocol
    - Confidence scoring
    - Full reasoning transparency
    - Zero hardcoding
    """
    
    def __init__(self, llm: Optional[dspy.LM] = None):
        super().__init__()
        self.llm = llm or dspy.settings.lm
        
        # Use dspy.settings.lm if available, otherwise ChainOfThought will use default
        self.resolver = dspy.ChainOfThought(ParameterMatchingSignature)
        logger.info("✅ AgenticParameterResolver initialized with LLM-based matching")
    
    def resolve_parameter(
        self,
        actor_name: str,
        parameter_name: str,
        parameter_type: Type,
        parameter_purpose: str,
        available_data: Dict[str, Dict[str, Any]],
        min_confidence: float = 0.7
    ) -> Tuple[Optional[str], float, str]:
        """
        Intelligently match a missing parameter to available data.
        
        Uses LLM reasoning to understand semantic relationships between
        the parameter need and available data.
        
        Args:
            actor_name: Name of the actor
            parameter_name: Missing parameter name
            parameter_type: Expected type
            parameter_purpose: What actor needs this for
            available_data: Dict of {key: {value, type, description, tags, source}}
            min_confidence: Minimum confidence to accept match (default: 0.7)
            
        Returns:
            (matched_key, confidence, reasoning) or (None, 0.0, reason_for_no_match)
        """
        logger.info(f"🔍 [AGENTIC RESOLVER] Resolving '{parameter_name}' for {actor_name}")
        # Handle parameter_type being a Type or a string
        type_name = parameter_type.__name__ if hasattr(parameter_type, '__name__') else str(parameter_type)
        logger.info(f"    Type: {type_name}, Purpose: {parameter_purpose}")
        logger.info(f"    Available data sources: {list(available_data.keys())}")
        
        # Prepare available_data for LLM (self-describing protocol)
        data_summary = {}
        for key, data_info in available_data.items():
            data_summary[key] = {
                'type': data_info.get('type', 'unknown'),
                'description': data_info.get('description', 'No description'),
                'tags': data_info.get('tags', []),
                'source': data_info.get('source', 'unknown'),
                'value_preview': str(data_info.get('value', ''))[:150]  # Preview for context
            }
        
        try:
            # Handle parameter_type being a Type or a string
            type_name = parameter_type.__name__ if hasattr(parameter_type, '__name__') else str(parameter_type)
            
            result = self.resolver(
                actor_name=actor_name,
                parameter_name=parameter_name,
                parameter_type=type_name,
                parameter_purpose=parameter_purpose,
                available_data=json.dumps(data_summary, indent=2)
            )
            
            best_match = result.best_match_key
            confidence = float(result.confidence)
            reasoning = result.reasoning
            explanation = result.semantic_explanation
            
            logger.info(f"✅ [AGENTIC RESOLVER] Best match: '{best_match}' (confidence: {confidence:.2f})")
            logger.info(f"    Reasoning: {reasoning[:200]}...")
            logger.info(f"    Explanation: {explanation[:150]}...")
            
            if best_match == "NO_MATCH" or confidence < min_confidence:
                logger.warning(f"⚠️  [AGENTIC RESOLVER] No confident match found (confidence: {confidence:.2f})")
                logger.warning(f"    Reasoning: {reasoning[:200]}...")
                return None, confidence, reasoning
            
            return best_match, confidence, explanation
            
        except Exception as e:
            logger.error(f"❌ [AGENTIC RESOLVER] Error: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None, 0.0, f"Resolver failed: {e}"
    
    def build_self_describing_data(
        self,
        key: str,
        value: Any,
        description: str,
        tags: List[str],
        source: str,
        generated_at: Optional[str] = None,
        purpose: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a self-describing data artifact.
        
        This is the standard protocol for all data in ReVal.
        Every piece of data should include semantic metadata.
        
        Args:
            key: Unique identifier for this data
            value: The actual data
            description: Human-readable description
            tags: Semantic tags for discovery
            source: Where this came from
            generated_at: Which actor generated it
            purpose: What this data is for
            
        Returns:
            Self-describing data dict
        """
        return {
            'key': key,
            'value': value,
            'type': type(value).__name__,
            'description': description,
            'tags': tags,
            'source': source,
            'generated_at': generated_at or source,
            'purpose': purpose or description
        }


__all__ = ['AgenticParameterResolver', 'ParameterMatchingSignature']

