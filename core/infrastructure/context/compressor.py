"""
 AGENTIC COMPRESSOR
====================

NO RULE-BASED COMPRESSION!
NO [:5000] SLICING!

Uses LLM to intelligently compress content based on:
- Task context (what the agent needs to do)
- Priority hints (what information is critical)
- Target token budget (how much space available)

The compressor UNDERSTANDS the content and preserves what's important!
"""

import dspy
from typing import Dict, List, Optional, Any
import logging

from ..utils.tokenizer import SmartTokenizer
from . import utils as ctx_utils

logger = logging.getLogger(__name__)


class CompressionSignature(dspy.Signature):
    """
    Intelligently compress content while preserving task-critical information.
    
     A-TEAM ENHANCEMENT: Shapley-impact-based prioritization.
    Content that contributed to past successes gets higher preservation priority.
    """
    
    full_content = dspy.InputField(desc="Full content to compress (NO slicing!)")
    task_description = dspy.InputField(desc="What the agent needs to do with this content")
    target_tokens = dspy.InputField(desc="Target token count after compression")
    priority_keywords = dspy.InputField(desc="Keywords/concepts that MUST be preserved")
    content_type = dspy.InputField(desc="Type of content: metadata, conversation, code, etc.")
    # NEW: Shapley impact hints
    high_impact_items = dspy.InputField(desc="Items with HIGH Shapley credit (preserve these first!)")
    low_impact_items = dspy.InputField(desc="Items with LOW Shapley credit (can be compressed/removed)")
    
    compressed_content = dspy.OutputField(desc="Compressed content preserving critical information")
    compression_ratio = dspy.OutputField(desc="Percentage of original content retained")
    what_was_removed = dspy.OutputField(desc="Brief summary of what information was removed")
    quality_score = dspy.OutputField(desc="Self-assessed quality score (0-10)")


class AgenticCompressor:
    """
    Smart compression agent that understands task context.
    
    NO rule-based slicing! Uses LLM to intelligently preserve important information.
    """
    
    def __init__(self, lm: Any = None) -> None:
        """
        Initialize compressor.
        
        Args:
            lm: Optional DSPy language model. If None, uses dspy.settings.lm.
        """
        self.lm = lm
        if lm is None:
            # Use global DSPy LM if available
            if hasattr(dspy.settings, 'lm') and dspy.settings.lm:
                self.lm = dspy.settings.lm
        
        self.compressor = dspy.ChainOfThought(CompressionSignature)
        self.compression_stats = []
    
    async def compress(
        self,
        content: str,
        task_context: Dict,
        target_tokens: int,
        shapley_credits: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Intelligently compress content based on task needs.
        
         A-TEAM ENHANCEMENT: Shapley-impact-based prioritization.
        
        Args:
            content: FULL content (NO pre-slicing!)
            task_context: {
                'actor_name': str,
                'goal': str,
                'query': str,
                'priority_keywords': List[str],  # What's most important
                'content_type': str  # 'metadata', 'conversation', etc.
            }
            target_tokens: Target token count
            shapley_credits: Optional dict of {item: credit} for prioritization
        
        Returns:
            Compressed content preserving critical information
        """
        current_tokens = ctx_utils.estimate_tokens(content)
        logger.info(f" Agentic compression for {task_context.get('actor_name', 'unknown')}...")
        logger.info(f"   Original length: {len(content)} chars (~{current_tokens} tokens)")
        logger.info(f"   Target: {target_tokens} tokens")
        
        if current_tokens <= target_tokens:
            logger.info(f" No compression needed ({current_tokens} <= {target_tokens} tokens)")
            return content
        
        # Build priority keywords
        priority_keywords = ", ".join(task_context.get('priority_keywords', []))
        if not priority_keywords:
            # Extract from query/goal
            query = task_context.get('query', '')
            goal = task_context.get('goal', '')
            priority_keywords = f"Query: {query}, Goal: {goal}"
        
        # A-TEAM: Extract high/low impact items from Shapley credits
        high_impact_items = ""
        low_impact_items = ""
        if shapley_credits:
            # Sort by credit value
            sorted_credits = sorted(shapley_credits.items(), key=lambda x: x[1], reverse=True)
            # Top 20% are high impact
            top_n = max(1, len(sorted_credits) // 5)
            high_impact = [item for item, credit in sorted_credits[:top_n]]
            low_impact = [item for item, credit in sorted_credits[-top_n:] if credit < 0.3]
            
            high_impact_items = ", ".join(high_impact) if high_impact else "None specified"
            low_impact_items = ", ".join(low_impact) if low_impact else "None - preserve all"
            
            logger.info(f"   High-impact items: {high_impact_items}")
            logger.info(f"   Low-impact items: {low_impact_items}")
        else:
            high_impact_items = "None specified - use task context to infer"
            low_impact_items = "None specified - preserve important information"
        
        # Call the agentic compressor
        with dspy.context(lm=self.lm):
            result = self.compressor(
                full_content=content, # FULL content, NO slicing!
                high_impact_items=high_impact_items,
                low_impact_items=low_impact_items,
                task_description=f"Actor '{task_context.get('actor_name')}' needs to: {task_context.get('goal', 'process this content')}",
                target_tokens=str(target_tokens),
                priority_keywords=priority_keywords,
                content_type=task_context.get('content_type', 'general')
            )
        
        compressed = result.compressed_content
        compressed_tokens = ctx_utils.estimate_tokens(compressed)
        
        logger.info(f" Compressed: {current_tokens} → {compressed_tokens} tokens ({result.compression_ratio})")
        logger.info(f"   Quality score: {result.quality_score}/10")
        logger.info(f"   Removed: {result.what_was_removed}")
        
        # Record stats for monitoring
        self.compression_stats.append({
            'actor': task_context.get('actor_name'),
            'original_tokens': current_tokens,
            'compressed_tokens': compressed_tokens,
            'ratio': result.compression_ratio,
            'quality': result.quality_score,
            'removed': result.what_was_removed,
        })
        
        # Verify quality
        try:
            quality = float(result.quality_score)
            if quality < 5.0:
                logger.warning(f" Low compression quality ({quality}/10)!")
                logger.warning(f"   Consider increasing target_tokens or reviewing compression")
        except (ValueError, TypeError, AttributeError) as e:
            logger.debug(f"Quality score parsing failed: {e}")
            pass
        
        return compressed
    
    # NEW: AgentSlack-compatible simpler API
    async def compress_simple(
        self,
        data: str,
        target_ratio: float = 0.5,
        preserve_critical: bool = False
    ) -> str:
        """
        Simple compression API for AgentSlack.
        
        Args:
            data: Input data (string)
            target_ratio: Target compression ratio (0.0-1.0), e.g. 0.5 = 50% of original
            preserve_critical: Try to preserve CRITICAL/IMPORTANT markers
        
        Returns:
            Compressed string
        """
        logger.info(f" [AgentSlack API] Compressing: {len(data)} chars, target_ratio={target_ratio}")

        if not data:
            logger.info(" Empty data, returning empty string")
            return ""

        # Calculate target tokens using shared utility
        current_tokens = ctx_utils.estimate_tokens(data)
        target_tokens = int(current_tokens * target_ratio)
        
        if current_tokens <= target_tokens:
            logger.info(f" No compression needed ({current_tokens} <= {target_tokens} tokens)")
            return data
        
        # If we have LM, use intelligent compression
        if self.lm:
            try:
                # Build simple task context
                task_context = {
                    'actor_name': 'AgentSlack',
                    'goal': 'compress data for transmission',
                    'priority_keywords': ['CRITICAL', 'IMPORTANT'] if preserve_critical else [],
                    'content_type': 'general'
                }
                
                return await self.compress(data, task_context, target_tokens)
            except Exception as e:
                logger.warning(f" LLM compression failed: {e}, using simple truncation")
        
        # Fallback: simple truncation (preserving CRITICAL sections if requested)
        if preserve_critical:
            # Extract CRITICAL sections
            lines = data.split('\n')
            critical_lines = [l for l in lines if 'CRITICAL' in l.upper() or 'IMPORTANT' in l.upper()]
            other_lines = [l for l in lines if l not in critical_lines]
            
            # Calculate how many other lines we can include
            critical_text = '\n'.join(critical_lines)
            critical_chars = len(critical_text)
            target_chars = target_tokens * 4
            
            remaining_chars = target_chars - critical_chars
            
            if remaining_chars > 0:
                # Include some other lines
                other_text = '\n'.join(other_lines)
                truncated_other = other_text[:remaining_chars]
                result = critical_text + '\n' + truncated_other
            else:
                result = critical_text[:target_chars]
            
            logger.info(f" Compressed with CRITICAL preservation: {len(data)} → {len(result)} chars")
            return result
        else:
            # Simple truncation
            target_chars = target_tokens * 4
            result = data[:target_chars]
            logger.info(f" Simple truncation: {len(data)} → {len(result)} chars")
            return result
    
    def get_stats(self) -> Dict:
        """Get compression statistics."""
        if not self.compression_stats:
            return {}
        
        total_compressions = len(self.compression_stats)
        avg_quality = sum(float(s.get('quality', 0)) for s in self.compression_stats) / total_compressions
        
        return {
            'total_compressions': total_compressions,
            'average_quality': avg_quality,
            'recent': self.compression_stats[-10:],  # Last 10
        }

