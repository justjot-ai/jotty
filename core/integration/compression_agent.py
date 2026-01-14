"""
Compression Agent - Intelligent Content Compression

NO HARDCODING, NO SLICING, NO RULES.
Agent decides what's important based on purpose, goal, and context.
"""

import dspy
from typing import Optional


class CompressionSignature(dspy.Signature):
    """
    Intelligently compress content while preserving what's important.
    
    The agent understands:
    - Why is this being compressed? (validation, memory, retrieval)
    - What's the overall goal?
    - What context is this part of?
    
    And decides what to keep based on importance, NOT arbitrary rules.
    """
    
    content = dspy.InputField(desc="Content to compress")
    purpose = dspy.InputField(desc="Why compress? (for_validation, for_memory, for_retrieval)")
    max_tokens = dspy.InputField(desc="Maximum tokens in compressed output")
    goal = dspy.InputField(desc="Overall system goal this relates to")
    context = dspy.InputField(desc="Surrounding context for relevance")
    
    analysis = dspy.OutputField(desc="What's important in this content and why")
    compressed_content = dspy.OutputField(desc="Compressed content preserving important parts")
    compression_ratio = dspy.OutputField(desc="Ratio of compression (e.g., '0.3' means 30% of original)")


class RelevanceRankerSignature(dspy.Signature):
    """
    Rank trajectory items by relevance to current task.
    
    NO ARBITRARY LIMITS (like  or ).
    Agent decides what's relevant based on semantic similarity and importance.
    """
    
    trajectory = dspy.InputField(desc="Full trajectory history as JSON")
    current_task = dspy.InputField(desc="Current task being executed")
    goal = dspy.InputField(desc="Overall system goal")
    max_items = dspy.InputField(desc="Maximum number of items to return")
    
    reasoning = dspy.OutputField(desc="Why these items are relevant")
    relevant_indices = dspy.OutputField(desc="Indices of relevant items (comma-separated)")
    relevance_scores = dspy.OutputField(desc="Relevance scores for each (comma-separated)")

