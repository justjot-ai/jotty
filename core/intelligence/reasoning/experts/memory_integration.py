"""
Memory Integration for Expert Agent Improvements

Integrates expert agent improvements with Jotty's SwarmMemory system
instead of file-based storage.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel

from ..memory.cortex import SwarmMemory

logger = logging.getLogger(__name__)


def store_improvement_to_memory(
    memory: SwarmMemory, improvement: Dict[str, Any], expert_name: str, domain: str
) -> None:
    """
    Store an improvement to Jotty's SwarmMemory system.

    Improvements are stored as PROCEDURAL or META level memories:
    - PROCEDURAL: How to generate correct outputs (action sequences)
    - META: Wisdom about when to use what patterns (learning wisdom)

    Args:
        memory: SwarmMemory instance
        improvement: Improvement dictionary
        expert_name: Name of the expert agent
        domain: Domain of the expert (e.g., "mermaid", "pipeline")
    """
    if not memory:
        logger.warning("No memory system provided, cannot store improvement")
        return

    # Determine memory level based on improvement type
    # PROCEDURAL: Specific patterns for generating outputs
    # META: General wisdom about when to use patterns
    if improvement.get("improvement_type") == "teacher_correction":
        level = MemoryLevel.PROCEDURAL  # Specific correction pattern
    else:
        level = MemoryLevel.META  # General learning wisdom

    # Create content for memory
    content = json.dumps(improvement, indent=2, ensure_ascii=False)

    # Create context
    context = {
        "expert_name": expert_name,
        "domain": domain,
        "task": improvement.get("task", ""),
        "iteration": improvement.get("iteration", 0),
        "improvement_type": improvement.get("improvement_type", "unknown"),
        "source": "optimization_pipeline",
    }

    # Goal for retrieval
    goal = f"expert_{domain}_improvements"

    # Store in memory
    try:
        memory_entry = memory.store(
            content=content,
            level=level,
            context=context,
            goal=goal,
            initial_value=1.0,  # High value for learned improvements
        )
        logger.info(f"Stored improvement to memory: {memory_entry.key} (level: {level.value})")
        return memory_entry
    except Exception as e:
        logger.error(f"Failed to store improvement to memory: {e}")
        return None


def retrieve_improvements_from_memory(
    memory: SwarmMemory,
    expert_name: str,
    domain: str,
    task: Optional[str] = None,
    max_results: int = 20,
) -> List[Dict[str, Any]]:
    """
    Retrieve improvements from Jotty's SwarmMemory system.

    Args:
        memory: SwarmMemory instance
        expert_name: Name of the expert agent
        domain: Domain of the expert
        task: Optional task filter
        max_results: Maximum number of improvements to retrieve

    Returns:
        List of improvement dictionaries
    """
    if not memory:
        return []

    # Build query
    query_parts = [
        f"expert agent {expert_name}",
        f"domain {domain}",
        "improvements learned patterns",
    ]
    if task:
        query_parts.append(f"task {task}")
    query = " ".join(query_parts)

    # Retrieve from PROCEDURAL and META levels
    try:
        memory_entries = memory.retrieve(
            query=query,
            goal=f"expert_{domain}_improvements",
            budget_tokens=max_results * 200,  # Estimate tokens
            levels=[MemoryLevel.PROCEDURAL, MemoryLevel.META],
        )

        improvements = []
        for entry in memory_entries[:max_results]:
            try:
                # Try to parse as JSON
                improvement_data = json.loads(entry.content)
                if isinstance(improvement_data, dict):
                    improvements.append(improvement_data)
                elif isinstance(improvement_data, list):
                    improvements.extend(improvement_data)
            except (json.JSONDecodeError, TypeError):
                # If not JSON, create from memory entry
                improvements.append(
                    {
                        "timestamp": (
                            entry.last_accessed.isoformat()
                            if entry.last_accessed
                            else datetime.now().isoformat()
                        ),
                        "task": entry.context.get("task", "Unknown"),
                        "learned_pattern": entry.content,
                        "source": "memory",
                        "memory_level": entry.level.value,
                        "memory_key": entry.key,
                    }
                )

        logger.info(f"Retrieved {len(improvements)} improvements from memory")
        return improvements
    except Exception as e:
        logger.error(f"Failed to retrieve improvements from memory: {e}")
        return []


def sync_improvements_to_memory(
    memory: SwarmMemory, improvements: List[Dict[str, Any]], expert_name: str, domain: str
) -> int:
    """
    Sync a list of improvements to memory system.

    Args:
        memory: SwarmMemory instance
        improvements: List of improvement dictionaries
        expert_name: Name of the expert agent
        domain: Domain of the expert

    Returns:
        Number of improvements stored
    """
    if not memory:
        return 0

    stored_count = 0
    for improvement in improvements:
        try:
            entry = store_improvement_to_memory(memory, improvement, expert_name, domain)
            if entry:
                stored_count += 1
        except Exception as e:
            logger.warning(f"Failed to store improvement to memory: {e}")

    logger.info(f"Synced {stored_count}/{len(improvements)} improvements to memory")
    return stored_count


def retrieve_synthesized_improvements(
    memory: SwarmMemory, expert_name: str, domain: str, task: Optional[str] = None
) -> str:
    """
    Retrieve and synthesize improvements from memory into coherent wisdom.

    Uses memory system's synthesis capability to create a consolidated summary
    of all improvements, finding patterns and resolving contradictions.

    Args:
        memory: SwarmMemory instance
        expert_name: Name of the expert agent
        domain: Domain of the expert
        task: Optional task filter

    Returns:
        Synthesized improvements as a coherent text summary
    """
    if not memory:
        return ""

    # Build query
    query_parts = [
        f"expert agent {expert_name}",
        f"domain {domain}",
        "improvements learned patterns",
    ]
    if task:
        query_parts.append(f"task {task}")
    query = " ".join(query_parts)

    goal = f"expert_{domain}_improvements"

    try:
        # Use retrieve_and_synthesize for brain-inspired synthesis
        synthesized = memory.retrieve_and_synthesize(
            query=query,
            goal=goal,
            levels=[MemoryLevel.PROCEDURAL, MemoryLevel.META, MemoryLevel.SEMANTIC],
            context_hints=f"Expert agent improvements for {domain} domain. Focus on patterns and best practices.",
        )

        logger.info(f"Retrieved synthesized improvements (length: {len(synthesized)} chars)")
        return synthesized
    except Exception as e:
        logger.error(f"Failed to synthesize improvements from memory: {e}")
        return ""


async def retrieve_synthesized_improvements_async(
    memory: SwarmMemory, expert_name: str, domain: str, task: Optional[str] = None
) -> str:
    """
    Async version of retrieve_synthesized_improvements.
    """
    if not memory:
        return ""

    query_parts = [
        f"expert agent {expert_name}",
        f"domain {domain}",
        "improvements learned patterns",
    ]
    if task:
        query_parts.append(f"task {task}")
    query = " ".join(query_parts)

    goal = f"expert_{domain}_improvements"

    try:
        synthesized = await memory.retrieve_and_synthesize_async(
            query=query,
            goal=goal,
            levels=[MemoryLevel.PROCEDURAL, MemoryLevel.META, MemoryLevel.SEMANTIC],
            context_hints=f"Expert agent improvements for {domain} domain. Focus on patterns and best practices.",
        )

        logger.info(f"Retrieved synthesized improvements (length: {len(synthesized)} chars)")
        return synthesized
    except Exception as e:
        logger.error(f"Failed to synthesize improvements from memory: {e}")
        return ""


def consolidate_improvements(memory: SwarmMemory, expert_name: str, domain: str) -> Dict[str, Any]:
    """
    Consolidate improvements stored in memory.

    Similar to memory consolidation cycle:
    - Consolidates PROCEDURAL improvements into SEMANTIC patterns
    - Groups similar improvements together
    - Extracts common patterns and best practices
    - Promotes important patterns to META level

    Args:
        memory: SwarmMemory instance
        expert_name: Name of the expert agent
        domain: Domain of the expert

    Returns:
        Dictionary with consolidation results
    """
    if not memory:
        return {"consolidated": 0, "preferences": 0, "merged": 0}

    try:
        # Get all PROCEDURAL improvements (raw patterns)
        procedural_improvements = memory.retrieve(
            query=f"expert agent {expert_name} domain {domain} improvements",
            goal=f"expert_{domain}_improvements",
            budget_tokens=10000,
            levels=[MemoryLevel.PROCEDURAL],
        )

        if len(procedural_improvements) < 2:
            logger.info("Not enough improvements to consolidate (need at least 2)")
            return {"consolidated": 0, "preferences": 0, "merged": 0}

        # Group similar improvements by pattern type
        pattern_groups = {}
        for entry in procedural_improvements:
            try:
                improvement_data = json.loads(entry.content)
                pattern = improvement_data.get("learned_pattern", "")
                task = improvement_data.get("task", "")

                # Extract pattern type (syntax, complexity, tags, etc.)
                pattern_type = "general"
                if "plantuml" in pattern.lower() or "mermaid" in pattern.lower():
                    pattern_type = "syntax_format"
                elif "simple" in pattern.lower() or "complex" in pattern.lower():
                    pattern_type = "complexity"
                elif "@startuml" in pattern.lower() or "@enduml" in pattern.lower():
                    pattern_type = "tags"
                elif task:
                    pattern_type = f"task_{task[:30]}"

                if pattern_type not in pattern_groups:
                    pattern_groups[pattern_type] = []
                pattern_groups[pattern_type].append(entry)
            except Exception:
                continue

        consolidated_count = 0
        merged_count = 0

        # Consolidate each group
        for pattern_type, entries in pattern_groups.items():
            if len(entries) >= 2:  # Consolidate groups with 2+ similar improvements
                # Extract common pattern
                patterns = []
                for entry in entries:
                    try:
                        improvement_data = json.loads(entry.content)
                        pattern = improvement_data.get("learned_pattern", "")
                        if pattern:
                            patterns.append(pattern)
                    except Exception:
                        continue

                if patterns:
                    # Create consolidated pattern summary
                    consolidated_pattern = f"Common pattern for {pattern_type}: " + "; ".join(
                        [
                            p[:100] + "..." if len(p) > 100 else p
                            for p in patterns[:3]  # Use first 3 patterns
                        ]
                    )

                    # Use synthesis if available (requires LLM)
                    try:
                        synthesized = memory.retrieve_and_synthesize(
                            query=f"Extract common pattern from {len(patterns)} similar improvements about {pattern_type}",
                            goal=f"expert_{domain}_improvements",
                            levels=[MemoryLevel.PROCEDURAL],
                            context_hints=f"These {len(patterns)} improvements are about {pattern_type}. Extract the common rule or pattern.",
                        )

                        if synthesized and len(synthesized) > 50:  # Valid synthesis
                            consolidated_pattern = synthesized
                    except Exception:
                        # Fallback to simple consolidation
                        pass

                    # Store consolidated pattern as SEMANTIC memory
                    consolidated_entry = memory.store(
                        content=consolidated_pattern,
                        level=MemoryLevel.SEMANTIC,
                        context={
                            "expert_name": expert_name,
                            "domain": domain,
                            "source": "consolidation",
                            "pattern_type": pattern_type,
                            "consolidated_from": len(entries),
                            "original_entries": [e.key for e in entries],
                        },
                        goal=f"expert_{domain}_improvements",
                        initial_value=1.0,
                    )

                    consolidated_count += 1
                    merged_count += len(entries)
                    logger.info(
                        f"Consolidated {len(entries)} improvements of type '{pattern_type}' into SEMANTIC pattern"
                    )

        logger.info(
            f"Consolidation complete: {consolidated_count} patterns consolidated, {merged_count} improvements merged"
        )
        return {"consolidated": consolidated_count, "preferences": 0, "merged": merged_count}

    except Exception as e:
        logger.error(f"Failed to consolidate improvements: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return {"consolidated": 0, "preferences": 0, "merged": 0}


async def run_improvement_consolidation_cycle(
    memory: SwarmMemory, expert_name: str, domain: str
) -> Dict[str, Any]:
    """
    Run full consolidation cycle for expert agent improvements.

    Similar to memory consolidation cycle:
    1. Consolidate PROCEDURAL improvements to SEMANTIC patterns
    2. Extract preferences/patterns
    3. Promote important patterns to META level

    Args:
        memory: SwarmMemory instance
        expert_name: Name of the expert agent
        domain: Domain of the expert

    Returns:
        Dictionary with consolidation results
    """
    if not memory:
        return {"consolidated": 0, "preferences": 0}

    logger.info(f"Running improvement consolidation cycle for expert {expert_name} ({domain})")

    # Consolidate improvements
    consolidation_result = consolidate_improvements(memory, expert_name, domain)

    logger.info(
        f"Consolidation complete: {consolidation_result.get('consolidated', 0)} semantic patterns, "
        f"{consolidation_result.get('preferences', 0)} preferences"
    )

    return consolidation_result
