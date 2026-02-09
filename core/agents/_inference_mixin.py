"""AgenticPlanner Inference Mixin - Task type and capability inference."""

import json
import logging
from typing import Dict, Any, List, Optional

from .agentic_planner import _get_task_type

logger = logging.getLogger(__name__)


class InferenceMixin:
    # Per-session cache: avoids redundant LLM calls for same task string.
    # Key = first 200 chars of task (stable identifier), Value = (TaskType, reasoning, confidence)
    _task_type_cache: dict = {}

    def infer_task_type(self, task: str):
        """
        Infer task type using LLM semantic understanding.

        Cached: repeated calls for the same task return instantly (saves ~3-5s per call).

        Args:
            task: Task description

        Returns:
            (TaskType, reasoning, confidence)
        """
        TaskType = _get_task_type()

        # Check cache first — task type for same text doesn't change
        cache_key = task[:200]
        if cache_key in InferenceMixin._task_type_cache:
            cached = InferenceMixin._task_type_cache[cache_key]
            logger.info(f"📋 Task type inferred: {cached[0].value} (confidence: {cached[2]:.2f}) [cached]")
            return cached

        try:
            import dspy
            import asyncio
            import re

            # Prepare task for inference (preserves full context)
            task_for_inference = self._abstract_task_for_planning(task)

            # Use fast LM (Haiku) for classification - much faster than Sonnet
            classification_lm = self._fast_lm or dspy.settings.lm

            # Call with fast LM for quick classification
            if classification_lm:
                with dspy.context(lm=classification_lm):
                    result = self.task_type_inferrer(task_description=task_for_inference)
                logger.debug(f"Task type inference using fast model: {self._fast_model}")
            else:
                # Fallback to default LM
                result = self.task_type_inferrer(task_description=task_for_inference)

            # Parse task_type field
            task_type_str = str(result.task_type).lower().strip().split()[0] if result.task_type else 'unknown'

            task_type_map = {
                'research': TaskType.RESEARCH,
                'comparison': TaskType.COMPARISON,
                'creation': TaskType.CREATION,
                'communication': TaskType.COMMUNICATION,
                'analysis': TaskType.ANALYSIS,
                'automation': TaskType.AUTOMATION,
            }
            task_type = task_type_map.get(task_type_str, TaskType.UNKNOWN)

            # Override if LLM returned unknown but keywords clearly indicate creation
            if task_type == TaskType.UNKNOWN:
                task_lower = task.lower()
                if any(w in task_lower for w in ['create', 'build', 'make', 'write', 'generate', 'implement']):
                    task_type = TaskType.CREATION
                    logger.info(f"Overriding 'unknown' to 'creation' based on keywords")

            # Parse confidence
            try:
                confidence_match = re.search(r'(\d+\.?\d*)', str(result.confidence))
                confidence = float(confidence_match.group(1)) if confidence_match else 0.7
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError, AttributeError):
                confidence = 0.7

            reasoning = str(result.reasoning).strip() if result.reasoning else f"Inferred as {task_type_str}"

            logger.info(f"📋 Task type inferred: {task_type.value} (confidence: {confidence:.2f})")
            result_tuple = (task_type, reasoning, confidence)
            InferenceMixin._task_type_cache[cache_key] = result_tuple
            return result_tuple

        except Exception as e:
            error_str = str(e)
            logger.warning(f"Task type inference failed: {e}, using keyword fallback")

            # Detect if LLM asked for clarification instead of classifying
            clarification_markers = [
                'could you', 'please provide', 'what', 'which', 'can you',
                'i need', 'clarify', 'more information', 'specify'
            ]
            if any(marker in error_str.lower() for marker in clarification_markers):
                logger.info("LLM asked for clarification - defaulting to analysis")
                return TaskType.ANALYSIS, "LLM requested clarification - default to analysis", 0.5

            # Enhanced keyword fallback with more coverage
            task_lower = task.lower()

            # Creation keywords
            if any(w in task_lower for w in ['create', 'generate', 'make', 'build', 'write', 'implement', 'develop', 'code']):
                return TaskType.CREATION, "Keyword fallback: creation task", 0.6
            # Comparison keywords
            elif any(w in task_lower for w in ['compare', 'vs', 'versus', 'comparison', 'difference', 'better']):
                return TaskType.COMPARISON, "Keyword fallback: comparison task", 0.6
            # Research keywords
            elif any(w in task_lower for w in ['research', 'find', 'search', 'discover', 'look up', 'what is']):
                return TaskType.RESEARCH, "Keyword fallback: research task", 0.6
            # Analysis keywords (including calculation)
            elif any(w in task_lower for w in ['analyze', 'analysis', 'evaluate', 'calculate', 'compute', 'answer', 'solve', 'sum', 'result']):
                return TaskType.ANALYSIS, "Keyword fallback: analysis task", 0.6
            # Automation keywords
            elif any(w in task_lower for w in ['automate', 'schedule', 'pipeline', 'workflow', 'cron']):
                return TaskType.AUTOMATION, "Keyword fallback: automation task", 0.6
            # Communication keywords
            elif any(w in task_lower for w in ['send', 'email', 'notify', 'message', 'communicate']):
                return TaskType.COMMUNICATION, "Keyword fallback: communication task", 0.6

            # Default to ANALYSIS for any ambiguous task (not UNKNOWN)
            # This ensures we always try to do something useful
            fallback = (TaskType.ANALYSIS, "Ambiguous task - defaulting to analysis", 0.4)
            InferenceMixin._task_type_cache[cache_key] = fallback
            return fallback

    def infer_capabilities(self, task: str) -> tuple[List[str], str]:
        """
        Infer required capabilities from task description.

        Uses fast LLM to determine what capabilities (data-fetch, communicate,
        visualize, etc.) are needed to complete the task.

        Args:
            task: Task description

        Returns:
            (capabilities, reasoning) - List of capability strings and explanation
        """
        # Default capabilities based on common patterns
        default_capabilities = ["analyze"]

        try:
            import dspy

            # Clean task for inference
            task_for_inference = self._abstract_task_for_planning(task)

            # Use fast LM for quick classification
            result = self._call_with_retry(
                module=self.capability_inferrer,
                kwargs={'task_description': task_for_inference},
                max_retries=2,
                lm=self._fast_lm
            )

            # Parse capabilities from result
            capabilities_str = str(result.capabilities).strip()

            # Try to parse as JSON
            try:
                if capabilities_str.startswith('['):
                    capabilities = json.loads(capabilities_str)
                else:
                    # Extract from text like "data-fetch, communicate"
                    capabilities = [c.strip().lower() for c in capabilities_str.replace('"', '').replace('[', '').replace(']', '').split(',')]
            except json.JSONDecodeError:
                # Fallback: extract known capabilities from text
                known_caps = ['data-fetch', 'research', 'analyze', 'visualize', 'document', 'communicate', 'file-ops', 'code', 'media']
                capabilities = [c for c in known_caps if c in capabilities_str.lower()]

            # Validate and clean
            capabilities = [c.strip().lower() for c in capabilities if c.strip()][:4]

            if not capabilities:
                capabilities = default_capabilities

            reasoning = str(result.reasoning).strip() if result.reasoning else "Inferred from task"

            logger.info(f"🎯 Capabilities inferred: {capabilities}")
            return capabilities, reasoning

        except Exception as e:
            logger.warning(f"Capability inference failed: {e}, using keyword fallback")

            # Keyword-based fallback
            task_lower = task.lower()
            capabilities = []

            if any(w in task_lower for w in ['weather', 'stock', 'price', 'data', 'fetch', 'get']):
                capabilities.append('data-fetch')
            if any(w in task_lower for w in ['research', 'search', 'find', 'look up']):
                capabilities.append('research')
            if any(w in task_lower for w in ['chart', 'graph', 'slide', 'visual', 'diagram']):
                capabilities.append('visualize')
            if any(w in task_lower for w in ['pdf', 'report', 'document']):
                capabilities.append('document')
            if any(w in task_lower for w in ['telegram', 'slack', 'email', 'send', 'notify']):
                capabilities.append('communicate')
            if any(w in task_lower for w in ['file', 'save', 'write', 'read']):
                capabilities.append('file-ops')
            if any(w in task_lower for w in ['analyze', 'calculate', 'compute']):
                capabilities.append('analyze')

            if not capabilities:
                capabilities = default_capabilities

            return capabilities, "Keyword-based fallback"

    # Skills that depend on external services that may be unreliable
    # These are deprioritized in skill selection to prefer more reliable alternatives
    DEPRIORITIZED_SKILLS = {
        'search-to-justjot-idea',      # JustJot API issues
        'mcp-justjot-mcp-client',      # MCP timeout issues
        'mcp-justjot',                 # MCP timeout issues
        'justjot-mcp-http',            # JustJot API issues
        'notion-research-documentation',  # Requires Notion API setup
        'reddit-trending-to-justjot',  # JustJot API issues
        'notebooklm-pdf',              # Requires browser sign-in
        'oauth-automation',            # Requires browser interaction
    }

