"""
Intent Classification System
============================

Semantic task classification using LLM-based understanding.
Replaces naive keyword matching with proper intent detection.

This is the foundation for routing tasks to the right execution path.
"""

import re
import logging
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class TaskIntent(Enum):
    """Task intent categories."""

    FACT_RETRIEVAL = "fact_retrieval"      # Direct Q&A (GAIA tasks)
    CODE_GENERATION = "code_generation"    # Write/debug code
    CONTENT_CREATION = "content_creation"  # Write articles/reports
    DATA_ANALYSIS = "data_analysis"        # Analyze datasets
    RESEARCH = "research"                  # Multi-source research
    CONVERSATION = "conversation"          # Chat/discuss
    TASK_EXECUTION = "task_execution"      # Execute specific actions


@dataclass
class IntentAnalysis:
    """Result of intent classification."""

    intent: TaskIntent
    confidence: float
    required_tools: List[str]
    is_multi_step: bool
    reasoning: str


class IntentClassifier:
    """Classify task intent using LLM-based semantic understanding."""

    def __init__(self, lm=None):
        """
        Initialize classifier.

        Args:
            lm: Language model for classification (optional, lazy-loaded)
        """
        self._lm = lm
        self._cache: Dict[str, IntentAnalysis] = {}

    @property
    def lm(self):
        """Lazy-load language model."""
        if self._lm is None:
            # Import here to avoid circular dependency
            from Jotty.core.foundation.unified_lm_provider import get_lm
            self._lm = get_lm(model='gpt-4o-mini', temperature=0.0)
        return self._lm

    def classify(
        self,
        task: str,
        attachments: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> IntentAnalysis:
        """
        Classify task intent using semantic understanding.

        Args:
            task: Task description or question
            attachments: List of attachment filenames (if any)
            context: Additional context

        Returns:
            IntentAnalysis with intent, tools, and reasoning
        """
        # Check cache
        cache_key = f"{task}:{attachments}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Build classification prompt
        prompt = self._build_classification_prompt(task, attachments, context)

        # Get LLM classification
        try:
            response = self.lm(prompt)
            analysis = self._parse_classification(response, task, attachments)

            # Cache result
            self._cache[cache_key] = analysis

            logger.info(f"Intent classified: {analysis.intent.value} (confidence: {analysis.confidence:.2f})")
            return analysis

        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            # Fallback to heuristic classification
            return self._heuristic_fallback(task, attachments)

    def _build_classification_prompt(
        self,
        task: str,
        attachments: Optional[List[str]],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Build classification prompt for LLM."""

        prompt = f"""Classify the intent of this task. Analyze what the user wants to accomplish.

Task: {task}
"""

        if attachments:
            prompt += f"\nAttachments: {', '.join(attachments)}"

        if context:
            prompt += f"\nContext: {context}"

        prompt += """

Choose ONE intent category:

1. FACT_RETRIEVAL
   - Answering specific questions (who/what/when/where/why)
   - Looking up facts or information
   - Calculations or conversions
   - Examples: "What is the capital of France?", "Calculate 234 * 567"

2. CODE_GENERATION
   - Writing, debugging, or reviewing code
   - Programming tasks
   - Technical implementation
   - Examples: "Write a Python function to...", "Fix this bug"

3. CONTENT_CREATION
   - Writing articles, reports, documents
   - Creating presentations or content
   - Drafting emails or messages
   - Examples: "Write a blog post about...", "Draft a report on..."

4. DATA_ANALYSIS
   - Analyzing datasets or numbers
   - Creating visualizations or charts
   - Statistical analysis
   - Examples: "Analyze this CSV", "Create a chart showing..."

5. RESEARCH
   - Multi-source research projects
   - Literature reviews
   - Comprehensive investigations
   - Examples: "Research AI trends for 2024", "Compare X and Y"

6. CONVERSATION
   - General chat or discussion
   - Clarification or explanation
   - Open-ended conversation
   - Examples: "Tell me about...", "What do you think..."

7. TASK_EXECUTION
   - Specific actions to perform
   - Workflow execution
   - Multi-step processes
   - Examples: "Send an email to...", "Create a presentation and..."

Provide your answer in this EXACT format:

INTENT: [category name]
CONFIDENCE: [0.0-1.0]
MULTI_STEP: [yes/no]
REQUIRED_TOOLS: [comma-separated tool names, or "none"]
REASONING: [one sentence explanation]

Think carefully about the core intent, not just keywords."""

        return prompt

    def _parse_classification(
        self,
        response: str,
        task: str,
        attachments: Optional[List[str]]
    ) -> IntentAnalysis:
        """Parse LLM classification response."""

        lines = response.strip().split('\n')
        data = {}

        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                data[key.strip().lower()] = value.strip()

        # Extract fields
        intent_str = data.get('intent', 'conversation').lower()
        confidence = float(data.get('confidence', '0.8'))
        multi_step = data.get('multi_step', 'no').lower() == 'yes'
        tools_str = data.get('required_tools', 'none')
        reasoning = data.get('reasoning', 'Based on task analysis')

        # Map intent string to enum
        intent_map = {
            'fact_retrieval': TaskIntent.FACT_RETRIEVAL,
            'code_generation': TaskIntent.CODE_GENERATION,
            'content_creation': TaskIntent.CONTENT_CREATION,
            'data_analysis': TaskIntent.DATA_ANALYSIS,
            'research': TaskIntent.RESEARCH,
            'conversation': TaskIntent.CONVERSATION,
            'task_execution': TaskIntent.TASK_EXECUTION,
        }

        intent = intent_map.get(intent_str, TaskIntent.CONVERSATION)

        # Parse tools
        if tools_str.lower() == 'none':
            required_tools = []
        else:
            required_tools = [t.strip() for t in tools_str.split(',')]

        # Auto-detect additional tools based on task and attachments
        auto_tools = self._auto_detect_tools(task, attachments, intent)
        required_tools.extend([t for t in auto_tools if t not in required_tools])

        return IntentAnalysis(
            intent=intent,
            confidence=confidence,
            required_tools=required_tools,
            is_multi_step=multi_step,
            reasoning=reasoning
        )

    def _auto_detect_tools(
        self,
        task: str,
        attachments: Optional[List[str]],
        intent: TaskIntent
    ) -> List[str]:
        """Auto-detect required tools based on patterns."""

        tools = []
        task_lower = task.lower()

        # Calculator patterns
        if any(op in task for op in ['+', '-', '*', '/', '=']) or \
           any(word in task_lower for word in ['calculate', 'compute', 'sum', 'multiply', 'divide']):
            tools.append('calculator')

        # Web search patterns
        if any(word in task_lower for word in ['search', 'find', 'look up', 'what is', 'who is', 'when was']):
            tools.append('web-search')

        # Attachment-based tools
        if attachments:
            for filename in attachments:
                ext = filename.split('.')[-1].lower()

                if ext in ['mp3', 'wav', 'm4a', 'ogg']:
                    tools.append('whisper')
                elif ext in ['pdf', 'docx', 'txt']:
                    tools.append('document-reader')
                elif ext in ['jpg', 'jpeg', 'png', 'gif']:
                    tools.append('vision')
                elif ext in ['csv', 'xlsx', 'json']:
                    tools.append('data-analysis')

        # Intent-specific tools
        if intent == TaskIntent.DATA_ANALYSIS:
            if 'data-analysis' not in tools:
                tools.append('data-analysis')

        if intent == TaskIntent.CODE_GENERATION:
            tools.append('code-interpreter')

        return tools

    def _heuristic_fallback(
        self,
        task: str,
        attachments: Optional[List[str]]
    ) -> IntentAnalysis:
        """Fallback to heuristic classification if LLM fails."""

        task_lower = task.lower()

        # Check for question patterns (FACT_RETRIEVAL)
        question_patterns = [
            r'^(what|who|when|where|why|how)\s',
            r'\?$',
            r'(calculate|compute|find|determine|identify)',
        ]

        if any(re.search(p, task_lower) for p in question_patterns):
            intent = TaskIntent.FACT_RETRIEVAL
        elif any(word in task_lower for word in ['code', 'function', 'program', 'script', 'debug']):
            intent = TaskIntent.CODE_GENERATION
        elif any(word in task_lower for word in ['write', 'draft', 'create', 'compose']):
            intent = TaskIntent.CONTENT_CREATION
        elif any(word in task_lower for word in ['analyze', 'data', 'chart', 'visualization']):
            intent = TaskIntent.DATA_ANALYSIS
        elif any(word in task_lower for word in ['research', 'investigate', 'study', 'compare']):
            intent = TaskIntent.RESEARCH
        else:
            intent = TaskIntent.CONVERSATION

        tools = self._auto_detect_tools(task, attachments, intent)

        return IntentAnalysis(
            intent=intent,
            confidence=0.6,  # Lower confidence for heuristic
            required_tools=tools,
            is_multi_step=False,
            reasoning="Heuristic classification (LLM failed)"
        )

    def clear_cache(self) -> None:
        """Clear classification cache."""
        self._cache.clear()


# Singleton instance
_classifier_instance: Optional[IntentClassifier] = None


def get_intent_classifier() -> IntentClassifier:
    """Get or create singleton classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = IntentClassifier()
    return _classifier_instance


def classify_task_intent(
    task: str,
    attachments: Optional[List[str]] = None,
    context: Optional[Dict[str, Any]] = None
) -> IntentAnalysis:
    """
    Convenience function to classify task intent.

    Args:
        task: Task description
        attachments: Optional attachment filenames
        context: Optional additional context

    Returns:
        IntentAnalysis with classification results
    """
    classifier = get_intent_classifier()
    return classifier.classify(task, attachments, context)
