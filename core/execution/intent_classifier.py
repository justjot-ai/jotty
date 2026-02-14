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
            # Use the already-configured DSPy LM (from benchmark setup)
            # This avoids interfering with existing DSPy configuration
            import dspy
            if dspy.settings.lm is not None:
                self._lm = dspy.settings.lm
            else:
                # Fallback: create new LM if DSPy not configured
                from Jotty.core.foundation.unified_lm_provider import UnifiedLMProvider
                self._lm = UnifiedLMProvider.create_lm('anthropic', model='claude-sonnet-4-5-20250929', temperature=0.0)
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

            # Extract text from DSPy response (can be list, dict, or string)
            if isinstance(response, list):
                response_text = response[0] if response else ""
            elif isinstance(response, dict):
                response_text = response.get('content', '') or response.get('text', '') or str(response)
            else:
                response_text = str(response)

            analysis = self._parse_classification(response_text, task, attachments)

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


# =============================================================================
# TaskClassifier — Smart Swarm Routing
# =============================================================================

CONFIDENCE_THRESHOLD = 0.4

# Intent → swarm mapping (signal 1)
_INTENT_SWARM_MAP: Dict[TaskIntent, Optional[str]] = {
    TaskIntent.CODE_GENERATION: "coding",
    TaskIntent.DATA_ANALYSIS: "data_analysis",
    TaskIntent.RESEARCH: "research",
    TaskIntent.CONTENT_CREATION: "idea_writer",
    TaskIntent.TASK_EXECUTION: None,
    TaskIntent.FACT_RETRIEVAL: None,
    TaskIntent.CONVERSATION: None,
}

# Domain keyword map (signal 2) — covers all 11 swarms.
# Multi-word phrases are matched exactly; single words use suffix tolerance
# (see _keyword_match). More specific swarms (testing, review, devops, etc.)
# should have enough keywords to outweigh generic swarms on relevant goals.
_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "coding": [
        "code", "program", "implement", "develop", "function", "class",
        "api", "debug", "refactor", "compile", "syntax",
        "algorithm", "software", "repository", "git", "backend", "frontend",
        "microservice", "endpoint", "sdk", "library", "websocket",
    ],
    "research": [
        "research", "investigate", "analyze company",
        "stock analysis", "market research", "deep dive",
    ],
    "testing": [
        "test", "coverage", "unit test", "integration test", "qa",
        "pytest", "regression", "benchmark", "assert", "load test",
        "end-to-end test", "e2e test", "test suite", "test case",
        "cypress", "selenium", "locust",
    ],
    "review": [
        "review", "audit", "check code", "pull request", "code review",
        "peer review", "pr review", "code quality", "vulnerability",
        "vulnerabilities", "security audit", "owasp",
    ],
    "data_analysis": [
        "dataset", "statistics", "visualization", "csv",
        "chart", "graph", "analytics", "dashboard", "pandas", "dataframe",
        "spreadsheet", "excel", "plot", "histogram", "correlation",
        "outlier", "trend", "metric", "a/b test",
    ],
    "devops": [
        "deploy", "docker", "ci/cd", "infrastructure", "kubernetes",
        "terraform", "ansible", "pipeline", "monitoring", "nginx",
        "server", "cloud", "aws", "gcp", "azure", "elk", "log aggregation",
        "grafana", "prometheus", "provision", "auto-scaling", "cloudformation",
        "helm", "container",
    ],
    "idea_writer": [
        "write article", "blog post", "essay", "content creation",
        "creative writing", "copywriting", "newsletter", "editorial",
        "blog", "article", "whitepaper", "case study", "draft article",
        "draft report", "write copy",
    ],
    "fundamental": [
        "stock", "valuation", "financial", "earnings", "investment",
        "portfolio", "market cap", "dividend", "pe ratio", "fundamental",
        "balance sheet", "income statement", "cash flow", "price target",
        "intrinsic value", "revenue growth", "profit margin", "earnings report",
    ],
    "learning": [
        "curriculum", "teach", "training material", "study guide",
        "lesson plan", "education", "tutorial", "course", "certification",
        "workshop", "education program", "training program", "training course",
        "study plan", "syllabus",
    ],
    "arxiv_learning": [
        "arxiv", "paper", "academic paper", "research paper",
        "preprint", "journal", "scientific paper", "literature review",
        "scholarly", "citation", "journal article",
    ],
    "olympiad_learning": [
        "olympiad", "competition", "competitive", "math olympiad",
        "science olympiad", "imo", "ioi", "contest", "olympiad problem",
        "competition problem",
    ],
}

# Skill category → swarm mapping (signal 3)
_CATEGORY_SWARM_MAP: Dict[str, str] = {
    "development": "coding",
    "code": "coding",
    "programming": "coding",
    "data": "data_analysis",
    "analysis": "data_analysis",
    "research": "research",
    "content": "idea_writer",
    "writing": "idea_writer",
    "infrastructure": "devops",
    "devops": "devops",
    "testing": "testing",
    "qa": "testing",
}


@dataclass
class TaskClassification:
    """Result of swarm classification."""

    swarm_name: Optional[str]
    confidence: float
    reasoning: str
    intent: TaskIntent


class TaskClassifier:
    """
    Smart swarm router combining intent classification, keyword domain
    matching, and skill category voting to select the best domain swarm.

    Weights are tuned so keyword is dominant (0.55) because it's the only
    signal that distinguishes all 11 swarms. Intent (7 categories) is too
    coarse — e.g. CODE_GENERATION absorbs testing/review/devops tasks.
    """

    INTENT_WEIGHT = 0.30
    KEYWORD_WEIGHT = 0.55
    SKILL_WEIGHT = 0.15

    def __init__(self):
        self._intent_classifier = get_intent_classifier()

    def classify_swarm(self, goal: str) -> TaskClassification:
        """
        Classify which swarm should handle a goal.

        Combines three signals:
        1. IntentClassifier → swarm mapping (weight=0.30)
        2. Domain keyword matching (weight=0.55)
        3. Skill category voting (weight=0.15)

        Returns TaskClassification with swarm_name=None if confidence
        is below CONFIDENCE_THRESHOLD (fallback to Orchestrator auto-swarm).
        """
        # Signal 1: Intent classification
        intent_analysis = self._intent_classifier.classify(goal)

        # Guard: CONVERSATION intent means no task to route
        if intent_analysis.intent == TaskIntent.CONVERSATION:
            return TaskClassification(
                swarm_name=None,
                confidence=0.0,
                reasoning=f"Conversational (intent={intent_analysis.intent.value})",
                intent=intent_analysis.intent,
            )

        intent_swarm = _INTENT_SWARM_MAP.get(intent_analysis.intent)

        # Signal 2: Domain keyword matching
        keyword_swarm = self._keyword_match(goal)

        # Signal 3: Skill category voting
        skill_swarm = self._skill_category_vote(goal)

        # Tally weighted votes
        votes: Dict[str, float] = {}
        signals_used = []

        if intent_swarm:
            votes[intent_swarm] = votes.get(intent_swarm, 0.0) + self.INTENT_WEIGHT
            signals_used.append(f"intent={intent_analysis.intent.value}")

        if keyword_swarm:
            votes[keyword_swarm] = votes.get(keyword_swarm, 0.0) + self.KEYWORD_WEIGHT
            signals_used.append(f"keyword={keyword_swarm}")

        if skill_swarm:
            votes[skill_swarm] = votes.get(skill_swarm, 0.0) + self.SKILL_WEIGHT
            signals_used.append(f"skill={skill_swarm}")

        if not votes:
            return TaskClassification(
                swarm_name=None,
                confidence=0.0,
                reasoning=f"No signal matched (intent={intent_analysis.intent.value})",
                intent=intent_analysis.intent,
            )

        # Winner = highest total weight
        winner = max(votes, key=votes.get)
        total_weight_cast = sum(votes.values())
        confidence = votes[winner] / total_weight_cast if total_weight_cast > 0 else 0.0

        reasoning = f"signals=[{', '.join(signals_used)}]"

        if confidence < CONFIDENCE_THRESHOLD:
            return TaskClassification(
                swarm_name=None,
                confidence=confidence,
                reasoning=f"Below threshold ({confidence:.2f}<{CONFIDENCE_THRESHOLD}): {reasoning}",
                intent=intent_analysis.intent,
            )

        return TaskClassification(
            swarm_name=winner,
            confidence=confidence,
            reasoning=reasoning,
            intent=intent_analysis.intent,
        )

    # Common English suffixes for fuzzy keyword matching
    _SUFFIX_PATTERN = r'(?:s|ing|ed|er|ment|tion|ize|ise|ly|ity|ness|ous|ive|al|able|ible)?'

    def _keyword_match(self, goal: str) -> Optional[str]:
        """Match goal against domain keywords using suffix-tolerant regex.

        Single-word keywords allow common English suffixes so that
        'test' matches 'testing', 'learn' matches 'learning', etc.
        Multi-word keywords require the exact phrase (with optional suffix
        on the last word).
        """
        goal_lower = goal.lower()
        best_swarm = None
        best_count = 0

        for swarm_name, keywords in _DOMAIN_KEYWORDS.items():
            count = 0
            for kw in keywords:
                pattern = r'\b' + re.escape(kw) + self._SUFFIX_PATTERN + r'\b'
                if re.search(pattern, goal_lower):
                    # Multi-word phrases are stronger signals (more specific)
                    count += 2 if ' ' in kw else 1
            if count > best_count:
                best_count = count
                best_swarm = swarm_name

        return best_swarm

    def _skill_category_vote(self, goal: str) -> Optional[str]:
        """Use skill discovery to vote for a swarm based on top-5 skill categories."""
        try:
            from Jotty.core.registry.skills_registry import get_skills_registry
            registry = get_skills_registry()
            if not registry.loaded_skills:
                registry.init()

            discovered = registry.discover(goal, max_results=5)
            if not discovered:
                return None

            # Tally category votes (only scored skills)
            category_counts: Dict[str, int] = {}
            for skill_dict in discovered:
                if skill_dict.get('relevance_score', 0) <= 0:
                    continue
                cat = (skill_dict.get('category') or '').lower()
                mapped = _CATEGORY_SWARM_MAP.get(cat)
                if mapped:
                    category_counts[mapped] = category_counts.get(mapped, 0) + 1

            if not category_counts:
                return None

            return max(category_counts, key=category_counts.get)

        except Exception as e:
            logger.debug(f"Skill category voting failed: {e}")
            return None


# Singleton instance
_task_classifier_instance: Optional[TaskClassifier] = None


def get_task_classifier() -> TaskClassifier:
    """Get or create singleton TaskClassifier instance."""
    global _task_classifier_instance
    if _task_classifier_instance is None:
        _task_classifier_instance = TaskClassifier()
    return _task_classifier_instance
