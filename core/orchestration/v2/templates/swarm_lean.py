"""
SwarmLean - Claude Code-like Lean Execution Template
=====================================================

A simple, efficient template for general-purpose tasks.
Inspired by Claude Code's clean, linear execution model.

Philosophy:
- KISS: Keep It Simple, Stupid
- One agent, linear pipeline
- No ensemble, no multi-perspective
- Fast, direct execution
- Clean progress reporting

Use Cases:
- Document creation (checklists, reports, summaries)
- Research tasks (web search, content aggregation)
- File operations (read, write, convert)
- Simple analysis tasks

This template avoids the over-engineering that can hurt simple tasks:
- NO multi-agent decomposition
- NO prompt ensembling
- NO transfer learning context injection
- NO Q-learning pollution

Pipeline:
1. UNDERSTAND - Parse and understand the task
2. RESEARCH - Gather information (search, read files)
3. EXECUTE - Perform the main task
4. OUTPUT - Generate final output

Usage:
    from jotty import Swarm

    result = await Swarm.solve(
        template="lean",
        task="Create a checklist for BaFin KGAB compliance"
    )
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import re

from .base import (
    SwarmTemplate, AgentConfig, StageConfig, FeedbackConfig, ModelTier
)


class SwarmLean(SwarmTemplate):
    """
    Lean execution template - Claude Code style.

    Single agent, linear pipeline, no over-engineering.
    Designed for simple to moderate complexity tasks that
    don't require multi-agent collaboration.

    Characteristics:
    - Single generalist agent
    - Sequential stages (no parallelism overhead)
    - No ensemble/perspectives
    - Direct skill execution
    - Clean, predictable output
    """

    name = "SwarmLean"
    version = "1.0.0"
    description = "Lean execution: simple tasks done well, Claude Code style"

    supported_problem_types = [
        "research",
        "document",
        "checklist",
        "summary",
        "analysis",
        "file_operation",
        "search",
    ]

    # ================================================================
    # SINGLE AGENT CONFIGURATION
    # ================================================================
    agents = {
        "executor": AgentConfig(
            name="executor",
            skills=[
                # Core research skills
                "web-search",
                "claude-cli-llm",
                "content-research-writer",

                # Document creation
                "docx-document-generator",
                "markdown-writer",

                # File operations
                "file-read",
                "file-write",
                "file-edit",

                # Utilities
                "shell-exec",
            ],
            model=ModelTier.BALANCED,  # Sonnet - good balance
            max_concurrent=1,  # Sequential for simplicity
            timeout=300,
            retry_count=2,
        ),
    }

    # ================================================================
    # LINEAR PIPELINE (4 stages, like Claude Code)
    # ================================================================
    pipeline = [
        # Stage 1: Understand the task
        StageConfig(
            name="UNDERSTAND",
            agents=["executor"],
            parallel=False,
            inputs=["task", "context"],
            outputs=["parsed_task", "task_type", "required_skills"],
            weight=5,
            description="Parse task and identify requirements",
        ),

        # Stage 2: Research/Gather information
        StageConfig(
            name="RESEARCH",
            agents=["executor"],
            parallel=False,
            inputs=["parsed_task", "task_type"],
            outputs=["research_results", "sources"],
            weight=25,
            description="Search and gather relevant information",
            skip_on_failure=False,  # Research is optional for some tasks
        ),

        # Stage 3: Execute main task
        StageConfig(
            name="EXECUTE",
            agents=["executor"],
            parallel=False,
            inputs=["parsed_task", "research_results", "sources"],
            outputs=["execution_result", "artifacts"],
            weight=50,
            description="Execute the main task",
        ),

        # Stage 4: Generate output
        StageConfig(
            name="OUTPUT",
            agents=["executor"],
            parallel=False,
            inputs=["execution_result", "artifacts", "task_type"],
            outputs=["final_output", "output_path"],
            weight=20,
            description="Generate and save final output",
        ),
    ]

    # No feedback loop - keep it simple
    feedback_config = None

    # ================================================================
    # SIMPLE, DIRECT PROMPTS
    # ================================================================
    llm_prompts = {
        "task_parser": """Analyze this task and extract key requirements:

Task: {task}

Identify:
1. Task type (research, document, checklist, summary, analysis, file_operation)
2. Main objective (one sentence)
3. Required information sources (web search needed? files to read?)
4. Expected output format (document, list, summary, file)

Be concise. Output JSON:
{{
    "task_type": "...",
    "objective": "...",
    "needs_search": true/false,
    "needs_file_read": true/false,
    "output_format": "...",
    "key_terms": ["...", "..."]
}}""",

        "search_query": """Create a focused search query for:

Task: {task}

Return ONLY the search query, nothing else. Keep it under 10 words.
Focus on the core topic, exclude meta-instructions.""",

        "synthesize": """Based on the research results below, create a comprehensive response.

Task: {task}

Research Results:
{research}

Create a well-structured response that directly addresses the task.
Be thorough but concise. Use bullet points and sections where appropriate.""",

        "document_create": """Create a professional document based on:

Task: {task}
Content: {content}

Format the content appropriately for the requested output type.
Use clear headings, bullet points, and professional formatting.""",
    }

    def __init__(self):
        """Initialize SwarmLean template."""
        super().__init__()
        self._task_type = None

    def detect_task_type(self, task: str) -> str:
        """
        Detect the type of task from the description.

        Returns:
            One of: research, document, checklist, summary, analysis, file_operation
        """
        task_lower = task.lower()

        # Checklist patterns
        if any(kw in task_lower for kw in ['checklist', 'todo', 'list of', 'steps to']):
            return "checklist"

        # Document patterns
        if any(kw in task_lower for kw in ['document', 'docx', 'report', 'write a']):
            return "document"

        # Summary patterns
        if any(kw in task_lower for kw in ['summarize', 'summary', 'overview', 'brief']):
            return "summary"

        # Research patterns
        if any(kw in task_lower for kw in ['research', 'find', 'search', 'look up', 'what is']):
            return "research"

        # Analysis patterns
        if any(kw in task_lower for kw in ['analyze', 'analysis', 'compare', 'evaluate']):
            return "analysis"

        # File operation patterns
        if any(kw in task_lower for kw in ['read file', 'write file', 'edit file', 'create file']):
            return "file_operation"

        # Default to research (needs information gathering)
        return "research"

    def needs_web_search(self, task: str) -> bool:
        """Determine if the task needs web search."""
        task_lower = task.lower()

        # Explicit search indicators
        if any(kw in task_lower for kw in ['search', 'find', 'look up', 'research', 'latest', 'current']):
            return True

        # Domain-specific terms often need search
        if any(kw in task_lower for kw in ['framework', 'regulation', 'compliance', 'bafin', 'sec', 'law']):
            return True

        # File operations don't need search
        if self.detect_task_type(task) == "file_operation":
            return False

        return True

    def get_output_format(self, task: str) -> str:
        """Determine expected output format."""
        task_lower = task.lower()

        if 'docx' in task_lower or 'word' in task_lower:
            return "docx"
        if 'pdf' in task_lower:
            return "pdf"
        if 'markdown' in task_lower or '.md' in task_lower:
            return "markdown"
        if self.detect_task_type(task) == "checklist":
            return "docx"  # Checklists are nice as docx
        if self.detect_task_type(task) == "document":
            return "docx"

        return "text"

    def validate_inputs(self, **kwargs) -> bool:
        """Validate that required inputs are provided."""
        task = kwargs.get('task')
        return task is not None and len(str(task).strip()) > 0

    def clean_task_for_execution(self, task: str) -> str:
        """
        Clean task string for execution.

        Removes any polluting context like Q-learning lessons,
        transfer learning suggestions, etc.
        """
        # Markers that indicate enrichment context (to be stripped)
        context_markers = [
            '\n[Multi-Perspective Analysis',
            '\nLearned Insights:',
            '\n# Transferable Learnings',
            '\n# Q-Learning Lessons',
            '\n## Task Type Pattern',
            '\n## Role Advice',
            '\n## Meta-Learning Advice',
            '\n\n---\n',  # Common separator before context
        ]

        cleaned = task
        for marker in context_markers:
            if marker in cleaned:
                cleaned = cleaned.split(marker)[0]

        return cleaned.strip()

    def get_search_query(self, task: str) -> str:
        """
        Extract clean search query from task.

        This is CRITICAL to avoid query pollution.
        """
        # First clean the task
        clean_task = self.clean_task_for_execution(task)

        # Remove common prefixes
        prefixes_to_strip = [
            'create a ', 'create ', 'make a ', 'make ',
            'help me ', 'please ', 'can you ',
            'i need ', 'i want ', 'generate a ', 'generate ',
            'write a ', 'write ', 'draft a ', 'draft ',
        ]

        query = clean_task.lower()
        for prefix in prefixes_to_strip:
            if query.startswith(prefix):
                query = query[len(prefix):]

        # Extract key topic (usually first clause)
        if ' for ' in query:
            # "checklist for oversight over SPVs" -> "oversight over SPVs"
            parts = query.split(' for ', 1)
            query = parts[1] if len(parts) > 1 else parts[0]

        # Limit query length
        words = query.split()
        if len(words) > 15:
            query = ' '.join(words[:15])

        return query.strip()

    def get_skills_for_task(self, task_type: str) -> List[str]:
        """Get the relevant skills for a task type."""
        skill_map = {
            "research": ["web-search", "claude-cli-llm"],
            "document": ["claude-cli-llm", "docx-document-generator"],
            "checklist": ["web-search", "claude-cli-llm", "docx-document-generator"],
            "summary": ["claude-cli-llm", "content-research-writer"],
            "analysis": ["web-search", "claude-cli-llm"],
            "file_operation": ["file-read", "file-write", "file-edit"],
        }
        return skill_map.get(task_type, ["claude-cli-llm"])


# Alias for convenience
Lean = SwarmLean
