"""Pilot Swarm - Agent implementations.

6 specialized agents for autonomous goal completion:
- PilotPlannerAgent: decomposes goals into subtask plans (Sonnet)
- PilotSearchAgent: web search + synthesis
- PilotCoderAgent: code/file generation (Sonnet)
- PilotTerminalAgent: safe shell command generation
- PilotSkillWriterAgent: creates new Jotty skills (Sonnet)
- PilotValidatorAgent: validates goal completion
"""

import logging
from typing import Any, Dict


from Jotty.core.intelligence.swarms.olympiad_learning_swarm.agents import BaseOlympiadAgent

from .signatures import (
    CoderSignature,
    PlannerSignature,
    SearchSignature,
    SkillWriterSignature,
    TerminalSignature,
    ValidatorSignature,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PLANNER AGENT
# =============================================================================


class PilotPlannerAgent(BaseOlympiadAgent):
    """Decomposes goals into ordered subtask plans.

    Uses Sonnet for higher quality planning — the plan drives
    all downstream execution.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._ensure_sonnet_lm()
        self._planner = self._create_module(PlannerSignature)

    def _ensure_sonnet_lm(self) -> Any:
        """Create Sonnet LM for planning."""
        try:
            from Jotty.core.infrastructure.foundation.direct_anthropic_lm import (
                DirectAnthropicLM,
                is_api_key_available,
            )

            if is_api_key_available():
                self._lm = DirectAnthropicLM(model="sonnet", max_tokens=4096)
                logger.info("PilotPlanner using Sonnet model")
                return
        except Exception as e:
            logger.debug(f"Sonnet not available for PilotPlanner: {e}")
        self._get_lm()

    async def plan(
        self, goal: str, available_swarms: str = "", context: str = ""
    ) -> Dict[str, Any]:
        """Decompose a goal into an ordered subtask plan."""
        try:
            result = self._call_with_own_lm(
                self._planner,
                goal=goal,
                available_swarms=available_swarms,
                context=context or "First attempt — no previous context.",
            )

            subtasks = self._parse_json_output(str(result.subtasks_json))
            reasoning = str(result.reasoning)

            self._broadcast(
                "plan_created",
                {
                    "goal": goal[:80],
                    "subtask_count": len(subtasks) if isinstance(subtasks, list) else 0,
                },
            )

            return {
                "subtasks": subtasks if isinstance(subtasks, list) else [],
                "reasoning": reasoning,
            }

        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return {"subtasks": [], "reasoning": f"Planning failed: {e}"}


# =============================================================================
# SEARCH AGENT
# =============================================================================


class PilotSearchAgent(BaseOlympiadAgent):
    """Searches for information and synthesizes findings.

    Generates search queries via LLM, optionally executes them
    using the web-search skill, then synthesizes results.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._searcher = self._create_module(SearchSignature)

    async def search(self, task: str, context: str = "") -> Dict[str, Any]:
        """Search for information and synthesize findings."""
        try:
            result = self._call_with_own_lm(
                self._searcher,
                task=task,
                context=context or "No previous context.",
            )

            queries = [q.strip() for q in str(result.search_queries).split("|") if q.strip()]
            findings = [f.strip() for f in str(result.key_findings).split("|") if f.strip()]

            # Try to execute actual web searches
            search_results = await self._execute_web_searches(queries)

            self._broadcast(
                "search_completed",
                {
                    "task": task[:50],
                    "queries": len(queries),
                    "web_results": len(search_results),
                },
            )

            return {
                "queries": queries,
                "synthesis": str(result.synthesis),
                "key_findings": findings,
                "search_results": search_results,
            }

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"queries": [], "synthesis": "", "key_findings": [], "search_results": []}

    async def _execute_web_searches(self, queries: list) -> list:
        """Execute web searches using the web-search skill if available."""
        results = []
        try:
            import importlib.util
            from pathlib import Path

            tools_path = str(
                Path(__file__).parent.parent.parent.parent / "skills" / "web-search" / "tools.py"
            )
            spec = importlib.util.spec_from_file_location("web_search_tools", tools_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            search_web_tool = getattr(mod, "search_web_tool")

            for query in queries[:3]:
                sr = search_web_tool({"query": query, "max_results": 5})
                if isinstance(sr, dict) and sr.get("results"):
                    for r in sr["results"][:3]:
                        if isinstance(r, dict):
                            results.append(
                                {
                                    "title": r.get("title", ""),
                                    "url": r.get("url", r.get("link", "")),
                                    "snippet": r.get("snippet", r.get("description", ""))[:200],
                                }
                            )
        except Exception as e:
            logger.debug(f"Web search skill not available: {e}")
        return results


# =============================================================================
# CODER AGENT
# =============================================================================


class PilotCoderAgent(BaseOlympiadAgent):
    """Generates code and file content.

    Uses Sonnet for higher quality code generation.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._ensure_sonnet_lm()
        self._coder = self._create_module(CoderSignature)

    def _ensure_sonnet_lm(self) -> Any:
        """Create Sonnet LM for code generation."""
        try:
            from Jotty.core.infrastructure.foundation.direct_anthropic_lm import (
                DirectAnthropicLM,
                is_api_key_available,
            )

            if is_api_key_available():
                self._lm = DirectAnthropicLM(model="sonnet", max_tokens=8192)
                logger.info("PilotCoder using Sonnet model")
                return
        except Exception as e:
            logger.debug(f"Sonnet not available for PilotCoder: {e}")
        self._get_lm()

    async def code(self, task: str, context: str = "") -> Dict[str, Any]:
        """Generate code or file content."""
        try:
            result = self._call_with_own_lm(
                self._coder,
                task=task,
                context=context or "No previous context.",
            )

            file_ops = self._parse_json_output(str(result.file_operations_json))

            self._broadcast("code_generated", {"task": task[:50]})

            return {
                "file_operations": file_ops if isinstance(file_ops, list) else [],
                "explanation": str(result.explanation),
            }

        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return {"file_operations": [], "explanation": f"Failed: {e}"}


# =============================================================================
# TERMINAL AGENT
# =============================================================================


class PilotTerminalAgent(BaseOlympiadAgent):
    """Generates shell commands for system tasks.

    Produces commands with safety flags — the swarm decides
    whether to actually execute them.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._terminal = self._create_module(TerminalSignature)

    async def execute(self, task: str, context: str = "") -> Dict[str, Any]:
        """Generate shell commands for a task."""
        try:
            result = self._call_with_own_lm(
                self._terminal,
                task=task,
                context=context or "No previous context. OS: Linux.",
            )

            commands = self._parse_json_output(str(result.commands_json))
            safety = str(result.safety_assessment)

            self._broadcast("terminal_planned", {"task": task[:50]})

            return {
                "commands": commands if isinstance(commands, list) else [],
                "safety_assessment": safety,
            }

        except Exception as e:
            logger.error(f"Terminal planning failed: {e}")
            return {"commands": [], "safety_assessment": f"Failed: {e}"}


# =============================================================================
# SKILL WRITER AGENT
# =============================================================================

SKILL_REFERENCE = '''Example skill.yaml:
name: my-skill
description: "Brief description of what the skill does"
version: 1.0.0
category: utility
tools:
  - my_tool_name
dependencies:
  - requests

Example tools.py:
"""My Skill - does something useful."""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def my_tool_name(params: Dict[str, Any]) -> Dict[str, Any]:
    """Do something useful.

    Args:
        params: Dictionary containing:
            - input (str, required): The input to process
            - option (bool, optional): An option flag

    Returns:
        dict with 'result' key on success, 'error' key on failure
    """
    try:
        value = params.get('input', '')
        if not value:
            return {'error': 'input parameter is required'}
        processed = value.upper()  # example processing
        return {'result': processed}
    except Exception as e:
        return {'error': str(e)}
'''


class PilotSkillWriterAgent(BaseOlympiadAgent):
    """Creates new Jotty skills (skill.yaml + tools.py).

    Uses Sonnet for higher quality skill generation.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._ensure_sonnet_lm()
        self._writer = self._create_module(SkillWriterSignature)

    def _ensure_sonnet_lm(self) -> Any:
        """Create Sonnet LM for skill writing."""
        try:
            from Jotty.core.infrastructure.foundation.direct_anthropic_lm import (
                DirectAnthropicLM,
                is_api_key_available,
            )

            if is_api_key_available():
                self._lm = DirectAnthropicLM(model="sonnet", max_tokens=8192)
                logger.info("PilotSkillWriter using Sonnet model")
                return
        except Exception as e:
            logger.debug(f"Sonnet not available for PilotSkillWriter: {e}")
        self._get_lm()

    async def write_skill(
        self,
        description: str,
        skill_name: str,
        reference_patterns: str = "",
    ) -> Dict[str, Any]:
        """Generate a complete Jotty skill (skill.yaml + tools.py)."""
        try:
            result = self._call_with_own_lm(
                self._writer,
                skill_description=description,
                skill_name=skill_name,
                reference_patterns=reference_patterns or SKILL_REFERENCE,
            )

            self._broadcast("skill_written", {"skill_name": skill_name})

            return {
                "skill_yaml": str(result.skill_yaml),
                "tools_py": str(result.tools_py),
                "usage_example": str(result.usage_example),
            }

        except Exception as e:
            logger.error(f"Skill writing failed: {e}")
            return {"skill_yaml": "", "tools_py": "", "usage_example": ""}


# =============================================================================
# VALIDATOR AGENT
# =============================================================================


class PilotValidatorAgent(BaseOlympiadAgent):
    """Validates whether a goal has been achieved."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._validator = self._create_module(ValidatorSignature)

    async def validate(self, goal: str, results_summary: str) -> Dict[str, Any]:
        """Validate goal completion against results."""
        try:
            result = self._call_with_own_lm(
                self._validator,
                goal=goal,
                results_summary=results_summary,
            )

            success = str(result.success).strip().lower() == "true"
            gaps_raw = str(result.remaining_gaps).strip()
            gaps = [g.strip() for g in gaps_raw.split("|") if g.strip()] if gaps_raw else []

            self._broadcast("validation_completed", {"success": success})

            return {
                "success": success,
                "assessment": str(result.assessment),
                "remaining_gaps": gaps,
            }

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {"success": False, "assessment": f"Validation failed: {e}", "remaining_gaps": []}


__all__ = [
    "PilotPlannerAgent",
    "PilotSearchAgent",
    "PilotCoderAgent",
    "PilotTerminalAgent",
    "PilotSkillWriterAgent",
    "PilotValidatorAgent",
]
