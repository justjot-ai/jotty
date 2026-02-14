"""TaskPlanner Skill Selection Mixin - Single LLM call skill selection."""

import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class SkillSelectionMixin:
    """Skill selection via single LLM call with compact skill catalog."""

    # Skills that depend on unreliable external services - deprioritized
    DEPRIORITIZED_SKILLS = {
        'search-to-justjot-idea',
        'notion-research-documentation',
        'reddit-trending-to-justjot',
        'notebooklm-pdf',
        'oauth-automation',
    }

    # Task type → relevant capabilities mapping (heuristic pre-filter)
    _TASK_CAPABILITY_MAP = {
        'research':      {'research', 'data-fetch', 'analyze', 'document'},
        'comparison':    {'research', 'analyze', 'data-fetch'},
        'creation':      {'code', 'document', 'media', 'visualize'},
        'analysis':      {'analyze', 'data-fetch', 'research', 'visualize'},
        'communication': {'communicate', 'document'},
        'automation':    {'code', 'file-ops', 'data-fetch', 'automation', 'workflow-trigger'},
    }

    # Skills that are always relevant regardless of task type (core tooling)
    _ALWAYS_INCLUDE = {
        'claude-cli-llm',      # LLM reasoning — needed for almost everything
        'file-operations',     # File I/O
    }

    def _prefilter_skills(
        self,
        task: str,
        available_skills: List[Dict[str, Any]],
        task_type: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Pre-filter skills by task capabilities before LLM selection.

        Reduces 127 skills → ~30-50 relevant ones. Saves ~60% LLM tokens.
        Falls back to full list if task type is unknown or filter too aggressive.
        """
        # Get relevant capabilities for this task type
        relevant_caps = self._TASK_CAPABILITY_MAP.get(task_type, set())
        if not relevant_caps:
            return available_skills  # Unknown task type: send all

        # Extract task keywords for name-based matching (catches n8n, telegram, etc.)
        task_words = {w.lower() for w in task.split() if len(w) > 2}

        filtered = []
        for s in available_skills:
            name = s.get('name', '')
            caps = set(s.get('capabilities', []))

            # Always include core skills
            if name in self._ALWAYS_INCLUDE:
                filtered.append(s)
            # Include if capabilities overlap with relevant caps
            elif caps & relevant_caps:
                filtered.append(s)
            # Include composites that reference relevant base skills
            elif s.get('skill_type') == 'composite' and s.get('base_skills'):
                # A composite is relevant if any of its base skills would be relevant
                filtered.append(s)
            # Include if any task keyword appears in skill name (direct mention)
            elif any(w in name for w in task_words):
                filtered.append(s)

        # Safety: if filter is too aggressive (<10 skills), fall back to full list
        if len(filtered) < 10:
            return available_skills

        logger.info(f"Pre-filtered skills: {len(available_skills)} → {len(filtered)} (task_type={task_type})")
        return filtered

    def select_skills(
        self,
        task: str,
        available_skills: List[Dict[str, Any]],
        max_skills: int = 8,
        task_type: str = "",
    ) -> tuple[List[Dict[str, Any]], str]:
        """
        Select best skills for a task via single LLM call.

        Pre-filters by task capabilities (127 → ~30-50 skills), then
        sends compact format to LLM for final selection.

        Args:
            task: Task description
            available_skills: List of skill dicts (from registry)
            max_skills: Maximum skills to select
            task_type: Inferred task type for pre-filtering

        Returns:
            (selected_skills, reasoning)
        """
        if not available_skills:
            return [], "No skills available"

        # Pre-filter by task capabilities (saves ~60% LLM tokens)
        relevant_skills = self._prefilter_skills(task, available_skills, task_type)

        # Move deprioritized skills to end (LLM still sees them but they're last)
        reliable = [s for s in relevant_skills if s.get('name') not in self.DEPRIORITIZED_SKILLS]
        deprioritized = [s for s in relevant_skills if s.get('name') in self.DEPRIORITIZED_SKILLS]
        ordered_skills = reliable + deprioritized

        # Format ALL skills compactly for the LLM (~5K tokens for 126 skills)
        formatted = self._format_skills_compact(ordered_skills)
        skills_json = json.dumps(formatted, separators=(',', ':'))

        llm_selected_names = []
        llm_reasoning = ""
        skill_priorities = {}

        try:
            import dspy

            result = self._call_with_retry(
                module=self.skill_selector,
                kwargs={
                    'task_description': task,
                    'available_skills': skills_json,
                    'max_skills': max_skills,
                },
                compressible_fields=['available_skills'],
                max_retries=self._max_compression_retries,
                lm=self._fast_lm,
            )
            logger.debug(f"Skill selection using fast model: {self._fast_model}")

            # Parse selected skills
            llm_selected_names = self._parse_selected_skills(result.selected_skills)
            llm_reasoning = result.reasoning or "LLM selection"

            # Parse skill priorities for ordering
            try:
                priorities_str = str(result.skill_priorities).strip()
                if priorities_str.startswith('{'):
                    skill_priorities = json.loads(priorities_str)
            except (json.JSONDecodeError, ValueError):
                pass

            logger.info(f"LLM selected {len(llm_selected_names)} skills: {llm_selected_names}")

        except Exception as e:
            logger.warning(f"LLM selection failed: {e}")

        # Build final selection
        if llm_selected_names:
            final_names = list(set(llm_selected_names))[:max_skills]
            reasoning = llm_reasoning
        else:
            final_names, reasoning = self._keyword_fallback(task, ordered_skills, max_skills)

        # Resolve to full skill dicts
        name_set = set(final_names)
        selected = [s for s in ordered_skills if s.get('name') in name_set]
        if not selected and ordered_skills:
            selected = ordered_skills[:max_skills]

        # Order by LLM priorities
        selected.sort(key=lambda s: -skill_priorities.get(s.get('name'), 0.5))

        # Enrich with tool names from registry
        selected = self._enrich_skills_with_tools(selected)

        selected = selected[:max_skills]
        logger.info(f"Selected {len(selected)} skills: {[s.get('name') for s in selected]}")
        return selected, reasoning

    async def aselect_skills(
        self,
        task: str,
        available_skills: List[Dict[str, Any]],
        max_skills: int = 8,
        task_type: str = "",
    ) -> tuple[List[Dict[str, Any]], str]:
        """
        Async version of select_skills using DSPy .acall().

        Non-blocking: uses _acall_with_retry (asyncio.sleep for backoff,
        module.acall for LLM calls). No thread pool needed.
        """
        if not available_skills:
            return [], "No skills available"

        # Pre-filter + format (CPU-only, fast)
        relevant_skills = self._prefilter_skills(task, available_skills, task_type)
        reliable = [s for s in relevant_skills if s.get('name') not in self.DEPRIORITIZED_SKILLS]
        deprioritized = [s for s in relevant_skills if s.get('name') in self.DEPRIORITIZED_SKILLS]
        ordered_skills = reliable + deprioritized
        formatted = self._format_skills_compact(ordered_skills)
        skills_json = json.dumps(formatted, separators=(',', ':'))

        llm_selected_names = []
        llm_reasoning = ""
        skill_priorities = {}

        try:
            result = await self._acall_with_retry(
                module=self.skill_selector,
                kwargs={
                    'task_description': task,
                    'available_skills': skills_json,
                    'max_skills': max_skills,
                },
                compressible_fields=['available_skills'],
                max_retries=self._max_compression_retries,
                lm=self._fast_lm,
            )

            llm_selected_names = self._parse_selected_skills(result.selected_skills)
            llm_reasoning = result.reasoning or "LLM selection"

            try:
                priorities_str = str(result.skill_priorities).strip()
                if priorities_str.startswith('{'):
                    skill_priorities = json.loads(priorities_str)
            except (json.JSONDecodeError, ValueError):
                pass

            logger.info(f"LLM selected {len(llm_selected_names)} skills (async): {llm_selected_names}")

        except Exception as e:
            logger.warning(f"Async LLM selection failed: {e}")

        # Resolve + order (CPU-only)
        if llm_selected_names:
            final_names = list(set(llm_selected_names))[:max_skills]
            reasoning = llm_reasoning
        else:
            final_names, reasoning = self._keyword_fallback(task, ordered_skills, max_skills)

        name_set = set(final_names)
        selected = [s for s in ordered_skills if s.get('name') in name_set]
        if not selected and ordered_skills:
            selected = ordered_skills[:max_skills]

        selected.sort(key=lambda s: -skill_priorities.get(s.get('name'), 0.5))
        selected = self._enrich_skills_with_tools(selected)
        selected = selected[:max_skills]
        logger.info(f"Selected {len(selected)} skills (async): {[s.get('name') for s in selected]}")
        return selected, reasoning

    def _format_skills_compact(self, skills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format skills compactly for LLM context (~5K tokens for 126 skills).

        Each skill gets: name, short description, type, capabilities, and
        for composite/derived: base_skills and hint.
        """
        formatted = []
        for s in skills:
            d = {
                'name': s.get('name', ''),
                'description': (s.get('description', '') or '')[:80],
                'type': s.get('skill_type', 'base'),
                'caps': s.get('capabilities', []),
            }

            base_skills = s.get('base_skills', [])
            skill_type = s.get('skill_type', 'base')

            if skill_type == 'composite' and base_skills:
                d['combines'] = base_skills
                if s.get('execution_mode'):
                    d['exec'] = s['execution_mode']
            elif skill_type == 'derived' and base_skills:
                d['extends'] = base_skills[0]

            if s.get('use_when'):
                d['use_when'] = s['use_when'][:60]

            formatted.append(d)
        return formatted

    def _parse_selected_skills(self, raw: Any) -> List[str]:
        """Parse LLM output into list of skill names."""
        import re

        selected_str = str(raw).strip()

        # Strip markdown code blocks
        if selected_str.startswith('```'):
            match = re.search(r'```(?:json)?\s*\n(.*?)\n```', selected_str, re.DOTALL)
            if match:
                selected_str = match.group(1).strip()

        # Try JSON parse
        try:
            names = json.loads(selected_str)
            if isinstance(names, list):
                return [str(n) for n in names]
            return [str(names)]
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: extract quoted strings or skill-name-like patterns
        names = re.findall(r'"([a-z0-9][a-z0-9-]*)"', selected_str)
        if not names:
            names = re.findall(r'\b([a-z][a-z0-9]+-[a-z0-9-]+)\b', selected_str)
        return names

    def _keyword_fallback(
        self, task: str, skills: List[Dict[str, Any]], max_skills: int
    ) -> tuple[List[str], str]:
        """Keyword-based fallback when LLM selection fails."""
        task_lower = task.lower()

        keyword_map = {
            ('file', 'create', 'write', 'save'): 'file-operations',
            ('generate', 'llm', 'text', 'content'): 'claude-cli-llm',
            ('search', 'web', 'find', 'lookup'): 'web-search',
            ('terminal', 'shell', 'command', 'run'): 'terminal',
            ('research', 'report'): 'research-to-pdf',
            ('image', 'picture', 'photo'): 'image-generator',
            ('calculate', 'math', 'compute', 'sum', 'add', 'multiply'): 'claude-cli-llm',
            ('what', 'how', 'why', 'explain', 'answer', 'tell', 'help'): 'claude-cli-llm',
            ('n8n', 'workflow', 'automation'): 'n8n-workflows',
        }

        available_names = {s.get('name') for s in skills}
        matched = []
        for keywords, skill_name in keyword_map.items():
            if skill_name in available_names and any(kw in task_lower for kw in keywords):
                matched.append(skill_name)

        if matched:
            return matched[:max_skills], f"Keyword fallback: {matched}"

        # Last resort
        for preferred in ['claude-cli-llm', 'calculator', 'web-search']:
            if preferred in available_names:
                return [preferred], f"Fallback: {preferred}"

        return [skills[0].get('name')] if skills else [], "Fallback: first available"

    def _enrich_skills_with_tools(self, selected_skills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich skill dicts with tool names from registry."""
        try:
            from ..registry.skills_registry import get_skills_registry
            registry = get_skills_registry()
            if not registry:
                return selected_skills

            enriched = []
            for skill_dict in selected_skills:
                skill_obj = registry.get_skill(skill_dict.get('name'))
                if skill_obj:
                    enriched_skill = skill_dict.copy()
                    enriched_skill['tools'] = list(skill_obj.tools.keys()) if skill_obj.tools else []
                    if not enriched_skill.get('description') and skill_obj.description:
                        enriched_skill['description'] = skill_obj.description
                    enriched.append(enriched_skill)
                else:
                    enriched.append(skill_dict)
            return enriched
        except Exception as e:
            logger.warning(f"Could not enrich skills: {e}")
            return selected_skills
