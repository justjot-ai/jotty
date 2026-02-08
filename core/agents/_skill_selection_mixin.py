"""AgenticPlanner Skill Selection Mixin - Skill discovery and matching."""

import json
import logging
import re
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class SkillSelectionMixin:
    def select_skills(
        self,
        task: str,
        available_skills: List[Dict[str, Any]],
        max_skills: int = 8,
        use_capability_filter: bool = True
    ) -> tuple[List[Dict[str, Any]], str]:
        """
        Select best skills for task using capability filtering + LLM semantic matching.

        Flow:
        1. Infer required capabilities from task (fast LLM call)
        2. Filter skills by capabilities (126 → ~10-20)
        3. LLM selects best from filtered set

        Falls back to using first available skills if LLM fails.
        Deprioritizes skills that depend on unreliable external services.

        Args:
            task: Task description
            available_skills: List of available skills
            max_skills: Maximum skills to select
            use_capability_filter: Whether to filter by capabilities first (default: True)

        Returns:
            (selected_skills, reasoning)
        """
        if not available_skills:
            return [], "No skills available"

        original_count = len(available_skills)

        # Step 1: Filter by capabilities (if enabled)
        if use_capability_filter and len(available_skills) > 15:
            try:
                capabilities, cap_reasoning = self.infer_capabilities(task)
                logger.info(f"🎯 Inferred capabilities: {capabilities}")

                # Filter skills by capabilities
                capability_filtered = [
                    s for s in available_skills
                    if self._skill_matches_capabilities(s, capabilities)
                ]

                if capability_filtered:
                    logger.info(f"📉 Capability filter: {original_count} → {len(capability_filtered)} skills")
                    available_skills = capability_filtered
                else:
                    logger.debug("Capability filter returned 0 skills, using all")
            except Exception as e:
                logger.debug(f"Capability filtering failed: {e}, using all skills")

        # Step 2: Filter out deprioritized skills (move to end of list)
        reliable_skills = [s for s in available_skills if s.get('name') not in self.DEPRIORITIZED_SKILLS]
        deprioritized = [s for s in available_skills if s.get('name') in self.DEPRIORITIZED_SKILLS]
        available_skills = reliable_skills + deprioritized  # Reliable first

        if deprioritized:
            logger.debug(f"Deprioritized {len(deprioritized)} unreliable skills: {[s.get('name') for s in deprioritized]}")

        llm_selected_names = []
        llm_reasoning = ""

        # Try LLM selection
        try:
            # Format skills with type info for LLM
            formatted_skills = []
            for s in available_skills[:50]:
                skill_dict = {
                    'name': s.get('name', ''),
                    'description': s.get('description', ''),
                    'tools': s.get('tools', []),
                    'skill_type': s.get('skill_type', 'base'),
                }
                # Add dependency info for derived/composite skills
                base_skills = s.get('base_skills', [])
                if base_skills:
                    skill_dict['base_skills'] = base_skills
                if s.get('skill_type') == 'composite':
                    skill_dict['hint'] = f"Pre-built workflow combining: {', '.join(base_skills)}"
                    if s.get('execution_mode'):
                        skill_dict['execution_mode'] = s.get('execution_mode')
                elif s.get('skill_type') == 'derived':
                    skill_dict['hint'] = f"Specialized version of: {', '.join(base_skills)}"
                if s.get('use_when'):
                    skill_dict['use_when'] = s.get('use_when')
                formatted_skills.append(skill_dict)

            skills_json = json.dumps(formatted_skills, indent=2)

            import dspy

            # Call skill selector with retry and context compression
            # Use fast LM (Haiku) for classification - much faster than Sonnet
            selector_kwargs = {
                'task_description': task,
                'available_skills': skills_json,
                'max_skills': max_skills
            }

            result = self._call_with_retry(
                module=self.skill_selector,
                kwargs=selector_kwargs,
                compressible_fields=['available_skills'],
                max_retries=self._max_compression_retries,
                lm=self._fast_lm  # Use fast model for skill selection
            )
            logger.debug(f"Skill selection using fast model: {self._fast_model}")

            # Parse selected skills
            try:
                selected_skills_str = str(result.selected_skills).strip()
                if selected_skills_str.startswith('```'):
                    import re
                    json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', selected_skills_str, re.DOTALL)
                    if json_match:
                        selected_skills_str = json_match.group(1).strip()

                llm_selected_names = json.loads(selected_skills_str)
                if not isinstance(llm_selected_names, list):
                    llm_selected_names = [llm_selected_names]
            except (json.JSONDecodeError, ValueError):
                llm_selected_names = self._extract_skill_names_from_text(result.selected_skills)

            llm_reasoning = result.reasoning or "LLM semantic matching"

            # Parse skill priorities for ordering
            try:
                priorities_str = str(result.skill_priorities).strip()
                if priorities_str.startswith('{'):
                    skill_priorities = json.loads(priorities_str)
                else:
                    skill_priorities = {}
            except (json.JSONDecodeError, ValueError):
                skill_priorities = {}

            logger.info(f"LLM selected {len(llm_selected_names)} skills: {llm_selected_names}")
            if skill_priorities:
                logger.info(f"Skill priorities: {skill_priorities}")

        except Exception as e:
            logger.warning(f"LLM selection failed: {e}")
            skill_priorities = {}

        # Build final selection
        if llm_selected_names:
            final_names = list(set(llm_selected_names))[:max_skills]
            reasoning = llm_reasoning
        else:
            # Smart fallback: match skills based on task keywords
            task_lower = task.lower()
            matched_skills = []

            # Priority keyword mappings for common tasks
            # IMPORTANT: claude-cli-llm should be the default for simple Q&A tasks
            keyword_skill_map = {
                ('file', 'create', 'write', 'save'): 'file-operations',
                ('generate', 'llm', 'text', 'content'): 'claude-cli-llm',
                ('search', 'web', 'find', 'lookup'): 'web-search',
                ('terminal', 'shell', 'command', 'run'): 'terminal',
                ('research', 'report'): 'research-to-pdf',
                ('image', 'picture', 'photo'): 'image-generator',
                ('calculate', 'math', 'compute', '+', '-', '*', '/', 'sum', 'add', 'multiply', 'divide'): 'claude-cli-llm',
                ('what', 'how', 'why', 'explain', 'answer', 'tell', 'help'): 'claude-cli-llm',  # Q&A tasks
            }

            for keywords, skill_name in keyword_skill_map.items():
                if any(kw in task_lower for kw in keywords):
                    # Check if this skill is available
                    for s in available_skills:
                        if s.get('name') == skill_name:
                            matched_skills.append(skill_name)
                            break

            if matched_skills:
                final_names = matched_skills[:max_skills]
                reasoning = f"Keyword-matched fallback: {matched_skills}"
                logger.info(f"Keyword fallback matched: {matched_skills}")
            else:
                # Last resort: prefer claude-cli-llm for general Q&A, then first available
                final_names = []
                preferred_fallbacks = ['claude-cli-llm', 'calculator', 'web-search']
                for preferred in preferred_fallbacks:
                    for s in available_skills:
                        if s.get('name') == preferred:
                            final_names = [preferred]
                            reasoning = f"Fallback: using {preferred} for general task"
                            logger.info(f"Fallback to preferred skill: {preferred}")
                            break
                    if final_names:
                        break

                if not final_names:
                    # True last resort: use first available skills
                    final_names = [s.get('name') for s in available_skills[:max_skills]]
                    reasoning = "Fallback: using first available skills"

        # Filter to available skills
        selected_skills = [s for s in available_skills if s.get('name') in final_names]

        if not selected_skills and available_skills:
            selected_skills = available_skills[:max_skills]

        # Order skills by LLM-assigned priorities (no hardcoded flow order)
        def get_skill_order(skill):
            # Use LLM priority (higher priority = earlier execution)
            priority = skill_priorities.get(skill.get('name'), 0.5)
            return -priority  # Negate so higher priority comes first

        selected_skills = sorted(selected_skills, key=get_skill_order)
        logger.info(f"Ordered skills: {[s.get('name') for s in selected_skills]}")

        # Enrich skills with tools from registry
        selected_skills = self._enrich_skills_with_tools(selected_skills)

        selected_skills = selected_skills[:max_skills]

        logger.info(f"Selected {len(selected_skills)} skills: {[s.get('name') for s in selected_skills]}")
        return selected_skills, reasoning

    def _skill_matches_capabilities(
        self,
        skill: Dict[str, Any],
        required_capabilities: List[str]
    ) -> bool:
        """
        Check if a skill matches any of the required capabilities.

        A skill matches if:
        1. It has capabilities defined and at least one matches, OR
        2. It has no capabilities defined (legacy skill - include by default), OR
        3. It's a composite/derived skill with base_skills (likely covers multiple capabilities)

        Args:
            skill: Skill dict with 'capabilities', 'skill_type', 'base_skills' fields
            required_capabilities: List of required capability strings

        Returns:
            True if skill should be included
        """
        if not required_capabilities:
            return True

        required_set = set(c.lower() for c in required_capabilities)

        # Check skill's own capabilities
        skill_caps = skill.get('capabilities', [])
        if skill_caps:
            skill_caps_set = set(c.lower() for c in skill_caps)
            if required_set & skill_caps_set:
                return True

        # Skills without capabilities - include by default but with lower priority
        if not skill_caps:
            return True

        # Composite skills with base_skills likely cover multiple capabilities
        if skill.get('skill_type') == 'composite' and skill.get('base_skills'):
            return True

        return False

    def _enrich_skills_with_tools(self, selected_skills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich skill dicts with tool names and descriptions from registry."""
        try:
            from ..registry.skills_registry import get_skills_registry
            registry = get_skills_registry()
            if not registry:
                return selected_skills

            enriched = []
            for skill_dict in selected_skills:
                skill_name = skill_dict.get('name')
                skill_obj = registry.get_skill(skill_name)
                if skill_obj:
                    enriched_skill = skill_dict.copy()
                    if skill_obj.tools:
                        enriched_skill['tools'] = list(skill_obj.tools.keys())
                    else:
                        enriched_skill['tools'] = []
                    if not enriched_skill.get('description') and skill_obj.description:
                        enriched_skill['description'] = skill_obj.description
                    enriched.append(enriched_skill)
                else:
                    enriched.append(skill_dict)
            return enriched
        except Exception as e:
            logger.warning(f"Could not enrich skills: {e}")
            return selected_skills
    
