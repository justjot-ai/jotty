"""Olympiad Learning Swarm - Agent implementations.

Each agent is a specialist in one aspect of world-class teaching:
- CurriculumArchitect: Designs the learning roadmap
- ConceptDecomposer: Breaks concepts into building blocks
- IntuitionBuilder: Builds deep understanding through analogy
- PatternHunter: Identifies competition patterns
- ProblemCrafter: Creates progressive problem sets
- SolutionStrategist: Teaches problem-solving strategies
- MistakeAnalyzer: Identifies common pitfalls
- ConnectionMapper: Maps inter-topic connections
- ContentAssembler: Assembles everything into a flowing lesson
- UnifiedTopicAgent: Single-pass deep content generation
"""

import asyncio
import logging
import json
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

import dspy

from Jotty.core.agents.base import DomainAgent, DomainAgentConfig, BaseSwarmAgent
from .types import (
    Subject, DifficultyTier, LessonDepth, TeachingMode,
    BuildingBlock, ConceptCore, PatternEntry, Problem,
    StrategyCard, MistakeEntry, LessonSection, LessonContent,
    format_steps_on_newlines, tier_to_level,
)
from .signatures import (
    CurriculumArchitectSignature, ConceptDecomposerSignature,
    IntuitionBuilderSignature, PatternHunterSignature,
    ProblemCrafterSignature, SolutionStrategistSignature,
    MistakeAnalyzerSignature, ConnectionMapperSignature,
    ContentAssemblerSignature, SingleTopicDeepSignature,
    NarrativeEditorSignature, RankTipsSignature,
)

logger = logging.getLogger(__name__)


# =============================================================================
# BASE AGENT
# =============================================================================

class BaseOlympiadAgent(BaseSwarmAgent):
    """Base class for all olympiad learning agents. Extends BaseSwarmAgent with LLM model selection."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = "",
                 model: str = "haiku", use_fast_predict: bool = True, llm_timeout: int = 90):
        super().__init__(memory=memory, context=context, bus=bus,
                         learned_context=learned_context, signature=None)
        self.model = model
        self.use_fast_predict = use_fast_predict
        self.llm_timeout = llm_timeout
        self._lm = None

    def _get_lm(self):
        """Get or create LLM instance. Tries Direct API first, then CLI fallback."""
        if self._lm is None:
            # If already configured globally, reuse it
            if hasattr(dspy.settings, 'lm') and dspy.settings.lm is not None:
                self._lm = dspy.settings.lm
                return self._lm

            # Try direct Anthropic API (fastest)
            try:
                from Jotty.core.foundation.direct_anthropic_lm import DirectAnthropicLM, is_api_key_available
                if is_api_key_available():
                    self._lm = DirectAnthropicLM(model=self.model, max_tokens=8192)
                    dspy.configure(lm=self._lm)
                    return self._lm
            except Exception as e:
                logger.debug(f"DirectAnthropicLM not available: {e}")

            # Fallback to Claude CLI
            try:
                from Jotty.core.foundation.persistent_claude_lm import PersistentClaudeCLI
                self._lm = PersistentClaudeCLI(model=self.model)
                dspy.configure(lm=self._lm)
            except Exception as e:
                logger.warning(f"Could not init LLM: {e}")
        return self._lm

    def _create_module(self, signature):
        """Create dspy module - Predict (fast) or ChainOfThought (reasoning)."""
        if self._lm is None:
            self._get_lm()
        if self.use_fast_predict:
            return dspy.Predict(signature)
        else:
            return dspy.ChainOfThought(signature)

    def _call_with_own_lm(self, module, **kwargs):
        """Call a DSPy module using this agent's LM via dspy.context().

        Uses dspy.context() (not dspy.configure()) to avoid async task conflicts.
        Safe for agents with per-instance LM overrides (e.g. Sonnet for planner/editor).
        """
        if self._lm is None:
            return module(**kwargs)
        with dspy.context(lm=self._lm):
            return module(**kwargs)

    def _parse_json_output(self, raw: str) -> Any:
        """Safely parse JSON from LLM output, handling markdown fences, newlines, and truncation."""
        text = str(raw).strip()
        # Strip markdown code blocks
        if text.startswith('```'):
            lines = text.split('\n')
            lines = [l for l in lines if not l.strip().startswith('```')]
            text = '\n'.join(lines)

        # Try strict parsing first, then lenient (strict=False allows
        # literal newlines/tabs inside strings, which LLMs commonly produce)
        for strict in (True, False):
            try:
                return json.loads(text, strict=strict)
            except (json.JSONDecodeError, ValueError):
                pass

        # Try to find JSON in the text
        start = text.find('[')
        if start < 0:
            start = text.find('{')
        end = max(text.rfind(']'), text.rfind('}'))
        if start >= 0 and end > start:
            substr = text[start:end + 1]
            for strict in (True, False):
                try:
                    return json.loads(substr, strict=strict)
                except (json.JSONDecodeError, ValueError):
                    pass

        # Try to repair truncated JSON (output token limit hit)
        if start >= 0:
            truncated = text[start:]
            repaired = self._repair_truncated_json(truncated)
            if repaired is not None:
                logger.info(f"Repaired truncated JSON from {self.__class__.__name__}")
                return repaired

        logger.warning(f"Failed to parse JSON from {self.__class__.__name__}")
        return []

    def _repair_truncated_json(self, text: str) -> Any:
        """Attempt to repair truncated JSON array by extracting complete objects.

        When Haiku hits its output token limit, JSON gets cut mid-object.
        Strategy: find each complete top-level object in the array and collect them.
        """
        # Strategy 1: Find complete objects using regex (handles nested braces)
        if text.startswith('['):
            objects = []
            depth = 0
            in_string = False
            escape_next = False
            obj_start = None

            for i, ch in enumerate(text):
                if escape_next:
                    escape_next = False
                    continue
                if ch == '\\' and in_string:
                    escape_next = True
                    continue
                if ch == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue

                if ch == '{':
                    if depth == 0:
                        obj_start = i
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0 and obj_start is not None:
                        obj_str = text[obj_start:i + 1]
                        try:
                            obj = json.loads(obj_str)
                            objects.append(obj)
                        except json.JSONDecodeError:
                            pass
                        obj_start = None

            if objects:
                return objects

        # Strategy 2: Progressive trim (fallback)
        for trim_from in range(len(text) - 1, max(0, len(text) - 500), -1):
            char = text[trim_from]
            if char in ('}', ','):
                candidate = text[:trim_from + 1].rstrip(',').rstrip()
                opens = candidate.count('[') - candidate.count(']')
                braces = candidate.count('{') - candidate.count('}')
                suffix = '}' * max(braces, 0) + ']' * max(opens, 0)
                try:
                    result = json.loads(candidate + suffix)
                    if isinstance(result, list) and len(result) > 0:
                        return result
                except json.JSONDecodeError:
                    continue
        return None


# =============================================================================
# CURRICULUM ARCHITECT AGENT
# =============================================================================

class CurriculumArchitectAgent(BaseOlympiadAgent):
    """Designs the complete learning roadmap from basics to olympiad level.

    Uses Sonnet model for higher quality planning — the planner drives
    all downstream content depth and running example quality.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_sonnet_lm()
        self._architect = self._create_module(CurriculumArchitectSignature)

    def _ensure_sonnet_lm(self):
        """Create Sonnet LM instance for curriculum planning (does NOT set global)."""
        try:
            from Jotty.core.foundation.direct_anthropic_lm import DirectAnthropicLM, is_api_key_available
            if is_api_key_available():
                self._lm = DirectAnthropicLM(model="sonnet", max_tokens=8192)
                logger.info("CurriculumArchitect using Sonnet model")
                return
        except Exception as e:
            logger.debug(f"Sonnet not available for CurriculumArchitect: {e}")
        self._get_lm()

    async def design(
        self,
        subject: str,
        topic: str,
        student_name: str,
        current_level: str = "beginner",
        target_level: str = "olympiad"
    ) -> Dict[str, Any]:
        """Design the curriculum for a topic."""
        try:
            ctx = f"\n\n{self.learned_context}" if self.learned_context else ""
            result = self._call_with_own_lm(
                self._architect,
                subject=subject + ctx,
                topic=topic,
                student_name=student_name,
                current_level=current_level,
                target_level=target_level,
            )

            blocks_data = self._parse_json_output(str(result.building_blocks_json))
            building_blocks = []
            for b in (blocks_data if isinstance(blocks_data, list) else []):
                building_blocks.append(BuildingBlock(
                    name=b.get('name', ''),
                    description=b.get('description', ''),
                    why_needed=b.get('why_needed', ''),
                    quick_review=b.get('quick_review', ''),
                    check_question=b.get('check_question', ''),
                    difficulty=int(b.get('difficulty', 1))
                ))

            sequence = [s.strip() for s in str(result.learning_sequence).split('|') if s.strip()]
            checkpoints = [c.strip() for c in str(result.milestone_checkpoints).split('|') if c.strip()]

            # Extract new planner fields
            running_example_scenario = str(getattr(result, 'running_example_scenario', ''))
            section_depth_plan = str(getattr(result, 'section_depth_plan', ''))

            self._broadcast("curriculum_designed", {'topic': topic, 'blocks': len(building_blocks)})

            return {
                'building_blocks': building_blocks,
                'learning_sequence': sequence,
                'checkpoints': checkpoints,
                'estimated_sessions': str(result.estimated_sessions),
                'running_example_scenario': running_example_scenario,
                'section_depth_plan': section_depth_plan,
            }

        except Exception as e:
            logger.error(f"Curriculum design failed: {e}")
            return {'building_blocks': [], 'learning_sequence': [], 'checkpoints': []}


# =============================================================================
# CONCEPT DECOMPOSER AGENT
# =============================================================================

class ConceptDecomposerAgent(BaseOlympiadAgent):
    """Breaks down concepts into the simplest possible building blocks."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._decomposer = self._create_module(ConceptDecomposerSignature)

    async def decompose(
        self,
        concept_name: str,
        concept_description: str,
        subject: str,
        prerequisites: str,
        student_name: str
    ) -> Dict[str, Any]:
        """Decompose a concept into building blocks."""
        try:
            ctx = f"\n\n{self.learned_context}" if self.learned_context else ""
            result = self._decomposer(
                concept_name=concept_name,
                concept_description=concept_description + ctx,
                subject=subject,
                prerequisites=prerequisites,
                student_name=student_name
            )

            misconceptions = [m.strip() for m in str(result.common_misconceptions).split('|') if m.strip()]

            self._broadcast("concept_decomposed", {'concept': concept_name})

            return {
                'simplest_example': str(result.simplest_example),
                'pattern_discovery': str(result.pattern_discovery),
                'formal_definition': str(result.formal_definition),
                'why_it_works': str(result.why_it_works),
                'key_insight': str(result.key_insight),
                'common_misconceptions': misconceptions
            }

        except Exception as e:
            logger.error(f"Concept decomposition failed: {e}")
            return {}


# =============================================================================
# INTUITION BUILDER AGENT
# =============================================================================

class IntuitionBuilderAgent(BaseOlympiadAgent):
    """Builds deep intuition through real-world analogies and progressive examples."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._builder = self._create_module(IntuitionBuilderSignature)

    async def build(
        self,
        concept: str,
        why_it_matters: str,
        student_name: str,
        audience_level: str,
        subject: str
    ) -> Dict[str, Any]:
        """Build intuition for a concept."""
        try:
            ctx = f"\n\n{self.learned_context}" if self.learned_context else ""
            result = self._builder(
                concept=concept + ctx,
                why_it_matters=why_it_matters,
                student_name=student_name,
                audience_level=audience_level,
                subject=subject
            )

            self._broadcast("intuition_built", {'concept': concept[:50]})

            return {
                'real_world_hook': str(result.real_world_hook),
                'analogy': str(result.analogy),
                'intuition_build': str(result.intuition_build),
                'aha_moment': str(result.aha_moment),
                'visual_description': str(result.visual_description)
            }

        except Exception as e:
            logger.error(f"Intuition building failed: {e}")
            return {}


# =============================================================================
# PATTERN HUNTER AGENT
# =============================================================================

class PatternHunterAgent(BaseOlympiadAgent):
    """Identifies problem-solving patterns that appear across olympiad problems."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._hunter = self._create_module(PatternHunterSignature)

    async def hunt(
        self,
        topic: str,
        subject: str,
        concepts: str,
        target_level: str
    ) -> Dict[str, Any]:
        """Find patterns in a topic."""
        try:
            ctx = f"\n\n{self.learned_context}" if self.learned_context else ""
            result = self._hunter(
                topic=topic + ctx,
                subject=subject,
                concepts=concepts,
                target_level=target_level
            )

            patterns_data = self._parse_json_output(str(result.patterns_json))
            patterns = []
            for p in (patterns_data if isinstance(patterns_data, list) else []):
                patterns.append(PatternEntry(
                    name=p.get('name', ''),
                    description=p.get('description', ''),
                    when_to_use=p.get('when_to_use', ''),
                    example_trigger=p.get('example_trigger', ''),
                    template=p.get('template', '')
                ))

            self._broadcast("patterns_found", {'topic': topic, 'count': len(patterns)})

            return {
                'patterns': patterns,
                'pattern_connections': str(result.pattern_connections),
                'meta_strategy': str(result.meta_strategy)
            }

        except Exception as e:
            logger.error(f"Pattern hunting failed: {e}")
            return {'patterns': [], 'pattern_connections': '', 'meta_strategy': ''}


# =============================================================================
# PROBLEM CRAFTER AGENT
# =============================================================================

class ProblemCrafterAgent(BaseOlympiadAgent):
    """Creates progressive problem sets from foundation to olympiad level."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._crafter = self._create_module(ProblemCrafterSignature)

    async def craft(
        self,
        topic: str,
        subject: str,
        concepts: str,
        patterns: str,
        tier: str,
        count: int,
        student_name: str,
        running_example: str = "",
    ) -> List[Problem]:
        """Craft problems for a specific difficulty tier."""
        try:
            ctx = f"\n\n{self.learned_context}" if self.learned_context else ""
            result = self._crafter(
                topic=topic + ctx,
                subject=subject,
                concepts=concepts,
                patterns=patterns,
                running_example=running_example or f"A scenario involving {topic}",
                tier=tier,
                count=str(count),
                student_name=student_name
            )

            raw_json = str(result.problems_json)
            problems_data = self._parse_json_output(raw_json)
            tier_enum = DifficultyTier(tier) if tier in [t.value for t in DifficultyTier] else DifficultyTier.INTERMEDIATE

            problems = []
            for p in (problems_data if isinstance(problems_data, list) else []):
                hints = p.get('hints', [])
                if isinstance(hints, str):
                    hints = [h.strip() for h in hints.split('|') if h.strip()]
                mistakes = p.get('common_mistakes', [])
                if isinstance(mistakes, str):
                    mistakes = [m.strip() for m in mistakes.split('|') if m.strip()]

                problems.append(Problem(
                    statement=p.get('statement', ''),
                    tier=tier_enum,
                    hints=hints,
                    solution=p.get('solution', ''),
                    strategy_used=p.get('strategy_used', ''),
                    time_estimate_minutes=int(p.get('time_estimate_minutes', 10)),
                    key_insight=p.get('key_insight', ''),
                    common_mistakes=mistakes,
                    relates_to_pattern=p.get('relates_to_pattern', ''),
                ))

            self._broadcast("problems_crafted", {'tier': tier, 'count': len(problems)})
            return problems

        except Exception as e:
            logger.error(f"Problem crafting failed for tier {tier}: {e}")
            return []


# =============================================================================
# SOLUTION STRATEGIST AGENT
# =============================================================================

class SolutionStrategistAgent(BaseOlympiadAgent):
    """Teaches problem-solving strategies specific to competition settings."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._strategist = self._create_module(SolutionStrategistSignature)

    async def strategize(
        self,
        topic: str,
        subject: str,
        concepts: str,
        target_level: str
    ) -> Dict[str, Any]:
        """Generate strategies for a topic."""
        try:
            ctx = f"\n\n{self.learned_context}" if self.learned_context else ""
            result = self._strategist(
                topic=topic + ctx,
                subject=subject,
                concepts=concepts,
                target_level=target_level
            )

            strategies_data = self._parse_json_output(str(result.strategies_json))
            strategies = []
            for s in (strategies_data if isinstance(strategies_data, list) else []):
                steps = s.get('steps', [])
                if isinstance(steps, str):
                    steps = [st.strip() for st in steps.split('|') if st.strip()]
                pitfalls = s.get('pitfalls', [])
                if isinstance(pitfalls, str):
                    pitfalls = [p.strip() for p in pitfalls.split('|') if p.strip()]

                strategies.append(StrategyCard(
                    name=s.get('name', ''),
                    description=s.get('description', ''),
                    when_to_use=s.get('when_to_use', ''),
                    steps=steps,
                    example_problem=s.get('example_problem', ''),
                    example_solution=s.get('example_solution', ''),
                    pitfalls=pitfalls
                ))

            speed = [t.strip() for t in str(result.speed_techniques).split('|') if t.strip()]
            stuck = [t.strip() for t in str(result.stuck_toolkit).split('|') if t.strip()]

            self._broadcast("strategies_created", {'topic': topic, 'count': len(strategies)})

            return {
                'strategies': strategies,
                'speed_techniques': speed,
                'stuck_toolkit': stuck
            }

        except Exception as e:
            logger.error(f"Strategy generation failed: {e}")
            return {'strategies': [], 'speed_techniques': [], 'stuck_toolkit': []}


# =============================================================================
# MISTAKE ANALYZER AGENT
# =============================================================================

class MistakeAnalyzerAgent(BaseOlympiadAgent):
    """Identifies common mistakes and how to avoid them."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._analyzer = self._create_module(MistakeAnalyzerSignature)

    async def analyze(
        self,
        topic: str,
        subject: str,
        concepts: str,
        target_level: str,
        running_example: str = "",
    ) -> Dict[str, Any]:
        """Analyze common mistakes for a topic."""
        try:
            ctx = f"\n\n{self.learned_context}" if self.learned_context else ""
            result = self._analyzer(
                topic=topic + ctx,
                subject=subject,
                concepts=concepts,
                running_example=running_example or f"A scenario involving {topic}",
                target_level=target_level
            )

            mistakes_data = self._parse_json_output(str(result.mistakes_json))
            mistakes = []
            for m in (mistakes_data if isinstance(mistakes_data, list) else []):
                mistakes.append(MistakeEntry(
                    description=m.get('description', ''),
                    why_it_happens=m.get('why_it_happens', ''),
                    how_to_avoid=m.get('how_to_avoid', ''),
                    example_wrong=m.get('example_wrong', ''),
                    example_correct=m.get('example_correct', '')
                ))

            traps = [t.strip() for t in str(result.trap_problems).split('|') if t.strip()]

            self._broadcast("mistakes_analyzed", {'topic': topic, 'count': len(mistakes)})

            return {
                'mistakes': mistakes,
                'trap_problems': traps
            }

        except Exception as e:
            logger.error(f"Mistake analysis failed: {e}")
            return {'mistakes': [], 'trap_problems': []}


# =============================================================================
# CONNECTION MAPPER AGENT
# =============================================================================

class ConnectionMapperAgent(BaseOlympiadAgent):
    """Maps connections between topics for deeper understanding."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mapper = self._create_module(ConnectionMapperSignature)

    async def map_connections(
        self,
        topic: str,
        subject: str,
        concepts: str
    ) -> Dict[str, Any]:
        """Map connections to other topics."""
        try:
            ctx = f"\n\n{self.learned_context}" if self.learned_context else ""
            result = self._mapper(
                topic=topic + ctx,
                subject=subject,
                concepts=concepts
            )

            direct = [c.strip() for c in str(result.direct_connections).split('|') if c.strip()]
            surprising = [c.strip() for c in str(result.surprising_connections).split('|') if c.strip()]
            powerful = [c.strip() for c in str(result.powerful_combinations).split('|') if c.strip()]
            next_topics = [t.strip() for t in str(result.next_topics).split('|') if t.strip()]

            self._broadcast("connections_mapped", {'topic': topic})

            return {
                'direct_connections': direct,
                'surprising_connections': surprising,
                'powerful_combinations': powerful,
                'next_topics': next_topics
            }

        except Exception as e:
            logger.error(f"Connection mapping failed: {e}")
            return {'direct_connections': [], 'surprising_connections': [], 'powerful_combinations': [], 'next_topics': []}


# =============================================================================
# CONTENT ASSEMBLER AGENT
# =============================================================================

class ContentAssemblerAgent(BaseOlympiadAgent):
    """Assembles all components into a beautifully flowing lesson."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._assembler = self._create_module(ContentAssemblerSignature)

    async def assemble(
        self,
        student_name: str,
        topic: str,
        subject: str,
        building_blocks: str,
        intuition: str,
        decomposition: str,
        patterns: str,
        strategies: str,
        problems_summary: str,
        mistakes: str,
        connections: str,
        celebration_word: str
    ) -> Dict[str, Any]:
        """Assemble all components into a complete lesson."""
        try:
            result = self._assembler(
                student_name=student_name,
                topic=topic,
                subject=subject,
                building_blocks=building_blocks,
                intuition=intuition,
                decomposition=decomposition,
                patterns=patterns,
                strategies=strategies,
                problems_summary=problems_summary,
                mistakes=mistakes,
                connections=connections,
                celebration_word=celebration_word
            )

            insights = [i.strip() for i in str(result.key_insights).split('|') if i.strip()]
            tips = [t.strip() for t in str(result.competition_tips).split('|') if t.strip()]

            self._broadcast("content_assembled", {'topic': topic})

            return {
                'complete_content': str(result.complete_content),
                'key_insights': insights,
                'summary': str(result.summary),
                'competition_tips': tips
            }

        except Exception as e:
            logger.error(f"Content assembly failed: {e}")
            return {'complete_content': '', 'key_insights': [], 'summary': '', 'competition_tips': []}


# =============================================================================
# UNIFIED TOPIC AGENT (SINGLE-PASS DEEP GENERATION)
# =============================================================================

class UnifiedTopicAgent(BaseOlympiadAgent):
    """Single-pass deep content generator for maximum quality per concept.

    Uses one comprehensive LLM call per concept for parallel execution.
    Produces the most thorough content possible.
    """

    _content_cache: Dict[str, Dict] = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._deep_generator = self._create_module(SingleTopicDeepSignature)

    async def generate_deep(
        self,
        student_name: str,
        topic: str,
        subject: str,
        target_level: str = "olympiad",
        celebration_word: str = "Brilliant!"
    ) -> Dict[str, Any]:
        """Generate comprehensive deep content for a single topic."""
        cache_key = f"deep_{subject}_{topic}_{target_level}"
        if cache_key in self._content_cache:
            logger.info(f"Using cached deep content for {topic}")
            return self._content_cache[cache_key]

        try:
            result = self._deep_generator(
                student_name=student_name,
                topic=topic,
                subject=subject,
                target_level=target_level,
                celebration_word=celebration_word
            )

            content = {
                'hook': str(result.hook),
                'building_blocks': str(result.building_blocks),
                'concept_discovery': str(result.concept_discovery),
                'formal_definition': str(result.formal_definition),
                'key_insight': str(result.key_insight),
                'patterns': str(result.patterns),
                'strategies': str(result.strategies),
                'problems_foundation': str(result.problems_foundation),
                'problems_intermediate': str(result.problems_intermediate),
                'problems_advanced': str(result.problems_advanced),
                'common_mistakes': str(result.common_mistakes),
                'connections': str(result.connections),
                'competition_tips': str(result.competition_tips),
            }

            self._content_cache[cache_key] = content
            self._broadcast("deep_content_generated", {'topic': topic})
            return content

        except Exception as e:
            logger.error(f"Deep content generation failed for {topic}: {e}")
            return {}

    async def generate_parallel(
        self,
        student_name: str,
        topics: List[str],
        subject: str,
        target_level: str = "olympiad",
        celebration_word: str = "Brilliant!",
        max_concurrent: int = 3
    ) -> List[Dict[str, Any]]:
        """Generate deep content for multiple topics in parallel."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def gen_one(topic: str) -> Dict[str, Any]:
            async with semaphore:
                return await self.generate_deep(
                    student_name=student_name,
                    topic=topic,
                    subject=subject,
                    target_level=target_level,
                    celebration_word=celebration_word
                )

        logger.info(f"Generating deep content for {len(topics)} topics in parallel...")
        start = datetime.now()
        tasks = [gen_one(t) for t in topics]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = (datetime.now() - start).total_seconds()
        logger.info(f"Parallel generation complete in {elapsed:.1f}s")

        return [r if isinstance(r, dict) else {} for r in results]


# =============================================================================
# NARRATIVE EDITOR AGENT
# =============================================================================

class NarrativeEditorAgent(BaseOlympiadAgent):
    """Post-assembly editor that weaves content into a coherent, curious narrative.

    Uses Sonnet model with 16K output tokens — Haiku was too weak for large rewrites.
    Sonnet has 200K context so no input truncation needed.

    Takes assembled content from multiple agents and rewrites for:
    - Running example threading throughout all sections
    - Socratic questions before major reveals
    - Specific (not generic) breakthrough moments
    - Natural transitions between sections
    - Curiosity gaps and mini-mysteries
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_sonnet_lm()
        # Use Predict (not ChainOfThought) to maximize output token budget for edited content
        self._editor = dspy.Predict(NarrativeEditorSignature)

    def _ensure_sonnet_lm(self):
        """Create Sonnet LM instance with 16K output for narrative editing (does NOT set global)."""
        try:
            from Jotty.core.foundation.direct_anthropic_lm import DirectAnthropicLM, is_api_key_available
            if is_api_key_available():
                self._lm = DirectAnthropicLM(model="sonnet", max_tokens=16384)
                logger.info("NarrativeEditor using Sonnet model (16K output)")
                return
        except Exception as e:
            logger.debug(f"Sonnet not available for NarrativeEditor: {e}")
        self._get_lm()

    async def edit(
        self,
        assembled_content: str,
        running_example: str,
        key_insights: List[str],
        pattern_names: List[str],
        student_name: str,
        topic: str,
    ) -> Dict[str, Any]:
        """Edit assembled content for coherence and curiosity.

        Args:
            assembled_content: Full markdown lesson content
            running_example: Running example/scenario to thread throughout
            key_insights: Key insights for specific breakthrough moments
            pattern_names: Pattern names from Pattern Library
            student_name: Student's name
            topic: Topic being taught

        Returns:
            Dict with edited_content, socratic_questions, breakthrough_moments
        """
        try:
            # Sonnet has 200K context — send full content, no truncation needed
            content_input = assembled_content
            logger.info(f"NarrativeEditor: processing {len(assembled_content)} chars with Sonnet")

            insights_str = ' | '.join(i for i in key_insights if i)
            patterns_str = ' | '.join(p for p in pattern_names if p)

            result = self._call_with_own_lm(
                self._editor,
                assembled_content=content_input,
                running_example=running_example or "the scenario from the introduction",
                key_insights=insights_str or "Key concept insights",
                pattern_names=patterns_str or "Core patterns",
                student_name=student_name,
                topic=topic,
            )

            edited = str(result.edited_content)
            socratic = [q.strip() for q in str(result.socratic_questions).split('|') if q.strip()]
            breakthroughs = [b.strip() for b in str(result.breakthrough_moments).split('|') if b.strip()]

            self._broadcast("narrative_edited", {
                'topic': topic,
                'socratic_count': len(socratic),
                'breakthrough_count': len(breakthroughs),
            })

            logger.info(f"NarrativeEditor: {len(socratic)} Socratic questions, {len(breakthroughs)} breakthroughs")

            return {
                'edited_content': edited,
                'socratic_questions': socratic,
                'breakthrough_moments': breakthroughs,
            }

        except Exception as e:
            logger.warning(f"NarrativeEditor failed, returning original content: {e}")
            return {
                'edited_content': assembled_content,
                'socratic_questions': [],
                'breakthrough_moments': [],
            }


# =============================================================================
# RANK TIPS AGENT
# =============================================================================

class RankTipsAgent(BaseOlympiadAgent):
    """Generates 20-30 actionable tips to help secure the #1 rank for a topic."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_direct_api_lm()
        self._generator = self._create_module(RankTipsSignature)

    def _ensure_direct_api_lm(self):
        """Create DirectAnthropicLM instance for reliable generation (does NOT set global)."""
        try:
            from Jotty.core.foundation.direct_anthropic_lm import DirectAnthropicLM, is_api_key_available
            if is_api_key_available():
                self._lm = DirectAnthropicLM(model=self.model, max_tokens=8192)
                return
        except Exception as e:
            logger.debug(f"DirectAnthropicLM not available for RankTips: {e}")
        self._get_lm()

    async def generate(
        self,
        topic: str,
        subject: str,
        target_level: str,
        student_name: str,
        patterns_summary: str,
        mistakes_summary: str,
    ) -> List[str]:
        """Generate 20-30 rank-securing tips.

        Returns:
            List of tip strings.
        """
        try:
            ctx = f"\n\n{self.learned_context}" if self.learned_context else ""
            result = self._call_with_own_lm(
                self._generator,
                topic=topic + ctx,
                subject=subject,
                target_level=target_level,
                student_name=student_name,
                patterns_summary=patterns_summary,
                mistakes_summary=mistakes_summary,
            )

            raw_tips = str(result.rank_tips)
            # Try pipe separator first
            tips = [t.strip() for t in raw_tips.split('|') if t.strip()]
            # If LLM used numbered lines instead of pipes, split on numbered patterns
            if len(tips) < 5:
                tips = re.split(r'\n\s*\d+[\.\)]\s+', '\n' + raw_tips)
                tips = [t.strip() for t in tips if t.strip()]
            # If still too few, split on double newlines
            if len(tips) < 5:
                tips = [t.strip() for t in raw_tips.split('\n\n') if t.strip()]
            # Final fallback: split on single newlines if they look like separate tips
            if len(tips) < 5:
                tips = [t.strip() for t in raw_tips.split('\n') if t.strip() and len(t.strip()) > 20]
            self._broadcast("rank_tips_generated", {'topic': topic, 'count': len(tips)})
            logger.info(f"RankTipsAgent: generated {len(tips)} tips for {topic}")
            return tips

        except Exception as e:
            logger.error(f"Rank tips generation failed: {e}")
            return []


__all__ = [
    'BaseOlympiadAgent',
    'CurriculumArchitectAgent', 'ConceptDecomposerAgent',
    'IntuitionBuilderAgent', 'PatternHunterAgent',
    'ProblemCrafterAgent', 'SolutionStrategistAgent',
    'MistakeAnalyzerAgent', 'ConnectionMapperAgent',
    'ContentAssemblerAgent', 'UnifiedTopicAgent',
    'NarrativeEditorAgent', 'RankTipsAgent',
]
