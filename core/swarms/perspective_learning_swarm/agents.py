"""Perspective Learning Swarm - Agent implementations.

Each agent is a specialist in one aspect of multi-perspective teaching:
- CurriculumDesignerAgent: Designs the multi-perspective learning plan
- IntuitiveExplainerAgent: Visual/step-by-step discovery
- FrameworkBuilderAgent: Mental models and structured thinking
- StorytellerAgent: Narrative-based learning
- DebateArchitectAgent: Critical thinking and argumentation
- ProjectDesignerAgent: Hands-on activities and projects
- RealWorldConnectorAgent: Real-world connections and applications
- MultilingualAgent: Content in Hindi, Kannada, French
- ContentAssemblerAgent: Assembles everything into one document
- NarrativeEditorAgent: Final polish and coherence pass
"""

import logging
from typing import Dict, Any, List

import dspy

from Jotty.core.swarms.olympiad_learning_swarm.agents import BaseOlympiadAgent
from .types import format_steps_on_newlines
from .signatures import (
    CurriculumDesignerSignature, IntuitiveExplainerSignature,
    FrameworkBuilderSignature, StorytellerSignature,
    DebateArchitectSignature, ProjectDesignerSignature,
    RealWorldConnectorSignature, MultilingualContentSignature,
    ContentAssemblerSignature, NarrativeEditorSignature,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CURRICULUM DESIGNER AGENT
# =============================================================================

class CurriculumDesignerAgent(BaseOlympiadAgent):
    """Designs the complete multi-perspective learning plan.

    Uses Sonnet model for higher quality planning — the planner drives
    all downstream content depth and running example quality.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_sonnet_lm()
        self._designer = self._create_module(CurriculumDesignerSignature)

    def _ensure_sonnet_lm(self) -> None:
        """Create Sonnet LM instance for curriculum planning (does NOT set global)."""
        try:
            from Jotty.core.foundation.direct_anthropic_lm import DirectAnthropicLM, is_api_key_available
            if is_api_key_available():
                self._lm = DirectAnthropicLM(model="sonnet", max_tokens=8192)
                logger.info("CurriculumDesigner using Sonnet model")
                return
        except Exception as e:
            logger.debug(f"Sonnet not available for CurriculumDesigner: {e}")
        self._get_lm()

    async def design(
        self,
        topic: str,
        student_name: str,
        age_group: str = "primary",
        central_idea: str = "",
    ) -> Dict[str, Any]:
        """Design the multi-perspective curriculum for a topic."""
        try:
            ctx = f"\n\n{self.learned_context}" if self.learned_context else ""
            result = self._call_with_own_lm(
                self._designer,
                topic=topic + ctx,
                student_name=student_name,
                age_group=age_group,
                central_idea=central_idea or topic,
            )

            objectives = [o.strip() for o in str(result.learning_objectives).split('|') if o.strip()]
            concepts = self._parse_json_output(str(result.key_concepts_json))
            vocabulary = self._parse_json_output(str(result.vocabulary_json))
            running_example = str(getattr(result, 'running_example_scenario', ''))
            section_depth_plan = str(getattr(result, 'section_depth_plan', ''))
            transdisciplinary = str(getattr(result, 'transdisciplinary_connections', ''))

            self._broadcast("curriculum_designed", {'topic': topic, 'objectives': len(objectives)})

            return {
                'learning_objectives': objectives,
                'key_concepts': concepts if isinstance(concepts, list) else [],
                'vocabulary': vocabulary if isinstance(vocabulary, list) else [],
                'running_example_scenario': running_example,
                'section_depth_plan': section_depth_plan,
                'transdisciplinary_connections': transdisciplinary,
            }

        except Exception as e:
            logger.error(f"Curriculum design failed: {e}")
            return {
                'learning_objectives': [], 'key_concepts': [], 'vocabulary': [],
                'running_example_scenario': '', 'section_depth_plan': '',
                'transdisciplinary_connections': '',
            }


# =============================================================================
# INTUITIVE EXPLAINER AGENT
# =============================================================================

class IntuitiveExplainerAgent(BaseOlympiadAgent):
    """Explains concepts through visual/step-by-step discovery."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._explainer = self._create_module(IntuitiveExplainerSignature)

    async def explain(
        self,
        topic: str,
        concepts: str,
        student_name: str,
        age_group: str,
        running_example: str,
    ) -> Dict[str, Any]:
        """Generate intuitive visual explanation."""
        try:
            ctx = f"\n\n{self.learned_context}" if self.learned_context else ""
            result = self._call_with_own_lm(
                self._explainer,
                topic=topic + ctx,
                concepts=concepts,
                student_name=student_name,
                age_group=age_group,
                running_example=running_example,
            )

            visuals = [v.strip() for v in str(result.visual_descriptions).split('|') if v.strip()]
            questions = [q.strip() for q in str(result.check_your_understanding).split('|') if q.strip()]

            self._broadcast("intuitive_explained", {'topic': topic[:50]})

            return {
                'simplest_example': format_steps_on_newlines(str(result.simplest_example)),
                'step_by_step_build': format_steps_on_newlines(str(result.step_by_step_build)),
                'visual_descriptions': visuals,
                'aha_moment': str(result.aha_moment),
                'check_your_understanding': questions,
            }

        except Exception as e:
            logger.error(f"Intuitive explanation failed: {e}")
            return {}


# =============================================================================
# FRAMEWORK BUILDER AGENT
# =============================================================================

class FrameworkBuilderAgent(BaseOlympiadAgent):
    """Builds mental models and structured thinking frameworks."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._builder = self._create_module(FrameworkBuilderSignature)

    async def build(
        self,
        topic: str,
        concepts: str,
        student_name: str,
        age_group: str,
        running_example: str,
    ) -> Dict[str, Any]:
        """Generate structured frameworks."""
        try:
            ctx = f"\n\n{self.learned_context}" if self.learned_context else ""
            result = self._call_with_own_lm(
                self._builder,
                topic=topic + ctx,
                concepts=concepts,
                student_name=student_name,
                age_group=age_group,
                running_example=running_example,
            )

            frameworks = self._parse_json_output(str(result.frameworks_json))
            principles = [p.strip() for p in str(result.key_principles).split('|') if p.strip()]
            checklist = [c.strip() for c in str(result.thinking_checklist).split('|') if c.strip()]

            self._broadcast("frameworks_built", {'topic': topic[:50]})

            return {
                'frameworks': frameworks if isinstance(frameworks, list) else [],
                'decision_tree': str(result.decision_tree),
                'comparison_matrix': str(result.comparison_matrix),
                'key_principles': principles,
                'thinking_checklist': checklist,
            }

        except Exception as e:
            logger.error(f"Framework building failed: {e}")
            return {'frameworks': [], 'decision_tree': '', 'comparison_matrix': '', 'key_principles': [], 'thinking_checklist': []}


# =============================================================================
# STORYTELLER AGENT
# =============================================================================

class StorytellerAgent(BaseOlympiadAgent):
    """Creates narrative-based learning content."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._storyteller = self._create_module(StorytellerSignature)

    async def tell(
        self,
        topic: str,
        concepts: str,
        student_name: str,
        age_group: str,
        running_example: str,
    ) -> Dict[str, Any]:
        """Generate a story that teaches the concept."""
        try:
            ctx = f"\n\n{self.learned_context}" if self.learned_context else ""
            result = self._call_with_own_lm(
                self._storyteller,
                topic=topic + ctx,
                concepts=concepts,
                student_name=student_name,
                age_group=age_group,
                running_example=running_example,
            )

            characters = self._parse_json_output(str(result.characters_json))
            questions = [q.strip() for q in str(result.discussion_questions).split('|') if q.strip()]

            self._broadcast("story_told", {'topic': topic[:50]})

            return {
                'story': str(result.story),
                'characters': characters if isinstance(characters, list) else [],
                'moral_or_lesson': str(result.moral_or_lesson),
                'discussion_questions': questions,
                'connect_to_life': str(result.connect_to_life),
            }

        except Exception as e:
            logger.error(f"Storytelling failed: {e}")
            return {}


# =============================================================================
# DEBATE ARCHITECT AGENT
# =============================================================================

class DebateArchitectAgent(BaseOlympiadAgent):
    """Builds critical thinking and debate content."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._architect = self._create_module(DebateArchitectSignature)

    async def architect(
        self,
        topic: str,
        concepts: str,
        student_name: str,
        age_group: str,
        running_example: str,
    ) -> Dict[str, Any]:
        """Generate debate/critical thinking content."""
        try:
            ctx = f"\n\n{self.learned_context}" if self.learned_context else ""
            result = self._call_with_own_lm(
                self._architect,
                topic=topic + ctx,
                concepts=concepts,
                student_name=student_name,
                age_group=age_group,
                running_example=running_example,
            )

            points_for = self._parse_json_output(str(result.points_for_json))
            points_against = self._parse_json_output(str(result.points_against_json))
            bias_tips = [t.strip() for t in str(result.bias_spotting_tips).split('|') if t.strip()]
            critical_qs = [q.strip() for q in str(result.critical_questions).split('|') if q.strip()]

            self._broadcast("debate_architected", {'topic': topic[:50]})

            return {
                'central_question': str(result.central_question),
                'points_for': points_for if isinstance(points_for, list) else [],
                'points_against': points_against if isinstance(points_against, list) else [],
                'bias_spotting_tips': bias_tips,
                'form_your_opinion': str(result.form_your_opinion),
                'critical_questions': critical_qs,
            }

        except Exception as e:
            logger.error(f"Debate architecture failed: {e}")
            return {'central_question': '', 'points_for': [], 'points_against': [], 'bias_spotting_tips': [], 'form_your_opinion': '', 'critical_questions': []}


# =============================================================================
# PROJECT DESIGNER AGENT
# =============================================================================

class ProjectDesignerAgent(BaseOlympiadAgent):
    """Designs hands-on project activities."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._designer = self._create_module(ProjectDesignerSignature)

    async def design(
        self,
        topic: str,
        concepts: str,
        student_name: str,
        age_group: str,
        running_example: str,
    ) -> Dict[str, Any]:
        """Generate hands-on project activities."""
        try:
            ctx = f"\n\n{self.learned_context}" if self.learned_context else ""
            result = self._call_with_own_lm(
                self._designer,
                topic=topic + ctx,
                concepts=concepts,
                student_name=student_name,
                age_group=age_group,
                running_example=running_example,
            )

            projects = self._parse_json_output(str(result.projects_json))
            presentation = [p.strip() for p in str(result.presentation_outline).split('|') if p.strip()]

            self._broadcast("projects_designed", {'topic': topic[:50]})

            return {
                'projects': projects if isinstance(projects, list) else [],
                'poster_design_brief': str(result.poster_design_brief),
                'role_play_scenario': str(result.role_play_scenario),
                'presentation_outline': presentation,
                'reflection_activity': str(result.reflection_activity),
            }

        except Exception as e:
            logger.error(f"Project design failed: {e}")
            return {'projects': [], 'poster_design_brief': '', 'role_play_scenario': '', 'presentation_outline': [], 'reflection_activity': ''}


# =============================================================================
# REAL WORLD CONNECTOR AGENT
# =============================================================================

class RealWorldConnectorAgent(BaseOlympiadAgent):
    """Connects topics to real-world applications."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._connector = self._create_module(RealWorldConnectorSignature)

    async def connect(
        self,
        topic: str,
        concepts: str,
        student_name: str,
        age_group: str,
        running_example: str,
    ) -> Dict[str, Any]:
        """Generate real-world connections."""
        try:
            ctx = f"\n\n{self.learned_context}" if self.learned_context else ""
            result = self._call_with_own_lm(
                self._connector,
                topic=topic + ctx,
                concepts=concepts,
                student_name=student_name,
                age_group=age_group,
                running_example=running_example,
            )

            daily = [d.strip() for d in str(result.daily_life_connections).split('|') if d.strip()]
            careers = [c.strip() for c in str(result.career_connections).split('|') if c.strip()]
            events = [e.strip() for e in str(result.current_events_link).split('|') if e.strip()]
            interviews = [q.strip() for q in str(result.interview_questions).split('|') if q.strip()]

            self._broadcast("real_world_connected", {'topic': topic[:50]})

            return {
                'daily_life_connections': daily,
                'career_connections': careers,
                'current_events_link': events,
                'future_impact': str(result.future_impact),
                'interview_questions': interviews,
            }

        except Exception as e:
            logger.error(f"Real-world connection failed: {e}")
            return {'daily_life_connections': [], 'career_connections': [], 'current_events_link': [], 'future_impact': '', 'interview_questions': []}


# =============================================================================
# MULTILINGUAL AGENT
# =============================================================================

class MultilingualAgent(BaseOlympiadAgent):
    """Generates content in Hindi, Kannada, and French."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._translator = self._create_module(MultilingualContentSignature)

    async def generate(
        self,
        topic: str,
        key_concepts: str,
        vocabulary: str,
        student_name: str,
        running_example: str,
        target_language: str,
    ) -> Dict[str, Any]:
        """Generate content in the target language."""
        try:
            ctx = f"\n\n{self.learned_context}" if self.learned_context else ""
            result = self._call_with_own_lm(
                self._translator,
                topic=topic + ctx,
                key_concepts=key_concepts,
                vocabulary=vocabulary,
                student_name=student_name,
                running_example=running_example,
                target_language=target_language,
            )

            vocab = [v.strip() for v in str(result.key_vocabulary_translated).split('|') if v.strip()]
            prompts = [p.strip() for p in str(result.reflection_prompts).split('|') if p.strip()]
            slogans = [s.strip() for s in str(result.slogans).split('|') if s.strip()]

            self._broadcast("multilingual_generated", {'language': target_language})

            return {
                'summary': str(result.summary),
                'key_vocabulary': vocab,
                'reflection_prompts': prompts,
                'activity': str(result.activity),
                'slogans': slogans,
            }

        except Exception as e:
            logger.error(f"Multilingual generation failed for {target_language}: {e}")
            return {'summary': '', 'key_vocabulary': [], 'reflection_prompts': [], 'activity': '', 'slogans': []}


# =============================================================================
# CONTENT ASSEMBLER AGENT
# =============================================================================

class ContentAssemblerAgent(BaseOlympiadAgent):
    """Assembles all perspectives and languages into one cohesive document."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._assembler = self._create_module(ContentAssemblerSignature)

    async def assemble(
        self,
        student_name: str,
        topic: str,
        central_idea: str,
        running_example: str,
        learning_objectives: str,
        intuitive_content: str,
        framework_content: str,
        story_content: str,
        debate_content: str,
        project_content: str,
        real_world_content: str,
        celebration_word: str,
    ) -> Dict[str, Any]:
        """Assemble all components into a complete lesson."""
        try:
            result = self._call_with_own_lm(
                self._assembler,
                student_name=student_name,
                topic=topic,
                central_idea=central_idea,
                running_example=running_example,
                learning_objectives=learning_objectives,
                intuitive_content=intuitive_content,
                framework_content=framework_content,
                story_content=story_content,
                debate_content=debate_content,
                project_content=project_content,
                real_world_content=real_world_content,
                celebration_word=celebration_word,
            )

            insights = [i.strip() for i in str(result.key_insights).split('|') if i.strip()]
            toc = [t.strip() for t in str(result.table_of_contents).split('|') if t.strip()]

            self._broadcast("content_assembled", {'topic': topic})

            return {
                'assembled_content': str(result.assembled_content),
                'table_of_contents': toc,
                'key_insights': insights,
            }

        except Exception as e:
            logger.error(f"Content assembly failed: {e}")
            return {'assembled_content': '', 'table_of_contents': [], 'key_insights': []}


# =============================================================================
# NARRATIVE EDITOR AGENT
# =============================================================================

class NarrativeEditorAgent(BaseOlympiadAgent):
    """Generates supplementary content: Socratic questions, parent guide, key takeaways.

    Uses Haiku for fast generation — no full-content rewrite needed.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._editor = dspy.Predict(NarrativeEditorSignature)

    async def edit(
        self,
        assembled_content: str,
        running_example: str,
        student_name: str,
        topic: str,
    ) -> Dict[str, Any]:
        """Generate supplementary content (Socratic questions, parent guide, key takeaways)."""
        try:
            # Truncate to first 2000 chars as a summary — enough context for supplements
            content_summary = assembled_content[:2000] if assembled_content else ""
            logger.info(f"NarrativeEditor: generating supplements from {len(content_summary)} char summary")

            result = self._call_with_own_lm(
                self._editor,
                content_summary=content_summary,
                running_example=running_example or "the scenario from the introduction",
                student_name=student_name,
                topic=topic,
            )

            socratic = [q.strip() for q in str(result.socratic_questions).split('|') if q.strip()]
            parent_guide = str(result.parent_guide)
            key_takeaways = [t.strip() for t in str(result.key_takeaways).split('|') if t.strip()]

            self._broadcast("narrative_edited", {
                'topic': topic,
                'socratic_count': len(socratic),
                'takeaways_count': len(key_takeaways),
            })

            logger.info(f"NarrativeEditor: {len(socratic)} Socratic questions, "
                        f"{len(key_takeaways)} takeaways, parent guide {len(parent_guide)} chars")

            return {
                'socratic_questions': socratic,
                'parent_guide': parent_guide,
                'key_takeaways': key_takeaways,
            }

        except Exception as e:
            logger.warning(f"NarrativeEditor failed: {e}")
            return {
                'socratic_questions': [],
                'parent_guide': '',
                'key_takeaways': [],
            }


__all__ = [
    'CurriculumDesignerAgent', 'IntuitiveExplainerAgent',
    'FrameworkBuilderAgent', 'StorytellerAgent',
    'DebateArchitectAgent', 'ProjectDesignerAgent',
    'RealWorldConnectorAgent', 'MultilingualAgent',
    'ContentAssemblerAgent', 'NarrativeEditorAgent',
]
