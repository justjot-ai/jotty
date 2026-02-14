"""Perspective Learning Swarm - Main swarm orchestrator.

Multi-perspective educational swarm that takes ANY topic and creates
comprehensive learning content from 6 distinct perspectives in 4 languages,
producing professional PDF + HTML output.

Teaching Philosophy (embedded in every agent):
- Start with WHY — why does this matter to a child's life?
- Explore from every angle: visual, structured, narrative, critical, hands-on, real-world
- Learning in multiple languages deepens understanding
- Celebrate insight moments
- Age-appropriate, personalized for the student

Usage:
    # Quick start
    result = await teach_perspectives("Media and its influence on decisions", student_name="Aria")

    # Full control
    swarm = PerspectiveLearningSwarm(PerspectiveLearningConfig(
        student_name="Aria",
        age_group=AgeGroup.PRIMARY,
        depth=ContentDepth.STANDARD,
    ))
    result = await swarm.teach(topic="Media and its influence on decisions")
"""

import asyncio
import logging
import json
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

import dspy

from ..base_swarm import (
    BaseSwarm, SwarmBaseConfig, SwarmResult, AgentRole,
    register_swarm, ExecutionTrace
)
from ..base import DomainSwarm, AgentTeam, PhaseExecutor

from .types import (
    PerspectiveType, Language, AgeGroup, ContentDepth,
    PerspectiveLearningConfig, PerspectiveSection, LanguageContent,
    LessonContent, PerspectiveLearningResult,
    format_steps_on_newlines, PERSPECTIVE_LABELS, LANGUAGE_LABELS,
)
from .agents import (
    CurriculumDesignerAgent, IntuitiveExplainerAgent,
    FrameworkBuilderAgent, StorytellerAgent,
    DebateArchitectAgent, ProjectDesignerAgent,
    RealWorldConnectorAgent, MultilingualAgent,
    ContentAssemblerAgent, NarrativeEditorAgent,
)

# Telegram support
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "skills" / "telegram-sender"))
    from tools import send_telegram_message_tool, send_telegram_file_tool
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    send_telegram_message_tool = None
    send_telegram_file_tool = None

logger = logging.getLogger(__name__)

# Log Telegram availability at import time
if TELEGRAM_AVAILABLE:
    _has_token = bool(os.environ.get('TELEGRAM_TOKEN'))
    _has_chat = bool(os.environ.get('TELEGRAM_CHAT_ID'))
    if _has_token and _has_chat:
        logger.info("Telegram: available (token + chat_id present)")
    else:
        logger.warning(f"Telegram: tools imported but env vars missing "
                       f"(TELEGRAM_TOKEN={'set' if _has_token else 'MISSING'}, "
                       f"TELEGRAM_CHAT_ID={'set' if _has_chat else 'MISSING'})")
else:
    logger.info("Telegram: tools not available (import failed)")


@register_swarm("perspective_learning")
class PerspectiveLearningSwarm(DomainSwarm):
    """
    Multi-Perspective Learning Swarm.

    Creates comprehensive, multi-perspective learning content exploring ANY topic
    from 6 distinct angles in 4 languages.

    10 Specialized Agents:
    1. CurriculumDesigner - Designs the multi-perspective learning plan
    2. IntuitiveExplainer - Visual/step-by-step discovery
    3. FrameworkBuilder - Mental models and structured thinking
    4. Storyteller - Narrative-based learning
    5. DebateArchitect - Critical thinking and argumentation
    6. ProjectDesigner - Hands-on activities and projects
    7. RealWorldConnector - Real-world connections
    8. Multilingual - Content in Hindi, Kannada, French
    9. ContentAssembler - Assembles everything into one document
    10. NarrativeEditor - Final polish and coherence pass
    """

    AGENT_TEAM = AgentTeam.define(
        (CurriculumDesignerAgent, "CurriculumDesigner", "_curriculum_designer"),
        (IntuitiveExplainerAgent, "IntuitiveExplainer", "_intuitive_explainer"),
        (FrameworkBuilderAgent, "FrameworkBuilder", "_framework_builder"),
        (StorytellerAgent, "Storyteller", "_storyteller"),
        (DebateArchitectAgent, "DebateArchitect", "_debate_architect"),
        (ProjectDesignerAgent, "ProjectDesigner", "_project_designer"),
        (RealWorldConnectorAgent, "RealWorldConnector", "_real_world_connector"),
        (MultilingualAgent, "Multilingual", "_multilingual"),
        (ContentAssemblerAgent, "ContentAssembler", "_content_assembler"),
        (NarrativeEditorAgent, "NarrativeEditor", "_narrative_editor"),
    )

    def __init__(self, config: PerspectiveLearningConfig = None) -> None:
        super().__init__(config or PerspectiveLearningConfig())
        self._optimization_mode = self.config.optimization_mode

    async def _execute_domain(self, topic: str = None, **kwargs: Any) -> PerspectiveLearningResult:
        """Execute learning content generation."""
        return await self.teach(topic=topic, **kwargs)

    async def teach(
        self,
        topic: str,
        student_name: str = None,
        age_group: str = None,
        depth: str = None,
        languages: List[Language] = None,
        perspectives: List[PerspectiveType] = None,
        central_idea: str = "",
        send_telegram: bool = None,
    ) -> PerspectiveLearningResult:
        """
        Create multi-perspective learning content for a topic.

        Args:
            topic: The topic to teach
            student_name: Override student name from config
            age_group: Override age group (string or AgeGroup enum)
            depth: Override content depth (string or ContentDepth enum)
            languages: Override languages to generate
            perspectives: Override perspectives to generate
            central_idea: Optional central idea statement
            send_telegram: Whether to send to Telegram

        Returns:
            PerspectiveLearningResult with complete lesson content
        """
        self._init_agents()

        config = self.config
        student_name = student_name or config.student_name
        age_group_str = age_group or (config.age_group.value if isinstance(config.age_group, AgeGroup) else str(config.age_group))
        languages = languages or config.languages
        perspectives = perspectives or config.perspectives

        logger.info(f"PerspectiveLearningSwarm starting: {topic} for {student_name}")

        async def _run_phases(executor: PhaseExecutor) -> PerspectiveLearningResult:
            return await self._execute_phases(
                executor, topic, student_name, age_group_str,
                languages, perspectives, central_idea, config, send_telegram,
            )

        return await self._safe_execute_domain(
            task_type='perspective_teaching',
            default_tools=['curriculum', 'perspectives', 'multilingual', 'assembly'],
            result_class=PerspectiveLearningResult,
            execute_fn=_run_phases,
            output_data_fn=lambda r: {
                'perspectives_count': r.perspectives_generated if hasattr(r, 'perspectives_generated') else 0,
                'languages_count': r.languages_generated if hasattr(r, 'languages_generated') else 0,
                'word_count': r.content.total_words if hasattr(r, 'content') and r.content else 0,
            },
            input_data_fn=lambda: {'topic': topic, 'student_name': student_name},
        )

    async def _execute_phases(
        self,
        executor: PhaseExecutor,
        topic: str,
        student_name: str,
        age_group: str,
        languages: List[Language],
        perspectives: List[PerspectiveType],
        central_idea: str,
        config: PerspectiveLearningConfig,
        send_telegram: bool = None,
    ) -> PerspectiveLearningResult:
        """Execute all teaching phases using the PhaseExecutor."""

        # ==============================================================
        # PHASE 1: Curriculum Design (serial — drives everything)
        # ==============================================================
        curriculum = await executor.run_phase(
            1, "Curriculum Design", "CurriculumDesigner", AgentRole.PLANNER,
            self._curriculum_designer.design(
                topic=topic, student_name=student_name,
                age_group=age_group, central_idea=central_idea,
            ),
            input_data={'topic': topic, 'student_name': student_name, 'age_group': age_group},
            tools_used=['curriculum_design'],
        )

        learning_objectives = curriculum.get('learning_objectives', [])
        key_concepts = curriculum.get('key_concepts', [])
        vocabulary = curriculum.get('vocabulary', [])
        running_example = curriculum.get('running_example_scenario', '')
        transdisciplinary = curriculum.get('transdisciplinary_connections', '')

        # Build concepts string for downstream agents
        concepts_str = ', '.join(c.get('name', '') for c in key_concepts if isinstance(c, dict))
        if not concepts_str:
            concepts_str = topic

        vocab_str = ', '.join(v.get('term', '') for v in vocabulary if isinstance(v, dict))

        if running_example:
            logger.info(f"Running example: {running_example[:80]}...")

        # ==============================================================
        # PHASE 2: All 6 Perspectives in parallel (semaphore max 5)
        # ==============================================================
        semaphore = asyncio.Semaphore(config.max_concurrent_llm)

        async def rate_limited(coro: Any) -> Any:
            async with semaphore:
                return await coro

        perspective_tasks = []

        if PerspectiveType.INTUITIVE_VISUAL in perspectives:
            perspective_tasks.append(
                ("IntuitiveExplainer", AgentRole.ACTOR,
                 rate_limited(self._intuitive_explainer.explain(
                     topic=topic, concepts=concepts_str,
                     student_name=student_name, age_group=age_group,
                     running_example=running_example,
                 )),
                 ['intuitive_explain'])
            )
        if PerspectiveType.STRUCTURED_FRAMEWORK in perspectives:
            perspective_tasks.append(
                ("FrameworkBuilder", AgentRole.ACTOR,
                 rate_limited(self._framework_builder.build(
                     topic=topic, concepts=concepts_str,
                     student_name=student_name, age_group=age_group,
                     running_example=running_example,
                 )),
                 ['framework_build'])
            )
        if PerspectiveType.STORYTELLING in perspectives:
            perspective_tasks.append(
                ("Storyteller", AgentRole.ACTOR,
                 rate_limited(self._storyteller.tell(
                     topic=topic, concepts=concepts_str,
                     student_name=student_name, age_group=age_group,
                     running_example=running_example,
                 )),
                 ['storytelling'])
            )
        if PerspectiveType.DEBATE_CRITICAL in perspectives:
            perspective_tasks.append(
                ("DebateArchitect", AgentRole.ACTOR,
                 rate_limited(self._debate_architect.architect(
                     topic=topic, concepts=concepts_str,
                     student_name=student_name, age_group=age_group,
                     running_example=running_example,
                 )),
                 ['debate_architect'])
            )
        if PerspectiveType.HANDS_ON_PROJECT in perspectives:
            perspective_tasks.append(
                ("ProjectDesigner", AgentRole.ACTOR,
                 rate_limited(self._project_designer.design(
                     topic=topic, concepts=concepts_str,
                     student_name=student_name, age_group=age_group,
                     running_example=running_example,
                 )),
                 ['project_design'])
            )
        if PerspectiveType.REAL_WORLD_APPLICATION in perspectives:
            perspective_tasks.append(
                ("RealWorldConnector", AgentRole.ACTOR,
                 rate_limited(self._real_world_connector.connect(
                     topic=topic, concepts=concepts_str,
                     student_name=student_name, age_group=age_group,
                     running_example=running_example,
                 )),
                 ['real_world_connect'])
            )

        perspective_results = await executor.run_parallel(
            2, f"6 Perspectives in Parallel ({len(perspective_tasks)} agents)",
            perspective_tasks,
        )

        # Unpack results by matching order to the tasks we added
        result_map = {}
        task_idx = 0
        for ptype in [PerspectiveType.INTUITIVE_VISUAL, PerspectiveType.STRUCTURED_FRAMEWORK,
                       PerspectiveType.STORYTELLING, PerspectiveType.DEBATE_CRITICAL,
                       PerspectiveType.HANDS_ON_PROJECT, PerspectiveType.REAL_WORLD_APPLICATION]:
            if ptype in perspectives:
                r = perspective_results[task_idx] if task_idx < len(perspective_results) else {}
                if isinstance(r, dict) and 'error' in r:
                    r = {}
                result_map[ptype] = r
                task_idx += 1

        intuitive_result = result_map.get(PerspectiveType.INTUITIVE_VISUAL, {})
        framework_result = result_map.get(PerspectiveType.STRUCTURED_FRAMEWORK, {})
        story_result = result_map.get(PerspectiveType.STORYTELLING, {})
        debate_result = result_map.get(PerspectiveType.DEBATE_CRITICAL, {})
        project_result = result_map.get(PerspectiveType.HANDS_ON_PROJECT, {})
        realworld_result = result_map.get(PerspectiveType.REAL_WORLD_APPLICATION, {})

        # ==============================================================
        # PHASE 3: Multilingual content in parallel (1 call per non-English language)
        # ==============================================================
        non_english = [lang for lang in languages if lang != Language.ENGLISH]
        if non_english:
            lang_tasks = []
            for lang in non_english:
                lang_tasks.append(
                    (f"Multilingual_{lang.value}", AgentRole.ACTOR,
                     rate_limited(self._multilingual.generate(
                         topic=topic, key_concepts=concepts_str,
                         vocabulary=vocab_str, student_name=student_name,
                         running_example=running_example,
                         target_language=lang.value,
                     )),
                     ['multilingual_generation'])
                )

            lang_results = await executor.run_parallel(
                3, f"Multilingual Content ({len(non_english)} languages)",
                lang_tasks,
            )
        else:
            lang_results = []

        # Build language content dict
        language_content = {}
        for i, lang in enumerate(non_english):
            r = lang_results[i] if i < len(lang_results) else {}
            if isinstance(r, dict) and 'error' in r:
                r = {}
            language_content[lang] = r

        # ==============================================================
        # PHASE 4: Content Assembly (serial)
        # ==============================================================
        # Build complete content programmatically (no LLM call needed)
        complete_content = self._build_complete_content(
            student_name=student_name, topic=topic,
            central_idea=central_idea or topic,
            learning_objectives=learning_objectives,
            key_concepts=key_concepts, vocabulary=vocabulary,
            running_example=running_example,
            intuitive=intuitive_result, framework=framework_result,
            story=story_result, debate=debate_result,
            project=project_result, realworld=realworld_result,
            language_content=language_content, config=config,
            transdisciplinary=transdisciplinary,
        )
        logger.info("Phase 4: Assembled multi-perspective lesson content")

        # ==============================================================
        # PHASE 5: Supplement Generation (Socratic questions, parent guide, takeaways)
        # ==============================================================
        narrative_result = await executor.run_phase(
            5, "Supplement Generation", "NarrativeEditor", AgentRole.REVIEWER,
            self._narrative_editor.edit(
                assembled_content=complete_content,
                running_example=running_example,
                student_name=student_name,
                topic=topic,
            ),
            tools_used=['narrative_edit'],
        )

        socratic_questions = narrative_result.get('socratic_questions', []) if isinstance(narrative_result, dict) else []
        parent_guide = narrative_result.get('parent_guide', '') if isinstance(narrative_result, dict) else ''
        key_takeaways = narrative_result.get('key_takeaways', []) if isinstance(narrative_result, dict) else []

        logger.info(f"Phase 5: {len(socratic_questions)} Socratic questions, "
                    f"{len(key_takeaways)} takeaways, parent guide {len(parent_guide)} chars")

        # Append supplement sections to the assembled content
        supplement_parts = []
        if parent_guide:
            supplement_parts.append(f"\n## Parent's Guide\n\n{parent_guide}\n")
        if key_takeaways:
            supplement_parts.append("\n## Key Takeaways\n")
            for takeaway in key_takeaways:
                supplement_parts.append(f"- {takeaway}")
            supplement_parts.append("")
        if socratic_questions:
            supplement_parts.append("\n## Questions to Think About\n")
            for i, q in enumerate(socratic_questions, 1):
                supplement_parts.append(f"{i}. {q}")
            supplement_parts.append("")

        if supplement_parts:
            complete_content += '\n'.join(supplement_parts)

        # Build structured LessonContent
        perspective_sections = self._build_perspective_sections(
            intuitive=intuitive_result, framework=framework_result,
            story=story_result, debate=debate_result,
            project=project_result, realworld=realworld_result,
            config=config,
        )

        language_sections = self._build_language_sections(language_content)

        key_insights = [
            intuitive_result.get('aha_moment', ''),
            story_result.get('moral_or_lesson', ''),
            debate_result.get('form_your_opinion', ''),
            realworld_result.get('future_impact', ''),
        ]
        key_insights = [i for i in key_insights if i]

        content = LessonContent(
            topic=topic,
            student_name=student_name,
            central_idea=central_idea or topic,
            learning_objectives=learning_objectives,
            key_concepts=key_concepts,
            running_example=running_example,
            vocabulary=vocabulary,
            perspectives=perspective_sections,
            language_sections=language_sections,
            key_insights=key_insights,
            parent_guide=parent_guide,
            socratic_questions=socratic_questions,
            total_words=len(complete_content.split()),
            key_takeaways=key_takeaways,
            transdisciplinary_connections=transdisciplinary,
        )

        # Build final result
        exec_time = executor.elapsed()

        final_result = PerspectiveLearningResult(
            success=True,
            swarm_name=self.config.name,
            domain=self.config.domain,
            output={
                'topic': topic,
                'student': student_name,
                'content_preview': complete_content[:500],
            },
            execution_time=exec_time,
            content=content,
            student_name=student_name,
            topic=topic,
            perspectives_generated=len(perspective_sections),
            languages_generated=len(language_sections),
        )

        logger.info(f"PerspectiveLearningSwarm complete: {topic}")
        logger.info(f"  {len(perspective_sections)} perspectives, "
                    f"{len(language_sections)} languages, "
                    f"{content.total_words} words")

        # ==============================================================
        # PHASE 6: Output Generation (PDF + HTML in parallel)
        # ==============================================================
        pdf_path = None
        html_path = None

        async def _noop() -> Any:
            return None

        output_results = await executor.run_parallel(
            6, "Output Generation (PDF + HTML)",
            [
                ("PDFGenerator", AgentRole.ACTOR,
                 self._generate_pdf(topic, student_name, content) if config.generate_pdf else _noop(),
                 ['pdf_generation']),
                ("HTMLGenerator", AgentRole.ACTOR,
                 self._generate_html(topic, student_name, content) if config.generate_html else _noop(),
                 ['html_generation']),
            ],
        )

        pdf_path = output_results[0] if not isinstance(output_results[0], dict) or 'error' not in output_results[0] else None
        html_path = output_results[1] if not isinstance(output_results[1], dict) or 'error' not in output_results[1] else None

        final_result.pdf_path = pdf_path
        final_result.html_path = html_path

        # Cost summary
        try:
            from Jotty.core.foundation.direct_anthropic_lm import get_cost_tracker
            tracker = get_cost_tracker()
            metrics = tracker.get_metrics()
            logger.info(
                f"COST SUMMARY: ${metrics.total_cost:.4f} | "
                f"{metrics.total_calls} calls | "
                f"{metrics.total_input_tokens}+{metrics.total_output_tokens} tokens"
            )
        except Exception:
            pass

        # ==============================================================
        # PHASE 7: Telegram (if enabled)
        # ==============================================================
        should_send = send_telegram if send_telegram is not None else config.send_telegram
        if should_send and not TELEGRAM_AVAILABLE:
            logger.warning("send_telegram=True but Telegram tools not available")
        elif should_send and not os.environ.get('TELEGRAM_TOKEN'):
            logger.warning("send_telegram=True but TELEGRAM_TOKEN not set in environment")
        if should_send and TELEGRAM_AVAILABLE and content:
            logger.info(f"Sending to Telegram (token={'set' if os.environ.get('TELEGRAM_TOKEN') else 'MISSING'})...")
            await self._send_to_telegram(
                topic, student_name, content,
                complete_content,
                pdf_path=pdf_path,
                html_path=html_path,
            )

        return final_result

    # =========================================================================
    # CONTENT BUILDERS
    # =========================================================================

    def _build_complete_content(
        self,
        student_name: str,
        topic: str,
        central_idea: str,
        learning_objectives: List[str],
        key_concepts: List[Dict],
        vocabulary: List[Dict],
        running_example: str,
        intuitive: Dict,
        framework: Dict,
        story: Dict,
        debate: Dict,
        project: Dict,
        realworld: Dict,
        language_content: Dict[Language, Dict],
        config: PerspectiveLearningConfig,
        transdisciplinary: str = "",
    ) -> str:
        """Build complete markdown lesson content."""
        celebration = config.celebration_word
        parts = []

        # Title
        parts.append(f"# {topic}")
        parts.append(f"### A Multi-Perspective Learning Journey for {student_name}\n")
        parts.append(f"*Central Idea: {central_idea}*\n")
        parts.append("---\n")

        # Table of Contents
        parts.append("## Table of Contents")
        parts.append("1. Why This Matters")
        parts.append("2. See It Clearly (Intuitive Visual)")
        parts.append("3. Think It Through (Structured Frameworks)")
        parts.append("4. Feel the Story (Narrative)")
        parts.append("5. Debate It (Critical Thinking)")
        parts.append("6. Build It (Hands-On Projects)")
        parts.append("7. Live It (Real-World Connections)")
        if Language.HINDI in language_content:
            parts.append("\u0939\u093f\u0928\u094d\u0926\u0940 \u092e\u0947\u0902 (In Hindi)")
        if Language.KANNADA in language_content:
            parts.append("\u0c95\u0ca8\u0ccd\u0ca8\u0ca1\u0ca6\u0cb2\u0ccd\u0cb2\u0cbf (In Kannada)")
        if Language.FRENCH in language_content:
            parts.append("En Fran\u00e7ais (In French)")
        parts.append("11. Parent's Guide")
        parts.append("12. Key Insights & Reflection\n")
        parts.append("---\n")

        # Why This Matters
        parts.append(f"## Why This Matters, {student_name}\n")
        if running_example:
            parts.append(f"{running_example}\n")
        if learning_objectives:
            parts.append("**What you'll learn:**")
            for obj in learning_objectives:
                parts.append(f"- {obj}")
            parts.append("")

        # ===== PERSPECTIVE 1: Intuitive Visual =====
        parts.append(f"\n## See It Clearly\n")
        if intuitive.get('simplest_example'):
            parts.append(f"### Let's Start Simple\n{intuitive['simplest_example']}\n")
        if intuitive.get('step_by_step_build'):
            parts.append(f"### Building Understanding\n{intuitive['step_by_step_build']}\n")
        if intuitive.get('visual_descriptions'):
            parts.append("### Picture This")
            for desc in intuitive['visual_descriptions']:
                parts.append(f"- {desc}")
            parts.append("")
        if intuitive.get('aha_moment'):
            parts.append(f"\n**{celebration}** {intuitive['aha_moment']}\n")
        if intuitive.get('check_your_understanding'):
            parts.append("### Check Your Understanding")
            for q in intuitive['check_your_understanding']:
                parts.append(f"- {q}")
            parts.append("")

        # ===== PERSPECTIVE 2: Structured Framework =====
        parts.append(f"\n## Think It Through\n")
        if framework.get('frameworks'):
            for fw in framework['frameworks']:
                if isinstance(fw, dict):
                    parts.append(f"### {fw.get('name', 'Framework')}")
                    parts.append(f"{fw.get('description', '')}")
                    if fw.get('how_to_use'):
                        parts.append(f"\n**How to use:** {fw['how_to_use']}")
                    if fw.get('visual_layout'):
                        parts.append(f"\n**Visual:** {fw['visual_layout']}")
                    if fw.get('example_applied'):
                        parts.append(f"\n**Applied:** {fw['example_applied']}\n")
        if framework.get('decision_tree'):
            parts.append(f"### Decision Tree\n{framework['decision_tree']}\n")
        if framework.get('comparison_matrix'):
            parts.append(f"### Comparison\n{framework['comparison_matrix']}\n")
        if framework.get('key_principles'):
            parts.append("### Key Principles")
            for p in framework['key_principles']:
                parts.append(f"- {p}")
            parts.append("")
        if framework.get('thinking_checklist'):
            parts.append("### Thinking Checklist")
            for i, step in enumerate(framework['thinking_checklist'], 1):
                parts.append(f"{i}. {step}")
            parts.append("")

        # ===== PERSPECTIVE 3: Storytelling =====
        parts.append(f"\n## Feel the Story\n")
        if story.get('story'):
            parts.append(f"{story['story']}\n")
        if story.get('moral_or_lesson'):
            parts.append(f"\n**{celebration}** {story['moral_or_lesson']}\n")
        if story.get('discussion_questions'):
            parts.append("### Let's Discuss")
            for q in story['discussion_questions']:
                parts.append(f"- {q}")
            parts.append("")
        if story.get('connect_to_life'):
            parts.append(f"### How This Connects to You\n{story['connect_to_life']}\n")

        # ===== PERSPECTIVE 4: Debate / Critical Thinking =====
        parts.append(f"\n## Debate It\n")
        if debate.get('central_question'):
            parts.append(f"**The Big Question:** {debate['central_question']}\n")
        if debate.get('points_for'):
            parts.append("### Arguments FOR")
            for pt in debate['points_for']:
                if isinstance(pt, dict):
                    parts.append(f"- **{pt.get('position', '')}** \u2014 {pt.get('argument', '')}")
                    if pt.get('evidence'):
                        parts.append(f"  *Evidence:* {pt['evidence']}")
        if debate.get('points_against'):
            parts.append("\n### Arguments AGAINST")
            for pt in debate['points_against']:
                if isinstance(pt, dict):
                    parts.append(f"- **{pt.get('position', '')}** \u2014 {pt.get('argument', '')}")
                    if pt.get('evidence'):
                        parts.append(f"  *Evidence:* {pt['evidence']}")
        if debate.get('bias_spotting_tips'):
            parts.append("\n### Spot the Bias")
            for tip in debate['bias_spotting_tips']:
                parts.append(f"- {tip}")
        if debate.get('form_your_opinion'):
            parts.append(f"\n### What Do YOU Think?\n{debate['form_your_opinion']}\n")
        if debate.get('critical_questions'):
            parts.append("### Critical Questions")
            for q in debate['critical_questions']:
                parts.append(f"- {q}")
            parts.append("")

        # ===== PERSPECTIVE 5: Hands-On Projects =====
        parts.append(f"\n## Build It\n")
        if project.get('projects'):
            for proj in project['projects']:
                if isinstance(proj, dict):
                    parts.append(f"### {proj.get('title', 'Activity')}")
                    parts.append(f"{proj.get('description', '')}")
                    if proj.get('materials'):
                        parts.append("\n**Materials:** " + ', '.join(proj['materials']))
                    if proj.get('steps'):
                        parts.append("\n**Steps:**")
                        for j, step in enumerate(proj['steps'], 1):
                            parts.append(f"{j}. {step}")
                    if proj.get('learning_outcome'):
                        parts.append(f"\n*What you'll learn:* {proj['learning_outcome']}\n")
        if project.get('poster_design_brief'):
            parts.append(f"### Poster Design Brief\n{project['poster_design_brief']}\n")
        if project.get('role_play_scenario'):
            parts.append(f"### Role-Play\n{project['role_play_scenario']}\n")
        if project.get('reflection_activity'):
            parts.append(f"### Reflect\n{project['reflection_activity']}\n")

        # ===== PERSPECTIVE 6: Real-World Connections =====
        parts.append(f"\n## Live It\n")
        if realworld.get('daily_life_connections'):
            parts.append("### In Your Daily Life")
            for conn in realworld['daily_life_connections']:
                parts.append(f"- {conn}")
            parts.append("")
        if realworld.get('career_connections'):
            parts.append("### Careers That Use This")
            for career in realworld['career_connections']:
                parts.append(f"- {career}")
            parts.append("")
        if realworld.get('current_events_link'):
            parts.append("### In the News")
            for event in realworld['current_events_link']:
                parts.append(f"- {event}")
            parts.append("")
        if realworld.get('future_impact'):
            parts.append(f"### Your Future\n{realworld['future_impact']}\n")
        if realworld.get('interview_questions'):
            parts.append("### Ask Someone You Know")
            for q in realworld['interview_questions']:
                parts.append(f"- {q}")
            parts.append("")

        # ===== LANGUAGE SECTIONS =====
        for lang in [Language.HINDI, Language.KANNADA, Language.FRENCH]:
            if lang in language_content:
                lc = language_content[lang]
                lang_label = LANGUAGE_LABELS.get(lang, lang.value.title())
                parts.append(f"\n## {lang_label} \u2014 {topic} in {lang.value.title()}\n")
                if lc.get('summary'):
                    parts.append(f"{lc['summary']}\n")
                if lc.get('key_vocabulary'):
                    parts.append("**Key Vocabulary:**")
                    for term in lc['key_vocabulary']:
                        parts.append(f"- {term}")
                    parts.append("")
                if lc.get('reflection_prompts'):
                    parts.append("**Reflect:**")
                    for prompt in lc['reflection_prompts']:
                        parts.append(f"- {prompt}")
                    parts.append("")
                if lc.get('activity'):
                    parts.append(f"**Activity:** {lc['activity']}\n")
                if lc.get('slogans'):
                    for slogan in lc['slogans']:
                        parts.append(f"*\u201c{slogan}\u201d*")
                    parts.append("")

        # Transdisciplinary connections
        if transdisciplinary:
            parts.append(f"\n## Connections Across Subjects\n{transdisciplinary}\n")

        return '\n'.join(parts)

    def _build_perspective_sections(
        self,
        intuitive: Dict,
        framework: Dict,
        story: Dict,
        debate: Dict,
        project: Dict,
        realworld: Dict,
        config: PerspectiveLearningConfig,
    ) -> List[PerspectiveSection]:
        """Build PerspectiveSection objects from agent results."""
        sections = []

        if intuitive:
            content_parts = []
            if intuitive.get('simplest_example'):
                content_parts.append(intuitive['simplest_example'])
            if intuitive.get('step_by_step_build'):
                content_parts.append(intuitive['step_by_step_build'])
            sections.append(PerspectiveSection(
                perspective=PerspectiveType.INTUITIVE_VISUAL,
                title="See It Clearly",
                content='\n\n'.join(content_parts),
                key_takeaway=intuitive.get('aha_moment', ''),
                visual_description='\n'.join(intuitive.get('visual_descriptions', [])),
            ))

        if framework:
            content_parts = []
            if framework.get('decision_tree'):
                content_parts.append(framework['decision_tree'])
            if framework.get('comparison_matrix'):
                content_parts.append(framework['comparison_matrix'])
            sections.append(PerspectiveSection(
                perspective=PerspectiveType.STRUCTURED_FRAMEWORK,
                title="Think It Through",
                content='\n\n'.join(content_parts),
                key_takeaway='\n'.join(framework.get('key_principles', [])[:2]),
            ))

        if story:
            sections.append(PerspectiveSection(
                perspective=PerspectiveType.STORYTELLING,
                title="Feel the Story",
                content=story.get('story', ''),
                key_takeaway=story.get('moral_or_lesson', ''),
            ))

        if debate:
            content_parts = []
            if debate.get('central_question'):
                content_parts.append(f"**The Big Question:** {debate['central_question']}")
            if debate.get('form_your_opinion'):
                content_parts.append(debate['form_your_opinion'])
            sections.append(PerspectiveSection(
                perspective=PerspectiveType.DEBATE_CRITICAL,
                title="Debate It",
                content='\n\n'.join(content_parts),
                key_takeaway=debate.get('form_your_opinion', ''),
            ))

        if project:
            content_parts = []
            if project.get('poster_design_brief'):
                content_parts.append(project['poster_design_brief'])
            if project.get('role_play_scenario'):
                content_parts.append(project['role_play_scenario'])
            sections.append(PerspectiveSection(
                perspective=PerspectiveType.HANDS_ON_PROJECT,
                title="Build It",
                content='\n\n'.join(content_parts),
                key_takeaway=project.get('reflection_activity', ''),
                activity=project.get('poster_design_brief', ''),
            ))

        if realworld:
            content_parts = []
            if realworld.get('future_impact'):
                content_parts.append(realworld['future_impact'])
            sections.append(PerspectiveSection(
                perspective=PerspectiveType.REAL_WORLD_APPLICATION,
                title="Live It",
                content='\n\n'.join(content_parts),
                key_takeaway=realworld.get('future_impact', ''),
            ))

        return sections

    def _build_language_sections(
        self, language_content: Dict[Language, Dict]
    ) -> List[LanguageContent]:
        """Build LanguageContent objects from multilingual agent results."""
        sections = []
        for lang, data in language_content.items():
            if not data:
                continue
            sections.append(LanguageContent(
                language=lang,
                summary=data.get('summary', ''),
                key_vocabulary=data.get('key_vocabulary', []),
                reflection_prompts=data.get('reflection_prompts', []),
                activity=data.get('activity', ''),
                slogans=data.get('slogans', []),
            ))
        return sections

    # =========================================================================
    # PDF & HTML GENERATION
    # =========================================================================

    async def _generate_pdf(
        self,
        topic: str,
        student_name: str,
        content: LessonContent,
    ) -> Optional[str]:
        """Generate professional PDF from lesson content."""
        try:
            from .pdf_generator import generate_perspective_pdf

            safe_topic = topic.replace(' ', '_').replace(':', '').replace(',', '')[:50]
            output_path = f"/tmp/perspective_{safe_topic}_{student_name.replace(' ', '_')}_lesson.pdf"

            pdf_path = await generate_perspective_pdf(
                content=content,
                output_path=output_path,
                celebration_word=self.config.celebration_word,
            )

            if pdf_path:
                logger.info(f"Generated PDF: {pdf_path}")
            return pdf_path

        except Exception as e:
            logger.warning(f"PDF generation failed: {e}")
            return None

    async def _generate_html(
        self,
        topic: str,
        student_name: str,
        content: LessonContent,
    ) -> Optional[str]:
        """Generate interactive HTML from lesson content."""
        try:
            from .pdf_generator import generate_perspective_html

            safe_topic = topic.replace(' ', '_').replace(':', '').replace(',', '')[:50]
            output_path = f"/tmp/perspective_{safe_topic}_{student_name.replace(' ', '_')}_slides.html"

            html_path = await generate_perspective_html(
                content=content,
                output_path=output_path,
                celebration_word=self.config.celebration_word,
            )

            if html_path:
                logger.info(f"Generated HTML: {html_path}")
            return html_path

        except Exception as e:
            logger.warning(f"HTML generation failed: {e}")
            return None

    # =========================================================================
    # TELEGRAM
    # =========================================================================

    async def _send_to_telegram(self, topic: str, student_name: str, content: LessonContent, full_content: str, pdf_path: Optional[str] = None, html_path: Optional[str] = None) -> Any:
        """Send lesson summary + PDF + HTML to Telegram."""
        if not TELEGRAM_AVAILABLE:
            logger.warning("Telegram tools not available")
            return

        try:
            has_pdf = pdf_path and Path(pdf_path).exists()
            has_html = html_path and Path(html_path).exists()
            files_label = ""
            if has_pdf:
                files_label += " + PDF"
            if has_html:
                files_label += " + HTML"

            # Summary message
            header = f"*Lesson: {topic}*\n"
            header += f"*Student:* {student_name}\n\n"

            stats = f"*{len(content.perspectives)} perspectives* | "
            stats += f"*{len(content.language_sections)} languages* | "
            stats += f"*{len(content.key_concepts)} concepts*\n"
            stats += f"*{content.total_words} words*{files_label}\n\n"

            insights = ""
            if content.key_insights:
                insights = f"*Key Insights ({self.config.celebration_word})*\n"
                for i, ins in enumerate(content.key_insights[:4], 1):
                    if ins:
                        insights += f"{i}. {ins[:150]}\n"
                insights += "\n"

            message = header + stats + insights
            if len(message) > 4000:
                message = message[:3950] + "\n..."

            result = await send_telegram_message_tool({
                'message': message, 'parse_mode': 'Markdown'
            })

            if result.get('success'):
                logger.info(f"Sent summary to Telegram")
            else:
                logger.error(f"Telegram message failed: {result.get('error')}")

            # Send PDF
            if has_pdf:
                file_result = await send_telegram_file_tool({
                    'file_path': pdf_path,
                    'caption': f"{topic} - Learning Guide for {student_name}"
                })
                if file_result.get('success'):
                    logger.info("Sent PDF to Telegram")

            # Send HTML
            if has_html:
                file_result = await send_telegram_file_tool({
                    'file_path': html_path,
                    'caption': f"{topic} - Interactive Slides for {student_name}"
                })
                if file_result.get('success'):
                    logger.info("Sent HTML to Telegram")

            # Fallback: send markdown if no PDF
            if not has_pdf:
                safe_topic = topic.replace(' ', '_').replace(':', '')[:40]
                temp_path = Path(f"/tmp/perspective_{safe_topic}_{student_name.replace(' ', '_')}.md")
                with open(temp_path, 'w') as f:
                    f.write(full_content)

                file_result = await send_telegram_file_tool({
                    'file_path': str(temp_path),
                    'caption': f"{topic} - Lesson for {student_name} (Markdown)"
                })

                try:
                    temp_path.unlink()
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # GOLD STANDARDS
    # =========================================================================

    def seed_gold_standards(self) -> None:
        """Seed default gold standards for evaluation."""
        self._init_agents()
        self.add_gold_standard(
            task_type='perspective_teaching',
            input_data={'topic': 'any'},
            expected_output={
                'perspectives_count': 6,
                'languages_count': 3,
                'has_running_example': True,
                'has_parent_guide': True,
                'word_count': 3000,
            },
            evaluation_criteria={
                'perspectives_count': 0.30,
                'languages_count': 0.20,
                'has_running_example': 0.15,
                'has_parent_guide': 0.15,
                'word_count': 0.20,
            }
        )
        logger.info("Seeded gold standards for PerspectiveLearningSwarm")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def teach_perspectives(
    topic: str,
    student_name: str = "Student",
    age_group: str = "primary",
    depth: str = "standard",
    languages: List[str] = None,
    central_idea: str = "",
    send_telegram: bool = False,
) -> PerspectiveLearningResult:
    """
    One-liner multi-perspective learning.

    Usage:
        from Jotty.core.swarms.perspective_learning_swarm import teach_perspectives

        # Basic
        result = await teach_perspectives("Media and its influence on decisions", student_name="Aria")

        # Full control
        result = await teach_perspectives(
            "Media is a tool that influences the decisions people make",
            student_name="Aria",
            age_group="primary",
            depth="deep",
            send_telegram=True,
        )
    """
    age_enum = AgeGroup(age_group) if age_group in [a.value for a in AgeGroup] else AgeGroup.PRIMARY
    depth_enum = ContentDepth(depth) if depth in [d.value for d in ContentDepth] else ContentDepth.STANDARD

    lang_enums = None
    if languages:
        lang_enums = [Language(l) if l in [la.value for la in Language] else Language.ENGLISH for l in languages]

    config = PerspectiveLearningConfig(
        student_name=student_name,
        age_group=age_enum,
        depth=depth_enum,
    )
    if lang_enums:
        config.languages = lang_enums

    swarm = PerspectiveLearningSwarm(config)
    return await swarm.teach(topic=topic, central_idea=central_idea, send_telegram=send_telegram)


def teach_perspectives_sync(
    topic: str,
    student_name: str = "Student",
    age_group: str = "primary",
    depth: str = "standard",
    languages: List[str] = None,
    central_idea: str = "",
    send_telegram: bool = False,
) -> PerspectiveLearningResult:
    """Synchronous version of teach_perspectives."""
    return asyncio.run(teach_perspectives(
        topic=topic, student_name=student_name, age_group=age_group,
        depth=depth, languages=languages, central_idea=central_idea,
        send_telegram=send_telegram,
    ))


__all__ = [
    'PerspectiveLearningSwarm',
    'PerspectiveLearningConfig',
    'PerspectiveLearningResult',
    'teach_perspectives',
    'teach_perspectives_sync',
]
