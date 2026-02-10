"""Olympiad Learning Swarm - Main swarm orchestrator.

World-class educational swarm that takes any subject + topic and creates
comprehensive learning content from building blocks to olympiad mastery.

Teaching Philosophy (embedded in every agent):
- Start with WHY before WHAT
- Build from the simplest possible concrete example
- Every formula earns its place by solving a real problem
- Pattern recognition is the key to competition success
- Progressive difficulty: foundation -> intermediate -> advanced -> olympiad
- Celebrate breakthrough moments
- Personalized for the student

Usage:
    # Quick start
    result = await learn_topic("mathematics", "Number Theory", "Aria")

    # Full control
    swarm = OlympiadLearningSwarm(OlympiadLearningConfig(
        subject=Subject.MATHEMATICS,
        student_name="Aria",
        depth=LessonDepth.DEEP,
        target_tier=DifficultyTier.OLYMPIAD,
    ))
    result = await swarm.teach(topic="Number Theory")
"""

import asyncio
import logging
import json
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import dspy

from ..base_swarm import (
    BaseSwarm, SwarmConfig, SwarmResult, AgentRole,
    register_swarm, ExecutionTrace
)
from ..base import DomainSwarm, AgentTeam

from .types import (
    Subject, DifficultyTier, LessonDepth, TeachingMode,
    OlympiadLearningConfig, BuildingBlock, ConceptCore,
    PatternEntry, Problem, StrategyCard, MistakeEntry,
    LessonSection, LessonContent, OlympiadLearningResult,
    format_steps_on_newlines, tier_to_level,
)
from .agents import (
    BaseOlympiadAgent, CurriculumArchitectAgent, ConceptDecomposerAgent,
    IntuitionBuilderAgent, PatternHunterAgent, ProblemCrafterAgent,
    SolutionStrategistAgent, MistakeAnalyzerAgent, ConnectionMapperAgent,
    ContentAssemblerAgent, UnifiedTopicAgent, NarrativeEditorAgent,
    RankTipsAgent,
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


@register_swarm("olympiad_learning")
class OlympiadLearningSwarm(DomainSwarm):
    """
    World-Class Olympiad Learning Swarm.

    Creates comprehensive, progressive learning content that builds
    from absolute basics to international olympiad level.

    10 Specialized Agents:
    1. CurriculumArchitect - Designs the learning roadmap
    2. ConceptDecomposer - Breaks concepts into building blocks
    3. IntuitionBuilder - Builds deep understanding through analogy
    4. PatternHunter - Identifies competition patterns
    5. ProblemCrafter - Creates progressive problem sets
    6. SolutionStrategist - Teaches problem-solving strategies
    7. MistakeAnalyzer - Identifies common pitfalls
    8. ConnectionMapper - Maps inter-topic connections
    9. ContentAssembler - Assembles everything into flowing lesson
    10. UnifiedTopicAgent - Single-pass deep content generation

    Supports: Mathematics, Physics, Chemistry, CS, Biology, Astronomy
    """

    AGENT_TEAM = AgentTeam.define(
        (CurriculumArchitectAgent, "CurriculumArchitect", "_curriculum_architect"),
        (ConceptDecomposerAgent, "ConceptDecomposer", "_concept_decomposer"),
        (IntuitionBuilderAgent, "IntuitionBuilder", "_intuition_builder"),
        (PatternHunterAgent, "PatternHunter", "_pattern_hunter"),
        (ProblemCrafterAgent, "ProblemCrafter", "_problem_crafter"),
        (SolutionStrategistAgent, "SolutionStrategist", "_solution_strategist"),
        (MistakeAnalyzerAgent, "MistakeAnalyzer", "_mistake_analyzer"),
        (ConnectionMapperAgent, "ConnectionMapper", "_connection_mapper"),
        (ContentAssemblerAgent, "ContentAssembler", "_content_assembler"),
        (UnifiedTopicAgent, "UnifiedTopic", "_unified_topic"),
        (NarrativeEditorAgent, "NarrativeEditor", "_narrative_editor"),
        (RankTipsAgent, "RankTips", "_rank_tips"),
    )

    def __init__(self, config: OlympiadLearningConfig = None):
        super().__init__(config or OlympiadLearningConfig())
        self._optimization_mode = self.config.optimization_mode

    async def _execute_domain(
        self,
        topic: str = None,
        **kwargs
    ) -> OlympiadLearningResult:
        """Execute learning content generation."""
        return await self.teach(topic=topic, **kwargs)

    async def teach(
        self,
        topic: str,
        subject: Subject = None,
        student_name: str = None,
        depth: LessonDepth = None,
        target_tier: DifficultyTier = None,
        send_telegram: bool = None,
    ) -> OlympiadLearningResult:
        """
        Create world-class learning content for a topic.

        Args:
            topic: The topic to teach (e.g., "Number Theory", "Combinatorics")
            subject: Override subject from config
            student_name: Override student name from config
            depth: Override lesson depth from config
            target_tier: Override target tier from config
            send_telegram: Whether to send to Telegram

        Returns:
            OlympiadLearningResult with complete lesson content
        """
        start_time = datetime.now()
        self._init_agents()

        config = self.config
        subject = subject or config.subject
        student_name = student_name or config.student_name
        depth = depth or config.depth
        target_tier = target_tier or config.target_tier
        subject_str = subject.value if isinstance(subject, Subject) else str(subject)
        target_str = target_tier.value if isinstance(target_tier, DifficultyTier) else str(target_tier)

        logger.info(f"OlympiadLearningSwarm starting: {topic} ({subject_str}) for {student_name}")

        try:
            if self._optimization_mode == "parallel_deep":
                result = await self._execute_parallel_deep(
                    topic, subject_str, student_name, depth, target_str, config
                )
            elif self._optimization_mode == "unified":
                result = await self._execute_unified(
                    topic, subject_str, student_name, depth, target_str, config
                )
            else:
                result = await self._execute_sequential(
                    topic, subject_str, student_name, depth, target_str, config
                )

            # Build final result
            exec_time = (datetime.now() - start_time).total_seconds()
            content = result.get('content')

            # Estimate learning time
            if depth == LessonDepth.QUICK:
                learning_time = "15-20 minutes"
            elif depth == LessonDepth.STANDARD:
                learning_time = "45-60 minutes"
            elif depth == LessonDepth.DEEP:
                learning_time = "90-120 minutes"
            else:
                learning_time = "2-3 hours"

            total_problems = len(content.problems) if content else 0
            breakthrough_count = sum(1 for s in content.sections if s.has_breakthrough_moment) if content else 0

            final_result = OlympiadLearningResult(
                success=True,
                swarm_name=self.config.name,
                domain=self.config.domain,
                output={
                    'topic': topic,
                    'subject': subject_str,
                    'student': student_name,
                    'content_preview': result.get('complete_content', '')[:500]
                },
                execution_time=exec_time,
                content=content,
                student_name=student_name,
                topic=topic,
                subject=subject,
                learning_time_estimate=learning_time,
                concepts_covered=len(content.core_concepts) if content else 0,
                problems_generated=total_problems,
                breakthrough_moments=breakthrough_count,
                difficulty_progression=[s.level for s in content.sections] if content else [],
            )

            logger.info(f"OlympiadLearningSwarm complete: {topic}")
            logger.info(f"  {len(content.core_concepts) if content else 0} concepts, "
                        f"{total_problems} problems, "
                        f"{breakthrough_count} breakthrough moments")

            # =============================================================
            # GENERATE OUTPUTS (PDF + HTML in parallel)
            # =============================================================
            pdf_path = None
            html_path = None

            if content:
                logger.info("Generating outputs (PDF, HTML)...")

                async def gen_pdf():
                    if config.generate_pdf:
                        return await self._generate_pdf(topic, student_name, content, learning_time)
                    return None

                async def gen_html():
                    if config.generate_html:
                        return await self._generate_html(topic, student_name, content, learning_time)
                    return None

                output_results = await asyncio.gather(
                    gen_pdf(), gen_html(), return_exceptions=True
                )

                pdf_path = output_results[0] if not isinstance(output_results[0], Exception) else None
                html_path = output_results[1] if not isinstance(output_results[1], Exception) else None

                for i, r in enumerate(output_results):
                    if isinstance(r, Exception):
                        logger.warning(f"Output generation {i} failed: {r}")

            final_result.pdf_path = pdf_path
            final_result.html_path = html_path

            # ── Live cost summary ──
            try:
                from Jotty.core.foundation.direct_anthropic_lm import get_cost_tracker
                tracker = get_cost_tracker()
                metrics = tracker.get_metrics()
                logger.info(
                    f"💰 COST SUMMARY: ${metrics.total_cost:.4f} | "
                    f"{metrics.total_calls} calls | "
                    f"{metrics.total_input_tokens}+{metrics.total_output_tokens} tokens | "
                    f"avg ${metrics.avg_cost_per_call:.4f}/call"
                )
                for model, cost in metrics.cost_by_model.items():
                    calls = metrics.calls_by_model.get(model, 0)
                    logger.info(f"  {model}: ${cost:.4f} ({calls} calls)")
            except Exception as cost_err:
                logger.debug(f"Cost summary unavailable: {cost_err}")

            # Self-improvement learning
            await self._post_execute_learning(
                success=True,
                execution_time=exec_time,
                tools_used=self._get_active_tools(['curriculum', 'decompose', 'intuition', 'patterns', 'problems']),
                task_type='olympiad_teaching',
                output_data={
                    'concepts_count': len(content.core_concepts) if content else 0,
                    'problems_count': total_problems,
                    'breakthrough_moments': breakthrough_count,
                    'word_count': content.total_words if content else 0,
                },
                input_data={'topic': topic, 'subject': subject_str}
            )

            # =============================================================
            # SEND TO TELEGRAM (summary + PDF + HTML)
            # =============================================================
            should_send = send_telegram if send_telegram is not None else config.send_telegram
            if should_send and TELEGRAM_AVAILABLE and content:
                logger.info("Sending to Telegram...")
                await self._send_to_telegram(
                    topic, student_name, content,
                    result.get('complete_content', ''),
                    pdf_path=pdf_path,
                    html_path=html_path,
                )

            return final_result

        except Exception as e:
            logger.error(f"OlympiadLearningSwarm error: {e}")
            import traceback
            traceback.print_exc()

            exec_time = (datetime.now() - start_time).total_seconds()
            await self._post_execute_learning(
                success=False, execution_time=exec_time,
                tools_used=[], task_type='olympiad_teaching'
            )

            return OlympiadLearningResult(
                success=False,
                swarm_name=self.config.name,
                domain=self.config.domain,
                output={},
                execution_time=exec_time,
                error=str(e)
            )

    # =========================================================================
    # PARALLEL DEEP MODE - Best quality + speed
    # =========================================================================

    async def _execute_parallel_deep(
        self,
        topic: str,
        subject: str,
        student_name: str,
        depth: LessonDepth,
        target_level: str,
        config: OlympiadLearningConfig
    ) -> Dict[str, Any]:
        """Execute with parallel deep generation - best quality."""
        logger.info("Phase 1: Curriculum Architecture...")

        # Phase 1: Curriculum design (must be first - defines structure)
        curriculum = await self._curriculum_architect.design(
            subject=subject,
            topic=topic,
            student_name=student_name,
            target_level=target_level
        )
        building_blocks = curriculum.get('building_blocks', [])

        # Extract concept names for parallel generation
        concept_names = curriculum.get('learning_sequence', [topic])
        concepts_str = ', '.join(concept_names[:5])

        # Extract planner's running example (canonical — used by all downstream agents)
        planner_running_example = curriculum.get('running_example_scenario', '')
        section_depth_plan = curriculum.get('section_depth_plan', '')
        if planner_running_example:
            logger.info(f"Planner running example: {planner_running_example[:80]}...")

        # Phase 2: Parallel generation of all independent components
        logger.info(f"Phase 2: Parallel deep generation ({len(concept_names)} concepts)...")

        semaphore = asyncio.Semaphore(config.max_concurrent_llm)

        async def rate_limited(coro):
            async with semaphore:
                return await coro

        # Build enriched descriptions using planner's depth plan
        concept_desc = f"{topic} - core concepts for {subject} preparation"
        if section_depth_plan:
            concept_desc += f". Content depth plan: {section_depth_plan}"

        # Run decomposition, intuition, patterns, strategies, mistakes, connections in parallel
        phase2_tasks = [
            rate_limited(self._concept_decomposer.decompose(
                concept_name=topic,
                concept_description=concept_desc,
                subject=subject,
                prerequisites=', '.join(b.name for b in building_blocks[:3]) if building_blocks else "Basic knowledge",
                student_name=student_name
            )),
            rate_limited(self._intuition_builder.build(
                concept=f"{topic}: Core concepts in {subject}",
                why_it_matters=f"Essential for international {subject} success",
                student_name=student_name,
                audience_level="intermediate",
                subject=subject
            )),
            rate_limited(self._pattern_hunter.hunt(
                topic=topic,
                subject=subject,
                concepts=concepts_str,
                target_level=target_level
            )),
            rate_limited(self._solution_strategist.strategize(
                topic=topic,
                subject=subject,
                concepts=concepts_str,
                target_level=target_level
            )),
            rate_limited(self._mistake_analyzer.analyze(
                topic=topic,
                subject=subject,
                concepts=concepts_str,
                running_example=planner_running_example or f"A real-world scenario involving {topic}",
                target_level=target_level
            )),
            rate_limited(self._connection_mapper.map_connections(
                topic=topic,
                subject=subject,
                concepts=concepts_str
            )),
        ]

        results = await asyncio.gather(*phase2_tasks, return_exceptions=True)

        decomposition = results[0] if not isinstance(results[0], Exception) else {}
        intuition = results[1] if not isinstance(results[1], Exception) else {}
        pattern_result = results[2] if not isinstance(results[2], Exception) else {'patterns': []}
        strategy_result = results[3] if not isinstance(results[3], Exception) else {'strategies': []}
        mistake_result = results[4] if not isinstance(results[4], Exception) else {'mistakes': []}
        connection_result = results[5] if not isinstance(results[5], Exception) else {}

        # Phase 3: Problem generation (parallel across tiers)
        logger.info("Phase 3: Crafting progressive problem sets...")

        # Include full pattern fields so problems can reference patterns properly
        patterns_str = json.dumps([{
            'name': p.name, 'description': p.description,
            'when_to_use': p.when_to_use, 'template': p.template,
            'trigger': p.example_trigger
        } for p in pattern_result.get('patterns', [])])

        # Running example: prefer planner's (richer, designed for threading), fallback to intuition
        running_example = planner_running_example or intuition.get('real_world_hook', f"A real-world scenario involving {topic}")

        tier_configs = [
            ("foundation", config.foundation_problems),
            ("intermediate", config.intermediate_problems),
            ("advanced", config.advanced_problems),
            ("olympiad", config.olympiad_problems),
        ]

        # Split into 1-per-call parallel requests to avoid JSON truncation
        # from Haiku hitting output token limits on large batches
        problem_tasks = []
        for tier, count in tier_configs:
            for _ in range(count):
                problem_tasks.append(
                    rate_limited(self._problem_crafter.craft(
                        topic=topic, subject=subject, concepts=concepts_str,
                        patterns=patterns_str, running_example=running_example,
                        tier=tier, count=1, student_name=student_name
                    ))
                )

        logger.info(f"Phase 3: {len(problem_tasks)} individual problem calls in parallel...")
        problem_results = await asyncio.gather(*problem_tasks, return_exceptions=True)

        all_problems = []
        for result in problem_results:
            if isinstance(result, list):
                all_problems.extend(result)

        # Phase 4: Assemble content
        logger.info("Phase 4: Assembling world-class lesson...")

        complete_content = self._build_complete_content(
            student_name=student_name,
            topic=topic,
            subject=subject,
            building_blocks=building_blocks,
            decomposition=decomposition,
            intuition=intuition,
            patterns=pattern_result.get('patterns', []),
            strategies=strategy_result.get('strategies', []),
            problems=all_problems,
            mistakes=mistake_result.get('mistakes', []),
            connections=connection_result,
            config=config,
            meta_strategy=pattern_result.get('meta_strategy', ''),
            speed_techniques=strategy_result.get('speed_techniques', []),
            stuck_toolkit=strategy_result.get('stuck_toolkit', []),
        )

        # Phase 5: Narrative editing + Rank tips (in parallel)
        logger.info("Phase 5: Narrative editing + Rank tips generation...")

        key_insights = [
            decomposition.get('key_insight', ''),
            intuition.get('aha_moment', ''),
            pattern_result.get('meta_strategy', ''),
        ]
        pattern_names = [p.name for p in pattern_result.get('patterns', [])]
        patterns_summary = ', '.join(pattern_names) if pattern_names else topic
        mistakes_summary = ', '.join(
            m.description for m in mistake_result.get('mistakes', [])[:5]
        ) or f"Common mistakes in {topic}"

        # Run narrative editing and rank tips in parallel
        phase5_results = await asyncio.gather(
            self._narrative_editor.edit(
                assembled_content=complete_content,
                running_example=running_example,
                key_insights=key_insights,
                pattern_names=pattern_names,
                student_name=student_name,
                topic=topic,
            ),
            self._rank_tips.generate(
                topic=topic,
                subject=subject,
                target_level=target_level,
                student_name=student_name,
                patterns_summary=patterns_summary,
                mistakes_summary=mistakes_summary,
            ),
            return_exceptions=True,
        )

        narrative_result = phase5_results[0] if not isinstance(phase5_results[0], Exception) else {}
        rank_tips = phase5_results[1] if not isinstance(phase5_results[1], Exception) else []
        if isinstance(rank_tips, Exception):
            logger.warning(f"Rank tips generation failed: {rank_tips}")
            rank_tips = []

        # Accept edited content only if length > 50% of original (safety check)
        edited = narrative_result.get('edited_content', '') if isinstance(narrative_result, dict) else ''
        if edited and len(edited) > len(complete_content) * 0.5:
            complete_content = edited
            logger.info(f"Phase 5: Accepted edited content ({len(edited)} chars)")
        else:
            logger.warning(f"Phase 5: Rejected edited content (too short: {len(edited)} vs {len(complete_content)})")

        # Append rank tips section to content
        if rank_tips:
            complete_content += self._build_rank_tips_section(student_name, topic, rank_tips)
            logger.info(f"Phase 5: Added {len(rank_tips)} rank tips")

        # Build sections for the LessonContent
        sections = self._build_sections(
            topic, student_name, building_blocks, decomposition,
            intuition, pattern_result, strategy_result, all_problems,
            mistake_result, connection_result, config
        )

        # Core concepts from decomposition
        core_concepts = [ConceptCore(
            name=topic,
            description=decomposition.get('formal_definition', ''),
            why_it_matters=f"Essential for {subject} olympiad",
            real_world_hook=intuition.get('real_world_hook', ''),
            prerequisites=[b.name for b in building_blocks],
            difficulty=4,
            key_insight=decomposition.get('key_insight', ''),
            common_misconceptions=decomposition.get('common_misconceptions', []),
            related_topics=connection_result.get('direct_connections', []),
        )]

        content = LessonContent(
            subject=Subject(subject) if subject in [s.value for s in Subject] else Subject.GENERAL,
            topic=topic,
            student_name=student_name,
            building_blocks=building_blocks,
            core_concepts=core_concepts,
            patterns=pattern_result.get('patterns', []),
            strategies=strategy_result.get('strategies', []),
            problems=all_problems,
            mistakes=mistake_result.get('mistakes', []),
            sections=sections,
            key_insights=key_insights,
            summary=f"Complete {topic} lesson for {student_name}: from building blocks to olympiad mastery.",
            next_topics=connection_result.get('next_topics', []),
            total_words=len(complete_content.split()),
            competition_tips=strategy_result.get('speed_techniques', []),
            connections=connection_result.get('surprising_connections', []),
            running_example=running_example,
            rank_tips=rank_tips if isinstance(rank_tips, list) else [],
        )

        return {
            'content': content,
            'complete_content': complete_content,
        }

    # =========================================================================
    # UNIFIED MODE - Fast single-pass
    # =========================================================================

    async def _execute_unified(
        self,
        topic: str,
        subject: str,
        student_name: str,
        depth: LessonDepth,
        target_level: str,
        config: OlympiadLearningConfig
    ) -> Dict[str, Any]:
        """Execute with unified single-pass generation."""
        logger.info("Unified mode: Generating deep content in single pass...")

        deep_content = await self._unified_topic.generate_deep(
            student_name=student_name,
            topic=topic,
            subject=subject,
            target_level=target_level,
            celebration_word=config.celebration_word
        )

        complete_content = self._build_unified_content(
            student_name, topic, subject, deep_content, config
        )

        sections = [
            LessonSection(title="Introduction", content=deep_content.get('hook', ''), level=1, has_breakthrough_moment=False),
            LessonSection(title="Building Blocks", content=deep_content.get('building_blocks', ''), level=1),
            LessonSection(title="Discovery", content=deep_content.get('concept_discovery', ''), level=2, has_breakthrough_moment=True),
            LessonSection(title="Formal Framework", content=deep_content.get('formal_definition', ''), level=3),
            LessonSection(title="Patterns", content=deep_content.get('patterns', ''), level=3),
            LessonSection(title="Strategies", content=deep_content.get('strategies', ''), level=3),
            LessonSection(title="Foundation Problems", content=deep_content.get('problems_foundation', ''), level=2),
            LessonSection(title="Intermediate Problems", content=deep_content.get('problems_intermediate', ''), level=3),
            LessonSection(title="Advanced Problems", content=deep_content.get('problems_advanced', ''), level=4, has_breakthrough_moment=True),
            LessonSection(title="Common Mistakes", content=deep_content.get('common_mistakes', ''), level=3),
            LessonSection(title="Connections", content=deep_content.get('connections', ''), level=3),
        ]

        content = LessonContent(
            subject=Subject(subject) if subject in [s.value for s in Subject] else Subject.GENERAL,
            topic=topic,
            student_name=student_name,
            building_blocks=[],
            core_concepts=[ConceptCore(
                name=topic,
                description=deep_content.get('formal_definition', topic),
                why_it_matters=f"Essential for {subject} olympiad",
                real_world_hook=deep_content.get('hook', ''),
                prerequisites=[],
                difficulty=4,
                key_insight=deep_content.get('key_insight', ''),
            )],
            patterns=[],
            strategies=[],
            problems=[],
            mistakes=[],
            sections=sections,
            key_insights=[deep_content.get('key_insight', '')],
            summary=f"Complete {topic} lesson for {student_name}.",
            next_topics=[],
            total_words=len(complete_content.split()),
            competition_tips=[deep_content.get('competition_tips', '')],
        )

        return {'content': content, 'complete_content': complete_content}

    # =========================================================================
    # SEQUENTIAL MODE - Original step-by-step
    # =========================================================================

    async def _execute_sequential(
        self,
        topic: str,
        subject: str,
        student_name: str,
        depth: LessonDepth,
        target_level: str,
        config: OlympiadLearningConfig
    ) -> Dict[str, Any]:
        """Execute agents sequentially with full control."""
        logger.info("Sequential mode: Step-by-step generation...")

        # Step 1: Curriculum
        logger.info("Step 1/8: Designing curriculum...")
        curriculum = await self._curriculum_architect.design(
            subject=subject, topic=topic,
            student_name=student_name, target_level=target_level
        )
        building_blocks = curriculum.get('building_blocks', [])
        concepts_str = ', '.join(curriculum.get('learning_sequence', [topic])[:5])
        await asyncio.sleep(0.5)

        # Step 2: Decompose
        logger.info("Step 2/8: Decomposing concept...")
        decomposition = await self._concept_decomposer.decompose(
            concept_name=topic,
            concept_description=f"{topic} in {subject}",
            subject=subject,
            prerequisites=', '.join(b.name for b in building_blocks[:3]) if building_blocks else "Basic knowledge",
            student_name=student_name
        )
        await asyncio.sleep(0.5)

        # Step 3: Build intuition
        logger.info("Step 3/8: Building intuition...")
        intuition = await self._intuition_builder.build(
            concept=f"{topic}: {subject} olympiad",
            why_it_matters=f"Essential for international {subject} olympiad",
            student_name=student_name,
            audience_level="intermediate",
            subject=subject
        )
        await asyncio.sleep(0.5)

        # Step 4: Hunt patterns
        logger.info("Step 4/8: Hunting patterns...")
        pattern_result = await self._pattern_hunter.hunt(
            topic=topic, subject=subject,
            concepts=concepts_str, target_level=target_level
        )
        await asyncio.sleep(0.5)

        # Step 5: Strategies
        logger.info("Step 5/8: Building strategies...")
        strategy_result = await self._solution_strategist.strategize(
            topic=topic, subject=subject,
            concepts=concepts_str, target_level=target_level
        )
        await asyncio.sleep(0.5)

        # Step 6: Problems
        logger.info("Step 6/8: Crafting problems...")
        all_problems = []
        patterns_str = json.dumps([{
            'name': p.name, 'description': p.description,
            'when_to_use': p.when_to_use, 'template': p.template,
            'trigger': p.example_trigger
        } for p in pattern_result.get('patterns', [])])
        # Prefer planner's running example (richer), fallback to intuition
        seq_running_example = curriculum.get('running_example_scenario', '') or intuition.get('real_world_hook', f"A scenario involving {topic}")

        for tier, count in [("foundation", config.foundation_problems),
                            ("intermediate", config.intermediate_problems),
                            ("advanced", config.advanced_problems),
                            ("olympiad", config.olympiad_problems)]:
            # Generate 1 problem per call to avoid JSON truncation
            for _ in range(count):
                problems = await self._problem_crafter.craft(
                    topic=topic, subject=subject, concepts=concepts_str,
                    patterns=patterns_str, running_example=seq_running_example,
                    tier=tier, count=1, student_name=student_name
                )
                all_problems.extend(problems)
                await asyncio.sleep(0.3)

        # Step 7: Mistakes
        logger.info("Step 7/8: Analyzing mistakes...")
        mistake_result = await self._mistake_analyzer.analyze(
            topic=topic, subject=subject,
            concepts=concepts_str,
            running_example=seq_running_example,
            target_level=target_level
        )
        await asyncio.sleep(0.5)

        # Step 8: Connections
        logger.info("Step 8/8: Mapping connections...")
        connection_result = await self._connection_mapper.map_connections(
            topic=topic, subject=subject, concepts=concepts_str
        )

        # Assemble
        logger.info("Assembling final lesson...")
        complete_content = self._build_complete_content(
            student_name=student_name, topic=topic, subject=subject,
            building_blocks=building_blocks, decomposition=decomposition,
            intuition=intuition, patterns=pattern_result.get('patterns', []),
            strategies=strategy_result.get('strategies', []),
            problems=all_problems, mistakes=mistake_result.get('mistakes', []),
            connections=connection_result, config=config,
            meta_strategy=pattern_result.get('meta_strategy', ''),
            speed_techniques=strategy_result.get('speed_techniques', []),
            stuck_toolkit=strategy_result.get('stuck_toolkit', []),
        )

        sections = self._build_sections(
            topic, student_name, building_blocks, decomposition,
            intuition, pattern_result, strategy_result, all_problems,
            mistake_result, connection_result, config
        )

        core_concepts = [ConceptCore(
            name=topic,
            description=decomposition.get('formal_definition', ''),
            why_it_matters=f"Essential for {subject} olympiad",
            real_world_hook=intuition.get('real_world_hook', ''),
            prerequisites=[b.name for b in building_blocks],
            difficulty=4,
            key_insight=decomposition.get('key_insight', ''),
            common_misconceptions=decomposition.get('common_misconceptions', []),
            related_topics=connection_result.get('direct_connections', []),
        )]

        content = LessonContent(
            subject=Subject(subject) if subject in [s.value for s in Subject] else Subject.GENERAL,
            topic=topic, student_name=student_name,
            building_blocks=building_blocks, core_concepts=core_concepts,
            patterns=pattern_result.get('patterns', []),
            strategies=strategy_result.get('strategies', []),
            problems=all_problems, mistakes=mistake_result.get('mistakes', []),
            sections=sections,
            key_insights=[decomposition.get('key_insight', ''), intuition.get('aha_moment', '')],
            summary=f"Complete {topic} lesson for {student_name}.",
            next_topics=connection_result.get('next_topics', []),
            total_words=len(complete_content.split()),
            competition_tips=strategy_result.get('speed_techniques', []),
            connections=connection_result.get('surprising_connections', []),
        )

        return {'content': content, 'complete_content': complete_content}

    # =========================================================================
    # CONTENT BUILDERS
    # =========================================================================

    def _build_complete_content(
        self,
        student_name: str,
        topic: str,
        subject: str,
        building_blocks: List[BuildingBlock],
        decomposition: Dict,
        intuition: Dict,
        patterns: List[PatternEntry],
        strategies: List[StrategyCard],
        problems: List[Problem],
        mistakes: List[MistakeEntry],
        connections: Dict,
        config: OlympiadLearningConfig,
        meta_strategy: str = "",
        speed_techniques: List[str] = None,
        stuck_toolkit: List[str] = None,
    ) -> str:
        """Build complete markdown lesson content."""
        celebration = config.celebration_word
        parts = []

        # Title
        parts.append(f"# {topic} - Olympiad Masterclass")
        parts.append(f"### Personalized for {student_name}\n")

        # Hook
        if intuition.get('real_world_hook'):
            parts.append(f"## Why Should You Care, {student_name}?\n")
            parts.append(f"{intuition['real_world_hook']}\n")

        # Building Blocks
        if building_blocks:
            parts.append(f"\n## Foundation Check\n")
            parts.append(f"Before we dive in, {student_name}, let's make sure these building blocks are solid:\n")
            for i, block in enumerate(building_blocks, 1):
                parts.append(f"### {i}. {block.name}")
                parts.append(f"{block.quick_review}")
                if block.check_question:
                    parts.append(f"\n*Quick check:* {block.check_question}\n")

        # Discovery
        if decomposition.get('simplest_example'):
            parts.append(f"\n*Now that you have the building blocks, let's discover something exciting, {student_name}...*\n")
            parts.append(f"\n## The Discovery\n")
            parts.append(f"### Start Simple\n{decomposition['simplest_example']}\n")

        if decomposition.get('pattern_discovery'):
            parts.append(f"\n### See the Pattern\n{decomposition['pattern_discovery']}\n")

        if decomposition.get('key_insight'):
            parts.append(f"\n**{celebration}** {decomposition['key_insight']}\n")

        # Intuition
        if intuition.get('analogy'):
            parts.append(f"\n## The Big Picture\n")
            parts.append(f"**Think of it like this:** {intuition['analogy']}\n")

        if intuition.get('intuition_build'):
            parts.append(f"\n### Building Understanding\n")
            parts.append(format_steps_on_newlines(intuition['intuition_build']))

        if intuition.get('aha_moment'):
            parts.append(f"\n**{celebration}** {intuition['aha_moment']}\n")

        # Formal Definition
        if decomposition.get('formal_definition'):
            parts.append(f"\n## The Precise Framework\n")
            parts.append(f"Now that you see WHY it works, {student_name}, here's the formal language:\n")
            parts.append(f"{decomposition['formal_definition']}\n")

        if decomposition.get('why_it_works'):
            parts.append(f"\n### Why It Works\n{decomposition['why_it_works']}\n")

        # Pattern Library
        if patterns:
            parts.append(f"\n*Here's where it gets exciting, {student_name}...*\n")
            parts.append(f"\n## Pattern Library\n")
            parts.append(f"These are the patterns that competition winners recognize instantly, {student_name}:\n")
            for i, p in enumerate(patterns, 1):
                parts.append(f"### Pattern {i}: {p.name}")
                parts.append(f"{p.description}")
                if p.when_to_use:
                    parts.append(f"\n**When to use:** {p.when_to_use}")
                if p.example_trigger:
                    parts.append(f"\n**Trigger:** {p.example_trigger}")
                if p.template:
                    parts.append(f"\n**Template:** {p.template}\n")

        if meta_strategy:
            parts.append(f"\n### Meta-Strategy\n{meta_strategy}\n")

        # Strategies
        if strategies:
            parts.append(f"\n## Strategy Toolkit\n")
            for s in strategies:
                parts.append(f"### {s.name}")
                parts.append(f"{s.description}")
                if s.when_to_use:
                    parts.append(f"\n**When:** {s.when_to_use}")
                if s.steps:
                    parts.append("\n**Steps:**")
                    for j, step in enumerate(s.steps, 1):
                        parts.append(f"{j}. {step}")
                if s.example_problem:
                    parts.append(f"\n**Example:** {s.example_problem}")
                if s.example_solution:
                    parts.append(f"\n**Solution:** {s.example_solution}\n")

        if speed_techniques:
            parts.append(f"\n### Speed Techniques\n")
            for t in speed_techniques:
                parts.append(f"- {t}")

        if stuck_toolkit:
            parts.append(f"\n### When You're Stuck\n")
            for i, t in enumerate(stuck_toolkit, 1):
                parts.append(f"{i}. {t}")

        # Problems - Progressive
        if problems:
            hook = intuition.get('real_world_hook', '')
            if hook:
                parts.append(f"\n*Remember our scenario: {hook[:120]}... Many problems connect back to it.*\n")
            parts.append(f"\n## Problem Ladder\n")
            parts.append(f"Time to build mastery, {student_name}! Work through these in order.\n")

            for tier in DifficultyTier:
                tier_problems = [p for p in problems if p.tier == tier]
                if not tier_problems:
                    continue

                tier_labels = {
                    DifficultyTier.FOUNDATION: "Foundation (Warm-up)",
                    DifficultyTier.INTERMEDIATE: "Intermediate (School Competition)",
                    DifficultyTier.ADVANCED: "Advanced (National Level)",
                    DifficultyTier.OLYMPIAD: "Olympiad (International Level)",
                    DifficultyTier.BEYOND: "Beyond (Research Extension)",
                }
                parts.append(f"\n### {tier_labels.get(tier, tier.value)}\n")

                for i, prob in enumerate(tier_problems, 1):
                    parts.append(f"**Problem {i}** ({prob.time_estimate_minutes} min)")
                    parts.append(f"{prob.statement}\n")

                    if prob.hints and config.include_hints:
                        parts.append("<details><summary>Hints</summary>\n")
                        for h_idx, hint in enumerate(prob.hints, 1):
                            parts.append(f"Hint {h_idx}: {hint}")
                        parts.append("</details>\n")

                    if prob.solution and config.include_full_solutions:
                        parts.append("<details><summary>Solution</summary>\n")
                        parts.append(f"{prob.solution}")
                        if prob.key_insight:
                            parts.append(f"\n**Key insight:** {prob.key_insight}")
                        if prob.strategy_used:
                            parts.append(f"\n**Strategy:** {prob.strategy_used}")
                        parts.append("</details>\n")

        # Common Mistakes
        if mistakes:
            parts.append(f"\n## Trap Alert!\n")
            parts.append(f"Watch out for these, {student_name} - they catch even experienced students:\n")
            for i, m in enumerate(mistakes, 1):
                parts.append(f"### Trap {i}: {m.description}")
                parts.append(f"**Why it happens:** {m.why_it_happens}")
                parts.append(f"**Wrong:** {m.example_wrong}")
                parts.append(f"**Correct:** {m.example_correct}")
                parts.append(f"**How to avoid:** {m.how_to_avoid}\n")

        # Connections
        if connections.get('direct_connections') or connections.get('surprising_connections'):
            parts.append(f"\n## Where This Leads\n")
            if connections.get('direct_connections'):
                parts.append("**Builds directly into:**")
                for c in connections['direct_connections']:
                    parts.append(f"- {c}")
            if connections.get('surprising_connections'):
                parts.append(f"\n**Surprising connections:**")
                for c in connections['surprising_connections']:
                    parts.append(f"- {c}")
            if connections.get('powerful_combinations'):
                parts.append(f"\n**Powerful when combined with:**")
                for c in connections['powerful_combinations']:
                    parts.append(f"- {c}")

        # Next steps
        if connections.get('next_topics'):
            parts.append(f"\n## What's Next, {student_name}?\n")
            parts.append("Recommended study order:")
            for i, t in enumerate(connections['next_topics'], 1):
                parts.append(f"{i}. {t}")

        return '\n'.join(parts)

    def _build_unified_content(
        self,
        student_name: str,
        topic: str,
        subject: str,
        deep_content: Dict,
        config: OlympiadLearningConfig
    ) -> str:
        """Build content from unified single-pass result."""
        celebration = config.celebration_word
        parts = []

        parts.append(f"# {topic} - Olympiad Masterclass")
        parts.append(f"### Personalized for {student_name}\n")

        if deep_content.get('hook'):
            parts.append(f"## Why This Matters\n{deep_content['hook']}\n")
        if deep_content.get('building_blocks'):
            parts.append(f"\n## Foundation Check\n{deep_content['building_blocks']}\n")
        if deep_content.get('concept_discovery'):
            parts.append(f"\n## The Discovery\n{deep_content['concept_discovery']}\n")
        if deep_content.get('key_insight'):
            parts.append(f"\n**{celebration}** {deep_content['key_insight']}\n")
        if deep_content.get('formal_definition'):
            parts.append(f"\n## The Precise Framework\n{deep_content['formal_definition']}\n")
        if deep_content.get('patterns'):
            parts.append(f"\n## Pattern Library\n{deep_content['patterns']}\n")
        if deep_content.get('strategies'):
            parts.append(f"\n## Strategy Toolkit\n{deep_content['strategies']}\n")
        if deep_content.get('problems_foundation'):
            parts.append(f"\n## Foundation Problems\n{deep_content['problems_foundation']}\n")
        if deep_content.get('problems_intermediate'):
            parts.append(f"\n## Intermediate Problems\n{deep_content['problems_intermediate']}\n")
        if deep_content.get('problems_advanced'):
            parts.append(f"\n## Advanced & Olympiad Problems\n{deep_content['problems_advanced']}\n")
        if deep_content.get('common_mistakes'):
            parts.append(f"\n## Trap Alert!\n{deep_content['common_mistakes']}\n")
        if deep_content.get('connections'):
            parts.append(f"\n## Where This Leads\n{deep_content['connections']}\n")
        if deep_content.get('competition_tips'):
            parts.append(f"\n## Competition Tips\n{deep_content['competition_tips']}\n")

        return '\n'.join(parts)

    def _build_rank_tips_section(
        self, student_name: str, topic: str, rank_tips: List[str]
    ) -> str:
        """Build the '20-30 Tips to Secure #1 Rank' markdown section."""
        parts = [f"\n\n## Tips to Secure #1 Rank in {topic}\n"]
        parts.append(f"*{student_name}, here are the specific tips that separate the #1 scorer from everyone else:*\n")
        for i, tip in enumerate(rank_tips, 1):
            # Strip existing numbering if the LLM already added it
            clean_tip = re.sub(r'^\d+[\.\)]\s*', '', tip)
            parts.append(f"**{i}.** {clean_tip}\n")
        parts.append(f"\n*Master these {len(rank_tips)} tips and you won't just pass — you'll dominate, {student_name}!*\n")
        return '\n'.join(parts)

    def _build_sections(
        self, topic, student_name, building_blocks, decomposition,
        intuition, pattern_result, strategy_result, problems,
        mistake_result, connection_result, config
    ) -> List[LessonSection]:
        """Build LessonSection objects for structured output."""
        sections = []

        # Hook
        sections.append(LessonSection(
            title=f"Why {topic} Matters",
            content=intuition.get('real_world_hook', f"Let's master {topic}!"),
            level=1, has_breakthrough_moment=False
        ))

        # Building blocks
        if building_blocks:
            blocks_content = '\n'.join(f"- **{b.name}**: {b.quick_review}" for b in building_blocks)
            sections.append(LessonSection(
                title="Foundation Check", content=blocks_content, level=1
            ))

        # Discovery
        if decomposition.get('simplest_example'):
            sections.append(LessonSection(
                title="The Discovery",
                content=f"{decomposition.get('simplest_example', '')}\n\n{decomposition.get('pattern_discovery', '')}",
                level=2, has_breakthrough_moment=bool(decomposition.get('key_insight')),
                breakthrough_content=decomposition.get('key_insight', ''),
                transition_text=f"Now that you have the building blocks, let's discover something exciting, {student_name}...",
            ))

        # Intuition
        if intuition.get('intuition_build'):
            sections.append(LessonSection(
                title="Building Deep Understanding",
                content=f"{intuition.get('analogy', '')}\n\n{intuition.get('intuition_build', '')}",
                level=2, has_breakthrough_moment=bool(intuition.get('aha_moment')),
                breakthrough_content=intuition.get('aha_moment', ''),
            ))

        # Patterns
        if pattern_result.get('patterns'):
            patterns_content = '\n'.join(
                f"**{p.name}**: {p.description} (Trigger: {p.example_trigger})"
                for p in pattern_result['patterns']
            )
            sections.append(LessonSection(
                title="Pattern Library", content=patterns_content, level=3,
                transition_text=f"Here's where it gets exciting, {student_name} — these are the patterns that competition winners recognize instantly...",
            ))

        # Problems by tier
        hook = intuition.get('real_world_hook', '')
        for i, tier in enumerate(DifficultyTier):
            tier_probs = [p for p in problems if p.tier == tier]
            if tier_probs:
                transition = ""
                if i == 0 and hook:
                    transition = f"Remember our scenario? Many problems connect back to it. Time to build mastery, {student_name}!"
                sections.append(LessonSection(
                    title=f"Problems: {tier.value.title()}",
                    content='\n'.join(p.statement for p in tier_probs),
                    level=tier_to_level(tier),
                    problems=tier_probs,
                    transition_text=transition,
                ))

        # Mistakes
        if mistake_result.get('mistakes'):
            sections.append(LessonSection(
                title="Common Traps",
                content='\n'.join(f"- {m.description}: {m.how_to_avoid}" for m in mistake_result['mistakes']),
                level=3
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
        learning_time: str,
    ) -> Optional[str]:
        """Generate professional PDF from lesson content.

        Returns:
            Path to generated PDF, or None on failure.
        """
        try:
            from .pdf_generator import generate_lesson_pdf

            safe_topic = topic.replace(' ', '_').replace(':', '').replace(',', '')[:50]
            output_path = f"/tmp/olympiad_{safe_topic}_{student_name.replace(' ', '_')}_lesson.pdf"

            pdf_path = await generate_lesson_pdf(
                content=content,
                output_path=output_path,
                celebration_word=self.config.celebration_word,
                learning_time=learning_time,
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
        learning_time: str,
    ) -> Optional[str]:
        """Generate interactive HTML from lesson content.

        Returns:
            Path to generated HTML, or None on failure.
        """
        try:
            from .pdf_generator import generate_lesson_html

            safe_topic = topic.replace(' ', '_').replace(':', '').replace(',', '')[:50]
            output_path = f"/tmp/olympiad_{safe_topic}_{student_name.replace(' ', '_')}_slides.html"

            html_path = await generate_lesson_html(
                content=content,
                output_path=output_path,
                celebration_word=self.config.celebration_word,
                learning_time=learning_time,
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

    async def _send_to_telegram(
        self,
        topic: str,
        student_name: str,
        content: LessonContent,
        full_content: str,
        pdf_path: Optional[str] = None,
        html_path: Optional[str] = None,
    ):
        """Send lesson summary + PDF + HTML to Telegram.

        Args:
            topic: Topic name
            student_name: Student name
            content: LessonContent data
            full_content: Full markdown text
            pdf_path: Path to generated PDF (if any)
            html_path: Path to generated HTML (if any)
        """
        if not TELEGRAM_AVAILABLE:
            logger.warning("Telegram tools not available")
            return

        try:
            celebration = self.config.celebration_word
            subject_str = content.subject.value if content.subject else "general"

            # Determine what files we have
            has_pdf = pdf_path and Path(pdf_path).exists()
            has_html = html_path and Path(html_path).exists()
            files_label = ""
            if has_pdf:
                files_label += " + PDF"
            if has_html:
                files_label += " + HTML"

            # =================================================================
            # SEND SUMMARY MESSAGE
            # =================================================================
            header = f"*Lesson: {topic}*\n"
            header += f"*Subject:* {subject_str.title()}\n"
            header += f"*Student:* {student_name}\n\n"

            stats = f"*{len(content.core_concepts)} concepts* | "
            stats += f"*{len(content.problems)} problems* | "
            stats += f"*{len(content.patterns)} patterns* | "
            stats += f"*{len(content.strategies)} strategies*\n"
            stats += f"*{content.total_words} words*{files_label}\n\n"

            insights = ""
            if content.key_insights:
                insights = f"*Key Insights ({celebration})*\n"
                for i, ins in enumerate(content.key_insights[:4], 1):
                    if ins:
                        insights += f"{i}. {ins[:150]}\n"
                insights += "\n"

            concepts_section = ""
            if content.building_blocks:
                concepts_section = "*Building Blocks:*\n"
                for b in content.building_blocks[:5]:
                    concepts_section += f"- {b.name}\n"
                concepts_section += "\n"

            message = header + stats + insights + concepts_section
            if len(message) > 4000:
                message = message[:3950] + "\n..."

            result = await send_telegram_message_tool({
                'message': message, 'parse_mode': 'Markdown'
            })

            if result.get('success'):
                logger.info(f"Sent summary to Telegram: message_id {result.get('message_id')}")
            else:
                logger.error(f"Telegram message failed: {result.get('error')}")

            # =================================================================
            # SEND PDF FILE
            # =================================================================
            if has_pdf:
                file_result = await send_telegram_file_tool({
                    'file_path': pdf_path,
                    'caption': f"{topic} - Learning Guide for {student_name}"
                })
                if file_result.get('success'):
                    logger.info("Sent PDF to Telegram")
                else:
                    logger.error(f"PDF send failed: {file_result.get('error')}")

            # =================================================================
            # SEND HTML FILE
            # =================================================================
            if has_html:
                file_result = await send_telegram_file_tool({
                    'file_path': html_path,
                    'caption': f"{topic} - Interactive Slides for {student_name}"
                })
                if file_result.get('success'):
                    logger.info("Sent HTML to Telegram")
                else:
                    logger.error(f"HTML send failed: {file_result.get('error')}")

            # =================================================================
            # FALLBACK: SEND MARKDOWN IF NO PDF
            # =================================================================
            if not has_pdf:
                safe_topic = topic.replace(' ', '_').replace(':', '')[:40]
                temp_path = Path(f"/tmp/olympiad_{safe_topic}_{student_name.replace(' ', '_')}.md")
                with open(temp_path, 'w') as f:
                    f.write(full_content)

                file_result = await send_telegram_file_tool({
                    'file_path': str(temp_path),
                    'caption': f"{topic} - Lesson for {student_name} (Markdown)"
                })
                if file_result.get('success'):
                    logger.info("Sent markdown to Telegram")

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

    def seed_gold_standards(self):
        """Seed default gold standards for evaluation."""
        self._init_agents()
        self.add_gold_standard(
            task_type='olympiad_teaching',
            input_data={'topic': 'any'},
            expected_output={
                'concepts_count': 3,
                'problems_count': 8,
                'breakthrough_moments': 2,
                'has_patterns': True,
                'has_strategies': True,
                'has_mistakes': True,
                'word_count': 2000,
            },
            evaluation_criteria={
                'concepts_count': 0.15,
                'problems_count': 0.25,
                'breakthrough_moments': 0.15,
                'has_patterns': 0.15,
                'has_strategies': 0.10,
                'has_mistakes': 0.10,
                'word_count': 0.10,
            }
        )
        logger.info("Seeded gold standards for OlympiadLearningSwarm")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def learn_topic(
    subject: str = "mathematics",
    topic: str = "Number Theory",
    student_name: str = "Student",
    depth: str = "standard",
    target: str = "olympiad",
    send_telegram: bool = False,
) -> OlympiadLearningResult:
    """
    One-liner olympiad learning.

    Usage:
        from core.swarms.olympiad_learning_swarm import learn_topic

        # Mathematics
        result = await learn_topic("mathematics", "Number Theory", "Aria")

        # Physics
        result = await learn_topic("physics", "Mechanics - Newton's Laws", "Aria")

        # With Telegram
        result = await learn_topic("mathematics", "Combinatorics", "Aria", send_telegram=True)
    """
    subject_enum = Subject(subject) if subject in [s.value for s in Subject] else Subject.GENERAL
    depth_enum = LessonDepth(depth) if depth in [d.value for d in LessonDepth] else LessonDepth.STANDARD
    target_enum = DifficultyTier(target) if target in [t.value for t in DifficultyTier] else DifficultyTier.OLYMPIAD

    config = OlympiadLearningConfig(
        subject=subject_enum,
        student_name=student_name,
        depth=depth_enum,
        target_tier=target_enum,
    )
    swarm = OlympiadLearningSwarm(config)
    return await swarm.teach(topic=topic, send_telegram=send_telegram)


def learn_topic_sync(
    subject: str = "mathematics",
    topic: str = "Number Theory",
    student_name: str = "Student",
    depth: str = "standard",
    target: str = "olympiad",
    send_telegram: bool = False,
) -> OlympiadLearningResult:
    """Synchronous version of learn_topic."""
    return asyncio.run(learn_topic(
        subject=subject, topic=topic, student_name=student_name,
        depth=depth, target=target, send_telegram=send_telegram
    ))


__all__ = [
    'OlympiadLearningSwarm',
    'OlympiadLearningConfig',
    'OlympiadLearningResult',
    'learn_topic',
    'learn_topic_sync',
]
