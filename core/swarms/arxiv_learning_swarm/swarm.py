"""ArXiv Learning Swarm - Main swarm orchestrator."""

import asyncio
import logging
import re
from typing import Dict, Any, Optional, List
from pathlib import Path

import dspy

from ..base_swarm import (
    BaseSwarm, SwarmConfig, SwarmResult, AgentRole,
    register_swarm, ExecutionTrace
)
from ..base import DomainSwarm, AgentTeam

from .types import (
    LearningDepth, ContentStyle, AudienceLevel,
    ArxivLearningConfig, PaperInfo, Concept, LearningSection,
    LearningContent, ArxivLearningResult,
    format_steps_on_newlines,
)
from .agents import (
    BaseSwarmAgent, PaperFetcherAgent, ConceptExtractorAgent,
    IntuitionBuilderAgent, MathSimplifierAgent, ExampleGeneratorAgent,
    ProgressiveBuilderAgent, ContentPolisherAgent, UnifiedLearningAgent,
)

# Import Telegram sender tools
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

@register_swarm("arxiv_learning")
class ArxivLearningSwarm(DomainSwarm):
    """
    World-Class ArXiv Learning Swarm.

    Creates engaging, progressive learning content from academic papers.
    Builds understanding from basics to advanced, always explaining WHY.
    """

    AGENT_TEAM = AgentTeam.define(
        (PaperFetcherAgent, "PaperFetcher", "_paper_fetcher"),
        (ConceptExtractorAgent, "ConceptExtractor", "_concept_extractor"),
        (IntuitionBuilderAgent, "IntuitionBuilder", "_intuition_builder"),
        (MathSimplifierAgent, "MathSimplifier", "_math_simplifier"),
        (ExampleGeneratorAgent, "ExampleGenerator", "_example_generator"),
        (ProgressiveBuilderAgent, "ProgressiveBuilder", "_progressive_builder"),
        (ContentPolisherAgent, "ContentPolisher", "_content_polisher"),
        (UnifiedLearningAgent, "UnifiedLearner", "_unified_learner"),
    )

    def __init__(self, config: ArxivLearningConfig = None):
        super().__init__(config or ArxivLearningConfig())
        # Optimization mode from config: "unified" (fast, 2 LLM calls) or "sequential" (original, 10+ calls)
        self._optimization_mode = self.config.optimization_mode

    def set_optimization_mode(self, mode: str):
        """
        Switch optimization mode.

        Args:
            mode: "unified" (fast, ~2 LLM calls) or "sequential" (original, ~10 calls)
        """
        if mode not in ["unified", "sequential"]:
            raise ValueError(f"Invalid mode: {mode}. Use 'unified' or 'sequential'")
        self._optimization_mode = mode
        self._agents_initialized = False  # Force re-init
        logger.info(f" Optimization mode set to: {mode}")

    async def _execute_domain(
        self,
        paper_id: str = None,
        topic: str = None,
        **kwargs
    ) -> ArxivLearningResult:
        """Execute learning content generation."""
        return await self.learn(paper_id=paper_id, topic=topic, **kwargs)

    async def learn(
        self,
        paper_id: str = None,
        topic: str = None,
        depth: LearningDepth = None,
        send_telegram: bool = None
    ) -> ArxivLearningResult:
        """
        Create learning content from an ArXiv paper.

        Delegates to _safe_execute_domain which handles try/except,
        timing, and post-execute learning automatically via PhaseExecutor.

        Args:
            paper_id: ArXiv paper ID (e.g., "1706.03762")
            topic: Search topic (alternative to paper_id)
            depth: Learning depth (quick, standard, deep)
            send_telegram: Whether to send result to Telegram

        Returns:
            ArxivLearningResult with complete learning content
        """
        # Initialize agents before using them
        self._init_agents()

        config = self.config
        learning_depth = depth or config.depth
        # Convert string to enum if needed
        if isinstance(learning_depth, str):
            learning_depth = LearningDepth(learning_depth.lower())

        logger.info(f" ArxivLearningSwarm starting...")

        return await self._safe_execute_domain(
            task_type='paper_learning',
            default_tools=['arxiv_fetch', 'concept_extract', 'content_generate'],
            result_class=ArxivLearningResult,
            execute_fn=lambda executor: self._execute_phases(
                executor, paper_id, topic, learning_depth, send_telegram, config
            ),
            output_data_fn=lambda result: {
                'concepts_count': getattr(result, 'concepts_covered', 0),
                'bingo_moments': getattr(result, 'bingo_moments', 0),
                'has_hook': bool(getattr(result, 'content', None) and getattr(result.content, 'hook', '')),
                'has_summary': bool(getattr(result, 'content', None) and getattr(result.content, 'summary', '')),
                'has_examples': getattr(result, 'concepts_covered', 0) > 0,
                'word_count': getattr(result, 'content', None) and getattr(result.content, 'total_words', 0) or 0,
            },
            input_data_fn=lambda: {
                'paper_id': paper_id,
                'topic': topic,
                'depth': learning_depth.value if learning_depth else None,
            },
        )

    async def _execute_phases(
        self,
        executor,
        paper_id: str,
        topic: str,
        learning_depth: LearningDepth,
        send_telegram: bool,
        config,
    ) -> ArxivLearningResult:
        """Execute all ArXiv learning phases using PhaseExecutor.

        Args:
            executor: PhaseExecutor instance for tracing and timing
            paper_id: ArXiv paper ID
            topic: Search topic (alternative to paper_id)
            learning_depth: Learning depth enum
            send_telegram: Whether to send result to Telegram
            config: ArxivLearningConfig instance

        Returns:
            ArxivLearningResult with complete learning content
        """
        # =================================================================
        # PHASE 1: FETCH PAPER
        # =================================================================
        paper = await executor.run_phase(
            1, "Fetching Paper", "PaperFetcher", AgentRole.ACTOR,
            self._fetch_paper(paper_id, topic),
            input_data={'paper_id': paper_id, 'topic': topic},
            tools_used=['arxiv_fetch'],
        )

        if not paper:
            return ArxivLearningResult(
                success=False,
                swarm_name=self.config.name,
                domain=self.config.domain,
                output={},
                execution_time=executor.elapsed(),
                error="Could not fetch paper"
            )

        logger.info(f"  Found: {paper.title[:60]}...")

        # =================================================================
        # PHASE 2: EXTRACT CONCEPTS (with swarm caching)
        # =================================================================
        concepts = await executor.run_phase(
            2, "Extracting Concepts", "ConceptExtractor", AgentRole.EXPERT,
            self._extract_concepts(paper, config),
            input_data={'paper_title': paper.title},
            tools_used=['concept_extract'],
        )

        # Limit concepts based on depth to prevent timeouts
        if learning_depth == LearningDepth.QUICK:
            max_concepts = config.max_concepts_quick
        elif learning_depth == LearningDepth.STANDARD:
            max_concepts = config.max_concepts_standard
        else:
            max_concepts = 5  # DEEP

        if len(concepts) > max_concepts:
            logger.info(f"  Limiting concepts: {len(concepts)} -> {max_concepts} (depth={learning_depth.value})")
            concepts = concepts[:max_concepts]

        logger.info(f"  Extracted {len(concepts)} concepts")

        # =================================================================
        # PHASE 3-7: CONTENT GENERATION
        # =================================================================
        content_data = await self._generate_content(
            executor, paper, concepts, learning_depth, config
        )

        intuitions = content_data['intuitions']
        math_explanations = content_data['math_explanations']
        examples = content_data['examples']
        draft_content = content_data['draft_content']
        progressive_result = content_data['progressive_result']
        polished = content_data['polished']

        # =================================================================
        # BUILD LEARNING SECTIONS
        # =================================================================
        sections = self._build_learning_sections(
            paper, concepts, intuitions, math_explanations, examples, config
        )

        # Count bingo moments
        bingo_count = sum(1 for s in sections if s.has_bingo_moment)
        bingo_count += len(progressive_result.get('key_insights', []))

        # Build final content
        final_content = polished.get('polished_content', '') or draft_content

        # Hook text
        hook_text = ""
        for concept_name, intuition in intuitions.items():
            if intuition.get('hook'):
                hook_text = intuition['hook']
                break

        learning_content = LearningContent(
            paper=paper,
            hook=hook_text,
            concepts=concepts,
            sections=sections,
            key_insights=progressive_result.get('key_insights', []),
            summary=progressive_result.get('summary', ''),
            next_steps=progressive_result.get('next_steps', []),
            total_words=len(final_content.split())
        )

        # =================================================================
        # PHASE 8: GENERATE OUTPUTS (PARALLEL)
        # =================================================================
        pdf_path, pptx_path, pptx_pdf_path, html_path = await self._generate_outputs(
            executor, paper, learning_content
        )

        # =================================================================
        # BUILD RESULT
        # =================================================================
        # Estimate learning time
        if learning_depth == LearningDepth.QUICK:
            learning_time = "5-10 minutes"
        elif learning_depth == LearningDepth.STANDARD:
            learning_time = "20-30 minutes"
        else:
            learning_time = "45-60 minutes"

        result = ArxivLearningResult(
            success=True,
            swarm_name=self.config.name,
            domain=self.config.domain,
            output={
                'title': paper.title,
                'arxiv_id': paper.arxiv_id,
                'content': final_content
            },
            execution_time=executor.elapsed(),
            paper=paper,
            content=learning_content,
            learning_time_estimate=learning_time,
            concepts_covered=len(concepts),
            bingo_moments=bingo_count,
            difficulty_progression=[s.level for s in sections],
            pdf_path=pdf_path,
            pptx_path=pptx_path,
            pptx_pdf_path=pptx_pdf_path,
            html_path=html_path
        )

        logger.info(f" ArxivLearningSwarm complete: {paper.title[:40]}...")
        logger.info(f"   {len(concepts)} concepts, {bingo_count} {config.celebration_word} moments")

        # Record orchestrator-level execution trace
        self._record_trace(
            agent_name="ArxivLearningSwarm",
            agent_role=AgentRole.ORCHESTRATOR,
            input_data={'paper_id': paper.arxiv_id, 'topic': paper.title},
            output_data={
                'concepts': len(concepts),
                'sections': len(sections),
                'bingo_moments': bingo_count,
                'words': learning_content.total_words
            },
            execution_time=executor.elapsed(),
            success=True
        )

        # Log LLM metrics
        try:
            if hasattr(dspy.settings, 'lm') and hasattr(dspy.settings.lm, 'get_metrics'):
                metrics = dspy.settings.lm.get_metrics()
                logger.info(f" LLM Metrics: {metrics.get('successful_calls', 0)}/{metrics.get('total_calls', 0)} calls succeeded ({metrics.get('success_rate', 'N/A')})")
                if metrics.get('retried_calls', 0) > 0:
                    logger.info(f" Retried {metrics['retried_calls']} calls")
        except Exception:
            pass

        # Send to Telegram (if enabled)
        should_send = send_telegram if send_telegram is not None else config.send_telegram

        if should_send and TELEGRAM_AVAILABLE:
            logger.info(" Sending to Telegram...")
            await self._send_to_telegram(paper, learning_content, final_content, pdf_path=pdf_path, pptx_path=pptx_path, pptx_pdf_path=pptx_pdf_path, html_path=html_path)
        elif should_send and not TELEGRAM_AVAILABLE:
            logger.warning(" Telegram sending requested but tools not available")

        return result

    async def _fetch_paper(self, paper_id: str, topic: str):
        """Fetch paper by ID or topic. Returns PaperInfo or None."""
        paper = None
        if paper_id:
            paper = await self._paper_fetcher.fetch_by_id(paper_id)
        elif topic:
            papers = await self._paper_fetcher.search_by_topic(topic, 1)
            paper = papers[0] if papers else None
        return paper

    async def _extract_concepts(self, paper, config):
        """Extract concepts from paper with swarm caching. Returns list of Concept."""
        cache_key = f"concepts_{paper.arxiv_id}"
        concepts = None
        if config.use_swarm_cache:
            concepts = self._get_cached(cache_key)
            if concepts:
                logger.info(f" Loaded {len(concepts)} concepts from cache")

        if not concepts:
            concepts = await self._concept_extractor.extract(paper)

            if not concepts:
                # Create default concept from abstract
                concepts = [Concept(
                    name="Main Contribution",
                    description=paper.abstract[:200],
                    why_it_matters="This is the paper's key innovation",
                    prerequisites=["Basic understanding of the field"],
                    difficulty=3,
                    math_required=True
                )]

            # Cache for future use
            if config.use_swarm_cache:
                self._cache_result(cache_key, concepts, ttl=7200)  # 2 hour TTL

        return concepts

    async def _generate_content(
        self, executor, paper, concepts, learning_depth, config
    ) -> Dict[str, Any]:
        """Generate content using the appropriate optimization mode.

        Dispatches to parallel_deep, unified, parallel, or sequential mode
        based on self._optimization_mode. Uses executor for phase tracing.

        Args:
            executor: PhaseExecutor instance
            paper: PaperInfo
            concepts: List of Concept
            learning_depth: LearningDepth enum
            config: ArxivLearningConfig

        Returns:
            Dict with keys: intuitions, math_explanations, examples,
            draft_content, progressive_result, polished
        """
        if self._optimization_mode == "parallel_deep" and self._unified_learner:
            return await self._content_parallel_deep(executor, paper, concepts, config)
        elif self._optimization_mode == "unified" and self._unified_learner:
            return await self._content_unified(executor, paper, concepts, config)
        elif self._optimization_mode == "parallel":
            return await self._content_parallel(executor, paper, concepts, learning_depth, config)
        else:
            return await self._content_sequential(executor, paper, concepts, learning_depth, config)

    async def _content_parallel_deep(self, executor, paper, concepts, config) -> Dict[str, Any]:
        """Parallel deep mode: full quality with parallel per-concept generation."""
        # Check cache first
        content_cache_key = f"parallel_deep_{paper.arxiv_id}_{config.audience.value}_{len(concepts)}"
        unified_result = None
        if config.use_swarm_cache:
            unified_result = self._get_cached(content_cache_key)
            if unified_result:
                logger.info(f" Loaded parallel deep content from cache")

        if not unified_result:
            unified_result = await executor.run_phase(
                3, "Parallel Deep Content Generation", "ParallelDeepLearner", AgentRole.PLANNER,
                self._unified_learner.generate_parallel(
                    paper=paper,
                    concepts=concepts,
                    audience_level=config.audience.value,
                    celebration_word=config.celebration_word,
                    max_concurrent=config.max_concurrent_llm
                ),
                input_data={'concepts_count': len(concepts), 'audience': config.audience.value},
                tools_used=['parallel_deep_learning'],
            )
            if config.use_swarm_cache and unified_result:
                self._cache_result(content_cache_key, unified_result, ttl=3600)

        draft_content = unified_result.get('complete_content', '')
        return {
            'intuitions': unified_result.get('intuitions', {}),
            'math_explanations': unified_result.get('math_explanations', {}),
            'examples': unified_result.get('examples', {}),
            'draft_content': draft_content,
            'progressive_result': {
                'complete_content': draft_content,
                'key_insights': unified_result.get('key_insights', []),
                'summary': unified_result.get('summary', ''),
                'next_steps': unified_result.get('next_steps', [])
            },
            'polished': {'polished_content': draft_content},
        }

    async def _content_unified(self, executor, paper, concepts, config) -> Dict[str, Any]:
        """Unified mode: single LLM call (experimental, may timeout)."""
        # Check swarm cache for unified content
        content_cache_key = f"unified_{paper.arxiv_id}_{config.audience.value}_{len(concepts)}"
        unified_result = None
        if config.use_swarm_cache:
            unified_result = self._get_cached(content_cache_key)
            if unified_result:
                logger.info(f" Loaded unified content from cache")

        if not unified_result:
            unified_result = await executor.run_phase(
                3, "Unified Content Generation", "UnifiedLearner", AgentRole.PLANNER,
                self._unified_learner.generate_all(
                    paper=paper,
                    concepts=concepts,
                    audience_level=config.audience.value,
                    celebration_word=config.celebration_word
                ),
                input_data={'concepts_count': len(concepts), 'audience': config.audience.value},
                tools_used=['unified_learning'],
            )
            # Cache for future use
            if config.use_swarm_cache and unified_result:
                self._cache_result(content_cache_key, unified_result, ttl=3600)

        draft_content = unified_result.get('complete_content', '')
        return {
            'intuitions': unified_result.get('intuitions', {}),
            'math_explanations': unified_result.get('math_explanations', {}),
            'examples': unified_result.get('examples', {}),
            'draft_content': draft_content,
            'progressive_result': {
                'complete_content': draft_content,
                'key_insights': unified_result.get('key_insights', []),
                'summary': unified_result.get('summary', ''),
                'next_steps': unified_result.get('next_steps', [])
            },
            'polished': {'polished_content': draft_content},
        }

    async def _content_parallel(self, executor, paper, concepts, learning_depth, config) -> Dict[str, Any]:
        """Parallel mode: run concept operations with controlled concurrency."""
        # Semaphore to limit concurrent LLM calls
        llm_semaphore = asyncio.Semaphore(2)

        async def rate_limited_call(coro):
            """Wrapper to limit concurrent LLM calls."""
            async with llm_semaphore:
                return await coro

        # Phase 3: Build intuitions in parallel (limited to 2 concurrent)
        intuition_tasks = [
            (f"IntuitionBuilder({c.name})", AgentRole.ACTOR,
             rate_limited_call(self._intuition_builder.build(c, config.audience.value)),
             ['intuition_build'])
            for c in concepts[:3]
        ]
        intuition_results = await executor.run_parallel(3, "Building Intuitions", intuition_tasks)

        intuitions = {}
        for c, result in zip(concepts[:3], intuition_results):
            if isinstance(result, dict) and 'error' not in result:
                intuitions[c.name] = result

        # Phase 4: Build math explanations in parallel (for concepts that need it)
        math_explanations = {}
        if learning_depth in [LearningDepth.STANDARD, LearningDepth.DEEP]:
            math_concepts = [c for c in concepts[:3] if c.math_required]
            if math_concepts:
                math_tasks = [
                    (f"MathSimplifier({c.name})", AgentRole.ACTOR,
                     rate_limited_call(self._math_simplifier.simplify(c, intuitions.get(c.name, {}), config.audience.value)),
                     ['math_simplify'])
                    for c in math_concepts
                ]
                math_results = await executor.run_parallel(4, "Simplifying Math", math_tasks)

                for c, result in zip(math_concepts, math_results):
                    if isinstance(result, dict) and 'error' not in result:
                        math_explanations[c.name] = result

        # Phase 5: Generate examples in parallel (for top 2 concepts)
        examples = {}
        if config.include_code_examples:
            example_tasks = [
                (f"ExampleGenerator({c.name})", AgentRole.ACTOR,
                 rate_limited_call(self._example_generator.generate(
                     c, intuitions.get(c.name, {}), math_explanations.get(c.name, {}))),
                 ['example_generate'])
                for c in concepts[:2]
            ]
            example_results = await executor.run_parallel(5, "Generating Examples", example_tasks)

            for c, result in zip(concepts[:2], example_results):
                if isinstance(result, dict) and 'error' not in result:
                    examples[c.name] = result

        logger.info(f"     {len(intuitions)} intuitions, {len(math_explanations)} math, {len(examples)} examples")

        # Phase 6: Build content directly (SKIP ProgressiveBuilder to avoid 76KB prompt)
        draft_content = self._build_direct_content(paper, concepts, intuitions, math_explanations, examples, config)

        # Extract key insights from aha_moments
        key_insights = [
            intuitions[c.name].get('aha_moment', '')
            for c in concepts[:3] if c.name in intuitions and intuitions[c.name].get('aha_moment')
        ]

        return {
            'intuitions': intuitions,
            'math_explanations': math_explanations,
            'examples': examples,
            'draft_content': draft_content,
            'progressive_result': {
                'complete_content': draft_content,
                'key_insights': key_insights,
                'summary': f"This paper introduces {concepts[0].name if concepts else 'key concepts'} and related innovations.",
                'next_steps': ['Explore related papers', 'Implement the concepts', 'Read the full paper']
            },
            'polished': {'polished_content': draft_content},
        }

    async def _content_sequential(self, executor, paper, concepts, learning_depth, config) -> Dict[str, Any]:
        """Sequential mode: original multi-call approach."""
        # Phase 3: Build intuitions sequentially
        intuitions = {}
        for concept in concepts[:3]:
            try:
                result = await executor.run_phase(
                    3, f"Building Intuition ({concept.name})", "IntuitionBuilder", AgentRole.ACTOR,
                    self._intuition_builder.build(concept, config.audience.value),
                    input_data={'concept': concept.name},
                    tools_used=['intuition_build'],
                )
                if isinstance(result, dict):
                    intuitions[concept.name] = result
                await asyncio.sleep(1)
            except Exception as e:
                logger.warning(f"Intuition building failed for {concept.name}: {e}")

        # Phase 4: Math
        math_explanations = {}
        if learning_depth in [LearningDepth.STANDARD, LearningDepth.DEEP]:
            for concept in concepts[:3]:
                if concept.math_required:
                    try:
                        result = await executor.run_phase(
                            4, f"Simplifying Math ({concept.name})", "MathSimplifier", AgentRole.ACTOR,
                            self._math_simplifier.simplify(
                                concept, intuitions.get(concept.name, {}), config.audience.value),
                            input_data={'concept': concept.name},
                            tools_used=['math_simplify'],
                        )
                        if isinstance(result, dict):
                            math_explanations[concept.name] = result
                        await asyncio.sleep(1)
                    except Exception as e:
                        logger.warning(f"Math simplification failed for {concept.name}: {e}")

        # Phase 5: Examples
        examples = {}
        if config.include_code_examples:
            for concept in concepts[:2]:
                try:
                    result = await executor.run_phase(
                        5, f"Generating Example ({concept.name})", "ExampleGenerator", AgentRole.ACTOR,
                        self._example_generator.generate(
                            concept, intuitions.get(concept.name, {}), math_explanations.get(concept.name, {})),
                        input_data={'concept': concept.name},
                        tools_used=['example_generate'],
                    )
                    if isinstance(result, dict):
                        examples[concept.name] = result
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.warning(f"Example generation failed for {concept.name}: {e}")

        # Phase 6: Progressive content
        progressive_result = await executor.run_phase(
            6, "Building Progressive Content", "ProgressiveBuilder", AgentRole.PLANNER,
            self._progressive_builder.build(
                paper, concepts, intuitions, math_explanations, examples, config.celebration_word),
            input_data={'concepts_count': len(concepts), 'has_math': bool(math_explanations), 'has_examples': bool(examples)},
            tools_used=['progressive_build'],
        )

        # Phase 7: Polish
        draft_content = progressive_result.get('complete_content', '')
        polished = await executor.run_phase(
            7, "Polishing Content", "ContentPolisher", AgentRole.REVIEWER,
            self._content_polisher.polish(draft_content, config.style.value, config.audience.value),
            input_data={'draft_length': len(draft_content)},
            tools_used=['content_polish'],
        )

        return {
            'intuitions': intuitions,
            'math_explanations': math_explanations,
            'examples': examples,
            'draft_content': draft_content,
            'progressive_result': progressive_result,
            'polished': polished,
        }

    def _build_direct_content(self, paper, concepts, intuitions, math_explanations, examples, config) -> str:
        """Build content directly from sections without an extra LLM call."""
        draft_parts = []
        draft_parts.append(f"# {paper.title}\n")

        # Add hook from first intuition
        for name, intuition in intuitions.items():
            if intuition.get('hook'):
                draft_parts.append(f"\n## Why Should You Care?\n{intuition['hook']}\n")
                break

        # Add concept sections
        for concept in concepts[:3]:
            intuition = intuitions.get(concept.name, {})
            math = math_explanations.get(concept.name, {})
            example = examples.get(concept.name, {})

            draft_parts.append(f"\n## {concept.name}\n")
            if intuition.get('analogy'):
                draft_parts.append(f"**Analogy:** {intuition['analogy']}\n")
            if intuition.get('intuition_build'):
                draft_parts.append(f"\n{intuition['intuition_build']}\n")
            if intuition.get('aha_moment'):
                draft_parts.append(f"\n **{config.celebration_word}!** {intuition['aha_moment']}\n")
            if math.get('step_by_step'):
                draft_parts.append(f"\n### The Math\n{math['step_by_step']}\n")
            if example.get('code_example'):
                draft_parts.append(f"\n### Code\n```python\n{example['code_example']}\n```\n")

        return '\n'.join(draft_parts)

    def _build_learning_sections(self, paper, concepts, intuitions, math_explanations, examples, config) -> List[LearningSection]:
        """Build learning sections from generated content data."""
        sections = []

        # Hook section
        hook_text = ""
        for concept_name, intuition in intuitions.items():
            if intuition.get('hook'):
                hook_text = intuition['hook']
                break

        sections.append(LearningSection(
            title="Why Should You Care?",
            content=hook_text or f"Let's understand {paper.title}",
            level=1,
            has_bingo_moment=False
        ))

        # Intuition sections - apply step formatting post-processing
        for concept in concepts[:3]:
            intuition = intuitions.get(concept.name, {})
            if intuition:
                intuition_content = format_steps_on_newlines(intuition.get('intuition_build', ''))
                sections.append(LearningSection(
                    title=f"Understanding {concept.name}",
                    content=f"{intuition.get('analogy', '')}\n\n{intuition_content}",
                    level=2,
                    has_bingo_moment=True if intuition.get('aha_moment') else False
                ))

        # Math sections - apply step formatting post-processing
        for concept_name, math in math_explanations.items():
            step_content = format_steps_on_newlines(math.get('step_by_step', ''))
            sections.append(LearningSection(
                title=f"The Math: {concept_name}",
                content=f"{math.get('math_motivation', '')}\n\n{step_content}",
                level=3,
                has_bingo_moment=False
            ))

        # Example sections - apply step formatting post-processing
        for concept_name, ex in examples.items():
            raw_code = ex.get('code_example', '')
            raw_code = re.sub(r'^```\w*\n?', '', raw_code)
            raw_code = re.sub(r'\n?```$', '', raw_code).strip()
            simple_example = format_steps_on_newlines(ex.get('simple_example', ''))
            sections.append(LearningSection(
                title=f"See It In Action: {concept_name}",
                content=f"**Simple Example:**\n{simple_example}",
                level=4,
                has_bingo_moment=False,
                code_example=raw_code
            ))

        return sections

    async def _generate_outputs(self, executor, paper, learning_content) -> tuple:
        """Generate PDF, PPTX, and HTML outputs in parallel.

        Args:
            executor: PhaseExecutor instance
            paper: PaperInfo
            learning_content: LearningContent

        Returns:
            Tuple of (pdf_path, pptx_path, pptx_pdf_path, html_path)
        """
        logger.info(" Generating outputs (PDF, PPTX, HTML in parallel)...")

        async def gen_pdf():
            return await self._generate_pdf(paper, learning_content)

        async def gen_pptx():
            if self.config.generate_pptx:
                return await self._generate_pptx(paper, learning_content)
            return (None, None)

        async def gen_html():
            if self.config.generate_html:
                return await self._generate_html(paper, learning_content)
            return None

        output_results = await executor.run_parallel(
            8, "Generating Outputs",
            [
                ("PDFGenerator", AgentRole.ACTOR, gen_pdf(), ['pdf_generate']),
                ("PPTXGenerator", AgentRole.ACTOR, gen_pptx(), ['pptx_generate']),
                ("HTMLGenerator", AgentRole.ACTOR, gen_html(), ['html_generate']),
            ]
        )

        # Extract results with safe defaults for errors
        pdf_path = output_results[0] if not isinstance(output_results[0], dict) or 'error' not in output_results[0] else None
        pptx_result = output_results[1] if not isinstance(output_results[1], dict) or 'error' not in output_results[1] else (None, None)
        pptx_path, pptx_pdf_path = pptx_result if isinstance(pptx_result, tuple) else (None, None)
        html_path = output_results[2] if not isinstance(output_results[2], dict) or 'error' not in output_results[2] else None

        return pdf_path, pptx_path, pptx_pdf_path, html_path

    async def search_and_learn(
        self,
        topic: str,
        max_papers: int = 1,
        rank_by: str = "Which {abstract} is most exciting and impactful for learning?"
    ) -> List[ArxivLearningResult]:
        """
        Search for papers and create learning content.

        If LOTUS is enabled and available, uses semantic ranking to find
        the most valuable papers for learning. Otherwise falls back to
        standard relevance-based search.

        Args:
            topic: Search topic
            max_papers: Number of papers to process
            rank_by: Natural language ranking criterion (LOTUS only)

        Returns:
            List of ArxivLearningResult objects
        """
        self._init_agents()

        # Use LOTUS for semantic search if available and enabled
        if self.config.use_lotus and LOTUS_AVAILABLE:
            logger.info(" Using LOTUS for semantic paper search...")
            papers = await self._paper_fetcher.search_and_rank_with_lotus(
                topic=topic,
                max_results=max_papers,
                rank_by=rank_by,
                lotus_model=self.config.lotus_model
            )
        else:
            papers = await self._paper_fetcher.search_by_topic(topic, max_papers)

        results = []
        for paper in papers:
            result = await self.learn(paper_id=paper.arxiv_id)
            results.append(result)

        return results

    async def _generate_pdf(
        self,
        paper: PaperInfo,
        content: LearningContent
    ) -> Optional[str]:
        """Generate professional PDF with visualizations. Returns path or None."""
        try:
            from ..skills.research.learning_pdf_template import (
                convert_learning_to_pdf,
                generate_concept_visualization
            )

            celebration = self.config.celebration_word

            # Prepare concepts for PDF
            concepts_data = [
                {
                    'name': c.name,
                    'description': c.description,
                    'difficulty': c.difficulty,
                    'why_it_matters': c.why_it_matters
                }
                for c in content.concepts
            ]

            # =============================================================
            # GENERATE VISUALIZATIONS FOR KEY CONCEPTS (PARALLEL)
            # =============================================================
            logger.info(" Generating concept visualizations (parallel)...")
            visualization_paths = {}

            # Build a map of concept name -> intuition text from sections
            concept_intuitions = {}
            for s in content.sections:
                for c in content.concepts:
                    if c.name.lower() in s.title.lower() and c.name not in concept_intuitions:
                        concept_intuitions[c.name] = s.content[:500]

            # OPTIMIZATION: Generate visualizations in PARALLEL using asyncio.gather
            async def generate_viz_for_concept(concept):
                """Generate visualization for a single concept."""
                try:
                    description_parts = [concept.description, concept.why_it_matters]
                    if concept.name in concept_intuitions:
                        description_parts.append(concept_intuitions[concept.name])
                    rich_description = " ".join(filter(None, description_parts))

                    viz_path = await generate_concept_visualization(
                        concept_name=concept.name,
                        concept_description=rich_description,
                        output_dir="/tmp"
                    )
                    if viz_path:
                        return (concept.name, viz_path)
                except Exception as e:
                    logger.debug(f"Viz generation failed for {concept.name}: {e}")
                return None

            # Run all visualization generations in parallel
            viz_tasks = [generate_viz_for_concept(c) for c in content.concepts[:3]]  # Top 3 concepts
            viz_results = await asyncio.gather(*viz_tasks, return_exceptions=True)

            for result in viz_results:
                if result and isinstance(result, tuple):
                    concept_name, viz_path = result
                    visualization_paths[concept_name] = viz_path
                    logger.info(f"  Generated viz for: {concept_name}")

            # Prepare sections for PDF with visualizations
            used_visualizations = set()
            sections_data = []
            for s in content.sections:
                section_dict = {
                    'title': s.title,
                    'content': s.content,
                    'level': s.level,
                    'has_bingo_moment': s.has_bingo_moment,
                    'code_example': s.code_example
                }
                for concept_name, viz_path in visualization_paths.items():
                    if concept_name not in used_visualizations and concept_name.lower() in s.title.lower():
                        section_dict['visualization_path'] = viz_path
                        used_visualizations.add(concept_name)
                        break
                sections_data.append(section_dict)

            # Estimate learning time
            if content.total_words < 1000:
                learning_time = "10-15 min"
            elif content.total_words < 2000:
                learning_time = "20-30 min"
            else:
                learning_time = "45-60 min"

            # Generate PDF
            output_path = f"/tmp/arxiv_{paper.arxiv_id.replace('.', '_').replace('/', '_')}_learning.pdf"

            pdf_path = await convert_learning_to_pdf(
                paper_title=paper.title,
                arxiv_id=paper.arxiv_id,
                authors=paper.authors,
                hook=content.hook,
                concepts=concepts_data,
                sections=sections_data,
                key_insights=content.key_insights,
                summary=content.summary,
                next_steps=content.next_steps,
                output_path=output_path,
                bingo_word=celebration,
                learning_time=learning_time,
                total_words=content.total_words
            )

            logger.info(f" Generated PDF: {pdf_path}")
            return pdf_path

        except Exception as e:
            logger.warning(f"PDF generation failed: {e}")
            return None

    async def _generate_pptx(
        self,
        paper: PaperInfo,
        content: LearningContent
    ) -> tuple:
        """Generate professional PowerPoint presentation.

        Returns:
            Tuple of (pptx_path, pptx_pdf_path). Either can be None if generation/conversion failed.
        """
        pptx_path = None
        pptx_pdf_path = None

        try:
            from ..skills.research.pptx_generator import (
                generate_learning_pptx,
                convert_pptx_to_pdf,
                is_libreoffice_available
            )

            celebration = self.config.celebration_word

            # Prepare concepts for PPTX
            concepts_data = [
                {
                    'name': c.name,
                    'description': c.description,
                    'difficulty': c.difficulty,
                    'why_it_matters': c.why_it_matters
                }
                for c in content.concepts
            ]

            # Prepare sections for PPTX
            sections_data = [
                {
                    'title': s.title,
                    'content': s.content,
                    'level': s.level,
                    'has_bingo_moment': s.has_bingo_moment,
                    'code_example': s.code_example
                }
                for s in content.sections
            ]

            # Estimate learning time
            if content.total_words < 1000:
                learning_time = "10-15 min"
            elif content.total_words < 2000:
                learning_time = "20-30 min"
            else:
                learning_time = "45-60 min"

            # Generate PPTX (use _presentation suffix to avoid conflict with _learning.pdf)
            output_path = f"/tmp/arxiv_{paper.arxiv_id.replace('.', '_').replace('/', '_')}_presentation.pptx"

            pptx_path = await generate_learning_pptx(
                paper_title=paper.title,
                arxiv_id=paper.arxiv_id,
                authors=paper.authors,
                hook=content.hook,
                concepts=concepts_data,
                sections=sections_data,
                key_insights=content.key_insights,
                summary=content.summary,
                next_steps=content.next_steps,
                output_path=output_path,
                bingo_word=celebration,
                learning_time=learning_time,
                total_words=content.total_words
            )

            if pptx_path:
                logger.info(f" Generated PPTX: {pptx_path}")

                # Convert PPTX to PDF if enabled (default: True)
                if self.config.convert_pptx_to_pdf:
                    if is_libreoffice_available():
                        logger.info(" Converting PPTX to PDF...")
                        pptx_pdf_path = await convert_pptx_to_pdf(pptx_path)
                        if pptx_pdf_path:
                            logger.info(f" Converted PPTX to PDF: {pptx_pdf_path}")
                    else:
                        logger.warning(" PPTX-to-PDF conversion skipped (LibreOffice not installed)")

            return pptx_path, pptx_pdf_path

        except Exception as e:
            logger.warning(f"PPTX generation failed: {e}")
            return None, None

    async def _generate_html(
        self,
        paper: PaperInfo,
        content: LearningContent
    ) -> Optional[str]:
        """Generate interactive HTML slides.

        Returns:
            Path to generated HTML file, or None if generation failed.
        """
        try:
            from ..skills.research.pptx_generator import generate_learning_html

            celebration = self.config.celebration_word

            # Build comprehensive paper_data for HTML generator
            paper_data = {
                'paper_title': paper.title,
                'arxiv_id': paper.arxiv_id,
                'authors': paper.authors,
                'hook': content.hook,
                'summary': content.summary,
                'abstract': paper.abstract,
                'bingo_word': celebration,

                # Full concepts with all fields
                'concepts': [
                    {
                        'name': c.name,
                        'description': c.description,
                        'why_it_matters': c.why_it_matters,
                        'prerequisites': c.prerequisites,
                        'difficulty': c.difficulty,
                        'math_required': c.math_required,
                    }
                    for c in content.concepts
                ],

                # Full sections with all fields
                'sections': [
                    {
                        'title': s.title,
                        'content': s.content,
                        'level': s.level,
                        'has_bingo_moment': s.has_bingo_moment,
                        'code_example': s.code_example,
                        'visualization_desc': s.visualization_desc,
                        'exercises': s.exercises,
                    }
                    for s in content.sections
                ],

                'key_insights': content.key_insights,
                'next_steps': content.next_steps,

                # Estimate learning time
                'learning_time': "10-15 min" if content.total_words < 1000 else "20-30 min" if content.total_words < 2000 else "45-60 min",
            }

            # Generate HTML slides
            output_path = f"/tmp/arxiv_{paper.arxiv_id.replace('.', '_').replace('/', '_')}_slides.html"

            html_path = await generate_learning_html(paper_data, output_path)

            if html_path:
                logger.info(f" Generated HTML slides: {html_path}")

            return html_path

        except Exception as e:
            logger.warning(f"HTML slide generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def _send_to_telegram(
        self,
        paper: PaperInfo,
        content: LearningContent,
        full_content: str,
        pdf_path: Optional[str] = None,
        pptx_path: Optional[str] = None,
        pptx_pdf_path: Optional[str] = None,
        html_path: Optional[str] = None
    ):
        """Send learning content summary, PDF, PPTX, and HTML to Telegram.

        Args:
            paper: Paper information
            content: Learning content
            full_content: Full text content
            pdf_path: Path to generated PDF (learning guide)
            pptx_path: Path to generated PPTX
            pptx_pdf_path: Path to PDF converted from PPTX (preferred for Telegram)
            html_path: Path to generated HTML slides
        """
        if not TELEGRAM_AVAILABLE:
            logger.warning("Telegram tools not available")
            return

        try:
            celebration = self.config.celebration_word

            # Determine what files we'll send
            has_pptx_pdf = pptx_pdf_path and Path(pptx_pdf_path).exists()
            has_pptx = pptx_path and Path(pptx_path).exists()
            has_html = html_path and Path(html_path).exists()
            presentation_label = " Presentation PDF" if has_pptx_pdf else " PPTX"
            html_label = " + HTML Slides" if has_html else ""

            # =================================================================
            # SEND TELEGRAM MESSAGE (Summary)
            # =================================================================
            header = f" *ArXiv Learning: {paper.title[:60]}*\n"
            header += f" ID: `{paper.arxiv_id}`\n"
            header += f" {', '.join(paper.authors[:3])}\n"
            header += f" {paper.arxiv_url}\n\n"

            hook_section = f"* Why Should You Care?*\n{content.hook[:400] if content.hook else 'Learn about cutting-edge research!'}\n\n"

            insights_section = ""
            if content.key_insights:
                insights_section = f"* Key Insights ({celebration}!)*\n"
                for i, insight in enumerate(content.key_insights[:4], 1):
                    insights_section += f"{i}. {insight[:150]}\n"
                insights_section += "\n"

            concepts_section = f"* Concepts ({len(content.concepts)})*\n"
            for concept in content.concepts[:4]:
                concepts_section += f"• {concept.name}\n"
            concepts_section += "\n"

            stats = f" {content.total_words} words | {len(content.concepts)} concepts | {len(content.key_insights)} insights\n"
            stats += f" Learning PDF + {presentation_label}{html_label} attached below"

            message = header + hook_section + insights_section + concepts_section + stats

            if len(message) > 4000:
                message = message[:3950] + "\n..."

            result = await send_telegram_message_tool({
                'message': message,
                'parse_mode': 'Markdown'
            })

            if result.get('success'):
                logger.info(f" Sent summary to Telegram: message_id {result.get('message_id')}")
            else:
                logger.error(f" Telegram message failed: {result.get('error')}")

            # =================================================================
            # SEND PDF FILE (Learning Guide)
            # =================================================================
            if pdf_path and Path(pdf_path).exists():
                file_result = await send_telegram_file_tool({
                    'file_path': pdf_path,
                    'caption': f" {paper.title[:50]} - Learning Guide"
                })

                if file_result.get('success'):
                    logger.info(f" Sent Learning PDF to Telegram")
                else:
                    logger.error(f" Learning PDF send failed: {file_result.get('error')}")

            # =================================================================
            # SEND PRESENTATION (prefer PDF from PPTX, fallback to raw PPTX)
            # =================================================================
            if has_pptx_pdf:
                # Send PDF converted from PPTX (better Telegram experience)
                file_result = await send_telegram_file_tool({
                    'file_path': pptx_pdf_path,
                    'caption': f" {paper.title[:50]} - Presentation (PDF)"
                })

                if file_result.get('success'):
                    logger.info(f" Sent Presentation PDF to Telegram")
                else:
                    logger.error(f" Presentation PDF send failed: {file_result.get('error')}")
            elif has_pptx:
                # Fallback: Send raw PPTX (when LibreOffice not available)
                file_result = await send_telegram_file_tool({
                    'file_path': pptx_path,
                    'caption': f" {paper.title[:50]} - Presentation (PPTX)"
                })

                if file_result.get('success'):
                    logger.info(f" Sent PPTX to Telegram")
                else:
                    logger.error(f" PPTX send failed: {file_result.get('error')}")

            # =================================================================
            # SEND HTML SLIDES
            # =================================================================
            if html_path and Path(html_path).exists():
                file_result = await send_telegram_file_tool({
                    'file_path': html_path,
                    'caption': f" {paper.title[:50]} - Interactive HTML Slides"
                })

                if file_result.get('success'):
                    logger.info(f" Sent HTML slides to Telegram")
                else:
                    logger.error(f" HTML slides send failed: {file_result.get('error')}")

            if not pdf_path or not Path(pdf_path).exists():
                # Fallback: Send markdown
                temp_path = Path(f"/tmp/arxiv_{paper.arxiv_id.replace('.', '_')}_learning.md")
                with open(temp_path, 'w') as f:
                    f.write(f"# Learning: {paper.title}\n\n")
                    f.write(f"**ArXiv ID:** {paper.arxiv_id}\n")
                    f.write(f"**Authors:** {', '.join(paper.authors)}\n\n")
                    f.write("---\n\n")
                    f.write(full_content)

                file_result = await send_telegram_file_tool({
                    'file_path': str(temp_path),
                    'caption': f" Learning content for {paper.arxiv_id}"
                })

                if file_result.get('success'):
                    logger.info(f" Sent markdown to Telegram")

                try:
                    temp_path.unlink()
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            import traceback
            traceback.print_exc()

    def seed_gold_standards(self):
        """
        Seed default gold standards for paper learning evaluation.

        This establishes baseline quality expectations for:
        - Concept extraction
        - Content quality
        - Engagement metrics
        """
        self._init_agents()

        # Gold standard for a well-learned paper
        self.add_gold_standard(
            task_type='paper_learning',
            input_data={'arxiv_id': 'any'},
            expected_output={
                'concepts_count': 5,  # At least 5 concepts
                'bingo_moments': 3,   # At least 3 key insights
                'has_hook': True,
                'has_summary': True,
                'has_examples': True,
                'word_count': 1000    # At least 1000 words
            },
            evaluation_criteria={
                'concepts_count': 0.25,
                'bingo_moments': 0.20,
                'has_hook': 0.15,
                'has_summary': 0.15,
                'has_examples': 0.15,
                'word_count': 0.10
            }
        )

        logger.info(" Seeded gold standards for ArxivLearningSwarm")

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        stats = {
            'evaluations_count': len(self._evaluations) if self._evaluations else 0,
            'traces_count': len(self._traces) if self._traces else 0,
            'improvements_suggested': 0,
            'avg_score': 0.0
        }

        if self._evaluations:
            stats['avg_score'] = sum(e.overall_score for e in self._evaluations) / len(self._evaluations)

        if self._improvement_history:
            pending = self._improvement_history.get_pending_suggestions()
            stats['improvements_suggested'] = len(pending)

        return stats


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def learn_paper(
    paper_id: str = None,
    topic: str = None,
    depth: str = "standard",
    send_telegram: bool = False
) -> ArxivLearningResult:
    """
    One-liner paper learning.

    Usage:
        from core.swarms.arxiv_learning_swarm import learn_paper

        # By ID
        result = await learn_paper("1706.03762")  # Attention paper

        # By topic
        result = await learn_paper(topic="transformer attention")

        # With Telegram
        result = await learn_paper("2408.11574", send_telegram=True)
    """
    depth_enum = LearningDepth(depth) if isinstance(depth, str) else depth

    swarm = ArxivLearningSwarm()
    return await swarm.learn(paper_id=paper_id, topic=topic, depth=depth_enum, send_telegram=send_telegram)


def learn_paper_sync(
    paper_id: str = None,
    topic: str = None,
    depth: str = "standard",
    send_telegram: bool = False
) -> ArxivLearningResult:
    """Synchronous paper learning."""
    return asyncio.run(learn_paper(paper_id=paper_id, topic=topic, depth=depth, send_telegram=send_telegram))


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'ArxivLearningSwarm',
    'ArxivLearningConfig',
    'ArxivLearningResult',
    'LearningContent',
    'LearningSection',
    'Concept',
    'PaperInfo',
    'LearningDepth',
    'ContentStyle',
    'AudienceLevel',
    'learn_paper',
    'learn_paper_sync',
    # Agents
    'PaperFetcherAgent',
    'ConceptExtractorAgent',
    'IntuitionBuilderAgent',
    'MathSimplifierAgent',
    'ExampleGeneratorAgent',
    'ProgressiveBuilderAgent',
    'ContentPolisherAgent',
]
