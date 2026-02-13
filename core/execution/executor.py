"""
Unified Executor - V3 Core
===========================

Single entry point for all execution tiers.
Routes to appropriate tier based on config or auto-detection.

Tier 1 (DIRECT):    Single LLM call - implemented here
Tier 2 (AGENTIC):   Planning + orchestration - implemented here
Tier 3 (LEARNING):  Memory + validation - implemented here
Tier 4 (RESEARCH):  Delegates to V2 SwarmManager - NO BREAKAGE
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional, List, Callable
from datetime import datetime

from .types import (
    ExecutionConfig,
    ExecutionTier,
    ExecutionResult,
    ExecutionPlan,
    ExecutionStep,
    ValidationResult,
    MemoryContext,
)
from .tier_detector import TierDetector

logger = logging.getLogger(__name__)


class UnifiedExecutor:
    """
    V3 Unified Executor.

    Single entry point that routes to appropriate tier.
    Zero breakage: V2 code paths preserved as Tier 4.
    """

    def __init__(
        self,
        config: Optional[ExecutionConfig] = None,
        registry=None,
        provider=None,
    ):
        """
        Initialize executor.

        Args:
            config: Default execution config
            registry: UnifiedRegistry instance (lazy-loaded if None)
            provider: LLM provider instance (lazy-loaded if None)
        """
        self.config = config or ExecutionConfig()
        self._registry = registry
        self._provider = provider
        self._detector = TierDetector()

        # Lazy-loaded components
        self._planner = None
        self._memory = None
        self._validator = None

        logger.info("UnifiedExecutor initialized (V3)")

    @property
    def registry(self):
        """Lazy-load UnifiedRegistry."""
        if self._registry is None:
            from Jotty.core.registry import get_unified_registry
            self._registry = get_unified_registry()
        return self._registry

    @property
    def provider(self):
        """Lazy-load LLM provider."""
        if self._provider is None:
            from Jotty.core.foundation.unified_lm_provider import UnifiedLMProvider
            self._provider = UnifiedLMProvider(
                provider=self.config.provider,
                model=self.config.model,
            )
        return self._provider

    @property
    def planner(self):
        """Lazy-load AgenticPlanner."""
        if self._planner is None:
            from Jotty.core.agents.agentic_planner import AgenticPlanner
            self._planner = AgenticPlanner()
        return self._planner

    @property
    def memory(self):
        """Lazy-load memory backend."""
        if self._memory is None:
            self._memory = self._create_memory_backend()
        return self._memory

    @property
    def validator(self):
        """Lazy-load validator."""
        if self._validator is None:
            from Jotty.core.agents.inspector import InspectorAgent
            self._validator = InspectorAgent(role="auditor")
        return self._validator

    async def execute(
        self,
        goal: str,
        config: Optional[ExecutionConfig] = None,
        status_callback: Optional[Callable] = None,
        **kwargs
    ) -> ExecutionResult:
        """
        Execute task with appropriate tier.

        Args:
            goal: Task description
            config: Execution config (overrides default)
            status_callback: Optional progress callback
            **kwargs: Additional arguments passed to tier executor

        Returns:
            ExecutionResult with tier-specific data
        """
        start_time = time.time()
        config = config or self.config

        # Auto-detect tier if not specified
        if config.tier is None:
            config.tier = self._detector.detect(goal)
            logger.info(f"Auto-detected tier: {config.tier.name}")

        # Route to appropriate tier
        try:
            if config.tier == ExecutionTier.DIRECT:
                result = await self._execute_tier1(goal, config, status_callback, **kwargs)
            elif config.tier == ExecutionTier.AGENTIC:
                result = await self._execute_tier2(goal, config, status_callback, **kwargs)
            elif config.tier == ExecutionTier.LEARNING:
                result = await self._execute_tier3(goal, config, status_callback, **kwargs)
            else:  # RESEARCH
                result = await self._execute_tier4(goal, config, status_callback, **kwargs)

            # Set timing
            result.latency_ms = (time.time() - start_time) * 1000
            result.completed_at = datetime.now()

            logger.info(f"Execution complete: {result}")
            return result

        except Exception as e:
            logger.error(f"Execution failed: {e}", exc_info=True)
            return ExecutionResult(
                output=None,
                tier=config.tier,
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    # =========================================================================
    # TIER 1: DIRECT - Single LLM call
    # =========================================================================

    async def _execute_tier1(
        self,
        goal: str,
        config: ExecutionConfig,
        status_callback: Optional[Callable],
        **kwargs
    ) -> ExecutionResult:
        """
        Tier 1: Direct LLM call with tools.

        Fast path for simple queries:
        1. Discover skills
        2. Convert to Claude tools
        3. Single LLM call
        4. Return result

        Expected: 1 LLM call, 1-2s latency, $0.01 cost
        """
        logger.info(f"[Tier 1: DIRECT] Executing: {goal[:50]}...")

        if status_callback:
            status_callback("direct", "Discovering skills...")

        # Discover relevant skills
        discovery = self.registry.discover_for_task(goal)
        skill_names = discovery.get('skills', [])[:5]  # Limit to top 5

        if status_callback:
            status_callback("direct", f"Found {len(skill_names)} skills")

        # Get Claude-format tools
        tools = []
        if skill_names:
            tools = self.registry.get_claude_tools(skill_names)

        if status_callback:
            status_callback("direct", "Calling LLM...")

        # Single LLM call
        llm_start = time.time()
        response = await self.provider.generate(
            prompt=goal,
            tools=tools,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        llm_time = (time.time() - llm_start) * 1000

        # Estimate cost (rough: $0.015/1K tokens for Claude Sonnet)
        tokens = response.get('usage', {}).get('total_tokens', 500)
        cost = (tokens / 1000) * 0.015

        return ExecutionResult(
            output=response.get('content', response),
            tier=ExecutionTier.DIRECT,
            success=True,
            llm_calls=1,
            latency_ms=llm_time,
            cost_usd=cost,
            metadata={
                'skills_discovered': skill_names,
                'tools_used': len(tools),
                'tokens': tokens,
            }
        )

    # =========================================================================
    # TIER 2: AGENTIC - Planning + Orchestration
    # =========================================================================

    async def _execute_tier2(
        self,
        goal: str,
        config: ExecutionConfig,
        status_callback: Optional[Callable],
        **kwargs
    ) -> ExecutionResult:
        """
        Tier 2: Agentic execution with planning.

        Steps:
        1. Create execution plan (LLM call)
        2. Execute steps sequentially or in parallel
        3. Aggregate results

        Expected: 3-5 LLM calls, 3-5s latency, $0.03 cost
        """
        logger.info(f"[Tier 2: AGENTIC] Executing: {goal[:50]}...")

        total_llm_calls = 0
        total_cost = 0.0

        if status_callback:
            status_callback("planning", "Creating execution plan...")

        # Step 1: Plan
        plan_result = await self.planner.plan(goal)
        total_llm_calls += 1
        total_cost += 0.01  # Rough estimate

        # Convert planner output to ExecutionPlan
        plan = self._parse_plan(goal, plan_result)

        if status_callback:
            status_callback("planning", f"Plan created: {len(plan.steps)} steps")

        # Step 2: Execute steps
        results = []
        for step in plan.steps:
            if status_callback:
                status_callback("executing", f"Step {step.step_num}: {step.description}")

            step.started_at = datetime.now()

            try:
                # Execute step
                step_result = await self._execute_step(step, config)
                step.result = step_result.get('output')
                step.completed_at = datetime.now()

                total_llm_calls += step_result.get('llm_calls', 1)
                total_cost += step_result.get('cost', 0.01)

                results.append(step_result)

            except Exception as e:
                logger.error(f"Step {step.step_num} failed: {e}")
                step.error = str(e)
                step.completed_at = datetime.now()

        # Step 3: Aggregate
        final_output = self._aggregate_results(results, goal)

        return ExecutionResult(
            output=final_output,
            tier=ExecutionTier.AGENTIC,
            success=all(s.is_complete and not s.error for s in plan.steps),
            llm_calls=total_llm_calls,
            cost_usd=total_cost,
            plan=plan,
            steps=plan.steps,
        )

    # =========================================================================
    # TIER 3: LEARNING - Memory + Validation
    # =========================================================================

    async def _execute_tier3(
        self,
        goal: str,
        config: ExecutionConfig,
        status_callback: Optional[Callable],
        **kwargs
    ) -> ExecutionResult:
        """
        Tier 3: Learning execution with memory and validation.

        Steps:
        1. Retrieve memory context
        2. Enrich goal with context
        3. Execute with Tier 2 (agentic)
        4. Validate result
        5. Store in memory

        Expected: 5-10 LLM calls, 5-10s latency, $0.06 cost
        """
        logger.info(f"[Tier 3: LEARNING] Executing: {goal[:50]}...")

        total_llm_calls = 0
        total_cost = 0.0
        memory_context = None

        # Step 1: Retrieve memory
        if config.memory_backend != "none":
            if status_callback:
                status_callback("memory", "Retrieving context...")

            memory_start = time.time()
            memory_context = await self._retrieve_memory(goal, config)
            memory_time = (time.time() - memory_start) * 1000

            if memory_context and memory_context.total_retrieved > 0:
                logger.info(f"Retrieved {memory_context.total_retrieved} memory entries")

        # Step 2: Enrich goal
        enriched_goal = self._enrich_with_memory(goal, memory_context)

        # Step 3: Execute with Tier 2
        if status_callback:
            status_callback("executing", "Running agentic execution...")

        tier2_config = ExecutionConfig(
            tier=ExecutionTier.AGENTIC,
            max_planning_depth=config.max_planning_depth,
            enable_parallel_execution=config.enable_parallel_execution,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        result = await self._execute_tier2(enriched_goal, tier2_config, status_callback, **kwargs)
        total_llm_calls += result.llm_calls
        total_cost += result.cost_usd

        # Step 4: Validate
        validation = None
        if config.enable_validation and result.success:
            if status_callback:
                status_callback("validating", "Validating result...")

            validation = await self._validate_result(goal, result, config)
            total_llm_calls += 1
            total_cost += 0.01

            # Retry if validation fails and retries enabled
            if not validation.success and config.validation_retries > 0:
                logger.info(f"Validation failed, retrying... ({config.validation_retries} attempts)")

                for attempt in range(config.validation_retries):
                    if status_callback:
                        status_callback("retrying", f"Retry {attempt + 1}/{config.validation_retries}")

                    feedback_goal = f"{goal}\n\nPrevious attempt feedback: {validation.feedback}"
                    retry_result = await self._execute_tier2(
                        feedback_goal, tier2_config, status_callback, **kwargs
                    )

                    total_llm_calls += retry_result.llm_calls
                    total_cost += retry_result.cost_usd

                    retry_validation = await self._validate_result(goal, retry_result, config)
                    total_llm_calls += 1
                    total_cost += 0.01

                    if retry_validation.success:
                        result = retry_result
                        validation = retry_validation
                        break

        # Step 5: Store memory
        if config.memory_backend != "none":
            if status_callback:
                status_callback("memory", "Storing result...")

            await self._store_memory(goal, result, validation, config)

        return ExecutionResult(
            output=result.output,
            tier=ExecutionTier.LEARNING,
            success=validation.success if validation else result.success,
            llm_calls=total_llm_calls,
            cost_usd=total_cost,
            plan=result.plan,
            steps=result.steps,
            validation=validation,
            used_memory=memory_context is not None and memory_context.total_retrieved > 0,
            memory_context=memory_context,
        )

    # =========================================================================
    # TIER 4: RESEARCH - Delegate to V2 (NO BREAKAGE)
    # =========================================================================

    async def _execute_tier4(
        self,
        goal: str,
        config: ExecutionConfig,
        status_callback: Optional[Callable],
        **kwargs
    ) -> ExecutionResult:
        """
        Tier 4: Research - Full V2 features.

        Delegates to existing V2 SwarmManager.
        ZERO changes to V2 code - complete backwards compatibility.

        Expected: 20-40 LLM calls, 10-30s latency, $0.15 cost
        """
        logger.info(f"[Tier 4: RESEARCH] Delegating to V2 SwarmManager...")

        if status_callback:
            status_callback("research", "Initializing V2 SwarmManager...")

        # Import V2 (existing code, no modifications)
        from Jotty.core.orchestration.v2 import SwarmManager
        from Jotty.core.foundation.data_structures import JottyConfig

        # Convert V3 config to V2 format
        v2_config_dict = config.to_v2_config()
        v2_config = JottyConfig(**v2_config_dict)

        # Create SwarmManager (existing V2 code)
        swarm_manager = SwarmManager(config=v2_config)

        if status_callback:
            status_callback("research", "Executing with V2...")

        # Execute with V2 (existing code path)
        v2_result = await swarm_manager.run(
            goal=goal,
            status_callback=status_callback,
            **kwargs
        )

        # Wrap V2 result in V3 format
        return ExecutionResult(
            output=v2_result.output,
            tier=ExecutionTier.RESEARCH,
            success=v2_result.success,
            llm_calls=v2_result.metadata.get('llm_calls', 20),
            latency_ms=v2_result.metadata.get('latency_ms', 0),
            cost_usd=v2_result.metadata.get('cost_usd', 0.15),
            v2_episode=v2_result,  # Include full V2 data
            learning_data=v2_result.metadata.get('learning_data', {}),
            metadata={'v2_mode': True},
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _parse_plan(self, goal: str, plan_result: Any) -> ExecutionPlan:
        """Convert planner output to ExecutionPlan."""
        # Extract steps from planner result
        # This depends on AgenticPlanner output format
        steps_data = plan_result.get('steps', [])

        steps = []
        for i, step_data in enumerate(steps_data):
            step = ExecutionStep(
                step_num=i + 1,
                description=step_data.get('description', f'Step {i+1}'),
                skill=step_data.get('skill'),
                depends_on=step_data.get('depends_on', []),
                can_parallelize=step_data.get('can_parallelize', False),
            )
            steps.append(step)

        return ExecutionPlan(
            goal=goal,
            steps=steps,
            estimated_cost=plan_result.get('estimated_cost', 0.0),
            estimated_time_ms=plan_result.get('estimated_time_ms', 0.0),
        )

    async def _execute_step(self, step: ExecutionStep, config: ExecutionConfig) -> Dict[str, Any]:
        """Execute a single step."""
        # If step has a specific skill, use it
        if step.skill:
            skill = self.registry.get_skill(step.skill)
            tools = skill.to_claude_tools()
        else:
            # Discover skills for this step
            discovery = self.registry.discover_for_task(step.description)
            skill_names = discovery.get('skills', [])[:3]
            tools = self.registry.get_claude_tools(skill_names) if skill_names else []

        # Execute step
        response = await self.provider.generate(
            prompt=step.description,
            tools=tools,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        return {
            'output': response.get('content', response),
            'llm_calls': 1,
            'cost': 0.01,
        }

    def _aggregate_results(self, results: List[Dict], goal: str) -> str:
        """Aggregate step results into final output."""
        if not results:
            return "No results generated."

        if len(results) == 1:
            return results[0].get('output', '')

        # Multiple results: concatenate with context
        aggregated = f"Results for: {goal}\n\n"
        for i, result in enumerate(results, 1):
            output = result.get('output', '')
            aggregated += f"Step {i}:\n{output}\n\n"

        return aggregated.strip()

    async def _retrieve_memory(self, goal: str, config: ExecutionConfig) -> Optional[MemoryContext]:
        """Retrieve relevant memory entries."""
        # Simple memory retrieval (JSON backend for now)
        # TODO: Add Redis/Postgres backends
        try:
            entries = await self.memory.retrieve(goal, limit=5)
            if not entries:
                return None

            return MemoryContext(
                entries=entries,
                relevance_scores=[e.get('score', 0.0) for e in entries],
                total_retrieved=len(entries),
                retrieval_time_ms=10.0,  # Rough estimate
            )
        except Exception as e:
            logger.warning(f"Memory retrieval failed: {e}")
            return None

    def _enrich_with_memory(self, goal: str, context: Optional[MemoryContext]) -> str:
        """Enrich goal with memory context."""
        if not context or not context.entries:
            return goal

        # Add relevant memories to goal
        enriched = f"{goal}\n\nRelevant past experience:\n"
        for entry in context.entries[:3]:  # Top 3
            enriched += f"- {entry.get('summary', entry.get('result', ''))}\n"

        return enriched

    async def _validate_result(
        self,
        goal: str,
        result: ExecutionResult,
        config: ExecutionConfig
    ) -> ValidationResult:
        """Validate execution result."""
        # Use InspectorAgent (existing V2 component)
        validation_prompt = f"""
Task: {goal}

Result: {str(result.output)[:500]}

Is this result correct and complete? Provide:
1. Success (yes/no)
2. Confidence (0-1)
3. Feedback (brief)
4. Reasoning
"""

        response = await self.validator.validate(validation_prompt)

        return ValidationResult(
            success=response.get('success', True),
            confidence=response.get('confidence', 0.8),
            feedback=response.get('feedback', ''),
            reasoning=response.get('reasoning', ''),
        )

    async def _store_memory(
        self,
        goal: str,
        result: ExecutionResult,
        validation: Optional[ValidationResult],
        config: ExecutionConfig
    ):
        """Store result in memory."""
        try:
            await self.memory.store(
                goal=goal,
                result=str(result.output)[:1000],
                success=validation.success if validation else result.success,
                confidence=validation.confidence if validation else 1.0,
                ttl_hours=config.memory_ttl_hours,
            )
        except Exception as e:
            logger.warning(f"Memory storage failed: {e}")

    def _create_memory_backend(self):
        """Create memory backend based on config."""
        backend = self.config.memory_backend

        if backend == "json":
            from Jotty.core.execution.memory.json_memory import JSONMemory
            return JSONMemory()
        elif backend == "redis":
            from Jotty.core.execution.memory.redis_memory import RedisMemory
            return RedisMemory()
        else:
            # Default: no-op memory
            from Jotty.core.execution.memory.noop_memory import NoOpMemory
            return NoOpMemory()
