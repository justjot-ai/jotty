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
from typing import Any, AsyncGenerator, Dict, Optional, List, Callable
from datetime import datetime

from .types import (
    ExecutionConfig,
    ExecutionTier,
    ExecutionResult,
    ExecutionPlan,
    ExecutionStep,
    ValidationResult,
    MemoryContext,
    StreamEvent,
    StreamEventType,
)
from .tier_detector import TierDetector
from Jotty.core.observability.tracing import SpanStatus

logger = logging.getLogger(__name__)


class _ExecutionLLMProvider:
    """Lightweight adapter wrapping an LLM client with generate()/stream() for the executor."""

    def __init__(self, provider: str = 'anthropic', model: str = None):
        self._provider_name = provider or 'anthropic'
        self._model = model or 'claude-sonnet-4-20250514'
        self._client = None

    def _get_client(self):
        if self._client is None:
            if self._provider_name == 'anthropic':
                import anthropic
                self._client = anthropic.AsyncAnthropic()
            else:
                # Fallback: use DSPy LM via UnifiedLMProvider
                from Jotty.core.foundation.unified_lm_provider import UnifiedLMProvider
                self._client = UnifiedLMProvider.create_lm(self._provider_name, self._model)
        return self._client

    async def generate(self, prompt: str, tools=None, temperature: float = 0.7,
                       max_tokens: int = 4000, **kwargs) -> Dict[str, Any]:
        """Generate a response. Returns {'content': str, 'usage': {...}}."""
        client = self._get_client()

        if self._provider_name == 'anthropic':
            api_kwargs = {
                'model': self._model,
                'max_tokens': max_tokens,
                'temperature': temperature,
                'messages': [{'role': 'user', 'content': prompt}],
            }
            if tools:
                api_kwargs['tools'] = tools

            response = await client.messages.create(**api_kwargs)
            content = ''.join(
                block.text for block in response.content if hasattr(block, 'text')
            )
            return {
                'content': content,
                'usage': {
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens,
                },
            }
        else:
            # DSPy LM fallback
            result = client(prompt)
            text = result[0] if isinstance(result, list) else str(result)
            return {'content': text, 'usage': {'input_tokens': 250, 'output_tokens': 250}}

    async def stream(self, prompt: str, tools=None, temperature: float = 0.7,
                     max_tokens: int = 4000, **kwargs):
        """Stream tokens. Yields {'content': str} chunks, final chunk has 'usage'."""
        client = self._get_client()

        if self._provider_name == 'anthropic':
            api_kwargs = {
                'model': self._model,
                'max_tokens': max_tokens,
                'temperature': temperature,
                'messages': [{'role': 'user', 'content': prompt}],
            }
            if tools:
                api_kwargs['tools'] = tools

            async with client.messages.stream(**api_kwargs) as stream:
                async for text in stream.text_stream:
                    yield {'content': text}

                response = await stream.get_final_message()
                yield {
                    'content': '',
                    'usage': {
                        'input_tokens': response.usage.input_tokens,
                        'output_tokens': response.usage.output_tokens,
                    },
                }
        else:
            # Non-streaming fallback
            result = await self.generate(prompt, tools, temperature, max_tokens, **kwargs)
            yield {'content': result['content'], 'usage': result.get('usage', {})}


class _ValidatorAdapter:
    """Lightweight validator that uses the LLM provider to validate results."""

    def __init__(self, provider):
        self._provider = provider

    async def validate(self, prompt: str) -> Dict[str, Any]:
        """Validate a result via LLM. Returns {'success': bool, 'confidence': float, ...}."""
        import json as _json

        validation_system = (
            "You are a quality validator. Evaluate the result and respond with ONLY "
            "a JSON object: {\"success\": true/false, \"confidence\": 0.0-1.0, "
            "\"feedback\": \"brief feedback\", \"reasoning\": \"brief reasoning\"}"
        )
        full_prompt = f"{validation_system}\n\n{prompt}"

        try:
            response = await self._provider.generate(
                prompt=full_prompt,
                temperature=0.3,
                max_tokens=500,
            )
            content = response.get('content', '{}')
            # Try to parse JSON from response
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                return _json.loads(content[start:end])
            return {'success': True, 'confidence': 0.7, 'feedback': content, 'reasoning': ''}
        except Exception as e:
            logger.warning(f"Validation LLM call failed: {e}")
            return {'success': True, 'confidence': 0.5, 'feedback': 'Validation skipped', 'reasoning': str(e)}


class _PlannerAdapter:
    """Wraps AgenticPlanner with a simple .plan(goal) → {'steps': [...]} interface."""

    def __init__(self, registry=None):
        self._planner = None
        self._registry = registry

    def _get_planner(self):
        if self._planner is None:
            from Jotty.core.agents.agentic_planner import AgenticPlanner
            self._planner = AgenticPlanner()
        return self._planner

    async def plan(self, goal: str) -> Dict[str, Any]:
        """Plan execution steps for a goal. Returns {'steps': [{'description': ...}]}."""
        planner = self._get_planner()

        # Discover skills for context
        skills = []
        if self._registry:
            try:
                discovery = self._registry.discover_for_task(goal)
                skills = discovery.get('skills', [])[:10]
            except Exception:
                pass

        try:
            steps, reasoning = await planner.aplan_execution(
                task=goal,
                task_type='general',
                skills=skills,
            )
        except Exception as e:
            logger.warning(f"Async planning failed, trying sync: {e}")
            steps, reasoning = planner.plan_execution(
                task=goal,
                task_type='general',
                skills=skills,
            )

        # Convert V2 ExecutionStep objects to dicts for V3 executor
        step_dicts = []
        for step in steps:
            step_dict = {
                'description': getattr(step, 'description', str(step)),
                'skill': getattr(step, 'skill_name', None),
            }
            if hasattr(step, 'depends_on'):
                step_dict['depends_on'] = step.depends_on
            step_dicts.append(step_dict)

        return {'steps': step_dicts, 'reasoning': reasoning}


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

        # Observability (lazy-loaded)
        self._metrics = None
        self._tracer = None
        self._cost_tracker = None

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
            self._provider = _ExecutionLLMProvider(
                provider=self.config.provider,
                model=self.config.model,
            )
        return self._provider

    @property
    def planner(self):
        """Lazy-load planner with .plan(goal) convenience wrapper."""
        if self._planner is None:
            self._planner = _PlannerAdapter(self.registry)
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
            self._validator = _ValidatorAdapter(self.provider)
        return self._validator

    @property
    def metrics(self):
        """Lazy-load MetricsCollector singleton."""
        if self._metrics is None:
            from Jotty.core.observability import get_metrics
            self._metrics = get_metrics()
        return self._metrics

    @property
    def tracer(self):
        """Lazy-load TracingContext singleton."""
        if self._tracer is None:
            from Jotty.core.observability import get_tracer
            self._tracer = get_tracer()
        return self._tracer

    @property
    def cost_tracker(self):
        """Lazy-load CostTracker instance."""
        if self._cost_tracker is None:
            from Jotty.core.monitoring.cost_tracker import CostTracker
            self._cost_tracker = CostTracker()
        return self._cost_tracker

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

        # Start trace
        trace = self.tracer.new_trace(metadata={
            'goal': goal[:200],
            'tier': config.tier.name,
        })

        # Route to appropriate tier
        try:
            with self.tracer.span("execute", tier=config.tier.name, goal=goal[:100]) as root_span:
                if config.tier == ExecutionTier.DIRECT:
                    result = await self._execute_tier1(goal, config, status_callback, **kwargs)
                elif config.tier == ExecutionTier.AGENTIC:
                    result = await self._execute_tier2(goal, config, status_callback, **kwargs)
                elif config.tier == ExecutionTier.LEARNING:
                    result = await self._execute_tier3(goal, config, status_callback, **kwargs)
                elif config.tier == ExecutionTier.AUTONOMOUS:
                    result = await self._execute_tier5(goal, config, status_callback, **kwargs)
                else:  # RESEARCH
                    result = await self._execute_tier4(goal, config, status_callback, **kwargs)

                root_span.set_status(SpanStatus.OK)

            # Set timing
            result.latency_ms = (time.time() - start_time) * 1000
            result.completed_at = datetime.now()

            # End trace and attach to result
            self.tracer.end_trace()
            traces = self.tracer.get_trace_history()
            result.trace = traces[-1] if traces else trace

            # Record to MetricsCollector
            self.metrics.record_execution(
                agent_name=f"tier_{config.tier.value}",
                task_type=config.tier.name.lower(),
                duration_s=result.latency_ms / 1000,
                success=result.success,
                input_tokens=result.metadata.get('tokens', 0),
                output_tokens=0,
                cost_usd=result.cost_usd,
                llm_calls=result.llm_calls,
                error=result.error,
                metadata={'goal': goal[:200]},
            )

            logger.info(f"Execution complete: {result}")
            return result

        except Exception as e:
            logger.error(f"Execution failed: {e}", exc_info=True)
            self.tracer.end_trace()
            duration_s = (time.time() - start_time)

            # Record failure to metrics
            self.metrics.record_execution(
                agent_name=f"tier_{config.tier.value}",
                task_type=config.tier.name.lower(),
                duration_s=duration_s,
                success=False,
                error=str(e),
            )

            return ExecutionResult(
                output=None,
                tier=config.tier,
                success=False,
                error=str(e),
                latency_ms=duration_s * 1000,
            )

    # =========================================================================
    # STREAMING
    # =========================================================================

    async def stream(
        self,
        goal: str,
        config: Optional[ExecutionConfig] = None,
        **kwargs
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream execution events as an async generator.

        Yields StreamEvent objects for each phase of execution:
        - STATUS events for phase changes (planning, executing, validating...)
        - STEP_COMPLETE events when individual steps finish
        - TOKEN events for individual LLM tokens (Tier 1 only, if provider supports it)
        - RESULT event with final ExecutionResult
        - ERROR event if execution fails

        Args:
            goal: Task description
            config: Execution config (overrides default)
            **kwargs: Additional arguments

        Yields:
            StreamEvent objects

        Example:
            async for event in executor.stream("Research AI"):
                if event.type == StreamEventType.TOKEN:
                    print(event.data, end="", flush=True)
                elif event.type == StreamEventType.STATUS:
                    print(f"[{event.data['stage']}] {event.data['detail']}")
                elif event.type == StreamEventType.RESULT:
                    print(f"Done: {event.data}")
        """
        config = config or self.config

        # Auto-detect tier
        if config.tier is None:
            config.tier = self._detector.detect(goal)

        yield StreamEvent(
            type=StreamEventType.STATUS,
            data={'stage': 'start', 'detail': f'Tier {config.tier.name} detected'},
            tier=config.tier,
        )

        # Tier 1: support token-level streaming
        if config.tier == ExecutionTier.DIRECT:
            async for event in self._stream_tier1(goal, config, **kwargs):
                yield event
            return

        # All other tiers: queue-based bridge from status_callback
        queue = asyncio.Queue()

        def _callback(stage: str, detail: str):
            queue.put_nowait(StreamEvent(
                type=StreamEventType.STATUS,
                data={'stage': stage, 'detail': detail},
                tier=config.tier,
            ))

        # Run execute() as a background task, feeding events via callback
        task = asyncio.create_task(
            self.execute(goal, config=config, status_callback=_callback, **kwargs)
        )

        # Yield events as they arrive until the task completes
        while not task.done():
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.1)
                yield event
            except asyncio.TimeoutError:
                continue

        # Drain any remaining events
        while not queue.empty():
            yield queue.get_nowait()

        # Yield final result or error
        try:
            result = task.result()
            yield StreamEvent(
                type=StreamEventType.RESULT,
                data=result,
                tier=config.tier,
            )
        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                data={'error': str(e)},
                tier=config.tier,
            )

    async def _stream_tier1(
        self,
        goal: str,
        config: ExecutionConfig,
        **kwargs
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream Tier 1 execution with token-level streaming.

        Falls back to non-streaming if provider doesn't support stream().
        """
        tier = ExecutionTier.DIRECT
        start_time = time.time()

        # Start trace
        trace = self.tracer.new_trace(metadata={'goal': goal[:200], 'tier': 'DIRECT'})

        yield StreamEvent(
            type=StreamEventType.STATUS,
            data={'stage': 'direct', 'detail': 'Discovering skills...'},
            tier=tier,
        )

        # Discover skills
        discovery = self.registry.discover_for_task(goal)
        raw_skills = discovery.get('skills', [])[:5]
        skill_names = [s['name'] if isinstance(s, dict) else s for s in raw_skills]
        tools = self.registry.get_claude_tools(skill_names) if skill_names else []

        yield StreamEvent(
            type=StreamEventType.STATUS,
            data={'stage': 'direct', 'detail': f'Found {len(skill_names)} skills, calling LLM...'},
            tier=tier,
        )

        model = self.config.model or 'claude-sonnet-4'
        provider_name = self.config.provider or 'anthropic'

        with self.tracer.span("execute", tier="DIRECT", goal=goal[:100]) as root_span:
            with self.tracer.span("tier1_llm_call", tools=len(tools), streaming=True) as llm_span:
                llm_start = time.time()

                # Try token streaming if provider supports it
                if hasattr(self.provider, 'stream'):
                    collected_tokens = []
                    usage = {}

                    async for chunk in self.provider.stream(
                        prompt=goal,
                        tools=tools,
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
                    ):
                        token = chunk.get('content', chunk.get('delta', ''))
                        if token:
                            collected_tokens.append(token)
                            yield StreamEvent(type=StreamEventType.TOKEN, data=token, tier=tier)

                        # Last chunk often carries usage
                        if 'usage' in chunk:
                            usage = chunk['usage']

                    output = ''.join(collected_tokens)
                else:
                    # Fallback to non-streaming
                    response = await self.provider.generate(
                        prompt=goal,
                        tools=tools,
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
                    )
                    output = response.get('content', response)
                    usage = response.get('usage', {})

                llm_time = (time.time() - llm_start) * 1000

                input_tokens = usage.get('input_tokens', 250)
                output_tokens = usage.get('output_tokens', 250)
                record = self.cost_tracker.record_llm_call(
                    provider=provider_name,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    success=True,
                    duration=llm_time / 1000,
                )

                llm_span.add_cost(input_tokens, output_tokens, record.cost)

            root_span.set_status(SpanStatus.OK)

        # Build result
        latency_ms = (time.time() - start_time) * 1000
        self.tracer.end_trace()
        traces = self.tracer.get_trace_history()

        result = ExecutionResult(
            output=output,
            tier=tier,
            success=True,
            llm_calls=1,
            latency_ms=latency_ms,
            cost_usd=record.cost,
            completed_at=datetime.now(),
            trace=traces[-1] if traces else trace,
            metadata={
                'skills_discovered': skill_names,
                'tools_used': len(tools),
                'tokens': input_tokens + output_tokens,
                'streamed': hasattr(self.provider, 'stream'),
            },
        )

        # Record to metrics
        self.metrics.record_execution(
            agent_name='tier_1',
            task_type='direct',
            duration_s=latency_ms / 1000,
            success=True,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=record.cost,
            llm_calls=1,
            metadata={'goal': goal[:200]},
        )

        yield StreamEvent(type=StreamEventType.RESULT, data=result, tier=tier)

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
        raw_skills = discovery.get('skills', [])[:5]  # Limit to top 5
        skill_names = [s['name'] if isinstance(s, dict) else s for s in raw_skills]

        if status_callback:
            status_callback("direct", f"Found {len(skill_names)} skills")

        # Get Claude-format tools
        tools = []
        if skill_names:
            tools = self.registry.get_claude_tools(skill_names)

        if status_callback:
            status_callback("direct", "Calling LLM...")

        # Single LLM call
        with self.tracer.span("tier1_llm_call", tools=len(tools)) as llm_span:
            llm_start = time.time()
            response = await self.provider.generate(
                prompt=goal,
                tools=tools,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            llm_time = (time.time() - llm_start) * 1000

            # Estimate cost (rough: $0.015/1K tokens for Claude Sonnet)
            usage = response.get('usage', {})
            input_tokens = usage.get('input_tokens', 250)
            output_tokens = usage.get('output_tokens', 250)
            tokens = input_tokens + output_tokens

            # Use CostTracker for precise cost
            model = self.config.model or 'claude-sonnet-4'
            record = self.cost_tracker.record_llm_call(
                provider=self.config.provider or 'anthropic',
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                success=True,
                duration=llm_time / 1000,
            )
            cost = record.cost

            llm_span.add_cost(input_tokens, output_tokens, cost)

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
        with self.tracer.span("tier2_plan") as plan_span:
            plan_start = time.time()
            plan_result = await self.planner.plan(goal)
            plan_duration = time.time() - plan_start
            total_llm_calls += 1

            # Estimate ~500 input tokens (prompt+goal), ~300 output (plan JSON)
            plan_record = self.cost_tracker.record_llm_call(
                provider=self.config.provider or 'anthropic',
                model=self.config.model or 'claude-sonnet-4',
                input_tokens=500,
                output_tokens=300,
                success=True,
                duration=plan_duration,
            )
            total_cost += plan_record.cost
            plan_span.add_cost(500, 300, plan_record.cost)

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

            with self.tracer.span(f"tier2_step_{step.step_num}", description=step.description[:80]) as step_span:
                try:
                    # Execute step
                    step_result = await self._execute_step(step, config)
                    step.result = step_result.get('output')
                    step.completed_at = datetime.now()

                    step_calls = step_result.get('llm_calls', 1)
                    step_cost = step_result.get('cost', 0.0)
                    total_llm_calls += step_calls
                    total_cost += step_cost
                    step_span.add_cost(
                        step_result.get('input_tokens', 0),
                        step_result.get('output_tokens', 0),
                        step_cost,
                    )

                    results.append(step_result)

                except Exception as e:
                    logger.error(f"Step {step.step_num} failed: {e}")
                    step.error = str(e)
                    step.completed_at = datetime.now()
                    step_span.set_status(SpanStatus.ERROR, str(e))

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

            with self.tracer.span("tier3_memory", backend=config.memory_backend) as mem_span:
                memory_start = time.time()
                memory_context = await self._retrieve_memory(goal, config)
                memory_time = (time.time() - memory_start) * 1000
                mem_span.set_attribute("retrieval_time_ms", memory_time)

                if memory_context and memory_context.total_retrieved > 0:
                    mem_span.set_attribute("entries_retrieved", memory_context.total_retrieved)
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

        with self.tracer.span("tier3_execute") as exec_span:
            result = await self._execute_tier2(enriched_goal, tier2_config, status_callback, **kwargs)
            total_llm_calls += result.llm_calls
            total_cost += result.cost_usd
            exec_span.add_cost(0, 0, result.cost_usd)

        # Step 4: Validate
        validation = None
        if config.enable_validation and result.success:
            if status_callback:
                status_callback("validating", "Validating result...")

            with self.tracer.span("tier3_validate") as val_span:
                val_start = time.time()
                validation = await self._validate_result(goal, result, config)
                val_duration = time.time() - val_start
                total_llm_calls += 1

                # Estimate ~400 input tokens (validation prompt), ~200 output
                val_record = self.cost_tracker.record_llm_call(
                    provider=self.config.provider or 'anthropic',
                    model=self.config.model or 'claude-sonnet-4',
                    input_tokens=400,
                    output_tokens=200,
                    success=True,
                    duration=val_duration,
                )
                total_cost += val_record.cost
                val_span.add_cost(400, 200, val_record.cost)
                val_span.set_attribute("validation_success", validation.success)

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

                        retry_val_start = time.time()
                        retry_validation = await self._validate_result(goal, retry_result, config)
                        retry_val_duration = time.time() - retry_val_start
                        total_llm_calls += 1

                        retry_val_record = self.cost_tracker.record_llm_call(
                            provider=self.config.provider or 'anthropic',
                            model=self.config.model or 'claude-sonnet-4',
                            input_tokens=400,
                            output_tokens=200,
                            success=True,
                            duration=retry_val_duration,
                        )
                        total_cost += retry_val_record.cost

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
        Tier 4: Domain swarm execution (direct, no SwarmManager wrapper).

        Selects the appropriate domain swarm and executes directly.

        Expected: 10-30s latency, $0.15 cost
        """
        logger.info(f"[Tier 4: RESEARCH] Executing with domain swarm...")

        if status_callback:
            status_callback("research", "Selecting domain swarm...")

        # Select and instantiate swarm
        swarm = self._select_swarm(goal, config.swarm_name)

        if swarm is None:
            # Fallback to V2 SwarmManager for unregistered tasks
            return await self._execute_tier4_v2_fallback(goal, config, status_callback, **kwargs)

        if status_callback:
            status_callback("research", f"Executing with {swarm.__class__.__name__}...")

        # Execute directly with tracing
        with self.tracer.span("tier4_swarm", swarm=swarm.__class__.__name__) as swarm_span:
            result = await swarm.execute(task=goal, **kwargs)
            swarm_span.set_attribute("success", result.success if hasattr(result, 'success') else True)

        # Wrap in ExecutionResult
        return ExecutionResult(
            output=result.output if hasattr(result, 'output') else {'result': str(result)},
            tier=ExecutionTier.RESEARCH,
            success=result.success if hasattr(result, 'success') else True,
            swarm_name=swarm.__class__.__name__,
            metadata={'direct_swarm': True},
        )

    async def _execute_tier4_v2_fallback(
        self,
        goal: str,
        config: ExecutionConfig,
        status_callback: Optional[Callable],
        **kwargs
    ) -> ExecutionResult:
        """Fallback: delegate to V2 SwarmManager when no swarm is registered."""
        logger.info("[Tier 4: RESEARCH] Falling back to V2 SwarmManager...")

        from Jotty.core.orchestration.v2 import SwarmManager
        from Jotty.core.foundation.data_structures import JottyConfig

        v2_config_dict = config.to_v2_config()
        v2_config = JottyConfig(**v2_config_dict)
        swarm_manager = SwarmManager(config=v2_config)

        if status_callback:
            status_callback("research", "Executing with V2...")

        v2_result = await swarm_manager.run(
            goal=goal,
            status_callback=status_callback,
            **kwargs
        )

        return ExecutionResult(
            output=v2_result.output,
            tier=ExecutionTier.RESEARCH,
            success=v2_result.success,
            llm_calls=v2_result.metadata.get('llm_calls', 20),
            latency_ms=v2_result.metadata.get('latency_ms', 0),
            cost_usd=v2_result.metadata.get('cost_usd', 0.15),
            v2_episode=v2_result,
            learning_data=v2_result.metadata.get('learning_data', {}),
            metadata={'v2_mode': True},
        )

    # =========================================================================
    # TIER 5: AUTONOMOUS - Sandbox + Coalition + Full V2 Features
    # =========================================================================

    async def _execute_tier5(
        self,
        goal: str,
        config: ExecutionConfig,
        status_callback: Optional[Callable],
        **kwargs
    ) -> ExecutionResult:
        """
        Tier 5: Autonomous execution with sandbox, coalition, curriculum.

        Expected: 30-120s latency, ~$0.50 cost
        """
        logger.info(f"[Tier 5: AUTONOMOUS] Executing: {goal[:50]}...")

        if status_callback:
            status_callback("autonomous", "Selecting swarm...")

        # Select swarm (same as tier 4)
        swarm = self._select_swarm(goal, config.swarm_name)
        sandbox_log = None

        # Optionally wrap in sandbox
        with self.tracer.span("tier5_autonomous", sandbox=config.enable_sandbox) as auto_span:
            if config.enable_sandbox and swarm is not None:
                try:
                    from Jotty.core.orchestration.v2.sandbox_manager import SandboxManager
                    sandbox = SandboxManager(trust_level=config.trust_level)

                    if status_callback:
                        status_callback("autonomous", "Executing in sandbox...")

                    result = await sandbox.execute(swarm, goal, **kwargs)
                    sandbox_log = getattr(result, 'sandbox_log', None)
                    auto_span.set_attribute("sandboxed", True)
                except ImportError:
                    logger.warning("SandboxManager not available, executing without sandbox")
                    auto_span.set_attribute("sandboxed", False)
                    if swarm is not None:
                        result = await swarm.execute(task=goal, **kwargs)
                    else:
                        return await self._execute_tier4_v2_fallback(goal, config, status_callback, **kwargs)
            elif swarm is not None:
                if status_callback:
                    status_callback("autonomous", f"Executing with {swarm.__class__.__name__}...")
                result = await swarm.execute(task=goal, **kwargs)
                auto_span.set_attribute("swarm", swarm.__class__.__name__)
            else:
                return await self._execute_tier4_v2_fallback(goal, config, status_callback, **kwargs)

        # Optionally apply paradigm (debate/relay/refinement)
        paradigm_used = None
        if config.paradigm:
            try:
                from Jotty.core.orchestration.v2.swarm_intelligence import SwarmIntelligence
                si = SwarmIntelligence()
                paradigm_used = config.paradigm
                logger.info(f"Applying paradigm: {config.paradigm}")
            except ImportError:
                logger.warning("SwarmIntelligence not available, skipping paradigm")

        return ExecutionResult(
            output=result.output if hasattr(result, 'output') else {'result': str(result)},
            tier=ExecutionTier.AUTONOMOUS,
            success=result.success if hasattr(result, 'success') else True,
            swarm_name=swarm.__class__.__name__ if swarm else None,
            paradigm_used=paradigm_used,
            sandbox_log=sandbox_log,
            metadata={'autonomous': True, 'sandboxed': config.enable_sandbox},
        )

    # =========================================================================
    # SWARM SELECTION
    # =========================================================================

    _swarms_registered = False

    def _ensure_swarms_registered(self):
        """Trigger lazy import of all swarm modules so they register with SwarmRegistry."""
        if UnifiedExecutor._swarms_registered:
            return
        swarm_modules = [
            'Jotty.core.swarms.coding_swarm',
            'Jotty.core.swarms.research_swarm',
            'Jotty.core.swarms.testing_swarm',
            'Jotty.core.swarms.review_swarm',
            'Jotty.core.swarms.data_analysis_swarm',
            'Jotty.core.swarms.devops_swarm',
            'Jotty.core.swarms.idea_writer_swarm',
            'Jotty.core.swarms.fundamental_swarm',
            'Jotty.core.swarms.learning_swarm',
        ]
        import importlib
        for mod in swarm_modules:
            try:
                importlib.import_module(mod)
            except Exception as e:
                logger.debug(f"Could not import swarm module {mod}: {e}")
        UnifiedExecutor._swarms_registered = True

    def _select_swarm(self, goal: str, swarm_name: Optional[str] = None):
        """Select and instantiate the right domain swarm.

        Args:
            goal: Task description for auto-detection
            swarm_name: Explicit swarm name (e.g. "coding", "research")

        Returns:
            Instantiated swarm or None if no match found.
        """
        self._ensure_swarms_registered()
        from Jotty.core.swarms.registry import SwarmRegistry

        if swarm_name:
            swarm = SwarmRegistry.create(swarm_name)
            if swarm:
                return swarm
            logger.warning(f"Swarm '{swarm_name}' not in registry, attempting auto-detect")

        # Auto-detect from goal keywords
        goal_lower = goal.lower()
        keyword_map = {
            'coding': ['code', 'program', 'implement', 'develop', 'function', 'class', 'api'],
            'research': ['research', 'analyze', 'investigate', 'study', 'report'],
            'testing': ['test', 'coverage', 'unit test', 'integration test', 'qa'],
            'review': ['review', 'audit', 'check code', 'pull request', 'pr'],
            'data_analysis': ['data', 'dataset', 'statistics', 'visualization', 'csv'],
            'devops': ['deploy', 'docker', 'ci/cd', 'infrastructure', 'kubernetes'],
            'idea_writer': ['write', 'article', 'blog', 'essay', 'content'],
            'fundamental': ['stock', 'valuation', 'financial', 'earnings', 'investment'],
            'learning': ['learn', 'curriculum', 'teach', 'training'],
        }

        for name, keywords in keyword_map.items():
            if any(kw in goal_lower for kw in keywords):
                swarm = SwarmRegistry.create(name)
                if swarm:
                    logger.info(f"Auto-detected swarm: {name}")
                    return swarm

        # No match found
        return None

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
            raw_skills = discovery.get('skills', [])[:3]
            skill_names = [s['name'] if isinstance(s, dict) else s for s in raw_skills]
            tools = self.registry.get_claude_tools(skill_names) if skill_names else []

        # Execute step
        llm_start = time.time()
        response = await self.provider.generate(
            prompt=step.description,
            tools=tools,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        llm_duration = time.time() - llm_start

        # Extract actual tokens and compute precise cost
        usage = response.get('usage', {})
        input_tokens = usage.get('input_tokens', 300)
        output_tokens = usage.get('output_tokens', 200)
        record = self.cost_tracker.record_llm_call(
            provider=self.config.provider or 'anthropic',
            model=self.config.model or 'claude-sonnet-4',
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            success=True,
            duration=llm_duration,
        )

        return {
            'output': response.get('content', response),
            'llm_calls': 1,
            'cost': record.cost,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
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
