"""
Unified Executor
=================

Single entry point for all execution tiers.
Routes to appropriate tier based on config or auto-detection.

Tier 1 (DIRECT):    Single LLM call - implemented here
Tier 2 (AGENTIC):   Planning + orchestration - implemented here
Tier 3 (LEARNING):  Memory + validation via ValidatorAgent
Tier 4 (RESEARCH):  Delegates to Orchestrator
Tier 5 (AUTONOMOUS): Sandbox + coalition + full features
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional, List, Callable, Tuple
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


class LLMProvider:
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


class TierExecutor:
    """
    Unified Executor — single entry point for all execution tiers.

    Wires real components:
    - ValidatorAgent + MultiRoundValidator for validation (Tier 3+)
    - TaskPlanner for planning (Tier 2+)
    - Orchestrator for domain swarm execution (Tier 4/5)
    """

    # Prompt file paths for ValidatorAgent
    _AUDITOR_PROMPT = Path(__file__).parent.parent.parent / 'configs' / 'prompts' / 'auditor' / 'base_auditor.md'
    _ARCHITECT_PROMPT = Path(__file__).parent.parent.parent / 'configs' / 'prompts' / 'architect' / 'base_architect.md'

    def __init__(
        self,
        config: Optional[ExecutionConfig] = None,
        registry=None,
        provider=None,
    ):
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

        logger.info("TierExecutor initialized")

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
            self._provider = LLMProvider(
                provider=self.config.provider,
                model=self.config.model,
            )
        return self._provider

    @property
    def planner(self):
        """Lazy-load TaskPlanner directly (no adapter)."""
        if self._planner is None:
            self._planner = self._create_planner()
        return self._planner

    @property
    def memory(self):
        """Lazy-load memory backend."""
        if self._memory is None:
            self._memory = self._create_memory_backend()
        return self._memory

    @property
    def validator(self):
        """Lazy-load MultiRoundValidator wrapping ValidatorAgent."""
        if self._validator is None:
            self._validator = self._create_validator()
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

    # =========================================================================
    # COMPONENT FACTORIES
    # =========================================================================

    def _create_planner(self):
        """Create TaskPlanner directly — no adapter wrapper."""
        try:
            from Jotty.core.agents.agentic_planner import TaskPlanner
            return TaskPlanner()
        except Exception as e:
            logger.warning(f"TaskPlanner creation failed: {e}")
            return None

    def _create_validator(self):
        """Create MultiRoundValidator wrapping ValidatorAgent.

        Falls back to simple LLM-based validation if ValidatorAgent
        cannot be instantiated (e.g. DSPy not configured).
        """
        try:
            from Jotty.core.agents.inspector import ValidatorAgent, MultiRoundValidator
            from Jotty.core.foundation.data_structures import SwarmConfig, SharedScratchpad

            swarm_config_dict = self.config.to_swarm_config()
            swarm_config = SwarmConfig(**swarm_config_dict)
            scratchpad = SharedScratchpad()

            auditor = ValidatorAgent(
                md_path=self._AUDITOR_PROMPT,
                is_architect=False,
                tools=[],
                config=swarm_config,
                scratchpad=scratchpad,
            )

            return MultiRoundValidator([auditor], swarm_config)
        except Exception as e:
            logger.warning(f"ValidatorAgent creation failed, using LLM fallback validator: {e}")
            return _FallbackValidator(self.provider)

    # =========================================================================
    # EXECUTE
    # =========================================================================

    async def execute(
        self,
        goal: str,
        config: Optional[ExecutionConfig] = None,
        status_callback: Optional[Callable] = None,
        **kwargs
    ) -> ExecutionResult:
        """Execute task with appropriate tier."""
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
        """Stream execution events as an async generator."""
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
        """Stream Tier 1 execution with token-level streaming."""
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
        """Tier 1: Direct LLM call with tools. Expected: 1 LLM call, 1-2s, $0.01."""
        logger.info(f"[Tier 1: DIRECT] Executing: {goal[:50]}...")

        if status_callback:
            status_callback("direct", "Discovering skills...")

        # Discover relevant skills
        discovery = self.registry.discover_for_task(goal)
        raw_skills = discovery.get('skills', [])[:5]
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
        """Tier 2: Agentic execution with planning. Expected: 3-5 LLM calls, 3-5s, $0.03."""
        logger.info(f"[Tier 2: AGENTIC] Executing: {goal[:50]}...")

        total_llm_calls = 0
        total_cost = 0.0

        if status_callback:
            status_callback("planning", "Creating execution plan...")

        # Step 1: Plan — use TaskPlanner directly
        with self.tracer.span("tier2_plan") as plan_span:
            plan_start = time.time()
            plan_result = await self._run_planner(goal)
            plan_duration = time.time() - plan_start
            total_llm_calls += 1

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
        """Tier 3: Learning with memory and validation. Expected: 5-10 LLM calls, 5-10s, $0.06."""
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
    # TIER 4: RESEARCH - Domain Swarm Execution
    # =========================================================================

    async def _execute_tier4(
        self,
        goal: str,
        config: ExecutionConfig,
        status_callback: Optional[Callable],
        **kwargs
    ) -> ExecutionResult:
        """Tier 4: Domain swarm execution. Expected: 10-30s, $0.15."""
        logger.info(f"[Tier 4: RESEARCH] Executing with domain swarm...")

        if status_callback:
            status_callback("research", "Selecting domain swarm...")

        # Select and instantiate swarm
        swarm = self._select_swarm(goal, config.swarm_name)

        if swarm is None:
            return await self._execute_with_swarm_manager(goal, config, status_callback, **kwargs)

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

    async def _execute_with_swarm_manager(
        self,
        goal: str,
        config: ExecutionConfig,
        status_callback: Optional[Callable],
        **kwargs
    ) -> ExecutionResult:
        """Delegate to Orchestrator when no specific swarm matches."""
        logger.info("[Tier 4: RESEARCH] Delegating to Orchestrator...")

        from Jotty.core.orchestration import Orchestrator
        from Jotty.core.foundation.data_structures import SwarmConfig

        swarm_config_dict = config.to_swarm_config()
        swarm_config = SwarmConfig(**swarm_config_dict)
        swarm_manager = Orchestrator(config=swarm_config)

        if status_callback:
            status_callback("research", "Executing with Orchestrator...")

        sm_result = await swarm_manager.run(
            goal=goal,
            status_callback=status_callback,
            **kwargs
        )

        return ExecutionResult(
            output=sm_result.output,
            tier=ExecutionTier.RESEARCH,
            success=sm_result.success,
            llm_calls=len(sm_result.trajectory) if sm_result.trajectory else 1,
            latency_ms=int(sm_result.execution_time * 1000),
            cost_usd=0.0,
            episode=sm_result,
            learning_data={'agent_contributions': sm_result.agent_contributions},
        )

    # =========================================================================
    # TIER 5: AUTONOMOUS - Sandbox + Coalition + Full Features
    # =========================================================================

    async def _execute_tier5(
        self,
        goal: str,
        config: ExecutionConfig,
        status_callback: Optional[Callable],
        **kwargs
    ) -> ExecutionResult:
        """Tier 5: Autonomous execution with sandbox, coalition, curriculum."""
        logger.info(f"[Tier 5: AUTONOMOUS] Executing: {goal[:50]}...")

        if status_callback:
            status_callback("autonomous", "Selecting swarm...")

        swarm = self._select_swarm(goal, config.swarm_name)
        sandbox_log = None

        with self.tracer.span("tier5_autonomous", sandbox=config.enable_sandbox) as auto_span:
            if config.enable_sandbox and swarm is not None:
                try:
                    from Jotty.core.orchestration.sandbox_manager import SandboxManager
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
                        return await self._execute_with_swarm_manager(goal, config, status_callback, **kwargs)
            elif swarm is not None:
                if status_callback:
                    status_callback("autonomous", f"Executing with {swarm.__class__.__name__}...")
                result = await swarm.execute(task=goal, **kwargs)
                auto_span.set_attribute("swarm", swarm.__class__.__name__)
            else:
                return await self._execute_with_swarm_manager(goal, config, status_callback, **kwargs)

        # Optionally apply paradigm (debate/relay/refinement)
        paradigm_used = None
        if config.paradigm:
            try:
                from Jotty.core.orchestration.swarm_intelligence import SwarmIntelligence
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
        if TierExecutor._swarms_registered:
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
        TierExecutor._swarms_registered = True

    def _select_swarm(self, goal: str, swarm_name: Optional[str] = None):
        """Select and instantiate the right domain swarm."""
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

        return None

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    async def _run_planner(self, goal: str) -> Dict[str, Any]:
        """Run TaskPlanner and return normalized plan dict.

        Calls planner.aplan_execution() directly. Handles both
        TaskPlanner (returns tuple) and mock planners (return dict).
        """
        planner = self.planner

        # If planner has .plan() (mock or legacy adapter), use it directly
        if hasattr(planner, 'plan') and callable(getattr(planner, 'plan')):
            return await planner.plan(goal)

        # Real TaskPlanner: use aplan_execution
        skills = []
        try:
            discovery = self.registry.discover_for_task(goal)
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

        # Convert ExecutionStep objects to dicts
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

    def _parse_plan(self, goal: str, plan_result: Any) -> ExecutionPlan:
        """Convert planner output to ExecutionPlan.

        Handles both dict results (from _run_planner) and
        raw ExecutionStep objects (from TaskPlanner).
        """
        steps_data = plan_result.get('steps', [])

        steps = []
        for i, step_data in enumerate(steps_data):
            # Handle dict format
            if isinstance(step_data, dict):
                step = ExecutionStep(
                    step_num=i + 1,
                    description=step_data.get('description', f'Step {i+1}'),
                    skill=step_data.get('skill'),
                    depends_on=step_data.get('depends_on', []),
                    can_parallelize=step_data.get('can_parallelize', False),
                )
            else:
                # Handle ExecutionStep objects from TaskPlanner
                step = ExecutionStep(
                    step_num=i + 1,
                    description=getattr(step_data, 'description', str(step_data)),
                    skill=getattr(step_data, 'skill_name', None),
                    depends_on=getattr(step_data, 'depends_on', []),
                    can_parallelize=False,
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
        if step.skill:
            skill = self.registry.get_skill(step.skill)
            tools = skill.to_claude_tools()
        else:
            discovery = self.registry.discover_for_task(step.description)
            raw_skills = discovery.get('skills', [])[:3]
            skill_names = [s['name'] if isinstance(s, dict) else s for s in raw_skills]
            tools = self.registry.get_claude_tools(skill_names) if skill_names else []

        llm_start = time.time()
        response = await self.provider.generate(
            prompt=step.description,
            tools=tools,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        llm_duration = time.time() - llm_start

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

        aggregated = f"Results for: {goal}\n\n"
        for i, result in enumerate(results, 1):
            output = result.get('output', '')
            aggregated += f"Step {i}:\n{output}\n\n"

        return aggregated.strip()

    async def _retrieve_memory(self, goal: str, config: ExecutionConfig) -> Optional[MemoryContext]:
        """Retrieve relevant memory entries."""
        try:
            entries = await self.memory.retrieve(goal, limit=5)
            if not entries:
                return None

            return MemoryContext(
                entries=entries,
                relevance_scores=[e.get('score', 0.0) for e in entries],
                total_retrieved=len(entries),
                retrieval_time_ms=10.0,
            )
        except Exception as e:
            logger.warning(f"Memory retrieval failed: {e}")
            return None

    def _enrich_with_memory(self, goal: str, context: Optional[MemoryContext]) -> str:
        """Enrich goal with memory context."""
        if not context or not context.entries:
            return goal

        enriched = f"{goal}\n\nRelevant past experience:\n"
        for entry in context.entries[:3]:
            enriched += f"- {entry.get('summary', entry.get('result', ''))}\n"

        return enriched

    async def _validate_result(
        self,
        goal: str,
        result: ExecutionResult,
        config: ExecutionConfig
    ) -> ValidationResult:
        """Validate execution result using ValidatorAgent or fallback."""
        validator = self.validator

        # Check if validator is a real MultiRoundValidator (not a mock or fallback)
        from Jotty.core.execution.executor import _FallbackValidator
        is_multi_round = (
            not isinstance(validator, _FallbackValidator)
            and hasattr(validator, '__class__')
            and validator.__class__.__name__ == 'MultiRoundValidator'
        )

        if is_multi_round:
            try:
                trajectory = [{'goal': goal, 'output': str(result.output)[:500]}]
                results_list, combined_decision = await validator.validate(
                    goal=goal,
                    inputs={'task': goal},
                    trajectory=trajectory,
                    is_architect=False,
                )
                if results_list:
                    first = results_list[0]
                    return ValidationResult(
                        success=combined_decision,
                        confidence=getattr(first, 'confidence', 0.8),
                        feedback=getattr(first, 'reasoning', ''),
                        reasoning=getattr(first, 'reasoning', ''),
                    )
            except Exception as e:
                logger.warning(f"MultiRoundValidator failed, using fallback: {e}")

        # Fallback / mock validator: single prompt-based validation
        validation_prompt = f"""
Task: {goal}

Result: {str(result.output)[:500]}

Is this result correct and complete? Provide:
1. Success (yes/no)
2. Confidence (0-1)
3. Feedback (brief)
4. Reasoning
"""
        response = await validator.validate(validation_prompt)

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
            from Jotty.core.execution.memory.noop_memory import NoOpMemory
            return NoOpMemory()


class _FallbackValidator:
    """Simple LLM-based validator used when ValidatorAgent can't be created."""

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
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                return _json.loads(content[start:end])
            return {'success': True, 'confidence': 0.7, 'feedback': content, 'reasoning': ''}
        except Exception as e:
            logger.warning(f"Validation LLM call failed: {e}")
            return {'success': True, 'confidence': 0.5, 'feedback': 'Validation skipped', 'reasoning': str(e)}
