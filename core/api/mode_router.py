"""
Mode Router - Unified Request Routing
=====================================

Single entry point for all execution modes.
Routes requests to Chat, Workflow, or Agent mode based on ExecutionContext.

All entry points (CLI, Gateway, Web, SDK) flow through this router
for consistent behavior and context handling.

Usage:
    router = ModeRouter()

    # Route based on context
    result = await router.route(request, context)

    # Or use mode-specific methods
    result = await router.chat(message, context)
    result = await router.workflow(goal, context)
    async for event in router.stream(message, context):
        print(event)
"""

import logging
from typing import Dict, Any, Optional, List, AsyncIterator, Union
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

# Absolute imports - single source of truth
from Jotty.core.foundation.types.sdk_types import (
    ExecutionMode,
    ChannelType,
    SDKEventType,
    ResponseFormat,
    ExecutionContext,
    SDKEvent,
    SDKResponse,
    SDKRequest,
)


@dataclass
class RouteResult:
    """Result from mode routing."""
    success: bool
    content: Any
    mode: ExecutionMode
    execution_time: float = 0.0
    skills_used: List[str] = None
    agents_used: List[str] = None
    steps_executed: int = 0
    error: Optional[str] = None
    errors: List[str] = None  # Multiple errors from execution
    stopped_early: bool = False  # True if execution stopped due to failure
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.skills_used is None:
            self.skills_used = []
        if self.agents_used is None:
            self.agents_used = []
        if self.errors is None:
            self.errors = []
        if self.metadata is None:
            self.metadata = {}

    def to_sdk_response(self) -> SDKResponse:
        """Convert to SDKResponse."""
        return SDKResponse(
            success=self.success,
            content=self.content,
            mode=self.mode,
            execution_time=self.execution_time,
            skills_used=self.skills_used,
            agents_used=self.agents_used,
            steps_executed=self.steps_executed,
            error=self.error,
            errors=self.errors,
            stopped_early=self.stopped_early,
            metadata=self.metadata,
        )


class ModeRouter:
    """
    Unified router for all execution modes.

    Routes requests to:
    - TierExecutor (native LLM tool calling) for CHAT mode
    - AutoAgent for WORKFLOW mode
    - Direct skill/agent for SKILL/AGENT modes

    Provides consistent:
    - Context handling
    - Event emission
    - Error handling
    - Logging
    """

    def __init__(self) -> None:
        self._auto_agent = None
        self._registry = None
        self._initialized = False
        self._lm_configured = False

    def _ensure_lm_configured(self) -> bool:
        """Ensure DSPy LM is configured."""
        if self._lm_configured:
            return True

        try:
            from Jotty.core.foundation.unified_lm_provider import configure_dspy_lm
            lm = configure_dspy_lm()
            if lm:
                self._lm_configured = True
                return True
        except Exception as e:
            logger.warning(f"Could not configure LM: {e}")
        return False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of mode handlers."""
        if self._initialized:
            return

        # Configure LM first
        self._ensure_lm_configured()

        try:
            from Jotty.core.registry import get_unified_registry
            self._registry = get_unified_registry()
        except Exception as e:
            logger.warning(f"Could not initialize registry: {e}")

        self._initialized = True

    def _get_executor(self, context: Optional[ExecutionContext] = None) -> Any:
        """Get ChatExecutor with callbacks from context."""
        from Jotty.core.orchestration.unified_executor import ChatExecutor
        status_cb = context.status_callback if context else None
        stream_cb = context.stream_callback if context else None
        return ChatExecutor(
            status_callback=status_cb,
            stream_callback=stream_cb,
        )

    def _get_auto_agent(self, context: Optional[ExecutionContext] = None) -> None:
        """Get or create AutoAgent."""
        try:
            from Jotty.core.agents.auto_agent import AutoAgent
            return AutoAgent(
                max_steps=context.max_steps if context else 10,
                timeout=int(context.timeout) if context else 300
            )
        except ImportError:
            logger.warning("AutoAgent not available")
            return None

    async def route(
        self,
        request: Union[str, SDKRequest],
        context: ExecutionContext
    ) -> RouteResult:
        """
        Route request to appropriate mode handler.

        Args:
            request: Request content or SDKRequest
            context: Execution context

        Returns:
            RouteResult with execution output
        """
        self._ensure_initialized()
        start_time = datetime.now()

        # Extract content from request
        if isinstance(request, SDKRequest):
            content = request.content
            mode = request.mode
        else:
            content = request
            mode = context.mode

        # Emit start event
        context.emit_event(SDKEventType.START, {"content": content, "mode": mode.value})

        try:
            # Route to appropriate handler
            if mode == ExecutionMode.CHAT:
                result = await self._handle_chat(content, context)
            elif mode == ExecutionMode.WORKFLOW:
                result = await self._handle_workflow(content, context)
            elif mode == ExecutionMode.SKILL:
                if isinstance(request, SDKRequest) and request.skill_name:
                    result = await self._handle_skill(request.skill_name, content, context)
                else:
                    result = RouteResult(
                        success=False,
                        content=None,
                        mode=mode,
                        error="Skill name required for SKILL mode"
                    )
            elif mode == ExecutionMode.AGENT:
                if isinstance(request, SDKRequest) and request.agent_name:
                    result = await self._handle_agent(request.agent_name, content, context)
                else:
                    result = await self._handle_workflow(content, context)
            else:
                # Default to chat
                result = await self._handle_chat(content, context)

            # Calculate execution time
            result.execution_time = (datetime.now() - start_time).total_seconds()

            # Emit complete event
            context.emit_event(SDKEventType.COMPLETE, {
                "success": result.success,
                "execution_time": result.execution_time
            })

            return result

        except Exception as e:
            logger.error(f"Mode routing error: {e}", exc_info=True)
            context.emit_event(SDKEventType.ERROR, {"error": str(e)})
            return RouteResult(
                success=False,
                content=None,
                mode=mode,
                execution_time=(datetime.now() - start_time).total_seconds(),
                error=str(e)
            )

    async def _handle_chat(
        self,
        message: str,
        context: ExecutionContext
    ) -> RouteResult:
        """
        Handle chat mode via TierExecutor.

        TierExecutor provides native LLM tool-calling:
        - Direct, low-latency single-agent execution
        - LLM decides which tools to call (web search, file ops, etc.)
        - Streaming support via context.stream_callback
        - Status updates via context.status_callback

        For multi-step planning workflows, use WORKFLOW mode instead.
        """
        context.emit_event(SDKEventType.THINKING, {"message": message})

        executor = self._get_executor(context)

        try:
            result = await executor.execute(message)

            return RouteResult(
                success=result.success,
                content=result.content,
                mode=ExecutionMode.CHAT,
                skills_used=getattr(result, 'tools_used', []),
                steps_executed=getattr(result, 'steps_taken', 1),
                metadata={
                    "output_format": getattr(result, 'output_format', 'markdown'),
                    "output_path": getattr(result, 'output_path', None),
                    "was_streamed": getattr(result, 'was_streamed', False),
                },
            )

        except Exception as e:
            logger.error(f"Chat execution error: {e}", exc_info=True)
            return RouteResult(
                success=False,
                content=None,
                mode=ExecutionMode.CHAT,
                error=str(e),
                errors=[str(e)],
            )

    async def _handle_workflow(
        self,
        goal: str,
        context: ExecutionContext
    ) -> RouteResult:
        """Handle workflow mode request using AutoAgent."""
        context.emit_event(SDKEventType.PLANNING, {"goal": goal})

        agent = self._get_auto_agent(context)
        if agent is None:
            return RouteResult(
                success=False,
                content=None,
                mode=ExecutionMode.WORKFLOW,
                error="AutoAgent not available"
            )

        # Status callback to emit events during execution
        def status_callback(stage: str, detail: str = "") -> None:
            context.emit_event(SDKEventType.THINKING, {"status": stage, "message": detail})

        try:
            result = await agent.execute(goal, status_callback=status_callback)

            # Get errors from result (dict or object attribute)
            errors = result.get('errors', []) if isinstance(result, dict) else getattr(result, 'errors', [])
            stopped_early = result.get('stopped_early', False) if isinstance(result, dict) else getattr(result, 'stopped_early', False)
            success = result.get('success', True) if isinstance(result, dict) else getattr(result, 'success', True)

            return RouteResult(
                success=success,
                content=result.get('final_output') if isinstance(result, dict) else getattr(result, 'final_output', result),
                mode=ExecutionMode.WORKFLOW,
                skills_used=result.get('skills_used', []) if isinstance(result, dict) else getattr(result, 'skills_used', []),
                steps_executed=result.get('steps_executed', 0) if isinstance(result, dict) else getattr(result, 'steps_executed', 0),
                error=errors[0] if errors else None,
                errors=errors or [],
                stopped_early=stopped_early,
                metadata={"task_type": result.get('task_type', 'unknown') if isinstance(result, dict) else getattr(result, 'task_type', 'unknown')},
            )

        except Exception as e:
            logger.error(f"Workflow execution error: {e}", exc_info=True)
            return RouteResult(
                success=False,
                content=None,
                mode=ExecutionMode.WORKFLOW,
                error=str(e),
                errors=[str(e)],
                stopped_early=True,
            )

    async def _handle_skill(
        self,
        skill_name: str,
        params: Any,
        context: ExecutionContext
    ) -> RouteResult:
        """Handle direct skill execution."""
        context.emit_event(SDKEventType.SKILL_START, {"skill": skill_name})

        if self._registry is None:
            return RouteResult(
                success=False,
                content=None,
                mode=ExecutionMode.SKILL,
                error="Registry not available"
            )

        try:
            skill = self._registry.get_skill(skill_name)
            if skill is None:
                return RouteResult(
                    success=False,
                    content=None,
                    mode=ExecutionMode.SKILL,
                    error=f"Skill not found: {skill_name}"
                )

            # Parse params if string
            if isinstance(params, str):
                import json
                try:
                    params = json.loads(params)
                except (json.JSONDecodeError, ValueError):
                    params = {"input": params}

            result = await skill.execute(params) if hasattr(skill, 'execute') else skill.run(params)

            context.emit_event(SDKEventType.SKILL_COMPLETE, {"skill": skill_name, "result": result})

            return RouteResult(
                success=True,
                content=result,
                mode=ExecutionMode.SKILL,
                skills_used=[skill_name],
            )

        except Exception as e:
            return RouteResult(
                success=False,
                content=None,
                mode=ExecutionMode.SKILL,
                error=str(e)
            )

    async def _handle_agent(
        self,
        agent_name: str,
        task: str,
        context: ExecutionContext
    ) -> RouteResult:
        """Handle direct agent execution."""
        context.emit_event(SDKEventType.AGENT_START, {"agent": agent_name})

        # For now, route to AutoAgent (could be extended to support named agents)
        agent = self._get_auto_agent(context)
        if agent is None:
            return RouteResult(
                success=False,
                content=None,
                mode=ExecutionMode.AGENT,
                error="Agent not available"
            )

        try:
            result = await agent.execute(task)

            context.emit_event(SDKEventType.AGENT_COMPLETE, {"agent": agent_name})

            return RouteResult(
                success=result.success,
                content=result.final_output,
                mode=ExecutionMode.AGENT,
                skills_used=result.skills_used,
                agents_used=[agent_name],
                steps_executed=result.steps_executed,
            )

        except Exception as e:
            return RouteResult(
                success=False,
                content=None,
                mode=ExecutionMode.AGENT,
                error=str(e)
            )

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    async def chat(self, message: str, context: Optional[ExecutionContext] = None, **kwargs: Any) -> RouteResult:
        """Execute chat mode."""
        if context is None:
            context = ExecutionContext(
                mode=ExecutionMode.CHAT,
                channel=ChannelType.SDK,
                **kwargs
            )
        return await self.route(message, context.with_mode(ExecutionMode.CHAT))

    async def workflow(self, goal: str, context: Optional[ExecutionContext] = None, **kwargs: Any) -> RouteResult:
        """Execute workflow mode."""
        if context is None:
            context = ExecutionContext(
                mode=ExecutionMode.WORKFLOW,
                channel=ChannelType.SDK,
                **kwargs
            )
        return await self.route(goal, context.with_mode(ExecutionMode.WORKFLOW))

    async def skill(self, skill_name: str, params: Dict[str, Any], context: Optional[ExecutionContext] = None, **kwargs: Any) -> RouteResult:
        """Execute a skill directly."""
        if context is None:
            context = ExecutionContext(
                mode=ExecutionMode.SKILL,
                channel=ChannelType.SDK,
                **kwargs
            )
        request = SDKRequest(
            content=str(params),
            mode=ExecutionMode.SKILL,
            skill_name=skill_name,
        )
        return await self.route(request, context.with_mode(ExecutionMode.SKILL))

    async def stream(self, content: str, context: Optional[ExecutionContext] = None, **kwargs: Any) -> AsyncIterator[SDKEvent]:
        """
        Stream execution with events.

        Uses asyncio.Queue for zero-latency event delivery (no polling).
        Events are yielded immediately as they're emitted by the execution.
        """
        import asyncio

        if context is None:
            context = ExecutionContext(
                mode=ExecutionMode.CHAT,
                channel=ChannelType.SDK,
                streaming=True,
                **kwargs
            )

        # Queue-based event delivery (instant, no polling)
        event_queue: asyncio.Queue[Optional[SDKEvent]] = asyncio.Queue()
        original_callback = context.event_callback

        def queue_event(event: SDKEvent) -> None:
            event_queue.put_nowait(event)
            if original_callback:
                original_callback(event)

        context.event_callback = queue_event

        # Start execution in background
        task = asyncio.create_task(self.route(content, context))

        # Yield events as they arrive (zero-latency)
        while True:
            # Wait for next event or task completion
            done, _ = await asyncio.wait(
                [asyncio.ensure_future(event_queue.get()), task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Drain all available events from queue
            while not event_queue.empty():
                event = event_queue.get_nowait()
                if event is not None:
                    yield event

            # If a queue.get() completed, yield that event too
            for completed in done:
                if completed is not task:
                    try:
                        event = completed.result()
                        if event is not None:
                            yield event
                    except Exception:
                        pass

            # If task completed, drain remaining events and exit
            if task.done():
                while not event_queue.empty():
                    event = event_queue.get_nowait()
                    if event is not None:
                        yield event
                break

        # Get result and yield final complete event
        try:
            result = task.result()
            yield SDKEvent(
                type=SDKEventType.COMPLETE,
                data=result.to_sdk_response().to_dict()
            )
        except Exception as e:
            yield SDKEvent(
                type=SDKEventType.ERROR,
                data={"error": str(e)}
            )


# Singleton instance
_mode_router: Optional[ModeRouter] = None


def get_mode_router() -> ModeRouter:
    """Get the singleton mode router."""
    global _mode_router
    if _mode_router is None:
        _mode_router = ModeRouter()
    return _mode_router
