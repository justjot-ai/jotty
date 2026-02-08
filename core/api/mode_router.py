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

# Import SDK types
from ..foundation.types.sdk_types import (
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
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.skills_used is None:
            self.skills_used = []
        if self.agents_used is None:
            self.agents_used = []
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
            metadata=self.metadata,
        )


class ModeRouter:
    """
    Unified router for all execution modes.

    Routes requests to:
    - LeanExecutor (DSPy-based) or UnifiedExecutor for CHAT mode
    - AutoAgent for WORKFLOW mode
    - Direct skill/agent for SKILL/AGENT modes

    Provides consistent:
    - Context handling
    - Event emission
    - Error handling
    - Logging
    """

    def __init__(self):
        self._auto_agent = None
        self._registry = None
        self._initialized = False
        self._lm_configured = False

    def _ensure_lm_configured(self):
        """Ensure DSPy LM is configured."""
        if self._lm_configured:
            return True

        try:
            from ..foundation.unified_lm_provider import configure_dspy_lm
            lm = configure_dspy_lm()
            if lm:
                self._lm_configured = True
                return True
        except Exception as e:
            logger.warning(f"Could not configure LM: {e}")
        return False

    def _ensure_initialized(self):
        """Lazy initialization of mode handlers."""
        if self._initialized:
            return

        # Configure LM first
        self._ensure_lm_configured()

        try:
            from ..registry import get_unified_registry
            self._registry = get_unified_registry()
        except Exception as e:
            logger.warning(f"Could not initialize registry: {e}")

        self._initialized = True

    def _get_auto_agent(self, context: Optional[ExecutionContext] = None):
        """Get or create AutoAgent."""
        try:
            from ..agents.auto_agent import AutoAgent
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
        Handle chat mode - all requests go through AutoAgent.

        AutoAgent automatically handles:
        - Simple conversational queries (falls back to direct LLM)
        - Complex multi-step tasks (uses skill discovery and execution)

        No hardcoded routing - the swarm system decides what to do.
        """
        # All chat requests go through AutoAgent for unified handling
        return await self._handle_workflow(message, context)

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
        def status_callback(stage: str, detail: str = ""):
            context.emit_event(SDKEventType.THINKING, {"status": stage, "message": detail})

        try:
            result = await agent.execute(goal, status_callback=status_callback)

            return RouteResult(
                success=result.success,
                content=result.final_output,
                mode=ExecutionMode.WORKFLOW,
                skills_used=result.skills_used,
                steps_executed=result.steps_executed,
                metadata={"task_type": result.task_type.value if hasattr(result.task_type, 'value') else str(result.task_type)},
            )

        except Exception as e:
            logger.error(f"Workflow execution error: {e}", exc_info=True)
            return RouteResult(
                success=False,
                content=None,
                mode=ExecutionMode.WORKFLOW,
                error=str(e)
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
                except:
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

    async def chat(
        self,
        message: str,
        context: Optional[ExecutionContext] = None,
        **kwargs
    ) -> RouteResult:
        """Execute chat mode."""
        if context is None:
            context = ExecutionContext(
                mode=ExecutionMode.CHAT,
                channel=ChannelType.SDK,
                **kwargs
            )
        return await self.route(message, context.with_mode(ExecutionMode.CHAT))

    async def workflow(
        self,
        goal: str,
        context: Optional[ExecutionContext] = None,
        **kwargs
    ) -> RouteResult:
        """Execute workflow mode."""
        if context is None:
            context = ExecutionContext(
                mode=ExecutionMode.WORKFLOW,
                channel=ChannelType.SDK,
                **kwargs
            )
        return await self.route(goal, context.with_mode(ExecutionMode.WORKFLOW))

    async def skill(
        self,
        skill_name: str,
        params: Dict[str, Any],
        context: Optional[ExecutionContext] = None,
        **kwargs
    ) -> RouteResult:
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

    async def stream(
        self,
        content: str,
        context: Optional[ExecutionContext] = None,
        **kwargs
    ) -> AsyncIterator[SDKEvent]:
        """
        Stream execution with events.

        Yields SDKEvent objects for each update during execution.
        """
        if context is None:
            context = ExecutionContext(
                mode=ExecutionMode.CHAT,
                channel=ChannelType.SDK,
                streaming=True,
                **kwargs
            )

        # Collect events
        events = []
        original_callback = context.event_callback

        def collect_event(event: SDKEvent):
            events.append(event)
            if original_callback:
                original_callback(event)

        context.event_callback = collect_event

        # Start execution (non-blocking)
        import asyncio
        task = asyncio.create_task(self.route(content, context))

        # Yield events as they come
        last_yielded = 0
        while not task.done():
            await asyncio.sleep(0.05)  # Poll every 50ms
            while last_yielded < len(events):
                yield events[last_yielded]
                last_yielded += 1

        # Wait for completion
        result = await task

        # Yield remaining events
        while last_yielded < len(events):
            yield events[last_yielded]
            last_yielded += 1

        # Yield final result
        yield SDKEvent(
            type=SDKEventType.COMPLETE,
            data=result.to_sdk_response().to_dict()
        )


# Singleton instance
_mode_router: Optional[ModeRouter] = None


def get_mode_router() -> ModeRouter:
    """Get the singleton mode router."""
    global _mode_router
    if _mode_router is None:
        _mode_router = ModeRouter()
    return _mode_router
