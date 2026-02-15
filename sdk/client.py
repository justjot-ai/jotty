"""
Jotty SDK Client - World-Class Python SDK
==========================================

The unified Python client for Jotty AI Framework.
This is the reference implementation that other language SDKs follow.

Features:
- Clean, intuitive API (chat, workflow, stream, agent, skill, session)
- Event-driven with callbacks for real-time updates
- Persistent sessions across channels
- Full type hints and IDE support
- Async-first with sync wrappers

Usage:
    from jotty import Jotty

    # Initialize
    client = Jotty(base_url="http://localhost:8766")

    # Chat mode
    response = await client.chat("What's the weather?")
    print(response.content)

    # Workflow mode
    result = await client.workflow("Research AI trends and create slides")
    print(result.content)

    # Streaming
    async for event in client.stream("Analyze this data"):
        if event.type == SDKEventType.STREAM:
            print(event.data, end="")

    # Events
    client.on("skill_start", lambda e: print(f"Using {e.data['skill']}"))
    client.on("thinking", lambda e: print("Agent thinking..."))

    # Direct skill
    result = await client.skill("web-search").run(query="AI news")

    # Direct agent
    result = await client.agent("research").execute("Find papers on transformers")

    # Session management
    session = await client.session("user-123")
    session.set("preference", "dark mode")
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Callable, Dict, Generic, List, Optional, TypeVar, Union

# Import SDK types
from ..core.infrastructure.foundation.types.sdk_types import (
    ChannelType,
    ExecutionContext,
    ExecutionMode,
    ResponseFormat,
    SDKEvent,
    SDKEventType,
    SDKRequest,
    SDKResponse,
    SDKSession,
    SDKVoiceResponse,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# EVENT EMITTER
# =============================================================================


class EventEmitter:
    """
    Event emitter for SDK callbacks.

    Supports:
    - Multiple listeners per event
    - Once listeners (auto-remove after first call)
    - Wildcard listeners (listen to all events)
    - Async and sync callbacks
    """

    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = {}
        self._once_listeners: Dict[str, List[Callable]] = {}

    def on(self, event: Union[str, SDKEventType], callback: Callable) -> "EventEmitter":
        """Register an event listener."""
        event_name = event.value if isinstance(event, SDKEventType) else event
        if event_name not in self._listeners:
            self._listeners[event_name] = []
        self._listeners[event_name].append(callback)
        return self

    def once(self, event: Union[str, SDKEventType], callback: Callable) -> "EventEmitter":
        """Register a one-time event listener."""
        event_name = event.value if isinstance(event, SDKEventType) else event
        if event_name not in self._once_listeners:
            self._once_listeners[event_name] = []
        self._once_listeners[event_name].append(callback)
        return self

    def off(
        self, event: Union[str, SDKEventType], callback: Optional[Callable] = None
    ) -> "EventEmitter":
        """Remove event listener(s)."""
        event_name = event.value if isinstance(event, SDKEventType) else event
        if callback is None:
            self._listeners.pop(event_name, None)
            self._once_listeners.pop(event_name, None)
        else:
            if event_name in self._listeners:
                self._listeners[event_name] = [
                    cb for cb in self._listeners[event_name] if cb != callback
                ]
            if event_name in self._once_listeners:
                self._once_listeners[event_name] = [
                    cb for cb in self._once_listeners[event_name] if cb != callback
                ]
        return self

    async def emit(self, event: Union[str, SDKEvent]) -> None:
        """Emit an event to all listeners."""
        if isinstance(event, SDKEvent):
            event_name = event.type.value
            event_data = event
        else:
            event_name = event
            event_data = event

        # Call regular listeners
        for callback in self._listeners.get(event_name, []):
            try:
                result = callback(event_data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Event listener error for {event_name}: {e}")

        # Call and remove once listeners
        once_callbacks = self._once_listeners.pop(event_name, [])
        for callback in once_callbacks:
            try:
                result = callback(event_data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Once listener error for {event_name}: {e}")

        # Call wildcard listeners
        for callback in self._listeners.get("*", []):
            try:
                result = callback(event_data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Wildcard listener error: {e}")


# =============================================================================
# SKILL HANDLE
# =============================================================================


class SkillHandle:
    """
    Handle for direct skill execution.

    Usage:
        skill = client.skill("web-search")
        result = await skill.run(query="AI news")
    """

    def __init__(self, client: "Jotty", skill_name: str):
        self._client = client
        self._skill_name = skill_name
        self._params: Dict[str, Any] = {}

    def with_params(self, **params) -> "SkillHandle":
        """Set default parameters."""
        self._params.update(params)
        return self

    async def run(self, **params) -> SDKResponse:
        """Execute the skill with parameters."""
        all_params = {**self._params, **params}
        return await self._client._execute_skill(self._skill_name, all_params)

    async def info(self) -> Dict[str, Any]:
        """Get skill information."""
        return await self._client._get_skill_info(self._skill_name)


# =============================================================================
# AGENT HANDLE
# =============================================================================


class AgentHandle:
    """
    Handle for direct agent execution.

    Usage:
        agent = client.agent("research")
        result = await agent.execute("Find papers on transformers")
    """

    def __init__(self, client: "Jotty", agent_name: str):
        self._client = client
        self._agent_name = agent_name
        self._context: Dict[str, Any] = {}

    def with_context(self, **context) -> "AgentHandle":
        """Set execution context."""
        self._context.update(context)
        return self

    async def execute(self, task: str, **kwargs) -> SDKResponse:
        """Execute task with agent."""
        context = {**self._context, **kwargs}
        return await self._client._execute_agent(self._agent_name, task, context)

    async def info(self) -> Dict[str, Any]:
        """Get agent information."""
        return await self._client._get_agent_info(self._agent_name)


# =============================================================================
# SESSION HANDLE
# =============================================================================


class SessionHandle:
    """
    Handle for session management.

    Usage:
        session = await client.session("user-123")
        session.set("preference", "dark mode")
        history = session.get_history(10)
    """

    def __init__(self, client: "Jotty", session: SDKSession):
        self._client = client
        self._session = session

    @property
    def id(self) -> str:
        return self._session.session_id

    @property
    def user_id(self) -> str:
        return self._session.user_id

    def set(self, key: str, value: Any) -> "SessionHandle":
        """Set a preference."""
        self._session.preferences[key] = value
        return self

    def get(self, key: str, default: Any = None) -> Any:
        """Get a preference."""
        return self._session.preferences.get(key, default)

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent message history."""
        return self._session.get_history(limit)

    def link_channel(self, channel: ChannelType, channel_id: str) -> "SessionHandle":
        """Link a channel to this session."""
        self._session.link_channel(channel, channel_id)
        return self

    async def save(self) -> None:
        """Save session to persistent storage."""
        await self._client._save_session(self._session)

    async def clear(self) -> None:
        """Clear session history."""
        self._session.messages = []
        await self.save()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self._session.to_dict()


# =============================================================================
# VOICE HANDLE
# =============================================================================


class VoiceHandle:
    """
    Fluent handle for voice interactions.

    Usage:
        voice = client.voice("en-US-Jenny")
        response = await voice.chat(audio_data)
        audio = await voice.speak("Hello!")
    """

    def __init__(self, client: "Jotty", voice: Optional[str] = None):
        self._client = client
        self._voice = voice
        self._stt_provider = "auto"
        self._tts_provider = "auto"

    def with_voice(self, voice: str) -> "VoiceHandle":
        """Set voice for TTS."""
        self._voice = voice
        return self

    def with_stt_provider(self, provider: str) -> "VoiceHandle":
        """Set STT provider (auto, groq, whisper, local)."""
        self._stt_provider = provider
        return self

    def with_tts_provider(self, provider: str) -> "VoiceHandle":
        """Set TTS provider (auto, edge, openai, elevenlabs, local)."""
        self._tts_provider = provider
        return self

    async def transcribe(self, audio_data: bytes, mime_type: str = "audio/webm") -> str:
        """Transcribe audio to text."""
        response = await self._client.stt(
            audio_data=audio_data, mime_type=mime_type, provider=self._stt_provider
        )
        if response.success:
            return response.user_text or ""
        raise RuntimeError(response.error or "STT failed")

    async def speak(self, text: str) -> bytes:
        """Convert text to speech."""
        return await self._client.tts(text=text, voice=self._voice, provider=self._tts_provider)

    async def chat(self, audio_data: bytes, mime_type: str = "audio/webm") -> SDKVoiceResponse:
        """Voice chat: transcribe + process + respond."""
        return await self._client.voice_chat(
            audio_data=audio_data,
            mime_type=mime_type,
            voice=self._voice,
            stt_provider=self._stt_provider,
            tts_provider=self._tts_provider,
        )

    async def stream(
        self, audio_data: bytes, mime_type: str = "audio/webm"
    ) -> AsyncIterator[SDKEvent]:
        """Voice chat with streaming."""
        async for event in self._client.voice_stream(
            audio_data=audio_data,
            mime_type=mime_type,
            voice=self._voice,
            stt_provider=self._stt_provider,
            tts_provider=self._tts_provider,
        ):
            yield event


# =============================================================================
# JOTTY CLIENT
# =============================================================================


class Jotty(EventEmitter):
    """
    The unified Jotty SDK client.

    This is the main entry point for all SDK operations.
    Provides a clean, intuitive API for:
    - Chat interactions
    - Workflow execution
    - Streaming responses
    - Direct skill/agent invocation
    - Session management
    - Event-driven callbacks

    Usage:
        client = Jotty(base_url="http://localhost:8766")

        # Simple chat
        response = await client.chat("Hello!")

        # Workflow
        result = await client.workflow("Create a report on AI")

        # Streaming
        async for event in client.stream("Tell me a story"):
            print(event.data, end="")

        # Events
        client.on("thinking", lambda e: print("Thinking..."))
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8766",
        api_key: Optional[str] = None,
        timeout: float = 300.0,
        max_retries: int = 3,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        """
        Initialize Jotty client.

        Args:
            base_url: API base URL
            api_key: Optional API key for authentication
            timeout: Default request timeout in seconds
            max_retries: Maximum retry attempts
            session_id: Optional session ID for continuity
            user_id: Optional user ID for tracking
        """
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._session_id = session_id or str(uuid.uuid4())
        self._user_id = user_id

        # Local mode flag (use internal APIs instead of HTTP)
        self._local_mode = False
        self._local_api = None

        # Session cache
        self._sessions: Dict[str, SDKSession] = {}

        # HTTP client (lazy init)
        self._http_client = None

    async def _get_http_client(self):
        """Get or create HTTP client."""
        if self._http_client is None:
            try:
                import httpx

                self._http_client = httpx.AsyncClient(
                    base_url=self.base_url,
                    timeout=self.timeout,
                    headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                )
            except ImportError:
                logger.warning("httpx not installed. Install with: pip install httpx")
                raise
        return self._http_client

    def use_local(self, api=None) -> "Jotty":
        """
        Use local API instead of HTTP.

        This allows the SDK to work directly with the Jotty core
        without going through HTTP, useful for embedded usage.

        Args:
            api: Optional JottyAPI instance. If None, creates one.
        """
        self._local_mode = True
        if api:
            self._local_api = api
        return self

    def _create_context(
        self, mode: ExecutionMode, streaming: bool = False, **kwargs
    ) -> ExecutionContext:
        """Create execution context for a request."""

        def safe_event_callback(event):
            """Safe event callback that emits synchronously to avoid delayed output."""
            # Always emit synchronously to ensure events appear in order
            # before the command completes (not scheduled for later)
            event_name = event.type.value if hasattr(event.type, "value") else str(event.type)

            # Call regular listeners synchronously
            for callback in self._listeners.get(event_name, []):
                try:
                    result = callback(event)
                    # If callback returns a coroutine, we can't await it here
                    # so just skip async callbacks in sync context
                    if asyncio.iscoroutine(result):
                        result.close()  # Clean up unawaited coroutine
                except Exception:
                    pass

            # Call wildcard listeners
            for callback in self._listeners.get("*", []):
                try:
                    result = callback(event)
                    if asyncio.iscoroutine(result):
                        result.close()
                except Exception:
                    pass

        return ExecutionContext(
            mode=mode,
            channel=ChannelType.SDK,
            session_id=kwargs.get("session_id", self._session_id),
            user_id=kwargs.get("user_id", self._user_id),
            streaming=streaming,
            timeout=kwargs.get("timeout", self.timeout),
            max_steps=kwargs.get("max_steps", 10),
            response_format=kwargs.get("response_format", ResponseFormat.MARKDOWN),
            metadata=kwargs.get("metadata", {}),
            event_callback=safe_event_callback,
        )

    async def _emit_event(self, event_type: SDKEventType, data: Any = None):
        """Emit an SDK event."""
        event = SDKEvent(
            type=event_type, data=data, context_id=self._session_id, timestamp=datetime.now()
        )
        await self.emit(event)

    # =========================================================================
    # CHAT
    # =========================================================================

    async def chat(
        self, message: str, history: Optional[List[Dict[str, Any]]] = None, **kwargs
    ) -> SDKResponse:
        """
        Send a chat message and get a response.

        Args:
            message: The message to send
            history: Optional conversation history
            **kwargs: Additional parameters

        Returns:
            SDKResponse with the assistant's response
        """
        # Events are emitted by ModeRouter via context callback - no direct emission here
        context = self._create_context(ExecutionMode.CHAT, **kwargs)
        start_time = datetime.now()

        try:
            if self._local_mode:
                result = await self._local_chat(message, history, context)
            else:
                result = await self._remote_chat(message, history, context)

            execution_time = (datetime.now() - start_time).total_seconds()

            # Check result success - don't hardcode True
            is_success = result.get("success", True)
            errors = result.get("errors", [])
            stopped_early = result.get("stopped_early", False)

            # If there are errors, it's not a success
            if errors:
                is_success = False

            return SDKResponse(
                success=is_success,
                content=result.get("content", result),
                mode=ExecutionMode.CHAT,
                request_id=context.request_id,
                execution_time=execution_time,
                metadata=result.get("metadata", {}),
                errors=errors,
                stopped_early=stopped_early,
                error=errors[0] if errors else None,
            )

        except Exception as e:
            return SDKResponse.error_response(str(e))

    async def _local_chat(
        self, message: str, history: Optional[List], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Execute chat locally using ModeRouter."""
        try:
            from ..core.interface.api.mode_router import get_mode_router

            router = get_mode_router()

            # Pass history via context metadata, NOT embedded in message
            # Skills receive only the clean message, not history
            if history:
                context.metadata["conversation_history"] = history[-6:]

            result = await router.chat(message, context)
            return {
                "content": result.content,
                "success": result.success,
                "metadata": result.metadata or {},
                "errors": getattr(result, "errors", []) or [],
                "stopped_early": getattr(result, "stopped_early", False),
            }
        except Exception as e:
            # Fallback to ChatAssistant directly
            try:
                from ..core.interface.api.agents import ChatAssistant

                assistant = ChatAssistant()
                result = await assistant.run(goal=message)
                return {
                    "content": result.get("response", result),
                    "success": result.get("success", True),
                    "metadata": result.get("metadata", {}),
                    "errors": result.get("errors", []),
                    "stopped_early": result.get("stopped_early", False),
                }
            except Exception as e2:
                return {
                    "content": None,
                    "success": False,
                    "error": str(e2),
                    "errors": [str(e2)],
                    "stopped_early": True,
                }

    async def _remote_chat(
        self, message: str, history: Optional[List], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Execute chat via HTTP."""
        client = await self._get_http_client()
        response = await client.post(
            "/api/chat",
            json={
                "message": message,
                "history": history,
                "session_id": context.session_id,
                "context": context.to_dict(),
            },
        )
        response.raise_for_status()
        return response.json()

    # =========================================================================
    # WORKFLOW
    # =========================================================================

    async def workflow(
        self, goal: str, context: Optional[Dict[str, Any]] = None, **kwargs
    ) -> SDKResponse:
        """
        Execute a multi-step workflow.

        Args:
            goal: The workflow goal/task
            context: Optional context data
            **kwargs: Additional parameters

        Returns:
            SDKResponse with workflow results
        """
        # Events are emitted by ModeRouter via context callback - no direct emission here
        exec_context = self._create_context(ExecutionMode.WORKFLOW, **kwargs)
        start_time = datetime.now()

        try:
            if self._local_mode:
                result = await self._local_workflow(goal, context, exec_context)
            else:
                result = await self._remote_workflow(goal, context, exec_context)

            execution_time = (datetime.now() - start_time).total_seconds()

            # Check result success and propagate errors
            is_success = result.get("success", True)
            errors = result.get("errors", [])
            stopped_early = result.get("stopped_early", False)

            # If there are errors, it's not a success
            if errors:
                is_success = False

            return SDKResponse(
                success=is_success,
                content=result.get("final_output", result.get("content", result)),
                mode=ExecutionMode.WORKFLOW,
                request_id=exec_context.request_id,
                execution_time=execution_time,
                skills_used=result.get("skills_used", []),
                steps_executed=result.get("steps_executed", 0),
                metadata=result.get("metadata", {}),
                errors=errors,
                stopped_early=stopped_early,
                error=errors[0] if errors else None,
            )

        except Exception as e:
            return SDKResponse.error_response(str(e))

    async def _local_workflow(
        self, goal: str, context: Optional[Dict], exec_context: ExecutionContext
    ) -> Dict[str, Any]:
        """Execute workflow locally using ModeRouter."""
        try:
            from ..core.interface.api.mode_router import get_mode_router

            router = get_mode_router()
            result = await router.workflow(goal, exec_context)
            logger.debug(
                f"ModeRouter workflow result: success={result.success}, skills={result.skills_used}"
            )
            return {
                "content": result.content,
                "final_output": result.content,
                "success": result.success,
                "skills_used": result.skills_used,
                "steps_executed": result.steps_executed,
                "metadata": result.metadata or {},
                "errors": getattr(result, "errors", []) or [],
                "stopped_early": getattr(result, "stopped_early", False),
            }
        except Exception as e:
            logger.warning(f"ModeRouter failed, falling back to AutoAgent: {e}")
            # Fallback to AutoAgent directly
            try:
                from ..core.interface.api.agents import AutoAgent

                agent = AutoAgent()
                result = await agent.execute(goal)
                return {
                    "content": result.final_output,
                    "final_output": result.final_output,
                    "success": result.success,
                    "skills_used": result.skills_used,
                    "steps_executed": result.steps_executed,
                    "metadata": {},
                    "errors": getattr(result, "errors", []) or [],
                    "stopped_early": getattr(result, "stopped_early", False),
                }
            except Exception as e2:
                logger.error(f"AutoAgent fallback also failed: {e2}")
                return {"content": None, "success": False, "error": str(e2)}

    async def _remote_workflow(
        self, goal: str, context: Optional[Dict], exec_context: ExecutionContext
    ) -> Dict[str, Any]:
        """Execute workflow via HTTP."""
        client = await self._get_http_client()
        response = await client.post(
            "/api/workflow",
            json={
                "goal": goal,
                "context": context,
                "session_id": exec_context.session_id,
                "execution_context": exec_context.to_dict(),
            },
        )
        response.raise_for_status()
        return response.json()

    # =========================================================================
    # STREAMING
    # =========================================================================

    async def stream(
        self,
        message: str,
        history: Optional[List[Dict[str, Any]]] = None,
        mode: ExecutionMode = ExecutionMode.CHAT,
        **kwargs,
    ) -> AsyncIterator[SDKEvent]:
        """
        Stream responses with real-time events.

        Args:
            message: The message/goal
            history: Optional conversation history
            mode: Execution mode (chat or workflow)
            **kwargs: Additional parameters

        Yields:
            SDKEvent objects for each update
        """
        context = self._create_context(mode, streaming=True, **kwargs)

        # Build message with context if history provided
        if history:
            context_parts = []
            for msg in history[-6:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                context_parts.append(f"{role.upper()}: {content}")
            conversation_context = "\n".join(context_parts)
            full_message = (
                f"Previous conversation:\n{conversation_context}\n\nCurrent request: {message}"
            )
        else:
            full_message = message

        try:
            if self._local_mode:
                # Local mode: ModeRouter emits START/COMPLETE events
                async for event in self._local_stream(full_message, mode, context):
                    yield event
            else:
                # Remote mode: we emit START ourselves
                yield SDKEvent(type=SDKEventType.START, data={"message": message})
                async for event in self._remote_stream(full_message, mode, context):
                    yield event

        except Exception as e:
            yield SDKEvent(type=SDKEventType.ERROR, data={"error": str(e)})

    async def _local_stream(
        self, message: str, mode: ExecutionMode, context: ExecutionContext
    ) -> AsyncIterator[SDKEvent]:
        """Stream locally using ModeRouter."""
        try:
            from ..core.interface.api.mode_router import get_mode_router

            router = get_mode_router()

            # Use ModeRouter's stream method which yields SDKEvents
            stream_context = ExecutionContext(
                mode=mode,
                channel=ChannelType.SDK,
                session_id=context.session_id,
                user_id=context.user_id,
                streaming=True,
                timeout=context.timeout,
                max_steps=context.max_steps,
            )

            async for event in router.stream(message, stream_context):
                yield event

        except Exception as e:
            logger.error(f"Local stream error: {e}", exc_info=True)
            yield SDKEvent(type=SDKEventType.ERROR, data={"error": str(e)})

    async def _remote_stream(
        self, message: str, mode: ExecutionMode, context: ExecutionContext
    ) -> AsyncIterator[SDKEvent]:
        """Stream via WebSocket."""
        try:
            import websockets
        except ImportError:
            raise ImportError(
                "websockets required for streaming. Install with: pip install websockets"
            )

        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/ws"

        async with websockets.connect(ws_url) as ws:
            # Send request
            await ws.send(
                json.dumps(
                    {
                        "content": message,
                        "mode": mode.value,
                        "stream": True,
                        "context": context.to_dict(),
                    }
                )
            )

            # Receive events
            async for msg in ws:
                data = json.loads(msg)
                event_type = SDKEventType(data.get("type", "stream"))
                yield SDKEvent(
                    type=event_type, data=data.get("data"), context_id=context.request_id
                )

                if event_type == SDKEventType.COMPLETE:
                    break

    # =========================================================================
    # VOICE (MODALITIES)
    # =========================================================================

    def voice(self, voice: Optional[str] = None) -> VoiceHandle:
        """
        Get a voice handle for fluent voice interactions.

        Args:
            voice: Optional voice ID (e.g., "en-US-Jenny")

        Returns:
            VoiceHandle for voice operations

        Usage:
            voice = client.voice("en-US-Ava")
            text = await voice.transcribe(audio_data)
            audio = await voice.speak("Hello!")
        """
        return VoiceHandle(self, voice)

    async def stt(
        self,
        audio_data: bytes,
        mime_type: str = "audio/webm",
        provider: str = "auto",
        language: Optional[str] = None,
    ) -> SDKVoiceResponse:
        """
        Speech-to-text: transcribe audio to text.

        Args:
            audio_data: Audio bytes
            mime_type: MIME type (audio/webm, audio/wav, audio/mp3, audio/ogg)
            provider: STT provider (auto, groq, whisper, local)
            language: Optional language code (e.g., 'en', 'es')

        Returns:
            SDKVoiceResponse with transcribed text

        Usage:
            response = await client.stt(audio_bytes, "audio/webm")
            print(response.user_text)
        """
        await self._emit_event(
            SDKEventType.VOICE_STT_START, {"mime_type": mime_type, "provider": provider}
        )

        start_time = datetime.now()

        try:
            if self._local_mode:
                result = await self._local_stt(audio_data, mime_type, provider, language)
            else:
                result = await self._remote_stt(audio_data, mime_type, provider, language)

            execution_time = (datetime.now() - start_time).total_seconds()

            await self._emit_event(
                SDKEventType.VOICE_STT_COMPLETE,
                {
                    "text": result.get("text"),
                    "confidence": result.get("confidence", 1.0),
                    "provider": result.get("provider"),
                },
            )

            return SDKVoiceResponse(
                success=result.get("success", True),
                content=result.get("text", ""),
                user_text=result.get("text", ""),
                confidence=result.get("confidence", 1.0),
                provider=result.get("provider"),
                mode=ExecutionMode.VOICE,
                execution_time=execution_time,
                metadata=result.get("metadata", {}),
            )

        except Exception as e:
            await self._emit_event(SDKEventType.ERROR, {"error": str(e)})
            return SDKVoiceResponse(success=False, content=None, error=str(e))

    async def _local_stt(
        self, audio_data: bytes, mime_type: str, provider: str, language: Optional[str]
    ) -> Dict[str, Any]:
        """Execute STT locally using modalities layer."""
        import tempfile
        from pathlib import Path

        from ..core.interface.modalities.voice import speech_to_text

        # Save audio to temp file with proper extension
        suffix = {
            "audio/webm": ".webm",
            "audio/ogg": ".ogg",
            "audio/wav": ".wav",
            "audio/mp3": ".mp3",
        }.get(mime_type, ".webm")

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_data)
            temp_path = f.name

        try:
            # Uses modalities voice layer (auto-selects Groq by default)
            text = await speech_to_text(
                audio_file=temp_path, platform="generic", provider=provider, language=language
            )
            return {"success": True, "text": text, "confidence": 1.0, "provider": provider}
        finally:
            Path(temp_path).unlink(missing_ok=True)

    async def _remote_stt(
        self, audio_data: bytes, mime_type: str, provider: str, language: Optional[str]
    ) -> Dict[str, Any]:
        """Execute STT via HTTP."""
        client = await self._get_http_client()
        response = await client.post(
            "/api/voice/stt",
            files={"audio": ("audio", audio_data, mime_type)},
            data={"provider": provider, "language": language or ""},
        )
        response.raise_for_status()
        return response.json()

    async def tts(
        self, text: str, voice: Optional[str] = None, provider: str = "auto", speed: float = 1.0
    ) -> bytes:
        """
        Text-to-speech: convert text to audio.

        Args:
            text: Text to synthesize
            voice: Voice ID (provider-specific)
            provider: TTS provider (auto, edge, openai, elevenlabs, local)
            speed: Speech speed multiplier (0.5 to 2.0)

        Returns:
            Audio bytes (MP3 format)

        Usage:
            audio = await client.tts("Hello world!", voice="en-US-Ava")
            with open("output.mp3", "wb") as f:
                f.write(audio)
        """
        await self._emit_event(
            SDKEventType.VOICE_TTS_START, {"text": text[:50], "voice": voice, "provider": provider}
        )

        start_time = datetime.now()

        try:
            if self._local_mode:
                audio_bytes = await self._local_tts(text, voice, provider, speed)
            else:
                audio_bytes = await self._remote_tts(text, voice, provider, speed)

            execution_time = (datetime.now() - start_time).total_seconds()

            await self._emit_event(
                SDKEventType.VOICE_TTS_COMPLETE, {"size": len(audio_bytes), "provider": provider}
            )

            return audio_bytes

        except Exception as e:
            await self._emit_event(SDKEventType.ERROR, {"error": str(e)})
            raise

    async def _local_tts(
        self, text: str, voice: Optional[str], provider: str, speed: float
    ) -> bytes:
        """Execute TTS locally using modalities layer."""
        from ..core.interface.modalities.voice import text_to_speech

        # Uses modalities voice layer (auto-selects Edge TTS by default)
        audio_bytes = await text_to_speech(
            text=text, platform="generic", provider=provider, voice=voice
        )
        return audio_bytes

    async def _remote_tts(
        self, text: str, voice: Optional[str], provider: str, speed: float
    ) -> bytes:
        """Execute TTS via HTTP."""
        client = await self._get_http_client()
        response = await client.post(
            "/api/voice/tts",
            json={"text": text, "voice": voice, "provider": provider, "speed": speed},
        )
        response.raise_for_status()
        # Assume response contains audio bytes
        return response.content

    async def voice_chat(
        self,
        audio_data: bytes,
        mime_type: str = "audio/webm",
        voice: Optional[str] = None,
        stt_provider: str = "auto",
        tts_provider: str = "auto",
        mode: str = "standard",
    ) -> SDKVoiceResponse:
        """
        Voice chat: transcribe → process → respond with audio.

        Args:
            audio_data: Input audio bytes
            mime_type: Audio MIME type
            voice: Voice for TTS response
            stt_provider: STT provider
            tts_provider: TTS provider
            mode: Processing mode (standard, quick, deep)

        Returns:
            SDKVoiceResponse with text and audio

        Usage:
            response = await client.voice_chat(audio_bytes)
            print(f"User said: {response.user_text}")
            print(f"Assistant: {response.content}")
            # Play response.audio_data
        """
        # Step 1: Transcribe
        stt_result = await self.stt(audio_data, mime_type, stt_provider)
        if not stt_result.success:
            return stt_result

        user_text = stt_result.user_text or ""

        # Step 2: Process (chat)
        chat_result = await self.chat(user_text, mode=mode)
        response_text = chat_result.content

        # Step 3: Synthesize
        try:
            audio_bytes = await self.tts(response_text, voice, tts_provider)
        except Exception as e:
            # Return text response even if TTS fails
            logger.warning(f"TTS failed: {e}")
            audio_bytes = None

        return SDKVoiceResponse(
            success=True,
            content=response_text,
            user_text=user_text,
            audio_data=audio_bytes,
            audio_format="audio/mp3",
            confidence=stt_result.confidence,
            provider=f"stt:{stt_provider},tts:{tts_provider}",
            mode=ExecutionMode.VOICE,
            execution_time=stt_result.execution_time + chat_result.execution_time,
            metadata={"stt_provider": stt_provider, "tts_provider": tts_provider, "voice": voice},
        )

    async def voice_stream(
        self,
        audio_data: bytes,
        mime_type: str = "audio/webm",
        voice: Optional[str] = None,
        stt_provider: str = "auto",
        tts_provider: str = "auto",
    ) -> AsyncIterator[SDKEvent]:
        """
        Voice chat with streaming text responses.

        Args:
            audio_data: Input audio bytes
            mime_type: Audio MIME type
            voice: Voice for TTS
            stt_provider: STT provider
            tts_provider: TTS provider

        Yields:
            SDKEvent objects for STT, streaming text, and TTS

        Usage:
            async for event in client.voice_stream(audio_bytes):
                if event.type == SDKEventType.VOICE_STT_COMPLETE:
                    print(f"User: {event.data['text']}")
                elif event.type == SDKEventType.STREAM:
                    print(event.data['delta'], end="")
        """
        # Step 1: Transcribe
        yield SDKEvent(type=SDKEventType.VOICE_STT_START, data={})
        stt_result = await self.stt(audio_data, mime_type, stt_provider)
        if not stt_result.success:
            yield SDKEvent(type=SDKEventType.ERROR, data={"error": stt_result.error})
            return

        yield SDKEvent(
            type=SDKEventType.VOICE_STT_COMPLETE,
            data={"text": stt_result.user_text, "confidence": stt_result.confidence},
        )

        # Step 2: Stream chat response
        full_response = ""
        async for event in self.stream(stt_result.user_text or ""):
            if event.type == SDKEventType.STREAM:
                full_response += event.data.get("delta", "")
            yield event

        # Step 3: Synthesize response
        if full_response:
            yield SDKEvent(type=SDKEventType.VOICE_TTS_START, data={})
            try:
                audio_bytes = await self.tts(full_response, voice, tts_provider)
                yield SDKEvent(
                    type=SDKEventType.VOICE_TTS_COMPLETE,
                    data={
                        "audio_size": len(audio_bytes),
                        "audio_base64": audio_bytes.decode("latin1"),  # Base64 in real impl
                    },
                )
            except Exception as e:
                yield SDKEvent(type=SDKEventType.ERROR, data={"error": f"TTS failed: {e}"})

    async def list_voices(self, provider: str = "auto") -> Dict[str, str]:
        """
        List available voices for TTS.

        Args:
            provider: TTS provider (auto, edge, openai, elevenlabs, local)

        Returns:
            Dictionary of voice ID -> voice name

        Usage:
            voices = await client.list_voices("edge")
            for voice_id, voice_name in voices.items():
                print(f"{voice_id}: {voice_name}")
        """
        if self._local_mode:
            from ..core.interface.modalities.voice.providers.edge_tts import EdgeTTSProvider

            if provider == "auto" or provider == "edge":
                return EdgeTTSProvider.VOICES
            return {}
        else:
            client = await self._get_http_client()
            response = await client.get(f"/api/voice/voices?provider={provider}")
            return response.json()

    # =========================================================================
    # SWARM (MULTI-AGENT COORDINATION)
    # =========================================================================

    async def swarm(self, goal: str, swarm_type: Optional[str] = None, **kwargs) -> SDKResponse:
        """
        Execute a multi-agent swarm for complex coordination.

        Args:
            goal: The goal/task for the swarm
            swarm_type: Optional swarm type (coding, research, testing, data_analysis, devops)
            **kwargs: Additional swarm configuration

        Returns:
            SDKResponse with swarm results

        Usage:
            # Auto-select swarm based on goal
            result = await client.swarm("Build a web scraper")

            # Explicit swarm type
            result = await client.swarm("Analyze this dataset", swarm_type="data_analysis")
        """
        await self._emit_event(SDKEventType.START, {"goal": goal, "swarm_type": swarm_type})

        context = self._create_context(ExecutionMode.SWARM, **kwargs)
        start_time = datetime.now()

        try:
            if self._local_mode:
                result = await self._local_swarm(goal, swarm_type, context)
            else:
                result = await self._remote_swarm(goal, swarm_type, context)

            execution_time = (datetime.now() - start_time).total_seconds()

            return SDKResponse(
                success=result.get("success", True),
                content=result.get("final_output", result.get("content", result)),
                mode=ExecutionMode.SWARM,
                request_id=context.request_id,
                execution_time=execution_time,
                agents_used=result.get("agents_used", []),
                steps_executed=result.get("steps_executed", 0),
                metadata=result.get("metadata", {}),
                error=result.get("error"),
            )

        except Exception as e:
            await self._emit_event(SDKEventType.ERROR, {"error": str(e)})
            return SDKResponse.error_response(str(e))

    async def _local_swarm(
        self, goal: str, swarm_type: Optional[str], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Execute swarm locally using Orchestrator."""
        try:
            from ..core.intelligence.orchestration import Orchestrator

            # Create orchestrator with optional swarm type
            orchestrator = Orchestrator(agents=swarm_type if swarm_type else "auto")

            result = await orchestrator.run(goal=goal)

            return {
                "success": True,
                "final_output": result.get("final_output", ""),
                "content": result.get("final_output", ""),
                "agents_used": result.get("agents_used", []),
                "steps_executed": result.get("steps_executed", 0),
                "metadata": result.get("metadata", {}),
            }
        except Exception as e:
            logger.error(f"Swarm execution error: {e}", exc_info=True)
            return {"success": False, "error": str(e), "final_output": None}

    async def _remote_swarm(
        self, goal: str, swarm_type: Optional[str], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Execute swarm via HTTP."""
        client = await self._get_http_client()
        response = await client.post(
            "/api/swarm",
            json={"goal": goal, "swarm_type": swarm_type, "context": context.to_dict()},
        )
        response.raise_for_status()
        return response.json()

    async def swarm_stream(
        self, goal: str, swarm_type: Optional[str] = None, **kwargs
    ) -> AsyncIterator[SDKEvent]:
        """
        Execute swarm with streaming events.

        Args:
            goal: The goal/task
            swarm_type: Optional swarm type
            **kwargs: Additional parameters

        Yields:
            SDKEvent objects for swarm progress

        Usage:
            async for event in client.swarm_stream("Research AI trends"):
                if event.type == SDKEventType.SWARM_AGENT_START:
                    print(f"Agent started: {event.data['agent']}")
                elif event.type == SDKEventType.STREAM:
                    print(event.data['delta'], end="")
        """
        context = self._create_context(ExecutionMode.SWARM, streaming=True, **kwargs)

        try:
            if self._local_mode:
                async for event in self._local_swarm_stream(goal, swarm_type, context):
                    yield event
            else:
                async for event in self._remote_swarm_stream(goal, swarm_type, context):
                    yield event

        except Exception as e:
            yield SDKEvent(type=SDKEventType.ERROR, data={"error": str(e)})

    async def _local_swarm_stream(
        self, goal: str, swarm_type: Optional[str], context: ExecutionContext
    ) -> AsyncIterator[SDKEvent]:
        """Stream swarm execution locally."""
        try:
            from ..core.intelligence.orchestration import Orchestrator

            # Emit start event
            yield SDKEvent(type=SDKEventType.START, data={"goal": goal})

            # Create orchestrator
            orchestrator = Orchestrator(agents=swarm_type if swarm_type else "auto")

            # TODO: Orchestrator doesn't have streaming yet, so run and emit complete
            result = await orchestrator.run(goal=goal)

            # Emit completion
            yield SDKEvent(
                type=SDKEventType.COMPLETE,
                data={
                    "final_output": result.get("final_output"),
                    "agents_used": result.get("agents_used", []),
                },
            )

        except Exception as e:
            logger.error(f"Swarm stream error: {e}", exc_info=True)
            yield SDKEvent(type=SDKEventType.ERROR, data={"error": str(e)})

    async def _remote_swarm_stream(
        self, goal: str, swarm_type: Optional[str], context: ExecutionContext
    ) -> AsyncIterator[SDKEvent]:
        """Stream swarm execution via WebSocket."""
        try:
            import websockets
        except ImportError:
            raise ImportError(
                "websockets required for streaming. Install with: pip install websockets"
            )

        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/ws/swarm"

        async with websockets.connect(ws_url) as ws:
            await ws.send(
                json.dumps({"goal": goal, "swarm_type": swarm_type, "context": context.to_dict()})
            )

            async for msg in ws:
                data = json.loads(msg)
                event_type = SDKEventType(data.get("type", "stream"))
                yield SDKEvent(
                    type=event_type, data=data.get("data"), context_id=context.request_id
                )

                if event_type == SDKEventType.COMPLETE:
                    break

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    async def configure_lm(
        self, provider: Optional[str] = None, model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Configure the language model provider and model.

        Args:
            provider: LM provider (anthropic, openai, groq, etc.)
            model: Model name (claude-3-5-sonnet-20241022, gpt-4, etc.)

        Returns:
            Configuration status

        Usage:
            await client.configure_lm(provider="anthropic", model="claude-3-5-sonnet-20241022")
        """
        if self._local_mode:
            try:
                from ..core.intelligence.orchestration.llm_providers.provider_manager import (
                    configure_dspy_lm,
                )

                configure_dspy_lm(provider=provider, model=model)
                return {"success": True, "provider": provider, "model": model}
            except Exception as e:
                return {"success": False, "error": str(e)}
        else:
            client = await self._get_http_client()
            response = await client.post(
                "/api/configure/lm", json={"provider": provider, "model": model}
            )
            return response.json()

    async def configure_voice(
        self, stt_provider: Optional[str] = None, tts_provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Configure voice providers for STT/TTS.

        Args:
            stt_provider: STT provider (auto, groq, whisper, local)
            tts_provider: TTS provider (auto, edge, openai, elevenlabs, local)

        Returns:
            Available providers

        Usage:
            await client.configure_voice(stt_provider="groq", tts_provider="edge")
        """
        return {
            "stt_provider": stt_provider or "auto",
            "tts_provider": tts_provider or "auto",
            "available_stt": ["groq", "whisper", "local"],
            "available_tts": ["edge", "openai", "elevenlabs", "local"],
        }

    # =========================================================================
    # MEMORY
    # =========================================================================

    async def memory_store(self, content: str, level: str = "episodic", **kwargs) -> SDKResponse:
        """
        Store content in memory system.

        Args:
            content: Content to store
            level: Memory level (episodic, semantic, procedural, meta, causal)
            **kwargs: Additional metadata (goal, metadata, etc.)

        Returns:
            SDKResponse with memory ID

        Usage:
            result = await client.memory_store(
                "Task X succeeded using approach Y",
                level="episodic",
                goal="research",
                metadata={"reward": 1.0}
            )
        """
        try:
            if self._local_mode:
                from ..core.intelligence.memory.facade import get_memory_system

                mem = get_memory_system()
                mem_id = mem.store(
                    content=content,
                    level=level,
                    goal=kwargs.get("goal"),
                    metadata=kwargs.get("metadata", {}),
                )
                return SDKResponse(
                    success=True, content=mem_id, metadata={"level": level, "memory_id": mem_id}
                )
            else:
                client = await self._get_http_client()
                response = await client.post(
                    "/api/memory/store", json={"content": content, "level": level, **kwargs}
                )
                return SDKResponse(success=True, content=response.json())

        except Exception as e:
            return SDKResponse.error_response(str(e))

    async def memory_retrieve(self, query: str, top_k: int = 5, **kwargs) -> SDKResponse:
        """
        Retrieve relevant memories.

        Args:
            query: Query string
            top_k: Number of memories to retrieve
            **kwargs: Additional filters (goal, level, etc.)

        Returns:
            SDKResponse with list of memories

        Usage:
            result = await client.memory_retrieve("How to handle task X?", top_k=5)
            for memory in result.content:
                print(f"[{memory['level']}] {memory['content']}")
        """
        try:
            if self._local_mode:
                from ..core.intelligence.memory.facade import get_memory_system

                mem = get_memory_system()
                results = mem.retrieve(query=query, goal=kwargs.get("goal"), top_k=top_k)
                memories = [
                    {
                        "content": r.content,
                        "level": r.level,
                        "relevance": r.relevance,
                        "metadata": r.metadata,
                    }
                    for r in results
                ]
                return SDKResponse(
                    success=True,
                    content=memories,
                    metadata={"query": query, "count": len(memories)},
                )
            else:
                client = await self._get_http_client()
                response = await client.post(
                    "/api/memory/retrieve", json={"query": query, "top_k": top_k, **kwargs}
                )
                return SDKResponse(success=True, content=response.json())

        except Exception as e:
            return SDKResponse.error_response(str(e))

    async def memory_status(self) -> SDKResponse:
        """
        Get memory system status.

        Returns:
            SDKResponse with memory statistics

        Usage:
            status = await client.memory_status()
            print(f"Total memories: {status.content['total_memories']}")
        """
        try:
            if self._local_mode:
                from ..core.intelligence.memory.facade import get_memory_system

                mem = get_memory_system()
                status = mem.status()
                return SDKResponse(success=True, content=status)
            else:
                client = await self._get_http_client()
                response = await client.get("/api/memory/status")
                return SDKResponse(success=True, content=response.json())

        except Exception as e:
            return SDKResponse.error_response(str(e))

    # =========================================================================
    # DOCUMENTS & RAG
    # =========================================================================

    async def upload_document(self, file_content: bytes, filename: str, **kwargs) -> SDKResponse:
        """
        Upload a document for RAG.

        Args:
            file_content: File bytes
            filename: Filename with extension
            **kwargs: Additional metadata

        Returns:
            SDKResponse with document ID

        Usage:
            with open("report.pdf", "rb") as f:
                result = await client.upload_document(f.read(), "report.pdf")
            print(f"Document ID: {result.content['doc_id']}")
        """
        try:
            if self._local_mode:
                # Store in memory as semantic knowledge
                from ..core.intelligence.memory.facade import get_memory_system

                mem = get_memory_system()

                # Extract text (basic implementation, can be enhanced)
                if filename.endswith(".txt"):
                    content_text = file_content.decode("utf-8")
                else:
                    content_text = f"Document: {filename} ({len(file_content)} bytes)"

                doc_id = mem.store(
                    content=content_text,
                    level="semantic",
                    metadata={"filename": filename, "size": len(file_content), **kwargs},
                )

                return SDKResponse(
                    success=True,
                    content={"doc_id": doc_id, "filename": filename},
                    metadata={"size": len(file_content)},
                )
            else:
                client = await self._get_http_client()
                response = await client.post(
                    "/api/documents/upload", files={"file": (filename, file_content)}, data=kwargs
                )
                return SDKResponse(success=True, content=response.json())

        except Exception as e:
            return SDKResponse.error_response(str(e))

    async def search_documents(self, query: str, top_k: int = 5, **kwargs) -> SDKResponse:
        """
        Search uploaded documents.

        Args:
            query: Search query
            top_k: Number of results
            **kwargs: Additional filters

        Returns:
            SDKResponse with document chunks

        Usage:
            results = await client.search_documents("machine learning", top_k=3)
            for doc in results.content:
                print(f"{doc['filename']}: {doc['excerpt']}")
        """
        # Delegates to memory_retrieve with semantic level filter
        return await self.memory_retrieve(query, top_k=top_k, level="semantic", **kwargs)

    async def chat_with_documents(
        self, message: str, doc_ids: Optional[List[str]] = None, **kwargs
    ) -> SDKResponse:
        """
        Chat with context from documents.

        Args:
            message: User message
            doc_ids: Optional list of document IDs to use as context
            **kwargs: Additional parameters

        Returns:
            SDKResponse with answer grounded in documents

        Usage:
            result = await client.chat_with_documents(
                "What are the key findings?",
                doc_ids=["doc-123", "doc-456"]
            )
        """
        # Retrieve relevant document chunks
        if doc_ids:
            # Filter by doc_ids (would need enhanced memory_retrieve)
            doc_context = f"Reference documents: {', '.join(doc_ids)}\n\n"
        else:
            # Search all documents
            search_results = await self.search_documents(message, top_k=3)
            if search_results.success:
                doc_context = "\n\n".join(
                    [
                        f"[{m.get('filename', 'Document')}] {m.get('content', '')}"
                        for m in search_results.content
                    ]
                )
            else:
                doc_context = ""

        # Chat with document context
        full_message = f"{doc_context}\n\nUser question: {message}"
        return await self.chat(full_message, **kwargs)

    # =========================================================================
    # SKILL & AGENT HANDLES
    # =========================================================================

    def skill(self, name: str) -> SkillHandle:
        """
        Get a handle for direct skill execution.

        Args:
            name: Skill name

        Returns:
            SkillHandle for executing the skill
        """
        return SkillHandle(self, name)

    def agent(self, name: str) -> AgentHandle:
        """
        Get a handle for direct agent execution.

        Args:
            name: Agent name

        Returns:
            AgentHandle for executing with the agent
        """
        return AgentHandle(self, name)

    async def _execute_skill(self, name: str, params: Dict[str, Any]) -> SDKResponse:
        """Execute a skill directly."""
        await self._emit_event(SDKEventType.SKILL_START, {"skill": name, "params": params})

        start_time = datetime.now()

        try:
            if self._local_mode:
                from ..core.interface.api.registry import get_unified_registry

                registry = get_unified_registry()
                skill = registry.get_skill(name)
                if skill:
                    result = await skill.execute(params)
                else:
                    raise ValueError(f"Skill not found: {name}")
            else:
                client = await self._get_http_client()
                response = await client.post(f"/api/skill/{name}", json=params)
                response.raise_for_status()
                result = response.json()

            execution_time = (datetime.now() - start_time).total_seconds()
            await self._emit_event(SDKEventType.SKILL_COMPLETE, {"skill": name, "result": result})

            return SDKResponse(
                success=True,
                content=result,
                mode=ExecutionMode.SKILL,
                execution_time=execution_time,
                skills_used=[name],
            )

        except Exception as e:
            await self._emit_event(SDKEventType.ERROR, {"error": str(e)})
            return SDKResponse.error_response(str(e))

    async def _execute_agent(self, name: str, task: str, context: Dict[str, Any]) -> SDKResponse:
        """Execute with a specific agent."""
        await self._emit_event(SDKEventType.AGENT_START, {"agent": name, "task": task})

        start_time = datetime.now()

        try:
            if self._local_mode:
                # Get agent from registry and execute
                from ..core.interface.api.agents import AutoAgent

                agent = AutoAgent()
                result = await agent.execute(task)
                result_dict = {
                    "success": result.success,
                    "content": result.final_output,
                    "skills_used": result.skills_used,
                    "steps_executed": result.steps_executed,
                    "errors": getattr(result, "errors", []) or [],
                    "stopped_early": getattr(result, "stopped_early", False),
                }
            else:
                client = await self._get_http_client()
                response = await client.post(
                    f"/api/agent/{name}", json={"task": task, "context": context}
                )
                response.raise_for_status()
                result_dict = response.json()

            execution_time = (datetime.now() - start_time).total_seconds()
            await self._emit_event(
                SDKEventType.AGENT_COMPLETE, {"agent": name, "result": result_dict}
            )

            # Propagate errors properly
            errors = result_dict.get("errors", [])
            stopped_early = result_dict.get("stopped_early", False)
            is_success = result_dict.get("success", True)

            # If there are errors, it's not a success
            if errors:
                is_success = False

            return SDKResponse(
                success=is_success,
                content=result_dict.get("content", result_dict),
                mode=ExecutionMode.AGENT,
                execution_time=execution_time,
                agents_used=[name],
                skills_used=result_dict.get("skills_used", []),
                steps_executed=result_dict.get("steps_executed", 0),
                errors=errors,
                stopped_early=stopped_early,
                error=errors[0] if errors else None,
            )

        except Exception as e:
            await self._emit_event(SDKEventType.ERROR, {"error": str(e)})
            return SDKResponse.error_response(str(e))

    async def _get_skill_info(self, name: str) -> Dict[str, Any]:
        """Get skill information."""
        if self._local_mode:
            from ..core.interface.api.registry import get_unified_registry

            registry = get_unified_registry()
            skill = registry.get_skill(name)
            if skill:
                return {
                    "name": skill.name,
                    "description": skill.description,
                    "tools": [t.name for t in skill.tools] if hasattr(skill, "tools") else [],
                }
            return {"error": "Skill not found"}
        else:
            client = await self._get_http_client()
            response = await client.get(f"/api/skill/{name}/info")
            return response.json()

    async def _get_agent_info(self, name: str) -> Dict[str, Any]:
        """Get agent information."""
        if self._local_mode:
            return {"name": name, "type": "auto"}
        else:
            client = await self._get_http_client()
            response = await client.get(f"/api/agent/{name}/info")
            return response.json()

    # =========================================================================
    # SESSION
    # =========================================================================

    async def session(self, user_id: str) -> SessionHandle:
        """
        Get or create a session for a user.

        Args:
            user_id: User identifier

        Returns:
            SessionHandle for managing the session
        """
        # Check cache
        if user_id in self._sessions:
            return SessionHandle(self, self._sessions[user_id])

        # Try to load existing session
        session = await self._load_session(user_id)
        if session is None:
            # Create new session
            session = SDKSession(session_id=str(uuid.uuid4()), user_id=user_id)

        self._sessions[user_id] = session
        return SessionHandle(self, session)

    async def _load_session(self, user_id: str) -> Optional[SDKSession]:
        """Load session from storage."""
        if self._local_mode:
            # Try to load from local file
            from pathlib import Path

            session_path = Path.home() / "jotty" / "sessions" / f"{user_id}.json"
            if session_path.exists():
                data = json.loads(session_path.read_text())
                return SDKSession.from_dict(data)
            return None
        else:
            try:
                client = await self._get_http_client()
                response = await client.get(f"/api/session/{user_id}")
                if response.status_code == 200:
                    return SDKSession.from_dict(response.json())
            except Exception:
                pass
            return None

    async def _save_session(self, session: SDKSession) -> None:
        """Save session to storage."""
        if self._local_mode:
            from pathlib import Path

            session_dir = Path.home() / "jotty" / "sessions"
            session_dir.mkdir(parents=True, exist_ok=True)
            session_path = session_dir / f"{session.user_id}.json"
            session_path.write_text(json.dumps(session.to_dict(), indent=2))
        else:
            client = await self._get_http_client()
            await client.put(f"/api/session/{session.user_id}", json=session.to_dict())

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    async def health(self) -> Dict[str, Any]:
        """Check API health."""
        if self._local_mode:
            return {"status": "healthy", "mode": "local"}
        else:
            client = await self._get_http_client()
            response = await client.get("/health")
            return response.json()

    async def list_skills(self) -> List[str]:
        """List available skills."""
        if self._local_mode:
            from ..core.interface.api.registry import get_unified_registry

            registry = get_unified_registry()
            return registry.list_skills()
        else:
            client = await self._get_http_client()
            response = await client.get("/api/skills")
            return response.json().get("skills", [])

    async def list_agents(self) -> List[str]:
        """List available agents."""
        if self._local_mode:
            return ["auto", "research", "coding", "testing"]
        else:
            client = await self._get_http_client()
            response = await client.get("/api/agents")
            return response.json().get("agents", [])

    async def close(self) -> None:
        """Close the client and cleanup resources."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None


# =============================================================================
# SYNC WRAPPER
# =============================================================================


class JottySync:
    """
    Synchronous wrapper for Jotty client.

    For environments where async isn't convenient.

    Usage:
        client = JottySync()
        response = client.chat("Hello!")
    """

    def __init__(self, **kwargs):
        self._async_client = Jotty(**kwargs)

    def _run(self, coro):
        try:
            asyncio.get_running_loop()
            # Already in async context — offload to a thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        except RuntimeError:
            return asyncio.run(coro)

    def chat(self, message: str, **kwargs) -> SDKResponse:
        return self._run(self._async_client.chat(message, **kwargs))

    def workflow(self, goal: str, **kwargs) -> SDKResponse:
        return self._run(self._async_client.workflow(goal, **kwargs))

    def skill(self, name: str) -> SkillHandle:
        return self._async_client.skill(name)

    def agent(self, name: str) -> AgentHandle:
        return self._async_client.agent(name)

    def on(self, event: str, callback: Callable):
        self._async_client.on(event, callback)
        return self

    def close(self):
        self._run(self._async_client.close())


# =============================================================================
# CONVENIENCE ALIASES
# =============================================================================

# Alias for backward compatibility and convenience
Client = Jotty
SyncClient = JottySync
