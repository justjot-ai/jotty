"""
Chat Executor

Handles execution of chat interactions.
"""

from typing import Dict, Any, Optional, List, AsyncIterator
import logging
import time

from .chat_orchestrator import ChatOrchestrator
from .chat_context import ChatContext, ChatMessage
from Jotty.core.ui.a2ui import is_a2ui_response, convert_to_a2ui_response

logger = logging.getLogger(__name__)


class ChatExecutor:
    """
    Executes chat interactions with agents.
    """
    
    def __init__(
        self,
        conductor: Any,  # Conductor instance
        orchestrator: ChatOrchestrator,
        context: Optional[ChatContext] = None
    ):
        """
        Initialize chat executor.
        
        Args:
            conductor: Jotty Conductor instance
            orchestrator: Chat orchestrator for agent selection
            context: Chat context manager
        """
        self.conductor = conductor
        self.orchestrator = orchestrator
        self.context = context or ChatContext()
    
    async def execute(
        self,
        message: str,
        history: Optional[List[ChatMessage]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute chat interaction synchronously.
        
        Args:
            message: User message
            history: Conversation history
            context: Additional context
            
        Returns:
            Chat response dictionary
        """
        start_time = time.time()
        
        # Add user message to context
        self.context.add_message("user", message)
        
        # Select agent
        agent_id = self.orchestrator.select_agent(message, history, context)
        
        # Prepare agent context
        agent_context = self.orchestrator.prepare_agent_context(
            message, history, context
        )
        
        # Execute agent
        try:
            # Use conductor.run() since run_actor() doesn't exist
            # Pass agent context as kwargs which includes actor selection
            result = await self.conductor.run(
                goal=message,
                **agent_context
            )

            # Extract response
            response_text = self._extract_response(result)

            # Add assistant message to context
            self.context.add_message("assistant", response_text)

            execution_time = time.time() - start_time

            # Convert result to serializable dict (DSPy Prediction objects are not JSON serializable)
            result_dict = {}
            if isinstance(result, dict):
                result_dict = result
            elif hasattr(result, "_store"):
                # DSPy Prediction object - extract fields from _store
                result_dict = dict(result._store)
            else:
                result_dict = {"value": str(result)}

            return {
                "success": True,
                "message": response_text,
                "agent": agent_id,
                "execution_time": execution_time,
                "metadata": {
                    "result": result_dict,  # Now serializable!
                    "context_used": len(agent_context)
                }
            }
            
        except Exception as e:
            logger.error(f"Chat execution failed: {e}", exc_info=True)
            execution_time = time.time() - start_time
            
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "agent": agent_id,
                "execution_time": execution_time,
                "error": str(e)
            }
    
    async def stream(
        self,
        message: str,
        history: Optional[List[ChatMessage]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Execute chat interaction with streaming.
        
        Args:
            message: User message
            history: Conversation history
            context: Additional context
            
        Yields:
            Event dictionaries
        """
        # Add user message to context
        self.context.add_message("user", message)
        
        # Select agent
        agent_id = self.orchestrator.select_agent(message, history, context)
        
        # Emit agent selection event
        yield {
            "type": "agent_selected",
            "agent": agent_id,
            "timestamp": time.time()
        }
        
        # Prepare agent context
        agent_context = self.orchestrator.prepare_agent_context(
            message, history, context
        )
        
        # Stream agent execution
        try:
            if hasattr(self.conductor, 'run_actor_stream'):
                async for event in self.conductor.run_actor_stream(
                    actor_name=agent_id,
                    goal=message,
                    context=agent_context
                ):
                    # Transform to chat events
                    for chat_event in self._transform_to_chat_events(event):
                        yield chat_event
            else:
                # Fallback: execute synchronously and stream result
                # Conductor.run() executes with the specified actor via kwargs
                result = await self.conductor.run(
                    goal=message,
                    **agent_context  # Pass context as kwargs (includes actor_name if specified)
                )

                # Debug logging
                logger.info(f"Conductor result type: {type(result)}, value: {result}")

                # Extract final_output from SwarmResult if applicable
                actual_result = result.final_output if hasattr(result, 'final_output') else result

                # Check if result is A2UI format
                if is_a2ui_response(actual_result):
                    # Stream A2UI widgets
                    logger.info("Streaming A2UI widget response")
                    a2ui_response = convert_to_a2ui_response(actual_result)

                    # Extract text for context (combine all text blocks)
                    response_text = self._extract_text_from_a2ui(a2ui_response)
                    self.context.add_message("assistant", response_text)

                    # Emit A2UI widget event
                    yield {
                        "type": "a2ui_widget",
                        "content": a2ui_response["content"],
                        "timestamp": time.time()
                    }

                    yield {
                        "type": "done",
                        "message": response_text,
                        "timestamp": time.time()
                    }
                else:
                    # Stream plain text result
                    response_text = self._extract_response(result)
                    self.context.add_message("assistant", response_text)

                    # Emit text chunks
                    for chunk in self._chunk_text(response_text):
                        yield {
                            "type": "text_chunk",
                            "content": chunk,
                            "timestamp": time.time()
                        }

                    yield {
                        "type": "done",
                        "message": response_text,
                        "timestamp": time.time()
                    }
                
        except Exception as e:
            logger.error(f"Chat streaming failed: {e}", exc_info=True)
            yield {
                "type": "error",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _extract_response(self, result: Any) -> str:
        """Extract response text from agent result."""
        # Handle None result
        if result is None:
            logger.warning(f"Conductor returned None result, using empty string")
            return ""

        if isinstance(result, dict):
            return result.get("response", result.get("output", str(result)))
        elif hasattr(result, "response"):
            return result.response
        elif hasattr(result, "output"):
            return result.output
        elif hasattr(result, "final_output"):
            return result.final_output
        else:
            return str(result)

    def _extract_text_from_a2ui(self, a2ui_response: Dict[str, Any]) -> str:
        """
        Extract text content from A2UI response for conversation context.

        Args:
            a2ui_response: A2UI formatted response

        Returns:
            Combined text content
        """
        texts = []
        content = a2ui_response.get("content", [])

        for block in content:
            block_type = block.get("type", "")

            if block_type == "text":
                texts.append(block.get("text", ""))
            elif block_type == "card":
                if block.get("title"):
                    texts.append(block["title"])
                if block.get("subtitle"):
                    texts.append(block["subtitle"])
                # Recursively extract from body if it's a list
                body = block.get("body", [])
                if isinstance(body, list):
                    for item in body:
                        if isinstance(item, dict) and item.get("type") == "text":
                            texts.append(item.get("text", ""))
            elif block_type == "list":
                items = block.get("items", [])
                for item in items:
                    if isinstance(item, dict):
                        if item.get("title"):
                            texts.append(item["title"])
                        if item.get("subtitle"):
                            texts.append(item["subtitle"])

        return " ".join(texts) if texts else "A2UI widget response"
    
    def _transform_to_chat_events(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Transform agent events to chat events."""
        chat_events = []
        event_type = event.get("type", "")
        
        if event_type == "agent_complete":
            result = event.get("result", {})
            
            # Reasoning
            if result.get("reasoning"):
                chat_events.append({
                    "type": "reasoning",
                    "content": result["reasoning"],
                    "timestamp": time.time()
                })
            
            # Tool calls
            for tool_call in result.get("tool_calls", []):
                chat_events.append({
                    "type": "tool_call",
                    "tool": tool_call.get("name"),
                    "args": tool_call.get("arguments"),
                    "timestamp": time.time()
                })
            
            # Tool results
            for tool_result in result.get("tool_results", []):
                chat_events.append({
                    "type": "tool_result",
                    "result": tool_result.get("result"),
                    "timestamp": time.time()
                })
            
            # Text response
            response_text = result.get("response", "")
            if response_text:
                for chunk in self._chunk_text(response_text):
                    chat_events.append({
                        "type": "text_chunk",
                        "content": chunk,
                        "timestamp": time.time()
                    })
                
                chat_events.append({
                    "type": "done",
                    "message": response_text,
                    "timestamp": time.time()
                })
        
        return chat_events
    
    def _chunk_text(self, text: str, chunk_size: int = 50) -> List[str]:
        """Split text into chunks for streaming."""
        # Handle None or empty text
        if not text:
            return []

        # Simple sentence-based chunking
        sentences = text.split(". ")
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
