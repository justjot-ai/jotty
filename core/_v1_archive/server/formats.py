"""
SSE Format Converters

Converts Jotty events to different SSE formats for various clients:
- useChat (Vercel AI SDK)
- OpenAI format
- Anthropic format
- Raw Jotty format
"""

import json
from typing import Dict, Any, List
from abc import ABC, abstractmethod


class SSEFormatter:
    """Base class for SSE formatters."""
    
    def format_event(self, event: Dict[str, Any]) -> List[str]:
        """Format a single event to SSE format."""
        return []
    
    def end_marker(self) -> str:
        """Return end-of-stream marker."""
        return "data: [DONE]\n\n"
    
    def error_event(self, error: str) -> str:
        """Format error event."""
        return f"event: error\ndata: {json.dumps({'error': error})}\n\n"


class useChatFormatter(SSEFormatter):
    """
    useChat-compatible SSE formatter (Vercel AI SDK).
    
    Format: data: {"type":"text-delta","textDelta":"char"}
    End: data: [DONE]
    """
    
    def format_event(self, event: Dict[str, Any]) -> List[str]:
        """Format event for useChat."""
        events = []
        event_type = event.get('type', '')
        
        if event_type == 'text_chunk':
            # Stream character-by-character for typing effect
            content = event.get('content', '')
            for char in content:
                chunk = json.dumps({
                    "type": "text-delta",
                    "textDelta": char
                })
                events.append(f"data: {chunk}\n\n")
        
        elif event_type == 'done':
            # Final message
            final_text = event.get('message', '')
            if final_text:
                for char in final_text:
                    chunk = json.dumps({
                        "type": "text-delta",
                        "textDelta": char
                    })
                    events.append(f"data: {chunk}\n\n")
        
        elif event_type == 'error':
            error_msg = f"Error: {event.get('error', 'Unknown error')}"
            for char in error_msg:
                chunk = json.dumps({
                    "type": "text-delta",
                    "textDelta": char
                })
                events.append(f"data: {chunk}\n\n")
        
        return events
    
    def end_marker(self) -> str:
        """Return useChat end marker."""
        return "data: [DONE]\n\n"


class OpenAIFormatter(SSEFormatter):
    """
    OpenAI-compatible SSE formatter.
    
    Format: data: {"choices":[{"delta":{"content":"text"}}]}
    """
    
    def format_event(self, event: Dict[str, Any]) -> List[str]:
        """Format event for OpenAI format."""
        events = []
        event_type = event.get('type', '')
        
        if event_type == 'text_chunk':
            content = event.get('content', '')
            for char in content:
                chunk = {
                    "choices": [{
                        "delta": {
                            "content": char
                        }
                    }]
                }
                events.append(f"data: {json.dumps(chunk)}\n\n")
        
        elif event_type == 'done':
            # Final chunk with finish_reason
            chunk = {
                "choices": [{
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            events.append(f"data: {json.dumps(chunk)}\n\n")
            events.append("data: [DONE]\n\n")
        
        return events
    
    def end_marker(self) -> str:
        """Return OpenAI end marker."""
        return "data: [DONE]\n\n"


class AnthropicFormatter(SSEFormatter):
    """
    Anthropic-compatible SSE formatter.
    
    Format: event: content_block_delta\ndata: {"delta":{"text":"..."}}
    """
    
    def format_event(self, event: Dict[str, Any]) -> List[str]:
        """Format event for Anthropic format."""
        events = []
        event_type = event.get('type', '')
        
        if event_type == 'text_chunk':
            content = event.get('content', '')
            for char in content:
                chunk = {
                    "delta": {
                        "text": char
                    }
                }
                events.append(f"event: content_block_delta\ndata: {json.dumps(chunk)}\n\n")
        
        elif event_type == 'done':
            events.append("event: message_stop\ndata: {}\n\n")
        
        return events
    
    def end_marker(self) -> str:
        """Return Anthropic end marker."""
        return "event: message_stop\ndata: {}\n\n"
