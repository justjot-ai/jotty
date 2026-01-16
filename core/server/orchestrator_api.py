"""
Orchestrator API for JustJot.ai Integration
==========================================

Exposes core orchestration operations (generate, improve, summarize) via HTTP API.
Reuses Jotty's existing infrastructure - no duplication.

Endpoints:
- POST /api/orchestrator/generate - Content generation
- POST /api/orchestrator/improve - Content improvement
- POST /api/orchestrator/summarize - Content summarization
- POST /api/orchestrator/preprocess - Request preprocessing
"""

from flask import Blueprint, request, jsonify, Response
from typing import Dict, Any, Optional, List
import logging
import json
import asyncio
import dspy

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

from ..foundation.unified_lm_provider import UnifiedLMProvider
from ..use_cases.chat import ChatUseCase, ChatMessage

logger = logging.getLogger(__name__)

orchestrator_bp = Blueprint('orchestrator', __name__)


def _get_jotty_api():
    """Get Jotty API instance from Flask app context."""
    from flask import current_app
    if hasattr(current_app, 'jotty_api'):
        return current_app.jotty_api
    
    if hasattr(request, 'jotty_api'):
        return request.jotty_api
    
    raise RuntimeError("Jotty API not available in Flask app context")


class ContentGenerator:
    """Content generation using DSPy - reuses existing infrastructure."""
    
    def __init__(self, jotty_api):
        self.jotty_api = jotty_api
    
    async def generate(
        self,
        prompt: str,
        content_type: str = "text",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate content using DSPy.
        
        Reuses ChatUseCase with generation-focused prompt.
        """
        # Configure LM provider if specified
        if provider:
            try:
                lm = UnifiedLMProvider.create_lm(provider, model=model)
                dspy.configure(lm=lm)
                logger.info(f"🔵 GENERATE: Configured LM provider: {provider}, model: {model}")
            except Exception as e:
                logger.warning(f"⚠️  GENERATE: Failed to configure provider {provider}: {e}")
        
        # Build generation prompt
        generation_prompt = self._build_generation_prompt(prompt, content_type)
        
        # Use ChatUseCase for generation (reuses existing infrastructure)
        chat_use_case = ChatUseCase(
            conductor=self.jotty_api.conductor if hasattr(self.jotty_api, 'conductor') else None,
            mode='static'
        )
        
        result = await chat_use_case.execute(
            goal=generation_prompt,
            context={'content_type': content_type, 'temperature': temperature}
        )
        
        return {
            'success': result.success,
            'content': result.output,
            'metadata': result.metadata
        }
    
    def _build_generation_prompt(self, prompt: str, content_type: str) -> str:
        """Build generation-focused prompt."""
        type_instructions = {
            'text': 'Generate high-quality text content',
            'code': 'Generate clean, well-documented code',
            'markdown': 'Generate well-structured Markdown content',
            'json': 'Generate valid JSON content',
            'html': 'Generate clean HTML content',
        }
        
        instruction = type_instructions.get(content_type, 'Generate content')
        return f"{instruction} based on the following request:\n\n{prompt}"


class ContentImprover:
    """Content improvement using DSPy - reuses existing infrastructure."""
    
    def __init__(self, jotty_api):
        self.jotty_api = jotty_api
    
    async def improve(
        self,
        content: str,
        instruction: str,
        temperature: float = 0.7,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Improve content using DSPy.
        
        Reuses ChatUseCase with improvement-focused prompt.
        """
        # Configure LM provider if specified
        if provider:
            try:
                lm = UnifiedLMProvider.create_lm(provider, model=model)
                dspy.configure(lm=lm)
                logger.info(f"🔵 IMPROVE: Configured LM provider: {provider}, model: {model}")
            except Exception as e:
                logger.warning(f"⚠️  IMPROVE: Failed to configure provider {provider}: {e}")
        
        # Build improvement prompt
        improvement_prompt = self._build_improvement_prompt(content, instruction)
        
        # Use ChatUseCase for improvement (reuses existing infrastructure)
        chat_use_case = ChatUseCase(
            conductor=self.jotty_api.conductor if hasattr(self.jotty_api, 'conductor') else None,
            mode='static'
        )
        
        result = await chat_use_case.execute(
            goal=improvement_prompt,
            context={'temperature': temperature}
        )
        
        return {
            'success': result.success,
            'content': result.output,
            'metadata': result.metadata
        }
    
    def _build_improvement_prompt(self, content: str, instruction: str) -> str:
        """Build improvement-focused prompt."""
        return f"""Improve the following content based on these instructions:

Instructions: {instruction}

Original content:
{content}

Please provide the improved version:"""


class ContentSummarizer:
    """Content summarization using DSPy - reuses existing infrastructure."""
    
    def __init__(self, jotty_api):
        self.jotty_api = jotty_api
    
    async def summarize(
        self,
        content: str,
        max_length: Optional[int] = None,
        focus: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Summarize content using DSPy.
        
        Reuses ChatUseCase with summarization-focused prompt.
        """
        # Configure LM provider if specified
        if provider:
            try:
                lm = UnifiedLMProvider.create_lm(provider, model=model)
                dspy.configure(lm=lm)
                logger.info(f"🔵 SUMMARIZE: Configured LM provider: {provider}, model: {model}")
            except Exception as e:
                logger.warning(f"⚠️  SUMMARIZE: Failed to configure provider {provider}: {e}")
        
        # Build summarization prompt
        summary_prompt = self._build_summarization_prompt(content, max_length, focus)
        
        # Use ChatUseCase for summarization (reuses existing infrastructure)
        chat_use_case = ChatUseCase(
            conductor=self.jotty_api.conductor if hasattr(self.jotty_api, 'conductor') else None,
            mode='static'
        )
        
        result = await chat_use_case.execute(
            goal=summary_prompt,
            context={'max_length': max_length, 'focus': focus}
        )
        
        return {
            'success': result.success,
            'summary': result.output,
            'metadata': result.metadata
        }
    
    def _build_summarization_prompt(self, content: str, max_length: Optional[int], focus: Optional[str]) -> str:
        """Build summarization-focused prompt."""
        length_instruction = f" in approximately {max_length} words" if max_length else ""
        focus_instruction = f" focusing on {focus}" if focus else ""
        
        return f"""Summarize the following content{length_instruction}{focus_instruction}:

{content}

Summary:"""


@orchestrator_bp.route('/api/orchestrator/generate', methods=['POST'])
def generate():
    """
    Generate content.
    
    Request body:
    {
        "prompt": "Write a blog post about AI",
        "contentType": "text",  # text, code, markdown, json, html
        "temperature": 0.7,
        "maxTokens": 1000,
        "provider": "opencode",  # Optional
        "model": "default"  # Optional
    }
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Request body required"}), 400
        
        prompt = data.get('prompt')
        if not prompt:
            return jsonify({"error": "prompt is required"}), 400
        
        jotty_api = _get_jotty_api()
        generator = ContentGenerator(jotty_api)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                generator.generate(
                    prompt=prompt,
                    content_type=data.get('contentType', 'text'),
                    temperature=data.get('temperature', 0.7),
                    max_tokens=data.get('maxTokens'),
                    provider=data.get('provider'),
                    model=data.get('model')
                )
            )
            
            return jsonify({
                "success": result.get('success', True),
                "content": result.get('content', ''),
                "metadata": result.get('metadata', {})
            })
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Generate error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@orchestrator_bp.route('/api/orchestrator/improve', methods=['POST'])
def improve():
    """
    Improve content.
    
    Request body:
    {
        "content": "Original content...",
        "instruction": "Make it more concise",
        "temperature": 0.7,
        "provider": "opencode",  # Optional
        "model": "default"  # Optional
    }
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Request body required"}), 400
        
        content = data.get('content')
        instruction = data.get('instruction', 'Improve this content')
        
        if not content:
            return jsonify({"error": "content is required"}), 400
        
        jotty_api = _get_jotty_api()
        improver = ContentImprover(jotty_api)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                improver.improve(
                    content=content,
                    instruction=instruction,
                    temperature=data.get('temperature', 0.7),
                    provider=data.get('provider'),
                    model=data.get('model')
                )
            )
            
            return jsonify({
                "success": result.get('success', True),
                "content": result.get('content', ''),
                "metadata": result.get('metadata', {})
            })
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Improve error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@orchestrator_bp.route('/api/orchestrator/summarize', methods=['POST'])
def summarize():
    """
    Summarize content.
    
    Request body:
    {
        "content": "Long content...",
        "maxLength": 200,  # Optional
        "focus": "key points",  # Optional
        "provider": "opencode",  # Optional
        "model": "default"  # Optional
    }
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Request body required"}), 400
        
        content = data.get('content')
        if not content:
            return jsonify({"error": "content is required"}), 400
        
        jotty_api = _get_jotty_api()
        summarizer = ContentSummarizer(jotty_api)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                summarizer.summarize(
                    content=content,
                    max_length=data.get('maxLength'),
                    focus=data.get('focus'),
                    provider=data.get('provider'),
                    model=data.get('model')
                )
            )
            
            return jsonify({
                "success": result.get('success', True),
                "summary": result.get('summary', ''),
                "metadata": result.get('metadata', {})
            })
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Summarize error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@orchestrator_bp.route('/api/orchestrator/preprocess', methods=['POST'])
def preprocess():
    """
    Preprocess request (sanitization, normalization, intent detection).
    
    Request body:
    {
        "prompt": "User input...",
        "mode": "chat",  # Optional, for mode suggestion
        "context": {}  # Optional
    }
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Request body required"}), 400
        
        prompt = data.get('prompt', '')
        mode = data.get('mode')
        context = data.get('context', {})
        
        # Simple preprocessing (can be enhanced)
        # For now, return sanitized prompt and basic analysis
        sanitized = prompt.strip()
        
        # Basic intent detection
        intent = 'chat'
        if any(word in sanitized.lower() for word in ['generate', 'create', 'write']):
            intent = 'generate'
        elif any(word in sanitized.lower() for word in ['improve', 'enhance', 'better']):
            intent = 'improve'
        elif any(word in sanitized.lower() for word in ['summarize', 'summary', 'brief']):
            intent = 'summarize'
        
        return jsonify({
            "success": True,
            "sanitized": sanitized,
            "intent": intent,
            "suggestedMode": intent if not mode else mode,
            "metadata": {
                "length": len(sanitized),
                "wordCount": len(sanitized.split())
            }
        })
        
    except Exception as e:
        logger.error(f"Preprocess error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
