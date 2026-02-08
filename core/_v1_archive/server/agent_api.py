"""
Agent Execution API for JustJot.ai Integration
===============================================

Exposes agent execution endpoints that leverage Jotty's existing chat infrastructure.
Reuses ChatUseCase and ChatExecutor - no duplication.

Endpoints:
- POST /api/agents/execute - Execute agent synchronously
- POST /api/agents/stream - Execute agent with streaming
- GET /api/agents/list - List available agents (from MongoDB)
- GET /api/agents/{agentId} - Get agent details
"""

from flask import Blueprint, request, jsonify, Response
from typing import Dict, Any, Optional, List
import logging
import json
import asyncio

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

from ..use_cases.chat import ChatUseCase, ChatMessage
from ..foundation.unified_lm_provider import UnifiedLMProvider

logger = logging.getLogger(__name__)

agent_bp = Blueprint('agent', __name__)


def _get_jotty_api():
    """Get Jotty API instance from Flask app context."""
    from flask import current_app
    if hasattr(current_app, 'jotty_api'):
        return current_app.jotty_api
    
    # Fallback: try to get from request context
    if hasattr(request, 'jotty_api'):
        return request.jotty_api
    
    raise RuntimeError("Jotty API not available in Flask app context")


def _load_agent_from_mongodb(agent_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Load agent from MongoDB (shared with JustJot.ai).
    
    Reuses existing MongoDB connection from Jotty.
    """
    try:
        from pymongo import MongoClient
        import os
        
        # Get MongoDB connection from environment (same as JustJot.ai)
        mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/justjot')
        client = MongoClient(mongo_uri)
        db = client.get_database()
        
        # Check static agents first (in-memory registry)
        # Then check MongoDB custom agents
        if agent_id.startswith('custom-'):
            mongo_id = agent_id.replace('custom-', '')
            query = {'_id': mongo_id, 'isActive': True}
            if user_id:
                query['userId'] = user_id
            
            agent_doc = db.customagents.find_one(query)
            if agent_doc:
                # Convert MongoDB document to agent dict
                return {
                    'id': agent_id,
                    'name': agent_doc.get('name', 'Unknown'),
                    'description': agent_doc.get('description', ''),
                    'category': agent_doc.get('category', 'utility'),
                    'tools': agent_doc.get('tools', []),
                    'useDSPy': agent_doc.get('useDSPy', False),
                    'dspyModule': agent_doc.get('dspyModule'),
                }
        else:
            # Static agent - check if exists in Jotty's agent list
            # For now, return basic info (can be enhanced)
            return {
                'id': agent_id,
                'name': agent_id,
                'description': f'Agent {agent_id}',
                'category': 'core',
                'tools': [],
                'useDSPy': False,
            }
        
        return None
    except Exception as e:
        logger.error(f"Failed to load agent from MongoDB: {e}", exc_info=True)
        return None


@agent_bp.route('/api/agents/execute', methods=['POST'])
def execute_agent():
    """
    Execute agent synchronously.
    
    Request body:
    {
        "agentId": "research-assistant",
        "task": "Research AI trends",
        "userId": "user_123",
        "provider": "opencode",  # Optional
        "model": "default",  # Optional
        "context": {
            "ideaId": "idea_456",
            "conversationHistory": [...]
        }
    }
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Request body required"}), 400
        
        agent_id = data.get('agentId')
        task = data.get('task') or data.get('message') or data.get('prompt')
        user_id = data.get('userId')
        provider = data.get('provider')
        model = data.get('model')
        context = data.get('context', {})
        
        if not agent_id:
            return jsonify({"error": "agentId is required"}), 400
        if not task:
            return jsonify({"error": "task/message/prompt is required"}), 400
        
        # Configure LM provider if specified
        if provider:
            try:
                lm = UnifiedLMProvider.create_lm(provider, model=model)
                dspy.configure(lm=lm)
                logger.info(f"🔵 AGENT: Configured LM provider: {provider}, model: {model}")
            except Exception as e:
                logger.warning(f"⚠️  AGENT: Failed to configure provider {provider}: {e}")
        
        # Load agent info (for metadata)
        # Try MongoDB first, then check Jotty's static agents
        agent_info = _load_agent_from_mongodb(agent_id, user_id)
        if not agent_info:
            # Check if agent exists in Jotty's static registry
            jotty_api = _get_jotty_api()
            if hasattr(jotty_api, 'agents') and jotty_api.agents:
                for agent_config in jotty_api.agents:
                    if agent_config.name == agent_id:
                        agent_info = {
                            'id': agent_config.name,
                            'name': agent_config.name,
                            'description': getattr(agent_config, 'description', ''),
                            'category': 'core',
                            'tools': [],
                        }
                        break
            
            if not agent_info:
                return jsonify({"error": f"Agent not found: {agent_id}"}), 404
        
        # Get Jotty API (ChatUseCase)
        jotty_api = _get_jotty_api()
        
        # Build conversation history
        conversation_history = []
        if context.get('conversationHistory'):
            for msg in context['conversationHistory']:
                conversation_history.append(ChatMessage(
                    role=msg.get('role', 'user'),
                    content=msg.get('content', ''),
                    timestamp=msg.get('timestamp')
                ))
        
        # Execute agent using JottyAPI's chat_execute method
        # This reuses Jotty's existing chat infrastructure - NO duplication
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Use JottyAPI's chat_execute (reuses existing infrastructure)
            result = loop.run_until_complete(
                jotty_api.chat_execute(
                    message=task,
                    history=conversation_history,
                    agent_id=agent_id,
                    **context
                )
            )
            
            return jsonify({
                "success": result.get('success', True),
                "output": result.get('message', result.get('output', '')),
                "agentId": agent_id,
                "agentName": agent_info.get('name', agent_id),
                "metadata": result.get('metadata', {}),
                "executionTime": result.get('execution_time', 0)
            })
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Agent execution error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e) if hasattr(request, 'show_error_details') and request.show_error_details else "Agent execution failed"
        }), 500


@agent_bp.route('/api/agents/stream', methods=['POST'])
def stream_agent():
    """
    Execute agent with streaming (SSE).
    
    Request body: Same as /api/agents/execute
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Request body required"}), 400
        
        agent_id = data.get('agentId')
        task = data.get('task') or data.get('message') or data.get('prompt')
        user_id = data.get('userId')
        provider = data.get('provider')
        model = data.get('model')
        context = data.get('context', {})
        
        if not agent_id:
            return jsonify({"error": "agentId is required"}), 400
        if not task:
            return jsonify({"error": "task/message/prompt is required"}), 400
        
        # Configure LM provider if specified
        if provider:
            try:
                lm = UnifiedLMProvider.create_lm(provider, model=model)
                dspy.configure(lm=lm)
                logger.info(f"🔵 AGENT: Configured LM provider: {provider}, model: {model}")
            except Exception as e:
                logger.warning(f"⚠️  AGENT: Failed to configure provider {provider}: {e}")
        
        # Load agent info
        # Try MongoDB first, then check Jotty's static agents
        agent_info = _load_agent_from_mongodb(agent_id, user_id)
        if not agent_info:
            # Check if agent exists in Jotty's static registry
            jotty_api = _get_jotty_api()
            if hasattr(jotty_api, 'agents') and jotty_api.agents:
                for agent_config in jotty_api.agents:
                    if agent_config.name == agent_id:
                        agent_info = {
                            'id': agent_config.name,
                            'name': agent_config.name,
                            'description': getattr(agent_config, 'description', ''),
                            'category': 'core',
                            'tools': [],
                        }
                        break
            
            if not agent_info:
                return jsonify({"error": f"Agent not found: {agent_id}"}), 404
        
        # Get Jotty API
        jotty_api = _get_jotty_api()
        
        # Build conversation history
        conversation_history = []
        if context.get('conversationHistory'):
            for msg in context['conversationHistory']:
                conversation_history.append(ChatMessage(
                    role=msg.get('role', 'user'),
                    content=msg.get('content', ''),
                    timestamp=msg.get('timestamp')
                ))
        
        # Get SSE formatter from Jotty HTTP server
        def generate_events():
            """Generate SSE events."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Use JottyAPI's chat_stream (reuses existing infrastructure)
                async_gen = jotty_api.chat_stream(
                    message=task,
                    history=conversation_history,
                    agent_id=agent_id,
                    **context
                )
                
                # Get formatter from HTTP server (reuse existing formatter)
                formatter = None
                from flask import current_app
                if hasattr(current_app, 'jotty_server'):
                    jotty_server = current_app.jotty_server
                    if hasattr(jotty_server, '_get_sse_formatter'):
                        formatter = jotty_server._get_sse_formatter()
                
                # Emit agent start event
                start_event = {
                    "type": "agent-start",
                    "data": {
                        "agentId": agent_id,
                        "agentName": agent_info.get('name', agent_id),
                    }
                }
                if formatter:
                    for formatted in formatter.format_event(start_event):
                        yield formatted
                else:
                    yield f"data: {json.dumps(start_event)}\n\n"
                
                # Stream and transform chat events to agent-progress format
                while True:
                    try:
                        event = loop.run_until_complete(async_gen.__anext__())
                        
                        # Transform chat events to agent-progress events
                        transformed = None
                        if event.get('type') == 'text_chunk':
                            transformed = {
                                "type": "agent-progress",
                                "data": {
                                    "agentId": agent_id,
                                    "textDelta": event.get('content', ''),
                                }
                            }
                        elif event.get('type') == 'tool_call':
                            transformed = {
                                "type": "agent-progress",
                                "data": {
                                    "agentId": agent_id,
                                    "toolCall": {
                                        "toolName": event.get('tool'),
                                        "args": event.get('args', {}),
                                    }
                                }
                            }
                        elif event.get('type') == 'done':
                            transformed = {
                                "type": "agent-finish",
                                "data": {
                                    "agentId": agent_id,
                                    "agentName": agent_info.get('name', agent_id),
                                    "success": True,
                                    "output": event.get('message', ''),
                                }
                            }
                        elif event.get('type') == 'error':
                            transformed = {
                                "type": "error",
                                "data": {"error": event.get('error', 'Unknown error')}
                            }
                        else:
                            transformed = event  # Pass through
                        
                        if transformed:
                            if formatter:
                                for formatted in formatter.format_event(transformed):
                                    yield formatted
                            else:
                                yield f"data: {json.dumps(transformed)}\n\n"
                                
                    except StopAsyncIteration:
                        break
                
                # End marker (reuse existing formatter)
                if formatter:
                    yield formatter.end_marker()
                else:
                    yield "data: [DONE]\n\n"
                
            except Exception as e:
                logger.error(f"Agent streaming error: {e}", exc_info=True)
                error_event = {
                    "type": "error",
                    "data": {"error": str(e)}
                }
                if formatter:
                    yield formatter.error_event(str(e))
                else:
                    yield f"data: {json.dumps(error_event)}\n\n"
            finally:
                loop.close()
        
        return Response(
            generate_events(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )
        
    except Exception as e:
        logger.error(f"Agent stream error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@agent_bp.route('/api/agents/list', methods=['GET'])
def list_agents():
    """
    List all available agents (from MongoDB + static registry).
    
    Query params:
    - userId: Filter by user ID (optional)
    - category: Filter by category (optional)
    """
    try:
        user_id = request.args.get('userId')
        category = request.args.get('category')
        
        agents = []
        
        # Load from MongoDB (custom agents)
        try:
            from pymongo import MongoClient
            import os
            
            mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/justjot')
            client = MongoClient(mongo_uri)
            db = client.get_database()
            
            query = {'isActive': True}
            if user_id:
                query['userId'] = user_id
            if category:
                query['category'] = category
            
            for agent_doc in db.customagents.find(query):
                agents.append({
                    'id': f"custom-{agent_doc['_id']}",
                    'name': agent_doc.get('name', 'Unknown'),
                    'description': agent_doc.get('description', ''),
                    'category': agent_doc.get('category', 'utility'),
                    'tools': agent_doc.get('tools', []),
                })
        except Exception as e:
            logger.warning(f"Failed to load agents from MongoDB: {e}")
        
        # Add static agents from Jotty
        jotty_api = _get_jotty_api()
        if hasattr(jotty_api, 'agents') and jotty_api.agents:
            for agent_config in jotty_api.agents:
                agents.append({
                    'id': agent_config.name,
                    'name': agent_config.name,
                    'description': getattr(agent_config, 'description', ''),
                    'category': 'core',
                    'tools': [],
                })
        
        return jsonify({
            "success": True,
            "agents": agents,
            "count": len(agents)
        })
        
    except Exception as e:
        logger.error(f"List agents error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@agent_bp.route('/api/agents/<agent_id>', methods=['GET'])
def get_agent(agent_id: str):
    """
    Get agent details by ID.
    """
    try:
        user_id = request.args.get('userId')
        
        # Try MongoDB first, then check Jotty's static agents
        agent_info = _load_agent_from_mongodb(agent_id, user_id)
        if not agent_info:
            # Check if agent exists in Jotty's static registry
            jotty_api = _get_jotty_api()
            if hasattr(jotty_api, 'agents') and jotty_api.agents:
                for agent_config in jotty_api.agents:
                    if agent_config.name == agent_id:
                        agent_info = {
                            'id': agent_config.name,
                            'name': agent_config.name,
                            'description': getattr(agent_config, 'description', ''),
                            'category': 'core',
                            'tools': [],
                        }
                        break
            
            if not agent_info:
                return jsonify({"error": f"Agent not found: {agent_id}"}), 404
        
        return jsonify({
            "success": True,
            "agent": agent_info
        })
        
    except Exception as e:
        logger.error(f"Get agent error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
