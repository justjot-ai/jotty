"""
Swarm API for JustJot.ai Integration
====================================

Exposes swarm orchestration operations via HTTP API.
Reuses Jotty's existing SwarmManager infrastructure - no duplication.

Endpoints:
- POST /api/swarm/execute - Execute swarm (auto or manual mode)
- POST /api/swarm/stream - Execute swarm with streaming
- GET /api/swarm/{swarmId} - Get swarm execution status
"""

from flask import Blueprint, request, jsonify, Response
from typing import Dict, Any, Optional, List
import logging
import json
import asyncio
import uuid

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

from ..foundation.unified_lm_provider import UnifiedLMProvider
from ..orchestration import SwarmManager
from ..foundation.data_structures import JottyConfig
from ..foundation.agent_config import AgentConfig

logger = logging.getLogger(__name__)

swarm_bp = Blueprint('swarm', __name__)


def _get_jotty_api():
    """Get Jotty API instance from Flask app context."""
    from flask import current_app
    if hasattr(current_app, 'jotty_api'):
        return current_app.jotty_api
    
    if hasattr(request, 'jotty_api'):
        return request.jotty_api
    
    raise RuntimeError("Jotty API not available in Flask app context")


def _load_agents_from_mongodb(agent_ids: List[str], user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load agents from MongoDB (shared with JustJot.ai)."""
    agents = []
    try:
        from pymongo import MongoClient
        import os
        
        mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/justjot')
        client = MongoClient(mongo_uri)
        db = client.get_database()
        
        for agent_id in agent_ids:
            if agent_id.startswith('custom-'):
                mongo_id = agent_id.replace('custom-', '')
                query = {'_id': mongo_id, 'isActive': True}
                if user_id:
                    query['userId'] = user_id
                
                agent_doc = db.customagents.find_one(query)
                if agent_doc:
                    agents.append({
                        'id': agent_id,
                        'name': agent_doc.get('name', 'Unknown'),
                        'description': agent_doc.get('description', ''),
                    })
            else:
                # Static agent - check Jotty's agent list
                jotty_api = _get_jotty_api()
                if hasattr(jotty_api, 'agents') and jotty_api.agents:
                    for agent_config in jotty_api.agents:
                        if agent_config.name == agent_id:
                            agents.append({
                                'id': agent_id,
                                'name': agent_config.name,
                                'description': getattr(agent_config, 'description', ''),
                            })
                            break
    except Exception as e:
        logger.error(f"Failed to load agents from MongoDB: {e}", exc_info=True)
    
    return agents


@swarm_bp.route('/api/swarm/execute', methods=['POST'])
def execute_swarm():
    """
    Execute swarm synchronously.
    
    Request body:
    {
        "task": "Create comprehensive documentation",
        "mode": "auto",  # "auto" or "manual"
        "agents": ["research-assistant", "writer"],  # Required for manual mode
        "userId": "user_123",
        "provider": "opencode",  # Optional
        "model": "default",  # Optional
        "context": {
            "ideaId": "idea_456",
            "maxIterations": 5,
            "parallel": true
        }
    }
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Request body required"}), 400
        
        task = data.get('task') or data.get('prompt')
        mode = data.get('mode', 'auto')
        agent_ids = data.get('agents', [])
        user_id = data.get('userId')
        provider = data.get('provider')
        model = data.get('model')
        context = data.get('context', {})
        max_iterations = data.get('maxIterations') or context.get('maxIterations', 3)
        parallel = data.get('parallel') or context.get('parallel', False)
        
        if not task:
            return jsonify({"error": "task/prompt is required"}), 400
        
        if mode == 'manual' and not agent_ids:
            return jsonify({"error": "agents array is required for manual mode"}), 400
        
        # Configure LM provider if specified
        if provider:
            try:
                lm = UnifiedLMProvider.create_lm(provider, model=model)
                dspy.configure(lm=lm)
                logger.info(f"🔵 SWARM: Configured LM provider: {provider}, model: {model}")
            except Exception as e:
                logger.warning(f"⚠️  SWARM: Failed to configure provider {provider}: {e}")
        
        # Get Jotty API
        jotty_api = _get_jotty_api()
        
        # Load agents
        if mode == 'manual':
            agents_info = _load_agents_from_mongodb(agent_ids, user_id)
            if len(agents_info) != len(agent_ids):
                return jsonify({"error": "Some agents not found"}), 404
        else:
            # Auto mode - use all available agents from Jotty
            agents_info = []
            if hasattr(jotty_api, 'agents') and jotty_api.agents:
                agents_info = [
                    {'id': agent_config.name, 'name': agent_config.name}
                    for agent_config in jotty_api.agents
                ]
        
        # Create swarm ID
        swarm_id = str(uuid.uuid4())
        
        # Execute swarm using Jotty's SwarmManager (reuses existing infrastructure)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Use Jotty's SwarmManager for swarm execution
            conductor = jotty_api.conductor if hasattr(jotty_api, 'conductor') else None
            if not conductor:
                return jsonify({"error": "SwarmManager not available"}), 500
            
            # Run swarm using SwarmManager
            # SwarmManager already handles multi-actor orchestration
            result = loop.run_until_complete(
                conductor.run(
                    goal=task,
                    max_iterations=max_iterations,
                    parallel=parallel,
                    **context
                )
            )
            
            # Transform SwarmManager result to swarm result format
            return jsonify({
                "success": True,
                "swarmId": swarm_id,
                "results": [
                    {
                        "agentId": agent_info['id'],
                        "agentName": agent_info['name'],
                        "success": True,
                        "output": str(result) if result else "",
                    }
                    for agent_info in agents_info
                ],
                "aggregatedOutput": str(result) if result else "",
                "totalDuration": 0,  # SwarmManager doesn't track this separately
                "metadata": {
                    "mode": mode,
                    "agents": [a['id'] for a in agents_info],
                }
            })
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Swarm execution error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@swarm_bp.route('/api/swarm/stream', methods=['POST'])
def stream_swarm():
    """
    Execute swarm with streaming (SSE).
    
    Request body: Same as /api/swarm/execute
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Request body required"}), 400
        
        task = data.get('task') or data.get('prompt')
        mode = data.get('mode', 'auto')
        agent_ids = data.get('agents', [])
        user_id = data.get('userId')
        provider = data.get('provider')
        model = data.get('model')
        context = data.get('context', {})
        max_iterations = data.get('maxIterations') or context.get('maxIterations', 3)
        parallel = data.get('parallel') or context.get('parallel', False)
        
        if not task:
            return jsonify({"error": "task/prompt is required"}), 400
        
        if mode == 'manual' and not agent_ids:
            return jsonify({"error": "agents array is required for manual mode"}), 400
        
        # Configure LM provider if specified
        if provider:
            try:
                lm = UnifiedLMProvider.create_lm(provider, model=model)
                dspy.configure(lm=lm)
                logger.info(f"🔵 SWARM: Configured LM provider: {provider}, model: {model}")
            except Exception as e:
                logger.warning(f"⚠️  SWARM: Failed to configure provider {provider}: {e}")
        
        # Get Jotty API
        jotty_api = _get_jotty_api()
        
        # Load agents
        if mode == 'manual':
            agents_info = _load_agents_from_mongodb(agent_ids, user_id)
            if len(agents_info) != len(agent_ids):
                return jsonify({"error": "Some agents not found"}), 404
        else:
            # Auto mode - use all available agents from Jotty
            agents_info = []
            if hasattr(jotty_api, 'agents') and jotty_api.agents:
                agents_info = [
                    {'id': agent_config.name, 'name': agent_config.name}
                    for agent_config in jotty_api.agents
                ]
        
        # Create swarm ID
        swarm_id = str(uuid.uuid4())
        
        # Get formatter from HTTP server
        formatter = None
        from flask import current_app
        if hasattr(current_app, 'jotty_server'):
            jotty_server = current_app.jotty_server
            if hasattr(jotty_server, '_get_sse_formatter'):
                formatter = jotty_server._get_sse_formatter()
        
        def generate_events():
            """Generate SSE events."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Use Jotty's SwarmManager for swarm execution (reuses existing infrastructure)
                conductor = jotty_api.conductor if hasattr(jotty_api, 'conductor') else None
                if not conductor:
                    error_event = {
                        "type": "error",
                        "data": {"error": "SwarmManager not available"}
                    }
                    if formatter:
                        yield formatter.error_event("SwarmManager not available")
                    else:
                        yield f"data: {json.dumps(error_event)}\n\n"
                    return
                
                # Emit swarm start event
                start_event = {
                    "type": "swarm-start",
                    "data": {
                        "swarmId": swarm_id,
                        "mode": mode,
                        "agents": [a['id'] for a in agents_info],
                        "totalAgents": len(agents_info),
                    }
                }
                if formatter:
                    for formatted in formatter.format_event(start_event):
                        yield formatted
                else:
                    yield f"data: {json.dumps(start_event)}\n\n"
                
                # Run swarm using SwarmManager
                async def run_swarm_execution():
                    result = await conductor.run(
                        goal=task,
                        max_iterations=max_iterations,
                        parallel=parallel,
                        **context
                    )
                    return result
                
                result = loop.run_until_complete(run_swarm_execution())
                
                # Emit agent progress events
                for i, agent_info in enumerate(agents_info):
                    progress_event = {
                        "type": "agent-progress",
                        "data": {
                            "swarmId": swarm_id,
                            "agentId": agent_info['id'],
                            "agentName": agent_info['name'],
                            "agentIndex": i,
                            "totalAgents": len(agents_info),
                            "textDelta": str(result) if result else "",
                        }
                    }
                    if formatter:
                        for formatted in formatter.format_event(progress_event):
                            yield formatted
                    else:
                        yield f"data: {json.dumps(progress_event)}\n\n"
                
                # Emit swarm finish event
                finish_event = {
                    "type": "swarm-finish",
                    "data": {
                        "swarmId": swarm_id,
                        "success": True,
                        "aggregatedOutput": str(result) if result else "",
                        "results": [
                            {
                                "agentId": agent_info['id'],
                                "agentName": agent_info['name'],
                                "success": True,
                                "output": str(result) if result else "",
                            }
                            for agent_info in agents_info
                        ],
                    }
                }
                if formatter:
                    for formatted in formatter.format_event(finish_event):
                        yield formatted
                else:
                    yield f"data: {json.dumps(finish_event)}\n\n"
                
                # End marker
                if formatter:
                    yield formatter.end_marker()
                else:
                    yield "data: [DONE]\n\n"
                
            except Exception as e:
                logger.error(f"Swarm streaming error: {e}", exc_info=True)
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
        logger.error(f"Swarm stream error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@swarm_bp.route('/api/swarm/<swarm_id>', methods=['GET'])
def get_swarm(swarm_id: str):
    """
    Get swarm execution status.
    
    Note: This is a placeholder - full implementation would require
    swarm state persistence in MongoDB or Jotty's persistence layer.
    """
    try:
        # TODO: Load swarm state from persistence
        return jsonify({
            "success": True,
            "swarmId": swarm_id,
            "status": "unknown",
            "message": "Swarm status retrieval not yet implemented"
        })
    except Exception as e:
        logger.error(f"Get swarm error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
