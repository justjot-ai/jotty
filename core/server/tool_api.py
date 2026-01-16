"""
Tool API for JustJot.ai Integration
====================================

Exposes tool registration and execution via HTTP API.
Allows JustJot.ai to register its tools with Jotty and execute them.

Endpoints:
- POST /api/tools/register - Register external tools
- POST /api/tools/execute - Execute a tool
- GET /api/tools/list - List all registered tools
- GET /api/tools/{toolName} - Get tool schema
"""

from flask import Blueprint, request, jsonify
from typing import Dict, Any, Optional, List
import logging
import json
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    try:
        import urllib.request
        import urllib.parse
        REQUESTS_AVAILABLE = False
        URLLIB_AVAILABLE = True
    except ImportError:
        REQUESTS_AVAILABLE = False
        URLLIB_AVAILABLE = False

logger = logging.getLogger(__name__)

tool_bp = Blueprint('tool', __name__)

# External tool registry (tools registered by JustJot.ai)
_external_tools: Dict[str, Dict[str, Any]] = {}


def _get_jotty_api():
    """Get Jotty API instance from Flask app context."""
    from flask import current_app
    if hasattr(current_app, 'jotty_api'):
        return current_app.jotty_api
    
    if hasattr(request, 'jotty_api'):
        return request.jotty_api
    
    raise RuntimeError("Jotty API not available in Flask app context")


@tool_bp.route('/api/tools/register', methods=['POST'])
def register_tools():
    """
    Register external tools (from JustJot.ai).
    
    Request body:
    {
        "tools": [
            {
                "name": "getIdea",
                "description": "Get a single idea by ID",
                "category": "idea",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"}
                    },
                    "required": ["id"]
                },
                "executionUrl": "http://justjot-ai-blue:3000/api/tools/execute",
                "executionMethod": "POST"
            }
        ]
    }
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Request body required"}), 400
        
        tools = data.get('tools', [])
        if not tools:
            return jsonify({"error": "tools array is required"}), 400
        
        registered = []
        for tool in tools:
            tool_name = tool.get('name')
            if not tool_name:
                continue
            
            _external_tools[tool_name] = {
                'name': tool_name,
                'description': tool.get('description', ''),
                'category': tool.get('category', 'utility'),
                'parameters': tool.get('parameters', {}),
                'returns': tool.get('returns'),
                'executionUrl': tool.get('executionUrl'),
                'executionMethod': tool.get('executionMethod', 'POST'),
            }
            registered.append(tool_name)
            logger.info(f"✅ Registered external tool: {tool_name}")
        
        return jsonify({
            "success": True,
            "registered": registered,
            "total": len(_external_tools)
        })
        
    except Exception as e:
        logger.error(f"Tool registration error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@tool_bp.route('/api/tools/execute', methods=['POST'])
def execute_tool():
    """
    Execute a tool.
    
    Request body:
    {
        "toolName": "getIdea",
        "arguments": {
            "id": "idea_123"
        },
        "userId": "user_123"  # Optional, for auth
    }
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Request body required"}), 400
        
        tool_name = data.get('toolName')
        arguments = data.get('arguments', {})
        user_id = data.get('userId')
        
        if not tool_name:
            return jsonify({"error": "toolName is required"}), 400
        
        # Check if tool is registered
        if tool_name not in _external_tools:
            # Check Jotty's internal tool registry
            jotty_api = _get_jotty_api()
            if hasattr(jotty_api, 'tools_registry'):
                tool = jotty_api.tools_registry.get(tool_name)
                if tool:
                    # Execute internal tool
                    if tool_name in jotty_api.tools_registry._implementations:
                        implementation = jotty_api.tools_registry._implementations[tool_name]
                        result = implementation(**arguments)
                        return jsonify({
                            "success": True,
                            "result": result
                        })
            
            return jsonify({"error": f"Tool not found: {tool_name}"}), 404
        
        # Execute external tool via HTTP callback
        tool_info = _external_tools[tool_name]
        execution_url = tool_info.get('executionUrl')
        
        if not execution_url:
            return jsonify({"error": f"Tool {tool_name} has no execution URL"}), 500
        
        # Call external tool execution endpoint
        if not REQUESTS_AVAILABLE and not URLLIB_AVAILABLE:
            return jsonify({
                "success": False,
                "error": "No HTTP library available for external tool execution (install requests or use urllib)"
            }), 500
        
        try:
            if REQUESTS_AVAILABLE:
                response = requests.post(
                    execution_url,
                    json={
                        "toolName": tool_name,
                        "arguments": arguments,
                        "userId": user_id,
                    },
                    headers={
                        "Content-Type": "application/json",
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                result_data = response.json()
            else:
                # Fallback to urllib
                import urllib.request
                import urllib.parse
                import json as json_lib
                
                data = json_lib.dumps({
                    "toolName": tool_name,
                    "arguments": arguments,
                    "userId": user_id,
                }).encode('utf-8')
                
                req = urllib.request.Request(
                    execution_url,
                    data=data,
                    headers={'Content-Type': 'application/json'}
                )
                
                with urllib.request.urlopen(req, timeout=30.0) as response:
                    result_data = json_lib.loads(response.read().decode('utf-8'))
            
            return jsonify({
                "success": True,
                "result": result_data.get('result') if isinstance(result_data, dict) else result_data
            })
                
        except Exception as e:
            logger.error(f"External tool execution error: {e}", exc_info=True)
            return jsonify({
                "success": False,
                "error": f"Tool execution failed: {str(e)}"
            }), 500
        
    except Exception as e:
        logger.error(f"Tool execution error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@tool_bp.route('/api/tools/list', methods=['GET'])
def list_tools():
    """
    List all registered tools (external + internal).
    
    Query params:
    - category: Filter by category (optional)
    """
    try:
        category = request.args.get('category')
        
        tools = []
        
        # Add external tools
        for tool_name, tool_info in _external_tools.items():
            if category and tool_info.get('category') != category:
                continue
            tools.append({
                'name': tool_name,
                'description': tool_info.get('description', ''),
                'category': tool_info.get('category', 'utility'),
                'parameters': tool_info.get('parameters', {}),
                'returns': tool_info.get('returns'),
                'source': 'external',
            })
        
        # Add internal tools from Jotty
        jotty_api = _get_jotty_api()
        if hasattr(jotty_api, 'tools_registry'):
            for tool in jotty_api.tools_registry.get_all():
                if category and tool.category != category:
                    continue
                tools.append({
                    'name': tool.name,
                    'description': tool.description,
                    'category': tool.category,
                    'parameters': tool.parameters,
                    'returns': tool.returns,
                    'source': 'internal',
                    'mcp_enabled': tool.mcp_enabled,
                })
        
        return jsonify({
            "success": True,
            "tools": tools,
            "count": len(tools)
        })
        
    except Exception as e:
        logger.error(f"List tools error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@tool_bp.route('/api/tools/<tool_name>', methods=['GET'])
def get_tool(tool_name: str):
    """
    Get tool schema by name.
    """
    try:
        # Check external tools first
        if tool_name in _external_tools:
            tool_info = _external_tools[tool_name]
            return jsonify({
                "success": True,
                "tool": {
                    'name': tool_name,
                    'description': tool_info.get('description', ''),
                    'category': tool_info.get('category', 'utility'),
                    'parameters': tool_info.get('parameters', {}),
                    'returns': tool_info.get('returns'),
                    'source': 'external',
                }
            })
        
        # Check internal tools
        jotty_api = _get_jotty_api()
        if hasattr(jotty_api, 'tools_registry'):
            tool = jotty_api.tools_registry.get(tool_name)
            if tool:
                return jsonify({
                    "success": True,
                    "tool": tool.to_dict()
                })
        
        return jsonify({"error": f"Tool not found: {tool_name}"}), 404
        
    except Exception as e:
        logger.error(f"Get tool error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
