"""
MCP JustJot.ai Skill (MCP Client Version)

Uses MCP protocol client instead of HTTP API.
Communicates via stdio transport like Claude Desktop.
"""
import asyncio
import logging
import os
import json
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


async def _call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call JustJot.ai MCP tool using MCP client.
    
    Args:
        tool_name: MCP tool name (e.g., "create_idea")
        arguments: Tool arguments
    
    Returns:
        Tool execution result
    """
    try:
        from core.integration.mcp_client import call_justjot_mcp_tool
        
        # Prepare environment
        env = {}
        if os.getenv("MONGODB_URI"):
            env["MONGODB_URI"] = os.getenv("MONGODB_URI")
        if os.getenv("CLERK_SECRET_KEY"):
            env["CLERK_SECRET_KEY"] = os.getenv("CLERK_SECRET_KEY")
        
        # Get MongoDB URI
        mongodb_uri = os.getenv("MONGODB_URI")
        
        # Call MCP tool
        result = await call_justjot_mcp_tool(
            tool_name=tool_name,
            arguments=arguments,
            env=env,
            mongodb_uri=mongodb_uri
        )
        
        return {
            'success': True,
            'data': result
        }
    except Exception as e:
        logger.error(f"MCP tool call error: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'MCP tool call failed: {str(e)}'
        }


# ============================================
# Ideas Management Tools (MCP Client Version)
# ============================================

async def list_ideas_mcp_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """List all ideas using MCP client."""
    result = await _call_mcp_tool("list_ideas", params)
    
    if result.get('success'):
        data = result.get('data', {})
        ideas = data.get('content', []) if isinstance(data, dict) else data
        return {
            'success': True,
            'ideas': ideas,
            'count': len(ideas) if isinstance(ideas, list) else 0
        }
    else:
        return result


async def create_idea_mcp_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new idea using MCP client."""
    title = params.get('title')
    if not title:
        return {
            'success': False,
            'error': 'title parameter is required'
        }
    
    result = await _call_mcp_tool("create_idea", params)
    
    if result.get('success'):
        data = result.get('data', {})
        
        # MCP returns content as text string, need to parse JSON
        if isinstance(data, str):
            try:
                import json
                data = json.loads(data)
            except:
                pass
        
        # Handle different response formats
        idea = {}
        idea_id = None
        
        if isinstance(data, dict):
            # Check if it's already parsed JSON with id
            if 'id' in data:
                idea_id = data['id']
                idea = data
            elif '_id' in data:
                idea_id = data['_id']
                idea = data
            elif 'content' in data:
                # MCP format with content array
                content = data.get('content', [])
                if isinstance(content, list) and len(content) > 0:
                    text_content = content[0].get('text', '')
                    try:
                        parsed = json.loads(text_content)
                        idea = parsed
                        idea_id = parsed.get('_id') or parsed.get('id')
                    except:
                        # Try to extract ID from text if it's a JSON string
                        if 'id' in text_content or '_id' in text_content:
                            try:
                                # Look for JSON-like structure in text
                                import re
                                id_match = re.search(r'"id"\s*:\s*"([^"]+)"', text_content)
                                if not id_match:
                                    id_match = re.search(r'"_id"\s*:\s*"([^"]+)"', text_content)
                                if id_match:
                                    idea_id = id_match.group(1)
                            except:
                                pass
                        idea = {'id': idea_id, 'text': text_content}
                else:
                    idea = data
            else:
                idea = data
        else:
            idea = {'id': None, 'data': data}
        
        return {
            'success': True,
            'idea': idea,
            'id': idea_id or idea.get('_id') or idea.get('id')
        }
    else:
        return result


async def get_idea_mcp_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get idea by ID using MCP client."""
    idea_id = params.get('idea_id') or params.get('id')
    if not idea_id:
        return {
            'success': False,
            'error': 'idea_id parameter is required'
        }
    
    result = await _call_mcp_tool("get_idea", {'id': idea_id})
    
    if result.get('success'):
        data = result.get('data', {})
        idea = data.get('content', {}) if isinstance(data, dict) else data
        return {
            'success': True,
            'idea': idea
        }
    else:
        return result


async def update_idea_mcp_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Update idea using MCP client."""
    idea_id = params.get('idea_id') or params.get('id')
    if not idea_id:
        return {
            'success': False,
            'error': 'idea_id parameter is required'
        }
    
    update_params = {'id': idea_id}
    for key in ['title', 'description', 'status', 'tags']:
        if key in params:
            update_params[key] = params[key]
    
    result = await _call_mcp_tool("update_idea", update_params)
    
    if result.get('success'):
        data = result.get('data', {})
        idea = data.get('content', {}) if isinstance(data, dict) else data
        return {
            'success': True,
            'idea': idea
        }
    else:
        return result


async def delete_idea_mcp_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Delete idea using MCP client."""
    idea_id = params.get('idea_id') or params.get('id')
    if not idea_id:
        return {
            'success': False,
            'error': 'idea_id parameter is required'
        }
    
    result = await _call_mcp_tool("delete_idea", {'id': idea_id})
    
    if result.get('success'):
        return {
            'success': True,
            'message': f'Idea {idea_id} deleted'
        }
    else:
        return result


__all__ = [
    'list_ideas_mcp_tool',
    'create_idea_mcp_tool',
    'get_idea_mcp_tool',
    'update_idea_mcp_tool',
    'delete_idea_mcp_tool'
]
