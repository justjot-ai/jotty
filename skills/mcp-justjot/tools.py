"""
MCP JustJot.ai Skill

Wraps JustJot.ai MCP tools as Jotty skills.
Integrates ideas, templates, sections, and tags management.
"""
import asyncio
import logging
import os
from typing import Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)

# Default JustJot.ai API URL
# Priority: JUSTJOT_API_URL > NEXT_PUBLIC_API_URL > JUSTJOT_BASE_URL > cmd.dev > localhost
def _get_api_url() -> str:
    """Get JustJot.ai API URL with fallback priority."""
    # Check environment variables in priority order
    url = (
        os.getenv("JUSTJOT_API_URL") or 
        os.getenv("NEXT_PUBLIC_API_URL") or 
        os.getenv("JUSTJOT_BASE_URL")
    )
    
    if url:
        return url
    
    # Fallback to cmd.dev (production deployment)
    return "https://justjot.ai.cmd.dev"

DEFAULT_API_URL = _get_api_url()


def _call_justjot_api(
    method: str,
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Call JustJot.ai API endpoint.
    
    Args:
        method: HTTP method (GET, POST, PUT, DELETE)
        endpoint: API endpoint path
        params: Query parameters
        data: Request body data
        headers: Request headers
    
    Returns:
        API response as dictionary
    """
    url = f"{DEFAULT_API_URL}{endpoint}"
    
    default_headers = {
        "Content-Type": "application/json"
    }
    
    # Add auth token if available
    auth_token = os.getenv("JUSTJOT_AUTH_TOKEN")
    if auth_token:
        default_headers["Authorization"] = f"Bearer {auth_token}"
    
    if headers:
        default_headers.update(headers)
    
    try:
        response = requests.request(
            method=method,
            url=url,
            params=params,
            json=data,
            headers=default_headers,
            timeout=30
        )
        
        response.raise_for_status()
        
        # Try to parse JSON
        try:
            return {
                'success': True,
                'data': response.json()
            }
        except ValueError:
            return {
                'success': True,
                'data': response.text
            }
    
    except requests.exceptions.RequestException as e:
        logger.error(f"JustJot API call failed: {e}")
        return {
            'success': False,
            'error': f'API call failed: {str(e)}'
        }


# ============================================
# Ideas Management Tools
# ============================================

async def list_ideas_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List all ideas.
    
    Args:
        params: Dictionary containing:
            - full (bool, optional): Include full content (default: False)
            - search (str, optional): Search query
    
    Returns:
        Dictionary with:
            - success (bool): Whether operation succeeded
            - ideas (list): List of ideas
            - error (str, optional): Error message if failed
    """
    query_params = {}
    if params.get('full'):
        query_params['full'] = 'true'
    
    result = _call_justjot_api('GET', '/api/ideas', params=query_params)
    
    if result.get('success'):
        ideas = result.get('data', [])
        return {
            'success': True,
            'ideas': ideas,
            'count': len(ideas)
        }
    else:
        return result


async def create_idea_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new idea.
    
    Args:
        params: Dictionary containing:
            - title (str, required): Idea title
            - description (str, optional): Idea description
            - templateName (str, optional): Template name
            - tags (list, optional): List of tags
            - status (str, optional): Idea status
    
    Returns:
        Dictionary with:
            - success (bool): Whether operation succeeded
            - idea (dict): Created idea
            - error (str, optional): Error message if failed
    """
    title = params.get('title')
    if not title:
        return {
            'success': False,
            'error': 'title parameter is required'
        }
    
    idea_data = {
        'title': title,
        'description': params.get('description', ''),
        'templateName': params.get('templateName', 'default'),
        'tags': params.get('tags', []),
        'status': params.get('status', 'active')
    }
    
    result = _call_justjot_api('POST', '/api/ideas', data=idea_data)
    
    if result.get('success'):
        return {
            'success': True,
            'idea': result.get('data'),
            'id': result.get('data', {}).get('_id') or result.get('data', {}).get('id')
        }
    else:
        return result


async def get_idea_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get idea by ID.
    
    Args:
        params: Dictionary containing:
            - idea_id (str, required): Idea ID
    
    Returns:
        Dictionary with:
            - success (bool): Whether operation succeeded
            - idea (dict): Idea data
            - error (str, optional): Error message if failed
    """
    idea_id = params.get('idea_id') or params.get('id')
    if not idea_id:
        return {
            'success': False,
            'error': 'idea_id parameter is required'
        }
    
    result = _call_justjot_api('GET', f'/api/ideas/{idea_id}')
    
    if result.get('success'):
        return {
            'success': True,
            'idea': result.get('data')
        }
    else:
        return result


async def update_idea_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update an existing idea.
    
    Args:
        params: Dictionary containing:
            - idea_id (str, required): Idea ID
            - title (str, optional): New title
            - description (str, optional): New description
            - status (str, optional): New status
            - tags (list, optional): New tags
    
    Returns:
        Dictionary with:
            - success (bool): Whether operation succeeded
            - idea (dict): Updated idea
            - error (str, optional): Error message if failed
    """
    idea_id = params.get('idea_id') or params.get('id')
    if not idea_id:
        return {
            'success': False,
            'error': 'idea_id parameter is required'
        }
    
    update_data = {}
    for key in ['title', 'description', 'status', 'tags']:
        if key in params:
            update_data[key] = params[key]
    
    if not update_data:
        return {
            'success': False,
            'error': 'No fields to update'
        }
    
    result = _call_justjot_api('PUT', f'/api/ideas/{idea_id}', data=update_data)
    
    if result.get('success'):
        return {
            'success': True,
            'idea': result.get('data')
        }
    else:
        return result


async def delete_idea_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Delete an idea.
    
    Args:
        params: Dictionary containing:
            - idea_id (str, required): Idea ID
    
    Returns:
        Dictionary with:
            - success (bool): Whether operation succeeded
            - error (str, optional): Error message if failed
    """
    idea_id = params.get('idea_id') or params.get('id')
    if not idea_id:
        return {
            'success': False,
            'error': 'idea_id parameter is required'
        }
    
    result = _call_justjot_api('DELETE', f'/api/ideas/{idea_id}')
    
    if result.get('success'):
        return {
            'success': True,
            'message': f'Idea {idea_id} deleted'
        }
    else:
        return result


# ============================================
# Templates Management Tools
# ============================================

async def list_templates_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """List all templates."""
    result = _call_justjot_api('GET', '/api/templates')
    
    if result.get('success'):
        templates = result.get('data', [])
        return {
            'success': True,
            'templates': templates,
            'count': len(templates)
        }
    else:
        return result


async def get_template_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get template by ID or name."""
    template_id = params.get('template_id') or params.get('id') or params.get('name')
    if not template_id:
        return {
            'success': False,
            'error': 'template_id parameter is required'
        }
    
    result = _call_justjot_api('GET', f'/api/templates/{template_id}')
    
    if result.get('success'):
        return {
            'success': True,
            'template': result.get('data')
        }
    else:
        return result


# ============================================
# Sections Management Tools
# ============================================

async def add_section_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add section to idea.
    
    Args:
        params: Dictionary containing:
            - idea_id (str, required): Idea ID
            - title (str, required): Section title
            - type (str, required): Section type
            - content (str, optional): Section content
    """
    idea_id = params.get('idea_id') or params.get('id')
    if not idea_id:
        return {
            'success': False,
            'error': 'idea_id parameter is required'
        }
    
    section_data = {
        'title': params.get('title', 'New Section'),
        'type': params.get('type', 'text'),
        'content': params.get('content', '')
    }
    
    result = _call_justjot_api('POST', f'/api/ideas/{idea_id}/sections', data=section_data)
    
    if result.get('success'):
        return {
            'success': True,
            'section': result.get('data')
        }
    else:
        return result


async def update_section_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Update section in idea."""
    idea_id = params.get('idea_id') or params.get('id')
    section_index = params.get('section_index') or params.get('index')
    
    if not idea_id or section_index is None:
        return {
            'success': False,
            'error': 'idea_id and section_index parameters are required'
        }
    
    update_data = {}
    for key in ['title', 'type', 'content', 'completed']:
        if key in params:
            update_data[key] = params[key]
    
    result = _call_justjot_api(
        'PUT',
        f'/api/ideas/{idea_id}/sections/{section_index}',
        data=update_data
    )
    
    if result.get('success'):
        return {
            'success': True,
            'section': result.get('data')
        }
    else:
        return result


# ============================================
# Tags Management Tools
# ============================================

async def list_tags_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """List all tags."""
    result = _call_justjot_api('GET', '/api/tags')
    
    if result.get('success'):
        tags = result.get('data', [])
        return {
            'success': True,
            'tags': tags,
            'count': len(tags)
        }
    else:
        return result


__all__ = [
    'list_ideas_tool',
    'create_idea_tool',
    'get_idea_tool',
    'update_idea_tool',
    'delete_idea_tool',
    'list_templates_tool',
    'get_template_tool',
    'add_section_tool',
    'update_section_tool',
    'list_tags_tool'
]
