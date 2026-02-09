"""
JustJot.ai Skill — Single source of truth for all JustJot.ai integration.

Manages ideas, templates, sections, and tags via JustJot.ai REST API.
Supports multiple auth methods (internal service header, Clerk API key, bearer token).

Replaces the old justjot-mcp-http and mcp-justjot-mcp-client skills (DRY merge).
"""
import logging
import os
from typing import Dict, Any, Optional
import requests

from Jotty.core.utils.skill_status import SkillStatus

logger = logging.getLogger(__name__)

# Status emitter for progress updates
status = SkillStatus("mcp-justjot")


def _get_api_url() -> str:
    """Get JustJot.ai API URL with fallback priority."""
    return (
        os.getenv("JUSTJOT_API_URL")
        or os.getenv("NEXT_PUBLIC_API_URL")
        or os.getenv("JUSTJOT_BASE_URL")
        or "https://justjot.ai"
    )


DEFAULT_API_URL = _get_api_url()


def _get_auth_headers() -> Dict[str, str]:
    """
    Build auth headers with priority:
      1. Clerk API Key + User ID (service-to-service, best for Jotty)
      2. Internal service header (localhost / same-network)
      3. Bearer token (Clerk session, browser-based)
    """
    headers: Dict[str, str] = {"Content-Type": "application/json"}

    api_key = os.getenv("JUSTJOT_API_KEY") or os.getenv("CLERK_SECRET_KEY")
    user_id = os.getenv("JUSTJOT_USER_ID")
    auth_token = os.getenv("JUSTJOT_AUTH_TOKEN")

    if api_key and user_id:
        headers["x-api-key"] = api_key
        headers["x-user-id"] = user_id
    else:
        # Internal service header (works when on same network / localhost)
        headers["x-internal-service"] = "true"
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

    return headers


def _call_justjot_api(
    method: str,
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Call JustJot.ai API endpoint.

    Uses /api/internal/* endpoints for service-to-service calls when using
    internal service header auth. Falls back to /api/* for Clerk API key auth.
    """
    auth_headers = _get_auth_headers()

    # Use internal endpoints when authenticating via service header
    if "x-internal-service" in auth_headers:
        if endpoint.startswith('/api/ideas'):
            endpoint = endpoint.replace('/api/ideas', '/api/internal/ideas', 1)
        elif endpoint.startswith('/api/templates'):
            endpoint = endpoint.replace('/api/templates', '/api/internal/templates', 1)
        elif endpoint.startswith('/api/tags'):
            endpoint = endpoint.replace('/api/tags', '/api/internal/tags', 1)

    url = f"{DEFAULT_API_URL}{endpoint}"

    try:
        response = requests.request(
            method=method,
            url=url,
            params=params,
            json=data,
            headers=auth_headers,
            timeout=30,
        )
        response.raise_for_status()

        try:
            return {'success': True, 'data': response.json()}
        except ValueError:
            return {'success': True, 'data': response.text}

    except requests.exceptions.RequestException as e:
        logger.error(f"JustJot API call failed: {e}")
        return {'success': False, 'error': f'API call failed: {str(e)}'}


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
    status.set_callback(params.pop('_status_callback', None))

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
    status.set_callback(params.pop('_status_callback', None))

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
        'status': params.get('status', 'Draft')
    }

    # Include sections if provided
    if params.get('sections'):
        idea_data['sections'] = params['sections']

    # Include userId if provided (for assignment)
    if params.get('userId'):
        idea_data['userId'] = params['userId']

    # Include author if provided
    if params.get('author'):
        idea_data['author'] = params['author']

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
    status.set_callback(params.pop('_status_callback', None))

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
    status.set_callback(params.pop('_status_callback', None))

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
    status.set_callback(params.pop('_status_callback', None))

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
    status.set_callback(params.pop('_status_callback', None))

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
    status.set_callback(params.pop('_status_callback', None))

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
    status.set_callback(params.pop('_status_callback', None))

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
    status.set_callback(params.pop('_status_callback', None))

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
    status.set_callback(params.pop('_status_callback', None))

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


# ============================================
# Ideas by Tag (merged from justjot-mcp-http)
# ============================================

async def get_ideas_by_tag_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get ideas filtered by tag.

    Args:
        params: Dictionary containing:
            - tag (str, required): Tag to filter by
            - limit (int, optional): Maximum results

    Returns:
        Dictionary with filtered ideas list
    """
    status.set_callback(params.pop('_status_callback', None))

    tag = params.get('tag')
    if not tag:
        return {'success': False, 'error': 'tag parameter is required'}

    query_params = {'tag': tag}
    if params.get('limit'):
        query_params['limit'] = str(params['limit'])

    result = _call_justjot_api('GET', '/api/ideas', params=query_params)

    if result.get('success'):
        ideas = result.get('data', [])
        return {
            'success': True,
            'tag': tag,
            'ideas': ideas,
            'count': len(ideas) if isinstance(ideas, list) else 0,
        }
    return result


__all__ = [
    'list_ideas_tool',
    'create_idea_tool',
    'get_idea_tool',
    'update_idea_tool',
    'delete_idea_tool',
    'get_ideas_by_tag_tool',
    'list_templates_tool',
    'get_template_tool',
    'add_section_tool',
    'update_section_tool',
    'list_tags_tool',
]
