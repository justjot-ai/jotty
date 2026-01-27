"""
JustJot.ai MCP HTTP Skill

Accesses JustJot.ai MCP tools over HTTP (remote access).
Makes JustJot.ai MCP available as a global standard skill in Jotty.
"""
import logging
import requests
import json
from typing import Dict, Any, Optional, List, Generator
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class JustJotMCPHTTPClient:
    """
    HTTP client for JustJot.ai MCP server.
    
    Makes JustJot.ai MCP accessible over HTTP as a global standard.
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Initialize JustJot MCP HTTP client.
        
        Supports multiple authentication methods:
        1. API Key + User ID (service-to-service) - Recommended for Jotty
        2. Bearer Token (Clerk session token) - For browser-based auth
        
        Args:
            base_url: Base URL for JustJot.ai API (default: https://justjot.ai)
            auth_token: Authentication token (Clerk session token) - Method 2
            api_key: CLERK_SECRET_KEY for service auth - Method 1
            user_id: User ID (Clerk user ID like user_xxx) - Method 1
        """
        self.base_url = base_url or os.getenv('JUSTJOT_API_URL', 'https://justjot.ai')
        
        # Method 1: API Key + User ID (service-to-service)
        self.api_key = api_key or os.getenv('JUSTJOT_API_KEY') or os.getenv('CLERK_SECRET_KEY')
        self.user_id = user_id or os.getenv('JUSTJOT_USER_ID')
        
        # Method 2: Bearer Token (browser session)
        self.auth_token = auth_token or os.getenv('JUSTJOT_AUTH_TOKEN')
        
        self.tools_endpoint = f"{self.base_url}/api/mcp/tools"
        self.manifest_endpoint = f"{self.base_url}/api/mcp/manifest"
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Get request headers with authentication.
        
        Priority:
        1. API Key + User ID (service-to-service) - Best for Jotty
        2. Bearer Token (session token) - For browser-based auth
        """
        headers = {
            'Content-Type': 'application/json'
        }
        
        # Method 1: API Key + User ID (preferred for service-to-service)
        if self.api_key and self.user_id:
            headers['x-api-key'] = self.api_key
            headers['x-user-id'] = self.user_id
        # Method 2: Bearer Token (fallback)
        elif self.auth_token:
            headers['Authorization'] = f'Bearer {self.auth_token}'
        
        return headers
    
    def get_manifest(self) -> Dict[str, Any]:
        """Get MCP server manifest."""
        try:
            response = requests.get(
                self.manifest_endpoint,
                headers=self._get_headers(),
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting manifest: {e}")
            return {'error': str(e)}
    
    def list_tools(self) -> Dict[str, Any]:
        """List available MCP tools."""
        try:
            response = requests.get(
                f"{self.tools_endpoint}/list",
                headers=self._get_headers(),
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return {'error': str(e)}
    
    def call_tool(self, name: str, arguments: Dict[str, Any], stream: bool = False) -> Dict[str, Any]:
        """
        Call an MCP tool.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            stream: Whether to use streaming (SSE) endpoint
        
        Returns:
            Tool execution result
        """
        try:
            endpoint = f"{self.tools_endpoint}/call/stream" if stream else f"{self.tools_endpoint}/call"
            
            if stream:
                # Handle SSE streaming
                response = requests.post(
                    endpoint,
                    json={
                        'name': name,
                        'arguments': arguments
                    },
                    headers=self._get_headers(),
                    timeout=60,
                    stream=True
                )
                response.raise_for_status()
                
                # Parse SSE stream
                chunks = []
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data = line_str[6:]  # Remove 'data: ' prefix
                            try:
                                chunk_data = json.loads(data)
                                chunks.append(chunk_data.get('chunk', ''))
                            except:
                                pass
                
                # Combine chunks
                result_text = ''.join(chunks)
                try:
                    return {'success': True, **json.loads(result_text)}
                except:
                    return {'success': True, 'text': result_text}
            else:
                # Standard non-streaming request
                response = requests.post(
                    endpoint,
                    json={
                        'name': name,
                        'arguments': arguments
                    },
                    headers=self._get_headers(),
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                
                # Check rate limit headers
                if 'X-RateLimit-Remaining' in response.headers:
                    remaining = int(response.headers['X-RateLimit-Remaining'])
                    if remaining < 10:
                        logger.warning(f"Rate limit low: {remaining} requests remaining")
                
                # Parse MCP response format
                if 'content' in result and isinstance(result['content'], list):
                    if len(result['content']) > 0:
                        text_content = result['content'][0].get('text', '')
                        try:
                            parsed = json.loads(text_content)
                            return {'success': True, **parsed}
                        except:
                            return {'success': True, 'text': text_content}
                
                return result
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                reset_at = e.response.headers.get('X-RateLimit-Reset')
                return {
                    'success': False,
                    'error': 'Rate limit exceeded',
                    'resetAt': reset_at
                }
            logger.error(f"HTTP error calling tool {name}: {e}")
            return {'success': False, 'error': str(e)}
        except Exception as e:
            logger.error(f"Error calling tool {name}: {e}")
            return {'success': False, 'error': str(e)}


# Global client instance
_client: Optional[JustJotMCPHTTPClient] = None


def get_client() -> JustJotMCPHTTPClient:
    """Get or create global JustJot MCP HTTP client."""
    global _client
    if _client is None:
        _client = JustJotMCPHTTPClient()
    return _client


async def list_ideas_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List ideas from JustJot.ai via HTTP MCP.
    
    Args:
        params: Dictionary containing:
            - status (str, optional): Filter by status
            - limit (int, optional): Maximum results
            - tags (list, optional): Filter by tags
    
    Returns:
        Dictionary with ideas list
    """
    client = get_client()
    result = client.call_tool('list_ideas', params)
    
    if result.get('success'):
        return {
            'success': True,
            'ideas': result.get('ideas', []),
            'count': len(result.get('ideas', []))
        }
    else:
        return {
            'success': False,
            'error': result.get('error', 'Unknown error')
        }


async def create_idea_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create an idea in JustJot.ai via HTTP MCP.
    
    Args:
        params: Dictionary containing:
            - title (str, required): Idea title
            - description (str, optional): Idea description
            - tags (list, optional): Tags
            - status (str, optional): Status
            - sections (list, optional): Initial sections
            - userId (str, optional): User ID
            - author (str, optional): Author name
    
    Returns:
        Dictionary with created idea ID
    """
    client = get_client()
    result = client.call_tool('create_idea', params)
    
    if result.get('success'):
        return {
            'success': True,
            'idea_id': result.get('id'),
            'message': result.get('message', 'Idea created')
        }
    else:
        return {
            'success': False,
            'error': result.get('error', 'Unknown error')
        }


async def get_idea_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get an idea from JustJot.ai via HTTP MCP.
    
    Args:
        params: Dictionary containing:
            - id (str, required): Idea ID
    
    Returns:
        Dictionary with idea data
    """
    client = get_client()
    result = client.call_tool('get_idea', params)
    
    if result.get('success'):
        return {
            'success': True,
            'idea': result.get('idea')
        }
    else:
        return {
            'success': False,
            'error': result.get('error', 'Unknown error')
        }


async def update_idea_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update an idea in JustJot.ai via HTTP MCP.
    
    Args:
        params: Dictionary containing:
            - id (str, required): Idea ID
            - title (str, optional): Updated title
            - description (str, optional): Updated description
            - status (str, optional): Updated status
            - tags (list, optional): Updated tags
            - sections (list, optional): Updated sections
    
    Returns:
        Dictionary with update result
    """
    client = get_client()
    result = client.call_tool('update_idea', params)
    
    if result.get('success'):
        return {
            'success': True,
            'idea_id': result.get('id'),
            'message': result.get('message', 'Idea updated')
        }
    else:
        return {
            'success': False,
            'error': result.get('error', 'Unknown error')
        }


async def delete_idea_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Delete an idea from JustJot.ai via HTTP MCP.
    
    Args:
        params: Dictionary containing:
            - id (str, required): Idea ID
    
    Returns:
        Dictionary with deletion result
    """
    client = get_client()
    result = client.call_tool('delete_idea', params)
    
    if result.get('success'):
        return {
            'success': True,
            'message': result.get('message', 'Idea deleted')
        }
    else:
        return {
            'success': False,
            'error': result.get('error', 'Unknown error')
        }


async def get_ideas_by_tag_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get ideas by tag from JustJot.ai via HTTP MCP.
    
    Args:
        params: Dictionary containing:
            - tag (str, required): Tag to filter by
            - limit (int, optional): Maximum results
    
    Returns:
        Dictionary with ideas list
    """
    client = get_client()
    result = client.call_tool('get_ideas_by_tag', params)
    
    if result.get('success'):
        return {
            'success': True,
            'tag': result.get('tag'),
            'ideas': result.get('ideas', []),
            'count': len(result.get('ideas', []))
        }
    else:
        return {
            'success': False,
            'error': result.get('error', 'Unknown error')
        }
