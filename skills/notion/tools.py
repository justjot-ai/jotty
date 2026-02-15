"""
Notion Skill

Integrates with Notion API to search, read, create, and update pages and databases.
"""
import os
import logging
import requests
from typing import Dict, Any, Optional, List

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper

# Status emitter for progress updates
status = SkillStatus("notion")


logger = logging.getLogger(__name__)


class NotionClient:
    """Handles Notion API authentication and requests."""

    BASE_URL = "https://api.notion.com/v1"
    NOTION_VERSION = "2022-06-28"

    def __init__(self):
        self._api_key: Optional[str] = None

    def _get_api_key(self) -> str:
        """Get API key from environment or config file."""
        if self._api_key:
            return self._api_key

        # Check environment variable first
        api_key = os.environ.get('NOTION_API_KEY')
        if api_key:
            self._api_key = api_key
            return api_key

        # Check config file
        config_path = os.path.expanduser('~/.config/notion/api_key')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                api_key = f.read().strip()
                if api_key:
                    self._api_key = api_key
                    return api_key

        raise ValueError(
            "Notion API key not found. Set NOTION_API_KEY environment variable "
            "or create ~/.config/notion/api_key file."
        )

    def _get_headers(self) -> Dict[str, str]:
        """Build request headers with authentication."""
        return {
            "Authorization": f"Bearer {self._get_api_key()}",
            "Notion-Version": self.NOTION_VERSION,
            "Content-Type": "application/json"
        }

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """Make an authenticated request to Notion API."""
        url = f"{self.BASE_URL}{endpoint}"
        headers = self._get_headers()

        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=data,
                timeout=timeout
            )

            result = response.json()

            if response.status_code >= 400:
                error_msg = result.get('message', 'Unknown error')
                return {
                    'success': False,
                    'error': f"Notion API error ({response.status_code}): {error_msg}",
                    'status_code': response.status_code
                }

            return {
                'success': True,
                'data': result,
                'status_code': response.status_code
            }

        except requests.Timeout:
            return {
                'success': False,
                'error': f'Request timed out after {timeout} seconds'
            }
        except requests.RequestException as e:
            return {
                'success': False,
                'error': f'Request failed: {str(e)}'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Error making Notion API request: {str(e)}'
            }


# Global client instance
_client = NotionClient()


@tool_wrapper()
def search_pages_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search for pages and databases in Notion.

    Args:
        params: Dictionary containing:
            - query (str, required): Search query text
            - filter (dict, optional): Filter by object type - {"property": "object", "value": "page"} or "database"
            - sort (dict, optional): Sort results - {"direction": "ascending" or "descending", "timestamp": "last_edited_time"}
            - page_size (int, optional): Number of results (default: 10, max: 100)
            - start_cursor (str, optional): Pagination cursor
            - timeout (int, optional): Request timeout in seconds (default: 30)

    Returns:
        Dictionary with:
            - success (bool): Whether request succeeded
            - results (list): List of matching pages/databases
            - has_more (bool): Whether more results exist
            - next_cursor (str): Cursor for next page
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))
    query = params.get('query')
    if not query:
        return {'success': False, 'error': 'query parameter is required'}

    request_body = {
        'query': query,
        'page_size': min(params.get('page_size', 10), 100)
    }

    if params.get('filter'):
        request_body['filter'] = params['filter']

    if params.get('sort'):
        request_body['sort'] = params['sort']

    if params.get('start_cursor'):
        request_body['start_cursor'] = params['start_cursor']

    result = _client._make_request(
        'POST',
        '/search',
        request_body,
        timeout=params.get('timeout', 30)
    )

    if not result['success']:
        return result

    data = result['data']
    return {
        'success': True,
        'results': data.get('results', []),
        'has_more': data.get('has_more', False),
        'next_cursor': data.get('next_cursor'),
        'result_count': len(data.get('results', []))
    }


@tool_wrapper()
def get_page_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a page's properties and optionally its content blocks.

    Args:
        params: Dictionary containing:
            - page_id (str, required): The Notion page ID
            - include_content (bool, optional): Also fetch page content blocks (default: True)
            - timeout (int, optional): Request timeout in seconds (default: 30)

    Returns:
        Dictionary with:
            - success (bool): Whether request succeeded
            - page (dict): Page properties
            - content (list, optional): List of content blocks
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    page_id = params.get('page_id')
    if not page_id:
        return {'success': False, 'error': 'page_id parameter is required'}

    # Clean page ID (remove dashes if present, or extract from URL)
    page_id = page_id.replace('-', '')
    if '/' in page_id:
        page_id = page_id.split('/')[-1].split('?')[0]

    timeout = params.get('timeout', 30)
    include_content = params.get('include_content', True)

    # Get page properties
    page_result = _client._make_request('GET', f'/pages/{page_id}', timeout=timeout)

    if not page_result['success']:
        return page_result

    response = {
        'success': True,
        'page': page_result['data']
    }

    # Optionally get page content (blocks)
    if include_content:
        blocks_result = _client._make_request(
            'GET',
            f'/blocks/{page_id}/children?page_size=100',
            timeout=timeout
        )

        if blocks_result['success']:
            response['content'] = blocks_result['data'].get('results', [])
        else:
            response['content_error'] = blocks_result.get('error')

    return response


@tool_wrapper()
def create_page_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new page in Notion.

    Args:
        params: Dictionary containing:
            - parent_id (str, required): Parent page or database ID
            - parent_type (str, optional): 'page_id' or 'database_id' (default: 'page_id')
            - title (str, required): Page title
            - properties (dict, optional): Additional properties (for database pages)
            - content (list, optional): List of block objects for page content
            - icon (dict, optional): Page icon - {"type": "emoji", "emoji": "..."} or {"type": "external", "external": {"url": "..."}}
            - cover (dict, optional): Cover image - {"type": "external", "external": {"url": "..."}}
            - timeout (int, optional): Request timeout in seconds (default: 30)

    Returns:
        Dictionary with:
            - success (bool): Whether creation succeeded
            - page_id (str): ID of created page
            - url (str): URL of created page
            - page (dict): Full page object
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))
    parent_id = params.get('parent_id')
    title = params.get('title')

    if not parent_id:
        return {'success': False, 'error': 'parent_id parameter is required'}
    if not title:
        return {'success': False, 'error': 'title parameter is required'}

    # Clean parent ID
    parent_id = parent_id.replace('-', '')

    parent_type = params.get('parent_type', 'page_id')

    # Build request body
    request_body = {
        'parent': {parent_type: parent_id}
    }

    # Handle properties based on parent type
    if parent_type == 'database_id':
        # For database pages, title goes in properties
        properties = params.get('properties', {})
        properties['title'] = {
            'title': [{'text': {'content': title}}]
        }
        request_body['properties'] = properties
    else:
        # For regular pages, title is in properties.title
        request_body['properties'] = {
            'title': {
                'title': [{'text': {'content': title}}]
            }
        }

    # Add optional icon
    if params.get('icon'):
        request_body['icon'] = params['icon']

    # Add optional cover
    if params.get('cover'):
        request_body['cover'] = params['cover']

    # Add content blocks
    if params.get('content'):
        request_body['children'] = params['content']

    result = _client._make_request(
        'POST',
        '/pages',
        request_body,
        timeout=params.get('timeout', 30)
    )

    if not result['success']:
        return result

    page_data = result['data']
    return {
        'success': True,
        'page_id': page_data.get('id'),
        'url': page_data.get('url'),
        'page': page_data
    }


@tool_wrapper()
def update_page_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update a page's properties in Notion.

    Args:
        params: Dictionary containing:
            - page_id (str, required): The Notion page ID
            - properties (dict, optional): Properties to update
            - archived (bool, optional): Set to True to archive the page
            - icon (dict, optional): New page icon
            - cover (dict, optional): New cover image
            - timeout (int, optional): Request timeout in seconds (default: 30)

    Returns:
        Dictionary with:
            - success (bool): Whether update succeeded
            - page (dict): Updated page object
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    page_id = params.get('page_id')
    if not page_id:
        return {'success': False, 'error': 'page_id parameter is required'}

    # Clean page ID
    page_id = page_id.replace('-', '')

    request_body = {}

    if params.get('properties'):
        request_body['properties'] = params['properties']

    if params.get('archived') is not None:
        request_body['archived'] = params['archived']

    if params.get('icon'):
        request_body['icon'] = params['icon']

    if params.get('cover'):
        request_body['cover'] = params['cover']

    if not request_body:
        return {'success': False, 'error': 'No update parameters provided'}

    result = _client._make_request(
        'PATCH',
        f'/pages/{page_id}',
        request_body,
        timeout=params.get('timeout', 30)
    )

    if not result['success']:
        return result

    return {
        'success': True,
        'page': result['data']
    }


@tool_wrapper()
def query_database_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Query a Notion database with filters and sorts.

    Args:
        params: Dictionary containing:
            - database_id (str, required): The Notion database ID
            - filter (dict, optional): Filter conditions
            - sorts (list, optional): Sort conditions
            - page_size (int, optional): Number of results (default: 100, max: 100)
            - start_cursor (str, optional): Pagination cursor
            - timeout (int, optional): Request timeout in seconds (default: 30)

    Filter examples:
        - {"property": "Status", "select": {"equals": "Done"}}
        - {"property": "Due Date", "date": {"before": "2024-01-01"}}
        - {"and": [filter1, filter2]}
        - {"or": [filter1, filter2]}

    Sort examples:
        - [{"property": "Created", "direction": "descending"}]
        - [{"timestamp": "last_edited_time", "direction": "ascending"}]

    Returns:
        Dictionary with:
            - success (bool): Whether query succeeded
            - results (list): List of database items
            - has_more (bool): Whether more results exist
            - next_cursor (str): Cursor for next page
            - error (str, optional): Error message if failed
    """
    database_id = params.get('database_id')
    if not database_id:
        return {'success': False, 'error': 'database_id parameter is required'}

    # Clean database ID
    database_id = database_id.replace('-', '')

    request_body = {
        'page_size': min(params.get('page_size', 100), 100)
    }

    if params.get('filter'):
        request_body['filter'] = params['filter']

    if params.get('sorts'):
        request_body['sorts'] = params['sorts']

    if params.get('start_cursor'):
        request_body['start_cursor'] = params['start_cursor']

    result = _client._make_request(
        'POST',
        f'/databases/{database_id}/query',
        request_body,
        timeout=params.get('timeout', 30)
    )

    if not result['success']:
        return result

    data = result['data']
    return {
        'success': True,
        'results': data.get('results', []),
        'has_more': data.get('has_more', False),
        'next_cursor': data.get('next_cursor'),
        'result_count': len(data.get('results', []))
    }


@tool_wrapper()
def create_database_item_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a new item (page) to a Notion database.

    Args:
        params: Dictionary containing:
            - database_id (str, required): The Notion database ID
            - properties (dict, required): Properties for the new item
            - content (list, optional): List of block objects for page content
            - icon (dict, optional): Item icon
            - cover (dict, optional): Cover image
            - timeout (int, optional): Request timeout in seconds (default: 30)

    Property value examples:
        - Title: {"title": [{"text": {"content": "My Title"}}]}
        - Rich text: {"rich_text": [{"text": {"content": "Some text"}}]}
        - Number: {"number": 42}
        - Select: {"select": {"name": "Option A"}}
        - Multi-select: {"multi_select": [{"name": "Tag1"}, {"name": "Tag2"}]}
        - Date: {"date": {"start": "2024-01-01", "end": "2024-01-02"}}
        - Checkbox: {"checkbox": True}
        - URL: {"url": "https://example.com"}
        - Email: {"email": "test@example.com"}
        - Phone: {"phone_number": "+1234567890"}

    Returns:
        Dictionary with:
            - success (bool): Whether creation succeeded
            - item_id (str): ID of created item
            - url (str): URL of created item
            - item (dict): Full item object
            - error (str, optional): Error message if failed
    """
    database_id = params.get('database_id')
    properties = params.get('properties')

    if not database_id:
        return {'success': False, 'error': 'database_id parameter is required'}
    if not properties:
        return {'success': False, 'error': 'properties parameter is required'}

    # Clean database ID
    database_id = database_id.replace('-', '')

    request_body = {
        'parent': {'database_id': database_id},
        'properties': properties
    }

    if params.get('content'):
        request_body['children'] = params['content']

    if params.get('icon'):
        request_body['icon'] = params['icon']

    if params.get('cover'):
        request_body['cover'] = params['cover']

    result = _client._make_request(
        'POST',
        '/pages',
        request_body,
        timeout=params.get('timeout', 30)
    )

    if not result['success']:
        return result

    item_data = result['data']
    return {
        'success': True,
        'item_id': item_data.get('id'),
        'url': item_data.get('url'),
        'item': item_data
    }
