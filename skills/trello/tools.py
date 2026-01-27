"""
Trello Skill

Integrates with Trello API to manage boards, lists, and cards.
"""
import os
import logging
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class TrelloClient:
    """Handles Trello API authentication and requests."""

    BASE_URL = "https://api.trello.com/1"

    def __init__(self):
        self._api_key: Optional[str] = None
        self._token: Optional[str] = None

    def _get_credentials(self) -> tuple:
        """Get API key and token from environment or config file."""
        if self._api_key and self._token:
            return self._api_key, self._token

        # Check environment variables first
        api_key = os.environ.get('TRELLO_API_KEY')
        token = os.environ.get('TRELLO_TOKEN')

        if api_key and token:
            self._api_key = api_key
            self._token = token
            return api_key, token

        # Check config files
        config_dir = os.path.expanduser('~/.config/trello')
        api_key_path = os.path.join(config_dir, 'api_key')
        token_path = os.path.join(config_dir, 'token')

        if os.path.exists(api_key_path) and os.path.exists(token_path):
            with open(api_key_path, 'r') as f:
                api_key = f.read().strip()
            with open(token_path, 'r') as f:
                token = f.read().strip()
            if api_key and token:
                self._api_key = api_key
                self._token = token
                return api_key, token

        raise ValueError(
            "Trello credentials not found. Set TRELLO_API_KEY and TRELLO_TOKEN environment variables "
            "or create ~/.config/trello/api_key and ~/.config/trello/token files."
        )

    def _get_auth_params(self) -> Dict[str, str]:
        """Build authentication query parameters."""
        api_key, token = self._get_credentials()
        return {
            "key": api_key,
            "token": token
        }

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """Make an authenticated request to Trello API."""
        url = f"{self.BASE_URL}{endpoint}"

        # Merge auth params with any additional params
        request_params = self._get_auth_params()
        if params:
            request_params.update(params)

        try:
            response = requests.request(
                method=method,
                url=url,
                params=request_params,
                json=data if method in ['POST', 'PUT', 'PATCH'] else None,
                timeout=timeout
            )

            # Handle empty responses (like DELETE)
            if response.status_code == 200 and not response.text:
                return {
                    'success': True,
                    'data': None,
                    'status_code': response.status_code
                }

            try:
                result = response.json()
            except Exception:
                result = {'message': response.text}

            if response.status_code >= 400:
                error_msg = result.get('message', result.get('error', 'Unknown error'))
                return {
                    'success': False,
                    'error': f"Trello API error ({response.status_code}): {error_msg}",
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
                'error': f'Error making Trello API request: {str(e)}'
            }


# Global client instance
_client = TrelloClient()


def list_boards_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List all boards accessible to the authenticated user.

    Args:
        params: Dictionary containing:
            - filter (str, optional): Filter boards - 'all', 'open', 'closed', 'members', 'organization', 'public', 'starred' (default: 'all')
            - fields (str, optional): Comma-separated fields to include (default: 'name,desc,url,closed,starred')
            - timeout (int, optional): Request timeout in seconds (default: 30)

    Returns:
        Dictionary with:
            - success (bool): Whether request succeeded
            - boards (list): List of board objects
            - board_count (int): Number of boards returned
            - error (str, optional): Error message if failed
    """
    request_params = {
        'filter': params.get('filter', 'all'),
        'fields': params.get('fields', 'name,desc,url,closed,starred,idOrganization')
    }

    result = _client._make_request(
        'GET',
        '/members/me/boards',
        params=request_params,
        timeout=params.get('timeout', 30)
    )

    if not result['success']:
        return result

    boards = result['data']
    return {
        'success': True,
        'boards': boards,
        'board_count': len(boards)
    }


def get_board_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get board details including its lists.

    Args:
        params: Dictionary containing:
            - board_id (str, required): The Trello board ID or shortLink
            - include_lists (bool, optional): Include lists in response (default: True)
            - include_members (bool, optional): Include board members (default: False)
            - include_labels (bool, optional): Include board labels (default: False)
            - list_filter (str, optional): Filter lists - 'all', 'open', 'closed' (default: 'open')
            - timeout (int, optional): Request timeout in seconds (default: 30)

    Returns:
        Dictionary with:
            - success (bool): Whether request succeeded
            - board (dict): Board details
            - lists (list, optional): List of lists on the board
            - members (list, optional): List of board members
            - labels (list, optional): List of board labels
            - error (str, optional): Error message if failed
    """
    board_id = params.get('board_id')
    if not board_id:
        return {'success': False, 'error': 'board_id parameter is required'}

    timeout = params.get('timeout', 30)
    include_lists = params.get('include_lists', True)
    include_members = params.get('include_members', False)
    include_labels = params.get('include_labels', False)

    # Get board details
    board_result = _client._make_request(
        'GET',
        f'/boards/{board_id}',
        params={'fields': 'name,desc,url,closed,starred,idOrganization,prefs'},
        timeout=timeout
    )

    if not board_result['success']:
        return board_result

    response = {
        'success': True,
        'board': board_result['data']
    }

    # Optionally get lists
    if include_lists:
        list_filter = params.get('list_filter', 'open')
        lists_result = _client._make_request(
            'GET',
            f'/boards/{board_id}/lists',
            params={'filter': list_filter, 'fields': 'name,closed,pos'},
            timeout=timeout
        )
        if lists_result['success']:
            response['lists'] = lists_result['data']
        else:
            response['lists_error'] = lists_result.get('error')

    # Optionally get members
    if include_members:
        members_result = _client._make_request(
            'GET',
            f'/boards/{board_id}/members',
            params={'fields': 'fullName,username'},
            timeout=timeout
        )
        if members_result['success']:
            response['members'] = members_result['data']
        else:
            response['members_error'] = members_result.get('error')

    # Optionally get labels
    if include_labels:
        labels_result = _client._make_request(
            'GET',
            f'/boards/{board_id}/labels',
            timeout=timeout
        )
        if labels_result['success']:
            response['labels'] = labels_result['data']
        else:
            response['labels_error'] = labels_result.get('error')

    return response


def list_cards_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List cards in a list or board.

    Args:
        params: Dictionary containing:
            - list_id (str, optional): The Trello list ID (use this OR board_id)
            - board_id (str, optional): The Trello board ID (use this OR list_id)
            - filter (str, optional): Filter cards - 'all', 'open', 'closed', 'visible' (default: 'open')
            - fields (str, optional): Comma-separated fields to include
            - limit (int, optional): Maximum number of cards (default: 100, max: 1000)
            - timeout (int, optional): Request timeout in seconds (default: 30)

    Returns:
        Dictionary with:
            - success (bool): Whether request succeeded
            - cards (list): List of card objects
            - card_count (int): Number of cards returned
            - error (str, optional): Error message if failed
    """
    list_id = params.get('list_id')
    board_id = params.get('board_id')

    if not list_id and not board_id:
        return {'success': False, 'error': 'Either list_id or board_id parameter is required'}

    request_params = {
        'filter': params.get('filter', 'open'),
        'fields': params.get('fields', 'name,desc,url,closed,pos,due,dueComplete,idList,idLabels,idMembers,labels'),
        'limit': min(params.get('limit', 100), 1000)
    }

    if list_id:
        endpoint = f'/lists/{list_id}/cards'
    else:
        endpoint = f'/boards/{board_id}/cards'

    result = _client._make_request(
        'GET',
        endpoint,
        params=request_params,
        timeout=params.get('timeout', 30)
    )

    if not result['success']:
        return result

    cards = result['data']
    return {
        'success': True,
        'cards': cards,
        'card_count': len(cards)
    }


def create_card_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new card in a Trello list.

    Args:
        params: Dictionary containing:
            - list_id (str, required): The ID of the list to add the card to
            - name (str, required): The name/title of the card
            - desc (str, optional): The description of the card
            - pos (str/float, optional): Position - 'top', 'bottom', or a positive float (default: 'bottom')
            - due (str, optional): Due date in ISO 8601 format (e.g., '2024-12-31T23:59:59.000Z')
            - due_complete (bool, optional): Whether the due date is complete (default: False)
            - id_labels (list, optional): List of label IDs to add
            - id_members (list, optional): List of member IDs to assign
            - url_source (str, optional): URL to attach to the card
            - timeout (int, optional): Request timeout in seconds (default: 30)

    Returns:
        Dictionary with:
            - success (bool): Whether creation succeeded
            - card_id (str): ID of created card
            - card (dict): Full card object
            - url (str): URL of the created card
            - error (str, optional): Error message if failed
    """
    list_id = params.get('list_id')
    name = params.get('name')

    if not list_id:
        return {'success': False, 'error': 'list_id parameter is required'}
    if not name:
        return {'success': False, 'error': 'name parameter is required'}

    request_params = {
        'idList': list_id,
        'name': name,
        'pos': params.get('pos', 'bottom')
    }

    if params.get('desc'):
        request_params['desc'] = params['desc']

    if params.get('due'):
        request_params['due'] = params['due']

    if params.get('due_complete') is not None:
        request_params['dueComplete'] = params['due_complete']

    if params.get('id_labels'):
        request_params['idLabels'] = ','.join(params['id_labels'])

    if params.get('id_members'):
        request_params['idMembers'] = ','.join(params['id_members'])

    if params.get('url_source'):
        request_params['urlSource'] = params['url_source']

    result = _client._make_request(
        'POST',
        '/cards',
        params=request_params,
        timeout=params.get('timeout', 30)
    )

    if not result['success']:
        return result

    card_data = result['data']
    return {
        'success': True,
        'card_id': card_data.get('id'),
        'card': card_data,
        'url': card_data.get('url')
    }


def update_card_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update an existing card (move, rename, change description, etc.).

    Args:
        params: Dictionary containing:
            - card_id (str, required): The ID of the card to update
            - name (str, optional): New name/title for the card
            - desc (str, optional): New description for the card
            - closed (bool, optional): Archive (True) or unarchive (False) the card
            - id_list (str, optional): ID of the list to move the card to
            - id_board (str, optional): ID of the board to move the card to
            - pos (str/float, optional): New position - 'top', 'bottom', or a positive float
            - due (str, optional): New due date in ISO 8601 format (null to remove)
            - due_complete (bool, optional): Whether the due date is complete
            - id_labels (list, optional): List of label IDs (replaces existing)
            - id_members (list, optional): List of member IDs (replaces existing)
            - timeout (int, optional): Request timeout in seconds (default: 30)

    Returns:
        Dictionary with:
            - success (bool): Whether update succeeded
            - card (dict): Updated card object
            - error (str, optional): Error message if failed
    """
    card_id = params.get('card_id')
    if not card_id:
        return {'success': False, 'error': 'card_id parameter is required'}

    request_params = {}

    if params.get('name') is not None:
        request_params['name'] = params['name']

    if params.get('desc') is not None:
        request_params['desc'] = params['desc']

    if params.get('closed') is not None:
        request_params['closed'] = params['closed']

    if params.get('id_list'):
        request_params['idList'] = params['id_list']

    if params.get('id_board'):
        request_params['idBoard'] = params['id_board']

    if params.get('pos') is not None:
        request_params['pos'] = params['pos']

    if 'due' in params:
        request_params['due'] = params['due']

    if params.get('due_complete') is not None:
        request_params['dueComplete'] = params['due_complete']

    if params.get('id_labels') is not None:
        request_params['idLabels'] = ','.join(params['id_labels']) if params['id_labels'] else ''

    if params.get('id_members') is not None:
        request_params['idMembers'] = ','.join(params['id_members']) if params['id_members'] else ''

    if not request_params:
        return {'success': False, 'error': 'No update parameters provided'}

    result = _client._make_request(
        'PUT',
        f'/cards/{card_id}',
        params=request_params,
        timeout=params.get('timeout', 30)
    )

    if not result['success']:
        return result

    return {
        'success': True,
        'card': result['data']
    }


def add_comment_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a comment to a card.

    Args:
        params: Dictionary containing:
            - card_id (str, required): The ID of the card to comment on
            - text (str, required): The comment text
            - timeout (int, optional): Request timeout in seconds (default: 30)

    Returns:
        Dictionary with:
            - success (bool): Whether comment was added
            - comment_id (str): ID of the created comment
            - comment (dict): Full comment object
            - error (str, optional): Error message if failed
    """
    card_id = params.get('card_id')
    text = params.get('text')

    if not card_id:
        return {'success': False, 'error': 'card_id parameter is required'}
    if not text:
        return {'success': False, 'error': 'text parameter is required'}

    result = _client._make_request(
        'POST',
        f'/cards/{card_id}/actions/comments',
        params={'text': text},
        timeout=params.get('timeout', 30)
    )

    if not result['success']:
        return result

    comment_data = result['data']
    return {
        'success': True,
        'comment_id': comment_data.get('id'),
        'comment': comment_data
    }
