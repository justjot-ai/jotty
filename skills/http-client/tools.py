"""
HTTP Client Skill

Make HTTP requests with any method.
Refactored to use Jotty core utilities.
"""

import requests
from typing import Dict, Any

from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper

from Jotty.core.utils.skill_status import SkillStatus

# Status emitter for progress updates
status = SkillStatus("http-client")



def _parse_response_body(response: requests.Response) -> tuple:
    """Parse response body, returning (body, body_type) tuple."""
    try:
        return response.json(), 'json'
    except Exception:
        return response.text, 'text'


def _make_http_request(method: str, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Common HTTP request handler."""
    headers = params.get('headers', {})
    timeout = params.get('timeout', 30)
    json_data = params.get('json')
    data = params.get('data')
    query_params = params.get('params', {})

    try:
        kwargs = {'headers': headers, 'timeout': timeout}

        if method == 'GET':
            kwargs['params'] = query_params or json_data
        elif json_data is not None:
            kwargs['json'] = json_data
        elif data is not None:
            kwargs['data'] = data

        response = requests.request(method, url, **kwargs)
        body, body_type = _parse_response_body(response)

        return tool_response(
            method=method,
            status_code=response.status_code,
            headers=dict(response.headers),
            body=body,
            body_type=body_type,
            url=str(response.url)
        )

    except requests.Timeout:
        return tool_error(f'Request timed out after {timeout} seconds')
    except requests.RequestException as e:
        return tool_error(f'Request failed: {str(e)}')


@tool_wrapper(required_params=['url'])
def http_get_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform HTTP GET request.

    Args:
        params: Dictionary containing:
            - url (str, required): URL to request
            - headers (dict, optional): HTTP headers
            - params (dict, optional): URL query parameters
            - timeout (int, optional): Request timeout in seconds (default: 30)

    Returns:
        Dictionary with success, status_code, headers, body, body_type, url
    """
    status.set_callback(params.pop('_status_callback', None))

    return _make_http_request('GET', params['url'], params)


@tool_wrapper(required_params=['url'])
def http_post_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform HTTP POST request.

    Args:
        params: Dictionary containing:
            - url (str, required): URL to request
            - data (dict/str, optional): Request body data
            - json (dict, optional): JSON data to send
            - headers (dict, optional): HTTP headers
            - timeout (int, optional): Request timeout in seconds (default: 30)

    Returns:
        Dictionary with success, status_code, headers, body, body_type, url
    """
    status.set_callback(params.pop('_status_callback', None))

    return _make_http_request('POST', params['url'], params)


@tool_wrapper(required_params=['method', 'url'])
def http_request_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform HTTP request with any method.

    Args:
        params: Dictionary containing:
            - method (str, required): HTTP method (GET, POST, PUT, DELETE, PATCH, etc.)
            - url (str, required): URL to request
            - data (dict/str, optional): Request body data
            - json (dict, optional): JSON data to send
            - headers (dict, optional): HTTP headers
            - timeout (int, optional): Request timeout in seconds (default: 30)

    Returns:
        Dictionary with success, method, status_code, headers, body, body_type, url
    """
    status.set_callback(params.pop('_status_callback', None))

    method = params['method'].upper()

    if method not in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']:
        return tool_error(f'Unsupported HTTP method: {method}')

    return _make_http_request(method, params['url'], params)


__all__ = ['http_get_tool', 'http_post_tool', 'http_request_tool']
