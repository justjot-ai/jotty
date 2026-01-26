import requests
import json
from typing import Dict, Any


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
        Dictionary with:
            - success (bool): Whether request succeeded
            - status_code (int): HTTP status code
            - headers (dict): Response headers
            - body (str): Response body
            - error (str, optional): Error message if failed
    """
    try:
        url = params.get('url')
        if not url:
            return {
                'success': False,
                'error': 'url parameter is required'
            }
        
        headers = params.get('headers', {})
        query_params = params.get('params', {})
        timeout = params.get('timeout', 30)
        
        response = requests.get(
            url,
            headers=headers,
            params=query_params,
            timeout=timeout
        )
        
        # Try to parse as JSON, fallback to text
        try:
            body = response.json()
            body_type = 'json'
        except:
            body = response.text
            body_type = 'text'
        
        return {
            'success': True,
            'status_code': response.status_code,
            'headers': dict(response.headers),
            'body': body,
            'body_type': body_type,
            'url': response.url
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
            'error': f'Error making HTTP request: {str(e)}'
        }


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
        Dictionary with:
            - success (bool): Whether request succeeded
            - status_code (int): HTTP status code
            - headers (dict): Response headers
            - body (str): Response body
            - error (str, optional): Error message if failed
    """
    try:
        url = params.get('url')
        if not url:
            return {
                'success': False,
                'error': 'url parameter is required'
            }
        
        data = params.get('data')
        json_data = params.get('json')
        headers = params.get('headers', {})
        timeout = params.get('timeout', 30)
        
        # If json is provided, use it; otherwise use data
        if json_data is not None:
            response = requests.post(
                url,
                json=json_data,
                headers=headers,
                timeout=timeout
            )
        else:
            response = requests.post(
                url,
                data=data,
                headers=headers,
                timeout=timeout
            )
        
        # Try to parse as JSON, fallback to text
        try:
            body = response.json()
            body_type = 'json'
        except:
            body = response.text
            body_type = 'text'
        
        return {
            'success': True,
            'status_code': response.status_code,
            'headers': dict(response.headers),
            'body': body,
            'body_type': body_type,
            'url': response.url
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
            'error': f'Error making HTTP request: {str(e)}'
        }


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
        Dictionary with:
            - success (bool): Whether request succeeded
            - status_code (int): HTTP status code
            - headers (dict): Response headers
            - body (str): Response body
            - error (str, optional): Error message if failed
    """
    try:
        method = params.get('method', '').upper()
        url = params.get('url')
        
        if not method:
            return {
                'success': False,
                'error': 'method parameter is required'
            }
        
        if not url:
            return {
                'success': False,
                'error': 'url parameter is required'
            }
        
        if method not in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']:
            return {
                'success': False,
                'error': f'Unsupported HTTP method: {method}'
            }
        
        data = params.get('data')
        json_data = params.get('json')
        headers = params.get('headers', {})
        timeout = params.get('timeout', 30)
        
        # Use requests.request for flexibility
        kwargs = {
            'headers': headers,
            'timeout': timeout
        }
        
        if json_data is not None:
            kwargs['json'] = json_data
        elif data is not None:
            kwargs['data'] = data
        
        response = requests.request(method, url, **kwargs)
        
        # Try to parse as JSON, fallback to text
        try:
            body = response.json()
            body_type = 'json'
        except:
            body = response.text
            body_type = 'text'
        
        return {
            'success': True,
            'method': method,
            'status_code': response.status_code,
            'headers': dict(response.headers),
            'body': body,
            'body_type': body_type,
            'url': response.url
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
            'error': f'Error making HTTP request: {str(e)}'
        }
