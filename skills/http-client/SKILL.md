# HTTP Client Skill

## Description
Provides advanced HTTP request capabilities: GET, POST, PUT, DELETE requests with headers, authentication, and response handling.


## Type
base


## Capabilities
- data-fetch


## Triggers
- "make request"
- "api call"
- "http request"
- "fetch url"
- "post data"

## Category
workflow-automation

## Tools

### http_get_tool
Performs HTTP GET request.

**Parameters:**
- `url` (str, required): URL to request
- `headers` (dict, optional): HTTP headers
- `params` (dict, optional): URL query parameters
- `timeout` (int, optional): Request timeout in seconds (default: 30)

**Returns:**
- `success` (bool): Whether request succeeded
- `status_code` (int): HTTP status code
- `headers` (dict): Response headers
- `body` (str): Response body
- `error` (str, optional): Error message if failed

### http_post_tool
Performs HTTP POST request.

**Parameters:**
- `url` (str, required): URL to request
- `data` (dict/str, optional): Request body data
- `json` (dict, optional): JSON data to send
- `headers` (dict, optional): HTTP headers
- `timeout` (int, optional): Request timeout in seconds (default: 30)

**Returns:**
- `success` (bool): Whether request succeeded
- `status_code` (int): HTTP status code
- `headers` (dict): Response headers
- `body` (str): Response body
- `error` (str, optional): Error message if failed

### http_request_tool
Performs HTTP request with any method (GET, POST, PUT, DELETE, PATCH, etc.).

**Parameters:**
- `method` (str, required): HTTP method (GET, POST, PUT, DELETE, PATCH, etc.)
- `url` (str, required): URL to request
- `data` (dict/str, optional): Request body data
- `json` (dict, optional): JSON data to send
- `headers` (dict, optional): HTTP headers
- `timeout` (int, optional): Request timeout in seconds (default: 30)

**Returns:**
- `success` (bool): Whether request succeeded
- `status_code` (int): HTTP status code
- `headers` (dict): Response headers
- `body` (str): Response body
- `error` (str, optional): Error message if failed
