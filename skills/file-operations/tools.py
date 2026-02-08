"""
File Operations Skill

Read, write, and manage files and directories.
Refactored to use Jotty core utilities.
"""

import glob
import shutil
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper


@tool_wrapper(required_params=['path'])
def read_file_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read the contents of a file.

    Args:
        params: Dictionary containing:
            - path (str, required): Path to the file
            - encoding (str, optional): File encoding (default: 'utf-8')

    Returns:
        Dictionary with success, content, path, size
    """
    encoding = params.get('encoding', 'utf-8')
    file_path = Path(params['path'])

    if not file_path.exists():
        return tool_error(f'File not found: {params["path"]}')

    if not file_path.is_file():
        return tool_error(f'Path is not a file: {params["path"]}')

    try:
        content = file_path.read_text(encoding=encoding)
        return tool_response(content=content, path=str(file_path), size=len(content))
    except UnicodeDecodeError as e:
        return tool_error(f'Encoding error: {str(e)}. Try specifying encoding parameter.')


@tool_wrapper(required_params=['path', 'content'])
def write_file_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Write content to a file.

    Args:
        params: Dictionary containing:
            - path (str, required): Path to the file
            - content (str, required): Content to write
            - encoding (str, optional): File encoding (default: 'utf-8')
            - mode (str, optional): 'w' (overwrite) or 'a' (append), default: 'w'

    Returns:
        Dictionary with success, path, bytes_written
    """
    encoding = params.get('encoding', 'utf-8')
    file_path = Path(params['path'])

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(params['content'], encoding=encoding)

    return tool_response(
        path=str(file_path),
        bytes_written=len(params['content'].encode(encoding))
    )


@tool_wrapper(required_params=['path'])
def list_directory_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List files and directories in a path.

    Args:
        params: Dictionary containing:
            - path (str, required): Directory path to list
            - recursive (bool, optional): List recursively (default: False)
            - include_hidden (bool, optional): Include hidden files (default: False)

    Returns:
        Dictionary with success, items, count
    """
    recursive = params.get('recursive', False)
    include_hidden = params.get('include_hidden', False)

    dir_path = Path(params['path'])
    if not dir_path.exists():
        return tool_error(f'Directory not found: {params["path"]}')

    if not dir_path.is_dir():
        return tool_error(f'Path is not a directory: {params["path"]}')

    items = []
    iterator = dir_path.rglob('*') if recursive else dir_path.iterdir()

    for item in iterator:
        if not include_hidden and item.name.startswith('.'):
            continue

        try:
            stat = item.stat()
            items.append({
                'name': item.name,
                'path': str(item),
                'type': 'directory' if item.is_dir() else 'file',
                'size': stat.st_size if item.is_file() else None,
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        except (OSError, PermissionError):
            continue

    items.sort(key=lambda x: (x['type'] != 'directory', x['name'].lower()))
    return tool_response(items=items, count=len(items))


@tool_wrapper(required_params=['path'])
def create_directory_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a directory (and parent directories if needed).

    Args:
        params: Dictionary containing:
            - path (str, required): Directory path to create
            - parents (bool, optional): Create parent directories (default: True)

    Returns:
        Dictionary with success, path
    """
    parents = params.get('parents', True)
    dir_path = Path(params['path'])

    if dir_path.exists():
        if dir_path.is_dir():
            return tool_response(path=str(dir_path), message='Directory already exists')
        else:
            return tool_error(f'Path exists but is not a directory: {params["path"]}')

    dir_path.mkdir(parents=parents, exist_ok=True)
    return tool_response(path=str(dir_path))


@tool_wrapper(required_params=['path'])
def delete_file_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Delete a file or directory.

    Args:
        params: Dictionary containing:
            - path (str, required): Path to file/directory to delete
            - recursive (bool, optional): Delete directory recursively (default: False)

    Returns:
        Dictionary with success, path
    """
    recursive = params.get('recursive', False)
    item_path = Path(params['path'])

    if not item_path.exists():
        return tool_error(f'Path not found: {params["path"]}')

    if item_path.is_dir():
        if recursive:
            shutil.rmtree(item_path)
        else:
            try:
                item_path.rmdir()
            except OSError:
                return tool_error('Directory not empty. Set recursive=True to delete recursively.')
    else:
        item_path.unlink()

    return tool_response(path=str(item_path))


@tool_wrapper(required_params=['directory', 'pattern'])
def search_files_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search for files matching a pattern.

    Args:
        params: Dictionary containing:
            - directory (str, required): Directory to search in
            - pattern (str, required): Filename pattern (glob wildcards: *, ?, [])
            - recursive (bool, optional): Search recursively (default: True)

    Returns:
        Dictionary with success, matches, count
    """
    recursive = params.get('recursive', True)
    dir_path = Path(params['directory'])

    if not dir_path.exists() or not dir_path.is_dir():
        return tool_error(f'Invalid directory: {params["directory"]}')

    if recursive:
        search_path = dir_path / '**' / params['pattern']
        matches = [str(p) for p in glob.glob(str(search_path), recursive=True) if Path(p).is_file()]
    else:
        search_path = dir_path / params['pattern']
        matches = [str(p) for p in glob.glob(str(search_path)) if Path(p).is_file()]

    return tool_response(matches=matches, count=len(matches))


@tool_wrapper(required_params=['path'])
def get_file_info_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get metadata about a file or directory.

    Args:
        params: Dictionary containing:
            - path (str, required): Path to file/directory

    Returns:
        Dictionary with success, exists, type, size, modified, path
    """
    item_path = Path(params['path'])

    if not item_path.exists():
        return tool_response(exists=False)

    stat = item_path.stat()
    return tool_response(
        exists=True,
        type='directory' if item_path.is_dir() else 'file',
        size=stat.st_size if item_path.is_file() else None,
        modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
        path=str(item_path)
    )


__all__ = [
    'read_file_tool',
    'write_file_tool',
    'list_directory_tool',
    'create_directory_tool',
    'delete_file_tool',
    'search_files_tool',
    'get_file_info_tool'
]
