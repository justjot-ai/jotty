import os
import glob
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def read_file_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read the contents of a file.
    
    Args:
        params: Dictionary containing:
            - path (str, required): Path to the file to read
            - encoding (str, optional): File encoding (default: 'utf-8')
    
    Returns:
        Dictionary with:
            - success (bool): Whether operation succeeded
            - content (str): File contents
            - error (str, optional): Error message if failed
    """
    try:
        path = params.get('path')
        if not path:
            return {
                'success': False,
                'error': 'path parameter is required'
            }
        
        encoding = params.get('encoding', 'utf-8')
        
        file_path = Path(path)
        if not file_path.exists():
            return {
                'success': False,
                'error': f'File not found: {path}'
            }
        
        if not file_path.is_file():
            return {
                'success': False,
                'error': f'Path is not a file: {path}'
            }
        
        content = file_path.read_text(encoding=encoding)
        
        return {
            'success': True,
            'content': content,
            'path': str(file_path),
            'size': len(content)
        }
    except UnicodeDecodeError as e:
        return {
            'success': False,
            'error': f'Encoding error: {str(e)}. Try specifying encoding parameter.'
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error reading file: {str(e)}'
        }


def write_file_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Write content to a file.
    
    Args:
        params: Dictionary containing:
            - path (str, required): Path to the file to write
            - content (str, required): Content to write
            - encoding (str, optional): File encoding (default: 'utf-8')
            - mode (str, optional): Write mode - 'w' (overwrite) or 'a' (append), default: 'w'
    
    Returns:
        Dictionary with:
            - success (bool): Whether operation succeeded
            - path (str): Path to written file
            - bytes_written (int): Number of bytes written
            - error (str, optional): Error message if failed
    """
    try:
        path = params.get('path')
        content = params.get('content')
        
        if not path:
            return {
                'success': False,
                'error': 'path parameter is required'
            }
        
        if content is None:
            return {
                'success': False,
                'error': 'content parameter is required'
            }
        
        encoding = params.get('encoding', 'utf-8')
        mode = params.get('mode', 'w')
        
        file_path = Path(path)
        
        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        if mode == 'a':
            file_path.write_text(content, encoding=encoding)
        else:
            file_path.write_text(content, encoding=encoding)
        
        bytes_written = len(content.encode(encoding))
        
        return {
            'success': True,
            'path': str(file_path),
            'bytes_written': bytes_written
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error writing file: {str(e)}'
        }


def list_directory_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List files and directories in a path.
    
    Args:
        params: Dictionary containing:
            - path (str, required): Directory path to list
            - recursive (bool, optional): Whether to list recursively (default: False)
            - include_hidden (bool, optional): Whether to include hidden files (default: False)
    
    Returns:
        Dictionary with:
            - success (bool): Whether operation succeeded
            - items (list): List of file/directory info dicts
            - error (str, optional): Error message if failed
    """
    try:
        path = params.get('path')
        if not path:
            return {
                'success': False,
                'error': 'path parameter is required'
            }
        
        recursive = params.get('recursive', False)
        include_hidden = params.get('include_hidden', False)
        
        dir_path = Path(path)
        if not dir_path.exists():
            return {
                'success': False,
                'error': f'Directory not found: {path}'
            }
        
        if not dir_path.is_dir():
            return {
                'success': False,
                'error': f'Path is not a directory: {path}'
            }
        
        items = []
        
        if recursive:
            iterator = dir_path.rglob('*')
        else:
            iterator = dir_path.iterdir()
        
        for item in iterator:
            # Skip hidden files if not requested
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
                # Skip items we can't access
                continue
        
        # Sort: directories first, then by name
        items.sort(key=lambda x: (x['type'] != 'directory', x['name'].lower()))
        
        return {
            'success': True,
            'items': items,
            'count': len(items)
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error listing directory: {str(e)}'
        }


def create_directory_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a directory (and parent directories if needed).
    
    Args:
        params: Dictionary containing:
            - path (str, required): Directory path to create
            - parents (bool, optional): Create parent directories if needed (default: True)
    
    Returns:
        Dictionary with:
            - success (bool): Whether operation succeeded
            - path (str): Path to created directory
            - error (str, optional): Error message if failed
    """
    try:
        path = params.get('path')
        if not path:
            return {
                'success': False,
                'error': 'path parameter is required'
            }
        
        parents = params.get('parents', True)
        
        dir_path = Path(path)
        
        if dir_path.exists():
            if dir_path.is_dir():
                return {
                    'success': True,
                    'path': str(dir_path),
                    'message': 'Directory already exists'
                }
            else:
                return {
                    'success': False,
                    'error': f'Path exists but is not a directory: {path}'
                }
        
        dir_path.mkdir(parents=parents, exist_ok=True)
        
        return {
            'success': True,
            'path': str(dir_path)
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error creating directory: {str(e)}'
        }


def delete_file_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Delete a file or directory.
    
    Args:
        params: Dictionary containing:
            - path (str, required): Path to file/directory to delete
            - recursive (bool, optional): If True, delete directory recursively (default: False)
    
    Returns:
        Dictionary with:
            - success (bool): Whether operation succeeded
            - path (str): Path to deleted item
            - error (str, optional): Error message if failed
    """
    try:
        path = params.get('path')
        if not path:
            return {
                'success': False,
                'error': 'path parameter is required'
            }
        
        recursive = params.get('recursive', False)
        
        item_path = Path(path)
        
        if not item_path.exists():
            return {
                'success': False,
                'error': f'Path not found: {path}'
            }
        
        if item_path.is_dir():
            if recursive:
                import shutil
                shutil.rmtree(item_path)
            else:
                try:
                    item_path.rmdir()
                except OSError:
                    return {
                        'success': False,
                        'error': 'Directory not empty. Set recursive=True to delete recursively.'
                    }
        else:
            item_path.unlink()
        
        return {
            'success': True,
            'path': str(item_path)
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error deleting: {str(e)}'
        }


def search_files_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search for files matching a pattern.
    
    Args:
        params: Dictionary containing:
            - directory (str, required): Directory to search in
            - pattern (str, required): Filename pattern (supports glob wildcards: *, ?, [])
            - recursive (bool, optional): Search recursively (default: True)
    
    Returns:
        Dictionary with:
            - success (bool): Whether operation succeeded
            - matches (list): List of matching file paths
            - count (int): Number of matches
            - error (str, optional): Error message if failed
    """
    try:
        directory = params.get('directory')
        pattern = params.get('pattern')
        
        if not directory:
            return {
                'success': False,
                'error': 'directory parameter is required'
            }
        
        if not pattern:
            return {
                'success': False,
                'error': 'pattern parameter is required'
            }
        
        recursive = params.get('recursive', True)
        
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            return {
                'success': False,
                'error': f'Invalid directory: {directory}'
            }
        
        if recursive:
            search_path = dir_path / '**' / pattern
            matches = [str(p) for p in glob.glob(str(search_path), recursive=True) if p.is_file()]
        else:
            search_path = dir_path / pattern
            matches = [str(p) for p in glob.glob(str(search_path)) if p.is_file()]
        
        return {
            'success': True,
            'matches': matches,
            'count': len(matches)
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error searching files: {str(e)}'
        }


def get_file_info_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get metadata about a file or directory.
    
    Args:
        params: Dictionary containing:
            - path (str, required): Path to file/directory
    
    Returns:
        Dictionary with:
            - success (bool): Whether operation succeeded
            - exists (bool): Whether file exists
            - type (str): 'file' or 'directory'
            - size (int): Size in bytes (for files)
            - modified (str): Last modified timestamp (ISO format)
            - error (str, optional): Error message if failed
    """
    try:
        path = params.get('path')
        if not path:
            return {
                'success': False,
                'error': 'path parameter is required'
            }
        
        item_path = Path(path)
        
        if not item_path.exists():
            return {
                'success': True,
                'exists': False
            }
        
        stat = item_path.stat()
        
        return {
            'success': True,
            'exists': True,
            'type': 'directory' if item_path.is_dir() else 'file',
            'size': stat.st_size if item_path.is_file() else None,
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'path': str(item_path)
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error getting file info: {str(e)}'
        }
