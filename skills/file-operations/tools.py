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

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper

from Jotty.core.infrastructure.utils.skill_status import SkillStatus

# Status emitter for progress updates
status = SkillStatus("file-operations")



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
    status.set_callback(params.pop('_status_callback', None))

    encoding = params.get('encoding', 'utf-8')
    file_path = Path(params['path'])

    if not file_path.exists():
        return tool_error(f'File not found: {params["path"]}')

    if not file_path.is_file():
        return tool_error(f'Path is not a file: {params["path"]}')

    try:
        status.emit("Reading", f"📄 Reading {file_path.name}...")
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
    status.set_callback(params.pop('_status_callback', None))

    encoding = params.get('encoding', 'utf-8')
    file_path = Path(params['path'])
    content = params['content']

    # Guard: detect when 'content' is accidentally a tool result JSON
    # instead of actual file content (happens when resolve_params picks
    # the wrong step output). e.g., writing '{"success": true, "path": ...}'
    # to a .md file is clearly wrong.
    _content_stripped = content.strip()
    if (len(_content_stripped) < 200
            and _content_stripped.startswith('{"success"')
            and '"bytes_written"' in _content_stripped):
        return tool_error(
            f'Content appears to be a tool result JSON, not actual file content. '
            f'The step that generates the content may have resolved incorrectly.',
            path=str(file_path)
        )

    # ---------------------------------------------------------------
    # Smart content cleaning: strip markdown fences and LLM preamble
    # when writing code/data files. The LLM (claude-cli-llm) often
    # wraps code in ```python...``` blocks with conversational text
    # before them.  This produces invalid .py/.json/.csv files.
    # ---------------------------------------------------------------
    import re as _re
    _ext = file_path.suffix.lower()
    _code_exts = {'.py', '.js', '.ts', '.java', '.c', '.cpp', '.go', '.rs',
                  '.rb', '.php', '.sh', '.bash', '.sql', '.r', '.swift',
                  '.kt', '.cs', '.yaml', '.yml', '.toml', '.ini', '.cfg'}
    _data_exts = {'.json', '.csv', '.xml', '.html', '.css', '.txt'}
    _needs_clean = _ext in (_code_exts | _data_exts)

    if _needs_clean and '```' in content:
        # Extract content from markdown code fences
        # Match ```lang\n...\n``` or ```\n...\n```
        _fence_pattern = _re.compile(
            r'```(?:\w+)?\s*\n(.*?)```',
            _re.DOTALL,
        )
        _matches = _fence_pattern.findall(content)
        if _matches:
            if len(_matches) == 1:
                # Single code block — use it directly
                content = _matches[0].strip()
            else:
                # Multiple code blocks in same file
                # (e.g., LLM generated calculator.py + test_calculator.py in one response)
                # Use the FIRST block that looks like it belongs to this file
                _stem = file_path.stem.lower()
                _best = None
                for _m in _matches:
                    _m_lower = _m.lower()
                    # If the block mentions this file's name/class/function
                    if _stem in _m_lower or _stem.replace('_', '') in _m_lower:
                        _best = _m.strip()
                        break
                if _best is None:
                    # Fallback: use the first block
                    _best = _matches[0].strip()
                content = _best

            status.emit("Cleaned", f"🧹 Stripped markdown fences from {file_path.name}")

    # For .json files, also strip any LLM preamble before the JSON
    if _ext == '.json' and content.strip() and not content.strip()[0] in ('{', '['):
        # Try to find where the JSON actually starts
        _json_start = None
        for _i, _ch in enumerate(content):
            if _ch in ('{', '['):
                _json_start = _i
                break
        if _json_start is not None:
            _candidate = content[_json_start:]
            import json as _json
            try:
                _json.loads(_candidate)
                content = _candidate
                status.emit("Cleaned", f"🧹 Stripped LLM preamble from {file_path.name}")
            except ValueError:
                pass  # Not valid JSON after stripping — keep original

    status.emit("Writing", f"📝 Writing {file_path.name}...")
    # Guard: if path exists as a directory (created by mistake by create_directory_tool),
    # remove the empty directory so we can write the file
    if file_path.is_dir():
        try:
            file_path.rmdir()  # Only removes empty dirs — safe
        except OSError:
            return tool_error(
                f'Path is a non-empty directory: {params["path"]}. '
                f'Cannot overwrite a directory with a file.',
                path=str(file_path)
            )
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding=encoding)

    return tool_response(
        path=str(file_path),
        bytes_written=len(content.encode(encoding))
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
    status.set_callback(params.pop('_status_callback', None))

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
    status.set_callback(params.pop('_status_callback', None))

    parents = params.get('parents', True)
    dir_path = Path(params['path'])

    # Guard: if path looks like a file (has extension), create parent dir instead.
    # LLM planners often pass the full file path to create_directory when they
    # mean "ensure the parent directory exists for this file".
    _file_exts = {'.md', '.txt', '.py', '.js', '.ts', '.json', '.csv', '.html',
                  '.xml', '.yaml', '.yml', '.pdf', '.epub', '.docx', '.xlsx',
                  '.toml', '.ini', '.cfg', '.sh', '.bash', '.sql', '.css',
                  '.java', '.c', '.cpp', '.go', '.rs', '.rb', '.php', '.swift'}
    if dir_path.suffix.lower() in _file_exts:
        # This is a file path, not a directory — create the parent instead
        dir_path = dir_path.parent

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
    status.set_callback(params.pop('_status_callback', None))

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
    status.set_callback(params.pop('_status_callback', None))

    recursive = params.get('recursive', True)
    dir_path = Path(params['directory'])

    if not dir_path.exists() or not dir_path.is_dir():
        return tool_error(f'Invalid directory: {params["directory"]}')

    status.emit("Searching", f"🔍 Searching for {params['pattern']}...")
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
    status.set_callback(params.pop('_status_callback', None))

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
