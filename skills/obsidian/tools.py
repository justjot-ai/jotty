import os
import re
import glob
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from Jotty.core.utils.skill_status import SkillStatus
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper

# Status emitter for progress updates
status = SkillStatus("obsidian")



def _get_vault_path(params: Dict[str, Any]) -> str:
    """
    Get the vault path from params or environment variable.

    Args:
        params: Dictionary containing optional vault_path

    Returns:
        Resolved vault path string
    """
    vault_path = params.get('vault_path')
    if not vault_path:
        vault_path = os.environ.get('OBSIDIAN_VAULT', os.path.expanduser('~/.obsidian-vault'))
    return os.path.expanduser(vault_path)


def _validate_vault(vault_path: str) -> Optional[Dict[str, Any]]:
    """
    Validate that the vault path exists and is a directory.

    Args:
        vault_path: Path to validate

    Returns:
        Error dictionary if invalid, None if valid
    """
    if not os.path.exists(vault_path):
        return {
            'success': False,
            'error': f'Vault not found: {vault_path}'
        }
    if not os.path.isdir(vault_path):
        return {
            'success': False,
            'error': f'Vault path is not a directory: {vault_path}'
        }
    return None


def _is_markdown_file(path: Path) -> bool:
    """Check if a file is a markdown file."""
    return path.suffix.lower() in ['.md', '.markdown']


def _extract_links(content: str) -> List[str]:
    """
    Extract wiki-style links from markdown content.

    Args:
        content: Markdown content to parse

    Returns:
        List of linked note names (without brackets)
    """
    # Match [[link]] and [[link|alias]] patterns
    wiki_links = re.findall(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', content)
    return wiki_links


@tool_wrapper()
def list_notes_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List notes in an Obsidian vault.

    Args:
        params: Dictionary containing:
            - vault_path (str, optional): Path to Obsidian vault (default: env OBSIDIAN_VAULT or ~/.obsidian-vault)
            - folder (str, optional): Subfolder to list (relative to vault)
            - pattern (str, optional): Glob pattern to filter notes (default: '*.md')

    Returns:
        Dictionary with:
            - success (bool): Whether operation succeeded
            - notes (list): List of note info dicts with keys: name, path, relative_path, size, modified
            - count (int): Number of notes found
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        vault_path = _get_vault_path(params)
        error = _validate_vault(vault_path)
        if error:
            return error

        folder = params.get('folder', '')
        pattern = params.get('pattern', '*.md')

        search_path = Path(vault_path)
        if folder:
            search_path = search_path / folder
            if not search_path.exists():
                return {
                    'success': False,
                    'error': f'Folder not found: {folder}'
                }

        # Search recursively for markdown files
        search_pattern = search_path / '**' / pattern
        notes = []

        for file_path in glob.glob(str(search_pattern), recursive=True):
            path = Path(file_path)
            if path.is_file() and not path.name.startswith('.'):
                try:
                    stat = path.stat()
                    relative_path = path.relative_to(vault_path)
                    notes.append({
                        'name': path.stem,
                        'filename': path.name,
                        'path': str(path),
                        'relative_path': str(relative_path),
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
                except (OSError, PermissionError):
                    continue

        # Sort by name
        notes.sort(key=lambda x: x['name'].lower())

        return {
            'success': True,
            'notes': notes,
            'count': len(notes),
            'vault_path': vault_path
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error listing notes: {str(e)}'
        }


@tool_wrapper()
def read_note_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read the content of an Obsidian note.

    Args:
        params: Dictionary containing:
            - vault_path (str, optional): Path to Obsidian vault (default: env OBSIDIAN_VAULT or ~/.obsidian-vault)
            - note_path (str, required): Path to the note (relative to vault, e.g., 'folder/note.md' or just 'note')

    Returns:
        Dictionary with:
            - success (bool): Whether operation succeeded
            - content (str): Note contents
            - frontmatter (dict, optional): Parsed YAML frontmatter if present
            - links (list): Wiki-style links found in the note
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        vault_path = _get_vault_path(params)
        error = _validate_vault(vault_path)
        if error:
            return error

        note_path = params.get('note_path')
        if not note_path:
            return {
                'success': False,
                'error': 'note_path parameter is required'
            }

        # Normalize note path - add .md if not present
        if not note_path.endswith('.md') and not note_path.endswith('.markdown'):
            note_path = note_path + '.md'

        full_path = Path(vault_path) / note_path

        if not full_path.exists():
            return {
                'success': False,
                'error': f'Note not found: {note_path}'
            }

        if not full_path.is_file():
            return {
                'success': False,
                'error': f'Path is not a file: {note_path}'
            }

        content = full_path.read_text(encoding='utf-8')

        # Parse frontmatter if present
        frontmatter = None
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                try:
                    import yaml
                    frontmatter = yaml.safe_load(parts[1])
                except:
                    pass  # Ignore frontmatter parsing errors

        # Extract links
        links = _extract_links(content)

        stat = full_path.stat()

        return {
            'success': True,
            'content': content,
            'frontmatter': frontmatter,
            'links': links,
            'path': str(full_path),
            'relative_path': note_path,
            'size': len(content),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
        }
    except UnicodeDecodeError as e:
        return {
            'success': False,
            'error': f'Encoding error reading note: {str(e)}'
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error reading note: {str(e)}'
        }


@tool_wrapper()
def create_note_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new note in the Obsidian vault.

    Args:
        params: Dictionary containing:
            - vault_path (str, optional): Path to Obsidian vault (default: env OBSIDIAN_VAULT or ~/.obsidian-vault)
            - note_path (str, required): Path for the new note (relative to vault, e.g., 'folder/note.md' or just 'note')
            - content (str, required): Content for the note
            - template (str, optional): Template name or path to use (note content will be appended after template)

    Returns:
        Dictionary with:
            - success (bool): Whether operation succeeded
            - path (str): Full path to created note
            - relative_path (str): Path relative to vault
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        vault_path = _get_vault_path(params)
        error = _validate_vault(vault_path)
        if error:
            return error

        note_path = params.get('note_path')
        content = params.get('content')
        template = params.get('template')

        if not note_path:
            return {
                'success': False,
                'error': 'note_path parameter is required'
            }

        if content is None:
            return {
                'success': False,
                'error': 'content parameter is required'
            }

        # Normalize note path - add .md if not present
        if not note_path.endswith('.md') and not note_path.endswith('.markdown'):
            note_path = note_path + '.md'

        full_path = Path(vault_path) / note_path

        if full_path.exists():
            return {
                'success': False,
                'error': f'Note already exists: {note_path}. Use update_note_tool to modify it.'
            }

        # Create parent directories if needed
        full_path.parent.mkdir(parents=True, exist_ok=True)

        final_content = ''

        # Load template if specified
        if template:
            # Try multiple template locations
            template_paths = [
                Path(vault_path) / 'Templates' / template,
                Path(vault_path) / 'templates' / template,
                Path(vault_path) / template,
            ]

            # Add .md extension variations
            expanded_paths = []
            for tp in template_paths:
                expanded_paths.append(tp)
                if not str(tp).endswith('.md'):
                    expanded_paths.append(Path(str(tp) + '.md'))

            template_content = None
            for tp in expanded_paths:
                if tp.exists() and tp.is_file():
                    template_content = tp.read_text(encoding='utf-8')
                    break

            if template_content:
                final_content = template_content + '\n\n'

        final_content += content

        full_path.write_text(final_content, encoding='utf-8')

        return {
            'success': True,
            'path': str(full_path),
            'relative_path': note_path,
            'bytes_written': len(final_content)
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error creating note: {str(e)}'
        }


@tool_wrapper()
def update_note_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update an existing note in the Obsidian vault.

    Args:
        params: Dictionary containing:
            - vault_path (str, optional): Path to Obsidian vault (default: env OBSIDIAN_VAULT or ~/.obsidian-vault)
            - note_path (str, required): Path to the note (relative to vault)
            - content (str, optional): New content to replace entire note
            - append (str, optional): Content to append to the note
            - prepend (str, optional): Content to prepend to the note (after frontmatter if present)

    Returns:
        Dictionary with:
            - success (bool): Whether operation succeeded
            - path (str): Full path to updated note
            - bytes_written (int): Number of bytes written
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        vault_path = _get_vault_path(params)
        error = _validate_vault(vault_path)
        if error:
            return error

        note_path = params.get('note_path')
        content = params.get('content')
        append = params.get('append')
        prepend = params.get('prepend')

        if not note_path:
            return {
                'success': False,
                'error': 'note_path parameter is required'
            }

        if content is None and append is None and prepend is None:
            return {
                'success': False,
                'error': 'Either content, append, or prepend parameter is required'
            }

        # Normalize note path
        if not note_path.endswith('.md') and not note_path.endswith('.markdown'):
            note_path = note_path + '.md'

        full_path = Path(vault_path) / note_path

        if not full_path.exists():
            return {
                'success': False,
                'error': f'Note not found: {note_path}. Use create_note_tool to create it.'
            }

        if content is not None:
            # Replace entire content
            final_content = content
        else:
            # Read existing content
            existing_content = full_path.read_text(encoding='utf-8')

            if append:
                final_content = existing_content + '\n' + append
            elif prepend:
                # Try to preserve frontmatter
                if existing_content.startswith('---'):
                    parts = existing_content.split('---', 2)
                    if len(parts) >= 3:
                        final_content = '---' + parts[1] + '---\n' + prepend + '\n' + parts[2]
                    else:
                        final_content = prepend + '\n' + existing_content
                else:
                    final_content = prepend + '\n' + existing_content
            else:
                final_content = existing_content

        full_path.write_text(final_content, encoding='utf-8')

        return {
            'success': True,
            'path': str(full_path),
            'relative_path': note_path,
            'bytes_written': len(final_content)
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error updating note: {str(e)}'
        }


@tool_wrapper()
def search_notes_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search notes in the vault by content.

    Args:
        params: Dictionary containing:
            - vault_path (str, optional): Path to Obsidian vault (default: env OBSIDIAN_VAULT or ~/.obsidian-vault)
            - query (str, required): Search query (case-insensitive substring match)
            - folder (str, optional): Limit search to a specific folder
            - include_content (bool, optional): Include matching content snippets (default: True)
            - max_results (int, optional): Maximum number of results (default: 50)

    Returns:
        Dictionary with:
            - success (bool): Whether operation succeeded
            - results (list): List of matching notes with keys: name, path, relative_path, matches (list of snippets)
            - count (int): Number of matching notes
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        vault_path = _get_vault_path(params)
        error = _validate_vault(vault_path)
        if error:
            return error

        query = params.get('query')
        if not query:
            return {
                'success': False,
                'error': 'query parameter is required'
            }

        folder = params.get('folder', '')
        include_content = params.get('include_content', True)
        max_results = params.get('max_results', 50)

        search_path = Path(vault_path)
        if folder:
            search_path = search_path / folder
            if not search_path.exists():
                return {
                    'success': False,
                    'error': f'Folder not found: {folder}'
                }

        query_lower = query.lower()
        results = []

        # Search all markdown files
        for file_path in glob.glob(str(search_path / '**' / '*.md'), recursive=True):
            if len(results) >= max_results:
                break

            path = Path(file_path)
            if path.name.startswith('.'):
                continue

            try:
                content = path.read_text(encoding='utf-8')
                if query_lower in content.lower():
                    relative_path = path.relative_to(vault_path)

                    result = {
                        'name': path.stem,
                        'filename': path.name,
                        'path': str(path),
                        'relative_path': str(relative_path)
                    }

                    if include_content:
                        # Extract matching lines with context
                        matches = []
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if query_lower in line.lower():
                                # Get surrounding context
                                start = max(0, i - 1)
                                end = min(len(lines), i + 2)
                                snippet = '\n'.join(lines[start:end])
                                matches.append({
                                    'line_number': i + 1,
                                    'snippet': snippet[:500]  # Limit snippet size
                                })
                                if len(matches) >= 3:  # Limit matches per file
                                    break
                        result['matches'] = matches

                    results.append(result)
            except (OSError, PermissionError, UnicodeDecodeError):
                continue

        return {
            'success': True,
            'results': results,
            'count': len(results),
            'query': query,
            'vault_path': vault_path
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error searching notes: {str(e)}'
        }


@tool_wrapper()
def get_backlinks_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find notes that link to a specific note (backlinks).

    Args:
        params: Dictionary containing:
            - vault_path (str, optional): Path to Obsidian vault (default: env OBSIDIAN_VAULT or ~/.obsidian-vault)
            - note_path (str, required): Path to the target note (relative to vault)

    Returns:
        Dictionary with:
            - success (bool): Whether operation succeeded
            - backlinks (list): List of notes linking to target with keys: name, path, relative_path, link_contexts
            - count (int): Number of backlinks found
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        vault_path = _get_vault_path(params)
        error = _validate_vault(vault_path)
        if error:
            return error

        note_path = params.get('note_path')
        if not note_path:
            return {
                'success': False,
                'error': 'note_path parameter is required'
            }

        # Get the note name to search for (without extension and path)
        note_name = Path(note_path).stem

        # Patterns to search for: [[note_name]] and [[note_name|alias]]
        link_patterns = [
            re.compile(r'\[\[' + re.escape(note_name) + r'(\|[^\]]+)?\]\]', re.IGNORECASE),
            re.compile(r'\[\[' + re.escape(note_path.rstrip('.md')) + r'(\|[^\]]+)?\]\]', re.IGNORECASE)
        ]

        backlinks = []

        # Search all markdown files
        for file_path in glob.glob(str(Path(vault_path) / '**' / '*.md'), recursive=True):
            path = Path(file_path)

            # Skip the target note itself
            if path.stem.lower() == note_name.lower():
                continue

            if path.name.startswith('.'):
                continue

            try:
                content = path.read_text(encoding='utf-8')

                # Check if any pattern matches
                found_contexts = []
                for pattern in link_patterns:
                    for match in pattern.finditer(content):
                        # Get surrounding context
                        start = max(0, match.start() - 50)
                        end = min(len(content), match.end() + 50)
                        context = content[start:end]
                        # Clean up context
                        if start > 0:
                            context = '...' + context
                        if end < len(content):
                            context = context + '...'
                        found_contexts.append(context)

                if found_contexts:
                    relative_path = path.relative_to(vault_path)
                    backlinks.append({
                        'name': path.stem,
                        'filename': path.name,
                        'path': str(path),
                        'relative_path': str(relative_path),
                        'link_contexts': found_contexts[:5]  # Limit contexts
                    })
            except (OSError, PermissionError, UnicodeDecodeError):
                continue

        # Sort by name
        backlinks.sort(key=lambda x: x['name'].lower())

        return {
            'success': True,
            'backlinks': backlinks,
            'count': len(backlinks),
            'target_note': note_name,
            'vault_path': vault_path
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error finding backlinks: {str(e)}'
        }
