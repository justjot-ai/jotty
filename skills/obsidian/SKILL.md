# Obsidian Skill

## Description
Provides file-based tools for interacting with Obsidian markdown vaults. Supports listing, reading, creating, updating, and searching notes, as well as finding backlinks between notes.


## Type
base

## Configuration
Set the default vault path via environment variable:
```bash
export OBSIDIAN_VAULT=/path/to/your/vault
```

If not set, defaults to `~/.obsidian-vault`.

## Tools

### list_notes_tool
Lists notes in an Obsidian vault recursively.

**Parameters:**
- `vault_path` (str, optional): Path to Obsidian vault (default: env OBSIDIAN_VAULT or ~/.obsidian-vault)
- `folder` (str, optional): Subfolder to list (relative to vault)
- `pattern` (str, optional): Glob pattern to filter notes (default: '*.md')

**Returns:**
- `success` (bool): Whether operation succeeded
- `notes` (list): List of note info dicts with keys: name, filename, path, relative_path, size, modified
- `count` (int): Number of notes found
- `vault_path` (str): The vault path used
- `error` (str, optional): Error message if failed

### read_note_tool
Reads the content of an Obsidian note.

**Parameters:**
- `vault_path` (str, optional): Path to Obsidian vault (default: env OBSIDIAN_VAULT or ~/.obsidian-vault)
- `note_path` (str, required): Path to the note (relative to vault, e.g., 'folder/note.md' or just 'note')

**Returns:**
- `success` (bool): Whether operation succeeded
- `content` (str): Note contents
- `frontmatter` (dict, optional): Parsed YAML frontmatter if present
- `links` (list): Wiki-style links found in the note
- `path` (str): Full path to note
- `relative_path` (str): Path relative to vault
- `size` (int): Content size in bytes
- `modified` (str): Last modified timestamp (ISO format)
- `error` (str, optional): Error message if failed

### create_note_tool
Creates a new note in the Obsidian vault.

**Parameters:**
- `vault_path` (str, optional): Path to Obsidian vault (default: env OBSIDIAN_VAULT or ~/.obsidian-vault)
- `note_path` (str, required): Path for the new note (relative to vault, e.g., 'folder/note.md' or just 'note')
- `content` (str, required): Content for the note
- `template` (str, optional): Template name or path to use (looks in Templates/, templates/, or vault root)

**Returns:**
- `success` (bool): Whether operation succeeded
- `path` (str): Full path to created note
- `relative_path` (str): Path relative to vault
- `bytes_written` (int): Number of bytes written
- `error` (str, optional): Error message if failed

### update_note_tool
Updates an existing note in the Obsidian vault.

**Parameters:**
- `vault_path` (str, optional): Path to Obsidian vault (default: env OBSIDIAN_VAULT or ~/.obsidian-vault)
- `note_path` (str, required): Path to the note (relative to vault)
- `content` (str, optional): New content to replace entire note
- `append` (str, optional): Content to append to the note
- `prepend` (str, optional): Content to prepend to the note (after frontmatter if present)

**Returns:**
- `success` (bool): Whether operation succeeded
- `path` (str): Full path to updated note
- `relative_path` (str): Path relative to vault
- `bytes_written` (int): Number of bytes written
- `error` (str, optional): Error message if failed

### search_notes_tool
Searches notes in the vault by content (case-insensitive).

**Parameters:**
- `vault_path` (str, optional): Path to Obsidian vault (default: env OBSIDIAN_VAULT or ~/.obsidian-vault)
- `query` (str, required): Search query (case-insensitive substring match)
- `folder` (str, optional): Limit search to a specific folder
- `include_content` (bool, optional): Include matching content snippets (default: True)
- `max_results` (int, optional): Maximum number of results (default: 50)

**Returns:**
- `success` (bool): Whether operation succeeded
- `results` (list): List of matching notes with keys: name, filename, path, relative_path, matches (list of snippets)
- `count` (int): Number of matching notes
- `query` (str): The search query used
- `vault_path` (str): The vault path used
- `error` (str, optional): Error message if failed

### get_backlinks_tool
Finds notes that link to a specific note (backlinks).

**Parameters:**
- `vault_path` (str, optional): Path to Obsidian vault (default: env OBSIDIAN_VAULT or ~/.obsidian-vault)
- `note_path` (str, required): Path to the target note (relative to vault)

**Returns:**
- `success` (bool): Whether operation succeeded
- `backlinks` (list): List of notes linking to target with keys: name, filename, path, relative_path, link_contexts
- `count` (int): Number of backlinks found
- `target_note` (str): The target note name searched for
- `vault_path` (str): The vault path used
- `error` (str, optional): Error message if failed

## Usage Examples

### List all notes in vault
```python
list_notes_tool({})
```

### List notes in a specific folder
```python
list_notes_tool({'folder': 'Daily Notes'})
```

### Read a note
```python
read_note_tool({'note_path': 'Projects/my-project'})
```

### Create a note with template
```python
create_note_tool({
    'note_path': 'Daily Notes/2024-01-15',
    'content': '## Tasks\n- [ ] First task',
    'template': 'daily-template'
})
```

### Update a note by appending
```python
update_note_tool({
    'note_path': 'Projects/my-project',
    'append': '\n## New Section\nAdditional content'
})
```

### Search notes
```python
search_notes_tool({
    'query': 'python programming',
    'folder': 'Notes'
})
```

### Find backlinks
```python
get_backlinks_tool({
    'note_path': 'Concepts/machine-learning'
})
```
