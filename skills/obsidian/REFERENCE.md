# Obsidian Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`list_notes_tool`](#list_notes_tool) | List notes in an Obsidian vault. |
| [`read_note_tool`](#read_note_tool) | Read the content of an Obsidian note. |
| [`create_note_tool`](#create_note_tool) | Create a new note in the Obsidian vault. |
| [`update_note_tool`](#update_note_tool) | Update an existing note in the Obsidian vault. |
| [`search_notes_tool`](#search_notes_tool) | Search notes in the vault by content. |
| [`get_backlinks_tool`](#get_backlinks_tool) | Find notes that link to a specific note (backlinks). |

---

## `list_notes_tool`

List notes in an Obsidian vault.

**Parameters:**

- **vault_path** (`str, optional`): Path to Obsidian vault (default: env OBSIDIAN_VAULT or ~/.obsidian-vault)
- **folder** (`str, optional`): Subfolder to list (relative to vault)
- **pattern** (`str, optional`): Glob pattern to filter notes (default: '*.md')

**Returns:** Dictionary with: - success (bool): Whether operation succeeded - notes (list): List of note info dicts with keys: name, path, relative_path, size, modified - count (int): Number of notes found - error (str, optional): Error message if failed

---

## `read_note_tool`

Read the content of an Obsidian note.

**Parameters:**

- **vault_path** (`str, optional`): Path to Obsidian vault (default: env OBSIDIAN_VAULT or ~/.obsidian-vault)
- **note_path** (`str, required`): Path to the note (relative to vault, e.g., 'folder/note.md' or just 'note')

**Returns:** Dictionary with: - success (bool): Whether operation succeeded - content (str): Note contents - frontmatter (dict, optional): Parsed YAML frontmatter if present - links (list): Wiki-style links found in the note - error (str, optional): Error message if failed

---

## `create_note_tool`

Create a new note in the Obsidian vault.

**Parameters:**

- **vault_path** (`str, optional`): Path to Obsidian vault (default: env OBSIDIAN_VAULT or ~/.obsidian-vault)
- **note_path** (`str, required`): Path for the new note (relative to vault, e.g., 'folder/note.md' or just 'note')
- **content** (`str, required`): Content for the note
- **template** (`str, optional`): Template name or path to use (note content will be appended after template)

**Returns:** Dictionary with: - success (bool): Whether operation succeeded - path (str): Full path to created note - relative_path (str): Path relative to vault - error (str, optional): Error message if failed

---

## `update_note_tool`

Update an existing note in the Obsidian vault.

**Parameters:**

- **vault_path** (`str, optional`): Path to Obsidian vault (default: env OBSIDIAN_VAULT or ~/.obsidian-vault)
- **note_path** (`str, required`): Path to the note (relative to vault)
- **content** (`str, optional`): New content to replace entire note
- **append** (`str, optional`): Content to append to the note
- **prepend** (`str, optional`): Content to prepend to the note (after frontmatter if present)

**Returns:** Dictionary with: - success (bool): Whether operation succeeded - path (str): Full path to updated note - bytes_written (int): Number of bytes written - error (str, optional): Error message if failed

---

## `search_notes_tool`

Search notes in the vault by content.

**Parameters:**

- **vault_path** (`str, optional`): Path to Obsidian vault (default: env OBSIDIAN_VAULT or ~/.obsidian-vault)
- **query** (`str, required`): Search query (case-insensitive substring match)
- **folder** (`str, optional`): Limit search to a specific folder
- **include_content** (`bool, optional`): Include matching content snippets (default: True)
- **max_results** (`int, optional`): Maximum number of results (default: 50)

**Returns:** Dictionary with: - success (bool): Whether operation succeeded - results (list): List of matching notes with keys: name, path, relative_path, matches (list of snippets) - count (int): Number of matching notes - error (str, optional): Error message if failed

---

## `get_backlinks_tool`

Find notes that link to a specific note (backlinks).

**Parameters:**

- **vault_path** (`str, optional`): Path to Obsidian vault (default: env OBSIDIAN_VAULT or ~/.obsidian-vault)
- **note_path** (`str, required`): Path to the target note (relative to vault)

**Returns:** Dictionary with: - success (bool): Whether operation succeeded - backlinks (list): List of notes linking to target with keys: name, path, relative_path, link_contexts - count (int): Number of backlinks found - error (str, optional): Error message if failed
