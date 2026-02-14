# File Operations Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`read_file_tool`](#read_file_tool) | Read the contents of a file. |
| [`write_file_tool`](#write_file_tool) | Write content to a file. |
| [`list_directory_tool`](#list_directory_tool) | List files and directories in a path. |
| [`create_directory_tool`](#create_directory_tool) | Create a directory (and parent directories if needed). |
| [`delete_file_tool`](#delete_file_tool) | Delete a file or directory. |
| [`search_files_tool`](#search_files_tool) | Search for files matching a pattern. |
| [`get_file_info_tool`](#get_file_info_tool) | Get metadata about a file or directory. |

---

## `read_file_tool`

Read the contents of a file.

**Parameters:**

- **path** (`str, required`): Path to the file
- **encoding** (`str, optional`): File encoding (default: 'utf-8')

**Returns:** Dictionary with success, content, path, size

---

## `write_file_tool`

Write content to a file.

**Parameters:**

- **path** (`str, required`): Path to the file
- **content** (`str, required`): Content to write
- **encoding** (`str, optional`): File encoding (default: 'utf-8')
- **mode** (`str, optional`): 'w' (overwrite) or 'a' (append), default: 'w'

**Returns:** Dictionary with success, path, bytes_written

---

## `list_directory_tool`

List files and directories in a path.

**Parameters:**

- **path** (`str, required`): Directory path to list
- **recursive** (`bool, optional`): List recursively (default: False)
- **include_hidden** (`bool, optional`): Include hidden files (default: False)

**Returns:** Dictionary with success, items, count

---

## `create_directory_tool`

Create a directory (and parent directories if needed).

**Parameters:**

- **path** (`str, required`): Directory path to create
- **parents** (`bool, optional`): Create parent directories (default: True)

**Returns:** Dictionary with success, path

---

## `delete_file_tool`

Delete a file or directory.

**Parameters:**

- **path** (`str, required`): Path to file/directory to delete
- **recursive** (`bool, optional`): Delete directory recursively (default: False)

**Returns:** Dictionary with success, path

---

## `search_files_tool`

Search for files matching a pattern.

**Parameters:**

- **directory** (`str, required`): Directory to search in
- **pattern** (`str, required`): Filename pattern (glob wildcards: *, ?, [])
- **recursive** (`bool, optional`): Search recursively (default: True)

**Returns:** Dictionary with success, matches, count

---

## `get_file_info_tool`

Get metadata about a file or directory.

**Parameters:**

- **path** (`str, required`): Path to file/directory

**Returns:** Dictionary with success, exists, type, size, modified, path
