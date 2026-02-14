---
name: managing-files
description: "Provides essential file system operations: read, write, list directories, create/delete files and directories, search files, and get file metadata. Use when the user wants to read file, write file, create file."
---

# File Operations Skill

## Description
Provides essential file system operations: read, write, list directories, create/delete files and directories, search files, and get file metadata.


## Type
base


## Capabilities
- file-ops


## Reference

For detailed tool documentation, see [REFERENCE.md](REFERENCE.md).

## Workflow

```
Task Progress:
- [ ] Step 1: Locate files
- [ ] Step 2: Read and inspect
- [ ] Step 3: Process and write
- [ ] Step 4: Verify results
```

**Step 1: Locate files**
Search and list files in the target directory.

**Step 2: Read and inspect**
Read file contents and check file metadata.

**Step 3: Process and write**
Create directories, write files, or modify content.

**Step 4: Verify results**
Confirm file operations completed successfully.

## Triggers
- "read file"
- "write file"
- "create file"
- "delete file"
- "list files"

## Category
workflow-automation

## Tools

### read_file_tool
Reads the contents of a file.

**Parameters:**
- `path` (str, required): Path to the file to read
- `encoding` (str, optional): File encoding (default: 'utf-8')

**Returns:**
- `success` (bool): Whether operation succeeded
- `content` (str): File contents
- `error` (str, optional): Error message if failed

### write_file_tool
Writes content to a file.

**Parameters:**
- `path` (str, required): Path to the file to write
- `content` (str, required): Content to write
- `encoding` (str, optional): File encoding (default: 'utf-8')
- `mode` (str, optional): Write mode - 'w' (overwrite) or 'a' (append), default: 'w'

**Returns:**
- `success` (bool): Whether operation succeeded
- `path` (str): Path to written file
- `bytes_written` (int): Number of bytes written
- `error` (str, optional): Error message if failed

### list_directory_tool
Lists files and directories in a path.

**Parameters:**
- `path` (str, required): Directory path to list
- `recursive` (bool, optional): Whether to list recursively (default: False)
- `include_hidden` (bool, optional): Whether to include hidden files (default: False)

**Returns:**
- `success` (bool): Whether operation succeeded
- `items` (list): List of file/directory info dicts with keys: name, path, type, size, modified
- `error` (str, optional): Error message if failed

### create_directory_tool
Creates a directory (and parent directories if needed).

**Parameters:**
- `path` (str, required): Directory path to create
- `parents` (bool, optional): Create parent directories if needed (default: True)

**Returns:**
- `success` (bool): Whether operation succeeded
- `path` (str): Path to created directory
- `error` (str, optional): Error message if failed

### delete_file_tool
Deletes a file or directory.

**Parameters:**
- `path` (str, required): Path to file/directory to delete
- `recursive` (bool, optional): If True, delete directory recursively (default: False)

**Returns:**
- `success` (bool): Whether operation succeeded
- `path` (str): Path to deleted item
- `error` (str, optional): Error message if failed

### search_files_tool
Searches for files matching a pattern.

**Parameters:**
- `directory` (str, required): Directory to search in
- `pattern` (str, required): Filename pattern (supports glob wildcards: *, ?, [])
- `recursive` (bool, optional): Search recursively (default: True)

**Returns:**
- `success` (bool): Whether operation succeeded
- `matches` (list): List of matching file paths
- `count` (int): Number of matches
- `error` (str, optional): Error message if failed

### get_file_info_tool
Gets metadata about a file or directory.

**Parameters:**
- `path` (str, required): Path to file/directory

**Returns:**
- `success` (bool): Whether operation succeeded
- `exists` (bool): Whether file exists
- `type` (str): 'file' or 'directory'
- `size` (int): Size in bytes (for files)
- `modified` (str): Last modified timestamp (ISO format)
- `error` (str, optional): Error message if failed
