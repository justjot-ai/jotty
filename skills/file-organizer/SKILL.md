# File Organizer Skill

Intelligently organizes files and folders by understanding context, finding duplicates, and suggesting better organizational structures.

## Description

This skill acts as your personal organization assistant, helping you maintain a clean, logical file structure across your computer without the mental overhead of constant manual organization.

## Tools

### `organize_files_tool`

Organize files and folders intelligently.

**Parameters:**
- `target_directory` (str, required): Directory to organize
- `organization_strategy` (str, optional): Strategy - 'by_type', 'by_date', 'by_purpose', 'auto' (default: 'auto')
- `find_duplicates` (bool, optional): Find and handle duplicates (default: True)
- `archive_old_files` (bool, optional): Archive files older than threshold (default: False)
- `age_threshold_days` (int, optional): Days threshold for archiving (default: 180)
- `dry_run` (bool, optional): Preview changes without executing (default: False)
- `output_report` (str, optional): Path to save organization report

**Returns:**
- `success` (bool): Whether organization succeeded
- `changes` (list): List of changes made (moves, renames, deletions)
- `duplicates_found` (list): List of duplicate file sets
- `statistics` (dict): Organization statistics
- `report_path` (str, optional): Path to saved report
- `error` (str, optional): Error message if failed

## Usage Examples

### Basic Usage

```python
result = await organize_files_tool({
    'target_directory': '~/Downloads',
    'organization_strategy': 'by_type'
})
```

### With Duplicate Detection

```python
result = await organize_files_tool({
    'target_directory': '~/Documents',
    'find_duplicates': True,
    'dry_run': True  # Preview first
})
```

## Dependencies

- `file-operations`: For file operations
- `claude-cli-llm`: For intelligent categorization
