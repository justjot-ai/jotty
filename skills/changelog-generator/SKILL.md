# Changelog Generator Skill

Automatically creates user-facing changelogs from git commits by analyzing commit history, categorizing changes, and transforming technical commits into clear, customer-friendly release notes.

## Description

This skill transforms technical git commits into polished, user-friendly changelogs that your customers and users will actually understand and appreciate. It saves hours of manual changelog writing by automating the process.

## Tools

### `generate_changelog_tool`

Generate a changelog from git commit history.

**Parameters:**
- `repo_path` (str, optional): Path to git repository (default: current directory)
- `since` (str, optional): Start date/commit/tag (e.g., "2024-01-01", "v2.4.0", "last-release")
- `until` (str, optional): End date/commit/tag (default: HEAD)
- `version` (str, optional): Version number for this release (e.g., "2.5.0")
- `output_file` (str, optional): Path to save changelog (default: CHANGELOG.md)
- `style_guide` (str, optional): Path to custom style guide file
- `exclude_patterns` (list, optional): Patterns to exclude (e.g., ["refactor", "test", "chore"])

**Returns:**
- `success` (bool): Whether generation succeeded
- `changelog` (str): Generated changelog content
- `output_path` (str): Path where changelog was saved
- `stats` (dict): Statistics about commits analyzed
- `error` (str, optional): Error message if failed

## Usage Examples

### Basic Usage

```python
result = await generate_changelog_tool({
    'since': 'last-release',
    'version': '2.5.0'
})
```

### With Date Range

```python
result = await generate_changelog_tool({
    'since': '2024-01-01',
    'until': '2024-01-31',
    'version': '2.5.0'
})
```

### Custom Style Guide

```python
result = await generate_changelog_tool({
    'since': 'v2.4.0',
    'style_guide': 'CHANGELOG_STYLE.md',
    'exclude_patterns': ['refactor', 'test', 'chore']
})
```

## Dependencies

- `gitpython`: Git repository access
- `claude-cli-llm`: AI-powered commit translation
