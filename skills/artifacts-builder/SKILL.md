---
name: artifacts-builder
description: "This skill helps create complex HTML artifacts for Claude.ai using modern frontend technologies. Supports multi-component artifacts requiring state management, routing, or shadcn/ui components. Use when the user wants to create."
---

# Artifacts Builder Skill

Creates elaborate HTML artifacts using React, TypeScript, Tailwind CSS, and shadcn/ui.

## Description

This skill helps create complex HTML artifacts for Claude.ai using modern frontend technologies. Supports multi-component artifacts requiring state management, routing, or shadcn/ui components.


## Type
derived

## Base Skills
- file-operations


## Capabilities
- code
- document

## Tools

### `init_artifact_project_tool`

Initialize a new artifact project.

**Parameters:**
- `project_name` (str, required): Name of the artifact project
- `output_directory` (str, optional): Output directory (default: current directory)
- `include_shadcn` (bool, optional): Include shadcn/ui components (default: True)
- `include_tailwind` (bool, optional): Include Tailwind CSS (default: True)

**Returns:**
- `success` (bool): Whether initialization succeeded
- `project_path` (str): Path to created project
- `files_created` (list): List of files created
- `next_steps` (list): Instructions for next steps
- `error` (str, optional): Error message if failed

### `bundle_artifact_tool`

Bundle artifact into single HTML file.

**Parameters:**
- `project_path` (str, required): Path to artifact project
- `output_file` (str, optional): Output HTML file path (default: bundle.html)

**Returns:**
- `success` (bool): Whether bundling succeeded
- `bundle_path` (str): Path to bundled HTML file
- `file_size` (int): Size of bundled file in bytes
- `error` (str, optional): Error message if failed

## Usage Examples

### Initialize Project

```python
result = await init_artifact_project_tool({
    'project_name': 'my-artifact',
    'include_shadcn': True
})
```

### Bundle Artifact

```python
result = await bundle_artifact_tool({
    'project_path': 'my-artifact',
    'output_file': 'artifact.html'
})
```

## Dependencies

- `shell-exec`: For running npm/node commands
- `file-operations`: For file operations

## Stack

- React 18 + TypeScript
- Vite (development)
- Parcel (bundling)
- Tailwind CSS
- shadcn/ui components

## Triggers
- "artifacts builder"
- "create"

## Category
media-creation
