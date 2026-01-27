# MCP Builder Skill

Helps create and validate MCP (Model Context Protocol) servers with best practices and evaluation tools.

## Description

This skill assists in building MCP servers by providing templates, validation, and best practices. Helps create Python and Node.js MCP servers following the Model Context Protocol specification.

## Tools

### `create_mcp_server_tool`

Create a new MCP server project.

**Parameters:**
- `server_name` (str, required): Name of the MCP server
- `language` (str, optional): Language - 'python' or 'node' (default: 'python')
- `output_directory` (str, optional): Output directory (default: current directory)
- `include_examples` (bool, optional): Include example tools (default: True)

**Returns:**
- `success` (bool): Whether creation succeeded
- `server_path` (str): Path to created server
- `files_created` (list): List of files created
- `error` (str, optional): Error message if failed

### `validate_mcp_server_tool`

Validate an MCP server structure.

**Parameters:**
- `server_path` (str, required): Path to MCP server directory

**Returns:**
- `success` (bool): Whether validation succeeded
- `valid` (bool): Whether server is valid
- `issues` (list): List of validation issues
- `warnings` (list): List of warnings

## Usage Examples

### Create Python MCP Server

```python
result = await create_mcp_server_tool({
    'server_name': 'my-mcp-server',
    'language': 'python',
    'include_examples': True
})
```

### Validate Server

```python
result = await validate_mcp_server_tool({
    'server_path': 'my-mcp-server'
})
```

## Dependencies

- `file-operations`: For creating files
