---
name: claude-api-llm
description: "This skill uses the Anthropic Messages API with `tool_use` for structured outputs. Unlike claude-cli-llm (DSPy/text-based), responses are clean JSON with zero preamble, zero code fences, and zero template collisions. Preferred when ANTHROPIC_API_KEY is available. Key advantages: - **generate_code_tool**: Forced tool_use returns clean code — no fences, no preamble - **agentic_generate_tool**: Multi-step tool loop (generate + write + execute) in one call - **structured_output_tool**: JSON schema-constrained output via tool_use - **generate_text_tool**: Drop-in replacement for claude-cli-llm's text generation."
---

# Claude API LLM Skill

Structured code generation and text output using Anthropic API `tool_use`.
Eliminates preamble/fence issues by forcing structured JSON responses.

## Description

This skill uses the Anthropic Messages API with `tool_use` for structured outputs.
Unlike claude-cli-llm (DSPy/text-based), responses are clean JSON with zero preamble,
zero code fences, and zero template collisions. Preferred when ANTHROPIC_API_KEY is available.

Key advantages:
- **generate_code_tool**: Forced tool_use returns clean code — no fences, no preamble
- **agentic_generate_tool**: Multi-step tool loop (generate + write + execute) in one call
- **structured_output_tool**: JSON schema-constrained output via tool_use
- **generate_text_tool**: Drop-in replacement for claude-cli-llm's text generation

## Type
base

## Capabilities
- analyze
- code
- generate

## Parameters

### generate_code_tool
- `prompt` (str, required): Code generation prompt
- `language` (str, optional): Target language (default: "python")
- `filename` (str, optional): Suggested filename

### generate_text_tool
- `prompt` (str, required): Text generation prompt
- `max_tokens` (int, optional): Maximum tokens (default: 4096)

### agentic_generate_tool
- `prompt` (str, required): Task prompt for agentic execution
- `tools` (list, optional): Tools to enable (default: all — write_file, execute_command, read_file, edit_file, search_replace)
- `working_directory` (str, optional): Working directory (default: "/tmp")
- `max_tool_rounds` (int, optional): Max tool-use rounds (default: 5)
- `stream` (bool, optional): Enable token-level streaming with status callbacks (default: false)
- `sandbox_level` (str, optional): Execution trust level — "trusted", "sandboxed", or "dangerous" (default: "sandboxed")
- `include_context` (bool, optional): Auto-discover project files for system prompt context (default: true)

### structured_output_tool
- `prompt` (str, required): Prompt for structured output
- `schema` (dict, optional): JSON schema for output structure

## Returns

### generate_code_tool
- `success` (bool): Whether generation succeeded
- `code` (str): Clean generated code (no fences, no preamble)
- `language` (str): Language of the code
- `filename` (str): Suggested filename
- `lint_passed` (bool): Whether code passed syntax validation

### generate_text_tool
- `success` (bool): Whether generation succeeded
- `text` (str): Generated text
- `model` (str): Model used
- `provider` (str): Always "anthropic-api"

### agentic_generate_tool
- `success` (bool): Whether execution succeeded
- `response` (str): Final response text
- `tool_calls` (list): Tool call history
- `files_created` (list): Paths of files created
- `execution_output` (str): Combined execution output

### structured_output_tool
- `success` (bool): Whether generation succeeded
- `data` (dict): Parsed JSON conforming to schema
- `model` (str): Model used

## Version
1.1.0

## Author
Jotty Framework

## Triggers
- "claude api llm"

## Category
workflow-automation
