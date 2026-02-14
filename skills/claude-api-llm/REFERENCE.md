# Claude API LLM Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`generate_code_tool`](#generate_code_tool) | Generate code using Anthropic API tool_use for structured output. |
| [`generate_text_tool`](#generate_text_tool) | Generate text using Anthropic API (drop-in replacement for claude-cli-llm). |
| [`execute_tool`](#execute_tool) | Execute a tool call and return the result as a string. |
| [`agentic_generate_tool`](#agentic_generate_tool) | Multi-step agentic code generation with tool loop. |
| [`structured_output_tool`](#structured_output_tool) | Generate structured JSON output using API tool_use with schema enforcement. |

### Helper Functions

| Function | Description |
|----------|-------------|
| [`get_instance`](#get_instance) | Get or create singleton instance. |
| [`reset`](#reset) | Reset singleton (for testing). |
| [`client`](#client) | Get the initialized Anthropic client. |
| [`model`](#model) | Get the configured model name. |
| [`call_with_tools`](#call_with_tools) | Make an API call with tool definitions. |
| [`call_messages`](#call_messages) | Make a standard messages API call (no tool_use). |
| [`stream_with_tools`](#stream_with_tools) | Stream API response with tool_use support. |
| [`validate_python`](#validate_python) | Validate Python code via ast. |
| [`validate`](#validate) | Validate code for the given language. |
| [`build_context`](#build_context) | Build a context summary of the project directory. |

---

## `generate_code_tool`

Generate code using Anthropic API tool_use for structured output.  Uses forced tool_choice to get clean code without preamble or fences. Includes lint-gating: validates syntax and retries on failure.

**Parameters:**

- **prompt** (`str, required`): Code generation prompt
- **language** (`str, optional`): Target language (default: "python")
- **filename** (`str, optional`): Suggested filename

**Returns:** Dictionary with: - success (bool): Whether generation succeeded - code (str): Clean generated code - language (str): Language of the code - filename (str): Suggested filename - lint_passed (bool): Whether code passed syntax validation

---

## `generate_text_tool`

Generate text using Anthropic API (drop-in replacement for claude-cli-llm).

**Parameters:**

- **prompt** (`str, required`): Text generation prompt
- **max_tokens** (`int, optional`): Maximum tokens (default: 4096)

**Returns:** Dictionary with: - success (bool): Whether generation succeeded - text (str): Generated text - model (str): Model used - provider (str): Always "anthropic-api"

---

## `execute_tool`

Execute a tool call and return the result as a string.

**Parameters:**

- **tool_name** (`str`)
- **tool_input** (`Dict[str, Any]`)

**Returns:** `str`

---

## `agentic_generate_tool`

Multi-step agentic code generation with tool loop.  Sends prompt to Claude with tool definitions (write_file, execute_command, read_file, edit_file, search_replace). Claude autonomously writes files, runs commands, and iterates until done.

**Parameters:**

- **prompt** (`str, required`): Task prompt for agentic execution
- **tools** (`list, optional`): Tool names to enable
- **working_directory** (`str, optional`): Working directory (default: "/tmp")
- **max_tool_rounds** (`int, optional`): Max tool-use rounds (default: 5)
- **stream** (`bool, optional`): Enable token-level streaming (default: False)
- **sandbox_level** (`str, optional`): "trusted", "sandboxed", or "dangerous" (default: "sandboxed")
- **include_context** (`bool, optional`): Auto-discover project files for context (default: True)

**Returns:** Dictionary with: - success (bool): Whether execution succeeded - response (str): Final response text - tool_calls (list): Tool call history - files_created (list): Paths of files created - execution_output (str): Combined execution output

---

## `structured_output_tool`

Generate structured JSON output using API tool_use with schema enforcement.

**Parameters:**

- **prompt** (`str, required`): Prompt for structured output
- **schema** (`dict, optional`): JSON schema for the output structure

**Returns:** Dictionary with: - success (bool): Whether generation succeeded - data (dict): Parsed JSON conforming to schema - model (str): Model used

---

## `get_instance`

Get or create singleton instance.

**Returns:** `'ClaudeAPIClient'`

---

## `reset`

Reset singleton (for testing).

**Returns:** `None`

---

## `client`

Get the initialized Anthropic client.

---

## `model`

Get the configured model name.

**Returns:** `str`

---

## `call_with_tools`

Make an API call with tool definitions.

**Parameters:**

- **messages** (`List[Dict[str, Any]]`)
- **tools** (`List[Dict[str, Any]]`)
- **tool_choice** (`Optional[Dict[str, Any]]`)
- **max_tokens** (`int`)
- **system** (`Optional[str]`)

**Returns:** API response object

---

## `call_messages`

Make a standard messages API call (no tool_use).

**Parameters:**

- **messages** (`List[Dict[str, Any]]`)
- **max_tokens** (`int`)
- **system** (`Optional[str]`)

**Returns:** API response object

---

## `stream_with_tools`

Stream API response with tool_use support.  Uses Anthropic's messages.stream() to yield text tokens in real-time while still supporting tool_use blocks in the final message.

**Parameters:**

- **messages** (`List[Dict[str, Any]]`)
- **tools** (`List[Dict[str, Any]]`)
- **tool_choice** (`Optional[Dict[str, Any]]`)
- **max_tokens** (`int`)
- **system** (`Optional[str]`)
- **on_token** (`Optional[Callable[[str], None]]`)

**Returns:** Final Message object (same shape as call_with_tools)

---

## `validate_python`

Validate Python code via ast.parse.  Returns None if valid, error message string if invalid.

**Parameters:**

- **code** (`str`)

**Returns:** `Optional[str]`

---

## `validate`

Validate code for the given language.  Returns None if valid, error message if invalid. Currently supports: python.

**Parameters:**

- **code** (`str`)
- **language** (`str`)

**Returns:** `Optional[str]`

---

## `build_context`

Build a context summary of the project directory.  Returns a string suitable for injection into the system prompt, containing directory tree + key file contents (within token budget).

**Parameters:**

- **working_directory** (`str`)
- **extensions** (`Optional[set]`)

**Returns:** `str`
