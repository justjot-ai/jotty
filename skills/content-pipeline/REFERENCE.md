# Content Pipeline Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`run_pipeline_tool`](#run_pipeline_tool) | Run a complete content pipeline: Source -> Processors -> Sinks. |
| [`run_source_tool`](#run_source_tool) | Run only the source stage to generate a Document. |
| [`process_document_tool`](#process_document_tool) | Process an existing document through processors. |
| [`sink_document_tool`](#sink_document_tool) | Write a document to one or more sinks. |

### Helper Functions

| Function | Description |
|----------|-------------|
| [`validate_input`](#validate_input) | No description available. |
| [`generate`](#generate) | No description available. |
| [`can_process`](#can_process) | No description available. |
| [`process`](#process) | No description available. |
| [`validate_document`](#validate_document) | No description available. |
| [`write`](#write) | No description available. |

---

## `run_pipeline_tool`

Run a complete content pipeline: Source -> Processors -> Sinks.

**Parameters:**

- **source_type** (`str, required`): Source adapter type ('markdown', 'arxiv', 'youtube', 'html', 'pdf')
- **source_params** (`dict, required`): Parameters for source
- **processors** (`list, optional`): Processor configurations
- **sinks** (`list, optional`): Sink configurations
- **output_dir** (`str, optional`): Output directory

**Returns:** Dictionary with: - success (bool): Whether pipeline succeeded - document (dict): Final document (serialized) - output_paths (list): Generated file paths - history (list): Pipeline execution history - error (str, optional): Error message if failed

---

## `run_source_tool`

Run only the source stage to generate a Document.

**Parameters:**

- **source_type** (`str, required`): Source adapter type ('markdown', 'arxiv', 'youtube', 'html', 'pdf')
- **source_params** (`dict, required`): Parameters for source

**Returns:** Dictionary with: - success (bool): Whether source generation succeeded - document (dict): Generated document (serialized) - error (str, optional): Error message if failed

---

## `process_document_tool`

Process an existing document through processors.

**Parameters:**

- **document** (`dict, required`): Document dictionary
- **processors** (`list, required`): Processor configurations

**Returns:** Dictionary with: - success (bool): Whether processing succeeded - document (dict): Processed document (serialized) - error (str, optional): Error message if failed

---

## `sink_document_tool`

Write a document to one or more sinks.

**Parameters:**

- **document** (`dict, required`): Document dictionary
- **sinks** (`list, required`): Sink configurations
- **output_dir** (`str, optional`): Output directory

**Returns:** Dictionary with: - success (bool): Whether sink writing succeeded - output_paths (list): Generated file paths - error (str, optional): Error message if failed

---

## `validate_input`

No description available.

---

## `generate`

No description available.

---

## `can_process`

No description available.

**Parameters:**

- **doc**

---

## `process`

No description available.

**Parameters:**

- **doc**

---

## `validate_document`

No description available.

**Parameters:**

- **doc**

---

## `write`

No description available.

**Parameters:**

- **doc**
- **output_path**
