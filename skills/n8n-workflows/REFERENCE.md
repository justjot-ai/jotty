# n8n Workflow Integration - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`list_n8n_workflows_tool`](#list_n8n_workflows_tool) | List all n8n workflows with trigger classification. |
| [`trigger_n8n_workflow_tool`](#trigger_n8n_workflow_tool) | Trigger an n8n workflow. |
| [`get_n8n_execution_tool`](#get_n8n_execution_tool) | Check the status and results of an n8n execution. |
| [`activate_n8n_workflow_tool`](#activate_n8n_workflow_tool) | Activate or deactivate an n8n workflow. |
| [`setup_n8n_workflows_tool`](#setup_n8n_workflows_tool) | One-time setup: create all standard n8n workflows. |

### Helper Functions

| Function | Description |
|----------|-------------|
| [`list_workflows`](#list_workflows) | Fetch all workflows from n8n. |
| [`get_workflow`](#get_workflow) | Fetch a single workflow by ID. |
| [`activate_workflow`](#activate_workflow) | Activate or deactivate a workflow. |
| [`trigger_via_webhook`](#trigger_via_webhook) | POST to a webhook-triggered workflow. |
| [`create_execution`](#create_execution) | Trigger a non-webhook workflow via executions API. |
| [`get_execution`](#get_execution) | Check execution status. |
| [`classify_trigger`](#classify_trigger) | Return trigger type: webhook | schedule | manual | execute | unknown. |
| [`get_webhook_path`](#get_webhook_path) | Extract webhook path from a webhook-triggered workflow. |
| [`summarize_workflow`](#summarize_workflow) | Produce a compact summary of a workflow. |
| [`infer_capabilities`](#infer_capabilities) | Union of tag, node, and name-based capabilities plus 'automation'. |
| [`infer_use_when`](#infer_use_when) | Natural-language use_when hint for LLM skill selection. |
| [`infer_description`](#infer_description) | Rich description including project context and trigger type. |
| [`register_all`](#register_all) | Register each n8n workflow as a derived skill. |
| [`save_cache`](#save_cache) | Write enriched workflow data to disk for offline startup. |
| [`load_cache`](#load_cache) | Load cached workflow skills from disk. |
| [`reset`](#reset) | Reset registration state (for testing). |
| [`create_schedule_to_http_to_telegram`](#create_schedule_to_http_to_telegram) | Build: ScheduleTrigger -> HTTP Request -> Telegram. |
| [`create_webhook_to_http_to_telegram`](#create_webhook_to_http_to_telegram) | Build: Webhook -> HTTP Request -> Telegram. |
| [`create_schedule_to_ssh_pipeline`](#create_schedule_to_ssh_pipeline) | Build: ScheduleTrigger -> SSH -> conditional Telegram on error. |
| [`get_all_workflow_definitions`](#get_all_workflow_definitions) | Return all 15 planned workflow definitions. |
| [`create_all_workflows`](#create_all_workflows) | Create all 15 planned workflows in n8n. |

---

## `list_n8n_workflows_tool`

List all n8n workflows with trigger classification.

**Parameters:**

- **active_only** (`bool, optional`): Only return active workflows
- **api_key** (`str, optional`): n8n API key (defaults to N8N_API_KEY env)
- **base_url** (`str, optional`): n8n instance URL (defaults to N8N_BASE_URL env)

**Returns:** Dictionary with success, workflows list, total count

---

## `trigger_n8n_workflow_tool`

Trigger an n8n workflow. Auto-detects webhook vs direct execution.

**Parameters:**

- **workflow_id** (`str, required`): Workflow ID to trigger
- **data** (`dict, optional`): Payload data to send
- **wait** (`bool, optional`): Wait for completion (default True)
- **timeout** (`int, optional`): Max seconds to wait (default 300)
- **api_key** (`str, optional`): n8n API key
- **base_url** (`str, optional`): n8n instance URL

**Returns:** Dictionary with success, execution_id, status, and data

---

## `get_n8n_execution_tool`

Check the status and results of an n8n execution.

**Parameters:**

- **execution_id** (`str, required`): Execution ID to check
- **api_key** (`str, optional`): n8n API key
- **base_url** (`str, optional`): n8n instance URL

**Returns:** Dictionary with success, execution_id, status, data

---

## `activate_n8n_workflow_tool`

Activate or deactivate an n8n workflow.

**Parameters:**

- **workflow_id** (`str, required`): Workflow ID
- **active** (`bool, optional`): True to activate, False to deactivate (default True)
- **api_key** (`str, optional`): n8n API key
- **base_url** (`str, optional`): n8n instance URL

**Returns:** Dictionary with success, workflow_id, active state

---

## `setup_n8n_workflows_tool`

One-time setup: create all standard n8n workflows.  Creates 15 workflows (11 PMI finance + 4 DevOps) in the configured n8n instance. Idempotent: skips workflows that already exist by name.

**Parameters:**

- **api_key** (`str, optional`): n8n API key
- **base_url** (`str, optional`): n8n instance URL

**Returns:** Dictionary with created count and total available

---

## `list_workflows`

Fetch all workflows from n8n.

**Parameters:**

- **active_only** (`bool`)

**Returns:** `Dict[str, Any]`

---

## `get_workflow`

Fetch a single workflow by ID.

**Parameters:**

- **workflow_id** (`str`)

**Returns:** `Dict[str, Any]`

---

## `activate_workflow`

Activate or deactivate a workflow.

**Parameters:**

- **workflow_id** (`str`)
- **active** (`bool`)

**Returns:** `Dict[str, Any]`

---

## `trigger_via_webhook`

POST to a webhook-triggered workflow.

**Parameters:**

- **webhook_path** (`str`)
- **data** (`Optional[Dict]`)

**Returns:** `Dict[str, Any]`

---

## `create_execution`

Trigger a non-webhook workflow via executions API.

**Parameters:**

- **workflow_id** (`str`)
- **data** (`Optional[Dict]`)

**Returns:** `Dict[str, Any]`

---

## `get_execution`

Check execution status.

**Parameters:**

- **execution_id** (`str`)

**Returns:** `Dict[str, Any]`

---

## `classify_trigger`

Return trigger type: webhook | schedule | manual | execute | unknown.

**Parameters:**

- **nodes** (`List[Dict[str, Any]]`)

**Returns:** `str`

---

## `get_webhook_path`

Extract webhook path from a webhook-triggered workflow.

**Parameters:**

- **nodes** (`List[Dict[str, Any]]`)

**Returns:** `Optional[str]`

---

## `summarize_workflow`

Produce a compact summary of a workflow.

**Parameters:**

- **workflow** (`Dict[str, Any]`)

**Returns:** `Dict[str, Any]`

---

## `infer_capabilities`

Union of tag, node, and name-based capabilities plus 'automation'.

**Parameters:**

- **workflow** (`Dict[str, Any]`)

**Returns:** `List[str]`

---

## `infer_use_when`

Natural-language use_when hint for LLM skill selection.

**Parameters:**

- **workflow** (`Dict[str, Any]`)

**Returns:** `str`

---

## `infer_description`

Rich description including project context and trigger type.

**Parameters:**

- **workflow** (`Dict[str, Any]`)
- **trigger_type** (`str`)

**Returns:** `str`

---

## `register_all`

Register each n8n workflow as a derived skill.  Uses WorkflowCapabilityInferrer for domain-specific capabilities, then persists enriched data to disk cache for offline startup.

**Parameters:**

- **registry**

**Returns:** Number of workflows registered.

---

## `save_cache`

Write enriched workflow data to disk for offline startup.

**Parameters:**

- **workflows** (`List[Dict[str, Any]]`)
- **base_url** (`str`)

**Returns:** `None`

---

## `load_cache`

Load cached workflow skills from disk.

**Returns:** Parsed cache envelope dict, or None if missing/invalid.

---

## `reset`

Reset registration state (for testing).

---

## `create_schedule_to_http_to_telegram`

Build: ScheduleTrigger -> HTTP Request -> Telegram.

**Parameters:**

- **name** (`str`)
- **cron** (`str`)
- **http_url** (`str`)
- **http_method** (`str`)
- **message_template** (`str`)
- **tags** (`List[str]`)

**Returns:** `Dict[str, Any]`

---

## `create_webhook_to_http_to_telegram`

Build: Webhook -> HTTP Request -> Telegram.

**Parameters:**

- **name** (`str`)
- **webhook_path** (`str`)
- **http_url** (`str`)
- **message_template** (`str`)
- **tags** (`List[str]`)

**Returns:** `Dict[str, Any]`

---

## `create_schedule_to_ssh_pipeline`

Build: ScheduleTrigger -> SSH -> conditional Telegram on error.

**Parameters:**

- **name** (`str`)
- **cron** (`str`)
- **ssh_command** (`str`)
- **tags** (`List[str]`)

**Returns:** `Dict[str, Any]`

---

## `get_all_workflow_definitions`

Return all 15 planned workflow definitions.

**Returns:** `List[Dict[str, Any]]`

---

## `create_all_workflows`

Create all 15 planned workflows in n8n. Idempotent (skips existing).

**Parameters:**

- **client** (`N8nAPIClient`)

**Returns:** Number of workflows created.
