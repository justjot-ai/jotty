# GitHub CLI Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`list_prs_tool`](#list_prs_tool) | List pull requests for a repository. |
| [`get_pr_tool`](#get_pr_tool) | Get details of a specific pull request. |
| [`create_pr_tool`](#create_pr_tool) | Create a new pull request. |
| [`list_issues_tool`](#list_issues_tool) | List issues for a repository. |
| [`create_issue_tool`](#create_issue_tool) | Create a new issue. |
| [`get_repo_info_tool`](#get_repo_info_tool) | Get information about a repository. |
| [`list_workflows_tool`](#list_workflows_tool) | List GitHub Actions workflows for a repository. |
| [`run_workflow_tool`](#run_workflow_tool) | Trigger a GitHub Actions workflow. |

### Helper Functions

| Function | Description |
|----------|-------------|
| [`run_command`](#run_command) | Execute a gh CLI command and return the result. |
| [`run_json_command`](#run_json_command) | Execute a gh CLI command that returns JSON. |

---

## `list_prs_tool`

List pull requests for a repository.

**Parameters:**

- **repo** (`str, optional`): Repository in owner/repo format
- **state** (`str, optional`): 'open', 'closed', 'merged', 'all' (default: 'open')
- **limit** (`int, optional`): Maximum PRs to return (default: 30) author/base/head/label (str, optional): Filters

**Returns:** Dictionary with success, prs list, count

---

## `get_pr_tool`

Get details of a specific pull request.

**Parameters:**

- **number** (`int, required`): PR number
- **repo** (`str, optional`): Repository in owner/repo format
- **include_diff** (`bool, optional`): Include diff stats
- **include_comments** (`bool, optional`): Include comments

**Returns:** Dictionary with success, pr object

---

## `create_pr_tool`

Create a new pull request.

**Parameters:**

- **title** (`str, required`): PR title
- **body** (`str, optional`): PR description base/head (str, optional): Base/head branch
- **repo** (`str, optional`): Repository in owner/repo format
- **draft** (`bool, optional`): Create as draft labels/assignees/reviewers (list, optional): Lists to add

**Returns:** Dictionary with success, pr object, message

---

## `list_issues_tool`

List issues for a repository.

**Parameters:**

- **repo** (`str, optional`): Repository in owner/repo format
- **state** (`str, optional`): 'open', 'closed', 'all' (default: 'open')
- **limit** (`int, optional`): Maximum issues to return (default: 30) author/assignee/label/milestone (str, optional): Filters

**Returns:** Dictionary with success, issues list, count

---

## `create_issue_tool`

Create a new issue.

**Parameters:**

- **title** (`str, required`): Issue title
- **body** (`str, optional`): Issue description
- **repo** (`str, optional`): Repository in owner/repo format labels/assignees (list, optional): Lists to add
- **milestone** (`str, optional`): Milestone name or number

**Returns:** Dictionary with success, issue object, message

---

## `get_repo_info_tool`

Get information about a repository.

**Parameters:**

- **repo** (`str, optional`): Repository in owner/repo format

**Returns:** Dictionary with success, repo object

---

## `list_workflows_tool`

List GitHub Actions workflows for a repository.

**Parameters:**

- **repo** (`str, optional`): Repository in owner/repo format
- **limit** (`int, optional`): Maximum workflows to return (default: 30)

**Returns:** Dictionary with success, workflows list, count

---

## `run_workflow_tool`

Trigger a GitHub Actions workflow.

**Parameters:**

- **workflow** (`str, required`): Workflow ID, name, or filename
- **repo** (`str, optional`): Repository in owner/repo format
- **ref** (`str, optional`): Branch or tag to run on
- **inputs** (`dict, optional`): Input parameters as key-value pairs

**Returns:** Dictionary with success, message, output

---

## `run_command`

Execute a gh CLI command and return the result.

**Parameters:**

- **args** (`List[str]`)
- **timeout** (`int`)

**Returns:** `Dict[str, Any]`

---

## `run_json_command`

Execute a gh CLI command that returns JSON.

**Parameters:**

- **args** (`List[str]`)
- **timeout** (`int`)

**Returns:** `Dict[str, Any]`
