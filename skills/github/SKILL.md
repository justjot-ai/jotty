---
name: github
description: "Provides tools for interacting with GitHub repositories using the `gh` CLI. Supports pull requests, issues, repository information, and GitHub Actions workflows. Use when the user wants to create pr, list issues, pull request."
---

# GitHub CLI Skill

## Description
Provides tools for interacting with GitHub repositories using the `gh` CLI. Supports pull requests, issues, repository information, and GitHub Actions workflows.


## Type
base


## Capabilities
- code


## Triggers
- "create pr"
- "list issues"
- "github"
- "pull request"
- "check ci"

## Category
workflow-automation

## Prerequisites
- GitHub CLI (`gh`) must be installed: https://cli.github.com/
- Must be authenticated: `gh auth login`

## Tools

### list_prs_tool
List pull requests for a repository.

**Parameters:**
- `repo` (str, optional): Repository in owner/repo format. Uses current repo if not specified.
- `state` (str, optional): Filter by state - 'open', 'closed', 'merged', 'all' (default: 'open')
- `limit` (int, optional): Maximum number of PRs to return (default: 30)
- `author` (str, optional): Filter by author username
- `base` (str, optional): Filter by base branch
- `head` (str, optional): Filter by head branch
- `label` (str, optional): Filter by label

**Returns:** `prs` (list), `count` (int)

---

### get_pr_tool
Get details of a specific pull request.

**Parameters:**
- `number` (int, required): PR number
- `repo` (str, optional): Repository in owner/repo format
- `include_diff` (bool, optional): Include diff stats (default: False)
- `include_comments` (bool, optional): Include comments (default: False)

**Returns:** `pr` (dict) with full PR details

---

### create_pr_tool
Create a new pull request.

**Parameters:**
- `title` (str, required): PR title
- `body` (str, optional): PR description/body
- `base` (str, optional): Base branch (default: default branch)
- `head` (str, optional): Head branch (default: current branch)
- `repo` (str, optional): Repository in owner/repo format
- `draft` (bool, optional): Create as draft PR (default: False)
- `labels` (list, optional): List of label names to add
- `assignees` (list, optional): List of usernames to assign
- `reviewers` (list, optional): List of usernames to request review from

**Returns:** `pr` (dict) with number and url

---

### list_issues_tool
List issues for a repository.

**Parameters:**
- `repo` (str, optional): Repository in owner/repo format
- `state` (str, optional): Filter by state - 'open', 'closed', 'all' (default: 'open')
- `limit` (int, optional): Maximum number of issues to return (default: 30)
- `author` (str, optional): Filter by author username
- `assignee` (str, optional): Filter by assignee username
- `label` (str, optional): Filter by label
- `milestone` (str, optional): Filter by milestone

**Returns:** `issues` (list), `count` (int)

---

### create_issue_tool
Create a new issue.

**Parameters:**
- `title` (str, required): Issue title
- `body` (str, optional): Issue description/body
- `repo` (str, optional): Repository in owner/repo format
- `labels` (list, optional): List of label names to add
- `assignees` (list, optional): List of usernames to assign
- `milestone` (str, optional): Milestone name or number

**Returns:** `issue` (dict) with number and url

---

### get_repo_info_tool
Get information about a repository.

**Parameters:**
- `repo` (str, optional): Repository in owner/repo format. Uses current repo if not specified.

**Returns:** `repo` (dict) with name, description, stars, forks, languages, etc.

---

### list_workflows_tool
List GitHub Actions workflows for a repository.

**Parameters:**
- `repo` (str, optional): Repository in owner/repo format
- `limit` (int, optional): Maximum number of workflows to return (default: 30)

**Returns:** `workflows` (list), `count` (int)

---

### run_workflow_tool
Trigger a GitHub Actions workflow.

**Parameters:**
- `workflow` (str, required): Workflow ID, name, or filename
- `repo` (str, optional): Repository in owner/repo format
- `ref` (str, optional): Branch or tag to run the workflow on (default: default branch)
- `inputs` (dict, optional): Input parameters for the workflow as key-value pairs

**Returns:** `message` (str) with status

## Examples

### List open PRs
```python
result = list_prs_tool({'repo': 'owner/repo', 'state': 'open', 'limit': 10})
```

### Get PR details with diff
```python
result = get_pr_tool({'number': 123, 'include_diff': True})
```

### Create a PR
```python
result = create_pr_tool({
    'title': 'Add new feature',
    'body': 'This PR adds...',
    'base': 'main',
    'draft': True,
    'labels': ['enhancement']
})
```

### List issues by label
```python
result = list_issues_tool({'label': 'bug', 'state': 'open'})
```

### Create an issue
```python
result = create_issue_tool({
    'title': 'Bug: Something is broken',
    'body': 'Steps to reproduce...',
    'labels': ['bug', 'high-priority']
})
```

### Get repository info
```python
result = get_repo_info_tool({'repo': 'facebook/react'})
```

### List and run workflows
```python
# List workflows
result = list_workflows_tool({'repo': 'owner/repo'})

# Run a workflow
result = run_workflow_tool({
    'workflow': 'deploy.yml',
    'ref': 'main',
    'inputs': {'environment': 'staging'}
})
```

## Error Handling
All tools return a dictionary with:
- `success` (bool): Whether the operation succeeded
- `error` (str): Error message if failed

Common errors:
- `gh CLI not found`: Install GitHub CLI from https://cli.github.com/
- `authentication required`: Run `gh auth login`
- `repository not found`: Check repo name or permissions
