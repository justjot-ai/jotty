"""
GitHub CLI Skill

Provides tools for interacting with GitHub using the gh CLI.
Supports PRs, issues, repo info, and GitHub Actions workflows.
Refactored to use Jotty core utilities.
"""

import subprocess
import json
import logging
from typing import Dict, Any, List

from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper

logger = logging.getLogger(__name__)


class GitHubCLI:
    """Helper class for executing gh CLI commands."""

    @staticmethod
    def run_command(args: List[str], timeout: int = 60) -> Dict[str, Any]:
        """Execute a gh CLI command and return the result."""
        try:
            cmd = ['gh'] + args
            logger.debug(f"Executing: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

            if result.returncode == 0:
                return tool_response(output=result.stdout.strip(), exit_code=0)
            else:
                return tool_error(
                    result.stderr.strip() or result.stdout.strip(),
                    exit_code=result.returncode
                )

        except subprocess.TimeoutExpired:
            return tool_error(f'Command timed out after {timeout} seconds')
        except FileNotFoundError:
            return tool_error('gh CLI not found. Install from: https://cli.github.com/')
        except Exception as e:
            logger.error(f"GitHub CLI error: {e}", exc_info=True)
            return tool_error(f'Command execution failed: {str(e)}')

    @staticmethod
    def run_json_command(args: List[str], timeout: int = 60) -> Dict[str, Any]:
        """Execute a gh CLI command that returns JSON."""
        result = GitHubCLI.run_command(args, timeout)

        if not result.get('success'):
            return result

        try:
            output = result.get('output', '')
            if output:
                data = json.loads(output)
                return tool_response(data=data)
            return tool_response(data=None)
        except json.JSONDecodeError as e:
            return tool_error(
                f'Failed to parse JSON response: {str(e)}',
                raw_output=result.get('output', '')[:500]
            )


def _add_repo_arg(args: List[str], repo: str) -> None:
    """Add --repo argument if provided."""
    if repo:
        args.extend(['--repo', repo])


def _add_limit_arg(args: List[str], limit: int) -> None:
    """Add --limit argument if provided."""
    if limit:
        args.extend(['--limit', str(limit)])


@tool_wrapper()
def list_prs_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List pull requests for a repository.

    Args:
        params: Dictionary containing:
            - repo (str, optional): Repository in owner/repo format
            - state (str, optional): 'open', 'closed', 'merged', 'all' (default: 'open')
            - limit (int, optional): Maximum PRs to return (default: 30)
            - author/base/head/label (str, optional): Filters

    Returns:
        Dictionary with success, prs list, count
    """
    args = ['pr', 'list', '--json',
            'number,title,state,author,createdAt,updatedAt,url,headRefName,baseRefName,labels,isDraft']

    _add_repo_arg(args, params.get('repo'))

    state = params.get('state', 'open')
    if state and state != 'all':
        args.extend(['--state', state])

    _add_limit_arg(args, params.get('limit', 30))

    for key in ('author', 'base', 'head', 'label'):
        if params.get(key):
            args.extend([f'--{key}', params[key]])

    result = GitHubCLI.run_json_command(args)

    if result.get('success'):
        prs = result.get('data', [])
        return tool_response(prs=prs, count=len(prs))

    return result


@tool_wrapper(required_params=['number'])
def get_pr_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get details of a specific pull request.

    Args:
        params: Dictionary containing:
            - number (int, required): PR number
            - repo (str, optional): Repository in owner/repo format
            - include_diff (bool, optional): Include diff stats
            - include_comments (bool, optional): Include comments

    Returns:
        Dictionary with success, pr object
    """
    fields = ['number', 'title', 'body', 'state', 'author', 'createdAt', 'updatedAt',
              'closedAt', 'mergedAt', 'url', 'headRefName', 'baseRefName', 'labels',
              'isDraft', 'mergeable', 'reviewDecision', 'additions', 'deletions',
              'changedFiles', 'commits']

    if params.get('include_comments'):
        fields.append('comments')

    args = ['pr', 'view', str(params['number']), '--json', ','.join(fields)]
    _add_repo_arg(args, params.get('repo'))

    result = GitHubCLI.run_json_command(args)

    if not result.get('success'):
        return result

    pr_data = result.get('data', {})

    # Optionally get diff stats
    if params.get('include_diff'):
        diff_args = ['pr', 'diff', str(params['number']), '--stat']
        _add_repo_arg(diff_args, params.get('repo'))
        diff_result = GitHubCLI.run_command(diff_args)
        if diff_result.get('success'):
            pr_data['diff_stat'] = diff_result.get('output')

    return tool_response(pr=pr_data)


@tool_wrapper(required_params=['title'])
def create_pr_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new pull request.

    Args:
        params: Dictionary containing:
            - title (str, required): PR title
            - body (str, optional): PR description
            - base/head (str, optional): Base/head branch
            - repo (str, optional): Repository in owner/repo format
            - draft (bool, optional): Create as draft
            - labels/assignees/reviewers (list, optional): Lists to add

    Returns:
        Dictionary with success, pr object, message
    """
    args = ['pr', 'create', '--title', params['title']]

    if params.get('body'):
        args.extend(['--body', params['body']])

    for key in ('base', 'head'):
        if params.get(key):
            args.extend([f'--{key}', params[key]])

    _add_repo_arg(args, params.get('repo'))

    if params.get('draft'):
        args.append('--draft')

    for key in ('labels', 'assignees', 'reviewers'):
        for item in params.get(key, []):
            args.extend([f'--{key[:-1]}', item])  # Remove 's' for flag name

    result = GitHubCLI.run_command(args)

    if not result.get('success'):
        return result

    output = result.get('output', '')
    pr_info = {'url': output}

    if '/pull/' in output:
        try:
            pr_info['number'] = int(output.split('/pull/')[-1].strip())
        except (ValueError, IndexError):
            pass

    return tool_response(pr=pr_info, message=f'Pull request created: {output}')


@tool_wrapper()
def list_issues_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List issues for a repository.

    Args:
        params: Dictionary containing:
            - repo (str, optional): Repository in owner/repo format
            - state (str, optional): 'open', 'closed', 'all' (default: 'open')
            - limit (int, optional): Maximum issues to return (default: 30)
            - author/assignee/label/milestone (str, optional): Filters

    Returns:
        Dictionary with success, issues list, count
    """
    args = ['issue', 'list', '--json',
            'number,title,state,author,createdAt,updatedAt,url,labels,assignees,milestone,body']

    _add_repo_arg(args, params.get('repo'))

    state = params.get('state', 'open')
    if state and state != 'all':
        args.extend(['--state', state])

    _add_limit_arg(args, params.get('limit', 30))

    for key in ('author', 'assignee', 'label', 'milestone'):
        if params.get(key):
            args.extend([f'--{key}', params[key]])

    result = GitHubCLI.run_json_command(args)

    if result.get('success'):
        issues = result.get('data', [])
        return tool_response(issues=issues, count=len(issues))

    return result


@tool_wrapper(required_params=['title'])
def create_issue_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new issue.

    Args:
        params: Dictionary containing:
            - title (str, required): Issue title
            - body (str, optional): Issue description
            - repo (str, optional): Repository in owner/repo format
            - labels/assignees (list, optional): Lists to add
            - milestone (str, optional): Milestone name or number

    Returns:
        Dictionary with success, issue object, message
    """
    args = ['issue', 'create', '--title', params['title']]

    if params.get('body'):
        args.extend(['--body', params['body']])

    _add_repo_arg(args, params.get('repo'))

    for key in ('labels', 'assignees'):
        for item in params.get(key, []):
            args.extend([f'--{key[:-1]}', item])

    if params.get('milestone'):
        args.extend(['--milestone', str(params['milestone'])])

    result = GitHubCLI.run_command(args)

    if not result.get('success'):
        return result

    output = result.get('output', '')
    issue_info = {'url': output}

    if '/issues/' in output:
        try:
            issue_info['number'] = int(output.split('/issues/')[-1].strip())
        except (ValueError, IndexError):
            pass

    return tool_response(issue=issue_info, message=f'Issue created: {output}')


@tool_wrapper()
def get_repo_info_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get information about a repository.

    Args:
        params: Dictionary containing:
            - repo (str, optional): Repository in owner/repo format

    Returns:
        Dictionary with success, repo object
    """
    args = ['repo', 'view', '--json',
            'name,owner,description,url,homepageUrl,defaultBranchRef,isPrivate,isFork,'
            'stargazerCount,forkCount,watchers,issues,pullRequests,createdAt,updatedAt,'
            'languages,licenseInfo,primaryLanguage']

    if params.get('repo'):
        args.append(params['repo'])

    result = GitHubCLI.run_json_command(args)

    if result.get('success'):
        return tool_response(repo=result.get('data', {}))

    return result


@tool_wrapper()
def list_workflows_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List GitHub Actions workflows for a repository.

    Args:
        params: Dictionary containing:
            - repo (str, optional): Repository in owner/repo format
            - limit (int, optional): Maximum workflows to return (default: 30)

    Returns:
        Dictionary with success, workflows list, count
    """
    args = ['workflow', 'list', '--json', 'id,name,state,path']

    _add_repo_arg(args, params.get('repo'))
    _add_limit_arg(args, params.get('limit', 30))

    result = GitHubCLI.run_json_command(args)

    if result.get('success'):
        workflows = result.get('data', [])
        return tool_response(workflows=workflows, count=len(workflows))

    return result


@tool_wrapper(required_params=['workflow'])
def run_workflow_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Trigger a GitHub Actions workflow.

    Args:
        params: Dictionary containing:
            - workflow (str, required): Workflow ID, name, or filename
            - repo (str, optional): Repository in owner/repo format
            - ref (str, optional): Branch or tag to run on
            - inputs (dict, optional): Input parameters as key-value pairs

    Returns:
        Dictionary with success, message, output
    """
    args = ['workflow', 'run', str(params['workflow'])]

    _add_repo_arg(args, params.get('repo'))

    if params.get('ref'):
        args.extend(['--ref', params['ref']])

    inputs = params.get('inputs', {})
    if inputs and isinstance(inputs, dict):
        for key, value in inputs.items():
            args.extend(['--field', f'{key}={value}'])

    result = GitHubCLI.run_command(args)

    if result.get('success'):
        return tool_response(
            message=f'Workflow "{params["workflow"]}" triggered successfully',
            output=result.get('output', '')
        )

    return result


__all__ = [
    'list_prs_tool',
    'get_pr_tool',
    'create_pr_tool',
    'list_issues_tool',
    'create_issue_tool',
    'get_repo_info_tool',
    'list_workflows_tool',
    'run_workflow_tool'
]
