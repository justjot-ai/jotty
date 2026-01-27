"""
GitHub CLI Skill

Provides tools for interacting with GitHub using the gh CLI.
Supports PRs, issues, repo info, and GitHub Actions workflows.
"""
import subprocess
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class GitHubCLI:
    """Helper class for executing gh CLI commands."""

    @staticmethod
    def run_command(args: List[str], timeout: int = 60) -> Dict[str, Any]:
        """
        Execute a gh CLI command and return the result.

        Args:
            args: List of command arguments (without 'gh' prefix)
            timeout: Command timeout in seconds

        Returns:
            Dictionary with success, output/error, and exit_code
        """
        try:
            cmd = ['gh'] + args
            logger.debug(f"Executing: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode == 0:
                return {
                    'success': True,
                    'output': result.stdout.strip(),
                    'exit_code': 0
                }
            else:
                return {
                    'success': False,
                    'error': result.stderr.strip() or result.stdout.strip(),
                    'exit_code': result.returncode
                }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f'Command timed out after {timeout} seconds'
            }
        except FileNotFoundError:
            return {
                'success': False,
                'error': 'gh CLI not found. Install from: https://cli.github.com/'
            }
        except Exception as e:
            logger.error(f"GitHub CLI error: {e}", exc_info=True)
            return {
                'success': False,
                'error': f'Command execution failed: {str(e)}'
            }

    @staticmethod
    def run_json_command(args: List[str], timeout: int = 60) -> Dict[str, Any]:
        """
        Execute a gh CLI command that returns JSON.

        Args:
            args: List of command arguments (without 'gh' prefix)
            timeout: Command timeout in seconds

        Returns:
            Dictionary with success, data/error
        """
        result = GitHubCLI.run_command(args, timeout)

        if not result.get('success'):
            return result

        try:
            output = result.get('output', '')
            if output:
                data = json.loads(output)
                return {
                    'success': True,
                    'data': data
                }
            return {
                'success': True,
                'data': None
            }
        except json.JSONDecodeError as e:
            return {
                'success': False,
                'error': f'Failed to parse JSON response: {str(e)}',
                'raw_output': result.get('output', '')[:500]
            }


def list_prs_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List pull requests for a repository.

    Args:
        params: Dictionary containing:
            - repo (str, optional): Repository in owner/repo format. Uses current repo if not specified.
            - state (str, optional): Filter by state - 'open', 'closed', 'merged', 'all' (default: 'open')
            - limit (int, optional): Maximum number of PRs to return (default: 30)
            - author (str, optional): Filter by author username
            - base (str, optional): Filter by base branch
            - head (str, optional): Filter by head branch
            - label (str, optional): Filter by label

    Returns:
        Dictionary with:
            - success (bool): Whether the operation succeeded
            - prs (list): List of PR objects with number, title, state, author, etc.
            - count (int): Number of PRs returned
            - error (str, optional): Error message if failed
    """
    repo = params.get('repo')
    state = params.get('state', 'open')
    limit = params.get('limit', 30)
    author = params.get('author')
    base = params.get('base')
    head = params.get('head')
    label = params.get('label')

    args = ['pr', 'list', '--json',
            'number,title,state,author,createdAt,updatedAt,url,headRefName,baseRefName,labels,isDraft']

    if repo:
        args.extend(['--repo', repo])

    if state and state != 'all':
        args.extend(['--state', state])

    if limit:
        args.extend(['--limit', str(limit)])

    if author:
        args.extend(['--author', author])

    if base:
        args.extend(['--base', base])

    if head:
        args.extend(['--head', head])

    if label:
        args.extend(['--label', label])

    result = GitHubCLI.run_json_command(args)

    if not result.get('success'):
        return result

    prs = result.get('data', [])

    return {
        'success': True,
        'prs': prs,
        'count': len(prs)
    }


def get_pr_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get details of a specific pull request.

    Args:
        params: Dictionary containing:
            - number (int, required): PR number
            - repo (str, optional): Repository in owner/repo format. Uses current repo if not specified.
            - include_diff (bool, optional): Include diff stats (default: False)
            - include_comments (bool, optional): Include comments (default: False)

    Returns:
        Dictionary with:
            - success (bool): Whether the operation succeeded
            - pr (dict): PR details including number, title, body, state, author, etc.
            - error (str, optional): Error message if failed
    """
    number = params.get('number')
    if not number:
        return {'success': False, 'error': 'number parameter is required'}

    repo = params.get('repo')
    include_diff = params.get('include_diff', False)
    include_comments = params.get('include_comments', False)

    # Build JSON fields to retrieve
    fields = ['number', 'title', 'body', 'state', 'author', 'createdAt', 'updatedAt',
              'closedAt', 'mergedAt', 'url', 'headRefName', 'baseRefName', 'labels',
              'isDraft', 'mergeable', 'reviewDecision', 'additions', 'deletions',
              'changedFiles', 'commits']

    if include_comments:
        fields.append('comments')

    args = ['pr', 'view', str(number), '--json', ','.join(fields)]

    if repo:
        args.extend(['--repo', repo])

    result = GitHubCLI.run_json_command(args)

    if not result.get('success'):
        return result

    pr_data = result.get('data', {})

    # Optionally get diff stats
    if include_diff:
        diff_args = ['pr', 'diff', str(number), '--stat']
        if repo:
            diff_args.extend(['--repo', repo])

        diff_result = GitHubCLI.run_command(diff_args)
        if diff_result.get('success'):
            pr_data['diff_stat'] = diff_result.get('output')

    return {
        'success': True,
        'pr': pr_data
    }


def create_pr_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new pull request.

    Args:
        params: Dictionary containing:
            - title (str, required): PR title
            - body (str, optional): PR description/body
            - base (str, optional): Base branch (default: default branch)
            - head (str, optional): Head branch (default: current branch)
            - repo (str, optional): Repository in owner/repo format
            - draft (bool, optional): Create as draft PR (default: False)
            - labels (list, optional): List of label names to add
            - assignees (list, optional): List of usernames to assign
            - reviewers (list, optional): List of usernames to request review from

    Returns:
        Dictionary with:
            - success (bool): Whether the operation succeeded
            - pr (dict): Created PR details with number, url, etc.
            - error (str, optional): Error message if failed
    """
    title = params.get('title')
    if not title:
        return {'success': False, 'error': 'title parameter is required'}

    body = params.get('body', '')
    base = params.get('base')
    head = params.get('head')
    repo = params.get('repo')
    draft = params.get('draft', False)
    labels = params.get('labels', [])
    assignees = params.get('assignees', [])
    reviewers = params.get('reviewers', [])

    args = ['pr', 'create', '--title', title]

    if body:
        args.extend(['--body', body])

    if base:
        args.extend(['--base', base])

    if head:
        args.extend(['--head', head])

    if repo:
        args.extend(['--repo', repo])

    if draft:
        args.append('--draft')

    if labels:
        for label in labels:
            args.extend(['--label', label])

    if assignees:
        for assignee in assignees:
            args.extend(['--assignee', assignee])

    if reviewers:
        for reviewer in reviewers:
            args.extend(['--reviewer', reviewer])

    result = GitHubCLI.run_command(args)

    if not result.get('success'):
        return result

    # Parse the output to get PR URL
    output = result.get('output', '')

    # Get PR details
    pr_info = {'url': output}

    # Extract PR number from URL if possible
    if '/pull/' in output:
        try:
            pr_number = output.split('/pull/')[-1].strip()
            pr_info['number'] = int(pr_number)
        except (ValueError, IndexError):
            pass

    return {
        'success': True,
        'pr': pr_info,
        'message': f'Pull request created: {output}'
    }


def list_issues_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List issues for a repository.

    Args:
        params: Dictionary containing:
            - repo (str, optional): Repository in owner/repo format. Uses current repo if not specified.
            - state (str, optional): Filter by state - 'open', 'closed', 'all' (default: 'open')
            - limit (int, optional): Maximum number of issues to return (default: 30)
            - author (str, optional): Filter by author username
            - assignee (str, optional): Filter by assignee username
            - label (str, optional): Filter by label
            - milestone (str, optional): Filter by milestone

    Returns:
        Dictionary with:
            - success (bool): Whether the operation succeeded
            - issues (list): List of issue objects
            - count (int): Number of issues returned
            - error (str, optional): Error message if failed
    """
    repo = params.get('repo')
    state = params.get('state', 'open')
    limit = params.get('limit', 30)
    author = params.get('author')
    assignee = params.get('assignee')
    label = params.get('label')
    milestone = params.get('milestone')

    args = ['issue', 'list', '--json',
            'number,title,state,author,createdAt,updatedAt,url,labels,assignees,milestone,body']

    if repo:
        args.extend(['--repo', repo])

    if state and state != 'all':
        args.extend(['--state', state])

    if limit:
        args.extend(['--limit', str(limit)])

    if author:
        args.extend(['--author', author])

    if assignee:
        args.extend(['--assignee', assignee])

    if label:
        args.extend(['--label', label])

    if milestone:
        args.extend(['--milestone', milestone])

    result = GitHubCLI.run_json_command(args)

    if not result.get('success'):
        return result

    issues = result.get('data', [])

    return {
        'success': True,
        'issues': issues,
        'count': len(issues)
    }


def create_issue_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new issue.

    Args:
        params: Dictionary containing:
            - title (str, required): Issue title
            - body (str, optional): Issue description/body
            - repo (str, optional): Repository in owner/repo format
            - labels (list, optional): List of label names to add
            - assignees (list, optional): List of usernames to assign
            - milestone (str, optional): Milestone name or number

    Returns:
        Dictionary with:
            - success (bool): Whether the operation succeeded
            - issue (dict): Created issue details with number, url, etc.
            - error (str, optional): Error message if failed
    """
    title = params.get('title')
    if not title:
        return {'success': False, 'error': 'title parameter is required'}

    body = params.get('body', '')
    repo = params.get('repo')
    labels = params.get('labels', [])
    assignees = params.get('assignees', [])
    milestone = params.get('milestone')

    args = ['issue', 'create', '--title', title]

    if body:
        args.extend(['--body', body])

    if repo:
        args.extend(['--repo', repo])

    if labels:
        for label in labels:
            args.extend(['--label', label])

    if assignees:
        for assignee in assignees:
            args.extend(['--assignee', assignee])

    if milestone:
        args.extend(['--milestone', str(milestone)])

    result = GitHubCLI.run_command(args)

    if not result.get('success'):
        return result

    # Parse the output to get issue URL
    output = result.get('output', '')

    issue_info = {'url': output}

    # Extract issue number from URL if possible
    if '/issues/' in output:
        try:
            issue_number = output.split('/issues/')[-1].strip()
            issue_info['number'] = int(issue_number)
        except (ValueError, IndexError):
            pass

    return {
        'success': True,
        'issue': issue_info,
        'message': f'Issue created: {output}'
    }


def get_repo_info_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get information about a repository.

    Args:
        params: Dictionary containing:
            - repo (str, optional): Repository in owner/repo format. Uses current repo if not specified.

    Returns:
        Dictionary with:
            - success (bool): Whether the operation succeeded
            - repo (dict): Repository details including name, description, stars, forks, etc.
            - error (str, optional): Error message if failed
    """
    repo = params.get('repo')

    args = ['repo', 'view', '--json',
            'name,owner,description,url,homepageUrl,defaultBranchRef,isPrivate,isFork,'
            'stargazerCount,forkCount,watchers,issues,pullRequests,createdAt,updatedAt,'
            'languages,licenseInfo,primaryLanguage']

    if repo:
        args.append(repo)

    result = GitHubCLI.run_json_command(args)

    if not result.get('success'):
        return result

    return {
        'success': True,
        'repo': result.get('data', {})
    }


def list_workflows_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List GitHub Actions workflows for a repository.

    Args:
        params: Dictionary containing:
            - repo (str, optional): Repository in owner/repo format. Uses current repo if not specified.
            - limit (int, optional): Maximum number of workflows to return (default: 30)

    Returns:
        Dictionary with:
            - success (bool): Whether the operation succeeded
            - workflows (list): List of workflow objects with id, name, state, path
            - count (int): Number of workflows returned
            - error (str, optional): Error message if failed
    """
    repo = params.get('repo')
    limit = params.get('limit', 30)

    args = ['workflow', 'list', '--json', 'id,name,state,path']

    if repo:
        args.extend(['--repo', repo])

    if limit:
        args.extend(['--limit', str(limit)])

    result = GitHubCLI.run_json_command(args)

    if not result.get('success'):
        return result

    workflows = result.get('data', [])

    return {
        'success': True,
        'workflows': workflows,
        'count': len(workflows)
    }


def run_workflow_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Trigger a GitHub Actions workflow.

    Args:
        params: Dictionary containing:
            - workflow (str, required): Workflow ID, name, or filename
            - repo (str, optional): Repository in owner/repo format. Uses current repo if not specified.
            - ref (str, optional): Branch or tag to run the workflow on (default: default branch)
            - inputs (dict, optional): Input parameters for the workflow as key-value pairs

    Returns:
        Dictionary with:
            - success (bool): Whether the operation succeeded
            - message (str): Success/status message
            - error (str, optional): Error message if failed
    """
    workflow = params.get('workflow')
    if not workflow:
        return {'success': False, 'error': 'workflow parameter is required'}

    repo = params.get('repo')
    ref = params.get('ref')
    inputs = params.get('inputs', {})

    args = ['workflow', 'run', str(workflow)]

    if repo:
        args.extend(['--repo', repo])

    if ref:
        args.extend(['--ref', ref])

    # Add workflow inputs
    if inputs and isinstance(inputs, dict):
        for key, value in inputs.items():
            args.extend(['--field', f'{key}={value}'])

    result = GitHubCLI.run_command(args)

    if not result.get('success'):
        return result

    return {
        'success': True,
        'message': f'Workflow "{workflow}" triggered successfully',
        'output': result.get('output', '')
    }
