"""
Changelog Generator Skill - Transform git commits into user-friendly changelogs.

Converts technical developer commits into polished, customer-facing release notes.
"""
import asyncio
import logging
import inspect
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import os

from Jotty.core.utils.skill_status import SkillStatus
from Jotty.core.utils.tool_helpers import tool_response, tool_error, async_tool_wrapper

# Status emitter for progress updates
status = SkillStatus("changelog-generator")


logger = logging.getLogger(__name__)

try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False
    logger.warning("gitpython not available - install with: pip install gitpython")


@async_tool_wrapper()
async def generate_changelog_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a user-friendly changelog from git commit history.
    
    Args:
        params:
            - repo_path (str, optional): Path to git repository
            - since (str, optional): Start point (date/tag/commit)
            - until (str, optional): End point (default: HEAD)
            - version (str, optional): Version number
            - output_file (str, optional): Output file path
            - style_guide (str, optional): Path to style guide
            - exclude_patterns (list, optional): Patterns to exclude
    
    Returns:
        Dictionary with changelog content, output path, stats
    """
    status.set_callback(params.pop('_status_callback', None))

    repo_path = params.get('repo_path', os.getcwd())
    since = params.get('since', 'last-release')
    until = params.get('until', 'HEAD')
    version = params.get('version', None)
    output_file = params.get('output_file', 'CHANGELOG.md')
    style_guide_path = params.get('style_guide')
    exclude_patterns = params.get('exclude_patterns', ['refactor', 'test', 'chore', 'ci', 'docs'])
    
    if not GIT_AVAILABLE:
        return {
            'success': False,
            'error': 'gitpython not installed. Install with: pip install gitpython'
        }
    
    try:
        repo = git.Repo(repo_path)
    except Exception as e:
        return {
            'success': False,
            'error': f'Failed to open git repository: {e}'
        }
    
    # Get commits
    try:
        if since == 'last-release':
            # Find last tag
            tags = sorted(repo.tags, key=lambda t: t.commit.committed_datetime, reverse=True)
            if tags:
                since = tags[0].name
            else:
                # No tags, use first commit
                since = repo.git.rev_list('--max-parents=0', 'HEAD').split('\n')[0]
        
        commits = list(repo.iter_commits(f'{since}..{until}'))
        
        if not commits:
            return {
                'success': False,
                'error': f'No commits found between {since} and {until}'
            }
    except Exception as e:
        return {
            'success': False,
            'error': f'Failed to get commits: {e}'
        }
    
    # Filter commits
    filtered_commits = []
    for commit in commits:
        message_lower = commit.message.lower()
        if not any(pattern in message_lower for pattern in exclude_patterns):
            filtered_commits.append(commit)
    
    # Categorize commits using AI
    categorized = await _categorize_commits(filtered_commits)
    
    # Generate changelog
    changelog = await _generate_changelog_content(
        categorized, version, since, until, style_guide_path
    )
    
    # Save to file
    output_path = Path(repo_path) / output_file
    try:
        output_path.write_text(changelog, encoding='utf-8')
    except Exception as e:
        return {
            'success': False,
            'error': f'Failed to write changelog: {e}'
        }
    
    stats = {
        'total_commits': len(commits),
        'filtered_commits': len(filtered_commits),
        'categories': {k: len(v) for k, v in categorized.items()}
    }
    
    return {
        'success': True,
        'changelog': changelog,
        'output_path': str(output_path),
        'stats': stats
    }


async def _categorize_commits(commits: List) -> Dict[str, List]:
    """Categorize commits into features, improvements, fixes, etc."""
    categories = {
        'features': [],
        'improvements': [],
        'fixes': [],
        'security': [],
        'breaking': [],
        'other': []
    }
    
    # Use AI to categorize if available
    try:
        try:
            from Jotty.core.registry.skills_registry import get_skills_registry
        except ImportError:
            from Jotty.core.registry.skills_registry import get_skills_registry
        
        registry = get_skills_registry()
        registry.init()
        claude_skill = registry.get_skill('claude-cli-llm')
        
        if claude_skill:
            generate_tool = claude_skill.tools.get('generate_text_tool')
            
            if generate_tool:
                # Prepare commit messages
                commit_text = "\n".join([
                    f"{i+1}. {commit.message.strip()[:200]}"
                    for i, commit in enumerate(commits[:50])  # Limit to 50 for prompt size
                ])
                
                prompt = f"""Categorize these git commits into: features, improvements, fixes, security, breaking, or other.

For each commit, return:
- Category (one word: features/improvements/fixes/security/breaking/other)
- User-friendly description (one sentence, customer-facing)

Commits:
{commit_text}

Return JSON format:
{{
  "1": {{"category": "features", "description": "User-friendly description"}},
  "2": {{"category": "fixes", "description": "User-friendly description"}},
  ...
}}"""

                if inspect.iscoroutinefunction(generate_tool):
                    result = await generate_tool({
                        'prompt': prompt,
                        'model': 'sonnet',
                        'timeout': 60
                    })
                else:
                    result = generate_tool({
                        'prompt': prompt,
                        'model': 'sonnet',
                        'timeout': 60
                    })
                
                if result.get('success'):
                    import json
                    text = result.get('text', '')
                    json_match = re.search(r'\{.*\}', text, re.DOTALL)
                    if json_match:
                        try:
                            categorized_data = json.loads(json_match.group())
                            for i, commit in enumerate(commits[:50]):
                                key = str(i+1)
                                if key in categorized_data:
                                    cat = categorized_data[key].get('category', 'other')
                                    desc = categorized_data[key].get('description', commit.message)
                                    if cat in categories:
                                        categories[cat].append({
                                            'commit': commit,
                                            'description': desc
                                        })
                                    else:
                                        categories['other'].append({
                                            'commit': commit,
                                            'description': desc
                                        })
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse AI categorization, using pattern matching")
    except Exception as e:
        logger.debug(f"AI categorization failed: {e}, using pattern matching")
    
    # Fallback: pattern-based categorization
    if not any(categories.values()):
        for commit in commits:
            msg_lower = commit.message.lower()
            if any(word in msg_lower for word in ['feat', 'add', 'new', 'implement']):
                categories['features'].append({'commit': commit, 'description': commit.message})
            elif any(word in msg_lower for word in ['fix', 'bug', 'resolve', 'correct']):
                categories['fixes'].append({'commit': commit, 'description': commit.message})
            elif any(word in msg_lower for word in ['security', 'vulnerability', 'cve']):
                categories['security'].append({'commit': commit, 'description': commit.message})
            elif any(word in msg_lower for word in ['break', 'remove', 'deprecate']):
                categories['breaking'].append({'commit': commit, 'description': commit.message})
            elif any(word in msg_lower for word in ['improve', 'enhance', 'optimize', 'update']):
                categories['improvements'].append({'commit': commit, 'description': commit.message})
            else:
                categories['other'].append({'commit': commit, 'description': commit.message})
    
    return categories


async def _generate_changelog_content(
    categorized: Dict[str, List],
    version: Optional[str],
    since: str,
    until: str,
    style_guide_path: Optional[str]
) -> str:
    """Generate changelog markdown content."""
    
    # Read style guide if provided
    style_guide = ""
    if style_guide_path and os.path.exists(style_guide_path):
        style_guide = Path(style_guide_path).read_text(encoding='utf-8')
    
    # Build changelog
    lines = []
    
    if version:
        lines.append(f"# Version {version}")
        lines.append("")
        lines.append(f"*Released: {datetime.now().strftime('%Y-%m-%d')}*")
    else:
        lines.append(f"# Changelog")
        lines.append("")
        lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d')}*")
    
    lines.append("")
    lines.append(f"*Changes from {since} to {until}*")
    lines.append("")
    
    # Features
    if categorized['features']:
        lines.append("## ✨ New Features")
        lines.append("")
        for item in categorized['features']:
            desc = item['description']
            # Clean up description
            desc = re.sub(r'^(feat|feature):\s*', '', desc, flags=re.IGNORECASE)
            desc = desc.strip()
            if not desc.endswith('.'):
                desc += '.'
            lines.append(f"- {desc}")
        lines.append("")
    
    # Improvements
    if categorized['improvements']:
        lines.append("## 🔧 Improvements")
        lines.append("")
        for item in categorized['improvements']:
            desc = item['description']
            desc = re.sub(r'^(improve|enhance|update):\s*', '', desc, flags=re.IGNORECASE)
            desc = desc.strip()
            if not desc.endswith('.'):
                desc += '.'
            lines.append(f"- {desc}")
        lines.append("")
    
    # Fixes
    if categorized['fixes']:
        lines.append("## 🐛 Bug Fixes")
        lines.append("")
        for item in categorized['fixes']:
            desc = item['description']
            desc = re.sub(r'^(fix|bugfix):\s*', '', desc, flags=re.IGNORECASE)
            desc = desc.strip()
            if not desc.endswith('.'):
                desc += '.'
            lines.append(f"- {desc}")
        lines.append("")
    
    # Security
    if categorized['security']:
        lines.append("## 🔒 Security")
        lines.append("")
        for item in categorized['security']:
            desc = item['description']
            lines.append(f"- {desc}")
        lines.append("")
    
    # Breaking changes
    if categorized['breaking']:
        lines.append("## ⚠️ Breaking Changes")
        lines.append("")
        for item in categorized['breaking']:
            desc = item['description']
            lines.append(f"- {desc}")
        lines.append("")
    
    # Other
    if categorized['other']:
        lines.append("## 📝 Other Changes")
        lines.append("")
        for item in categorized['other']:
            desc = item['description']
            lines.append(f"- {desc}")
        lines.append("")
    
    return "\n".join(lines)
