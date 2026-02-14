"""
Skill Discovery Tool

Provides tools for agents to discover and understand available skills.
Uses the SkillsManifest for categorization and metadata.
"""
import logging
from typing import Dict, Any, List, Optional

from Jotty.core.utils.skill_status import SkillStatus
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper

# Status emitter for progress updates
status = SkillStatus("skill-discovery")


logger = logging.getLogger(__name__)


@tool_wrapper()
def list_categories_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List all skill categories.

    Args:
        params: Dictionary (can be empty)

    Returns:
        Dictionary with:
            - success (bool): Whether operation succeeded
            - categories (list): List of category info
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        from Jotty.core.registry.skills_manifest import get_skills_manifest

        manifest = get_skills_manifest()
        categories = []

        for cat in manifest.get_categories():
            categories.append({
                "name": cat.name,
                "icon": cat.icon,
                "description": cat.description,
                "skill_count": len(cat.skills)
            })

        return {
            "success": True,
            "categories": categories,
            "total": len(categories)
        }

    except Exception as e:
        logger.error(f"Failed to list categories: {e}")
        return {"success": False, "error": str(e)}


@tool_wrapper()
def list_skills_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List skills with optional filtering.

    Args:
        params: Dictionary containing:
            - category (str, optional): Filter by category
            - tag (str, optional): Filter by tag
            - search (str, optional): Search query
            - include_uncategorized (bool, optional): Include uncategorized skills

    Returns:
        Dictionary with:
            - success (bool): Whether operation succeeded
            - skills (list): List of skill info
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        from Jotty.core.registry.skills_manifest import get_skills_manifest

        manifest = get_skills_manifest()

        category = params.get('category')
        tag = params.get('tag')
        search = params.get('search')
        include_uncategorized = params.get('include_uncategorized', True)

        if category:
            skills = manifest.get_skills_by_category(category)
        elif tag:
            skills = manifest.get_skills_by_tag(tag)
        elif search:
            skills = manifest.search_skills(search)
        else:
            skills = manifest.get_all_skills()

        if not include_uncategorized:
            skills = [s for s in skills if s.category != "uncategorized"]

        result = []
        for skill in skills:
            result.append({
                "name": skill.name,
                "category": skill.category,
                "icon": skill.icon,
                "tags": skill.tags,
                "requires_auth": skill.requires_auth,
                "env_vars": skill.env_vars,
                "is_new": skill.is_discovered
            })

        return {
            "success": True,
            "skills": result,
            "total": len(result)
        }

    except Exception as e:
        logger.error(f"Failed to list skills: {e}")
        return {"success": False, "error": str(e)}


@tool_wrapper()
def get_skill_info_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get detailed information about a specific skill.

    Args:
        params: Dictionary containing:
            - skill_name (str, required): Name of the skill

    Returns:
        Dictionary with skill details and available tools
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        from Jotty.core.registry.skills_manifest import get_skills_manifest
        try:
            from Jotty.core.registry.skills_registry import get_skills_registry
        except ImportError:
            from Jotty.core.registry.skills_registry import get_skills_registry

        skill_name = params.get('skill_name')
        if not skill_name:
            return {"success": False, "error": "skill_name parameter required"}

        manifest = get_skills_manifest()
        skill_info = manifest.get_skill(skill_name)

        if not skill_info:
            return {"success": False, "error": f"Skill not found: {skill_name}"}

        # Get tools from registry
        registry = get_skills_registry()
        registry.init()
        skill = registry.get_skill(skill_name)

        tools = []
        if skill and skill.tools:
            for tool_name, tool_func in skill.tools.items():
                doc = tool_func.__doc__ or ""
                # Extract first line of docstring as description
                desc = doc.strip().split('\n')[0] if doc else ""
                tools.append({
                    "name": tool_name,
                    "description": desc
                })

        return {
            "success": True,
            "skill": {
                "name": skill_info.name,
                "category": skill_info.category,
                "icon": skill_info.icon,
                "tags": skill_info.tags,
                "requires_auth": skill_info.requires_auth,
                "env_vars": skill_info.env_vars,
                "requires_cli": skill_info.requires_cli
            },
            "tools": tools,
            "tool_count": len(tools)
        }

    except Exception as e:
        logger.error(f"Failed to get skill info: {e}")
        return {"success": False, "error": str(e)}


@tool_wrapper()
def get_discovery_summary_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a summary of all skills for agent discovery.

    Args:
        params: Dictionary containing:
            - format (str, optional): 'json' or 'markdown' (default: 'json')

    Returns:
        Dictionary with skill summary
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        from Jotty.core.registry.skills_manifest import get_skills_manifest

        manifest = get_skills_manifest()
        format_type = params.get('format', 'json')

        if format_type == 'markdown':
            return {
                "success": True,
                "summary": manifest.get_discovery_prompt()
            }
        else:
            return {
                "success": True,
                "summary": manifest.get_summary()
            }

    except Exception as e:
        logger.error(f"Failed to get discovery summary: {e}")
        return {"success": False, "error": str(e)}


@tool_wrapper()
def refresh_manifest_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Refresh the skills manifest (reload + discover new skills).

    Args:
        params: Dictionary (can be empty)

    Returns:
        Dictionary with refresh status
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        from Jotty.core.registry.skills_manifest import get_skills_manifest

        manifest = get_skills_manifest(refresh=True)

        uncategorized = manifest.get_uncategorized_skills()

        return {
            "success": True,
            "total_skills": len(manifest.get_all_skills()),
            "total_categories": len(manifest.get_categories()),
            "uncategorized_count": len(uncategorized),
            "uncategorized_skills": [s.name for s in uncategorized]
        }

    except Exception as e:
        logger.error(f"Failed to refresh manifest: {e}")
        return {"success": False, "error": str(e)}


@tool_wrapper()
def categorize_skill_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Categorize a skill (move to a category).

    Args:
        params: Dictionary containing:
            - skill_name (str, required): Name of the skill
            - category (str, required): Target category

    Returns:
        Dictionary with operation status
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        from Jotty.core.registry.skills_manifest import get_skills_manifest

        skill_name = params.get('skill_name')
        category = params.get('category')

        if not skill_name:
            return {"success": False, "error": "skill_name parameter required"}
        if not category:
            return {"success": False, "error": "category parameter required"}

        manifest = get_skills_manifest()

        if manifest.add_skill_to_category(skill_name, category):
            return {
                "success": True,
                "skill": skill_name,
                "category": category,
                "message": f"Skill '{skill_name}' moved to category '{category}'"
            }
        else:
            return {
                "success": False,
                "error": f"Failed to categorize skill. Check skill and category names."
            }

    except Exception as e:
        logger.error(f"Failed to categorize skill: {e}")
        return {"success": False, "error": str(e)}


@tool_wrapper()
def find_skills_for_task_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find relevant skills for a given task description.

    Args:
        params: Dictionary containing:
            - task (str, required): Description of the task
            - max_results (int, optional): Maximum skills to return (default: 10)
            - use_llm (bool, optional): Use LLM for semantic matching (default: True)

    Returns:
        Dictionary with recommended skills
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        from Jotty.core.registry.skills_manifest import get_skills_manifest

        task = params.get('task', '').lower()
        max_results = params.get('max_results', 10)
        use_llm = params.get('use_llm', True)

        if not task:
            return {"success": False, "error": "task parameter required"}

        manifest = get_skills_manifest()

        # Keyword mapping to categories/skills
        keyword_map = {
            # Documents
            'pdf': ['pdf-tools', 'document-converter', 'research-to-pdf'],
            'excel': ['xlsx-tools'],
            'spreadsheet': ['xlsx-tools'],
            'word': ['docx-tools'],
            'document': ['docx-tools', 'pdf-tools', 'document-converter'],
            'powerpoint': ['pptx-editor', 'slide-generator', 'presenton'],
            'slides': ['slide-generator', 'pptx-editor', 'presenton'],
            'presentation': ['slide-generator', 'pptx-editor', 'presenton'],

            # Media
            'image': ['image-generator', 'openai-image-gen', 'algorithmic-art'],
            'gif': ['gif-creator'],
            'video': ['youtube-downloader'],
            'audio': ['openai-whisper-api'],
            'transcribe': ['openai-whisper-api'],
            'art': ['algorithmic-art', 'image-generator'],

            # Communication
            'telegram': ['telegram-sender'],
            'slack': ['slack'],
            'discord': ['discord'],
            'message': ['telegram-sender', 'slack', 'discord'],
            'send': ['telegram-sender', 'slack', 'discord'],

            # Research
            'search': ['web-search', 'web-scraper'],
            'research': ['stock-research-comprehensive', 'web-search', 'summarize', 'research-to-notion'],
            'leads': ['lead-research-assistant', 'research-to-notion'],
            'competitor': ['competitive-ads-extractor', 'product-launch-pipeline'],
            'stock': ['stock-research-comprehensive', 'stock-research-deep', 'screener-financials'],
            'finance': ['investing-commodities', 'screener-financials'],
            'summarize': ['summarize', 'claude-cli-llm'],
            
            # Composite workflows - Research & Documentation
            'research and document': ['research-to-notion'],
            'research to notion': ['research-to-notion'],
            'research workflow': ['research-to-notion'],
            
            # Composite workflows - Meeting Intelligence
            'meeting': ['meeting-insights-analyzer', 'meeting-intelligence-pipeline', 'notion-meeting-intelligence'],
            'meeting analysis': ['meeting-intelligence-pipeline'],
            'meeting intelligence': ['meeting-intelligence-pipeline'],
            'prepare meeting': ['meeting-intelligence-pipeline'],
            'meeting materials': ['meeting-intelligence-pipeline'],
            
            # Composite workflows - Content & Branding
            'domain': ['domain-name-brainstormer', 'product-launch-pipeline', 'content-branding-pipeline'],
            'brand': ['brand-guidelines', 'content-branding-pipeline'],
            'branding': ['brand-guidelines', 'content-branding-pipeline'],
            'theme': ['theme-factory', 'content-branding-pipeline'],
            'content creation': ['content-research-writer', 'content-branding-pipeline'],
            'branded content': ['content-branding-pipeline'],
            
            # Composite workflows - Development
            'changelog': ['changelog-generator', 'dev-workflow'],
            'create skill': ['skill-creator', 'dev-workflow'],
            'test webapp': ['webapp-testing', 'dev-workflow'],
            'test app': ['webapp-testing', 'dev-workflow'],
            'development workflow': ['dev-workflow'],
            'dev workflow': ['dev-workflow'],
            
            # Composite workflows - Media Production
            'enhance image': ['image-enhancer', 'media-production-pipeline'],
            'create gif': ['slack-gif-creator', 'gif-creator', 'media-production-pipeline'],
            'media production': ['media-production-pipeline'],
            'design': ['canvas-design', 'media-production-pipeline'],
            
            # Composite workflows - Notion Knowledge
            'knowledge': ['notion-knowledge-capture', 'notion-knowledge-pipeline'],
            'capture knowledge': ['notion-knowledge-pipeline'],
            'knowledge pipeline': ['notion-knowledge-pipeline'],
            'implementation plan': ['notion-spec-to-implementation', 'notion-knowledge-pipeline'],
            
            # Composite workflows - Product Launch
            'launch': ['product-launch-pipeline'],
            'product launch': ['product-launch-pipeline'],
            'launch product': ['product-launch-pipeline'],
            'product launch workflow': ['product-launch-pipeline'],

            # Productivity
            'github': ['github'],
            'notion': ['notion'],
            'trello': ['trello'],
            'notes': ['obsidian', 'notion'],
            'task': ['trello', 'notion'],

            # AI
            'ai': ['claude-cli-llm', 'gemini', 'openai-image-gen'],
            'llm': ['claude-cli-llm', 'gemini'],
            'claude': ['claude-cli-llm'],
            'gemini': ['gemini'],
            'openai': ['openai-image-gen', 'openai-whisper-api'],
            'gpt': ['openai-image-gen'],

            # Code Generation & Development
            'generate code': ['file-operations', 'skill-creator', 'dev-workflow'],
            'write code': ['file-operations', 'skill-creator'],
            'create code': ['file-operations', 'skill-creator'],
            'code generation': ['file-operations', 'skill-creator'],
            'implement': ['file-operations', 'skill-creator', 'dev-workflow'],
            'develop': ['file-operations', 'skill-creator', 'dev-workflow'],
            'programming': ['file-operations', 'skill-creator'],
            'write file': ['file-operations'],
            'create file': ['file-operations'],
            'code': ['file-operations', 'skill-creator'],
            
            # Other
            'weather': ['weather-checker'],
            'music': ['spotify'],
            'spotify': ['spotify'],
            'youtube': ['youtube-downloader'],
            'mindmap': ['mindmap-generator'],
            'file': ['file-operations'],
            'text': ['text-utils', 'text-chunker'],

            # Research/comparison patterns
            'vs': ['web-search', 'summarize', 'claude-cli-llm', 'slide-generator'],
            'versus': ['web-search', 'summarize', 'claude-cli-llm', 'slide-generator'],
            'compare': ['web-search', 'summarize', 'claude-cli-llm', 'slide-generator'],
            'comparison': ['web-search', 'summarize', 'claude-cli-llm', 'slide-generator'],
            'difference': ['web-search', 'summarize', 'claude-cli-llm'],
            'explain': ['web-search', 'summarize', 'claude-cli-llm'],
            'what is': ['web-search', 'summarize', 'claude-cli-llm'],
            'how does': ['web-search', 'summarize', 'claude-cli-llm'],
            'learn': ['web-search', 'summarize', 'slide-generator'],
            'tutorial': ['web-search', 'summarize', 'slide-generator'],

            # Technical topics trigger research
            'neural': ['web-search', 'summarize', 'claude-cli-llm', 'slide-generator'],
            'network': ['web-search', 'summarize', 'claude-cli-llm'],
            'machine learning': ['web-search', 'summarize', 'claude-cli-llm', 'slide-generator'],
            'deep learning': ['web-search', 'summarize', 'claude-cli-llm', 'slide-generator'],
            'algorithm': ['web-search', 'summarize', 'claude-cli-llm'],
            'model': ['web-search', 'summarize', 'claude-cli-llm'],
            'transformer': ['web-search', 'summarize', 'claude-cli-llm', 'slide-generator'],
            'rnn': ['web-search', 'summarize', 'claude-cli-llm', 'slide-generator'],
            'cnn': ['web-search', 'summarize', 'claude-cli-llm', 'slide-generator'],
            'lstm': ['web-search', 'summarize', 'claude-cli-llm'],
            'bert': ['web-search', 'summarize', 'claude-cli-llm'],
        }

        # Find matching skills
        matched_skills = set()

        # Check multi-word keywords first (longer matches take priority)
        # This ensures "generate code" matches before just "code"
        sorted_keywords = sorted(keyword_map.keys(), key=len, reverse=True)
        matched_keywords = set()
        for keyword in sorted_keywords:
            if keyword in task:
                # Skip if this keyword is a substring of an already matched keyword
                # (e.g., if "generate code" matched, don't also match "code")
                is_substring = any(kw != keyword and keyword in kw for kw in matched_keywords)
                if not is_substring:
                    matched_skills.update(keyword_map[keyword])
                    matched_keywords.add(keyword)

        # Also search in manifest (by name, category, tags)
        search_results = manifest.search_skills(task)
        for skill in search_results:
            matched_skills.add(skill.name)
        
        # Search for composite/pipeline workflows in task
        # Check if task mentions multiple actions that suggest a composite workflow
        workflow_keywords = {
            'research-to-notion': ['research', 'leads', 'competitor', 'document', 'notion'],
            'meeting-intelligence-pipeline': ['meeting', 'analyze', 'prepare', 'materials', 'insights'],
            'content-branding-pipeline': ['domain', 'brand', 'theme', 'content', 'create'],
            'dev-workflow': ['changelog', 'skill', 'test', 'webapp', 'development'],
            'media-production-pipeline': ['enhance', 'image', 'gif', 'design', 'media'],
            'notion-knowledge-pipeline': ['knowledge', 'capture', 'research', 'implementation', 'plan'],
            'product-launch-pipeline': ['launch', 'product', 'domain', 'competitor', 'leads', 'content']
        }
        
        # Check if task matches multiple keywords for a composite workflow
        task_lower = task.lower()
        for composite_name, keywords in workflow_keywords.items():
            matches = sum(1 for kw in keywords if kw in task_lower)
            if matches >= 2:  # At least 2 keywords match
                matched_skills.add(composite_name)

        # If no matches and use_llm, try to infer intent
        if not matched_skills and use_llm:
            # Infer task type from patterns
            inferred_skills = _infer_skills_from_task(task, manifest)
            matched_skills.update(inferred_skills)

        # Get skill info for matches
        results = []
        for skill_name in list(matched_skills)[:max_results]:
            skill = manifest.get_skill(skill_name)
            if skill:
                results.append({
                    "name": skill.name,
                    "category": skill.category,
                    "icon": skill.icon,
                    "requires_auth": skill.requires_auth
                })

        return {
            "success": True,
            "task": task,
            "recommended_skills": results,
            "total": len(results)
        }

    except Exception as e:
        logger.error(f"Failed to find skills: {e}")
        return {"success": False, "error": str(e)}


def _infer_skills_from_task(task: str, manifest) -> set:
    """Infer skills from task patterns when no keywords match."""
    inferred = set()

    # Pattern detection
    words = task.split()

    # If task has technical acronyms or jargon, likely needs research
    has_acronym = any(word.isupper() and len(word) >= 2 for word in words)
    has_comparison = any(w in task for w in ['vs', 'versus', 'or', 'and', 'compare'])
    has_question = any(task.startswith(w) for w in ['what', 'how', 'why', 'when', 'where', 'which'])

    if has_acronym or has_question:
        inferred.update(['web-search', 'summarize', 'claude-cli-llm'])

    if has_comparison:
        inferred.update(['web-search', 'summarize', 'claude-cli-llm', 'slide-generator'])

    # Short tasks (< 5 words) with no clear action likely need research
    if len(words) <= 5 and not any(w in task for w in ['create', 'make', 'send', 'generate', 'build']):
        inferred.update(['web-search', 'claude-cli-llm'])

    return inferred
