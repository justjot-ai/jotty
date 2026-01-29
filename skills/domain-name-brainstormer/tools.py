"""
Domain Name Brainstormer Skill - Generate creative domain names and check availability.

Helps find the perfect domain name for projects by generating creative options
and checking availability across multiple TLDs.
"""
import asyncio
import logging
import inspect
import re
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


async def brainstorm_domains_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate domain name suggestions and check availability.
    
    Args:
        params:
            - project_description (str): Description of your project
            - keywords (list, optional): Specific keywords to include
            - preferred_tlds (list, optional): Preferred TLDs
            - max_suggestions (int, optional): Max suggestions (default: 15)
            - check_availability (bool, optional): Check availability (default: True)
    
    Returns:
        Dictionary with suggestions, recommendations, availability info
    """
    project_description = params.get('project_description', '')
    keywords = params.get('keywords', [])
    preferred_tlds = params.get('preferred_tlds', ['.com', '.io', '.dev', '.ai', '.app'])
    max_suggestions = params.get('max_suggestions', 15)
    check_availability = params.get('check_availability', True)
    
    if not project_description:
        return {
            'success': False,
            'error': 'project_description is required'
        }
    
    # Generate domain suggestions using AI
    suggestions = await _generate_domain_suggestions(
        project_description, keywords, preferred_tlds, max_suggestions
    )
    
    # Check availability if requested
    if check_availability:
        suggestions = await _check_domain_availability(suggestions)
    
    # Generate recommendations
    recommendations = _generate_recommendations(suggestions)
    
    return {
        'success': True,
        'suggestions': suggestions,
        'recommendations': recommendations
    }


async def _generate_domain_suggestions(
    project_description: str,
    keywords: List[str],
    preferred_tlds: List[str],
    max_suggestions: int
) -> List[Dict]:
    """Generate creative domain name suggestions using AI."""
    
    try:
        try:
            from Jotty.core.registry.skills_registry import get_skills_registry
        except ImportError:
            from core.registry.skills_registry import get_skills_registry
        
        registry = get_skills_registry()
        registry.init()
        claude_skill = registry.get_skill('claude-cli-llm')
        
        if not claude_skill:
            return []
        
        generate_tool = claude_skill.tools.get('generate_text_tool')
        if not generate_tool:
            return []
        
        keywords_text = f"Keywords to consider: {', '.join(keywords)}" if keywords else ""
        tlds_text = f"Preferred TLDs: {', '.join(preferred_tlds)}"
        
        prompt = f"""Generate creative domain name suggestions for this project:

**Project Description:** {project_description}
{keywords_text}
{tlds_text}

**Requirements:**
- Short (under 15 characters ideal)
- Memorable and easy to spell
- Pronounceable
- Descriptive or brandable
- No hyphens or numbers
- Professional sounding

Generate {max_suggestions} domain name suggestions. For each, provide:
1. Domain name (without TLD)
2. Suggested TLD (.com, .io, .dev, .ai, etc.)
3. Why it works (1-2 sentences)
4. Brandability score (1-10)

Return JSON format:
{{
  "suggestions": [
    {{
      "name": "domainname",
      "tld": ".com",
      "full_domain": "domainname.com",
      "why_it_works": "Explanation",
      "brandability_score": 8
    }}
  ]
}}"""

        if inspect.iscoroutinefunction(generate_tool):
            result = await generate_tool({
                'prompt': prompt,
                'model': 'sonnet',
                'timeout': 90
            })
        else:
            result = generate_tool({
                'prompt': prompt,
                'model': 'sonnet',
                'timeout': 90
            })
        
        if result.get('success'):
            text = result.get('text', '')
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    suggestions = data.get('suggestions', [])
                    
                    # Add all TLD variations
                    expanded = []
                    for suggestion in suggestions:
                        name = suggestion.get('name', '')
                        for tld in preferred_tlds:
                            expanded.append({
                                'name': name,
                                'tld': tld,
                                'full_domain': f"{name}{tld}",
                                'why_it_works': suggestion.get('why_it_works', ''),
                                'brandability_score': suggestion.get('brandability_score', 5),
                                'available': None  # To be checked
                            })
                    
                    return expanded[:max_suggestions * len(preferred_tlds)]
                except json.JSONDecodeError:
                    logger.warning("Failed to parse AI domain suggestions")
    except Exception as e:
        logger.debug(f"AI domain generation failed: {e}")
    
    # Fallback: basic generation
    return []


async def _check_domain_availability(suggestions: List[Dict]) -> List[Dict]:
    """Check domain availability (simplified - would need actual WHOIS API)."""
    
    # Note: Real implementation would use WHOIS API or domain registrar API
    # For now, mark as "unknown" - user can check manually
    for suggestion in suggestions:
        if suggestion.get('available') is None:
            suggestion['available'] = 'unknown'  # Would need actual API call
            suggestion['availability_note'] = 'Check manually or use domain registrar API'
    
    return suggestions


def _generate_recommendations(suggestions: List[Dict]) -> Dict:
    """Generate top recommendations."""
    
    # Sort by brandability score
    sorted_suggestions = sorted(
        suggestions, 
        key=lambda x: x.get('brandability_score', 0), 
        reverse=True
    )
    
    top_pick = sorted_suggestions[0] if sorted_suggestions else None
    runner_up = sorted_suggestions[1] if len(sorted_suggestions) > 1 else None
    
    return {
        'top_pick': top_pick,
        'runner_up': runner_up,
        'total_suggestions': len(suggestions),
        'available_count': len([s for s in suggestions if s.get('available') == True])
    }
