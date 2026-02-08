"""
Lead Research Assistant Skill - Identify and qualify high-quality leads.

Helps find potential customers by analyzing your product, understanding ICP,
and providing actionable outreach strategies.
"""
import asyncio
import logging
import inspect
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import os

logger = logging.getLogger(__name__)


async def research_leads_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Research and identify potential leads for your product/service.
    
    Args:
        params:
            - product_description (str): Description of your product/service
            - industry (str, optional): Target industry
            - location (str, optional): Geographic location
            - company_size (str, optional): Company size range
            - pain_points (list, optional): Pain points your product solves
            - technologies (list, optional): Technologies they might use
            - max_leads (int, optional): Maximum leads to find (default: 10)
            - output_format (str, optional): Output format (default: 'markdown')
    
    Returns:
        Dictionary with leads list, summary, output file path
    """
    product_description = params.get('product_description', '')
    industry = params.get('industry', '')
    location = params.get('location', '')
    company_size = params.get('company_size', '')
    pain_points = params.get('pain_points', [])
    technologies = params.get('technologies', [])
    max_leads = params.get('max_leads', 10)
    output_format = params.get('output_format', 'markdown')
    
    if not product_description:
        return {
            'success': False,
            'error': 'product_description is required'
        }
    
    # Build search queries
    search_queries = []
    
    # Base query
    base_query = f"{product_description} companies"
    if industry:
        base_query += f" {industry} industry"
    if location:
        base_query += f" {location}"
    if company_size:
        base_query += f" {company_size}"
    
    search_queries.append(base_query)
    
    # Add pain point queries
    for pain_point in pain_points[:3]:  # Limit to 3
        search_queries.append(f"companies struggling with {pain_point} {industry or ''}")
    
    # Search for companies
    try:
        try:
            from Jotty.core.registry.skills_registry import get_skills_registry
        except ImportError:
            from Jotty.core.registry.skills_registry import get_skills_registry
        
        registry = get_skills_registry()
        registry.init()
        web_search_skill = registry.get_skill('web-search')
        
        if not web_search_skill:
            return {
                'success': False,
                'error': 'web-search skill not available'
            }
        
        search_tool = web_search_skill.tools.get('search_web_tool')
        if not search_tool:
            return {
                'success': False,
                'error': 'search_web_tool not found'
            }
        
        # Perform searches
        all_results = []
        for query in search_queries:
            if inspect.iscoroutinefunction(search_tool):
                result = await search_tool({
                    'query': query,
                    'max_results': 10
                })
            else:
                result = search_tool({
                    'query': query,
                    'max_results': 10
                })
            
            if result.get('success'):
                all_results.extend(result.get('results', []))
        
        # Analyze and qualify leads using AI
        leads = await _qualify_leads(
            all_results, product_description, industry, 
            pain_points, technologies, max_leads
        )
        
        # Generate output
        summary = {
            'total_found': len(leads),
            'high_priority': len([l for l in leads if l.get('fit_score', 0) >= 8]),
            'medium_priority': len([l for l in leads if 5 <= l.get('fit_score', 0) < 8]),
            'low_priority': len([l for l in leads if l.get('fit_score', 0) < 5])
        }
        
        # Save output
        output_file = None
        if output_format == 'markdown':
            output_content = _format_leads_markdown(leads, product_description, summary)
            output_file = f"leads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            Path(output_file).write_text(output_content, encoding='utf-8')
        elif output_format == 'json':
            output_content = json.dumps({'leads': leads, 'summary': summary}, indent=2)
            output_file = f"leads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            Path(output_file).write_text(output_content, encoding='utf-8')
        elif output_format == 'csv':
            output_content = _format_leads_csv(leads)
            output_file = f"leads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            Path(output_file).write_text(output_content, encoding='utf-8')
        
        return {
            'success': True,
            'leads': leads,
            'summary': summary,
            'output_file': output_file
        }
        
    except Exception as e:
        logger.error(f"Lead research failed: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


async def _qualify_leads(
    search_results: List[Dict],
    product_description: str,
    industry: str,
    pain_points: List[str],
    technologies: List[str],
    max_leads: int
) -> List[Dict]:
    """Qualify and score leads using AI analysis."""
    
    try:
        try:
            from Jotty.core.registry.skills_registry import get_skills_registry
        except ImportError:
            from Jotty.core.registry.skills_registry import get_skills_registry
        
        registry = get_skills_registry()
        registry.init()
        claude_skill = registry.get_skill('claude-cli-llm')
        
        if not claude_skill:
            return []
        
        generate_tool = claude_skill.tools.get('generate_text_tool')
        if not generate_tool:
            return []
        
        # Prepare search results summary
        results_text = "\n".join([
            f"{i+1}. {r.get('title', '')} - {r.get('url', '')}\n   {r.get('snippet', '')[:200]}"
            for i, r in enumerate(search_results[:30])  # Limit for prompt size
        ])
        
        prompt = f"""Analyze these search results and identify companies that would be good leads for this product:

**Product:** {product_description}
**Target Industry:** {industry or 'Any'}
**Pain Points Solved:** {', '.join(pain_points) if pain_points else 'Not specified'}
**Technologies:** {', '.join(technologies) if technologies else 'Not specified'}

**Search Results:**
{results_text}

For each potential lead company, provide:
1. Company name
2. Website URL
3. Why they're a good fit (2-3 specific reasons)
4. Fit score (1-10)
5. Target decision maker role/title
6. Value proposition for them
7. Outreach strategy (personalized approach)

Return JSON format:
{{
  "leads": [
    {{
      "company_name": "Company Name",
      "website": "https://...",
      "why_fit": "Reason 1, Reason 2, Reason 3",
      "fit_score": 8,
      "decision_maker": "VP of Engineering",
      "value_proposition": "How product helps them specifically",
      "outreach_strategy": "Personalized approach suggestion"
    }}
  ]
}}

Return up to {max_leads} best leads, sorted by fit_score descending."""

        if inspect.iscoroutinefunction(generate_tool):
            result = await generate_tool({
                'prompt': prompt,
                'model': 'sonnet',
                'timeout': 120
            })
        else:
            result = generate_tool({
                'prompt': prompt,
                'model': 'sonnet',
                'timeout': 120
            })
        
        if result.get('success'):
            import re
            text = result.get('text', '')
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    return data.get('leads', [])
                except json.JSONDecodeError:
                    logger.warning("Failed to parse AI lead qualification")
    except Exception as e:
        logger.debug(f"AI lead qualification failed: {e}")
    
    # Fallback: basic extraction
    leads = []
    seen_companies = set()
    
    for result in search_results[:max_leads]:
        title = result.get('title', '')
        url = result.get('url', '')
        
        # Extract company name from title/URL
        company_name = title.split(' - ')[0].split(' | ')[0]
        if company_name.lower() in seen_companies:
            continue
        
        seen_companies.add(company_name.lower())
        
        leads.append({
            'company_name': company_name,
            'website': url,
            'why_fit': result.get('snippet', '')[:200],
            'fit_score': 5,  # Default score
            'decision_maker': 'To be determined',
            'value_proposition': f'Could benefit from {product_description}',
            'outreach_strategy': 'Research company needs and personalize approach'
        })
    
    return leads[:max_leads]


def _format_leads_markdown(leads: List[Dict], product_description: str, summary: Dict) -> str:
    """Format leads as markdown."""
    lines = [
        "# Lead Research Results",
        "",
        f"**Product:** {product_description}",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"- Total leads found: {summary['total_found']}",
        f"- High priority (8-10): {summary['high_priority']}",
        f"- Medium priority (5-7): {summary['medium_priority']}",
        f"- Low priority (<5): {summary['low_priority']}",
        "",
        "---",
        ""
    ]
    
    for i, lead in enumerate(leads, 1):
        lines.extend([
            f"## Lead {i}: {lead.get('company_name', 'Unknown')}",
            "",
            f"**Website:** {lead.get('website', 'N/A')}",
            f"**Priority Score:** {lead.get('fit_score', 0)}/10",
            "",
            "**Why They're a Good Fit:**",
            lead.get('why_fit', 'Analysis pending'),
            "",
            f"**Target Decision Maker:** {lead.get('decision_maker', 'To be determined')}",
            "",
            "**Value Proposition:**",
            lead.get('value_proposition', 'Analysis pending'),
            "",
            "**Outreach Strategy:**",
            lead.get('outreach_strategy', 'Research and personalize'),
            "",
            "---",
            ""
        ])
    
    return "\n".join(lines)


def _format_leads_csv(leads: List[Dict]) -> str:
    """Format leads as CSV."""
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        'Company Name', 'Website', 'Fit Score', 'Decision Maker',
        'Why Fit', 'Value Proposition', 'Outreach Strategy'
    ])
    
    # Rows
    for lead in leads:
        writer.writerow([
            lead.get('company_name', ''),
            lead.get('website', ''),
            lead.get('fit_score', 0),
            lead.get('decision_maker', ''),
            lead.get('why_fit', ''),
            lead.get('value_proposition', ''),
            lead.get('outreach_strategy', '')
        ])
    
    return output.getvalue()
