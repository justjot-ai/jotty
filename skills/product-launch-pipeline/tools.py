"""
Product Launch Pipeline Composite Skill

Complete product launch workflow:
1. Brainstorm domain names (domain-name-brainstormer)
2. Research leads (lead-research-assistant)
3. Analyze competitors (competitive-ads-extractor)
4. Write content (content-research-writer)

Source → Processor → Processor → Sink pattern.
"""
import asyncio
import logging
import inspect
from typing import Dict, Any

from Jotty.core.utils.skill_status import SkillStatus

# Status emitter for progress updates
status = SkillStatus("product-launch-pipeline")


logger = logging.getLogger(__name__)


async def product_launch_pipeline_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Complete product launch workflow: domain → leads → competitors → content.
    
    Args:
        params:
            - product_description (str, required): Product description
            - product_name (str, optional): Product name (will brainstorm if not provided)
            - industry (str, optional): Target industry
            - location (str, optional): Geographic location
            - competitor_names (list, optional): List of competitor names
            - content_action (str, optional): 'outline', 'draft', 'full' (default: 'outline')
            - max_domain_suggestions (int, optional): Max domain suggestions (default: 10)
            - max_leads (int, optional): Max leads (default: 10)
            - max_ads_per_competitor (int, optional): Max ads per competitor (default: 5)
            - brainstorm_domains (bool, optional): Brainstorm domains (default: True)
            - research_leads (bool, optional): Research leads (default: True)
            - analyze_competitors (bool, optional): Analyze competitors (default: True)
            - write_content (bool, optional): Write content (default: True)
    
    Returns:
        Dictionary with domains, leads, competitor analysis, and content
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        try:
            from Jotty.core.registry.skills_registry import get_skills_registry
        except ImportError:
            from Jotty.core.registry.skills_registry import get_skills_registry
        
        registry = get_skills_registry()
        registry.init()
        
        product_description = params.get('product_description')
        if not product_description:
            return {
                'success': False,
                'error': 'product_description is required'
            }
        
        logger.info(f"🚀 Product Launch Pipeline: {product_description}")
        
        domains = []
        leads = []
        competitor_analysis = {}
        content_result = None
        
        # Step 1: Brainstorm domain names (Source)
        if params.get('brainstorm_domains', True):
            logger.info("💡 Step 1: Brainstorming domain names...")
            domain_skill = registry.get_skill('domain-name-brainstormer')
            if domain_skill:
                domain_tool = domain_skill.tools.get('brainstorm_domains_tool')
                if domain_tool:
                    domain_params = {
                        'project_description': product_description,
                        'max_suggestions': params.get('max_domain_suggestions', 10)
                    }
                    
                    if inspect.iscoroutinefunction(domain_tool):
                        domain_result = await domain_tool(domain_params)
                    else:
                        domain_result = domain_tool(domain_params)
                    
                    if domain_result.get('success'):
                        domains = domain_result.get('suggestions', [])
                        logger.info(f"✅ Generated {len(domains)} domain suggestions")
                    else:
                        logger.warning(f"⚠️  Domain brainstorming failed: {domain_result.get('error')}")
        
        product_name = params.get('product_name') or domains[0] if domains else 'product'
        
        # Step 2: Research leads (Processor)
        if params.get('research_leads', True):
            logger.info("👥 Step 2: Researching leads...")
            lead_skill = registry.get_skill('lead-research-assistant')
            if lead_skill:
                lead_tool = lead_skill.tools.get('research_leads_tool')
                if lead_tool:
                    lead_params = {
                        'product_description': product_description,
                        'max_leads': params.get('max_leads', 10),
                        **{k: v for k, v in params.items() if k in ['industry', 'location', 'company_size', 'pain_points', 'technologies']}
                    }
                    
                    if inspect.iscoroutinefunction(lead_tool):
                        lead_result = await lead_tool(lead_params)
                    else:
                        lead_result = lead_tool(lead_params)
                    
                    if lead_result.get('success'):
                        leads = lead_result.get('leads', [])
                        logger.info(f"✅ Found {len(leads)} leads")
                    else:
                        logger.warning(f"⚠️  Lead research failed: {lead_result.get('error')}")
        
        # Step 3: Analyze competitors (Processor)
        if params.get('analyze_competitors', True):
            logger.info("🔍 Step 3: Analyzing competitors...")
            competitor_names = params.get('competitor_names', [])
            
            if competitor_names:
                ads_skill = registry.get_skill('competitive-ads-extractor')
                if ads_skill:
                    ads_tool = ads_skill.tools.get('extract_competitive_ads_tool')
                    if ads_tool:
                        max_ads = params.get('max_ads_per_competitor', 5)
                        
                        # Analyze each competitor
                        for competitor_name in competitor_names:
                            ads_params = {
                                'competitor_name': competitor_name,
                                'max_ads': max_ads
                            }
                            
                            if inspect.iscoroutinefunction(ads_tool):
                                ads_result = await ads_tool(ads_params)
                            else:
                                ads_result = ads_tool(ads_params)
                            
                            if ads_result.get('success'):
                                competitor_analysis[competitor_name] = {
                                    'ads': ads_result.get('ads', []),
                                    'summary': ads_result.get('summary', '')
                                }
                                logger.info(f"✅ Analyzed {competitor_name}: {len(ads_result.get('ads', []))} ads")
                            else:
                                logger.warning(f"⚠️  Competitor analysis failed for {competitor_name}: {ads_result.get('error')}")
            else:
                logger.info("⚠️  No competitor names provided, skipping competitor analysis")
        
        # Step 4: Write content (Sink)
        if params.get('write_content', True):
            logger.info("✍️  Step 4: Writing content...")
            content_skill = registry.get_skill('content-research-writer')
            if content_skill:
                content_tool = content_skill.tools.get('write_content_with_research_tool')
                if content_tool:
                    # Build research context from leads and competitors
                    research_context = f"Product: {product_name}\nDescription: {product_description}\n\n"
                    
                    if leads:
                        research_context += f"Target Leads ({len(leads)}):\n"
                        for i, lead in enumerate(leads[:5], 1):
                            if isinstance(lead, dict):
                                name = lead.get('company_name', lead.get('name', ''))
                                desc = lead.get('description', lead.get('summary', ''))
                                research_context += f"{i}. {name}: {desc}\n"
                        research_context += "\n"
                    
                    if competitor_analysis:
                        research_context += "Competitor Analysis:\n"
                        for competitor, data in competitor_analysis.items():
                            research_context += f"- {competitor}: {len(data.get('ads', []))} ads analyzed\n"
                        research_context += "\n"
                    
                    content_action = params.get('content_action', 'outline')
                    content_params = {
                        'topic': f"{product_name} Launch Content",
                        'action': content_action,
                        'draft_content': research_context,
                        'research_topics': [product_description]
                    }
                    
                    if inspect.iscoroutinefunction(content_tool):
                        content_result = await content_tool(content_params)
                    else:
                        content_result = content_tool(content_params)
                    
                    if content_result.get('success'):
                        logger.info(f"✅ Content written: {content_action}")
                    else:
                        logger.warning(f"⚠️  Content writing failed: {content_result.get('error')}")
        
        return {
            'success': True,
            'product_description': product_description,
            'product_name': product_name,
            'domains': domains,
            'leads': leads,
            'competitor_analysis': competitor_analysis,
            'content': content_result
        }
        
    except Exception as e:
        logger.error(f"Product Launch Pipeline error: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Workflow failed: {str(e)}'
        }


__all__ = ['product_launch_pipeline_tool']
