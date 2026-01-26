#!/usr/bin/env python3
"""
Run comprehensive stock research for Colgate Palmolive.
"""
import asyncio
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """Run comprehensive research for Colgate Palmolive."""
    try:
        from core.registry.skills_registry import get_skills_registry
        
        logger.info("🚀 Starting comprehensive stock research for Colgate Palmolive")
        
        # Initialize registry
        registry = get_skills_registry()
        registry.init()
        
        # Check if skill exists
        skill = registry.get_skill('stock-research-comprehensive')
        if not skill:
            logger.error("❌ stock-research-comprehensive skill not found")
            logger.info("💡 Make sure the skill is in ~/jotty/skills/stock-research-comprehensive/")
            return
        
        logger.info("✅ Skill found")
        
        # Get the tool
        tool = skill.tools.get('comprehensive_stock_research_tool')
        if not tool:
            logger.error("❌ comprehensive_stock_research_tool not found")
            return
        
        # Run research
        logger.info("📊 Running comprehensive research...")
        
        result = await tool({
            'ticker': 'CL',
            'company_name': 'Colgate Palmolive',
            'title': 'Colgate Palmolive (CL) - Comprehensive Research Report',
            'author': 'Jotty Stock Research',
            'send_telegram': True,
            'max_results_per_aspect': 10,
            'page_size': 'a4'
        })
        
        if result.get('success'):
            logger.info("✅ Research completed successfully!")
            logger.info(f"📄 Markdown: {result.get('md_path')}")
            logger.info(f"📑 PDF: {result.get('pdf_path')}")
            logger.info(f"📊 Fundamentals: {result.get('fundamentals_research', {}).get('count', 0)} results")
            logger.info(f"📈 Technicals: {result.get('technicals_research', {}).get('count', 0)} results")
            logger.info(f"📰 Broker Reports: {result.get('broker_research', {}).get('count', 0)} results")
            logger.info(f"📱 Telegram Sent: {result.get('telegram_sent', False)}")
        else:
            logger.error(f"❌ Research failed: {result.get('error')}")
            if 'fundamentals_research' in result:
                logger.info(f"Fundamentals: {result.get('fundamentals_research')}")
            if 'technicals_research' in result:
                logger.info(f"Technicals: {result.get('technicals_research')}")
            if 'broker_research' in result:
                logger.info(f"Broker: {result.get('broker_research')}")
    
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)


if __name__ == '__main__':
    asyncio.run(main())
