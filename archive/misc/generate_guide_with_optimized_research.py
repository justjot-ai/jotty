#!/usr/bin/env python3
"""
Comprehensive Guide Generator with Optimized Multi-Agent Research
================================================================

Uses Jotty's multi-agent system + OptimizedWebSearchRAG:
1. Planner Agent: Determines what sections to generate
2. Research Agent: Uses OptimizedWebSearchRAG (Searx, Brave, multiple providers)
3. Content Generator: Creates comprehensive guide

Features:
- Multiple search providers with fallback
- Anti-CAPTCHA strategies
- Rate limiting and caching
- Better search results
"""

import sys
from pathlib import Path
import dspy
import logging
from typing import List

# Add Jotty to path
sys.path.insert(0, str(Path(__file__).parent))

from core.tools.content_generation import Document, SectionType, ContentGenerators
from core.integration.direct_claude_cli_lm import DirectClaudeCLI
from optimized_web_search_rag import OptimizedWebSearchRAG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# MULTI-AGENT SIGNATURES
# =============================================================================

class PlannerSignature(dspy.Signature):
    """Planner Agent - Determines what sections to generate for a topic"""
    topic: str = dspy.InputField(desc="Topic to create guide about")
    goal: str = dspy.InputField(desc="Specific goal for the guide")

    section_titles: str = dspy.OutputField(
        desc="List of 10-15 section titles that should be covered. "
        "Format: one title per line. "
        "Each title should be specific to the topic. "
        "Example for Poodles: History, Breed Types, Grooming, Health. "
        "Example for Python: Syntax Basics, Data Types, Control Flow, Functions."
    )
    research_queries: str = dspy.OutputField(
        desc="Search queries to research this topic. "
        "Format: one query per line. "
        "5-10 specific queries that will help gather comprehensive information."
    )


class ResearcherSignature(dspy.Signature):
    """Researcher Agent - Conducts web research"""
    topic: str = dspy.InputField(desc="Topic to research")
    queries: str = dspy.InputField(desc="Search queries (one per line)")
    research_data: str = dspy.InputField(desc="Raw research results from web")

    research_summary: str = dspy.OutputField(
        desc="Summary of research findings. "
        "Include key facts, statistics, and important information discovered. "
        "Write in flowing paragraphs."
    )


class ContentWriterSignature(dspy.Signature):
    """Content Writer Agent - Creates guide sections"""
    topic: str = dspy.InputField(desc="Topic of the guide")
    section_title: str = dspy.InputField(desc="Title of this section")
    research_context: str = dspy.InputField(desc="Research findings to use")

    section_content: str = dspy.OutputField(
        desc="Well-written section content for beginners. "
        "CRITICAL: Write in flowing paragraphs. "
        "DO NOT use hard line breaks within paragraphs. "
        "Use only natural paragraph breaks. "
        "Include specific details from research. "
        "Write 2-4 comprehensive paragraphs."
    )


# =============================================================================
# OPTIMIZED WEB SEARCH
# =============================================================================

def search_web_optimized(rag_tool: OptimizedWebSearchRAG, queries: List[str],
                        max_results_per_query: int = 3) -> List[dict]:
    """
    Search using OptimizedWebSearchRAG with multiple providers and fallback

    Args:
        rag_tool: Initialized OptimizedWebSearchRAG instance
        queries: List of search queries
        max_results_per_query: Max results per query

    Returns:
        List of search results with title, url, snippet
    """
    all_research = []

    for i, query in enumerate(queries, 1):
        print(f"   Query {i}/{len(queries)}: {query}")

        try:
            # Use OptimizedWebSearchRAG (tries Searx → Brave → Bing → etc.)
            results = rag_tool.search_and_extract(
                query=query,
                num_results=max_results_per_query,
                provider='searx'  # Start with Searx (free, no CAPTCHA)
            )

            if results:
                print(f"   ✅ Found {len(results)} results")
                for result in results:
                    all_research.append({
                        'query': query,
                        'title': result.get('title', ''),
                        'snippet': result.get('snippet', ''),
                        'url': result.get('url', '')
                    })
            else:
                print(f"   ⚠️  No results for this query")

        except Exception as e:
            print(f"   ⚠️  Search failed: {e}")

    return all_research


# =============================================================================
# MULTI-AGENT WORKFLOW
# =============================================================================

def run_multi_agent_research(topic: str, goal: str):
    """
    Run multi-agent workflow to create comprehensive guide

    Phase 1: Planning - Determine sections
    Phase 2: Research - Gather information with OptimizedWebSearchRAG
    Phase 3: Content Generation - Write guide
    """

    print("\n" + "=" * 100)
    print(f"  JOTTY OPTIMIZED MULTI-AGENT RESEARCH: '{topic} for Dummies'")
    print("=" * 100 + "\n")

    # Configure LLM
    print("🔧 Configuring Claude CLI...")
    try:
        lm = DirectClaudeCLI(model="haiku")
        dspy.configure(lm=lm)
        print("✅ Configured with Claude CLI (Haiku)\n")
    except Exception as e:
        print(f"❌ Failed to configure LLM: {e}")
        print("   Install: npm install -g @anthropic-ai/claude-code\n")
        return None

    # Initialize OptimizedWebSearchRAG
    print("🔧 Initializing OptimizedWebSearchRAG...")
    rag_tool = OptimizedWebSearchRAG(cache_dir="outputs/web_cache")
    print("✅ RAG tool initialized with Searx + multiple fallback providers\n")

    # =================================================================
    # PHASE 1: PLANNING
    # =================================================================

    print("=" * 100)
    print("  PHASE 1: PLANNING")
    print("=" * 100 + "\n")

    print("🤖 Agent 1: Planner")
    print(f"   Task: Determine sections for '{topic}'\n")

    planner = dspy.ChainOfThought(PlannerSignature)

    plan_result = planner(topic=topic, goal=goal)

    section_titles = [
        line.strip()
        for line in plan_result.section_titles.strip().split('\n')
        if line.strip()
    ]

    research_queries = [
        line.strip()
        for line in plan_result.research_queries.strip().split('\n')
        if line.strip()
    ]

    print(f"✅ Planner determined {len(section_titles)} sections:")
    for i, title in enumerate(section_titles, 1):
        print(f"   {i}. {title}")

    print(f"\n✅ Planner generated {len(research_queries)} research queries:")
    for i, query in enumerate(research_queries, 1):
        print(f"   {i}. {query}")

    # =================================================================
    # PHASE 2: OPTIMIZED WEB RESEARCH
    # =================================================================

    print("\n" + "=" * 100)
    print("  PHASE 2: OPTIMIZED WEB RESEARCH")
    print("=" * 100 + "\n")

    print("🔍 Searching with OptimizedWebSearchRAG (Searx + fallbacks)...\n")

    # Limit to first 5 queries to avoid rate limiting
    all_research = search_web_optimized(
        rag_tool=rag_tool,
        queries=research_queries[:5],
        max_results_per_query=3
    )

    print(f"\n✅ Gathered {len(all_research)} research results\n")

    # Synthesize research with LLM
    print("🤖 Agent 2: Researcher")
    print("   Task: Synthesize research findings\n")

    researcher = dspy.ChainOfThought(ResearcherSignature)

    research_context = "\n\n".join([
        f"Query: {r['query']}\nTitle: {r['title']}\nSnippet: {r['snippet']}\nURL: {r['url']}"
        for r in all_research
    ])

    research_result = researcher(
        topic=topic,
        queries="\n".join(research_queries[:5]),
        research_data=research_context if research_context else "No web results available. Use general knowledge."
    )

    print("✅ Research synthesized\n")

    # =================================================================
    # PHASE 3: CONTENT GENERATION
    # =================================================================

    print("=" * 100)
    print("  PHASE 3: CONTENT GENERATION")
    print("=" * 100 + "\n")

    print("🤖 Agent 3: Content Writer")
    print(f"   Task: Write {len(section_titles)} sections\n")

    writer = dspy.ChainOfThought(ContentWriterSignature)

    sections = []
    for i, title in enumerate(section_titles, 1):
        print(f"   Writing section {i}/{len(section_titles)}: {title}...")

        result = writer(
            topic=topic,
            section_title=title,
            research_context=research_result.research_summary
        )

        sections.append((title, result.section_content))
        print(f"   ✅ Completed ({len(result.section_content)} chars)")

    print(f"\n✅ All {len(sections)} sections generated!\n")

    return {
        'topic': topic,
        'sections': sections,
        'research_summary': research_result.research_summary,
        'research_sources': all_research
    }


def create_document(guide_data: dict) -> Document:
    """Create Document from multi-agent research results"""

    print("📄 Creating structured document...\n")

    doc = Document(
        title=f"{guide_data['topic']} for Dummies: A Comprehensive Guide",
        author="Jotty AI Optimized Multi-Agent Research Team",
        topic=guide_data['topic'],
        source_type="jotty-optimized-multi-agent-research"
    )

    # Add all sections
    for title, content in guide_data['sections']:
        doc.add_section(SectionType.TEXT, content, title=title)

    # Add research sources as appendix with shortened URLs
    if guide_data['research_sources']:
        sources_text = "This guide was created using research from the following sources:\n\n"
        for i, source in enumerate(guide_data['research_sources'][:10], 1):
            # Shorten URL if too long (helps with PDF wrapping)
            url = source['url']
            if len(url) > 80:
                url = url[:77] + "..."

            sources_text += f"{i}. **{source['title']}**\n"
            sources_text += f"   {url}\n\n"

        doc.add_section(SectionType.TEXT, sources_text, title="Research Sources")

    print(f"✅ Document created with {len(doc.sections)} sections")
    print(f"   Total content: {len(doc.full_content):,} characters\n")

    return doc


def generate_outputs(doc, topic: str):
    """Generate PDF, Markdown, HTML with optimized settings"""

    print("=" * 100)
    print("  GENERATING OUTPUT FILES")
    print("=" * 100 + "\n")

    topic_slug = topic.lower().replace(" ", "_").replace("/", "_")
    output_dir = Path(f"./outputs/{topic_slug}_guide")
    output_dir.mkdir(parents=True, exist_ok=True)

    generators = ContentGenerators()

    # Generate PDF with optimized margins and URL breaking
    print("📕 Generating PDF (optimized: 1in margins, URL wrapping)...")
    try:
        pdf_path = generators.generate_pdf(doc, output_path=output_dir, format='a4')
        print(f"✅ PDF: {pdf_path.name} ({pdf_path.stat().st_size/1024:.1f} KB)\n")
    except Exception as e:
        print(f"❌ PDF failed: {e}\n")
        return False

    # Generate Markdown
    print("📄 Generating Markdown...")
    try:
        md_path = generators.export_markdown(doc, output_path=output_dir)
        print(f"✅ Markdown: {md_path.name}\n")
    except Exception as e:
        print(f"⚠️  Markdown skipped: {e}\n")

    # Generate HTML
    print("🌐 Generating HTML...")
    try:
        html_path = generators.generate_html(doc, output_path=output_dir)
        print(f"✅ HTML: {html_path.name}\n")
    except Exception as e:
        print(f"⚠️  HTML skipped: {e}\n")

    print("=" * 100)
    print("  SUCCESS!")
    print("=" * 100 + "\n")

    print(f"📁 Output: {output_dir}")
    print(f"📚 Files:\n")

    for file in sorted(output_dir.glob("*")):
        size = file.stat().st_size / 1024
        print(f"   - {file.name} ({size:.1f} KB)")

    print(f"\n🎯 Main PDF: {pdf_path}")
    print(f"\n✨ Your researched '{topic} for Dummies' guide is ready!")
    print(f"   - 1 inch margins (reduced padding)")
    print(f"   - URLs automatically wrap in PDF")
    print(f"   - Optimized web search with multiple providers\n")

    return True


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate comprehensive guides with optimized multi-agent research"
    )
    parser.add_argument(
        '--topic',
        type=str,
        required=True,
        help='Topic to generate guide about. Examples: "Poodles", "Python Programming", "Chess"'
    )
    parser.add_argument(
        '--goal',
        type=str,
        default=None,
        help='Specific goal (optional)'
    )
    args = parser.parse_args()

    if args.goal is None:
        args.goal = f"Create a comprehensive beginner's guide to {args.topic}"

    # Run multi-agent research workflow
    guide_data = run_multi_agent_research(args.topic, args.goal)

    if not guide_data:
        print("❌ Multi-agent research failed\n")
        return 1

    # Create document
    doc = create_document(guide_data)

    # Generate outputs
    success = generate_outputs(doc, args.topic)

    if success:
        print(f"🎉 Optimized multi-agent research complete! Guide ready.\n")
        return 0
    else:
        print("❌ Output generation failed\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
