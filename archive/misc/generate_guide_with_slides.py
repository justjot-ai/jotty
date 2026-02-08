#!/usr/bin/env python3
"""
Comprehensive Guide + Slides Generator
======================================

Complete workflow:
1. Planner Agent → Determines sections
2. Research Agent → Gathers information
3. Content Writer → Creates guide
4. Content Tools → Generates PDF/HTML/MD
5. Slides Generator → Creates presentation deck

Output:
- Research guide (PDF, Markdown, HTML)
- Presentation slides (PDF deck + individual PNG slides)
"""

import sys
from pathlib import Path
import dspy
import logging
from typing import List

# Add Jotty to path
sys.path.insert(0, str(Path(__file__).parent))

from core.tools.content_generation import Document, SectionType, ContentGenerators
from core.tools.content_generation.slides_generator import SlidesGenerator
from core.integration.direct_claude_cli_lm import DirectClaudeCLI

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None
    print("⚠️  duckduckgo_search not available - research will use LLM knowledge only")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# MULTI-AGENT SIGNATURES (Reused from generate_guide_with_research.py)
# =============================================================================

class PlannerSignature(dspy.Signature):
    """Planner Agent - Determines what sections to generate for a topic"""
    topic: str = dspy.InputField(desc="Topic to create guide about")
    goal: str = dspy.InputField(desc="Specific goal for the guide")

    section_titles: str = dspy.OutputField(
        desc="List of 10-15 section titles that should be covered. "
        "Format: one title per line. "
        "Each title should be specific to the topic."
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

    research_summary: str = dspy.OutputField(
        desc="Summary of research findings. "
        "Include key facts, statistics, and important information discovered."
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
        "Include specific details from research. "
        "Write 2-4 comprehensive paragraphs."
    )


# =============================================================================
# WEB SEARCH TOOL
# =============================================================================

def search_web(query: str, max_results: int = 5) -> List[dict]:
    """Search the web using DuckDuckGo"""
    if DDGS is None:
        return []

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return results
    except Exception as e:
        logger.warning(f"Search failed for '{query}': {e}")
        return []


# =============================================================================
# MULTI-AGENT WORKFLOW
# =============================================================================

def run_multi_agent_research(topic: str, goal: str):
    """
    Run multi-agent workflow to create comprehensive guide

    Returns guide data for slides generation
    """

    print("\n" + "=" * 100)
    print(f"  JOTTY MULTI-AGENT: '{topic} for Dummies' + PRESENTATION")
    print("=" * 100 + "\n")

    # Configure LLM
    print("🔧 Configuring Claude CLI...")
    try:
        lm = DirectClaudeCLI(model="haiku")
        dspy.configure(lm=lm)
        print("✅ Configured with Claude CLI (Haiku)\n")
    except Exception as e:
        print(f"❌ Failed to configure LLM: {e}")
        return None

    # Phase 1: Planning
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

    print(f"✅ Planner determined {len(section_titles)} sections")
    print(f"✅ Planner generated {len(research_queries)} research queries\n")

    # Phase 2: Research
    print("=" * 100)
    print("  PHASE 2: WEB RESEARCH")
    print("=" * 100 + "\n")

    all_research = []
    if DDGS:
        print("🔍 Searching with DuckDuckGo...\n")
        for i, query in enumerate(research_queries[:5], 1):
            print(f"   Query {i}: {query}")
            results = search_web(query, max_results=3)
            if results:
                print(f"   ✅ Found {len(results)} results")
                for result in results:
                    all_research.append({
                        'query': query,
                        'title': result.get('title', ''),
                        'snippet': result.get('body', ''),
                        'url': result.get('href', '')
                    })
            else:
                print(f"   ⚠️  No results")
    else:
        print("⚠️  Skipping web search (DuckDuckGo not available)\n")

    print(f"\n✅ Gathered {len(all_research)} research results\n")

    # Synthesize research
    print("🤖 Agent 2: Researcher")
    print("   Task: Synthesize research findings\n")

    researcher = dspy.ChainOfThought(ResearcherSignature)
    research_result = researcher(
        topic=topic,
        queries="\n".join(research_queries[:5])
    )

    print("✅ Research synthesized\n")

    # Phase 3: Content Generation
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
        author="Jotty AI Multi-Agent Research Team",
        topic=guide_data['topic'],
        source_type="jotty-multi-agent-research"
    )

    # Add all sections
    for title, content in guide_data['sections']:
        doc.add_section(SectionType.TEXT, content, title=title)

    # Add research sources as appendix
    if guide_data['research_sources']:
        sources_text = "This guide was created using research from the following sources:\n\n"
        for i, source in enumerate(guide_data['research_sources'][:10], 1):
            url = source['url']
            if len(url) > 80:
                url = url[:77] + "..."
            sources_text += f"{i}. {source['title']}\n   {url}\n\n"

        doc.add_section(SectionType.TEXT, sources_text, title="Research Sources")

    print(f"✅ Document created with {len(doc.sections)} sections\n")

    return doc


def generate_outputs(doc, topic: str):
    """Generate PDF, Markdown, HTML"""

    print("=" * 100)
    print("  PHASE 4: GENERATING GUIDE FILES")
    print("=" * 100 + "\n")

    topic_slug = topic.lower().replace(" ", "_").replace("/", "_")
    output_dir = Path(f"./outputs/{topic_slug}_guide")
    output_dir.mkdir(parents=True, exist_ok=True)

    generators = ContentGenerators()

    # Generate PDF
    print("📕 Generating PDF (0.75in margins, blue links)...")
    try:
        pdf_path = generators.generate_pdf(doc, output_path=output_dir, format='a4')
        print(f"✅ PDF: {pdf_path.name} ({pdf_path.stat().st_size/1024:.1f} KB)\n")
    except Exception as e:
        print(f"❌ PDF failed: {e}\n")
        return None, output_dir

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

    print(f"✅ Guide files generated in: {output_dir}\n")

    return pdf_path, output_dir


def generate_slides(pdf_path: Path, topic: str, style: str = "academic", length: str = "medium"):
    """Generate presentation slides from guide PDF"""

    print("=" * 100)
    print("  PHASE 5: GENERATING PRESENTATION SLIDES")
    print("=" * 100 + "\n")

    try:
        generator = SlidesGenerator(fast_mode=True)  # Fast mode for quicker generation

        result = generator.generate_slides_sync(
            input_pdf=pdf_path,
            style=style,
            length=length,
            parallel_workers=1
        )

        print(f"✅ Slides generated successfully!")
        print(f"   Total slides: {result['num_slides']}")
        print(f"   PDF deck: {result['pdf']}")
        print(f"   Output: {result['slides_dir']}\n")

        return result

    except Exception as e:
        print(f"❌ Slides generation failed: {e}\n")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate comprehensive guide + presentation slides"
    )
    parser.add_argument(
        '--topic',
        type=str,
        required=True,
        help='Topic to generate guide about'
    )
    parser.add_argument(
        '--goal',
        type=str,
        default=None,
        help='Specific goal (optional)'
    )
    parser.add_argument(
        '--style',
        type=str,
        default='academic',
        help='Presentation style (academic, doraemon, or custom description)'
    )
    parser.add_argument(
        '--length',
        type=str,
        default='medium',
        choices=['short', 'medium', 'long'],
        help='Slides length'
    )
    parser.add_argument(
        '--skip-slides',
        action='store_true',
        help='Skip slides generation (guide only)'
    )

    args = parser.parse_args()

    if args.goal is None:
        args.goal = f"Create a comprehensive beginner's guide to {args.topic}"

    # Phase 1-3: Multi-agent research
    guide_data = run_multi_agent_research(args.topic, args.goal)

    if not guide_data:
        print("❌ Multi-agent research failed\n")
        return 1

    # Create document
    doc = create_document(guide_data)

    # Phase 4: Generate guide files
    pdf_path, output_dir = generate_outputs(doc, args.topic)

    if not pdf_path:
        print("❌ Guide generation failed\n")
        return 1

    # Phase 5: Generate slides
    slides_result = None
    if not args.skip_slides:
        slides_result = generate_slides(pdf_path, args.topic, args.style, args.length)

    # Final summary
    print("=" * 100)
    print("  🎉 SUCCESS!")
    print("=" * 100 + "\n")

    print(f"📁 Output Directory: {output_dir}\n")

    print("📚 Generated Files:\n")
    for file in sorted(output_dir.glob("*")):
        size = file.stat().st_size / 1024
        print(f"   - {file.name} ({size:.1f} KB)")

    print(f"\n🎯 Main Guide PDF: {pdf_path}")

    if slides_result:
        print(f"🎨 Presentation Slides: {slides_result['pdf']}")
        print(f"   Total slides: {slides_result['num_slides']}")

    print(f"\n✨ Your '{args.topic} for Dummies' guide + presentation is ready!\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
