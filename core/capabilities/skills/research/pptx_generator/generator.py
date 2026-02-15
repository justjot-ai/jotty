"""
PowerPoint Generator for ArXiv Learning
Uses PptxGenJS via Node.js for professional Goldman-Sachs style presentations.

Features:
- Intelligent diagram selection (no force-fit)
- LLM-as-Judge quality evaluation
- Auto-improvement loop until 10/10
"""

import asyncio
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from .deck_judge import (
    DiagramType,
    DiagramDecision,
    DiagramDecisionEngine,
    DeckScore,
    DeckJudge,
    AutoImprovementLoop,
    analyze_and_decide_diagrams,
)

from .visualization_planner import (
    LIDAStylePlanner,
    VisualizationSpec,
    convert_specs_to_pptx_data,
)

from .diagram_image_generator import (
    DiagramImageGenerator,
    MermaidDiagramGenerator,
)

# Import HTML slide generator
from ..html_slide_generator import (
    HTMLSlideGenerator,
    LearningSlideBuilder,
    PresentationConfig,
    SlideType,
)

logger = logging.getLogger(__name__)

# Path to the Node.js script
SCRIPT_DIR = Path(__file__).parent
GENERATE_SCRIPT = SCRIPT_DIR / "generate_pptx.js"

# Check for LibreOffice availability (for PPTX to PDF conversion)
_LIBREOFFICE_AVAILABLE = None


def _find_libreoffice() -> Optional[str]:
    """Find LibreOffice executable path."""
    global _LIBREOFFICE_AVAILABLE
    if _LIBREOFFICE_AVAILABLE is not None:
        return _LIBREOFFICE_AVAILABLE

    # Try common LibreOffice executable names
    for cmd in ['libreoffice', 'soffice', '/usr/bin/libreoffice', '/usr/bin/soffice',
                '/Applications/LibreOffice.app/Contents/MacOS/soffice']:
        if shutil.which(cmd):
            _LIBREOFFICE_AVAILABLE = cmd
            return cmd

    _LIBREOFFICE_AVAILABLE = ""
    return None


async def convert_pptx_to_pdf(pptx_path: str, output_dir: Optional[str] = None) -> Optional[str]:
    """
    Convert PPTX to PDF using LibreOffice.

    Args:
        pptx_path: Path to the PPTX file
        output_dir: Directory for output PDF (defaults to same directory as PPTX)

    Returns:
        Path to generated PDF file, or None if conversion failed
    """
    libreoffice = _find_libreoffice()
    if not libreoffice:
        logger.warning("⚠️ LibreOffice not found - PPTX to PDF conversion unavailable. "
                      "Install LibreOffice for this feature: apt install libreoffice")
        return None

    pptx_file = Path(pptx_path)
    if not pptx_file.exists():
        logger.error(f"PPTX file not found: {pptx_path}")
        return None

    out_dir = Path(output_dir) if output_dir else pptx_file.parent

    try:
        # Run LibreOffice headless conversion
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: subprocess.run(
                [
                    libreoffice,
                    '--headless',
                    '--convert-to', 'pdf',
                    '--outdir', str(out_dir),
                    str(pptx_file)
                ],
                capture_output=True,
                text=True,
                timeout=120  # 2 minutes timeout for conversion
            )
        )

        if result.returncode == 0:
            # LibreOffice names output file same as input but with .pdf extension
            pdf_path = out_dir / (pptx_file.stem + '.pdf')
            if pdf_path.exists():
                logger.info(f"✅ Converted PPTX to PDF: {pdf_path}")
                return str(pdf_path)
            else:
                logger.error(f"PDF not created at expected path: {pdf_path}")
                return None
        else:
            logger.error(f"LibreOffice conversion failed: {result.stderr or result.stdout}")
            return None

    except subprocess.TimeoutExpired:
        logger.error("PPTX to PDF conversion timed out")
        return None
    except Exception as e:
        logger.error(f"PPTX to PDF conversion error: {e}")
        return None


def is_libreoffice_available() -> bool:
    """Check if LibreOffice is available for PDF conversion."""
    return _find_libreoffice() is not None


async def generate_learning_pptx(
    paper_title: str,
    arxiv_id: str,
    authors: List[str],
    hook: str,
    concepts: List[Dict[str, Any]],
    sections: List[Dict[str, Any]],
    key_insights: List[str],
    summary: str,
    next_steps: List[str],
    output_path: str,
    bingo_word: str = "Bingo",
    learning_time: str = "20-30 min",
    total_words: int = 0,
    visualization_specs: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Generate a professional PowerPoint presentation from learning content.

    Uses LIDA-style visualization specs when provided for detailed diagrams.

    Args:
        paper_title: Title of the paper
        arxiv_id: ArXiv paper ID
        authors: List of author names
        hook: Opening hook/why it matters
        concepts: List of concept dicts with name, description, difficulty, why_it_matters
        sections: List of section dicts with title, content, level, has_bingo_moment, code_example
        key_insights: List of key insight strings
        summary: Summary text
        next_steps: List of next step strings
        output_path: Where to save the .pptx file
        bingo_word: Celebration word (default "Bingo")
        learning_time: Estimated learning time
        total_words: Total word count
        visualization_specs: Optional LIDA-style specs for detailed diagrams

    Returns:
        Path to generated PPTX file, or None if generation failed
    """
    try:
        # Prepare data for the Node.js script
        data = {
            "paper_title": paper_title,
            "arxiv_id": arxiv_id,
            "authors": authors,
            "hook": hook,
            "concepts": concepts,
            "sections": sections,
            "key_insights": key_insights,
            "summary": summary,
            "next_steps": next_steps,
            "bingo_word": bingo_word,
            "learning_time": learning_time,
            "total_words": total_words,
            "visualization_specs": visualization_specs or {}
        }

        # Write data to temp JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f, ensure_ascii=False)
            json_path = f.name

        try:
            # Run Node.js script
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    ['node', str(GENERATE_SCRIPT), json_path, output_path],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=str(SCRIPT_DIR)
                )
            )

            if result.returncode == 0 and 'SUCCESS:' in result.stdout:
                # Find the SUCCESS: line (may have console.log output before it)
                pptx_path = result.stdout.split('SUCCESS:')[1].strip().split('\n')[0]
                logger.info(f"✅ Generated PPTX: {pptx_path}")
                return pptx_path
            else:
                error_msg = result.stderr or result.stdout
                logger.error(f"PPTX generation failed: {error_msg}")
                return None

        finally:
            # Cleanup temp file
            Path(json_path).unlink(missing_ok=True)

    except subprocess.TimeoutExpired:
        logger.error("PPTX generation timed out")
        return None
    except Exception as e:
        logger.error(f"PPTX generation error: {e}")
        return None


async def generate_intelligent_pptx(
    paper_data: Dict[str, Any],
    output_path: str,
    target_score: float = 9.5,
    max_iterations: int = 3,
    use_lida_planning: bool = True
) -> Tuple[Optional[str], DeckScore]:
    """
    Generate a presentation with intelligent diagram selection and quality evaluation.

    This function combines LIDA-style LLM planning with PptxGenJS rendering:
    1. LIDA planner generates detailed visualization specs (LLM recommends)
    2. Decides which diagrams to include (no force-fit)
    3. PptxGenJS renders specs into pixel-perfect slides (implements)
    4. Evaluates quality using LLM-as-Judge

    Args:
        paper_data: Dict with paper_title, arxiv_id, authors, hook, concepts, sections, etc.
        output_path: Where to save the .pptx file
        target_score: Target quality score (1-10), default 9.5
        max_iterations: Max improvement iterations
        use_lida_planning: Whether to use LIDA-style LLM planning for detailed specs

    Returns:
        Tuple of (pptx_path, final_score)
    """
    visualization_specs = {}
    diagram_images = {}

    # Step 1: LIDA-style LLM planning (optional but recommended)
    if use_lida_planning:
        logger.info("🎨 LIDA-style visualization planning...")
        try:
            planner = LIDAStylePlanner()
            specs = planner.generate_all_specs(paper_data)
            visualization_specs = convert_specs_to_pptx_data(specs)
            logger.info(f"✅ Generated {len(specs)} detailed visualization specs")
        except Exception as e:
            logger.warning(f"LIDA planning failed, using defaults: {e}")

    # Step 1.5: Generate Mermaid diagram images (professional, pre-sized)
    logger.info("🎨 Generating Mermaid diagram images...")
    try:
        diagram_generator = DiagramImageGenerator()
        viz_specs = visualization_specs.get('visualization_specs', {})
        if not viz_specs and paper_data.get('visualization_specs'):
            viz_specs = paper_data.get('visualization_specs', {})

        diagram_images = await diagram_generator.generate_all_diagrams(
            paper_data,
            viz_specs
        )
        logger.info(f"✅ Generated {len(diagram_images)} diagram images: {list(diagram_images.keys())}")
    except Exception as e:
        logger.warning(f"Mermaid diagram generation failed: {e}")
        diagram_images = {}

    # Step 2: Analyze which diagrams should be included
    logger.info("📊 Analyzing paper for intelligent diagram selection...")
    diagram_decisions, approved_diagrams = analyze_and_decide_diagrams(paper_data)

    logger.info(f"✅ Approved diagrams: {approved_diagrams}")
    for dt, decision in diagram_decisions.items():
        if not decision.should_include:
            logger.info(f"❌ Skipping {dt.value}: {decision.reasoning}")

    # Step 3: Combine decisions with visualization specs and diagram images
    paper_data_with_specs = {
        **paper_data,
        'diagram_decisions': {
            dt.value: {
                'should_include': decision.should_include,
                'confidence': decision.confidence,
                'reasoning': decision.reasoning
            }
            for dt, decision in diagram_decisions.items()
        },
        'visualization_specs': visualization_specs.get('visualization_specs', {}),
        'diagram_images': diagram_images  # Mermaid-generated image paths
    }

    # Step 4: Generate the presentation (PptxGenJS uses images when available)
    viz_specs_data = visualization_specs.get('visualization_specs', {})
    viz_specs_data['diagram_images'] = diagram_images  # Include image paths

    pptx_path = await generate_learning_pptx(
        paper_title=paper_data.get('paper_title', 'Untitled'),
        arxiv_id=paper_data.get('arxiv_id', ''),
        authors=paper_data.get('authors', []),
        hook=paper_data.get('hook', ''),
        concepts=paper_data.get('concepts', []),
        sections=paper_data.get('sections', []),
        key_insights=paper_data.get('key_insights', []),
        summary=paper_data.get('summary', ''),
        next_steps=paper_data.get('next_steps', []),
        output_path=output_path,
        bingo_word=paper_data.get('bingo_word', 'Eureka'),
        learning_time=paper_data.get('learning_time', '30 min'),
        total_words=paper_data.get('total_words', 0),
        visualization_specs=viz_specs_data
    )

    if not pptx_path:
        return None, DeckScore()

    # Step 4: Evaluate the deck
    logger.info("🎯 Evaluating presentation quality...")
    judge = DeckJudge()

    # Calculate approximate slide count based on content
    # Title + Agenda + Hook + Architecture + Concepts overview + individual concepts + sections + flow + comparison + insights + metrics + next steps + thank you
    num_concepts = len(paper_data.get('concepts', []))
    num_sections = len(paper_data.get('sections', []))
    estimated_slides = 5 + num_concepts + num_sections + len(approved_diagrams) + 3  # base + concepts + sections + diagrams + closing

    deck_info = {
        'diagrams_included': approved_diagrams,
        'has_code_examples': any(s.get('code_example') for s in paper_data.get('sections', [])),
        'has_eureka_moments': any(s.get('has_bingo_moment') for s in paper_data.get('sections', [])),
        'slide_count': estimated_slides,
    }

    score = judge.evaluate(paper_data, deck_info, diagram_decisions)

    logger.info(f"📈 Quality Score: {score.overall:.1f}/10")
    logger.info(f"   - Content: {score.content_depth:.1f}")
    logger.info(f"   - Diagrams: {score.diagram_relevance:.1f}")
    logger.info(f"   - Clarity: {score.clarity:.1f}")

    if score.diagrams_to_remove:
        logger.info(f"   ⚠️ Diagrams to remove: {score.diagrams_to_remove}")

    if score.improvements:
        logger.info(f"   💡 Improvements: {score.improvements[:2]}")

    return pptx_path, score


async def generate_and_improve_pptx(
    paper_data: Dict[str, Any],
    output_path: str,
    target_score: float = 9.5,
    max_iterations: int = 5
) -> Tuple[Optional[str], DeckScore, str]:
    """
    Generate and iteratively improve a presentation until target score.

    Uses the auto-improvement loop:
    1. Generate → 2. Evaluate → 3. Identify improvements → 4. Regenerate → Repeat

    Args:
        paper_data: Dict with paper content
        output_path: Where to save the .pptx file
        target_score: Target quality score (1-10)
        max_iterations: Max improvement iterations

    Returns:
        Tuple of (pptx_path, final_score, progress_report)
    """
    loop = AutoImprovementLoop(target_score=target_score, max_iterations=max_iterations)
    iteration = 0
    current_data = paper_data.copy()
    best_path = None
    best_score = DeckScore()

    while True:
        iteration += 1
        logger.info(f"\n🔄 Iteration {iteration}/{max_iterations}")

        # Generate and evaluate
        iter_output = output_path.replace('.pptx', f'_v{iteration}.pptx')
        pptx_path, score = await generate_intelligent_pptx(
            current_data,
            iter_output if iteration > 1 else output_path,
            target_score=target_score,
            max_iterations=1  # Single generation per iteration
        )

        loop.record_iteration(iteration, score)

        # Track best result
        if score.overall > best_score.overall:
            best_score = score
            best_path = pptx_path

        # Check if we should continue
        if not loop.should_continue(score, iteration):
            break

        # Get improvement plan and apply
        plan = loop.get_improvement_plan(score)
        logger.info(f"📋 Improvement plan: {plan}")

        # Apply improvements to data for next iteration
        # (In a full implementation, this would modify content/structure)
        # For now, just update diagram decisions based on feedback
        if plan['diagrams_to_remove']:
            for diagram in plan['diagrams_to_remove']:
                if 'diagram_decisions' not in current_data:
                    current_data['diagram_decisions'] = {}
                current_data['diagram_decisions'][diagram] = {
                    'should_include': False,
                    'confidence': 1.0,
                    'reasoning': 'Removed by improvement loop'
                }

    progress_report = loop.get_progress_report()
    logger.info(f"\n{progress_report}")

    return best_path, best_score, progress_report


async def generate_learning_html_slides(
    paper_title: str,
    arxiv_id: str,
    authors: List[str],
    hook: str,
    concepts: List[Dict[str, Any]],
    sections: List[Dict[str, Any]],
    key_insights: List[str],
    summary: str,
    next_steps: List[str],
    output_path: str,
    bingo_word: str = "Eureka",
    learning_time: str = "20-30 min",
    total_words: int = 0,
    visualization_specs: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Generate interactive HTML slides from learning content.

    Uses the SAME signature as generate_learning_pptx and convert_learning_to_pdf
    for easy integration with the existing ArxivLearningSwarm.

    Args:
        paper_title: Title of the paper
        arxiv_id: ArXiv paper ID
        authors: List of author names
        hook: Opening hook/why it matters
        concepts: List of concept dicts with name, description, difficulty, why_it_matters
        sections: List of section dicts with title, content, level, has_bingo_moment
        key_insights: List of key insight strings
        summary: Summary text
        next_steps: List of next step strings
        output_path: Where to save the .html file
        bingo_word: Celebration word
        learning_time: Estimated learning time
        total_words: Total word count
        visualization_specs: Optional visualization specs

    Returns:
        Path to generated HTML file, or None if generation failed
    """
    try:
        logger.info("🌐 Generating interactive HTML slides...")

        # Build paper_data dict from individual args (same format as PPTX)
        paper_data = {
            "paper_title": paper_title,
            "arxiv_id": arxiv_id,
            "authors": authors,
            "hook": hook,
            "concepts": concepts,
            "sections": sections,
            "key_insights": key_insights,
            "summary": summary,
            "next_steps": next_steps,
            "bingo_word": bingo_word,
            "learning_time": learning_time,
            "total_words": total_words,
            "visualization_specs": visualization_specs or {},
        }

        # Transform to HTML slide format
        html_paper_data = _transform_for_html_slides(paper_data)

        # Build the presentation
        builder = LearningSlideBuilder()
        html_content = builder.build_from_paper_data(html_paper_data)

        # Save to file
        builder.save(html_content, output_path)

        logger.info(f"✅ Generated HTML slides: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"HTML slide generation error: {e}")
        return None


async def generate_learning_html(
    paper_data: Dict[str, Any],
    output_path: str
) -> Optional[str]:
    """
    Generate interactive HTML slides from paper data dict.

    Convenience wrapper that takes a single dict instead of individual args.

    Args:
        paper_data: Dict with paper content matching the PPTX generator format
        output_path: Where to save the .html file

    Returns:
        Path to generated HTML file, or None if generation failed
    """
    return await generate_learning_html_slides(
        paper_title=paper_data.get('paper_title', 'Untitled'),
        arxiv_id=paper_data.get('arxiv_id', ''),
        authors=paper_data.get('authors', []),
        hook=paper_data.get('hook', ''),
        concepts=paper_data.get('concepts', []),
        sections=paper_data.get('sections', []),
        key_insights=paper_data.get('key_insights', []),
        summary=paper_data.get('summary', ''),
        next_steps=paper_data.get('next_steps', []),
        output_path=output_path,
        bingo_word=paper_data.get('bingo_word', 'Eureka'),
        learning_time=paper_data.get('learning_time', '20-30 min'),
        total_words=paper_data.get('total_words', 0),
        visualization_specs=paper_data.get('visualization_specs'),
    )


def _transform_for_html_slides(paper_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform PPTX paper_data format to HTML slide builder format.
    """
    # Extract concepts with proper structure
    concepts = []
    for c in paper_data.get('concepts', []):
        concepts.append({
            "name": c.get('name', ''),
            "description": c.get('description', ''),
            "icon": _get_concept_icon(c.get('difficulty', 'intermediate')),
            "formula": c.get('why_it_matters', '')[:50] if c.get('why_it_matters') else '',
        })

    # Build methodology steps from sections
    methodology_steps = []
    for i, section in enumerate(paper_data.get('sections', [])[:4]):
        methodology_steps.append({
            "title": section.get('title', f'Step {i+1}'),
            "description": section.get('content', '')[:100] + '...' if len(section.get('content', '')) > 100 else section.get('content', ''),
        })

    # Build comparison from concepts (old vs new paradigm)
    comparison = {
        "title": "Innovation Comparison",
        "before": {
            "title": "Previous Approach",
            "items": ["Traditional methods", "Sequential processing", "Limited scalability", "Manual optimization"]
        },
        "after": {
            "title": paper_data.get('paper_title', 'New Approach')[:30],
            "items": paper_data.get('key_insights', ['Novel architecture', 'Parallel processing', 'Better scalability', 'Automated optimization'])[:4]
        }
    }

    # Build timeline (if available, otherwise use generic)
    timeline = paper_data.get('timeline', [
        {"year": "Background", "title": "Prior Work", "description": "Foundation of this research area", "highlight": False},
        {"year": "This Paper", "title": paper_data.get('paper_title', 'Current Work')[:30], "description": paper_data.get('hook', '')[:80], "highlight": True},
        {"year": "Impact", "title": "Applications", "description": "Widespread adoption in industry and research", "highlight": False},
    ])

    # Build takeaways from key_insights and next_steps
    takeaways = []
    for insight in paper_data.get('key_insights', [])[:3]:
        takeaways.append({"title": insight[:50], "description": insight[50:] if len(insight) > 50 else ""})
    for step in paper_data.get('next_steps', [])[:1]:
        takeaways.append({"title": "Next Step", "description": step})

    return {
        "title": paper_data.get('paper_title', 'Untitled Paper'),
        "arxiv_id": paper_data.get('arxiv_id', ''),
        "authors": paper_data.get('authors', []),
        "abstract": paper_data.get('hook', ''),
        "tags": _extract_tags(paper_data),
        "year": paper_data.get('year', '2024'),
        "citations": paper_data.get('citations', 'N/A'),
        "concepts": concepts,
        "methodology_steps": methodology_steps,
        "comparison": comparison,
        "timeline": timeline,
        "key_quote": paper_data.get('hook', '')[:200] if paper_data.get('hook') else '',
        "takeaways": takeaways,
        "affiliations": {},
    }


def _get_concept_icon(difficulty: str) -> str:
    """Get emoji icon based on concept difficulty."""
    icons = {
        "beginner": "🌱",
        "basics": "🌱",
        "intermediate": "🔧",
        "intuition": "💡",
        "advanced": "🎯",
        "math": "📐",
        "deep": "🔬",
        "application": "🚀",
    }
    return icons.get(difficulty.lower() if difficulty else 'intermediate', '💡')


def _extract_tags(paper_data: Dict[str, Any]) -> List[str]:
    """Extract relevant tags from paper data."""
    tags = []

    # Extract from concepts
    for c in paper_data.get('concepts', [])[:2]:
        if c.get('name'):
            tags.append(c['name'][:15])

    # Add difficulty-based tags
    difficulties = set(c.get('difficulty', '') for c in paper_data.get('concepts', []))
    if 'advanced' in difficulties or 'math' in difficulties:
        tags.append("Advanced")
    elif 'intermediate' in difficulties:
        tags.append("Intermediate")

    # Default tags
    if not tags:
        tags = ["Research", "AI"]

    return tags[:4]


async def generate_learning_html(
    paper_data: Dict[str, Any],
    output_path: str
) -> Optional[str]:
    """
    Generate HTML slides from paper data.

    This uses the same paper_data format as the PPTX and PDF generators
    for seamless integration with the research swarm.

    Passes through ALL content from the swarm to create comprehensive
    presentations with 20-40 slides.

    Args:
        paper_data: Dict containing paper analysis (title, concepts, sections, etc.)
        output_path: Where to save the HTML file

    Returns:
        Path to generated HTML file, or None if generation failed
    """
    try:
        # Pass through ALL data from swarm - don't truncate or limit!
        builder_data = {
            # Core paper info
            "title": paper_data.get("paper_title", paper_data.get("title", "Research Paper")),
            "arxiv_id": paper_data.get("arxiv_id", ""),
            "authors": paper_data.get("authors", []),

            # Hook and abstract - use the full content
            "hook": paper_data.get("hook", ""),
            "abstract": paper_data.get("abstract", paper_data.get("summary", "")),
            "summary": paper_data.get("summary", ""),

            # Metadata
            "tags": paper_data.get("tags", ["Research", "AI"]),
            "year": paper_data.get("year", "2024"),
            "citations": paper_data.get("citations", "N/A"),
            "learning_time": paper_data.get("learning_time", "20-30 min"),
            "bingo_word": paper_data.get("bingo_word", "Eureka!"),

            # FULL concepts with ALL fields - don't truncate!
            "concepts": [
                {
                    "name": c.get("name", ""),
                    "description": c.get("description", ""),
                    "why_it_matters": c.get("why_it_matters", ""),
                    "prerequisites": c.get("prerequisites", []),
                    "difficulty": c.get("difficulty", 3),
                    "math_required": c.get("math_required", False),
                    "icon": c.get("icon", "💡"),
                    "formula": c.get("formula", ""),
                    "code_example": c.get("code_example", ""),
                }
                for c in paper_data.get("concepts", [])
            ],

            # FULL sections with ALL fields - don't truncate!
            "sections": [
                {
                    "title": s.get("title", ""),
                    "content": s.get("content", ""),  # Full content!
                    "level": s.get("level", 1),
                    "has_bingo_moment": s.get("has_bingo_moment", False),
                    "code_example": s.get("code_example", ""),
                    "visualization_desc": s.get("visualization_desc", ""),
                    "exercises": s.get("exercises", []),
                }
                for s in paper_data.get("sections", [])
            ],

            # Key insights - full list
            "key_insights": paper_data.get("key_insights", []),

            # Takeaways
            "takeaways": [
                {"title": t.get("title", t) if isinstance(t, dict) else str(t), "description": t.get("description", "") if isinstance(t, dict) else ""}
                for t in paper_data.get("takeaways", paper_data.get("key_insights", []))
            ],

            # Next steps
            "next_steps": paper_data.get("next_steps", []),

            # Visual elements
            "timeline": paper_data.get("timeline", []),
            "comparison": paper_data.get("comparison", {}),
            "architecture": paper_data.get("architecture", {}),
            "results": paper_data.get("results", {}),
            "pros": paper_data.get("pros", []),
            "cons": paper_data.get("cons", []),

            # Quote
            "key_quote": paper_data.get("key_quote", ""),

            # Methodology steps (can be derived from sections if not provided)
            "methodology_steps": paper_data.get("methodology_steps", []),

            # Affiliations
            "affiliations": paper_data.get("affiliations", {}),

            # Visualization specs
            "visualization_specs": paper_data.get("visualization_specs", {}),
        }

        builder = LearningSlideBuilder()
        html_content = builder.build_from_paper_data(builder_data)
        builder.save(html_content, output_path)

        # Log the slide count for visibility
        slide_count = len(builder.generator.slides) if builder.generator else 0
        logger.info(f"✅ Generated HTML slides: {output_path} ({slide_count} slides)")
        return output_path

    except Exception as e:
        logger.error(f"HTML slide generation error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


async def generate_learning_html_slides(
    paper_title: str,
    arxiv_id: str,
    authors: List[str],
    hook: str,
    concepts: List[Dict[str, Any]],
    sections: List[Dict[str, Any]],
    key_insights: List[str],
    summary: str,
    next_steps: List[str],
    output_path: str,
    bingo_word: str = "Eureka",
    learning_time: str = "20-30 min",
    total_words: int = 0,
    visualization_specs: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Generate HTML slides with the SAME signature as generate_learning_pptx.

    This allows drop-in replacement in existing swarm workflows.

    Args:
        paper_title: Title of the paper
        arxiv_id: ArXiv paper ID
        authors: List of author names
        hook: Opening hook/why it matters
        concepts: List of concept dicts
        sections: List of section dicts
        key_insights: List of key insight strings
        summary: Summary text
        next_steps: List of next step strings
        output_path: Where to save the HTML file
        bingo_word: Celebration word (default "Eureka")
        learning_time: Estimated learning time
        total_words: Total word count
        visualization_specs: Optional visualization specs

    Returns:
        Path to generated HTML file, or None if generation failed
    """
    paper_data = {
        "paper_title": paper_title,
        "arxiv_id": arxiv_id,
        "authors": authors,
        "hook": hook,
        "concepts": concepts,
        "sections": sections,
        "key_insights": key_insights,
        "summary": summary,
        "next_steps": next_steps,
        "bingo_word": bingo_word,
        "learning_time": learning_time,
        "total_words": total_words,
        "visualization_specs": visualization_specs,
    }

    return await generate_learning_html(paper_data, output_path)


async def generate_all_formats(
    paper_data: Dict[str, Any],
    output_dir: str,
    generate_pptx: bool = True,
    generate_html: bool = True,
    generate_pdf: bool = True,
    target_score: float = 9.5,
    max_iterations: int = 3
) -> Dict[str, Optional[str]]:
    """
    Generate learning content in all requested formats.

    This is the main entry point for multi-format output generation.
    Produces PPTX, HTML slides, and PDF as requested.

    Args:
        paper_data: Dict with paper content
        output_dir: Directory to save output files
        generate_pptx: Whether to generate PowerPoint
        generate_html: Whether to generate HTML slides
        generate_pdf: Whether to convert PPTX to PDF
        target_score: Target quality score for PPTX
        max_iterations: Max improvement iterations for PPTX

    Returns:
        Dict mapping format to output path (or None if not generated)
    """
    from pathlib import Path
    import os

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate safe filename from paper title
    safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_'
                         for c in paper_data.get('paper_title', 'presentation')[:50])
    safe_title = safe_title.strip().replace(' ', '_')

    results = {
        "pptx": None,
        "html": None,
        "pdf": None,
    }

    # Generate PPTX
    if generate_pptx:
        pptx_path = str(output_dir / f"{safe_title}.pptx")
        pptx_result, score, _ = await generate_and_improve_pptx(
            paper_data,
            pptx_path,
            target_score=target_score,
            max_iterations=max_iterations
        )
        results["pptx"] = pptx_result
        logger.info(f"📊 PPTX generated with score {score.overall:.1f}/10")

        # Convert to PDF if requested
        if generate_pdf and pptx_result:
            pdf_path = await convert_pptx_to_pdf(pptx_result)
            results["pdf"] = pdf_path

    # Generate HTML slides
    if generate_html:
        html_path = str(output_dir / f"{safe_title}_slides.html")
        html_result = await generate_learning_html(paper_data, html_path)
        results["html"] = html_result

    logger.info(f"📦 Generated formats: {[k for k, v in results.items() if v]}")
    return results


