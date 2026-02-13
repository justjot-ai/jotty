#!/usr/bin/env python3
"""
Slides Generator for Jotty
===========================
Wrapper around Paper2Slides for converting research guides to presentation slides.

Integrates with Jotty's multi-agent workflow to automatically generate:
- Research guide (PDF/MD/HTML)
- Presentation slides (PNG slides + PDF deck)
"""

import sys
import os
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, List

# Add Paper2Slides to Python path
JOTTY_ROOT = Path(__file__).parent.parent.parent.parent
PAPER2SLIDES_PATH = JOTTY_ROOT / "Paper2Slides"
sys.path.insert(0, str(PAPER2SLIDES_PATH))

from paper2slides.core import (
    get_base_dir,
    get_config_dir,
    detect_start_stage,
    run_pipeline,
)
from paper2slides.utils.path_utils import (
    normalize_input_path,
    get_project_name,
    parse_style,
)

logger = logging.getLogger(__name__)


class SlidesGenerator:
    """
    Wrapper for Paper2Slides integration with Jotty

    Features:
    - Converts research PDFs to presentation slides
    - Supports custom styling (academic, doraemon, or custom descriptions)
    - Generates both PNG slides and consolidated PDF
    - Checkpoint-based resumable workflow
    """

    def __init__(
        self,
        output_base_dir: Optional[Path] = None,
        fast_mode: bool = False
    ):
        """
        Initialize slides generator

        Args:
            output_base_dir: Base directory for outputs (default: Jotty outputs dir)
            fast_mode: Skip RAG indexing for faster generation (for short docs)
        """
        self.fast_mode = fast_mode

        if output_base_dir is None:
            output_base_dir = JOTTY_ROOT / "outputs" / "slides"

        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        # Default configuration
        self.default_config = {
            "content_type": "general",  # "paper" or "general"
            "output_type": "slides",    # "slides" or "poster"
            "style": "academic",        # "academic", "doraemon", or custom
            "slides_length": "medium",  # "short", "medium", "long"
            "poster_density": "medium", # "sparse", "medium", "dense"
            "fast_mode": fast_mode,
            "max_workers": 1,           # Parallel processing workers
        }

    async def generate_slides(
        self,
        input_pdf: Path,
        style: str = "academic",
        length: str = "medium",
        output_type: str = "slides",
        parallel_workers: int = 1,
        from_stage: Optional[str] = None
    ) -> Dict[str, Path]:
        """
        Generate presentation slides from research PDF

        Args:
            input_pdf: Path to input PDF file
            style: Presentation style (academic, doraemon, or custom description)
            length: Slides length (short=5-8, medium=10-15, long=20+)
            output_type: Output type (slides or poster)
            parallel_workers: Number of parallel workers for slide generation
            from_stage: Force re-run from specific stage (rag, summary, plan, create)

        Returns:
            Dictionary with paths to generated files:
            {
                'slides_dir': Path to directory with individual PNG slides,
                'pdf': Path to consolidated PDF presentation,
                'checkpoint_dir': Path to checkpoint directory
            }
        """
        logger.info(f"🎨 Generating {output_type} from: {input_pdf.name}")
        logger.info(f"   Style: {style}")
        logger.info(f"   Length: {length}")

        # Normalize input path
        try:
            input_path = normalize_input_path(str(input_pdf))
        except FileNotFoundError as e:
            logger.error(f"❌ Input file not found: {e}")
            raise

        # Build config
        style_type, custom_style = parse_style(style)
        config = {
            **self.default_config,
            "input_path": input_path,
            "output_type": output_type,
            "style": style_type,
            "custom_style": custom_style,
            "slides_length": length,
            "max_workers": parallel_workers,
        }

        # Determine output paths
        project_name = get_project_name(str(input_pdf))
        base_dir = get_base_dir(
            str(self.output_base_dir),
            project_name,
            config["content_type"]
        )
        config_dir = get_config_dir(base_dir, config)

        logger.info(f"📁 Output directory: {base_dir}")
        logger.info(f"   Config: {config_dir.name}")

        # Determine start stage
        if from_stage:
            start_stage = from_stage
            logger.info(f"🔄 Force re-running from stage: {start_stage}")
        else:
            start_stage = detect_start_stage(base_dir, config_dir, config)
            if start_stage != "rag":
                logger.info(f"♻️  Reusing checkpoints, starting from: {start_stage}")

        # Run Paper2Slides pipeline
        logger.info("")
        logger.info("=" * 80)
        logger.info("  PAPER2SLIDES PIPELINE")
        logger.info("=" * 80)
        logger.info("")

        try:
            await run_pipeline(base_dir, config_dir, config, start_stage)
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise

        # Find generated files
        slides_dir = config_dir
        pdf_files = list(slides_dir.glob("**/*.pdf"))

        if not pdf_files:
            raise RuntimeError(f"No PDF generated in {slides_dir}")

        # Get the latest PDF (in case of multiple runs)
        pdf_path = sorted(pdf_files, key=lambda p: p.stat().st_mtime)[-1]

        # Count PNG slides
        png_files = list(slides_dir.glob("**/*.png"))

        logger.info("")
        logger.info("=" * 80)
        logger.info("  SLIDES GENERATION COMPLETE!")
        logger.info("=" * 80)
        logger.info("")
        logger.info(f"✅ Generated {len(png_files)} slides")
        logger.info(f"✅ PDF: {pdf_path.name} ({pdf_path.stat().st_size / 1024:.1f} KB)")
        logger.info(f"📁 Output: {slides_dir}")
        logger.info("")

        return {
            'slides_dir': slides_dir,
            'pdf': pdf_path,
            'png_files': png_files,
            'checkpoint_dir': config_dir,
            'num_slides': len(png_files)
        }

    def generate_slides_sync(self, *args, **kwargs) -> Dict[str, Path]:
        """Synchronous wrapper for generate_slides"""
        return asyncio.run(self.generate_slides(*args, **kwargs))


def generate_slides_from_pdf(
    pdf_path: Path,
    style: str = "academic",
    length: str = "medium",
    output_dir: Optional[Path] = None,
    fast_mode: bool = False
) -> Dict[str, Path]:
    """
    Convenience function to generate slides from a PDF

    Args:
        pdf_path: Path to input PDF
        style: Presentation style
        length: Slides length (short/medium/long)
        output_dir: Custom output directory
        fast_mode: Skip RAG indexing for faster generation

    Returns:
        Dictionary with paths to generated files

    Example:
        >>> from pathlib import Path
        >>> result = generate_slides_from_pdf(
        ...     Path("outputs/poodles_guide/Poodles_Guide.pdf"),
        ...     style="academic",
        ...     length="medium"
        ... )
        >>> print(f"Slides PDF: {result['pdf']}")
        >>> print(f"Total slides: {result['num_slides']}")
    """
    generator = SlidesGenerator(
        output_base_dir=output_dir,
        fast_mode=fast_mode
    )

    return generator.generate_slides_sync(
        input_pdf=pdf_path,
        style=style,
        length=length
    )


if __name__ == "__main__":
    # Simple CLI for testing
    import argparse

    parser = argparse.ArgumentParser(description="Generate slides from PDF")
    parser.add_argument("--input", "-i", required=True, help="Input PDF file")
    parser.add_argument("--style", default="academic", help="Presentation style")
    parser.add_argument("--length", default="medium", choices=["short", "medium", "long"])
    parser.add_argument("--fast", action="store_true", help="Fast mode (skip RAG)")
    parser.add_argument("--output-dir", help="Output directory")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Generate slides
    result = generate_slides_from_pdf(
        pdf_path=Path(args.input),
        style=args.style,
        length=args.length,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        fast_mode=args.fast
    )

    logger.info("Slides generation succeeded!")
    logger.info(f"Slides PDF: {result['pdf']}")
    logger.info(f"Total slides: {result['num_slides']}")
    logger.info(f"Output directory: {result['slides_dir']}")
