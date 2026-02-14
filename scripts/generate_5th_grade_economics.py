#!/usr/bin/env python3
"""Generate 5th Grade Economics Learning Material.

Uses the Olympiad Learning Swarm to create engaging, comprehensive
economics content covering:
- Income, spending, savings, budgeting, donating
- What is economics and its characteristics
- Real-life examples
- Producers and consumers, sellers and buyers
- Four types of economic systems
- Three sectors of economics
- Supply and demand

Output: Professional PDF + Interactive HTML with StatQuest-style engagement
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add Jotty to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.swarms.olympiad_learning_swarm import OlympiadLearningSwarm, OlympiadLearningConfig
from core.swarms.olympiad_learning_swarm.types import Subject, DifficultyTier, LessonDepth

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def generate_economics_material():
    """Generate comprehensive 5th grade economics learning material."""

    logger.info("=" * 80)
    logger.info("GENERATING 5TH GRADE ECONOMICS LEARNING MATERIAL")
    logger.info("=" * 80)

    # Configure for 5th grade level
    config = OlympiadLearningConfig(
        subject=Subject.GENERAL,  # Economics falls under general
        student_name="5th Grader",  # Generic for all 5th graders
        depth=LessonDepth.DEEP,  # Comprehensive coverage
        target_tier=DifficultyTier.FOUNDATION,  # Age-appropriate

        # Content settings
        build_from_basics=True,
        include_practice_problems=True,
        include_common_mistakes=True,
        include_connections=True,
        include_competition_tips=False,  # Not needed for 5th grade

        # Problem distribution (adjusted for 5th grade)
        foundation_problems=5,  # More foundation problems
        intermediate_problems=3,
        advanced_problems=0,  # Skip advanced for 5th grade
        olympiad_problems=0,  # Skip olympiad for 5th grade

        # Output
        generate_pdf=True,
        generate_html=True,
        send_telegram=False,

        # Engagement
        celebration_word="Amazing!",  # Age-appropriate celebration

        # Performance
        optimization_mode="parallel_deep",  # Fast + high quality
        max_concurrent_llm=5,
    )

    # Create the swarm
    swarm = OlympiadLearningSwarm(config)

    # Define the comprehensive economics topic
    topic = """Economics for 5th Graders: Understanding Money, Choices, and Our Economy

    This comprehensive lesson covers:
    1. What is Economics? - The study of choices about scarce resources
    2. Personal Finance - Income, spending, savings, budgeting, and donating
    3. Economic Roles - Producers vs Consumers, Sellers vs Buyers
    4. Economic Systems - Traditional, Command, Market, and Mixed economies
    5. Economic Sectors - Primary (natural resources), Secondary (manufacturing), Tertiary (services)
    6. Supply and Demand - How prices and quantities are determined
    7. Real-life examples throughout
    8. Characteristics of a healthy economy
    """

    # Generate the learning material
    logger.info(f"\nStarting content generation for: {topic[:100]}...\n")

    result = await swarm.teach(
        topic=topic,
        target_level="5th_grader",  # Explicit level for agents
    )

    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("GENERATION COMPLETE!")
    logger.info("=" * 80)

    if result.success:
        logger.info(f"✓ Success: Generated comprehensive economics lesson")
        logger.info(f"✓ Student: {result.student_name}")
        logger.info(f"✓ Topic: {result.topic[:100]}...")
        logger.info(f"✓ Learning Time: {result.learning_time_estimate}")
        logger.info(f"✓ Concepts Covered: {result.concepts_covered}")
        logger.info(f"✓ Problems Generated: {result.problems_generated}")
        logger.info(f"✓ Breakthrough Moments: {result.breakthrough_moments}")

        if result.content:
            logger.info(f"✓ Total Words: {result.content.total_words}")
            logger.info(f"✓ Building Blocks: {len(result.content.building_blocks)}")
            logger.info(f"✓ Patterns: {len(result.content.patterns)}")
            logger.info(f"✓ Strategies: {len(result.content.strategies)}")
            logger.info(f"✓ Common Mistakes: {len(result.content.mistakes)}")

        logger.info(f"\n📄 OUTPUT FILES:")
        if result.pdf_path:
            logger.info(f"   PDF: {result.pdf_path}")
        if result.html_path:
            logger.info(f"   HTML: {result.html_path}")

        logger.info(f"\n⏱️  Execution Time: {result.execution_time:.2f}s")

        # Show sample content
        if result.content and result.content.key_insights:
            logger.info(f"\n🎯 KEY INSIGHTS:")
            for i, insight in enumerate(result.content.key_insights[:3], 1):
                if insight:
                    logger.info(f"   {i}. {insight[:100]}...")

        return result
    else:
        logger.error(f"✗ Generation failed: {result.error}")
        return None


def main():
    """Main entry point."""
    try:
        result = asyncio.run(generate_economics_material())

        if result and result.success:
            print("\n" + "=" * 80)
            print("SUCCESS! Economics learning material generated.")
            print("=" * 80)
            if result.pdf_path:
                print(f"\n📖 Open the PDF: {result.pdf_path}")
            if result.html_path:
                print(f"🌐 Open the HTML: {result.html_path}")
            print()
            return 0
        else:
            print("\n❌ Generation failed. Check logs above.")
            return 1

    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
