#!/usr/bin/env python3
"""
Generate Clean, Properly Formatted PDF from Olympiad Content
Fixes the unformatted text issue by properly extracting content
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


async def main():
    from Jotty.core.swarms.olympiad_learning_swarm import learn_topic
    from Jotty.core.swarms.olympiad_learning_swarm.pdf_generator import (
        generate_lesson_pdf,
        generate_lesson_html
    )

    print("="*80)
    print("GENERATING CLEAN, FORMATTED PDF")
    print("="*80 + "\n")

    print("🚀 Generating comprehensive course content...")
    print("   Using olympiad_learning_swarm.learn_topic()\n")

    # Generate content
    result = await learn_topic(
        subject="general",
        topic="Multi-Agent Systems: Architectures, Coordination, Memory, and Learning",
        student_name="Advanced Researcher",
        depth="deep",
        target="advanced",
        send_telegram=False
    )

    if not result.success:
        print(f"❌ Content generation failed: {result.error}")
        sys.exit(1)

    print("✅ Content generated successfully!\n")

    # Extract clean LessonContent object (not the result wrapper)
    lesson_content = result.content  # This is the LessonContent dataclass

    # Generate properly formatted PDF
    pdf_path = "/home/coder/jotty/outputs/multiagent_systems_clean.pdf"
    html_path = "/home/coder/jotty/outputs/multiagent_systems_clean.html"

    print("📄 Generating styled PDF with WeasyPrint...")
    print(f"   Output: {pdf_path}\n")

    pdf_result = await generate_lesson_pdf(
        content=lesson_content,
        output_path=pdf_path,
        celebration_word="Brilliant!",
        learning_time="~90 min"
    )

    if pdf_result:
        size_kb = Path(pdf_result).stat().st_size / 1024
        print(f"✅ PDF generated: {size_kb:.1f} KB")
        print(f"   Path: {pdf_result}\n")
    else:
        print("❌ PDF generation failed\n")
        sys.exit(1)

    # Also generate HTML for preview
    print("🌐 Generating interactive HTML...")
    html_result = await generate_lesson_html(
        content=lesson_content,
        output_path=html_path,
        celebration_word="Brilliant!",
        learning_time="~90 min"
    )

    if html_result:
        print(f"✅ HTML generated: {html_result}\n")

    print("="*80)
    print("COMPLETE!")
    print(f"PDF: {pdf_path}")
    print(f"HTML: {html_path}")
    print("="*80)


if __name__ == '__main__':
    asyncio.run(main())
