"""
Generate Transformer presentation slides, convert to PDF, send to Telegram
"""

import asyncio
import os
import sys
from datetime import datetime

sys.path.insert(0, "/var/www/sites/personal/stock_market/Jotty")

PROMPT = """Create a SLIDE DECK (not a book!) about Transformer Architecture.

SLIDE DECK RULES - VERY IMPORTANT:
- Maximum 4-5 bullet points per slide
- Each bullet: 5-10 words MAX
- No paragraphs - only short punchy bullets
- Use --- to separate slides
- Think PRESENTATION not document

FORMAT:
---
## Slide Title

- Short bullet point
- Another short point
- Key takeaway

---

NOW CREATE 12 SLIDES:

1. Title: "Transformers: The AI Revolution"
2. What are Transformers? (3 bullets max)
3. Problem with RNNs (3 bullets)
4. Attention Mechanism (4 bullets with simple formula)
5. Multi-Head Attention (3-4 bullets)
6. Architecture Overview (simple diagram description, 4 bullets)
7. Positional Encoding (3 bullets)
8. Encoder Stack (4 bullets)
9. Decoder Stack (4 bullets)
10. Famous Models (BERT, GPT, T5, ViT - one line each)
11. Applications (4 categories, one line each)
12. Key Takeaways (3-4 final points)

REMEMBER: This is for SLIDES - keep it SHORT and VISUAL. Start now:"""


async def main():
    import inspect

    from core.registry.skills_registry import get_skills_registry

    registry = get_skills_registry()
    registry.init()

    # Get required skills
    claude_skill = registry.get_skill("claude-cli-llm")
    file_ops_skill = registry.get_skill("file-operations")
    doc_converter_skill = registry.get_skill("document-converter")
    telegram_skill = registry.get_skill("telegram-sender")

    print("Skills loaded:")
    print(f'  claude-cli-llm: {"✅" if claude_skill else "❌"}')
    print(f'  file-operations: {"✅" if file_ops_skill else "❌"}')
    print(f'  document-converter: {"✅" if doc_converter_skill else "❌"}')
    print(f'  telegram-sender: {"✅" if telegram_skill else "❌"}')

    if not all([claude_skill, file_ops_skill, doc_converter_skill, telegram_skill]):
        print("❌ Missing required skills")
        return None

    # Step 1: Generate presentation content with Claude
    print("\n🤖 Step 1: Generating presentation content with Claude...")

    generate_tool = claude_skill.tools.get("generate_text_tool")

    if inspect.iscoroutinefunction(generate_tool):
        claude_result = await generate_tool({"prompt": PROMPT, "model": "sonnet", "timeout": 300})
    else:
        claude_result = generate_tool({"prompt": PROMPT, "model": "sonnet", "timeout": 300})

    if not claude_result.get("success"):
        print(f'❌ Claude generation failed: {claude_result.get("error")}')
        return None

    markdown_content = claude_result.get("text", "")
    print(f"✅ Generated {len(markdown_content)} characters of content")

    # Step 2: Write markdown file
    print("\n📝 Step 2: Writing markdown file...")

    output_dir = os.path.expanduser("~/jotty/presentations")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_filename = f"transformer_presentation_{timestamp}.md"
    md_path = os.path.join(output_dir, md_filename)

    write_tool = file_ops_skill.tools.get("write_file_tool")
    write_result = write_tool({"path": md_path, "content": markdown_content})

    if not write_result.get("success"):
        print(f'❌ Failed to write markdown: {write_result.get("error")}')
        return None

    print(f"✅ Markdown saved: {md_path}")

    # Step 3: Convert to PDF
    print("\n📄 Step 3: Converting to PDF...")

    pdf_filename = f"transformer_presentation_{timestamp}.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)

    convert_tool = doc_converter_skill.tools.get("convert_to_pdf_tool")

    if inspect.iscoroutinefunction(convert_tool):
        pdf_result = await convert_tool(
            {
                "input_file": md_path,
                "output_file": pdf_path,
                "page_size": "a4",
                "title": "Transformer Architecture: Revolutionizing AI",
                "author": "Jotty Presentation Generator",
            }
        )
    else:
        pdf_result = convert_tool(
            {
                "input_file": md_path,
                "output_file": pdf_path,
                "page_size": "a4",
                "title": "Transformer Architecture: Revolutionizing AI",
                "author": "Jotty Presentation Generator",
            }
        )

    if not pdf_result.get("success"):
        print(f'❌ PDF conversion failed: {pdf_result.get("error")}')
        return None

    pdf_output_path = pdf_result.get("output_path", pdf_path)
    print(f"✅ PDF generated: {pdf_output_path}")

    # Step 4: Send to Telegram
    print("\n📱 Step 4: Sending to Telegram...")

    send_tool = telegram_skill.tools.get("send_telegram_file_tool")

    caption = """📊 Transformer Architecture Presentation

🤖 AI-Generated Slide Deck
📖 12 slides covering:
• What are Transformers
• Attention Mechanism
• Architecture Overview
• Famous Models (BERT, GPT, etc.)
• Applications & Future

Generated by Jotty"""

    if inspect.iscoroutinefunction(send_tool):
        telegram_result = await send_tool({"file_path": pdf_output_path, "caption": caption})
    else:
        telegram_result = send_tool({"file_path": pdf_output_path, "caption": caption})

    if telegram_result.get("success"):
        print("✅ PDF sent to Telegram!")
    else:
        print(f'⚠️ Telegram send failed: {telegram_result.get("error")}')

    return {
        "success": True,
        "md_path": md_path,
        "pdf_path": pdf_output_path,
        "telegram_sent": telegram_result.get("success", False),
    }


if __name__ == "__main__":
    result = asyncio.run(main())
    print(f"\n🎉 Final result: {result}")
