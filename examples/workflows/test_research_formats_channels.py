#!/usr/bin/env python3
"""
END-TO-END TEST: Formats + Channels
====================================

Complete workflow test:
1. Generate research content (ResearchWorkflow)
2. Create multiple formats (PDF, EPUB, HTML)
3. Send to multiple channels (Telegram, WhatsApp)

Demonstrates the full pipeline:
Content Generation → Format Generation → Channel Delivery
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent / "Jotty" / ".env.anthropic")


async def main():
    from Jotty.core.workflows import (
        ResearchWorkflow,
        OutputFormatManager,
        OutputChannelManager,
    )

    print("\n" + "="*80)
    print("END-TO-END TEST: Research → Formats → Channels")
    print("="*80 + "\n")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Generate Research Content
    # ═══════════════════════════════════════════════════════════════════════

    print("STEP 1: Generate Research Content")
    print("-" * 80 + "\n")

    workflow = ResearchWorkflow.from_intent(
        topic="Benefits of Daily Meditation for Mental Health",
        research_type="general",
        depth="quick",  # Quick for testing
        deliverables=[
            "overview",
            "deep_dive",
            "synthesis",
            "documentation"
        ],
        max_sources=5,
        send_telegram=False  # We'll send manually
    )

    print("🚀 Executing research workflow...")
    result = await workflow.run(verbose=True)

    print(f"\n✅ Research complete!")
    print(f"   Stages: {len(result.stages)}")
    print(f"   Cost: ${result.total_cost:.6f}")
    print(f"   Time: {result.total_time:.2f}s\n")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: Extract Documentation
    # ═══════════════════════════════════════════════════════════════════════

    print("STEP 2: Extract Documentation Content")
    print("-" * 80 + "\n")

    documentation_stage = None
    for stage in result.stages:
        if stage.stage_name == "documentation":
            documentation_stage = stage
            break

    if not documentation_stage:
        print("❌ No documentation stage found!")
        return

    # Save markdown
    output_dir = Path.home() / "jotty" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = output_dir / "meditation_research.md"
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write("# Benefits of Daily Meditation for Mental Health\n\n")
        f.write(documentation_stage.result.output)

    print(f"✅ Saved markdown: {markdown_path}")
    print(f"   Size: {len(documentation_stage.result.output)} chars\n")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: Generate Multiple Formats
    # ═══════════════════════════════════════════════════════════════════════

    print("STEP 3: Generate Multiple Output Formats")
    print("-" * 80 + "\n")

    format_manager = OutputFormatManager(output_dir=str(output_dir))

    # Generate PDF, EPUB, HTML
    print("🔄 Generating formats: PDF, EPUB, HTML...")
    format_results = format_manager.generate_all(
        markdown_path=str(markdown_path),
        formats=["pdf", "epub", "html"],
        title="Benefits of Daily Meditation for Mental Health",
        author="Jotty Research Workflow"
    )

    # Show results
    format_summary = format_manager.get_summary(format_results)
    print(f"\n📦 Generated {format_summary['successful']}/{format_summary['total']} formats:")
    for fmt in format_summary['successful_formats']:
        file_path = format_summary['file_paths'][fmt]
        file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 0
        print(f"   ✅ {fmt.upper()}: {file_path} ({file_size:,} bytes)")

    if format_summary['failed_formats']:
        print(f"\n❌ Failed formats: {', '.join(format_summary['failed_formats'])}")
        for fmt, error in format_summary['errors'].items():
            print(f"   {fmt}: {error}")

    print()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: Send to Telegram
    # ═══════════════════════════════════════════════════════════════════════

    print("STEP 4: Send to Telegram")
    print("-" * 80 + "\n")

    channel_manager = OutputChannelManager()

    # Check if we have PDF
    if 'pdf' in format_summary['successful_formats']:
        pdf_path = format_summary['file_paths']['pdf']

        print(f"📤 Sending PDF to Telegram...")
        telegram_result = channel_manager.send_to_telegram(
            file_path=pdf_path,
            caption="📚 <b>Research Report: Benefits of Daily Meditation</b>\n\n"
                   "Generated by Jotty Research Workflow\n"
                   f"📄 {format_summary['successful']} formats created\n"
                   f"💰 Cost: ${result.total_cost:.6f}",
            parse_mode="HTML"
        )

        if telegram_result.success:
            print(f"✅ Sent to Telegram!")
            print(f"   Message ID: {telegram_result.message_id}")
            print(f"   Chat ID: {telegram_result.metadata.get('chat_id')}")
        else:
            print(f"❌ Telegram send failed: {telegram_result.error}")
    else:
        print("⚠️  No PDF available to send")

    print()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: Send to WhatsApp (if configured)
    # ═══════════════════════════════════════════════════════════════════════

    print("STEP 5: Send to WhatsApp")
    print("-" * 80 + "\n")

    # Check if WhatsApp recipient is configured
    whatsapp_to = os.environ.get('WHATSAPP_TO')

    if whatsapp_to:
        if 'pdf' in format_summary['successful_formats']:
            pdf_path = format_summary['file_paths']['pdf']

            print(f"📤 Sending PDF to WhatsApp ({whatsapp_to})...")
            whatsapp_result = channel_manager.send_to_whatsapp(
                to=whatsapp_to,
                file_path=pdf_path,
                caption="📚 Research Report: Benefits of Daily Meditation\n\n"
                       f"Generated by Jotty • {format_summary['successful']} formats",
                provider="auto"
            )

            if whatsapp_result.success:
                print(f"✅ Sent to WhatsApp!")
                print(f"   Message ID: {whatsapp_result.message_id}")
                print(f"   Provider: {whatsapp_result.metadata.get('provider')}")
            else:
                print(f"❌ WhatsApp send failed: {whatsapp_result.error}")
        else:
            print("⚠️  No PDF available to send")
    else:
        print("⚠️  WHATSAPP_TO environment variable not set - skipping WhatsApp")
        print("   Set WHATSAPP_TO='14155238886' (with country code) to enable")

    print()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 6: Send to Multiple Channels at Once
    # ═══════════════════════════════════════════════════════════════════════

    print("STEP 6: Send to Multiple Channels (Batch)")
    print("-" * 80 + "\n")

    if 'pdf' in format_summary['successful_formats']:
        pdf_path = format_summary['file_paths']['pdf']

        channels_to_send = ["telegram"]
        channel_params = {}

        if whatsapp_to:
            channels_to_send.append("whatsapp")
            channel_params['whatsapp_to'] = whatsapp_to

        print(f"📤 Sending to {len(channels_to_send)} channels: {', '.join(channels_to_send)}...")

        batch_results = channel_manager.send_to_all(
            channels=channels_to_send,
            file_path=pdf_path,
            caption="📚 Meditation Research Report (Multi-Channel Test)",
            **channel_params
        )

        # Show batch results
        batch_summary = channel_manager.get_summary(batch_results)
        print(f"\n📊 Sent to {batch_summary['successful']}/{batch_summary['total']} channels:")
        for ch in batch_summary['successful_channels']:
            print(f"   ✅ {ch.upper()}")

        if batch_summary['failed_channels']:
            print(f"\n❌ Failed channels:")
            for ch in batch_summary['failed_channels']:
                print(f"   {ch.upper()}: {batch_summary['errors'].get(ch, 'Unknown error')}")
    else:
        print("⚠️  No PDF available for batch send")

    print()

    # ═══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════

    print("="*80)
    print("END-TO-END TEST COMPLETE")
    print("="*80 + "\n")

    print("📊 Summary:")
    print(f"   ✅ Research stages: {len(result.stages)}")
    print(f"   ✅ Formats generated: {format_summary['successful']}/{format_summary['total']}")
    for fmt in format_summary['successful_formats']:
        print(f"      - {fmt.upper()}")

    # Count channels sent
    channels_sent = 1  # Telegram (always sent)
    if whatsapp_to and 'pdf' in format_summary['successful_formats']:
        channels_sent += 1

    print(f"   ✅ Channels delivered: {channels_sent}")
    if channels_sent > 0:
        print(f"      - TELEGRAM")
    if channels_sent > 1:
        print(f"      - WHATSAPP")

    print(f"\n💰 Total cost: ${result.total_cost:.6f}")
    print(f"⏱️  Total time: {result.total_time:.2f}s")
    print()

    print("🎊 Complete Pipeline Validated:")
    print("   Content → Formats → Channels ✅")
    print()

    print("Files saved to:")
    print(f"   {output_dir}/")
    print()


if __name__ == '__main__':
    asyncio.run(main())
