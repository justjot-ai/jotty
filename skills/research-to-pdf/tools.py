"""
Research to PDF Skill - Composite Workflow

Consolidates multi-step research workflow into single tool call:
  web search → LLM analysis → PDF generation → optional Telegram delivery

Follows Anthropic best practices:
- Clear error messages with corrective examples
- Semantic response fields (no UUIDs)
- Status reporting for progress tracking
- Proper parameter validation
"""
from typing import Dict, Any
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, async_tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("research-to-pdf")


@async_tool_wrapper(required_params=['topic'])
async def research_to_pdf_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Research topic, analyze, create PDF, and optionally send to Telegram (all-in-one).

    This composite skill consolidates 4 separate tool calls into one seamless workflow:
    1. Web search for topic
    2. LLM analysis and synthesis
    3. PDF report generation
    4. Optional Telegram delivery

    Args:
        params: Dictionary containing:
            - topic (str, required): Topic to research
            - depth (str, optional): "quick", "standard", or "deep". Defaults to "standard"
            - send_telegram (bool, optional): Send PDF to Telegram. Defaults to False
            - telegram_chat_id (str, optional): Telegram chat ID if send_telegram=True

    Returns:
        Dictionary with success, pdf_path, topic, sources_count, summary_length, telegram_sent, error
    """
    status.set_callback(params.pop('_status_callback', None))

    topic = params.get('topic')
    depth = params.get('depth', 'standard')
    send_telegram = params.get('send_telegram', False)
    telegram_chat_id = params.get('telegram_chat_id')

    # Validate depth
    valid_depths = {'quick': 5, 'standard': 10, 'deep': 20}
    if depth not in valid_depths:
        return tool_error(
            f'Parameter "depth" must be one of: quick, standard, deep. Got: {depth}. '
            f'Example: {{"topic": "AI trends", "depth": "standard"}}'
        )

    limit = valid_depths[depth]

    # Validate Telegram params
    if send_telegram and not telegram_chat_id:
        return tool_error(
            'Parameter "telegram_chat_id" required when send_telegram=True. '
            'Example: {"send_telegram": True, "telegram_chat_id": "123456789"}'
        )

    # Get registry to access skills
    from Jotty.core.capabilities.registry import get_unified_registry
    registry = get_unified_registry()

    try:
        # Step 1: Web Search
        status.emit("Searching", f"🔍 Researching: {topic}")

        web_search_skill = registry.get_skill('web-search')
        if not web_search_skill:
            return tool_error(
                'Web search skill not available. Install web-search skill first. '
                'Run: python -m Jotty.cli skills install web-search'
            )

        web_search_tool = web_search_skill.get_tool('web_search_tool')
        search_results = await web_search_tool({
            'query': topic,
            'limit': limit
        })

        if not search_results.get('success'):
            return tool_error(
                f'Web search failed: {search_results.get("error")}. '
                f'Check internet connection and search API configuration.'
            )

        sources_count = len(search_results.get('results', []))

        # Step 2: LLM Analysis
        status.emit("Analyzing", f"📊 Analyzing {sources_count} sources...")

        llm_skill = registry.get_skill('claude-cli-llm')
        if not llm_skill:
            return tool_error(
                'LLM skill not available. Ensure claude-cli-llm skill is installed.'
            )

        llm_tool = llm_skill.get_tool('claude_cli_llm_tool')

        # Create comprehensive prompt
        results_text = "\n\n".join([
            f"Source {i+1}: {r.get('title', 'Unknown')}\n{r.get('snippet', '')}"
            for i, r in enumerate(search_results.get('results', []))
        ])

        analysis_prompt = f"""Analyze and synthesize these research findings on "{topic}":

{results_text}

Create a comprehensive report with:
1. Executive Summary
2. Key Findings (3-5 bullet points)
3. Detailed Analysis
4. Conclusions
5. Sources Referenced

Format in Markdown with clear headings."""

        summary_result = await llm_tool({
            'prompt': analysis_prompt,
            'max_tokens': 4000
        })

        if not summary_result.get('success'):
            return tool_error(
                f'LLM analysis failed: {summary_result.get("error")}. '
                f'Verify LLM provider (Claude/OpenAI/Groq) is configured. '
                f'Check ANTHROPIC_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY environment variables.'
            )

        summary_text = summary_result.get('response', '')
        summary_length = len(summary_text)

        # Step 3: Create PDF
        status.emit("Creating", "📄 Generating PDF report...")

        doc_skill = registry.get_skill('document-converter')
        if not doc_skill:
            return tool_error(
                'Document converter skill not available. Install document-converter skill.'
            )

        doc_tool = doc_skill.get_tool('document_converter_tool')

        # Add title and metadata to document
        import datetime
        full_document = f"""# Research Report: {topic}

**Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Sources Analyzed**: {sources_count}
**Research Depth**: {depth}

---

{summary_text}

---

*Generated by Jotty AI Research Assistant*
"""

        pdf_result = await doc_tool({
            'content': full_document,
            'format': 'pdf',
            'title': f'Research Report - {topic}'
        })

        if not pdf_result.get('success'):
            return tool_error(
                f'PDF generation failed: {pdf_result.get("error")}. '
                f'Check document-converter skill and filesystem write permissions.'
            )

        pdf_path = pdf_result.get('path')

        # Step 4: Optional Telegram Delivery
        telegram_sent = False
        if send_telegram:
            status.emit("Sending", "📤 Sending to Telegram...")

            telegram_skill = registry.get_skill('telegram-sender')
            if not telegram_skill:
                return tool_error(
                    'Telegram sender skill not available. Install telegram-sender skill. '
                    'Also set TELEGRAM_TOKEN environment variable.'
                )

            telegram_tool = telegram_skill.get_tool('telegram_send_tool')

            telegram_result = await telegram_tool({
                'file': pdf_path,
                'chat_id': telegram_chat_id,
                'caption': f'📊 Research Report: {topic}'
            })

            if not telegram_result.get('success'):
                # Non-fatal - PDF was created successfully
                status.emit("Warning", f"⚠️ Telegram send failed: {telegram_result.get('error')}")
            else:
                telegram_sent = True

        status.emit("Complete", "✅ Research report completed!")

        return tool_response(
            pdf_path=pdf_path,
            topic=topic,
            sources_count=sources_count,
            summary_length=summary_length,
            telegram_sent=telegram_sent,
            depth=depth
        )

    except Exception as e:
        return tool_error(
            f'Research workflow failed: {str(e)}. '
            f'Verify all required skills are installed: web-search, claude-cli-llm, document-converter.'
        )


__all__ = ['research_to_pdf_tool']
