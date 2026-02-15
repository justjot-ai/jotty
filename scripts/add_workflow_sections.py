#!/usr/bin/env python3
"""
Add ## Workflow sections to SKILL.md files that need them.

Qualifying skills:
1. ALL composite skills (have `## Type\ncomposite` in SKILL.md)
2. Skills with 5+ tools (complex operations)
3. Pipeline skills (name contains "pipeline")

Skips skills that already have `## Workflow` in their SKILL.md.
"""

import glob
import os
import re

SKILLS_DIR = "/var/www/sites/personal/stock_market/Jotty/skills"


def count_tools(content: str) -> int:
    """Count tool entries in a SKILL.md file."""
    pattern1 = re.findall(r"^- \*\*\w+\*\*", content, re.MULTILINE)
    pattern2 = re.findall(r"^### [`]?\w+[`]?", content, re.MULTILINE)
    return max(len(pattern1), len(pattern2))


def get_description(content: str) -> str:
    """Extract description from the ## Description section."""
    m = re.search(r"## Description\s*\n+(.*?)(?=\n##|\Z)", content, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r'^description:\s*"?(.+?)"?\s*$', content, re.MULTILINE)
    return m.group(1).strip() if m else ""


def get_base_skills(content: str) -> list:
    """Extract base skills list."""
    m = re.search(r"## Base Skills\s*\n((?:- .+\n?)+)", content)
    if m:
        return [
            line.strip("- \n")
            for line in m.group(1).strip().split("\n")
            if line.strip().startswith("-")
        ]
    return []


def get_tools(content: str) -> list:
    """Extract tool names."""
    tools = re.findall(r"^### [`]?(\w+)[`]?\s*$", content, re.MULTILINE)
    if tools:
        return tools
    tools = re.findall(r"^- \*\*(\w+)\*\*:", content, re.MULTILINE)
    return tools


def is_composite(content: str) -> bool:
    return bool(re.search(r"## Type\s*\ncomposite", content))


def is_pipeline(skill_dir_name: str) -> bool:
    return "pipeline" in skill_dir_name.lower()


def has_workflow(content: str) -> bool:
    return bool(re.search(r"^## Workflow", content, re.MULTILINE))


def humanize_skill_name(name: str) -> str:
    return name.replace("-", " ").replace("_", " ").title()


def humanize_tool_name(name: str) -> str:
    name = re.sub(r"_tool$", "", name)
    return name.replace("_", " ").title()


def build_workflow_from_steps(steps: list) -> str:
    """Build workflow markdown from a list of (step_name, detail) tuples."""
    lines = ["## Workflow", "", "```"]
    lines.append("Task Progress:")
    for i, (step_name, _) in enumerate(steps, 1):
        lines.append(f"- [ ] Step {i}: {step_name}")
    lines.append("```")
    lines.append("")
    for i, (step_name, detail) in enumerate(steps, 1):
        lines.append(f"**Step {i}: {step_name}**")
        lines.append(detail)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# Composite skill workflow overrides
COMPOSITE_OVERRIDES = {
    "search-summarize-pdf-telegram": [
        ("Search web for topic", "Use web-search to find relevant articles and data."),
        ("Summarize findings", "Condense search results into key insights using Claude LLM."),
        ("Generate PDF report", "Format the summary as a professional PDF document."),
        ("Send to Telegram", "Deliver the PDF report to the specified Telegram chat."),
    ],
    "search-summarize-pdf-telegram-v2": [
        ("Search web for topic", "Use web-search to find relevant articles and data."),
        ("Summarize findings", "Condense search results into key insights using Claude LLM."),
        ("Generate PDF report", "Format the summary as a professional PDF document."),
        ("Send to Telegram", "Deliver the PDF report to the specified Telegram chat."),
    ],
    "last30days-to-pdf-telegram": [
        (
            "Research recent topics",
            "Use last30days-claude-cli to research the topic from the past 30 days.",
        ),
        ("Generate PDF report", "Convert the research markdown into a formatted PDF document."),
        ("Send to Telegram", "Deliver the PDF report to the specified Telegram chat."),
    ],
    "last30days-to-epub-telegram": [
        (
            "Research recent topics",
            "Use last30days-claude-cli to research the topic from the past 30 days.",
        ),
        ("Generate EPUB", "Convert the research markdown into an EPUB ebook format."),
        ("Send to Telegram", "Deliver the EPUB file to the specified Telegram chat."),
    ],
    "last30days-to-pdf-remarkable": [
        (
            "Research recent topics",
            "Use last30days-claude-cli to research the topic from the past 30 days.",
        ),
        ("Generate PDF report", "Convert the research markdown into a formatted PDF document."),
        ("Send to reMarkable", "Upload the PDF to the reMarkable tablet for offline reading."),
    ],
    "research-to-pdf": [
        (
            "Search for topic information",
            "Use web-search to gather comprehensive data on the research topic.",
        ),
        (
            "Synthesize research with AI",
            "Use Claude LLM to analyze and synthesize findings into a structured report.",
        ),
        (
            "Generate PDF document",
            "Convert the markdown report into a professional PDF with proper formatting.",
        ),
    ],
    "research-to-notion": [
        (
            "Research the topic",
            "Use web-search, lead research, or competitive analysis to gather data.",
        ),
        ("Write content", "Use content-research-writer to draft structured content from research."),
        ("Save to Notion", "Create a Notion page with the documented research and citations."),
    ],
    "content-branding-pipeline": [
        ("Brainstorm domain names", "Generate creative domain name suggestions for the project."),
        ("Create artifact", "Build the HTML, presentation, or document artifact."),
        ("Apply brand guidelines", "Apply Anthropic brand colors, typography, and styling."),
        ("Apply theme", "Finalize the artifact with the selected visual theme."),
    ],
    "content-pipeline": [
        (
            "Ingest source content",
            "Load content from the source (markdown, ArXiv, HTML, PDF, or YouTube).",
        ),
        (
            "Process content",
            "Apply processors: render diagrams, handle LaTeX, download images, fix syntax.",
        ),
        (
            "Export to target format",
            "Write the processed document to one or more sinks (PDF, EPUB, DOCX, HTML, reMarkable).",
        ),
    ],
    "dev-workflow": [
        ("Generate changelog", "Analyze recent git commits to generate a version changelog."),
        ("Create skill", "Scaffold a new Jotty skill with proper structure and configuration."),
        ("Test application", "Run webapp tests with screenshots to verify functionality."),
    ],
    "media-production-pipeline": [
        ("Enhance image", "Upscale and enhance the source image to the target resolution."),
        ("Create design", "Generate a visual design from the brief using canvas-design."),
        ("Create GIF", "Produce an animated GIF optimized for Slack or web use."),
    ],
    "meeting-intelligence-pipeline": [
        (
            "Analyze meeting transcripts",
            "Process transcript files to extract speaking ratios, action items, and decisions.",
        ),
        (
            "Prepare meeting materials",
            "Create pre-read documents and agendas in Notion with relevant context.",
        ),
        ("Generate communications", "Draft internal communications summarizing meeting outcomes."),
    ],
    "notion-knowledge-pipeline": [
        (
            "Capture knowledge",
            "Save insights, decisions, and concepts to Notion with proper categorization.",
        ),
        (
            "Research and document",
            "Search Notion and web for related information, then create comprehensive documentation.",
        ),
        (
            "Create implementation plan",
            "Transform specifications into actionable implementation plans with tasks and milestones.",
        ),
    ],
    "product-launch-pipeline": [
        (
            "Brainstorm domain names",
            "Generate and validate domain name suggestions for the product.",
        ),
        ("Research leads", "Find and qualify potential leads in the target industry."),
        ("Analyze competitors", "Extract competitor ads and analyze competitive positioning."),
        ("Write launch content", "Create marketing content: outlines, drafts, or full articles."),
    ],
    "transformer-paper-pipeline": [
        (
            "Generate paper content",
            "Use Claude LLM to write a comprehensive transformer research paper.",
        ),
        ("Compile LaTeX PDF", "Convert the paper content to LaTeX and compile to PDF."),
        ("Send to Telegram", "Deliver the compiled PDF to the specified Telegram chat."),
    ],
    "investing-commodities-to-telegram": [
        ("Fetch commodities prices", "Retrieve latest commodities data from investing.com."),
        ("Format report", "Structure the price data into a formatted HTML/markdown report."),
        ("Send to Telegram", "Deliver the commodities report to the specified Telegram chat."),
    ],
    "screener-to-pdf-telegram": [
        (
            "Fetch financial data",
            "Retrieve company financials from screener.in for the specified symbols.",
        ),
        ("Analyze with AI", "Use Claude LLM to synthesize a comprehensive financial analysis."),
        ("Generate PDF", "Convert the analysis to a professionally formatted PDF."),
        ("Send to Telegram", "Deliver the PDF report to the specified Telegram chat."),
    ],
    "reddit-trending-to-justjot": [
        ("Search Reddit", "Find trending posts and discussions on the specified topic."),
        ("Format as markdown", "Structure the Reddit results into organized markdown content."),
        ("Create JustJot idea", "Save the formatted content as a JustJot idea via MCP client."),
    ],
    "trending-topics-to-ideas": [
        ("Get trending topics", "Fetch trending topics from Reddit, V2V, or web search."),
        ("Research each topic", "Gather additional details for each topic via web search."),
        ("Synthesize content", "Use Claude LLM to create structured summaries for each topic."),
        ("Create JustJot ideas", "Save each synthesized topic as a JustJot idea with sections."),
    ],
    "v2v-to-pdf-telegram-remarkable": [
        ("Search V2V trending", "Find trending topics and discussions on V2V.ai."),
        ("Generate PDF", "Convert the research results into a formatted PDF document."),
        ("Send to Telegram", "Deliver the PDF to the specified Telegram chat."),
        ("Upload to reMarkable", "Upload the PDF to the reMarkable tablet for reading."),
    ],
    "v2v-trending-search": [
        ("Search V2V.ai", "Query V2V.ai for trending topics matching the search query."),
        ("Generate report", "Format the search results into a structured markdown report."),
    ],
    "notion-knowledge-capture": [
        (
            "Parse content",
            "Analyze the input content to determine type (FAQ, how-to, decision, concept).",
        ),
        (
            "Format for Notion",
            "Structure the content with proper headers, tags, and categorization.",
        ),
        (
            "Save to Notion",
            "Create a Notion page in the appropriate wiki or database with linking.",
        ),
    ],
    "notion-meeting-intelligence": [
        (
            "Gather Notion context",
            "Search Notion for related project pages and prior meeting notes.",
        ),
        ("Enrich with research", "Use Claude LLM to add research context and analysis."),
        ("Create meeting materials", "Generate pre-read documents and external agendas in Notion."),
    ],
    "notion-research-documentation": [
        (
            "Search Notion workspace",
            "Query Notion for pages and databases related to the research topic.",
        ),
        ("Synthesize findings", "Analyze found pages and synthesize insights with AI assistance."),
        (
            "Create documentation",
            "Write comprehensive documentation with proper citations in Notion.",
        ),
    ],
    "notion-spec-to-implementation": [
        ("Read specification", "Fetch and parse the specification from the Notion page."),
        ("Generate plan", "Break down the spec into tasks, milestones, and dependencies."),
        (
            "Create task database",
            "Write the implementation plan to a Notion database with tracking.",
        ),
    ],
    "slack-gif-creator": [
        (
            "Define animation parameters",
            "Set GIF type (message/emoji), dimensions, and animation style.",
        ),
        (
            "Generate animated frames",
            "Create animation frames based on the description and parameters.",
        ),
        ("Optimize for Slack", "Compress and optimize the GIF to meet Slack size requirements."),
    ],
    "stock-research-deep": [
        ("Initial broad research", "Execute parallel searches across 12 financial aspects."),
        ("Gap analysis", "AI identifies missing information and knowledge gaps."),
        ("Targeted follow-up research", "Fill knowledge gaps with focused additional searches."),
        (
            "Progressive synthesis",
            "Build comprehensive report section by section with accumulated context.",
        ),
        (
            "Quality validation and refinement",
            "Validate coverage, identify weak sections, and refine with additional research.",
        ),
        ("Generate and deliver report", "Convert to PDF and send to Telegram."),
    ],
    "notebooklm-pdf": [
        ("Prepare content", "Read markdown or text content from input."),
        ("Upload to NotebookLM", "Send the content to Google NotebookLM for processing."),
        ("Generate PDF", "Retrieve the AI-formatted PDF from NotebookLM."),
    ],
}

# Multi-tool skill workflow definitions
MULTITOOL_WORKFLOWS = {
    "android-automation": [
        ("Connect to device", "Establish ADB connection to the target Android device."),
        ("Navigate to target", "Launch the app or navigate to the required screen."),
        ("Perform actions", "Execute taps, swipes, text input, and other UI interactions."),
        ("Capture results", "Take screenshots or extract text to verify outcomes."),
        ("Clean up", "Close apps and disconnect from the device."),
    ],
    "database-tools": [
        ("Connect to database", "Establish connection to the target database using credentials."),
        (
            "Inspect schema",
            "List tables and describe their structure to understand the data model.",
        ),
        ("Execute queries", "Run SQL queries to read, insert, update, or analyze data."),
        ("Export results", "Export query results to the desired format (CSV, JSON, etc.)."),
        ("Close connection", "Clean up by closing the database connection."),
    ],
    "github": [
        ("Authenticate and list repos", "Connect to GitHub and list available repositories."),
        ("Inspect repository", "Get repo details, list issues, and review pull requests."),
        ("Search code", "Search for specific code patterns across repositories."),
        ("Create artifacts", "Create issues, pull requests, or update repository files."),
    ],
    "file-operations": [
        ("Locate files", "Search and list files in the target directory."),
        ("Read and inspect", "Read file contents and check file metadata."),
        ("Process and write", "Create directories, write files, or modify content."),
        ("Verify results", "Confirm file operations completed successfully."),
    ],
    "financial-visualization": [
        (
            "Extract financial data",
            "Parse research results to extract structured financial metrics.",
        ),
        (
            "Select chart types",
            "AI-powered selection of optimal chart types based on data completeness.",
        ),
        (
            "Generate charts",
            "Create financial charts: revenue growth, profitability, valuation, health scores.",
        ),
        (
            "Generate data tables",
            "Build formatted tables for financial statements, ratios, and peer comparisons.",
        ),
        (
            "Add insights",
            "Generate AI narratives, detect anomalies, and forecast trends for each chart.",
        ),
    ],
    "claude-api-llm": [
        ("Prepare prompt", "Construct the prompt with system context and user input."),
        (
            "Select model and parameters",
            "Choose the appropriate Claude model and set temperature/tokens.",
        ),
        ("Execute LLM call", "Send the request to the Claude API and receive the response."),
        (
            "Process response",
            "Parse the response, extract structured content, and handle tool use.",
        ),
    ],
    "summarize": [
        ("Identify content source", "Determine input type: text, file, URL, or conversation."),
        ("Extract content", "Read and parse the source content into processable text."),
        (
            "Generate summary",
            "Use AI to create a summary at the requested length and detail level.",
        ),
        ("Format output", "Structure the summary with key points, sections, or bullet points."),
    ],
    "spotify": [
        ("Search catalog", "Search Spotify for artists, tracks, albums, or playlists."),
        ("Get details", "Retrieve detailed information about artists, albums, or tracks."),
        ("Analyze features", "Get audio features and recommendations based on preferences."),
        ("Manage playlists", "Create playlists and add recommended tracks."),
    ],
    "pptx-editor": [
        ("Load presentation", "Open the target PowerPoint file for editing."),
        ("Inspect slides", "List slides and analyze their current content and layout."),
        ("Modify content", "Update text, images, charts, and formatting across slides."),
        ("Apply styling", "Set themes, colors, fonts, and transitions."),
        ("Save presentation", "Export the modified presentation to the desired format."),
    ],
    "justjot-converters": [
        ("Fetch source content", "Retrieve the original idea or content from JustJot."),
        ("Select target format", "Determine the output format (PDF, EPUB, Markdown, etc.)."),
        ("Transform sections", "Convert each section type to the target format representation."),
        ("Assemble document", "Combine transformed sections into a cohesive document."),
        ("Export result", "Save or deliver the final converted document."),
    ],
    "xlsx-tools": [
        ("Load spreadsheet", "Open the target Excel file for reading or editing."),
        ("Inspect sheets", "List worksheets and examine their structure and data."),
        ("Read and analyze data", "Extract data from specific cells, ranges, or sheets."),
        ("Modify content", "Update cells, add formulas, create charts, or format data."),
        ("Save spreadsheet", "Write the modified spreadsheet to the output path."),
    ],
    "pdf-tools": [
        ("Load PDF", "Open the target PDF file for processing."),
        ("Extract content", "Read text, images, or metadata from the PDF."),
        ("Modify document", "Merge, split, rotate pages, or add watermarks."),
        ("Export result", "Save the processed PDF to the output path."),
    ],
    "docx-tools": [
        ("Load document", "Open the target Word document for processing."),
        ("Inspect content", "List sections, paragraphs, tables, and images."),
        ("Modify content", "Update text, add tables, insert images, or change formatting."),
        ("Export document", "Save the modified document to the desired format."),
    ],
    "skill-discovery": [
        ("Search skills catalog", "Query the registry for skills matching the task description."),
        ("Analyze capabilities", "Evaluate skill capabilities, tools, and compatibility."),
        ("Rank matches", "Score and rank skills by relevance to the task."),
        ("Return recommendations", "Provide the best-matching skills with usage examples."),
    ],
    "pmi-market-data": [
        ("Select data type", "Choose the market data category (stocks, indices, ETFs, etc.)."),
        ("Fetch market data", "Retrieve real-time or historical data from PlanMyInvesting API."),
        ("Process and format", "Parse the response and format data for analysis."),
        ("Return results", "Deliver structured market data with key metrics highlighted."),
    ],
    "algorithmic-art": [
        (
            "Choose algorithm",
            "Select the generative art algorithm (fractal, L-system, cellular automata, etc.).",
        ),
        (
            "Configure parameters",
            "Set colors, dimensions, iterations, and algorithm-specific settings.",
        ),
        ("Generate artwork", "Execute the algorithm to produce the visual output."),
        ("Export image", "Save the generated artwork in the desired format (PNG, SVG, etc.)."),
    ],
    "visual-inspector": [
        ("Capture target", "Take a screenshot or load the image to inspect."),
        ("Analyze visuals", "Use AI vision to identify UI elements, layout, and content."),
        ("Detect issues", "Find visual bugs, accessibility problems, or design inconsistencies."),
        ("Generate report", "Produce a structured inspection report with findings."),
    ],
    "trello": [
        ("Connect to board", "Authenticate and select the target Trello board."),
        ("Inspect board state", "List columns and cards to understand current project status."),
        ("Manage cards", "Create, update, or move cards between lists."),
        ("Report status", "Summarize board activity and current state."),
    ],
    "text-utils": [
        ("Parse input text", "Read and analyze the input text content."),
        ("Apply transformations", "Perform text operations: format, convert, extract, or clean."),
        ("Validate output", "Check the transformed text meets requirements."),
        ("Return result", "Deliver the processed text in the requested format."),
    ],
    "slack": [
        ("Connect to workspace", "Authenticate with the Slack workspace."),
        ("Select channel", "List channels and identify the target channel."),
        ("Compose message", "Prepare the message content with formatting."),
        ("Send to Slack", "Post the message or upload files to the channel."),
    ],
    "obsidian": [
        ("Open vault", "Connect to the target Obsidian vault directory."),
        ("Search and navigate", "Search notes and explore backlinks to find relevant content."),
        ("Read or create notes", "Read existing notes or create new ones with proper formatting."),
        ("Update and link", "Update note content and maintain bidirectional links."),
    ],
    "notion": [
        ("Connect to workspace", "Authenticate with the Notion workspace via API."),
        ("Search and navigate", "Search pages and databases to find relevant content."),
        (
            "Create or update pages",
            "Create new pages or update existing ones with structured content.",
        ),
        ("Organize content", "Manage databases, properties, and page relationships."),
    ],
    "gif-creator": [
        ("Define animation", "Specify the animation type, dimensions, and duration."),
        ("Generate frames", "Create individual frames for the animation sequence."),
        ("Assemble GIF", "Combine frames into an animated GIF with proper timing."),
        ("Optimize output", "Compress and optimize the GIF for target size requirements."),
    ],
    "discord": [
        ("Connect to server", "Authenticate with the Discord bot credentials."),
        ("Select channel", "List and identify the target Discord channel."),
        ("Compose content", "Prepare the message or file to send."),
        ("Send to Discord", "Post the message, upload files, or add reactions."),
    ],
    "whatsapp": [
        ("Connect to WhatsApp", "Authenticate with the WhatsApp Business API."),
        ("Select recipient", "Identify the target phone number or group."),
        ("Compose message", "Prepare text, file, location, or contact to send."),
        ("Send message", "Deliver the message via WhatsApp."),
    ],
    "pmi-watchlist": [
        ("Fetch watchlist", "Retrieve the user's watchlist from PlanMyInvesting."),
        ("Get stock data", "Fetch current prices and metrics for watchlist stocks."),
        ("Analyze performance", "Compare stocks against benchmarks and signals."),
        ("Generate alerts", "Create alerts for price targets or significant changes."),
    ],
    "pmi-trading": [
        ("Analyze position", "Review current portfolio positions and market conditions."),
        ("Calculate signals", "Compute technical and fundamental trading signals."),
        ("Generate recommendations", "Produce buy/sell/hold recommendations with reasoning."),
        ("Execute or report", "Place trades or deliver the analysis report."),
    ],
    "document-converter": [
        ("Load source document", "Read the input file (Markdown, HTML, DOCX, etc.)."),
        ("Parse content", "Extract text, images, and structure from the source."),
        ("Convert format", "Transform content to the target format (PDF, EPUB, DOCX, HTML)."),
        ("Save output", "Write the converted document to the output path."),
    ],
}


def insert_workflow(content: str, workflow_section: str) -> str:
    """Insert workflow section before ## Triggers or at the end."""
    triggers_match = re.search(r"^## Triggers", content, re.MULTILINE)
    if triggers_match:
        pos = triggers_match.start()
        return content[:pos] + workflow_section + "\n" + content[pos:]

    category_match = re.search(r"^## Category", content, re.MULTILINE)
    if category_match:
        pos = category_match.start()
        return content[:pos] + workflow_section + "\n" + content[pos:]

    return content.rstrip() + "\n\n" + workflow_section


def generate_workflow_for_composite(skill_name: str, description: str, base_skills: list) -> str:
    """Generate workflow for composite skills from their description and base skills."""
    skill_actions = {
        "web-search": (
            "Search the web",
            "Use web-search to find relevant articles and data on the topic.",
        ),
        "web-scraper": (
            "Scrape web content",
            "Scrape web pages to extract structured content and data.",
        ),
        "claude-cli-llm": (
            "Synthesize with AI",
            "Use Claude LLM to analyze, summarize, and synthesize the gathered information.",
        ),
        "claude-api-llm": ("Process with AI", "Use Claude API to analyze and process the content."),
        "document-converter": (
            "Generate document",
            "Convert the processed content into the target document format.",
        ),
        "telegram-sender": (
            "Send to Telegram",
            "Deliver the final document or message to the specified Telegram chat.",
        ),
        "remarkable-sender": (
            "Send to reMarkable",
            "Upload the document to the reMarkable tablet.",
        ),
        "slack": ("Send to Slack", "Post the results to the configured Slack channel."),
        "mcp-justjot": (
            "Create JustJot idea",
            "Save the synthesized content as a structured idea on JustJot.ai.",
        ),
        "notion": ("Save to Notion", "Create or update a Notion page with the structured content."),
        "file-operations": ("Manage files", "Read, write, or organize files on the filesystem."),
        "github": ("Interact with GitHub", "Perform GitHub operations."),
        "shell-exec": ("Execute shell commands", "Run shell commands for system operations."),
        "image-generator": ("Generate images", "Create images using AI models."),
        "gif-creator": ("Create GIF animation", "Generate animated GIF."),
        "voice": ("Process audio", "Handle speech-to-text or text-to-speech conversion."),
        "video-downloader": ("Download video", "Download video content from the specified source."),
        "arxiv-downloader": (
            "Fetch ArXiv paper",
            "Download and parse the academic paper from ArXiv.",
        ),
        "last30days-claude-cli": (
            "Research recent topics",
            "Research topics from the last 30 days using Claude CLI.",
        ),
        "screener-financials": (
            "Fetch financial data",
            "Retrieve financial data from screener.in.",
        ),
        "investing-commodities": (
            "Fetch commodities data",
            "Get latest commodities prices from investing.com.",
        ),
        "brand-guidelines": (
            "Apply brand styling",
            "Apply brand colors, typography, and guidelines.",
        ),
    }

    # Parse description for numbered steps
    desc_steps = re.findall(
        r"\d+\.\s+\*\*(.+?)\*\*(?:\s*\((\w+)\))?:\s*(.+?)(?=\n\d+\.|\n\n|\Z)",
        description,
        re.DOTALL,
    )

    steps = []
    if desc_steps and len(desc_steps) >= 2:
        for step_name, _role, step_skill in desc_steps:
            step_name = step_name.strip()
            step_skill = step_skill.strip()
            skill_key = step_skill.lower().strip()
            if skill_key in skill_actions:
                action, detail = skill_actions[skill_key]
                steps.append((step_name, detail))
            else:
                steps.append((step_name, f"Execute {step_skill} to handle {step_name.lower()}."))
    elif base_skills:
        for skill in base_skills:
            skill_key = skill.strip()
            if skill_key in skill_actions:
                action, detail = skill_actions[skill_key]
                steps.append((action, detail))
            else:
                human = humanize_skill_name(skill_key)
                steps.append(
                    (f"Run {human}", f"Execute the {skill_key} skill to process the data.")
                )

    if not steps:
        return ""

    return build_workflow_from_steps(steps)


def main():
    skill_dirs = sorted(glob.glob(os.path.join(SKILLS_DIR, "*/SKILL.md")))
    print(f"Found {len(skill_dirs)} SKILL.md files")

    updated_count = 0
    skipped_has_workflow = 0
    skipped_not_qualifying = 0
    updated_skills = []

    for skill_md_path in skill_dirs:
        skill_dir = os.path.basename(os.path.dirname(skill_md_path))

        with open(skill_md_path, "r") as f:
            content = f.read()

        if has_workflow(content):
            skipped_has_workflow += 1
            continue

        skill_composite = is_composite(content)
        skill_pipeline = is_pipeline(skill_dir)
        tool_count = count_tools(content)
        skill_multitool = tool_count >= 5

        if not (skill_composite or skill_pipeline or skill_multitool):
            skipped_not_qualifying += 1
            continue

        skill_name = skill_dir
        description = get_description(content)
        base_skills = get_base_skills(content)

        # Generate workflow
        workflow = ""

        if skill_name in COMPOSITE_OVERRIDES:
            workflow = build_workflow_from_steps(COMPOSITE_OVERRIDES[skill_name])
        elif skill_name in MULTITOOL_WORKFLOWS:
            workflow = build_workflow_from_steps(MULTITOOL_WORKFLOWS[skill_name])
        elif skill_composite or skill_pipeline:
            workflow = generate_workflow_for_composite(skill_name, description, base_skills)
        else:
            # Fallback: should not happen if we covered all multi-tool skills
            print(
                f"  WARNING: No workflow definition for multi-tool skill: {skill_name} ({tool_count} tools)"
            )
            skipped_not_qualifying += 1
            continue

        if not workflow:
            print(f"  WARNING: Could not generate workflow for {skill_name}")
            skipped_not_qualifying += 1
            continue

        new_content = insert_workflow(content, workflow)

        with open(skill_md_path, "w") as f:
            f.write(new_content)

        updated_count += 1
        reason = []
        if skill_composite:
            reason.append("composite")
        if skill_pipeline:
            reason.append("pipeline")
        if skill_multitool:
            reason.append(f"{tool_count} tools")
        updated_skills.append((skill_name, ", ".join(reason)))
        print(f"  Added workflow to: {skill_name} ({', '.join(reason)})")

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total SKILL.md files:         {len(skill_dirs)}")
    print(f"Already had workflow:          {skipped_has_workflow}")
    print(f"Not qualifying:               {skipped_not_qualifying}")
    print(f"Workflows added:              {updated_count}")
    print(f"{'=' * 60}")

    if updated_skills:
        print(f"\nUpdated skills:")
        for name, reason in updated_skills:
            print(f"  - {name} ({reason})")


if __name__ == "__main__":
    main()
