"""ArXiv to Report Skill — download ArXiv papers, extract content, analyze, generate report."""

import json
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("arxiv-to-report")

# ArXiv API base URL
ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_ID_PATTERN = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?$")

# Namespace used by Atom/ArXiv API responses
ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"

ANALYSIS_DEPTHS = {"quick", "standard", "deep"}
OUTPUT_FORMATS = {"markdown", "pdf", "html"}


def _validate_arxiv_id(arxiv_id: str) -> Optional[str]:
    """Validate and normalize an ArXiv ID. Returns cleaned ID or None."""
    cleaned = arxiv_id.strip()
    # Strip common prefixes
    for prefix in ("arxiv:", "arXiv:", "https://arxiv.org/abs/", "http://arxiv.org/abs/"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    cleaned = cleaned.strip("/").strip()
    if ARXIV_ID_PATTERN.match(cleaned):
        return cleaned
    return None


def _fetch_arxiv_metadata(arxiv_id: str) -> Dict[str, Any]:
    """Fetch paper metadata from ArXiv API.

    Returns dict with title, authors, abstract, categories, published, updated.
    Raises ValueError on API errors or paper not found.
    """
    url = f"{ARXIV_API_URL}?id_list={arxiv_id}&max_results=1"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read().decode("utf-8")
    except urllib.error.URLError as e:
        raise ValueError(f"Failed to fetch ArXiv API: {e}")

    root = ET.fromstring(data)

    # Check for entries
    entry = root.find(f"{{{ATOM_NS}}}entry")
    if entry is None:
        raise ValueError(f"No paper found for ArXiv ID: {arxiv_id}")

    # Check if ArXiv returned an error entry (no title or generic error)
    title_el = entry.find(f"{{{ATOM_NS}}}title")
    if title_el is None or title_el.text is None:
        raise ValueError(f"No paper found for ArXiv ID: {arxiv_id}")

    title = " ".join(title_el.text.strip().split())
    if title.lower().startswith("error"):
        raise ValueError(f"ArXiv API error for ID {arxiv_id}: {title}")

    # Authors
    authors = []
    for author_el in entry.findall(f"{{{ATOM_NS}}}author"):
        name_el = author_el.find(f"{{{ATOM_NS}}}name")
        if name_el is not None and name_el.text:
            authors.append(name_el.text.strip())

    # Abstract / summary
    summary_el = entry.find(f"{{{ATOM_NS}}}summary")
    abstract = ""
    if summary_el is not None and summary_el.text:
        abstract = " ".join(summary_el.text.strip().split())

    # Categories
    categories = []
    for cat_el in entry.findall(f"{{{ARXIV_NS}}}primary_category"):
        term = cat_el.get("term")
        if term:
            categories.append(term)
    for cat_el in entry.findall(f"{{{ATOM_NS}}}category"):
        term = cat_el.get("term")
        if term and term not in categories:
            categories.append(term)

    # Dates
    published_el = entry.find(f"{{{ATOM_NS}}}published")
    updated_el = entry.find(f"{{{ATOM_NS}}}updated")
    published = published_el.text.strip() if published_el is not None and published_el.text else ""
    updated = updated_el.text.strip() if updated_el is not None and updated_el.text else ""

    # PDF link
    pdf_link = ""
    for link_el in entry.findall(f"{{{ATOM_NS}}}link"):
        if link_el.get("title") == "pdf":
            pdf_link = link_el.get("href", "")
            break

    return {
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "categories": categories,
        "published": published,
        "updated": updated,
        "pdf_link": pdf_link,
    }


def _extract_key_findings(abstract: str, depth: str) -> List[str]:
    """Extract key findings from the abstract based on analysis depth.

    Uses sentence splitting heuristics to pull the most informative sentences.
    """
    if not abstract:
        return ["No abstract available for analysis."]

    # Split into sentences (handle common abbreviations)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", abstract)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not sentences:
        return [abstract[:300]]

    if depth == "quick":
        # Return first and last sentence (intro + conclusion)
        findings = [sentences[0]]
        if len(sentences) > 1:
            findings.append(sentences[-1])
        return findings[:2]
    elif depth == "standard":
        # Return up to 4 key sentences
        return sentences[:4]
    else:  # deep
        # Return all sentences as findings
        return sentences


def _generate_report_markdown(
    arxiv_id: str,
    metadata: Dict[str, Any],
    key_findings: List[str],
    depth: str,
) -> str:
    """Generate a Markdown report from paper metadata and analysis."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    authors_str = ", ".join(metadata["authors"]) if metadata["authors"] else "Unknown"
    categories_str = ", ".join(metadata["categories"]) if metadata["categories"] else "N/A"

    sections = [
        f"# ArXiv Paper Report: {metadata['title']}",
        "",
        f"**ArXiv ID:** {arxiv_id}  ",
        f"**Authors:** {authors_str}  ",
        f"**Categories:** {categories_str}  ",
        f"**Published:** {metadata.get('published', 'N/A')}  ",
        f"**Updated:** {metadata.get('updated', 'N/A')}  ",
        f"**Analysis Depth:** {depth}  ",
        f"**Report Generated:** {now}  ",
        "",
        "---",
        "",
        "## Abstract",
        "",
        metadata["abstract"] if metadata["abstract"] else "*No abstract available.*",
        "",
        "---",
        "",
        "## Key Findings",
        "",
    ]

    for i, finding in enumerate(key_findings, 1):
        sections.append(f"{i}. {finding}")

    sections.extend(
        [
            "",
            "---",
            "",
            "## Links",
            "",
            f"- [ArXiv Abstract](https://arxiv.org/abs/{arxiv_id})",
        ]
    )
    if metadata.get("pdf_link"):
        sections.append(f"- [PDF Download]({metadata['pdf_link']})")

    if depth == "deep":
        sections.extend(
            [
                "",
                "---",
                "",
                "## Metadata",
                "",
                f"- **Number of Authors:** {len(metadata['authors'])}",
                f"- **Primary Category:** {metadata['categories'][0] if metadata['categories'] else 'N/A'}",
                f"- **All Categories:** {categories_str}",
                f"- **Sentence Count in Abstract:** {len(key_findings)}",
            ]
        )

    sections.append("")
    return "\n".join(sections)


def _save_report(arxiv_id: str, content: str, output_format: str) -> str:
    """Save report to file and return the path.

    For markdown output, writes .md directly. For pdf/html, writes .md
    as the base (actual conversion would require external tools).
    """
    safe_id = arxiv_id.replace("/", "_").replace(".", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path.home() / "jotty" / "reports" / "arxiv"
    output_dir.mkdir(parents=True, exist_ok=True)

    ext = "md"
    if output_format == "html":
        ext = "html"
        # Wrap markdown content in minimal HTML
        content = (
            "<!DOCTYPE html>\n<html><head>"
            f"<title>ArXiv Report {arxiv_id}</title>"
            "<style>body{font-family:sans-serif;max-width:800px;margin:2em auto;"
            "padding:0 1em;line-height:1.6;}</style></head><body>\n"
            f"<pre>{content}</pre>\n</body></html>"
        )
    elif output_format == "pdf":
        # PDF generation would require external library; save as .md with note
        ext = "md"
        content = (
            f"<!-- PDF output requested. Convert this Markdown to PDF "
            f"using pandoc or similar tool. -->\n\n{content}"
        )

    file_path = output_dir / f"arxiv_{safe_id}_{timestamp}.{ext}"
    file_path.write_text(content, encoding="utf-8")
    return str(file_path)


@tool_wrapper(required_params=["arxiv_id"])
def arxiv_to_report_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Download ArXiv paper metadata, analyze abstract, generate report (all-in-one).

    Parameters:
        arxiv_id (str, required): ArXiv ID (e.g., "2301.12345")
        analysis_depth (str, optional): "quick", "standard", "deep". Default: "standard"
        output_format (str, optional): "markdown", "pdf", "html". Default: "pdf"

    Returns:
        success, arxiv_id, title, authors, report_path, key_findings
    """
    status.set_callback(params.pop("_status_callback", None))

    raw_id = params["arxiv_id"]
    depth = params.get("analysis_depth", "standard").lower()
    output_format = params.get("output_format", "pdf").lower()

    # Validate depth
    if depth not in ANALYSIS_DEPTHS:
        return tool_error(
            f"Invalid analysis_depth: '{depth}'. Must be one of: {sorted(ANALYSIS_DEPTHS)}"
        )

    # Validate output format
    if output_format not in OUTPUT_FORMATS:
        return tool_error(
            f"Invalid output_format: '{output_format}'. Must be one of: {sorted(OUTPUT_FORMATS)}"
        )

    # Validate ArXiv ID
    arxiv_id = _validate_arxiv_id(raw_id)
    if not arxiv_id:
        return tool_error(
            f"Invalid ArXiv ID: '{raw_id}'. Expected format like '2301.12345' or '2301.12345v2'"
        )

    # Fetch metadata
    status.fetching(f"https://arxiv.org/abs/{arxiv_id}")
    try:
        metadata = _fetch_arxiv_metadata(arxiv_id)
    except ValueError as e:
        return tool_error(str(e))

    # Analyze
    status.analyzing(metadata["title"][:50])
    key_findings = _extract_key_findings(metadata["abstract"], depth)

    # Generate report
    status.creating("report")
    report_content = _generate_report_markdown(arxiv_id, metadata, key_findings, depth)

    # Save report
    report_path = _save_report(arxiv_id, report_content, output_format)
    status.done(f"Report saved to {report_path}")

    return tool_response(
        arxiv_id=arxiv_id,
        title=metadata["title"],
        authors=metadata["authors"],
        report_path=report_path,
        key_findings=key_findings,
        categories=metadata["categories"],
        pdf_link=metadata.get("pdf_link", ""),
    )


__all__ = ["arxiv_to_report_tool"]
