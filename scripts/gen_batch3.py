"""Batch 3: Content, media, finance, and productivity skills (20 skills)."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from generate_skills import create_skill

# ── 41. email-template-builder ────────────────────────────────────
create_skill(
    name="email-template-builder",
    frontmatter_name="building-email-templates",
    description="Create HTML email templates with inline CSS for cross-client compatibility. Use when the user wants to build email, create email template, HTML email.",
    category="content-creation",
    capabilities=["generate", "document"],
    triggers=["email template", "html email", "newsletter", "email builder", "inline css email"],
    eval_tool="build_email_template_tool",
    eval_input={
        "subject": "Welcome!",
        "body": "Hello and welcome to our service.",
        "template": "basic",
    },
    tool_docs="""### build_email_template_tool
Build an HTML email template with inline CSS.

**Parameters:**
- `subject` (str, required): Email subject
- `body` (str, required): Email body text or HTML
- `template` (str, optional): Template style: basic, newsletter, promotional, transactional (default: basic)
- `brand_color` (str, optional): Primary brand color hex (default: #007bff)
- `footer_text` (str, optional): Footer text
- `preheader` (str, optional): Preheader text

**Returns:**
- `success` (bool)
- `html` (str): Complete HTML email
- `subject` (str): Email subject""",
    tools_code=r'''"""Email Template Builder Skill - create HTML emails with inline CSS."""
from typing import Dict, Any
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("email-template-builder")

TEMPLATES = {
    "basic": """<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{subject}</title></head><body style="margin:0;padding:0;background-color:#f4f4f4;font-family:Arial,Helvetica,sans-serif;">
<span style="display:none;max-height:0;overflow:hidden;">{preheader}</span>
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color:#f4f4f4;">
<tr><td align="center" style="padding:20px 0;">
<table role="presentation" width="600" cellpadding="0" cellspacing="0" style="background-color:#ffffff;border-radius:8px;overflow:hidden;">
<tr><td style="background-color:{brand_color};padding:30px;text-align:center;">
<h1 style="color:#ffffff;margin:0;font-size:24px;">{subject}</h1></td></tr>
<tr><td style="padding:30px;color:#333333;font-size:16px;line-height:1.6;">{body}</td></tr>
<tr><td style="padding:20px 30px;background-color:#f8f9fa;color:#666666;font-size:12px;text-align:center;border-top:1px solid #eeeeee;">{footer}</td></tr>
</table></td></tr></table></body></html>""",
    "newsletter": """<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{subject}</title></head><body style="margin:0;padding:0;background-color:#f4f4f4;font-family:Georgia,serif;">
<span style="display:none;max-height:0;overflow:hidden;">{preheader}</span>
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color:#f4f4f4;">
<tr><td align="center" style="padding:20px 0;">
<table role="presentation" width="600" cellpadding="0" cellspacing="0" style="background-color:#ffffff;">
<tr><td style="padding:20px;text-align:center;border-bottom:3px solid {brand_color};">
<h1 style="color:{brand_color};margin:0;font-size:28px;">{subject}</h1></td></tr>
<tr><td style="padding:30px;color:#333333;font-size:16px;line-height:1.8;">{body}</td></tr>
<tr><td style="padding:20px;background-color:#333333;color:#ffffff;font-size:12px;text-align:center;">{footer}</td></tr>
</table></td></tr></table></body></html>""",
    "promotional": """<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{subject}</title></head><body style="margin:0;padding:0;background-color:#f4f4f4;font-family:Arial,Helvetica,sans-serif;">
<span style="display:none;max-height:0;overflow:hidden;">{preheader}</span>
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color:#f4f4f4;">
<tr><td align="center" style="padding:20px 0;">
<table role="presentation" width="600" cellpadding="0" cellspacing="0" style="background-color:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.1);">
<tr><td style="background:linear-gradient(135deg,{brand_color},#333333);padding:40px;text-align:center;">
<h1 style="color:#ffffff;margin:0;font-size:32px;text-transform:uppercase;">{subject}</h1></td></tr>
<tr><td style="padding:30px;color:#333333;font-size:16px;line-height:1.6;text-align:center;">{body}</td></tr>
<tr><td style="padding:15px 30px;text-align:center;">
<a href="#" style="display:inline-block;background-color:{brand_color};color:#ffffff;text-decoration:none;padding:14px 40px;border-radius:6px;font-size:18px;font-weight:bold;">Shop Now</a></td></tr>
<tr><td style="padding:20px;color:#999999;font-size:11px;text-align:center;border-top:1px solid #eeeeee;">{footer}</td></tr>
</table></td></tr></table></body></html>""",
    "transactional": """<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{subject}</title></head><body style="margin:0;padding:0;background-color:#f4f4f4;font-family:Arial,Helvetica,sans-serif;">
<span style="display:none;max-height:0;overflow:hidden;">{preheader}</span>
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color:#f4f4f4;">
<tr><td align="center" style="padding:20px 0;">
<table role="presentation" width="600" cellpadding="0" cellspacing="0" style="background-color:#ffffff;">
<tr><td style="padding:20px 30px;border-bottom:2px solid {brand_color};">
<h2 style="color:#333333;margin:0;">{subject}</h2></td></tr>
<tr><td style="padding:30px;color:#333333;font-size:14px;line-height:1.6;">{body}</td></tr>
<tr><td style="padding:15px 30px;background-color:#f8f9fa;color:#666666;font-size:11px;text-align:center;">{footer}</td></tr>
</table></td></tr></table></body></html>""",
}


@tool_wrapper(required_params=["subject", "body"])
def build_email_template_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Build an HTML email template with inline CSS."""
    status.set_callback(params.pop("_status_callback", None))
    subject = params["subject"]
    body = params["body"]
    template_name = params.get("template", "basic").lower()
    brand_color = params.get("brand_color", "#007bff")
    footer = params.get("footer_text", "You received this email because you are subscribed.")
    preheader = params.get("preheader", "")

    if template_name not in TEMPLATES:
        return tool_error(f"Unknown template: {template_name}. Use: {list(TEMPLATES.keys())}")

    html = TEMPLATES[template_name].format(
        subject=subject, body=body, brand_color=brand_color,
        footer=footer, preheader=preheader,
    )
    return tool_response(html=html, subject=subject, template=template_name,
                         char_count=len(html))


__all__ = ["build_email_template_tool"]
''',
)

# ── 42. seo-content-optimizer ─────────────────────────────────────
create_skill(
    name="seo-content-optimizer",
    frontmatter_name="optimizing-seo-content",
    description="Analyze text for SEO: keyword density, readability score, meta tag suggestions. Use when the user wants to check SEO, keyword density, readability, optimize content.",
    category="content-creation",
    capabilities=["analyze"],
    triggers=["seo", "keyword density", "readability", "optimize content", "meta tags"],
    eval_tool="analyze_seo_tool",
    eval_input={
        "text": "Python is a great programming language for data science and machine learning.",
        "keywords": ["python", "data science"],
    },
    tool_docs="""### analyze_seo_tool
Analyze text for SEO metrics.

**Parameters:**
- `text` (str, required): Text content to analyze
- `keywords` (list, optional): Target keywords to check density
- `title` (str, optional): Page title to analyze

**Returns:**
- `success` (bool)
- `word_count` (int): Total words
- `readability` (dict): Readability scores
- `keyword_density` (dict): Keyword frequencies
- `suggestions` (list): SEO improvement suggestions""",
    tools_code=r'''"""SEO Content Optimizer Skill - analyze text for SEO metrics."""
import re
import math
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("seo-content-optimizer")


def _count_syllables(word: str) -> int:
    word = word.lower().strip()
    if not word:
        return 0
    if len(word) <= 3:
        return 1
    word = re.sub(r"(?:es|ed|e)$", "", word) or word
    vowels = re.findall(r"[aeiouy]+", word)
    return max(1, len(vowels))


def _flesch_reading_ease(text: str) -> float:
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    if not sentences or not words:
        return 0.0
    total_syllables = sum(_count_syllables(w) for w in words)
    asl = len(words) / len(sentences)
    asw = total_syllables / len(words)
    score = 206.835 - 1.015 * asl - 84.6 * asw
    return round(max(0, min(100, score)), 1)


def _reading_level(score: float) -> str:
    if score >= 90:
        return "5th grade (very easy)"
    elif score >= 80:
        return "6th grade (easy)"
    elif score >= 70:
        return "7th grade (fairly easy)"
    elif score >= 60:
        return "8th-9th grade (standard)"
    elif score >= 50:
        return "10th-12th grade (fairly difficult)"
    elif score >= 30:
        return "College (difficult)"
    return "College graduate (very difficult)"


@tool_wrapper(required_params=["text"])
def analyze_seo_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze text for SEO: keyword density, readability, suggestions."""
    status.set_callback(params.pop("_status_callback", None))
    text = params["text"]
    keywords = params.get("keywords", [])
    title = params.get("title", "")

    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    word_count = len(words)
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sentence_count = len(sentences)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Readability
    flesch = _flesch_reading_ease(text)
    avg_sentence_len = round(word_count / max(sentence_count, 1), 1)
    avg_word_len = round(sum(len(w) for w in words) / max(word_count, 1), 1)

    # Keyword density
    kw_density = {}
    for kw in keywords:
        kw_lower = kw.lower()
        count = text.lower().count(kw_lower)
        density = round((count / max(word_count, 1)) * 100, 2)
        kw_density[kw] = {"count": count, "density_pct": density}

    # Suggestions
    suggestions = []
    if word_count < 300:
        suggestions.append("Content is short. Aim for 300+ words for better SEO.")
    if avg_sentence_len > 25:
        suggestions.append("Sentences are long. Aim for under 20 words per sentence.")
    if flesch < 60:
        suggestions.append("Readability is low. Simplify language for broader audience.")
    if title and len(title) > 60:
        suggestions.append("Title is over 60 characters. Keep under 60 for search results.")
    if title and len(title) < 30:
        suggestions.append("Title is short. Aim for 30-60 characters.")
    for kw, info in kw_density.items():
        if info["density_pct"] < 0.5:
            suggestions.append(f"Keyword '{kw}' density is low ({info['density_pct']}%). Aim for 1-2%.")
        elif info["density_pct"] > 3.0:
            suggestions.append(f"Keyword '{kw}' density is high ({info['density_pct']}%). May be seen as stuffing.")
    if not any(s.startswith(("# ", "## ")) for s in text.split("\n")):
        suggestions.append("No headings detected. Use headings (H1, H2) to structure content.")

    return tool_response(
        word_count=word_count, sentence_count=sentence_count,
        paragraph_count=len(paragraphs),
        readability={"flesch_score": flesch, "level": _reading_level(flesch),
                     "avg_sentence_length": avg_sentence_len,
                     "avg_word_length": avg_word_len},
        keyword_density=kw_density,
        suggestions=suggestions,
    )


__all__ = ["analyze_seo_tool"]
''',
)

# ── 43. transcript-formatter ──────────────────────────────────────
create_skill(
    name="transcript-formatter",
    frontmatter_name="formatting-transcripts",
    description="Clean raw transcripts with speaker labels, timestamps, and paragraph breaks. Use when the user wants to clean transcript, format transcript, add speaker labels.",
    category="content-creation",
    capabilities=["generate", "analyze"],
    triggers=["transcript", "format transcript", "clean transcript", "speaker labels", "subtitles"],
    eval_tool="format_transcript_tool",
    eval_input={
        "text": "00:00:05 John: Hello everyone.\n00:00:08 Jane: Hi John, thanks for joining.\n00:00:12 John: Happy to be here. Let me share my thoughts on the project."
    },
    tool_docs="""### format_transcript_tool
Format a raw transcript with speaker labels and timestamps.

**Parameters:**
- `text` (str, required): Raw transcript text
- `merge_speakers` (bool, optional): Merge consecutive lines from same speaker (default: true)
- `include_timestamps` (bool, optional): Keep timestamps in output (default: true)

**Returns:**
- `success` (bool)
- `formatted` (str): Cleaned transcript
- `speakers` (list): Unique speakers found
- `duration` (str): Estimated duration""",
    tools_code=r'''"""Transcript Formatter Skill - clean raw transcripts."""
import re
from typing import Dict, Any, List, Optional, Tuple
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("transcript-formatter")

TS_PATTERNS = [
    re.compile(r"^(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)\s*[-:]?\s*(.*)"),
    re.compile(r"^\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.*)"),
    re.compile(r"^(\d{1,2}:\d{2})\s+(.*)"),
]

SPEAKER_PATTERN = re.compile(r"^([A-Za-z][A-Za-z0-9 _.'-]{0,40}):\s+(.*)")


def _parse_line(line: str) -> Tuple[Optional[str], Optional[str], str]:
    timestamp = None
    speaker = None
    text = line.strip()
    if not text:
        return None, None, ""
    for pat in TS_PATTERNS:
        m = pat.match(text)
        if m:
            timestamp = m.group(1)
            text = m.group(2).strip()
            break
    m = SPEAKER_PATTERN.match(text)
    if m:
        speaker = m.group(1).strip()
        text = m.group(2).strip()
    return timestamp, speaker, text


@tool_wrapper(required_params=["text"])
def format_transcript_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Format a raw transcript with speaker labels and timestamps."""
    status.set_callback(params.pop("_status_callback", None))
    raw = params["text"]
    merge = params.get("merge_speakers", True)
    show_ts = params.get("include_timestamps", True)

    lines = raw.strip().split("\n")
    entries = []
    for line in lines:
        ts, speaker, text = _parse_line(line)
        if text:
            entries.append({"timestamp": ts, "speaker": speaker, "text": text})

    if not entries:
        return tool_error("No transcript content found")

    if merge:
        merged = []
        for entry in entries:
            if (merged and entry["speaker"] and
                    merged[-1]["speaker"] == entry["speaker"]):
                merged[-1]["text"] += " " + entry["text"]
                if not merged[-1]["timestamp"] and entry["timestamp"]:
                    merged[-1]["timestamp"] = entry["timestamp"]
            else:
                merged.append(dict(entry))
        entries = merged

    speakers = sorted(set(e["speaker"] for e in entries if e["speaker"]))

    output_lines = []
    for e in entries:
        parts = []
        if show_ts and e["timestamp"]:
            parts.append(f"[{e['timestamp']}]")
        if e["speaker"]:
            parts.append(f"{e['speaker']}:")
        parts.append(e["text"])
        output_lines.append(" ".join(parts))

    first_ts = next((e["timestamp"] for e in entries if e["timestamp"]), None)
    last_ts = next((e["timestamp"] for e in reversed(entries) if e["timestamp"]), None)

    return tool_response(
        formatted="\n\n".join(output_lines),
        speakers=speakers, speaker_count=len(speakers),
        line_count=len(entries),
        first_timestamp=first_ts, last_timestamp=last_ts,
    )


__all__ = ["format_transcript_tool"]
''',
)

# ── 44. press-release-generator ───────────────────────────────────
create_skill(
    name="press-release-generator",
    frontmatter_name="generating-press-releases",
    description="Generate press release templates following AP style with proper structure. Use when the user wants to write press release, news release, PR template.",
    category="content-creation",
    capabilities=["generate", "document"],
    triggers=["press release", "news release", "PR template", "media release", "announcement"],
    eval_tool="generate_press_release_tool",
    eval_input={
        "headline": "Acme Corp Launches New Product",
        "company": "Acme Corp",
        "body_points": ["Revolutionary new widget", "Available Q1 2025", "Priced at $99"],
    },
    tool_docs="""### generate_press_release_tool
Generate a structured press release.

**Parameters:**
- `headline` (str, required): Press release headline
- `company` (str, required): Company name
- `body_points` (list, required): Key points to include
- `city` (str, optional): Dateline city (default: New York)
- `contact_name` (str, optional): Media contact name
- `contact_email` (str, optional): Media contact email
- `quote_attribution` (str, optional): Name for the quote
- `quote_title` (str, optional): Title of person quoted

**Returns:**
- `success` (bool)
- `press_release` (str): Formatted press release text
- `word_count` (int): Total word count""",
    tools_code=r'''"""Press Release Generator Skill - AP style press releases."""
from datetime import datetime
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("press-release-generator")


@tool_wrapper(required_params=["headline", "company", "body_points"])
def generate_press_release_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a structured press release in AP style."""
    status.set_callback(params.pop("_status_callback", None))
    headline = params["headline"]
    company = params["company"]
    body_points = params["body_points"]
    city = params.get("city", "New York").upper()
    contact_name = params.get("contact_name", "Media Relations")
    contact_email = params.get("contact_email", f"press@{company.lower().replace(' ', '')}.com")
    quote_attr = params.get("quote_attribution", f"CEO of {company}")
    quote_title = params.get("quote_title", "Chief Executive Officer")

    today = datetime.now().strftime("%B %d, %Y")

    # Build lead paragraph
    lead = (f"{city}, {today} -- {company} today announced {body_points[0].lower() if body_points else 'a major development'}. "
            f"{'This ' + body_points[1].lower() + '.' if len(body_points) > 1 else ''}")

    # Build body paragraphs
    body_paras = []
    for i, point in enumerate(body_points[2:], start=1):
        body_paras.append(point if point.endswith(".") else point + ".")

    # Build quote
    quote = (f'"We are thrilled to share this news with our customers and partners," '
             f'said {quote_attr}, {quote_title} of {company}. '
             f'"This represents a significant milestone for our organization."')

    # About section
    about = (f"About {company}\n"
             f"{company} is a leading organization committed to innovation and excellence. "
             f"For more information, visit www.{company.lower().replace(' ', '')}.com.")

    # Contact
    contact = (f"Media Contact:\n{contact_name}\n{company}\n{contact_email}")

    # Assemble
    sections = [
        "FOR IMMEDIATE RELEASE",
        "",
        headline.upper(),
        "",
        lead,
    ]
    if body_paras:
        sections.append("")
        sections.append("\n\n".join(body_paras))
    sections.extend(["", quote, "", about, "", contact, "", "###"])

    pr_text = "\n".join(sections)

    return tool_response(
        press_release=pr_text, word_count=len(pr_text.split()),
        headline=headline, company=company,
    )


__all__ = ["generate_press_release_tool"]
''',
)

# ── 45. copywriting-assistant ─────────────────────────────────────
create_skill(
    name="copywriting-assistant",
    frontmatter_name="assisting-copywriting",
    description="Generate marketing copy using proven frameworks: AIDA, PAS, BAB, 4Ps, FAB. Use when the user wants to write marketing copy, ad copy, sales copy, AIDA framework.",
    category="content-creation",
    capabilities=["generate"],
    triggers=[
        "copywriting",
        "marketing copy",
        "AIDA",
        "PAS",
        "ad copy",
        "sales copy",
        "landing page copy",
    ],
    eval_tool="generate_copy_tool",
    eval_input={
        "product": "AI Writing Assistant",
        "audience": "content marketers",
        "framework": "AIDA",
    },
    tool_docs="""### generate_copy_tool
Generate marketing copy using a copywriting framework.

**Parameters:**
- `product` (str, required): Product or service name
- `audience` (str, required): Target audience
- `framework` (str, optional): AIDA, PAS, BAB, 4Ps, FAB (default: AIDA)
- `key_benefit` (str, optional): Main benefit to highlight
- `tone` (str, optional): professional, casual, urgent, friendly (default: professional)

**Returns:**
- `success` (bool)
- `copy` (dict): Copy organized by framework sections
- `framework` (str): Framework used""",
    tools_code=r'''"""Copywriting Assistant Skill - marketing copy frameworks."""
from typing import Dict, Any
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("copywriting-assistant")

FRAMEWORKS = {
    "aida": {
        "name": "AIDA",
        "sections": ["Attention", "Interest", "Desire", "Action"],
        "descriptions": [
            "Hook that grabs attention",
            "Build interest with relevant information",
            "Create desire by showing benefits",
            "Clear call to action",
        ],
    },
    "pas": {
        "name": "PAS",
        "sections": ["Problem", "Agitate", "Solution"],
        "descriptions": [
            "Identify the pain point",
            "Amplify the problem and its consequences",
            "Present your solution",
        ],
    },
    "bab": {
        "name": "BAB",
        "sections": ["Before", "After", "Bridge"],
        "descriptions": [
            "Describe current situation (the struggle)",
            "Paint the ideal future state",
            "Show how product bridges the gap",
        ],
    },
    "4ps": {
        "name": "4Ps",
        "sections": ["Promise", "Picture", "Proof", "Push"],
        "descriptions": [
            "Lead with a bold promise",
            "Help them visualize the outcome",
            "Provide evidence and credibility",
            "Encourage immediate action",
        ],
    },
    "fab": {
        "name": "FAB",
        "sections": ["Features", "Advantages", "Benefits"],
        "descriptions": [
            "List the key features",
            "Explain advantages over alternatives",
            "Translate to real user benefits",
        ],
    },
}

TONE_MODIFIERS = {
    "professional": {"opener": "Introducing", "cta": "Get started today."},
    "casual": {"opener": "Hey there!", "cta": "Give it a try!"},
    "urgent": {"opener": "Don't miss out!", "cta": "Act now before it's too late!"},
    "friendly": {"opener": "We're excited to share", "cta": "Join us today!"},
}


@tool_wrapper(required_params=["product", "audience"])
def generate_copy_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate marketing copy using a copywriting framework."""
    status.set_callback(params.pop("_status_callback", None))
    product = params["product"]
    audience = params["audience"]
    fw_name = params.get("framework", "AIDA").lower()
    benefit = params.get("key_benefit", f"save time and boost productivity")
    tone = params.get("tone", "professional").lower()

    if fw_name not in FRAMEWORKS:
        return tool_error(f"Unknown framework: {fw_name}. Use: {list(FRAMEWORKS.keys())}")

    fw = FRAMEWORKS[fw_name]
    tone_mod = TONE_MODIFIERS.get(tone, TONE_MODIFIERS["professional"])

    copy_sections = {}

    if fw_name == "aida":
        copy_sections = {
            "Attention": f"{tone_mod['opener']} {product} - the solution {audience} have been waiting for.",
            "Interest": f"As a {audience.rstrip('s')} professional, you know the challenges of staying competitive. {product} addresses these challenges head-on with cutting-edge capabilities.",
            "Desire": f"Imagine being able to {benefit}. With {product}, you can transform your workflow and achieve results that set you apart from the competition.",
            "Action": f"{tone_mod['cta']} Try {product} and see the difference for yourself.",
        }
    elif fw_name == "pas":
        copy_sections = {
            "Problem": f"As {audience}, you're constantly struggling to keep up with demands. The current tools just aren't cutting it.",
            "Agitate": f"Every day without a proper solution means lost opportunities, wasted hours, and falling behind competitors. Can you really afford to keep doing things the old way?",
            "Solution": f"{product} helps you {benefit}. {tone_mod['cta']}",
        }
    elif fw_name == "bab":
        copy_sections = {
            "Before": f"Right now, {audience} spend too much time on manual processes that drain productivity and creativity.",
            "After": f"Imagine a world where you can {benefit} effortlessly. Where your workflow is smooth, efficient, and actually enjoyable.",
            "Bridge": f"{product} is the bridge. It transforms your current workflow into a streamlined process. {tone_mod['cta']}",
        }
    elif fw_name == "4ps":
        copy_sections = {
            "Promise": f"{product} will help {audience} {benefit} - guaranteed.",
            "Picture": f"Picture yourself completing projects faster, with better results, and with less stress. That's what life looks like with {product}.",
            "Proof": f"Trusted by professionals worldwide, {product} has helped thousands of {audience} transform their workflow.",
            "Push": f"{tone_mod['cta']} Start your free trial of {product} now.",
        }
    elif fw_name == "fab":
        copy_sections = {
            "Features": f"{product} comes equipped with advanced tools designed specifically for {audience}.",
            "Advantages": f"Unlike other solutions, {product} offers a seamless experience that integrates into your existing workflow without disruption.",
            "Benefits": f"The result? You {benefit}, giving you more time to focus on what truly matters. {tone_mod['cta']}",
        }

    return tool_response(
        copy=copy_sections,
        framework=fw["name"],
        framework_sections=fw["sections"],
        section_descriptions=dict(zip(fw["sections"], fw["descriptions"])),
        product=product, audience=audience, tone=tone,
    )


__all__ = ["generate_copy_tool"]
''',
)

# ── 46. blog-post-writer ──────────────────────────────────────────
create_skill(
    name="blog-post-writer",
    frontmatter_name="writing-blog-posts",
    description="Generate blog post outlines with SEO metadata, headings, and section templates. Use when the user wants to write blog post, blog outline, article structure.",
    category="content-creation",
    capabilities=["generate", "document"],
    triggers=["blog post", "blog outline", "article", "blog writing", "content outline"],
    eval_tool="generate_blog_outline_tool",
    eval_input={
        "title": "10 Best Practices for Remote Work",
        "keywords": ["remote work", "productivity"],
        "sections": 5,
    },
    tool_docs="""### generate_blog_outline_tool
Generate a blog post outline with SEO metadata.

**Parameters:**
- `title` (str, required): Blog post title
- `keywords` (list, optional): Target SEO keywords
- `sections` (int, optional): Number of sections (default: 5)
- `tone` (str, optional): professional, casual, academic (default: professional)
- `word_target` (int, optional): Target word count (default: 1500)

**Returns:**
- `success` (bool)
- `outline` (dict): Structured blog outline
- `seo_meta` (dict): SEO metadata suggestions""",
    tools_code=r'''"""Blog Post Writer Skill - generate blog outlines with SEO."""
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("blog-post-writer")

SECTION_TEMPLATES = [
    "Introduction - Hook the reader and state the problem",
    "Background - Provide context and establish expertise",
    "Key Concept - Explain the main idea in detail",
    "Practical Steps - Actionable advice with examples",
    "Common Mistakes - What to avoid and how to fix them",
    "Expert Tips - Advanced insights from the field",
    "Case Study - Real-world example or success story",
    "Tools & Resources - Recommended tools and further reading",
    "FAQ - Answer common questions",
    "Conclusion - Summarize key takeaways and CTA",
]


@tool_wrapper(required_params=["title"])
def generate_blog_outline_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a blog post outline with SEO metadata."""
    status.set_callback(params.pop("_status_callback", None))
    title = params["title"]
    keywords = params.get("keywords", [])
    num_sections = min(max(int(params.get("sections", 5)), 3), 10)
    tone = params.get("tone", "professional")
    word_target = int(params.get("word_target", 1500))

    # Build outline sections
    sections = []
    templates = SECTION_TEMPLATES[:num_sections]
    # Always include intro and conclusion
    if num_sections >= 2:
        templates[0] = SECTION_TEMPLATES[0]
        templates[-1] = SECTION_TEMPLATES[-1]

    words_per_section = word_target // num_sections

    for i, template in enumerate(templates):
        parts = template.split(" - ", 1)
        heading = parts[0]
        guidance = parts[1] if len(parts) > 1 else ""
        sections.append({
            "order": i + 1,
            "heading": f"## {heading}" if i > 0 else f"## Introduction",
            "guidance": guidance,
            "word_target": words_per_section,
            "key_points": [
                f"Point related to {keywords[0]}" if keywords else "Main argument",
                "Supporting evidence or example",
                "Transition to next section",
            ],
        })

    # SEO metadata
    meta_desc = f"Learn about {title.lower()}. "
    if keywords:
        meta_desc += f"Discover key insights on {', '.join(keywords[:3])}."
    else:
        meta_desc += "A comprehensive guide with practical tips and expert advice."
    meta_desc = meta_desc[:160]

    slug = title.lower()
    for ch in "?!@#$%^&*()+=[]{}|;:'\",.<>/":
        slug = slug.replace(ch, "")
    slug = slug.strip().replace(" ", "-")
    slug = "-".join(part for part in slug.split("-") if part)

    seo_meta = {
        "title_tag": title[:60],
        "meta_description": meta_desc,
        "slug": slug,
        "keywords": keywords,
        "word_target": word_target,
        "estimated_read_time": f"{max(1, word_target // 250)} min",
    }

    return tool_response(
        outline={"title": title, "sections": sections, "tone": tone},
        seo_meta=seo_meta,
        section_count=len(sections),
    )


__all__ = ["generate_blog_outline_tool"]
''',
)

# ── 47. rss-feed-reader ───────────────────────────────────────────
create_skill(
    name="rss-feed-reader",
    frontmatter_name="reading-rss-feeds",
    description="Fetch and parse RSS/Atom feeds to extract articles, titles, and summaries. Use when the user wants to read RSS, parse feed, get news feed, Atom feed.",
    category="data-analysis",
    capabilities=["data-fetch"],
    triggers=["rss", "atom feed", "news feed", "parse feed", "syndication"],
    eval_tool="fetch_rss_tool",
    eval_input={"url": "https://feeds.bbci.co.uk/news/rss.xml"},
    deps="requests",
    tool_docs="""### fetch_rss_tool
Fetch and parse an RSS/Atom feed.

**Parameters:**
- `url` (str, required): Feed URL
- `limit` (int, optional): Max items to return (default: 10)

**Returns:**
- `success` (bool)
- `feed_title` (str): Feed title
- `items` (list): Feed entries with title, link, description, date
- `item_count` (int): Number of items returned""",
    tools_code=r'''"""RSS Feed Reader Skill - fetch and parse RSS/Atom feeds."""
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("rss-feed-reader")

# Common RSS/Atom namespaces
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "dc": "http://purl.org/dc/elements/1.1/",
    "content": "http://purl.org/rss/1.0/modules/content/",
    "media": "http://search.yahoo.com/mrss/",
}


def _text(el: Optional[ET.Element], default: str = "") -> str:
    if el is None:
        return default
    return (el.text or "").strip()


def _parse_rss(root: ET.Element, limit: int) -> dict:
    channel = root.find("channel")
    if channel is None:
        return {"title": "", "items": []}
    title = _text(channel.find("title"))
    description = _text(channel.find("description"))
    items = []
    for item in channel.findall("item")[:limit]:
        items.append({
            "title": _text(item.find("title")),
            "link": _text(item.find("link")),
            "description": _text(item.find("description"))[:500],
            "pub_date": _text(item.find("pubDate")),
            "author": _text(item.find("dc:creator", NS)) or _text(item.find("author")),
        })
    return {"title": title, "description": description, "items": items}


def _parse_atom(root: ET.Element, limit: int) -> dict:
    ns = NS["atom"]
    title = _text(root.find(f"{{{ns}}}title"))
    items = []
    for entry in root.findall(f"{{{ns}}}entry")[:limit]:
        link_el = entry.find(f"{{{ns}}}link")
        link = link_el.get("href", "") if link_el is not None else ""
        summary = _text(entry.find(f"{{{ns}}}summary")) or _text(entry.find(f"{{{ns}}}content"))
        items.append({
            "title": _text(entry.find(f"{{{ns}}}title")),
            "link": link,
            "description": summary[:500],
            "pub_date": _text(entry.find(f"{{{ns}}}updated")) or _text(entry.find(f"{{{ns}}}published")),
            "author": _text(entry.find(f"{{{ns}}}author/{{{ns}}}name")),
        })
    return {"title": title, "description": "", "items": items}


@tool_wrapper(required_params=["url"])
def fetch_rss_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch and parse an RSS or Atom feed."""
    status.set_callback(params.pop("_status_callback", None))
    url = params["url"]
    limit = min(max(int(params.get("limit", 10)), 1), 100)

    try:
        import requests
        resp = requests.get(url, timeout=15, headers={"User-Agent": "JottyRSSReader/1.0"})
        resp.raise_for_status()
        xml_text = resp.text
    except Exception as e:
        return tool_error(f"Failed to fetch feed: {e}")

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        return tool_error(f"Failed to parse XML: {e}")

    tag = root.tag.lower().split("}")[-1] if "}" in root.tag else root.tag.lower()
    if tag == "rss" or root.find("channel") is not None:
        result = _parse_rss(root, limit)
    elif "feed" in tag:
        result = _parse_atom(root, limit)
    else:
        return tool_error(f"Unrecognized feed format: {root.tag}")

    return tool_response(
        feed_title=result["title"],
        feed_description=result.get("description", ""),
        items=result["items"],
        item_count=len(result["items"]),
        feed_url=url,
    )


__all__ = ["fetch_rss_tool"]
''',
)

# ── 48. notification-aggregator ───────────────────────────────────
create_skill(
    name="notification-aggregator",
    frontmatter_name="aggregating-notifications",
    description="Route and format notifications for multiple channels: email, webhook, log, console. Use when the user wants to send notifications, route alerts, aggregate messages.",
    category="workflow-automation",
    capabilities=["code"],
    triggers=["notification", "alert", "notify", "webhook", "send alert", "push notification"],
    eval_tool="send_notification_tool",
    eval_input={
        "message": "Deployment completed successfully",
        "channels": ["console", "log"],
        "level": "info",
    },
    tool_docs="""### send_notification_tool
Route a notification to specified channels.

**Parameters:**
- `message` (str, required): Notification message
- `channels` (list, optional): Channels: console, log, webhook, file (default: ["console"])
- `level` (str, optional): info, warning, error, critical (default: info)
- `title` (str, optional): Notification title
- `webhook_url` (str, optional): Webhook URL for webhook channel
- `file_path` (str, optional): File path for file channel

**Returns:**
- `success` (bool)
- `delivered` (list): Channels where delivery succeeded
- `failed` (list): Channels where delivery failed""",
    tools_code=r'''"""Notification Aggregator Skill - route notifications to channels."""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("notification-aggregator")
logger = logging.getLogger("jotty.skills.notification-aggregator")

LEVELS = {"info": "INFO", "warning": "WARNING", "error": "ERROR", "critical": "CRITICAL"}
LEVEL_ICONS = {"info": "[i]", "warning": "[!]", "error": "[X]", "critical": "[!!!]"}


def _format_notification(title: str, message: str, level: str) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level.upper(),
        "title": title,
        "message": message,
    }


def _send_console(notification: dict) -> bool:
    icon = LEVEL_ICONS.get(notification["level"].lower(), "[i]")
    ts = notification["timestamp"][:19]
    print(f"{icon} [{ts}] {notification['title']}: {notification['message']}")
    return True


def _send_log(notification: dict) -> bool:
    level_map = {"INFO": logging.INFO, "WARNING": logging.WARNING,
                 "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}
    log_level = level_map.get(notification["level"], logging.INFO)
    logger.log(log_level, "%s: %s", notification["title"], notification["message"])
    return True


def _send_webhook(notification: dict, url: str) -> bool:
    try:
        import requests
        resp = requests.post(url, json=notification, timeout=10)
        return resp.status_code < 400
    except Exception:
        return False


def _send_file(notification: dict, file_path: str) -> bool:
    try:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            f.write(json.dumps(notification) + "\n")
        return True
    except Exception:
        return False


@tool_wrapper(required_params=["message"])
def send_notification_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Route a notification to specified channels."""
    status.set_callback(params.pop("_status_callback", None))
    message = params["message"]
    channels = params.get("channels", ["console"])
    level = params.get("level", "info").lower()
    title = params.get("title", "Notification")
    webhook_url = params.get("webhook_url", "")
    file_path = params.get("file_path", "")

    if level not in LEVELS:
        return tool_error(f"Invalid level: {level}. Use: {list(LEVELS.keys())}")

    notification = _format_notification(title, message, level)
    delivered = []
    failed = []

    for channel in channels:
        ch = channel.lower().strip()
        ok = False
        if ch == "console":
            ok = _send_console(notification)
        elif ch == "log":
            ok = _send_log(notification)
        elif ch == "webhook":
            if not webhook_url:
                failed.append({"channel": ch, "error": "webhook_url required"})
                continue
            ok = _send_webhook(notification, webhook_url)
        elif ch == "file":
            if not file_path:
                failed.append({"channel": ch, "error": "file_path required"})
                continue
            ok = _send_file(notification, file_path)
        else:
            failed.append({"channel": ch, "error": f"Unknown channel: {ch}"})
            continue

        if ok:
            delivered.append(ch)
        else:
            failed.append({"channel": ch, "error": "delivery failed"})

    return tool_response(
        delivered=delivered, failed=failed,
        notification=notification,
        total_channels=len(channels),
    )


__all__ = ["send_notification_tool"]
''',
)

# ── 49. image-resizer ────────────────────────────────────────────
create_skill(
    name="image-resizer",
    frontmatter_name="resizing-images",
    description="Resize, crop, and convert images using Pillow. Use when the user wants to resize image, crop image, convert image format, thumbnail.",
    category="content-creation",
    capabilities=["generate"],
    triggers=["resize image", "crop image", "image thumbnail", "convert image", "scale image"],
    eval_tool="resize_image_tool",
    eval_input={"input_path": "photo.jpg", "width": 800, "height": 600},
    deps="Pillow",
    tool_docs="""### resize_image_tool
Resize an image to specified dimensions.

**Parameters:**
- `input_path` (str, required): Path to source image
- `output_path` (str, optional): Path for resized image (default: adds _resized suffix)
- `width` (int, optional): Target width in pixels
- `height` (int, optional): Target height in pixels
- `maintain_aspect` (bool, optional): Maintain aspect ratio (default: true)
- `quality` (int, optional): JPEG quality 1-100 (default: 85)
- `format` (str, optional): Output format: JPEG, PNG, WEBP (default: same as input)

**Returns:**
- `success` (bool)
- `output_path` (str): Path to resized image
- `original_size` (dict): Original width and height
- `new_size` (dict): New width and height""",
    tools_code=r'''"""Image Resizer Skill - resize images using Pillow."""
from pathlib import Path
from typing import Dict, Any
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("image-resizer")


@tool_wrapper(required_params=["input_path"])
def resize_image_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Resize an image to specified dimensions."""
    status.set_callback(params.pop("_status_callback", None))

    try:
        from PIL import Image
    except ImportError:
        return tool_error("Pillow is required. Install with: pip install Pillow")

    input_path = Path(params["input_path"])
    if not input_path.exists():
        return tool_error(f"File not found: {input_path}")

    width = params.get("width")
    height = params.get("height")
    maintain_aspect = params.get("maintain_aspect", True)
    quality = min(max(int(params.get("quality", 85)), 1), 100)
    out_format = params.get("format")

    if not width and not height:
        return tool_error("Specify at least width or height")

    try:
        img = Image.open(input_path)
        orig_w, orig_h = img.size

        if maintain_aspect:
            if width and height:
                ratio = min(width / orig_w, height / orig_h)
            elif width:
                ratio = width / orig_w
            else:
                ratio = height / orig_h
            new_w = int(orig_w * ratio)
            new_h = int(orig_h * ratio)
        else:
            new_w = int(width) if width else orig_w
            new_h = int(height) if height else orig_h

        resized = img.resize((new_w, new_h), Image.LANCZOS)

        # Determine output
        suffix = out_format.lower() if out_format else input_path.suffix.lstrip(".")
        if suffix == "jpg":
            suffix = "jpeg"
        out_path_str = params.get("output_path")
        if out_path_str:
            out_path = Path(out_path_str)
        else:
            out_path = input_path.with_name(f"{input_path.stem}_resized.{suffix}")

        save_kwargs = {}
        if suffix in ("jpeg", "jpg", "webp"):
            save_kwargs["quality"] = quality
        if suffix == "jpeg" and resized.mode in ("RGBA", "P"):
            resized = resized.convert("RGB")

        resized.save(str(out_path), **save_kwargs)

        return tool_response(
            output_path=str(out_path),
            original_size={"width": orig_w, "height": orig_h},
            new_size={"width": new_w, "height": new_h},
            format=suffix,
            quality=quality,
        )
    except Exception as e:
        return tool_error(f"Image processing failed: {e}")


__all__ = ["resize_image_tool"]
''',
)

# ── 50. ocr-extractor ────────────────────────────────────────────
create_skill(
    name="ocr-extractor",
    frontmatter_name="extracting-ocr-text",
    description="Extract text from images using OCR. Requires pytesseract and Tesseract engine. Use when the user wants to OCR, extract text from image, read image text.",
    category="data-analysis",
    capabilities=["analyze"],
    triggers=["ocr", "extract text", "image to text", "read image", "tesseract"],
    eval_tool="ocr_extract_tool",
    eval_input={"image_path": "screenshot.png"},
    deps="pytesseract, Pillow, Tesseract-OCR engine (system package)",
    tool_docs="""### ocr_extract_tool
Extract text from an image using OCR.

**Parameters:**
- `image_path` (str, required): Path to image file
- `language` (str, optional): OCR language code (default: eng)
- `psm` (int, optional): Page segmentation mode 0-13 (default: 3)

**Returns:**
- `success` (bool)
- `text` (str): Extracted text
- `confidence` (float): Average confidence score
- `word_count` (int): Number of words extracted

**Note:** Requires Tesseract OCR engine installed on the system.
Install: `sudo apt install tesseract-ocr` or `brew install tesseract`
Python: `pip install pytesseract Pillow`""",
    tools_code=r'''"""OCR Extractor Skill - extract text from images.

NOTE: Requires external dependencies:
  System: tesseract-ocr (apt install tesseract-ocr / brew install tesseract)
  Python: pytesseract, Pillow (pip install pytesseract Pillow)
"""
from pathlib import Path
from typing import Dict, Any
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("ocr-extractor")


@tool_wrapper(required_params=["image_path"])
def ocr_extract_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Extract text from an image using OCR (Tesseract)."""
    status.set_callback(params.pop("_status_callback", None))

    image_path = Path(params["image_path"])
    if not image_path.exists():
        return tool_error(f"File not found: {image_path}")

    language = params.get("language", "eng")
    psm = int(params.get("psm", 3))

    try:
        from PIL import Image
    except ImportError:
        return tool_error("Pillow is required. Install with: pip install Pillow")

    try:
        import pytesseract
    except ImportError:
        return tool_error(
            "pytesseract is required. Install with: pip install pytesseract\n"
            "Also requires Tesseract OCR engine: sudo apt install tesseract-ocr"
        )

    try:
        img = Image.open(image_path)
        custom_config = f"--psm {psm}"
        text = pytesseract.image_to_string(img, lang=language, config=custom_config)

        # Get confidence data
        try:
            data = pytesseract.image_to_data(img, lang=language, output_type=pytesseract.Output.DICT,
                                              config=custom_config)
            confidences = [int(c) for c in data.get("conf", []) if str(c).isdigit() and int(c) > 0]
            avg_confidence = round(sum(confidences) / len(confidences), 1) if confidences else 0.0
        except Exception:
            avg_confidence = 0.0

        cleaned = text.strip()
        word_count = len(cleaned.split()) if cleaned else 0

        return tool_response(
            text=cleaned, confidence=avg_confidence,
            word_count=word_count, language=language,
            image_path=str(image_path),
        )
    except Exception as e:
        return tool_error(f"OCR failed: {e}")


__all__ = ["ocr_extract_tool"]
''',
)

# ── 51. archive-manager ───────────────────────────────────────────
create_skill(
    name="archive-manager",
    frontmatter_name="managing-archives",
    description="Create and extract ZIP, TAR, and GZIP archives using Python stdlib. Use when the user wants to zip, unzip, tar, extract archive, compress files.",
    category="development",
    capabilities=["code"],
    triggers=["zip", "unzip", "tar", "archive", "compress", "extract", "gzip"],
    eval_tool="create_archive_tool",
    eval_input={"files": ["file1.txt", "file2.txt"], "output": "archive.zip", "format": "zip"},
    tool_docs="""### create_archive_tool
Create an archive from files.

**Parameters:**
- `files` (list, required): List of file paths to archive
- `output` (str, required): Output archive path
- `format` (str, optional): zip, tar, tar.gz, tar.bz2 (default: zip)

**Returns:**
- `success` (bool)
- `output_path` (str): Path to created archive
- `file_count` (int): Number of files archived
- `size_bytes` (int): Archive size in bytes

### extract_archive_tool
Extract an archive.

**Parameters:**
- `archive_path` (str, required): Path to archive
- `output_dir` (str, optional): Extraction directory (default: current dir)

**Returns:**
- `success` (bool)
- `extracted_files` (list): List of extracted file paths
- `output_dir` (str): Extraction directory

### list_archive_tool
List contents of an archive.

**Parameters:**
- `archive_path` (str, required): Path to archive

**Returns:**
- `success` (bool)
- `files` (list): List of files in archive
- `total_size` (int): Total uncompressed size""",
    tools_code=r'''"""Archive Manager Skill - create/extract ZIP, TAR, GZIP archives."""
import zipfile
import tarfile
import os
from pathlib import Path
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("archive-manager")

FORMAT_MAP = {
    "zip": "zip",
    "tar": "tar",
    "tar.gz": "tar.gz", "tgz": "tar.gz",
    "tar.bz2": "tar.bz2", "tbz2": "tar.bz2",
}


@tool_wrapper(required_params=["files", "output"])
def create_archive_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Create an archive from files."""
    status.set_callback(params.pop("_status_callback", None))
    files = params["files"]
    output = params["output"]
    fmt = FORMAT_MAP.get(params.get("format", "zip").lower(), "zip")

    existing = []
    for f in files:
        p = Path(f)
        if p.exists():
            existing.append(p)
        else:
            return tool_error(f"File not found: {f}")

    if not existing:
        return tool_error("No valid files to archive")

    try:
        if fmt == "zip":
            with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
                for fp in existing:
                    zf.write(fp, fp.name)
        else:
            mode_map = {"tar": "w", "tar.gz": "w:gz", "tar.bz2": "w:bz2"}
            with tarfile.open(output, mode_map[fmt]) as tf:
                for fp in existing:
                    tf.add(str(fp), arcname=fp.name)

        out_path = Path(output)
        return tool_response(
            output_path=str(out_path.resolve()),
            file_count=len(existing),
            size_bytes=out_path.stat().st_size,
            format=fmt,
        )
    except Exception as e:
        return tool_error(f"Archive creation failed: {e}")


@tool_wrapper(required_params=["archive_path"])
def extract_archive_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Extract an archive to a directory."""
    status.set_callback(params.pop("_status_callback", None))
    archive_path = Path(params["archive_path"])
    output_dir = params.get("output_dir", ".")

    if not archive_path.exists():
        return tool_error(f"Archive not found: {archive_path}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        extracted = []
        if zipfile.is_zipfile(str(archive_path)):
            with zipfile.ZipFile(str(archive_path), "r") as zf:
                zf.extractall(str(out))
                extracted = zf.namelist()
        elif tarfile.is_tarfile(str(archive_path)):
            with tarfile.open(str(archive_path), "r:*") as tf:
                tf.extractall(str(out))
                extracted = tf.getnames()
        else:
            return tool_error("Unsupported archive format")

        return tool_response(
            extracted_files=extracted,
            file_count=len(extracted),
            output_dir=str(out.resolve()),
        )
    except Exception as e:
        return tool_error(f"Extraction failed: {e}")


@tool_wrapper(required_params=["archive_path"])
def list_archive_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """List contents of an archive."""
    status.set_callback(params.pop("_status_callback", None))
    archive_path = Path(params["archive_path"])

    if not archive_path.exists():
        return tool_error(f"Archive not found: {archive_path}")

    try:
        files = []
        total_size = 0
        if zipfile.is_zipfile(str(archive_path)):
            with zipfile.ZipFile(str(archive_path), "r") as zf:
                for info in zf.infolist():
                    files.append({"name": info.filename, "size": info.file_size,
                                  "compressed": info.compress_size})
                    total_size += info.file_size
        elif tarfile.is_tarfile(str(archive_path)):
            with tarfile.open(str(archive_path), "r:*") as tf:
                for member in tf.getmembers():
                    files.append({"name": member.name, "size": member.size,
                                  "is_dir": member.isdir()})
                    total_size += member.size
        else:
            return tool_error("Unsupported archive format")

        return tool_response(files=files, file_count=len(files),
                             total_size=total_size,
                             archive_size=archive_path.stat().st_size)
    except Exception as e:
        return tool_error(f"Failed to list archive: {e}")


__all__ = ["create_archive_tool", "extract_archive_tool", "list_archive_tool"]
''',
)

# ── 52. epub-builder ──────────────────────────────────────────────
create_skill(
    name="epub-builder",
    frontmatter_name="building-epub-books",
    description="Build EPUB e-books from text or markdown content using stdlib zipfile. Use when the user wants to create epub, build ebook, convert text to epub.",
    category="document-creation",
    capabilities=["generate", "document"],
    triggers=["epub", "ebook", "e-book", "create epub", "build ebook"],
    eval_tool="build_epub_tool",
    eval_input={
        "title": "My Book",
        "author": "Author Name",
        "chapters": [{"title": "Chapter 1", "content": "This is the first chapter."}],
    },
    tool_docs="""### build_epub_tool
Build an EPUB e-book from chapters.

**Parameters:**
- `title` (str, required): Book title
- `author` (str, required): Author name
- `chapters` (list, required): List of {title, content} dicts
- `output_path` (str, optional): Output file path (default: title.epub)
- `language` (str, optional): Language code (default: en)
- `description` (str, optional): Book description

**Returns:**
- `success` (bool)
- `output_path` (str): Path to generated EPUB
- `chapter_count` (int): Number of chapters""",
    tools_code='"""EPUB Builder Skill - build EPUB ebooks from text/markdown."""\n'
    "import zipfile\n"
    "import uuid\n"
    "import re as _re\n"
    "from pathlib import Path\n"
    "from datetime import datetime, timezone\n"
    "from typing import Dict, Any, List\n"
    "from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper\n"
    "from Jotty.core.infrastructure.utils.skill_status import SkillStatus\n"
    "\n"
    'status = SkillStatus("epub-builder")\n'
    "\n"
    "_XHTML_TPL = (\n"
    '    \'<?xml version="1.0" encoding="UTF-8"?>\\n\'\n'
    "    '<!DOCTYPE html>\\n'\n"
    "    '<html xmlns=\"http://www.w3.org/1999/xhtml\">\\n'\n"
    "    '<head><title>{title}</title>\\n'\n"
    "    '<style>body{{font-family:serif;line-height:1.6;margin:1em;}}h1{{color:#333;}}</style>\\n'\n"
    "    '</head>\\n'\n"
    "    '<body><h1>{title}</h1>{body}</body></html>'\n"
    ")\n"
    "\n"
    "_OPF_TPL = (\n"
    '    \'<?xml version="1.0" encoding="UTF-8"?>\\n\'\n'
    '    \'<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">\\n\'\n'
    "    '<metadata xmlns:dc=\"http://purl.org/dc/elements/1.1/\">\\n'\n"
    "    '<dc:identifier id=\"bookid\">urn:uuid:{book_id}</dc:identifier>\\n'\n"
    "    '<dc:title>{title}</dc:title>\\n'\n"
    "    '<dc:creator>{author}</dc:creator>\\n'\n"
    "    '<dc:language>{language}</dc:language>\\n'\n"
    "    '<dc:description>{description}</dc:description>\\n'\n"
    "    '<meta property=\"dcterms:modified\">{now}</meta>\\n'\n"
    "    '</metadata>\\n'\n"
    "    '<manifest>\\n'\n"
    '    \'<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>\\n\'\n'
    "    '{manifest}\\n'\n"
    "    '</manifest>\\n'\n"
    "    '<spine>{spine}</spine>\\n'\n"
    "    '</package>'\n"
    ")\n"
    "\n"
    "_NAV_TPL = (\n"
    '    \'<?xml version="1.0" encoding="UTF-8"?>\\n\'\n'
    "    '<!DOCTYPE html>\\n'\n"
    '    \'<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">\\n\'\n'
    "    '<head><title>Table of Contents</title></head>\\n'\n"
    "    '<body><nav epub:type=\"toc\"><h1>Table of Contents</h1>\\n'\n"
    "    '<ol>{toc}</ol></nav></body></html>'\n"
    ")\n"
    "\n"
    "\n"
    "def _make_xhtml(title: str, body: str) -> str:\n"
    "    return _XHTML_TPL.format(title=title, body=body)\n"
    "\n"
    "\n"
    "def _text_to_html(text: str) -> str:\n"
    '    """Convert plain text to basic HTML paragraphs."""\n'
    '    paragraphs = text.strip().split("\\n\\n")\n'
    "    html_parts = []\n"
    "    for p in paragraphs:\n"
    "        p = p.strip()\n"
    "        if not p:\n"
    "            continue\n"
    '        m = _re.match(r"^(#{1,6})\\s+(.+)$", p)\n'
    "        if m:\n"
    "            level = len(m.group(1))\n"
    '            html_parts.append("<h{0}>{1}</h{0}>".format(level, m.group(2)))\n'
    "        else:\n"
    '            lines = p.replace("\\n", "<br/>")\n'
    '            html_parts.append("<p>{}</p>".format(lines))\n'
    '    return "\\n".join(html_parts)\n'
    "\n"
    "\n"
    '@tool_wrapper(required_params=["title", "author", "chapters"])\n'
    "def build_epub_tool(params: Dict[str, Any]) -> Dict[str, Any]:\n"
    '    """Build an EPUB e-book from chapters."""\n'
    '    status.set_callback(params.pop("_status_callback", None))\n'
    '    title = params["title"]\n'
    '    author = params["author"]\n'
    '    chapters = params["chapters"]\n'
    '    language = params.get("language", "en")\n'
    '    description = params.get("description", "")\n'
    "    book_id = str(uuid.uuid4())\n"
    "\n"
    "    if not chapters or not isinstance(chapters, list):\n"
    '        return tool_error("chapters must be a non-empty list of {title, content} dicts")\n'
    "\n"
    '    slug = title.lower().replace(" ", "_")[:50]\n'
    '    output_path = params.get("output_path", slug + ".epub")\n'
    '    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")\n'
    "\n"
    "    try:\n"
    '        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as epub:\n'
    '            epub.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)\n'
    "\n"
    '            epub.writestr("META-INF/container.xml",\n'
    '                \'<?xml version="1.0" encoding="UTF-8"?>\'\n'
    '                \'<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">\'\n'
    "                '<rootfiles><rootfile full-path=\"OEBPS/content.opf\" '\n"
    "                'media-type=\"application/oebps-package+xml\"/></rootfiles></container>')\n"
    "\n"
    "            manifest_items = []\n"
    "            spine_items = []\n"
    "            toc_items = []\n"
    "\n"
    "            for i, ch in enumerate(chapters):\n"
    '                ch_title = ch.get("title", "Chapter {}".format(i + 1))\n'
    '                ch_content = ch.get("content", "")\n'
    "                ch_html = _text_to_html(ch_content)\n"
    '                ch_id = "chapter{}".format(i + 1)\n'
    '                ch_file = ch_id + ".xhtml"\n'
    "\n"
    '                epub.writestr("OEBPS/" + ch_file, _make_xhtml(ch_title, ch_html))\n'
    '                manifest_items.append(\'<item id="{}" href="{}" media-type="application/xhtml+xml"/>\'.format(ch_id, ch_file))\n'
    "                spine_items.append('<itemref idref=\"{}\"/>'.format(ch_id))\n"
    "                toc_items.append('<li><a href=\"{}\">{}</a></li>'.format(ch_file, ch_title))\n"
    "\n"
    "            opf = _OPF_TPL.format(\n"
    "                book_id=book_id, title=title, author=author,\n"
    "                language=language, description=description, now=now,\n"
    '                manifest="\\n".join(manifest_items),\n'
    '                spine="\\n".join(spine_items),\n'
    "            )\n"
    '            epub.writestr("OEBPS/content.opf", opf)\n'
    "\n"
    '            nav = _NAV_TPL.format(toc="\\n".join(toc_items))\n'
    '            epub.writestr("OEBPS/nav.xhtml", nav)\n'
    "\n"
    "        out = Path(output_path)\n"
    "        return tool_response(\n"
    "            output_path=str(out.resolve()),\n"
    "            chapter_count=len(chapters),\n"
    "            size_bytes=out.stat().st_size,\n"
    "            book_id=book_id,\n"
    "        )\n"
    "    except Exception as e:\n"
    '        return tool_error("EPUB creation failed: {}".format(e))\n'
    "\n"
    "\n"
    '__all__ = ["build_epub_tool"]\n',
)

# ── 53. invoice-generator ─────────────────────────────────────────
create_skill(
    name="invoice-generator",
    frontmatter_name="generating-invoices",
    description="Generate invoice data as structured JSON with line items, taxes, and totals. Use when the user wants to create invoice, generate bill, invoice template.",
    category="workflow-automation",
    capabilities=["generate"],
    triggers=["invoice", "bill", "receipt", "generate invoice", "create invoice"],
    eval_tool="generate_invoice_tool",
    eval_input={
        "client_name": "Acme Corp",
        "items": [{"description": "Consulting", "quantity": 10, "rate": 150}],
    },
    tool_docs="""### generate_invoice_tool
Generate a structured invoice.

**Parameters:**
- `client_name` (str, required): Client/company name
- `items` (list, required): Line items [{description, quantity, rate}]
- `invoice_number` (str, optional): Invoice number (auto-generated if omitted)
- `tax_rate` (float, optional): Tax rate percentage (default: 0)
- `currency` (str, optional): Currency code (default: USD)
- `due_days` (int, optional): Payment due in days (default: 30)
- `notes` (str, optional): Additional notes

**Returns:**
- `success` (bool)
- `invoice` (dict): Complete invoice data with totals""",
    tools_code=r'''"""Invoice Generator Skill - generate structured invoice data."""
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("invoice-generator")

CURRENCY_SYMBOLS = {
    "USD": "$", "EUR": "\u20ac", "GBP": "\u00a3", "JPY": "\u00a5",
    "CAD": "CA$", "AUD": "A$", "CHF": "CHF", "INR": "\u20b9",
}


@tool_wrapper(required_params=["client_name", "items"])
def generate_invoice_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a structured invoice with line items and totals."""
    status.set_callback(params.pop("_status_callback", None))
    client = params["client_name"]
    items = params["items"]
    tax_rate = float(params.get("tax_rate", 0))
    currency = params.get("currency", "USD").upper()
    due_days = int(params.get("due_days", 30))
    notes = params.get("notes", "")

    if not items or not isinstance(items, list):
        return tool_error("items must be a non-empty list of {description, quantity, rate}")

    inv_number = params.get("invoice_number", f"INV-{uuid.uuid4().hex[:8].upper()}")
    now = datetime.now(timezone.utc)
    due_date = now + timedelta(days=due_days)
    symbol = CURRENCY_SYMBOLS.get(currency, currency + " ")

    line_items = []
    subtotal = 0.0
    for i, item in enumerate(items):
        desc = item.get("description", f"Item {i + 1}")
        qty = float(item.get("quantity", 1))
        rate = float(item.get("rate", 0))
        amount = round(qty * rate, 2)
        subtotal += amount
        line_items.append({
            "line": i + 1,
            "description": desc,
            "quantity": qty,
            "rate": rate,
            "amount": amount,
            "formatted_amount": f"{symbol}{amount:,.2f}",
        })

    subtotal = round(subtotal, 2)
    tax_amount = round(subtotal * tax_rate / 100, 2) if tax_rate else 0.0
    total = round(subtotal + tax_amount, 2)

    invoice = {
        "invoice_number": inv_number,
        "date": now.strftime("%Y-%m-%d"),
        "due_date": due_date.strftime("%Y-%m-%d"),
        "client_name": client,
        "currency": currency,
        "line_items": line_items,
        "subtotal": subtotal,
        "tax_rate": tax_rate,
        "tax_amount": tax_amount,
        "total": total,
        "formatted_subtotal": f"{symbol}{subtotal:,.2f}",
        "formatted_tax": f"{symbol}{tax_amount:,.2f}",
        "formatted_total": f"{symbol}{total:,.2f}",
        "notes": notes,
        "status": "pending",
    }

    return tool_response(invoice=invoice)


__all__ = ["generate_invoice_tool"]
''',
)

# ── 54. expense-tracker ───────────────────────────────────────────
create_skill(
    name="expense-tracker",
    frontmatter_name="tracking-expenses",
    description="Track expenses with categorization, budget limits, and summary reports. Uses JSON file storage. Use when the user wants to track expense, log spending, budget tracker.",
    category="workflow-automation",
    capabilities=["analyze"],
    triggers=["expense", "spending", "budget", "track expense", "log expense", "expense report"],
    eval_tool="add_expense_tool",
    eval_input={"amount": 45.99, "category": "food", "description": "Lunch meeting"},
    tool_docs="""### add_expense_tool
Add an expense entry.

**Parameters:**
- `amount` (float, required): Expense amount
- `category` (str, required): Category (food, transport, utilities, entertainment, shopping, health, other)
- `description` (str, optional): Description of expense
- `date` (str, optional): Date YYYY-MM-DD (default: today)
- `storage_file` (str, optional): JSON storage path (default: expenses.json)

**Returns:**
- `success` (bool)
- `expense` (dict): Added expense record
- `running_total` (float): Total expenses

### expense_summary_tool
Get expense summary and breakdown.

**Parameters:**
- `storage_file` (str, optional): JSON storage path
- `month` (str, optional): Filter by month YYYY-MM
- `category` (str, optional): Filter by category

**Returns:**
- `success` (bool)
- `total` (float): Total expenses
- `by_category` (dict): Breakdown by category
- `count` (int): Number of expenses""",
    tools_code=r'''"""Expense Tracker Skill - track expenses with JSON storage."""
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("expense-tracker")

CATEGORIES = {"food", "transport", "utilities", "entertainment", "shopping",
              "health", "education", "housing", "insurance", "savings", "other"}
DEFAULT_FILE = "expenses.json"


def _load(path: str) -> List[dict]:
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def _save(path: str, data: List[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


@tool_wrapper(required_params=["amount", "category"])
def add_expense_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Add an expense entry."""
    status.set_callback(params.pop("_status_callback", None))
    amount = round(float(params["amount"]), 2)
    category = params["category"].lower()
    description = params.get("description", "")
    date_str = params.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    storage = params.get("storage_file", DEFAULT_FILE)

    if amount <= 0:
        return tool_error("Amount must be positive")
    if category not in CATEGORIES:
        return tool_error(f"Invalid category. Use: {sorted(CATEGORIES)}")

    expenses = _load(storage)
    entry = {
        "id": uuid.uuid4().hex[:8],
        "amount": amount,
        "category": category,
        "description": description,
        "date": date_str,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    expenses.append(entry)
    _save(storage, expenses)

    running_total = round(sum(e["amount"] for e in expenses), 2)

    return tool_response(expense=entry, running_total=running_total,
                         total_entries=len(expenses))


@tool_wrapper()
def expense_summary_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get expense summary with category breakdown."""
    status.set_callback(params.pop("_status_callback", None))
    storage = params.get("storage_file", DEFAULT_FILE)
    month_filter = params.get("month", "")
    cat_filter = params.get("category", "").lower()

    expenses = _load(storage)

    if month_filter:
        expenses = [e for e in expenses if e.get("date", "").startswith(month_filter)]
    if cat_filter:
        expenses = [e for e in expenses if e.get("category") == cat_filter]

    total = round(sum(e["amount"] for e in expenses), 2)
    by_category = {}
    for e in expenses:
        cat = e.get("category", "other")
        if cat not in by_category:
            by_category[cat] = {"total": 0, "count": 0}
        by_category[cat]["total"] = round(by_category[cat]["total"] + e["amount"], 2)
        by_category[cat]["count"] += 1

    # Sort by total descending
    by_category = dict(sorted(by_category.items(), key=lambda x: x[1]["total"], reverse=True))

    return tool_response(
        total=total, count=len(expenses),
        by_category=by_category,
        filters={"month": month_filter, "category": cat_filter},
    )


__all__ = ["add_expense_tool", "expense_summary_tool"]
''',
)

# ── 55. loan-amortization-calculator ──────────────────────────────
create_skill(
    name="loan-amortization-calculator",
    frontmatter_name="calculating-loan-amortization",
    description="Generate loan amortization schedules with monthly payments, interest, and principal breakdown. Use when the user wants to calculate loan, amortization schedule, mortgage payment.",
    category="data-analysis",
    capabilities=["analyze"],
    triggers=["loan", "amortization", "mortgage", "monthly payment", "loan calculator", "interest"],
    eval_tool="amortization_schedule_tool",
    eval_input={"principal": 250000, "annual_rate": 6.5, "years": 30},
    tool_docs="""### amortization_schedule_tool
Generate a loan amortization schedule.

**Parameters:**
- `principal` (float, required): Loan principal amount
- `annual_rate` (float, required): Annual interest rate (percentage)
- `years` (int, required): Loan term in years
- `extra_payment` (float, optional): Extra monthly payment (default: 0)

**Returns:**
- `success` (bool)
- `monthly_payment` (float): Monthly payment amount
- `total_interest` (float): Total interest paid
- `total_paid` (float): Total amount paid
- `schedule` (list): Monthly breakdown (first 12 + last month)
- `payoff_months` (int): Actual months to payoff""",
    tools_code=r'''"""Loan Amortization Calculator Skill."""
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("loan-amortization-calculator")


@tool_wrapper(required_params=["principal", "annual_rate", "years"])
def amortization_schedule_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a loan amortization schedule."""
    status.set_callback(params.pop("_status_callback", None))
    principal = float(params["principal"])
    annual_rate = float(params["annual_rate"])
    years = int(params["years"])
    extra = float(params.get("extra_payment", 0))

    if principal <= 0:
        return tool_error("Principal must be positive")
    if annual_rate < 0:
        return tool_error("Interest rate cannot be negative")
    if years <= 0:
        return tool_error("Term must be positive")

    monthly_rate = annual_rate / 100 / 12
    num_payments = years * 12

    if monthly_rate == 0:
        monthly_payment = principal / num_payments
    else:
        monthly_payment = principal * (monthly_rate * (1 + monthly_rate) ** num_payments) / \
                          ((1 + monthly_rate) ** num_payments - 1)

    monthly_payment = round(monthly_payment, 2)

    # Generate schedule
    balance = principal
    total_interest = 0.0
    total_paid = 0.0
    schedule = []

    for month in range(1, num_payments + 1):
        interest_payment = round(balance * monthly_rate, 2)
        principal_payment = round(monthly_payment - interest_payment + extra, 2)

        if principal_payment > balance:
            principal_payment = balance
            interest_payment = round(balance * monthly_rate, 2)

        balance = round(balance - principal_payment, 2)
        total_interest += interest_payment
        total_paid += interest_payment + principal_payment

        entry = {
            "month": month,
            "payment": round(interest_payment + principal_payment, 2),
            "principal": principal_payment,
            "interest": interest_payment,
            "balance": max(0, balance),
        }
        schedule.append(entry)

        if balance <= 0:
            break

    # Return first 12 months + last month for brevity
    summary_schedule = schedule[:12]
    if len(schedule) > 12:
        summary_schedule.append(schedule[-1])

    return tool_response(
        monthly_payment=monthly_payment,
        total_interest=round(total_interest, 2),
        total_paid=round(total_paid, 2),
        principal=principal,
        annual_rate=annual_rate,
        term_years=years,
        extra_payment=extra,
        payoff_months=len(schedule),
        schedule=summary_schedule,
        interest_savings=round((monthly_payment * num_payments - total_paid), 2) if extra else 0,
    )


__all__ = ["amortization_schedule_tool"]
''',
)

# ── 56. tax-calculator ────────────────────────────────────────────
create_skill(
    name="tax-calculator",
    frontmatter_name="calculating-taxes",
    description="Calculate US federal income tax using current brackets, standard deduction, and effective rates. Use when the user wants to calculate tax, income tax, tax brackets, federal tax.",
    category="data-analysis",
    capabilities=["analyze"],
    triggers=[
        "tax",
        "income tax",
        "tax bracket",
        "federal tax",
        "tax calculator",
        "effective rate",
    ],
    eval_tool="calculate_federal_tax_tool",
    eval_input={"income": 85000, "filing_status": "single"},
    tool_docs="""### calculate_federal_tax_tool
Calculate US federal income tax.

**Parameters:**
- `income` (float, required): Gross annual income
- `filing_status` (str, optional): single, married_joint, married_separate, head_of_household (default: single)
- `deductions` (float, optional): Itemized deductions (uses standard if less)
- `year` (int, optional): Tax year (default: 2024)

**Returns:**
- `success` (bool)
- `tax_owed` (float): Total federal tax
- `effective_rate` (float): Effective tax rate percentage
- `marginal_rate` (float): Marginal tax bracket
- `breakdown` (list): Tax by bracket""",
    tools_code=r'''"""Tax Calculator Skill - US federal income tax brackets."""
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("tax-calculator")

# 2024 US Federal Tax Brackets
BRACKETS_2024 = {
    "single": [
        (11600, 0.10), (47150, 0.12), (100525, 0.22), (191950, 0.24),
        (243725, 0.32), (609350, 0.35), (float("inf"), 0.37),
    ],
    "married_joint": [
        (23200, 0.10), (94300, 0.12), (201050, 0.22), (383900, 0.24),
        (487450, 0.32), (731200, 0.35), (float("inf"), 0.37),
    ],
    "married_separate": [
        (11600, 0.10), (47150, 0.12), (100525, 0.22), (191950, 0.24),
        (243725, 0.32), (365600, 0.35), (float("inf"), 0.37),
    ],
    "head_of_household": [
        (16550, 0.10), (63100, 0.12), (100500, 0.22), (191950, 0.24),
        (243700, 0.32), (609350, 0.35), (float("inf"), 0.37),
    ],
}

STANDARD_DEDUCTIONS_2024 = {
    "single": 14600, "married_joint": 29200,
    "married_separate": 14600, "head_of_household": 21900,
}


def _calc_tax(taxable_income: float, brackets: list) -> tuple:
    tax = 0.0
    prev_limit = 0
    breakdown = []
    marginal_rate = 0.0

    for limit, rate in brackets:
        if taxable_income <= 0:
            break
        bracket_income = min(taxable_income, limit) - prev_limit
        if bracket_income <= 0:
            prev_limit = limit
            continue
        bracket_tax = round(bracket_income * rate, 2)
        tax += bracket_tax
        marginal_rate = rate
        breakdown.append({
            "bracket": f"{int(rate * 100)}%",
            "income_in_bracket": round(bracket_income, 2),
            "tax": bracket_tax,
            "range": f"${prev_limit:,.0f} - ${limit:,.0f}" if limit != float("inf") else f"${prev_limit:,.0f}+",
        })
        prev_limit = limit
        if taxable_income <= limit:
            break

    return round(tax, 2), marginal_rate, breakdown


@tool_wrapper(required_params=["income"])
def calculate_federal_tax_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate US federal income tax."""
    status.set_callback(params.pop("_status_callback", None))
    income = float(params["income"])
    filing = params.get("filing_status", "single").lower().replace(" ", "_")
    deductions = float(params.get("deductions", 0))

    if income < 0:
        return tool_error("Income cannot be negative")
    if filing not in BRACKETS_2024:
        return tool_error(f"Invalid filing status. Use: {list(BRACKETS_2024.keys())}")

    std_deduction = STANDARD_DEDUCTIONS_2024[filing]
    actual_deduction = max(deductions, std_deduction)
    deduction_type = "itemized" if deductions > std_deduction else "standard"
    taxable_income = max(0, income - actual_deduction)

    tax, marginal_rate, breakdown = _calc_tax(taxable_income, BRACKETS_2024[filing])
    effective_rate = round((tax / income) * 100, 2) if income > 0 else 0.0

    return tool_response(
        gross_income=income,
        deduction=actual_deduction,
        deduction_type=deduction_type,
        taxable_income=taxable_income,
        tax_owed=tax,
        effective_rate=effective_rate,
        marginal_rate=round(marginal_rate * 100, 1),
        filing_status=filing,
        breakdown=breakdown,
        formatted_tax=f"${tax:,.2f}",
    )


__all__ = ["calculate_federal_tax_tool"]
''',
)

# ── 57. pomodoro-timer ────────────────────────────────────────────
create_skill(
    name="pomodoro-timer",
    frontmatter_name="tracking-pomodoro",
    description="Track Pomodoro work sessions, breaks, and productivity stats. Use when the user wants to pomodoro timer, focus timer, work session, productivity tracker.",
    category="workflow-automation",
    capabilities=["analyze"],
    triggers=["pomodoro", "focus timer", "work session", "break timer", "productivity"],
    eval_tool="start_pomodoro_tool",
    eval_input={"task": "Write documentation", "work_minutes": 25},
    tool_docs="""### start_pomodoro_tool
Start a Pomodoro session.

**Parameters:**
- `task` (str, required): Task description
- `work_minutes` (int, optional): Work duration (default: 25)
- `break_minutes` (int, optional): Break duration (default: 5)
- `long_break_minutes` (int, optional): Long break after 4 sessions (default: 15)

**Returns:**
- `success` (bool)
- `session` (dict): Session details with start time and durations

### pomodoro_stats_tool
Get Pomodoro productivity statistics.

**Parameters:**
- `storage_file` (str, optional): JSON storage path

**Returns:**
- `success` (bool)
- `total_sessions` (int): Total completed sessions
- `total_focus_minutes` (int): Total focus time
- `by_task` (dict): Sessions per task""",
    tools_code=r'''"""Pomodoro Timer Skill - track focus sessions and breaks."""
import json
import uuid
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("pomodoro-timer")
DEFAULT_FILE = "pomodoro_sessions.json"


def _load(path: str) -> List[dict]:
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def _save(path: str, data: List[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


@tool_wrapper(required_params=["task"])
def start_pomodoro_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Start a Pomodoro session."""
    status.set_callback(params.pop("_status_callback", None))
    task = params["task"]
    work_min = int(params.get("work_minutes", 25))
    break_min = int(params.get("break_minutes", 5))
    long_break = int(params.get("long_break_minutes", 15))
    storage = params.get("storage_file", DEFAULT_FILE)

    now = datetime.now(timezone.utc)
    sessions = _load(storage)

    # Count today's sessions for this task
    today = now.strftime("%Y-%m-%d")
    today_count = sum(1 for s in sessions if s.get("date") == today)
    session_number = today_count + 1
    is_long_break = session_number % 4 == 0
    actual_break = long_break if is_long_break else break_min

    session = {
        "id": uuid.uuid4().hex[:8],
        "task": task,
        "date": today,
        "started_at": now.isoformat(),
        "work_minutes": work_min,
        "break_minutes": actual_break,
        "work_ends_at": (now + timedelta(minutes=work_min)).isoformat(),
        "break_ends_at": (now + timedelta(minutes=work_min + actual_break)).isoformat(),
        "session_number": session_number,
        "is_long_break": is_long_break,
        "status": "active",
    }

    sessions.append(session)
    _save(storage, sessions)

    return tool_response(
        session=session,
        message=f"Pomodoro #{session_number} started! Focus for {work_min} min, then {'long' if is_long_break else 'short'} break ({actual_break} min).",
    )


@tool_wrapper()
def pomodoro_stats_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get Pomodoro productivity statistics."""
    status.set_callback(params.pop("_status_callback", None))
    storage = params.get("storage_file", DEFAULT_FILE)
    sessions = _load(storage)

    total_focus = sum(s.get("work_minutes", 25) for s in sessions)
    total_break = sum(s.get("break_minutes", 5) for s in sessions)

    by_task = {}
    for s in sessions:
        task = s.get("task", "Unknown")
        if task not in by_task:
            by_task[task] = {"sessions": 0, "focus_minutes": 0}
        by_task[task]["sessions"] += 1
        by_task[task]["focus_minutes"] += s.get("work_minutes", 25)

    by_date = {}
    for s in sessions:
        date = s.get("date", "unknown")
        by_date[date] = by_date.get(date, 0) + 1

    return tool_response(
        total_sessions=len(sessions),
        total_focus_minutes=total_focus,
        total_break_minutes=total_break,
        by_task=by_task,
        by_date=by_date,
        avg_sessions_per_day=round(len(sessions) / max(len(by_date), 1), 1),
    )


__all__ = ["start_pomodoro_tool", "pomodoro_stats_tool"]
''',
)

# ── 58. habit-tracker ─────────────────────────────────────────────
create_skill(
    name="habit-tracker",
    frontmatter_name="tracking-habits",
    description="Track daily habits with streaks, completion rates, and statistics. Uses JSON storage. Use when the user wants to track habit, habit streak, daily tracker.",
    category="workflow-automation",
    capabilities=["analyze"],
    triggers=["habit", "habit tracker", "streak", "daily habit", "routine", "track habit"],
    eval_tool="log_habit_tool",
    eval_input={"habit": "exercise", "completed": True},
    tool_docs="""### log_habit_tool
Log a habit completion for today.

**Parameters:**
- `habit` (str, required): Habit name
- `completed` (bool, optional): Whether completed (default: true)
- `date` (str, optional): Date YYYY-MM-DD (default: today)
- `notes` (str, optional): Optional notes
- `storage_file` (str, optional): JSON storage path

**Returns:**
- `success` (bool)
- `habit` (str): Habit name
- `current_streak` (int): Current consecutive day streak

### habit_stats_tool
Get habit statistics and streaks.

**Parameters:**
- `habit` (str, optional): Specific habit (default: all)
- `storage_file` (str, optional): JSON storage path

**Returns:**
- `success` (bool)
- `habits` (dict): Per-habit stats with streaks and completion rates""",
    tools_code=r'''"""Habit Tracker Skill - track habits with streaks."""
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("habit-tracker")
DEFAULT_FILE = "habits.json"


def _load(path: str) -> dict:
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def _save(path: str, data: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


def _calc_streak(dates: List[str]) -> int:
    if not dates:
        return 0
    sorted_dates = sorted(set(dates), reverse=True)
    streak = 1
    for i in range(len(sorted_dates) - 1):
        curr = datetime.strptime(sorted_dates[i], "%Y-%m-%d")
        prev = datetime.strptime(sorted_dates[i + 1], "%Y-%m-%d")
        if (curr - prev).days == 1:
            streak += 1
        else:
            break
    return streak


def _longest_streak(dates: List[str]) -> int:
    if not dates:
        return 0
    sorted_dates = sorted(set(dates))
    longest = 1
    current = 1
    for i in range(1, len(sorted_dates)):
        curr = datetime.strptime(sorted_dates[i], "%Y-%m-%d")
        prev = datetime.strptime(sorted_dates[i - 1], "%Y-%m-%d")
        if (curr - prev).days == 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


@tool_wrapper(required_params=["habit"])
def log_habit_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Log a habit completion."""
    status.set_callback(params.pop("_status_callback", None))
    habit = params["habit"].lower().strip()
    completed = params.get("completed", True)
    date = params.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    notes = params.get("notes", "")
    storage = params.get("storage_file", DEFAULT_FILE)

    data = _load(storage)
    if habit not in data:
        data[habit] = {"entries": [], "created": date}

    # Remove existing entry for same date
    data[habit]["entries"] = [e for e in data[habit]["entries"] if e.get("date") != date]
    data[habit]["entries"].append({
        "date": date,
        "completed": completed,
        "notes": notes,
    })

    _save(storage, data)

    completed_dates = [e["date"] for e in data[habit]["entries"] if e["completed"]]
    streak = _calc_streak(completed_dates)

    return tool_response(
        habit=habit, date=date, completed=completed,
        current_streak=streak,
        total_completions=len(completed_dates),
    )


@tool_wrapper()
def habit_stats_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get habit statistics and streaks."""
    status.set_callback(params.pop("_status_callback", None))
    storage = params.get("storage_file", DEFAULT_FILE)
    habit_filter = params.get("habit", "").lower().strip()

    data = _load(storage)
    if habit_filter and habit_filter in data:
        data = {habit_filter: data[habit_filter]}
    elif habit_filter and habit_filter not in data:
        return tool_error(f"Habit '{habit_filter}' not found. Available: {list(data.keys())}")

    habits_stats = {}
    for habit, info in data.items():
        entries = info.get("entries", [])
        completed_dates = [e["date"] for e in entries if e.get("completed")]
        total_entries = len(entries)
        total_completed = len(completed_dates)

        habits_stats[habit] = {
            "total_entries": total_entries,
            "total_completed": total_completed,
            "completion_rate": round(total_completed / max(total_entries, 1) * 100, 1),
            "current_streak": _calc_streak(completed_dates),
            "longest_streak": _longest_streak(completed_dates),
            "created": info.get("created", "unknown"),
        }

    return tool_response(habits=habits_stats, habit_count=len(habits_stats))


__all__ = ["log_habit_tool", "habit_stats_tool"]
''',
)

# ── 59. decision-matrix-builder ───────────────────────────────────
create_skill(
    name="decision-matrix-builder",
    frontmatter_name="building-decision-matrices",
    description="Build weighted decision matrices to compare options across multiple criteria. Use when the user wants to compare options, decision matrix, weighted scoring, pros cons.",
    category="workflow-automation",
    capabilities=["analyze"],
    triggers=[
        "decision matrix",
        "compare options",
        "weighted scoring",
        "decision analysis",
        "pros cons",
    ],
    eval_tool="build_decision_matrix_tool",
    eval_input={
        "options": ["Option A", "Option B"],
        "criteria": [{"name": "Cost", "weight": 5}, {"name": "Quality", "weight": 8}],
        "scores": {"Option A": {"Cost": 8, "Quality": 6}, "Option B": {"Cost": 5, "Quality": 9}},
    },
    tool_docs="""### build_decision_matrix_tool
Build a weighted decision matrix.

**Parameters:**
- `options` (list, required): List of option names
- `criteria` (list, required): List of {name, weight} dicts (weight 1-10)
- `scores` (dict, required): {option: {criterion: score}} (scores 1-10)

**Returns:**
- `success` (bool)
- `results` (list): Ranked options with weighted scores
- `matrix` (str): Text-formatted matrix
- `winner` (str): Best option""",
    tools_code=r'''"""Decision Matrix Builder Skill - weighted multi-criteria comparison."""
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("decision-matrix-builder")


@tool_wrapper(required_params=["options", "criteria", "scores"])
def build_decision_matrix_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Build a weighted decision matrix to compare options."""
    status.set_callback(params.pop("_status_callback", None))
    options = params["options"]
    criteria = params["criteria"]
    scores = params["scores"]

    if len(options) < 2:
        return tool_error("Need at least 2 options to compare")
    if not criteria:
        return tool_error("Need at least 1 criterion")

    # Validate
    crit_names = [c["name"] for c in criteria]
    total_weight = sum(c.get("weight", 1) for c in criteria)

    results = []
    for option in options:
        if option not in scores:
            return tool_error(f"Missing scores for option: {option}")
        weighted_total = 0
        raw_total = 0
        detail = {}
        for crit in criteria:
            name = crit["name"]
            weight = crit.get("weight", 1)
            score = scores[option].get(name, 0)
            if not (0 <= score <= 10):
                return tool_error(f"Score for {option}/{name} must be 0-10, got {score}")
            ws = round(score * weight, 2)
            weighted_total += ws
            raw_total += score
            detail[name] = {"score": score, "weight": weight, "weighted": ws}

        normalized = round(weighted_total / total_weight, 2) if total_weight else 0
        results.append({
            "option": option,
            "weighted_total": round(weighted_total, 2),
            "normalized_score": normalized,
            "raw_total": raw_total,
            "detail": detail,
        })

    # Sort by weighted total descending
    results.sort(key=lambda x: x["weighted_total"], reverse=True)
    winner = results[0]["option"]

    # Build text matrix
    col_width = max(len(o) for o in options) + 2
    crit_width = max(len(c["name"]) for c in criteria) + 2
    header = "Criterion".ljust(crit_width) + "Wt  " + "  ".join(o.center(col_width) for o in options)
    sep = "-" * len(header)
    rows = [header, sep]
    for crit in criteria:
        name = crit["name"]
        weight = crit.get("weight", 1)
        row = name.ljust(crit_width) + f"{weight:<4}"
        for option in options:
            s = scores[option].get(name, 0)
            ws = round(s * weight, 2)
            row += f"{s}({ws})".center(col_width) + "  "
        rows.append(row)
    rows.append(sep)
    total_row = "TOTAL".ljust(crit_width) + "    "
    for r in sorted(results, key=lambda x: options.index(x["option"])):
        total_row += f"{r['weighted_total']}".center(col_width) + "  "
    rows.append(total_row)
    matrix_text = "\n".join(rows)

    return tool_response(
        results=results, winner=winner,
        matrix=matrix_text,
        criteria_count=len(criteria),
        option_count=len(options),
    )


__all__ = ["build_decision_matrix_tool"]
''',
)

# ── 60. project-timeline-generator ────────────────────────────────
create_skill(
    name="project-timeline-generator",
    frontmatter_name="generating-project-timelines",
    description="Generate text-based Gantt charts and project timelines from task lists. Use when the user wants to project timeline, Gantt chart, schedule tasks, project plan.",
    category="workflow-automation",
    capabilities=["generate", "analyze"],
    triggers=[
        "timeline",
        "gantt chart",
        "project plan",
        "schedule",
        "project timeline",
        "milestones",
    ],
    eval_tool="generate_timeline_tool",
    eval_input={
        "tasks": [
            {"name": "Design", "duration": 5, "start_day": 1},
            {"name": "Development", "duration": 10, "start_day": 6},
            {"name": "Testing", "duration": 5, "start_day": 16},
        ]
    },
    tool_docs="""### generate_timeline_tool
Generate a text-based Gantt chart / project timeline.

**Parameters:**
- `tasks` (list, required): List of {name, duration, start_day, depends_on (optional)}
- `title` (str, optional): Project title
- `scale` (str, optional): day, week (default: day)

**Returns:**
- `success` (bool)
- `gantt_chart` (str): Text-based Gantt chart
- `total_duration` (int): Total project duration in days
- `critical_path` (list): Tasks on critical path""",
    tools_code=r'''"""Project Timeline Generator Skill - text-based Gantt charts."""
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("project-timeline-generator")


def _resolve_dependencies(tasks: List[dict]) -> List[dict]:
    """Resolve task dependencies and adjust start days."""
    task_map = {t["name"]: t for t in tasks}
    resolved = []

    for task in tasks:
        t = dict(task)
        dep = t.get("depends_on")
        if dep and dep in task_map:
            dep_task = task_map[dep]
            dep_end = dep_task.get("start_day", 1) + dep_task.get("duration", 1)
            t["start_day"] = max(t.get("start_day", dep_end), dep_end)
        resolved.append(t)

    return resolved


@tool_wrapper(required_params=["tasks"])
def generate_timeline_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a text-based Gantt chart."""
    status.set_callback(params.pop("_status_callback", None))
    tasks = params["tasks"]
    title = params.get("title", "Project Timeline")
    scale = params.get("scale", "day").lower()

    if not tasks:
        return tool_error("Need at least one task")

    tasks = _resolve_dependencies(tasks)

    # Calculate project bounds
    max_end = 0
    for t in tasks:
        start = t.get("start_day", 1)
        duration = t.get("duration", 1)
        end = start + duration - 1
        t["_end"] = end
        max_end = max(max_end, end)

    total_days = max_end
    name_width = max(len(t.get("name", "Task")) for t in tasks) + 2

    # Scale factor for chart width
    if scale == "week":
        chart_width = min((total_days // 7) + 1, 52)
        scale_factor = 7
    else:
        chart_width = min(total_days, 80)
        scale_factor = max(1, total_days // chart_width) if chart_width > 0 else 1

    # Build header
    lines = [f"  {title}", "  " + "=" * (name_width + chart_width + 10)]

    # Day/week markers
    marker_line = " " * (name_width + 2)
    for i in range(0, chart_width, 5):
        day_num = i * scale_factor + 1
        label = str(day_num)
        marker_line += label.ljust(5)
    lines.append(marker_line)
    lines.append(" " * (name_width + 2) + "|" * chart_width)

    # Task bars
    for t in tasks:
        name = t.get("name", "Task")
        start = t.get("start_day", 1)
        duration = t.get("duration", 1)

        bar_start = (start - 1) // scale_factor
        bar_len = max(1, duration // scale_factor)

        bar_start = min(bar_start, chart_width - 1)
        bar_len = min(bar_len, chart_width - bar_start)

        bar = " " * bar_start + "\u2588" * bar_len
        padding = chart_width - len(bar)
        if padding > 0:
            bar += " " * padding

        day_range = f"(d{start}-d{start + duration - 1})"
        lines.append(f"  {name.ljust(name_width)}{bar}  {day_range}")

    lines.append(" " * (name_width + 2) + "-" * chart_width)
    lines.append(f"  Total duration: {total_days} days")

    gantt_text = "\n".join(lines)

    # Find critical path (longest path)
    critical = sorted(tasks, key=lambda t: t.get("_end", 0), reverse=True)
    critical_path = [t["name"] for t in critical if t.get("_end") == max_end]

    task_summary = []
    for t in tasks:
        task_summary.append({
            "name": t.get("name"),
            "start_day": t.get("start_day"),
            "duration": t.get("duration"),
            "end_day": t.get("_end"),
            "depends_on": t.get("depends_on"),
        })

    return tool_response(
        gantt_chart=gantt_text,
        total_duration=total_days,
        task_count=len(tasks),
        tasks=task_summary,
        critical_path=critical_path,
        title=title,
    )


__all__ = ["generate_timeline_tool"]
''',
)

print(f"\nBatch 3 complete: 20 skills created.")
