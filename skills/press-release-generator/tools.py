"""Press Release Generator Skill - AP style press releases."""
from datetime import datetime
from typing import Dict, Any, List
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

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
