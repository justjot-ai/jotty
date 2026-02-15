"""Blog Post Writer Skill - generate blog outlines with SEO."""

from typing import Any, Dict, List

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

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
        sections.append(
            {
                "order": i + 1,
                "heading": f"## {heading}" if i > 0 else f"## Introduction",
                "guidance": guidance,
                "word_target": words_per_section,
                "key_points": [
                    f"Point related to {keywords[0]}" if keywords else "Main argument",
                    "Supporting evidence or example",
                    "Transition to next section",
                ],
            }
        )

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
