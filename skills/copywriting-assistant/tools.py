"""Copywriting Assistant Skill - marketing copy frameworks."""

from typing import Any, Dict

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

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
        product=product,
        audience=audience,
        tone=tone,
    )


__all__ = ["generate_copy_tool"]
