"""Email Template Builder Skill - create HTML emails with inline CSS."""

from typing import Any, Dict

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

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
        subject=subject,
        body=body,
        brand_color=brand_color,
        footer=footer,
        preheader=preheader,
    )
    return tool_response(html=html, subject=subject, template=template_name, char_count=len(html))


__all__ = ["build_email_template_tool"]
