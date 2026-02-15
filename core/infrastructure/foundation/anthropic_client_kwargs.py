"""
Shared kwargs for Anthropic SDK client creation (CCR and direct API).

Use this wherever Jotty creates anthropic.Anthropic() or AsyncAnthropic()
so that a single env var (ANTHROPIC_BASE_URL) routes all traffic through
claude-code-router or any Anthropic-compatible proxy. Keys for downstream
providers are configured in CCR; Jotty only needs one base URL and one auth.
"""

import os
from typing import Optional, Dict, Any


def get_anthropic_client_kwargs(api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Build kwargs for anthropic.Anthropic(**kwargs) / AsyncAnthropic(**kwargs).

    - api_key: from param, or ANTHROPIC_API_KEY, or ANTHROPIC_AUTH_TOKEN (CCR).
    - base_url: from ANTHROPIC_BASE_URL when set (e.g. CCR at http://127.0.0.1:3456).

    Returns:
        Dict to pass as **kwargs to Anthropic() / AsyncAnthropic().
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    kwargs: Dict[str, Any] = {}
    if key:
        kwargs["api_key"] = key
    base = os.environ.get("ANTHROPIC_BASE_URL")
    if base:
        kwargs["base_url"] = base.rstrip("/")
    return kwargs
