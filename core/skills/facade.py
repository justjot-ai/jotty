"""
Skills & Providers Subsystem Facade
=====================================

Clean, discoverable API for skills, providers, and tool management.
No new business logic — just imports + convenience accessors.

Usage:
    from Jotty.core.skills.facade import get_registry, list_providers

    registry = get_registry()
    skills = registry.list_skills()

    providers = list_providers()
"""

from typing import Optional, Dict, List, Any


def get_registry():
    """
    Return the UnifiedRegistry (single entry point for all capabilities).

    Returns:
        UnifiedRegistry instance (singleton).
    """
    from Jotty.core.registry import get_unified_registry
    return get_unified_registry()


def list_providers() -> List[Dict[str, Any]]:
    """
    List all skill providers with their installation status.

    Returns:
        List of dicts with 'name', 'description', and 'installed' keys.
    """
    provider_specs = [
        ("browser-use", "browser_use", "Web automation via browser-use library"),
        ("openhands", "openhands", "Terminal/code via OpenHands SDK"),
        ("agent-s", "agent_s", "GUI/computer control via Agent-S"),
        ("open-interpreter", "interpreter", "Local code execution"),
        ("streamlit", "streamlit", "App building (open source)"),
        ("morph", "morph", "Cloud app building"),
        ("n8n", None, "Workflow automation (API-based)"),
        ("activepieces", None, "Flow automation (API-based)"),
    ]
    result = []
    for name, module, desc in provider_specs:
        installed = None
        if module:
            try:
                __import__(module)
                installed = True
            except ImportError:
                installed = False
        result.append({"name": name, "description": desc, "installed": installed})
    return result


def get_provider(name: str):
    """
    Get a specific skill provider by name.

    Args:
        name: Provider name (e.g. "browser-use", "openhands").

    Returns:
        Provider instance.

    Raises:
        ImportError: If the provider's dependencies are not installed.
        ValueError: If the provider name is unknown.
    """
    providers = {
        "browser-use": ("Jotty.core.skills.providers.browser_use_provider", "BrowserUseProvider"),
        "openhands": ("Jotty.core.skills.providers.openhands_provider", "OpenHandsProvider"),
        "agent-s": ("Jotty.core.skills.providers.agent_s_provider", "AgentSProvider"),
        "open-interpreter": ("Jotty.core.skills.providers.open_interpreter_provider", "OpenInterpreterProvider"),
        "streamlit": ("Jotty.core.skills.providers.streamlit_provider", "StreamlitProvider"),
        "morph": ("Jotty.core.skills.providers.morph_provider", "MorphProvider"),
    }
    if name not in providers:
        raise ValueError(
            f"Unknown provider: {name!r}. "
            f"Available: {list(providers.keys())}"
        )
    module_path, class_name = providers[name]
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls()


def list_skills(category: Optional[str] = None) -> List[str]:
    """
    List registered skill names, optionally filtered by category.

    Args:
        category: Optional category filter (e.g. "browser", "code_exec").

    Returns:
        List of skill names.
    """
    try:
        from Jotty.core.registry import get_unified_registry
        registry = get_unified_registry()
        all_skills = registry.list_skills()
        if category is None:
            return all_skills
        # Filter by category if skills have category metadata
        filtered = []
        for skill_name in all_skills:
            try:
                skill = registry.get_skill(skill_name)
                skill_cat = getattr(skill, 'category', None) or ''
                if category.lower() in str(skill_cat).lower():
                    filtered.append(skill_name)
            except Exception:
                continue
        return filtered
    except Exception:
        return []


def list_components() -> Dict[str, str]:
    """
    List all skills/provider subsystem components with descriptions.

    Returns:
        Dict mapping component name to description.
    """
    return {
        "UnifiedRegistry": "Single entry point for all capabilities (skills + UI + tools)",
        "SkillsRegistry": "Backend skill definitions, tool metadata, lazy loading",
        "UIRegistry": "UI component registry (16 renderers)",
        "ProviderRegistry": "Pluggable provider selection with RL-based routing",
        "SkillProvider": "Base class for custom skill providers",
        "SkillGenerator": "DSPy-powered skill generation from natural language",
        "SkillDependencyManager": "Manages inter-skill dependencies",
        "ToolCollection": "Groups related tools for agent consumption",
        "ToolValidator": "Validates tool schemas and attributes",
    }
