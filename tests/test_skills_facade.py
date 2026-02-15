"""
Tests for the Skills & Providers Subsystem Facade (Phase 2d).

Verifies provider listing, skill filtering, and accessor functions.
All tests run offline.
"""

import pytest


@pytest.mark.unit
class TestSkillsFacade:
    """Tests for skills facade accessor functions."""

    def test_get_registry_returns_unified_registry(self):
        from Jotty.core.capabilities.skills.facade import get_registry
        from Jotty.core.capabilities.registry import UnifiedRegistry
        registry = get_registry()
        assert isinstance(registry, UnifiedRegistry)

    def test_list_providers_returns_list(self):
        from Jotty.core.capabilities.skills.facade import list_providers
        providers = list_providers()
        assert isinstance(providers, list)
        assert len(providers) > 0

    def test_list_providers_has_known_providers(self):
        from Jotty.core.capabilities.skills.facade import list_providers
        providers = list_providers()
        names = [p["name"] for p in providers]
        assert "browser-use" in names
        assert "openhands" in names

    def test_list_providers_has_required_fields(self):
        from Jotty.core.capabilities.skills.facade import list_providers
        for p in list_providers():
            assert "name" in p
            assert "description" in p
            assert "installed" in p

    def test_get_provider_unknown_raises_value_error(self):
        from Jotty.core.capabilities.skills.facade import get_provider
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("nonexistent-provider")

    def test_list_skills_returns_list(self):
        from Jotty.core.capabilities.skills.facade import list_skills
        skills = list_skills()
        assert isinstance(skills, list)

    def test_list_components_returns_dict(self):
        from Jotty.core.capabilities.skills.facade import list_components
        components = list_components()
        assert isinstance(components, dict)
        assert len(components) > 0

    def test_list_components_has_key_classes(self):
        from Jotty.core.capabilities.skills.facade import list_components
        components = list_components()
        expected = [
            "UnifiedRegistry",
            "SkillsRegistry",
            "ProviderRegistry",
            "SkillProvider",
        ]
        for name in expected:
            assert name in components, f"Missing component: {name}"

    def test_list_components_values_are_strings(self):
        from Jotty.core.capabilities.skills.facade import list_components
        for name, desc in list_components().items():
            assert isinstance(desc, str)
            assert len(desc) > 0


@pytest.mark.unit
class TestSkillsFacadeFromInit:
    """Test facade functions are accessible from skills __init__."""

    def test_import_get_registry(self):
        from Jotty.core.capabilities.skills import get_registry
        assert callable(get_registry)

    def test_import_list_providers(self):
        from Jotty.core.capabilities.skills import list_providers
        assert callable(list_providers)

    def test_import_list_skills(self):
        from Jotty.core.capabilities.skills import list_skills
        assert callable(list_skills)
