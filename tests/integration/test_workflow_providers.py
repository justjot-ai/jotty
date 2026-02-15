"""
Test n8n and Activepieces as skill providers (localhost).

Run with: pytest tests/test_workflow_providers.py -v
With n8n on localhost:5678 you get real workflow list; without, providers init and return empty (OK).
"""

import os

import pytest


class TestWorkflowProvidersRegistration:
    """Providers must be registered and discoverable."""

    def test_registry_has_n8n_and_activepieces(self):
        from Jotty.core.capabilities.skills.providers import ProviderRegistry

        reg = ProviderRegistry()
        n8n = reg.get_provider("n8n")
        ap = reg.get_provider("activepieces")
        assert n8n is not None
        assert ap is not None
        assert n8n.name == "n8n"
        assert ap.name == "activepieces"

    def test_get_all_contributed_skills_returns_list(self):
        from Jotty.core.capabilities.skills.providers import ProviderRegistry

        reg = ProviderRegistry()
        skills = reg.get_all_contributed_skills()
        assert isinstance(skills, list)
        # Before init, n8n/activepieces caches are empty; jotty doesn't implement list_skills
        # so we may get 0 or more
        for s in skills:
            assert hasattr(s, "id")
            assert hasattr(s, "name")
            assert hasattr(s, "provider")


@pytest.mark.asyncio
class TestWorkflowProvidersLocalhost:
    """Test against localhost (n8n :5678, activepieces :8080). No server = empty list."""

    async def test_n8n_provider_initialize(self):
        from Jotty.core.capabilities.skills.providers import ProviderRegistry

        reg = ProviderRegistry()
        n8n = reg.get_provider("n8n")
        ok = await n8n.initialize()
        assert ok is True
        # With or without n8n running, init should succeed (is_available may be False)
        skills = n8n.list_skills()
        assert isinstance(skills, list)

    async def test_activepieces_provider_initialize(self):
        from Jotty.core.capabilities.skills.providers import ProviderRegistry

        reg = ProviderRegistry()
        ap = reg.get_provider("activepieces")
        ok = await ap.initialize()
        assert ok is True
        skills = ap.list_skills()
        assert isinstance(skills, list)

    async def test_n8n_execute_without_workflow_id_returns_error_result(self):
        from Jotty.core.capabilities.skills.providers import ProviderRegistry

        reg = ProviderRegistry()
        n8n = reg.get_provider("n8n")
        await n8n.initialize()
        result = await n8n.execute("", {})
        assert result.success is False
        assert "workflow_id" in result.error or "skill_id" in result.error.lower()

    async def test_contributed_skill_shape(self):
        from Jotty.core.capabilities.skills.providers import ContributedSkill, ProviderRegistry

        reg = ProviderRegistry()
        await reg.get_provider("n8n").initialize()
        await reg.get_provider("activepieces").initialize()
        all_skills = reg.get_all_contributed_skills()
        for s in all_skills:
            assert isinstance(s, ContributedSkill)
            assert s.id
            assert s.provider in ("n8n", "activepieces")


@pytest.mark.asyncio
class TestWorkflowProvidersRealServer:
    """
    Real server test (pmi.workflows). Run with:
      N8N_BASE_URL=http://localhost:5678 N8N_API_KEY=... python3 -m pytest ...
    Use SSH tunnel: ssh -f -N -L 5678:127.0.0.1:5678 pmi.workflows
    """

    async def test_n8n_real_server_reachable(self):
        import os

        base = os.getenv("N8N_BASE_URL")
        if not base or "localhost" not in base:
            pytest.skip("N8N_BASE_URL not set or not localhost (tunnel)")
        from Jotty.core.capabilities.skills.providers import ProviderRegistry

        reg = ProviderRegistry()
        n8n = reg.get_provider("n8n")
        await n8n.initialize()
        skills = n8n.list_skills()
        # Real server: either we get workflows (200) or 0 (401/connection)
        assert isinstance(skills, list)
        if skills:
            assert all(hasattr(s, "id") and s.id.startswith("n8n:workflow:") for s in skills)
