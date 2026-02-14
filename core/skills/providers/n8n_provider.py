"""
n8n Skill Provider
==================

Exposes n8n workflows as skills (one skill per workflow). DRY: connector logic
stays in n8n; we only list workflows and run by id. KISS: one provider, same
env as CLI (N8N_BASE_URL, N8N_API_KEY).
"""

import os
import time
import logging
from typing import Any, Dict, List

from .base import (
    SkillProvider,
    SkillCategory,
    ProviderCapability,
    ProviderResult,
    ContributedSkill,
)

logger = logging.getLogger(__name__)

DEFAULT_N8N_URL = "http://localhost:5678"


def _n8n_headers() -> Dict[str, str]:
    key = os.getenv("N8N_API_KEY")
    return {"X-N8N-API-KEY": key} if key else {}


async def _n8n_get_workflows(base_url: str) -> List[Dict[str, Any]]:
    """Fetch workflow list from n8n API. Reusable (DRY with CLI)."""
    try:
        import aiohttp
    except ImportError:
        logger.debug("aiohttp not installed; n8n provider list_skills will be empty")
        return []
    url = f"{base_url.rstrip('/')}/api/v1/workflows"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=_n8n_headers(), timeout=10) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                return data.get("data", [])
    except Exception as e:
        logger.debug("n8n list workflows failed: %s", e)
        return []


async def _n8n_run_workflow(base_url: str, workflow_id: str, payload: Dict[str, Any] = None) -> Dict[str, Any]:
    """Run workflow by id. Returns n8n response or raises."""
    try:
        import aiohttp
    except ImportError:
        raise RuntimeError("aiohttp required for n8n provider")
    url = f"{base_url.rstrip('/')}/api/v1/workflows/{workflow_id}/run"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=_n8n_headers(), json=payload or {}, timeout=60) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"n8n run failed {resp.status}: {text}")
            return await resp.json()


class N8nProvider(SkillProvider):
    """
    Skill provider that exposes n8n workflows as skills.
    One skill per workflow; execute runs that workflow via n8n API.
    """

    name = "n8n"
    version = "1.0.0"
    description = "n8n workflows as skills (one per workflow)"

    def __init__(self, config: Dict[str, Any] = None) -> None:
        super().__init__(config or {})
        self._base_url = self.config.get("base_url") or os.getenv("N8N_BASE_URL", DEFAULT_N8N_URL)
        self._skills_cache: List[ContributedSkill] = []
        self.capabilities = [
            ProviderCapability(
                category=SkillCategory.SCHEDULING,
                actions=["run_workflow", "trigger", "schedule"],
                estimated_latency_ms=2000,
            )
        ]

    def get_categories(self) -> List[SkillCategory]:
        return [SkillCategory.SCHEDULING]

    def list_skills(self) -> List[ContributedSkill]:
        """Sync wrapper: returns cached list; refresh via list_skills_async or first execute."""
        return list(self._skills_cache)

    async def _fetch_skills(self) -> List[ContributedSkill]:
        workflows = await _n8n_get_workflows(self._base_url)
        skills = []
        for w in workflows:
            wf_id = w.get("id")
            if wf_id is None:
                continue
            name = w.get("name", "Unnamed")
            skills.append(
                ContributedSkill(
                    id=f"n8n:workflow:{wf_id}",
                    name=name,
                    description=w.get("description") or f"n8n workflow: {name}",
                    provider=self.name,
                    metadata={"workflow_id": str(wf_id)},
                )
            )
        self._skills_cache = skills
        return skills

    async def initialize(self) -> bool:
        try:
            skills = await self._fetch_skills()
            self.is_initialized = True
            self.is_available = True
            logger.info(" n8n provider initialized (%d workflows as skills)", len(skills))
            return True
        except Exception as e:
            logger.warning("n8n provider init failed (will retry on use): %s", e)
            self.is_initialized = True
            self.is_available = False
            return True

    async def execute(self, task: str, context: Dict[str, Any] = None) -> ProviderResult:
        start = time.time()
        context = context or {}
        workflow_id = context.get("workflow_id") or context.get("skill_id")
        if workflow_id and isinstance(workflow_id, str) and workflow_id.startswith("n8n:workflow:"):
            workflow_id = workflow_id.replace("n8n:workflow:", "")
        if not workflow_id:
            return ProviderResult(
                success=False,
                output=None,
                error="Missing workflow_id or skill_id in context (e.g. n8n:workflow:ID)",
                execution_time=time.time() - start,
                provider_name=self.name,
                category=SkillCategory.SCHEDULING,
            )
        try:
            payload = context.get("payload", context.get("data", {}))
            data = await _n8n_run_workflow(self._base_url, str(workflow_id), payload)
            return ProviderResult(
                success=True,
                output=data,
                execution_time=time.time() - start,
                provider_name=self.name,
                category=SkillCategory.SCHEDULING,
                metadata={"workflow_id": workflow_id},
            )
        except Exception as e:
            return ProviderResult(
                success=False,
                output=None,
                error=str(e),
                execution_time=time.time() - start,
                provider_name=self.name,
                category=SkillCategory.SCHEDULING,
                retryable=True,
            )
