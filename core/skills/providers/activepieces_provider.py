"""
Activepieces Skill Provider
============================

Exposes Activepieces flows as skills (one skill per flow). DRY: connector
logic stays in Activepieces; we only list flows and run by id. KISS: one
provider, same pattern as N8nProvider. Config: ACTIVEPIECES_BASE_URL,
ACTIVEPIECES_API_KEY (or apiKey in config).
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

DEFAULT_ACTIVEPIECES_URL = "http://localhost:8080"


def _ap_headers(config: Dict[str, Any]) -> Dict[str, str]:
    key = config.get("api_key") or os.getenv("ACTIVEPIECES_API_KEY")
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


async def _ap_get_flows(base_url: str, headers: Dict[str, str]) -> List[Dict[str, Any]]:
    """Fetch flow list from Activepieces API."""
    try:
        import aiohttp
    except ImportError:
        logger.debug("aiohttp not installed; activepieces list_skills will be empty")
        return []
    # Common patterns: /api/v1/flows or /flows
    for path in ["/api/v1/flows", "/flows"]:
        url = f"{base_url.rstrip('/')}{path}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=10) as resp:
                    if resp.status != 200:
                        continue
                    data = await resp.json()
                    # Response may be { data: [...] } or direct list
                    if isinstance(data, list):
                        return data
                    return data.get("data", data.get("flows", []))
        except Exception as e:
            logger.debug("activepieces list %s: %s", path, e)
    return []


async def _ap_run_flow(
    base_url: str, headers: Dict[str, str], flow_id: str, payload: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Run flow by id."""
    try:
        import aiohttp
    except ImportError:
        raise RuntimeError("aiohttp required for activepieces provider")
    for path in [f"/api/v1/flows/{flow_id}/run", f"/flows/{flow_id}/run"]:
        url = f"{base_url.rstrip('/')}{path}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, headers=headers, json=payload or {}, timeout=60
                ) as resp:
                    if resp.status == 404:
                        continue
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"activepieces run failed {resp.status}: {text}")
                    return await resp.json()
        except RuntimeError:
            raise
        except Exception:
            continue
    raise RuntimeError(f"activepieces run not available for flow {flow_id}")


class ActivepiecesProvider(SkillProvider):
    """
    Skill provider that exposes Activepieces flows as skills.
    One skill per flow; execute runs that flow via Activepieces API.
    """

    name = "activepieces"
    version = "1.0.0"
    description = "Activepieces flows as skills (one per flow)"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self._base_url = self.config.get("base_url") or os.getenv(
            "ACTIVEPIECES_BASE_URL", DEFAULT_ACTIVEPIECES_URL
        )
        self._headers = _ap_headers(self.config)
        self._skills_cache: List[ContributedSkill] = []
        self.capabilities = [
            ProviderCapability(
                category=SkillCategory.SCHEDULING,
                actions=["run_flow", "trigger", "schedule"],
                estimated_latency_ms=2000,
            )
        ]

    def get_categories(self) -> List[SkillCategory]:
        return [SkillCategory.SCHEDULING]

    def list_skills(self) -> List[ContributedSkill]:
        """Sync: returns cached list (populated in initialize)."""
        return list(self._skills_cache)

    async def _fetch_skills(self) -> List[ContributedSkill]:
        flows = await _ap_get_flows(self._base_url, self._headers)
        skills = []
        for f in flows:
            flow_id = f.get("id")
            if flow_id is None:
                continue
            name = f.get("name", "Unnamed")
            skills.append(
                ContributedSkill(
                    id=f"activepieces:flow:{flow_id}",
                    name=name,
                    description=f.get("description") or f"Activepieces flow: {name}",
                    provider=self.name,
                    metadata={"flow_id": str(flow_id)},
                )
            )
        self._skills_cache = skills
        return skills

    async def initialize(self) -> bool:
        try:
            skills = await self._fetch_skills()
            self.is_initialized = True
            self.is_available = True
            logger.info(" activepieces provider initialized (%d flows as skills)", len(skills))
            return True
        except Exception as e:
            logger.warning("activepieces provider init failed (will retry on use): %s", e)
            self.is_initialized = True
            self.is_available = False
            return True

    async def execute(self, task: str, context: Dict[str, Any] = None) -> ProviderResult:
        start = time.time()
        context = context or {}
        flow_id = context.get("flow_id") or context.get("skill_id")
        if flow_id and isinstance(flow_id, str) and flow_id.startswith("activepieces:flow:"):
            flow_id = flow_id.replace("activepieces:flow:", "")
        if not flow_id:
            return ProviderResult(
                success=False,
                output=None,
                error="Missing flow_id or skill_id in context (e.g. activepieces:flow:ID)",
                execution_time=time.time() - start,
                provider_name=self.name,
                category=SkillCategory.SCHEDULING,
            )
        try:
            payload = context.get("payload", context.get("data", {}))
            data = await _ap_run_flow(self._base_url, self._headers, str(flow_id), payload)
            return ProviderResult(
                success=True,
                output=data,
                execution_time=time.time() - start,
                provider_name=self.name,
                category=SkillCategory.SCHEDULING,
                metadata={"flow_id": flow_id},
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
