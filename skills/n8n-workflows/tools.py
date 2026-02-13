"""
n8n Workflow Integration Skill

List, trigger, monitor, and manage n8n workflows.
Each workflow is also registered as a derived Jotty skill for planner discovery.
"""

import os
import re
import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

from Jotty.core.utils.env_loader import load_jotty_env
from Jotty.core.utils.api_client import BaseAPIClient
from Jotty.core.utils.tool_helpers import (
    tool_response, tool_error, async_tool_wrapper
)
from Jotty.core.utils.skill_status import SkillStatus

# Load environment variables
load_jotty_env()

status = SkillStatus("n8n-workflows")
logger = logging.getLogger(__name__)


# =============================================================================
# API CLIENT
# =============================================================================

class N8nAPIClient(BaseAPIClient):
    """n8n REST API client using X-N8N-API-KEY header auth."""

    AUTH_PREFIX = ""  # Not used — we override _get_headers
    TOKEN_ENV_VAR = "N8N_API_KEY"
    TOKEN_CONFIG_PATH = ""
    CONTENT_TYPE = "application/json"
    DEFAULT_TIMEOUT = 30

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(api_key or os.getenv("N8N_API_KEY"))
        self.BASE_URL = (
            base_url or os.getenv("N8N_BASE_URL", "")
        ).rstrip("/")

    def _get_headers(self) -> Dict[str, str]:
        """n8n uses X-N8N-API-KEY header instead of Authorization."""
        headers = {"Content-Type": self.CONTENT_TYPE}
        if self.token:
            headers["X-N8N-API-KEY"] = self.token
        return headers

    # -- Convenience wrappers around _make_request --

    def list_workflows(self, active_only: bool = False) -> Dict[str, Any]:
        """Fetch all workflows from n8n."""
        result = self._make_request("/api/v1/workflows", method="GET")
        if not result.get("success") and "data" not in result:
            return result
        workflows = result.get("data", [])
        if active_only:
            workflows = [w for w in workflows if w.get("active")]
        return {"success": True, "data": workflows}

    def get_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Fetch a single workflow by ID."""
        return self._make_request(f"/api/v1/workflows/{workflow_id}", method="GET")

    def activate_workflow(self, workflow_id: str, active: bool) -> Dict[str, Any]:
        """Activate or deactivate a workflow."""
        return self._make_request(
            f"/api/v1/workflows/{workflow_id}",
            method="PATCH",
            json_data={"active": active},
        )

    def trigger_via_webhook(self, webhook_path: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """POST to a webhook-triggered workflow."""
        url = f"{self.BASE_URL}/webhook/{webhook_path}"
        return self._make_request(url, method="POST", json_data=data or {})

    def create_execution(self, workflow_id: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Trigger a non-webhook workflow via executions API."""
        return self._make_request(
            "/api/v1/executions",
            method="POST",
            json_data={"workflowId": workflow_id, **(data or {})},
        )

    def get_execution(self, execution_id: str) -> Dict[str, Any]:
        """Check execution status."""
        return self._make_request(f"/api/v1/executions/{execution_id}", method="GET")


# =============================================================================
# WORKFLOW ANALYZER
# =============================================================================

class N8nWorkflowAnalyzer:
    """Classifies workflow trigger type from nodes array and produces summaries."""

    # Map node type substrings to trigger categories
    TRIGGER_MAP = {
        "webhook": "webhook",
        "scheduleTrigger": "schedule",
        "cron": "schedule",
        "manualTrigger": "manual",
        "executeWorkflow": "execute",
        "executeWorkflowTrigger": "execute",
    }

    @classmethod
    def classify_trigger(cls, nodes: List[Dict[str, Any]]) -> str:
        """Return trigger type: webhook | schedule | manual | execute | unknown."""
        for node in nodes:
            node_type = node.get("type", "")
            for key, trigger in cls.TRIGGER_MAP.items():
                if key in node_type:
                    return trigger
        return "unknown"

    @classmethod
    def get_webhook_path(cls, nodes: List[Dict[str, Any]]) -> Optional[str]:
        """Extract webhook path from a webhook-triggered workflow."""
        for node in nodes:
            if "webhook" in node.get("type", "").lower():
                params = node.get("parameters", {})
                return params.get("path") or params.get("options", {}).get("path")
        return None

    @classmethod
    def summarize_workflow(cls, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Produce a compact summary of a workflow."""
        nodes = workflow.get("nodes", [])
        return {
            "id": workflow.get("id"),
            "name": workflow.get("name", ""),
            "active": workflow.get("active", False),
            "trigger_type": cls.classify_trigger(nodes),
            "node_count": len(nodes),
            "tags": [t.get("name", "") for t in workflow.get("tags", [])],
            "updated_at": workflow.get("updatedAt", ""),
        }


# =============================================================================
# CAPABILITY INFERRER
# =============================================================================

class WorkflowCapabilityInferrer:
    """Infers domain-specific capabilities from workflow metadata.

    Examines three signals — tags, node types, and name segments — to build
    a rich capability set so the planner can match workflows to user tasks
    like "refresh my watchlists" or "download MCX quotes".
    """

    # Tag → capabilities
    TAG_MAP: Dict[str, set] = {
        "pmi": {"finance", "data-fetch"},
        "data": {"data-fetch"},
        "report": {"document", "analyze"},
        "cmd": {"devops"},
        "dev": {"devops"},
        "finance": {"finance"},
        "trading": {"finance"},
    }

    # Node type substring → capabilities
    NODE_MAP: Dict[str, set] = {
        "telegram": {"communicate"},
        "discord": {"communicate"},
        "slack": {"communicate"},
        "ssh": {"devops"},
        "httpRequest": {"data-fetch"},
        "googleGemini": {"analyze"},
        "openAi": {"analyze"},
        "code": {"devops"},
    }

    # Name segment → capabilities
    NAME_MAP: Dict[str, set] = {
        "download": {"data-fetch"},
        "refresh": {"data-fetch"},
        "sync": {"data-fetch"},
        "fetch": {"data-fetch"},
        "eodreport": {"document", "finance"},
        "watchlist": {"finance"},
        "quotes": {"finance"},
        "mcx": {"finance"},
        "nse": {"finance"},
        "bse": {"finance"},
        "report": {"document"},
    }

    # Project name mapping for use_when descriptions
    PROJECT_MAP: Dict[str, str] = {
        "pmi": "PlanMyInvesting",
        "cmd": "cmd.dev server",
    }

    @classmethod
    def infer_capabilities(cls, workflow: Dict[str, Any]) -> List[str]:
        """Union of tag, node, and name-based capabilities plus 'automation'."""
        caps: set = {"automation"}

        # Signal 1: tags
        tags = [t.get("name", "") if isinstance(t, dict) else str(t) for t in workflow.get("tags", [])]
        for tag in tags:
            tag_lower = tag.lower()
            for key, tag_caps in cls.TAG_MAP.items():
                if key in tag_lower:
                    caps |= tag_caps

        # Signal 2: node types
        for node in workflow.get("nodes", []):
            node_type = node.get("type", "")
            for key, node_caps in cls.NODE_MAP.items():
                if key in node_type:
                    caps |= node_caps

        # Signal 3: name segments
        name = workflow.get("name", "").lower()
        segments = re.split(r"[._\-\s]+", name)
        for segment in segments:
            for key, name_caps in cls.NAME_MAP.items():
                if key in segment:
                    caps |= name_caps

        return sorted(caps)

    @classmethod
    def infer_use_when(cls, workflow: Dict[str, Any]) -> str:
        """Natural-language use_when hint for LLM skill selection."""
        name = workflow.get("name", "")
        segments = re.split(r"[._\-\s]+", name.lower())

        # Detect project
        project = None
        for seg in segments:
            if seg in cls.PROJECT_MAP:
                project = cls.PROJECT_MAP[seg]
                break

        # Build action phrase from name
        # Filter out environment segments (prod, dev, staging) and project prefixes
        skip = {"prod", "dev", "staging", "test"} | set(cls.PROJECT_MAP.keys())
        action_parts = [s for s in segments if s not in skip and len(s) > 1]
        action = " ".join(action_parts) if action_parts else name

        if project:
            return f"User wants to {action} for {project}"
        return f"User wants to run {action}"

    @classmethod
    def infer_description(cls, workflow: Dict[str, Any], trigger_type: str) -> str:
        """Rich description including project context and trigger type."""
        name = workflow.get("name", "")
        segments = re.split(r"[._\-\s]+", name.lower())

        project = None
        for seg in segments:
            if seg in cls.PROJECT_MAP:
                project = cls.PROJECT_MAP[seg]
                break

        parts = [f"n8n workflow: {name}"]
        if project:
            parts.append(f"({project}, {trigger_type} trigger)")
        else:
            parts.append(f"({trigger_type} trigger)")

        return " ".join(parts)


# =============================================================================
# DYNAMIC SKILL REGISTRAR
# =============================================================================

class N8nDynamicSkillRegistrar:
    """
    Fetches all workflows and registers each as a derived SkillDefinition
    into the unified registry so the planner can discover them by name.

    Maintains a disk cache at ~/jotty/intelligence/workflow_skills_cache.json
    so derived skills are available at startup without an API call.
    """

    _registered = False

    CACHE_DIR = Path.home() / "jotty" / "intelligence"
    CACHE_FILE = "workflow_skills_cache.json"
    SCHEMA_VERSION = "1.0"

    @classmethod
    def register_all(cls, registry=None) -> int:
        """
        Register each n8n workflow as a derived skill.

        Uses WorkflowCapabilityInferrer for domain-specific capabilities,
        then persists enriched data to disk cache for offline startup.

        Returns:
            Number of workflows registered.
        """
        if cls._registered:
            return 0

        client = N8nAPIClient()
        if not client.token:
            logger.debug("N8N_API_KEY not set — skipping dynamic registration")
            return 0

        result = client.list_workflows()
        if not result.get("success"):
            logger.warning("Failed to list n8n workflows for registration: %s", result.get("error"))
            return 0

        if registry is None:
            try:
                from Jotty.core.registry import get_unified_registry
                registry = get_unified_registry()
            except Exception:
                logger.debug("Could not get registry for dynamic skill registration")
                return 0

        from Jotty.core.registry.skills_registry import SkillDefinition, SkillType

        count = 0
        workflows = result.get("data", [])
        for wf in workflows:
            skill_name = cls._workflow_to_skill_name(wf)
            if skill_name in getattr(registry, "loaded_skills", {}):
                continue

            summary = N8nWorkflowAnalyzer.summarize_workflow(wf)
            wf_id = wf.get("id", "")
            trigger_type = summary["trigger_type"]

            # Domain-specific inference
            capabilities = WorkflowCapabilityInferrer.infer_capabilities(wf)
            use_when = WorkflowCapabilityInferrer.infer_use_when(wf)
            description = WorkflowCapabilityInferrer.infer_description(wf, trigger_type)

            # Build a pre-bound trigger tool for this specific workflow
            def _make_trigger(wid=wf_id, sname=skill_name, wname=wf.get("name", wf_id)):
                async def _trigger(params: Dict[str, Any]) -> Dict[str, Any]:
                    params["workflow_id"] = wid
                    return await trigger_n8n_workflow_tool(params)
                _trigger.__name__ = f"trigger_{sname.replace('-', '_')}"
                _trigger.__doc__ = f"Trigger n8n workflow: {wname}"
                _trigger._required_params = []
                return _trigger

            skill = SkillDefinition(
                name=skill_name,
                description=description,
                tools={f"trigger_{skill_name.replace('-', '_')}": _make_trigger()},
                skill_type=SkillType.DERIVED,
                base_skills=["n8n-workflows"],
                capabilities=capabilities,
                use_when=use_when,
                metadata={
                    "n8n_workflow_id": wf_id,
                    "n8n_trigger_type": trigger_type,
                    "n8n_active": wf.get("active", False),
                },
            )

            if hasattr(registry, "loaded_skills"):
                registry.loaded_skills[skill_name] = skill
                count += 1

        # Persist cache for offline startup
        try:
            cls.save_cache(workflows, client.BASE_URL)
        except Exception as e:
            logger.warning("Failed to save workflow cache: %s", e)

        cls._registered = True
        logger.info("Registered %d n8n workflows as derived skills", count)
        return count

    @classmethod
    def save_cache(cls, workflows: List[Dict[str, Any]], base_url: str) -> None:
        """Write enriched workflow data to disk for offline startup.

        Args:
            workflows: Raw workflow dicts from the n8n API.
            base_url: n8n instance base URL (for provenance).
        """
        skills = []
        for wf in workflows:
            summary = N8nWorkflowAnalyzer.summarize_workflow(wf)
            trigger_type = summary["trigger_type"]
            tags = [t.get("name", "") if isinstance(t, dict) else str(t) for t in wf.get("tags", [])]

            skills.append({
                "skill_name": cls._workflow_to_skill_name(wf),
                "workflow_id": wf.get("id", ""),
                "workflow_name": wf.get("name", ""),
                "trigger_type": trigger_type,
                "active": wf.get("active", False),
                "description": WorkflowCapabilityInferrer.infer_description(wf, trigger_type),
                "capabilities": WorkflowCapabilityInferrer.infer_capabilities(wf),
                "use_when": WorkflowCapabilityInferrer.infer_use_when(wf),
                "tags": tags,
            })

        envelope = {
            "schema_version": cls.SCHEMA_VERSION,
            "provider": "n8n",
            "base_url": base_url,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "skills": skills,
        }

        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = cls.CACHE_DIR / cls.CACHE_FILE
        cache_path.write_text(json.dumps(envelope, indent=2))
        logger.info("Saved %d workflow skills to cache: %s", len(skills), cache_path)

    @classmethod
    def load_cache(cls) -> Optional[Dict[str, Any]]:
        """Load cached workflow skills from disk.

        Returns:
            Parsed cache envelope dict, or None if missing/invalid.
        """
        cache_path = cls.CACHE_DIR / cls.CACHE_FILE
        if not cache_path.exists():
            return None

        try:
            data = json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            logger.debug("Corrupt workflow cache — ignoring")
            return None

        if data.get("schema_version") != cls.SCHEMA_VERSION:
            logger.debug("Cache schema mismatch (got %s, want %s)", data.get("schema_version"), cls.SCHEMA_VERSION)
            return None

        return data

    @classmethod
    def reset(cls):
        """Reset registration state (for testing)."""
        cls._registered = False

    @staticmethod
    def _workflow_to_skill_name(workflow: Dict[str, Any]) -> str:
        """Convert workflow name to a valid Jotty skill name slug."""
        name = workflow.get("name", workflow.get("id", "unknown"))
        # Lowercase, replace non-alphanumeric with hyphens, collapse multiples
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        return f"n8n-{slug}"


# =============================================================================
# TOOL FUNCTIONS
# =============================================================================

def _get_client(params: Dict[str, Any]) -> tuple:
    """Get N8n client, returning (client, error) tuple."""
    client = N8nAPIClient(
        api_key=params.get("api_key"),
        base_url=params.get("base_url"),
    )
    if not client.token:
        return None, tool_error(
            "n8n API key required. Set N8N_API_KEY env var or provide api_key parameter"
        )
    if not client.BASE_URL:
        return None, tool_error(
            "n8n base URL required. Set N8N_BASE_URL env var or provide base_url parameter"
        )
    return client, None


@async_tool_wrapper()
async def list_n8n_workflows_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List all n8n workflows with trigger classification.

    Args:
        params: Dictionary containing:
            - active_only (bool, optional): Only return active workflows
            - api_key (str, optional): n8n API key (defaults to N8N_API_KEY env)
            - base_url (str, optional): n8n instance URL (defaults to N8N_BASE_URL env)

    Returns:
        Dictionary with success, workflows list, total count
    """
    status.set_callback(params.pop("_status_callback", None))

    client, error = _get_client(params)
    if error:
        return error

    status.emit("Fetching", "Fetching workflows from n8n...")
    result = client.list_workflows(active_only=params.get("active_only", False))

    if not result.get("success"):
        return tool_error(result.get("error", "Failed to list workflows"))

    workflows = result.get("data", [])
    summaries = [N8nWorkflowAnalyzer.summarize_workflow(wf) for wf in workflows]

    return tool_response(
        workflows=summaries,
        total=len(summaries),
    )


@async_tool_wrapper(required_params=["workflow_id"])
async def trigger_n8n_workflow_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Trigger an n8n workflow. Auto-detects webhook vs direct execution.

    Args:
        params: Dictionary containing:
            - workflow_id (str, required): Workflow ID to trigger
            - data (dict, optional): Payload data to send
            - wait (bool, optional): Wait for completion (default True)
            - timeout (int, optional): Max seconds to wait (default 300)
            - api_key (str, optional): n8n API key
            - base_url (str, optional): n8n instance URL

    Returns:
        Dictionary with success, execution_id, status, and data
    """
    status.set_callback(params.pop("_status_callback", None))

    client, error = _get_client(params)
    if error:
        return error

    workflow_id = params["workflow_id"]
    data = params.get("data", {})
    wait = params.get("wait", True)
    timeout = params.get("timeout", 300)

    # Fetch workflow to determine trigger type
    status.emit("Analyzing", f"Fetching workflow {workflow_id}...")
    wf_result = client.get_workflow(workflow_id)
    if not wf_result.get("success"):
        return tool_error(f"Could not fetch workflow {workflow_id}: {wf_result.get('error')}")

    nodes = wf_result.get("nodes", [])
    trigger_type = N8nWorkflowAnalyzer.classify_trigger(nodes)
    webhook_path = N8nWorkflowAnalyzer.get_webhook_path(nodes)

    # Route based on trigger type
    if trigger_type == "webhook" and webhook_path:
        status.emit("Triggering", f"POST to webhook /{webhook_path}...")
        result = client.trigger_via_webhook(webhook_path, data)
        if result.get("success"):
            return tool_response(
                execution_id=result.get("executionId"),
                trigger_type="webhook",
                data=result,
            )
        return tool_error(f"Webhook trigger failed: {result.get('error')}")

    # Non-webhook: use executions API
    status.emit("Triggering", f"Starting execution for workflow {workflow_id}...")
    result = client.create_execution(workflow_id, data)

    if not result.get("success"):
        return tool_error(f"Execution trigger failed: {result.get('error')}")

    execution_id = result.get("id") or result.get("executionId")
    if not execution_id:
        return tool_response(trigger_type=trigger_type, data=result)

    if not wait:
        return tool_response(
            execution_id=execution_id,
            trigger_type=trigger_type,
            status="started",
        )

    # Poll for completion
    return _poll_execution(client, execution_id, timeout)


def _poll_execution(client: N8nAPIClient, execution_id: str, timeout: int = 300) -> Dict[str, Any]:
    """Poll an execution until it completes or times out."""
    poll_interval = 2
    elapsed = 0

    while elapsed < timeout:
        result = client.get_execution(execution_id)
        if not result.get("success"):
            return tool_error(f"Failed to check execution {execution_id}: {result.get('error')}")

        exec_status = result.get("status", result.get("finished", ""))
        if exec_status in ("success", "error", "crashed", True):
            return tool_response(
                execution_id=execution_id,
                status="success" if exec_status in ("success", True) else exec_status,
                data=result.get("data"),
                finished=True,
            )

        time.sleep(poll_interval)
        elapsed += poll_interval

    return tool_error(f"Execution {execution_id} timed out after {timeout}s")


@async_tool_wrapper(required_params=["execution_id"])
async def get_n8n_execution_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check the status and results of an n8n execution.

    Args:
        params: Dictionary containing:
            - execution_id (str, required): Execution ID to check
            - api_key (str, optional): n8n API key
            - base_url (str, optional): n8n instance URL

    Returns:
        Dictionary with success, execution_id, status, data
    """
    status.set_callback(params.pop("_status_callback", None))

    client, error = _get_client(params)
    if error:
        return error

    execution_id = params["execution_id"]
    status.emit("Checking", f"Fetching execution {execution_id}...")
    result = client.get_execution(execution_id)

    if not result.get("success"):
        return tool_error(f"Failed to get execution: {result.get('error')}")

    return tool_response(
        execution_id=execution_id,
        status=result.get("status", "unknown"),
        finished=result.get("finished", False),
        data=result.get("data"),
        started_at=result.get("startedAt"),
        stopped_at=result.get("stoppedAt"),
    )


@async_tool_wrapper(required_params=["workflow_id"])
async def activate_n8n_workflow_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Activate or deactivate an n8n workflow.

    Args:
        params: Dictionary containing:
            - workflow_id (str, required): Workflow ID
            - active (bool, optional): True to activate, False to deactivate (default True)
            - api_key (str, optional): n8n API key
            - base_url (str, optional): n8n instance URL

    Returns:
        Dictionary with success, workflow_id, active state
    """
    status.set_callback(params.pop("_status_callback", None))

    client, error = _get_client(params)
    if error:
        return error

    workflow_id = params["workflow_id"]
    active = params.get("active", True)
    action = "Activating" if active else "Deactivating"

    status.emit(action, f"{action} workflow {workflow_id}...")
    result = client.activate_workflow(workflow_id, active)

    if not result.get("success"):
        return tool_error(f"Failed to {action.lower()} workflow: {result.get('error')}")

    return tool_response(
        workflow_id=workflow_id,
        active=result.get("active", active),
        name=result.get("name", ""),
    )


# =============================================================================
# WORKFLOW FACTORY
# =============================================================================

class N8nWorkflowFactory:
    """Creates n8n workflow definitions programmatically.

    Provides factory methods for common workflow patterns:
    - Schedule -> HTTP -> Telegram (scheduled reports)
    - Webhook -> HTTP -> Telegram (event-driven alerts)
    - Schedule -> SSH pipeline (devops automation)

    All methods return workflow JSON dicts suitable for n8n API POST.
    """

    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
    PMI_BASE_URL = os.getenv("PMI_API_URL", "http://localhost:5000")

    # Fields the n8n API rejects on POST (read-only).
    _READ_ONLY_FIELDS = {"active", "tags"}

    @classmethod
    def _base_workflow(cls, name: str, tags: list) -> Dict[str, Any]:
        """Create base workflow structure.

        ``tags`` are stored under a private ``_tags`` key so they survive
        round-tripping through ``get_all_workflow_definitions`` but are
        stripped before the POST to the n8n API (which treats ``tags``
        as read-only).  ``create_all_workflows`` handles tagging after
        creation via a separate PATCH call.
        """
        return {
            "name": name,
            "nodes": [],
            "connections": {},
            "settings": {"executionOrder": "v1"},
            "_tags": tags,  # private — stripped before POST
        }

    @classmethod
    def _schedule_trigger_node(cls, cron: str) -> Dict[str, Any]:
        """Create a schedule trigger node."""
        return {
            "id": "schedule-trigger",
            "name": "Schedule Trigger",
            "type": "n8n-nodes-base.scheduleTrigger",
            "typeVersion": 1.2,
            "position": [250, 300],
            "parameters": {
                "rule": {"interval": [{"field": "cronExpression", "expression": cron}]},
            },
        }

    @classmethod
    def _webhook_trigger_node(cls, webhook_path: str) -> Dict[str, Any]:
        """Create a webhook trigger node."""
        return {
            "id": "webhook-trigger",
            "name": "Webhook",
            "type": "n8n-nodes-base.webhook",
            "typeVersion": 2,
            "position": [250, 300],
            "parameters": {"path": webhook_path, "httpMethod": "POST"},
            "webhookId": webhook_path,
        }

    @classmethod
    def _http_request_node(cls, url: str, method: str = "GET",
                           node_id: str = "http-request") -> Dict[str, Any]:
        """Create an HTTP request node."""
        return {
            "id": node_id,
            "name": "HTTP Request",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "position": [470, 300],
            "parameters": {"url": url, "method": method},
        }

    @classmethod
    def _telegram_node(cls, message_template: str,
                       node_id: str = "telegram-send") -> Dict[str, Any]:
        """Create a Telegram send message node."""
        return {
            "id": node_id,
            "name": "Telegram",
            "type": "n8n-nodes-base.telegram",
            "typeVersion": 1.2,
            "position": [690, 300],
            "parameters": {
                "chatId": cls.TELEGRAM_CHAT_ID,
                "text": message_template,
                "additionalFields": {"parse_mode": "Markdown"},
            },
            "credentials": {"telegramApi": {"id": "telegram", "name": "Telegram"}},
        }

    @classmethod
    def _ssh_node(cls, command: str, node_id: str = "ssh-exec") -> Dict[str, Any]:
        """Create an SSH execute command node."""
        return {
            "id": node_id,
            "name": "SSH",
            "type": "n8n-nodes-base.ssh",
            "typeVersion": 1,
            "position": [470, 300],
            "parameters": {"command": command},
            "credentials": {"sshPassword": {"id": "ssh", "name": "SSH"}},
        }

    @classmethod
    def create_schedule_to_http_to_telegram(
        cls, name: str, cron: str, http_url: str, http_method: str,
        message_template: str, tags: List[str],
    ) -> Dict[str, Any]:
        """Build: ScheduleTrigger -> HTTP Request -> Telegram."""
        wf = cls._base_workflow(name, tags)
        trigger = cls._schedule_trigger_node(cron)
        http_req = cls._http_request_node(http_url, http_method)
        telegram = cls._telegram_node(message_template)
        wf["nodes"] = [trigger, http_req, telegram]
        wf["connections"] = {
            "Schedule Trigger": {"main": [[{"node": "HTTP Request", "type": "main", "index": 0}]]},
            "HTTP Request": {"main": [[{"node": "Telegram", "type": "main", "index": 0}]]},
        }
        return wf

    @classmethod
    def create_webhook_to_http_to_telegram(
        cls, name: str, webhook_path: str, http_url: str,
        message_template: str, tags: List[str],
    ) -> Dict[str, Any]:
        """Build: Webhook -> HTTP Request -> Telegram."""
        wf = cls._base_workflow(name, tags)
        trigger = cls._webhook_trigger_node(webhook_path)
        http_req = cls._http_request_node(http_url, "POST")
        telegram = cls._telegram_node(message_template)
        wf["nodes"] = [trigger, http_req, telegram]
        wf["connections"] = {
            "Webhook": {"main": [[{"node": "HTTP Request", "type": "main", "index": 0}]]},
            "HTTP Request": {"main": [[{"node": "Telegram", "type": "main", "index": 0}]]},
        }
        return wf

    @classmethod
    def create_schedule_to_ssh_pipeline(
        cls, name: str, cron: str, ssh_command: str, tags: List[str],
    ) -> Dict[str, Any]:
        """Build: ScheduleTrigger -> SSH -> conditional Telegram on error."""
        wf = cls._base_workflow(name, tags)
        trigger = cls._schedule_trigger_node(cron)
        ssh = cls._ssh_node(ssh_command)
        telegram = cls._telegram_node(
            f"[ALERT] {name} failed: {{{{$json.stderr}}}}",
            node_id="telegram-alert",
        )
        telegram["position"] = [690, 300]
        wf["nodes"] = [trigger, ssh, telegram]
        wf["connections"] = {
            "Schedule Trigger": {"main": [[{"node": "SSH", "type": "main", "index": 0}]]},
            "SSH": {"main": [[{"node": "Telegram", "type": "main", "index": 0}]]},
        }
        return wf

    @classmethod
    def get_all_workflow_definitions(cls) -> List[Dict[str, Any]]:
        """Return all 15 planned workflow definitions."""
        base = cls.PMI_BASE_URL
        return [
            # --- PMI Production Workflows ---
            cls.create_schedule_to_http_to_telegram(
                name="pmi.prod.morning_brief",
                cron="45 8 * * 1-5",
                http_url=f"{base}/v2/get_indices",
                http_method="GET",
                message_template="*Morning Brief*\n{{$json.data}}",
                tags=["pmi", "prod", "finance"],
            ),
            cls.create_schedule_to_http_to_telegram(
                name="pmi.prod.closing_brief",
                cron="45 15 * * 1-5",
                http_url=f"{base}/v2/get_pnl_summary",
                http_method="GET",
                message_template="*Closing Brief*\nP&L: {{$json.total_pnl}}",
                tags=["pmi", "prod", "finance"],
            ),
            cls.create_schedule_to_http_to_telegram(
                name="pmi.prod.portfolio_snapshot",
                cron="0 16 * * 1-5",
                http_url=f"{base}/v2/portfolio",
                http_method="GET",
                message_template="*Portfolio Snapshot*\n{{$json.data}}",
                tags=["pmi", "prod", "finance"],
            ),
            cls.create_schedule_to_http_to_telegram(
                name="pmi.prod.fii_dii_tracker",
                cron="0 18 * * 1-5",
                http_url="https://www.nseindia.com/api/fiidiiTradeReact",
                http_method="GET",
                message_template="*FII/DII Activity*\n{{$json.data}}",
                tags=["pmi", "prod", "finance"],
            ),
            cls.create_webhook_to_http_to_telegram(
                name="pmi.prod.strategy_signals",
                webhook_path="pmi-strategy-signals",
                http_url=f"{base}/api/strategies/generate-signals",
                message_template="*Strategy Signals*\n{{$json.signals}}",
                tags=["pmi", "prod", "finance"],
            ),
            cls.create_webhook_to_http_to_telegram(
                name="pmi.prod.order_alerts",
                webhook_path="pmi-order-alerts",
                http_url=f"{base}/v2/orders",
                message_template="*Order Alert*\n{{$json.data}}",
                tags=["pmi", "prod", "finance"],
            ),
            cls.create_schedule_to_http_to_telegram(
                name="pmi.prod.earnings_calendar",
                cron="0 9 * * 1",
                http_url=f"{base}/v2/earnings_calendar",
                http_method="GET",
                message_template="*Earnings This Week*\n{{$json.data}}",
                tags=["pmi", "prod", "finance"],
            ),
            cls.create_schedule_to_http_to_telegram(
                name="pmi.prod.global_markets",
                cron="0 7 * * 1-5",
                http_url=f"{base}/v2/get_global_indices",
                http_method="GET",
                message_template="*Global Markets*\n{{$json.data}}",
                tags=["pmi", "prod", "finance"],
            ),
            cls.create_schedule_to_http_to_telegram(
                name="pmi.prod.weekly_report",
                cron="0 18 * * 0",
                http_url=f"{base}/v2/portfolio",
                http_method="GET",
                message_template="*Weekly Portfolio Report*\n{{$json.data}}",
                tags=["pmi", "prod", "finance", "report"],
            ),
            cls.create_schedule_to_http_to_telegram(
                name="pmi.prod.ipo_tracker",
                cron="0 10 * * *",
                http_url=f"{base}/v2/ipo_list",
                http_method="GET",
                message_template="*IPO Tracker*\n{{$json.data}}",
                tags=["pmi", "prod", "finance"],
            ),
            cls.create_schedule_to_http_to_telegram(
                name="pmi.prod.watchlist_alerts",
                cron="*/15 9-16 * * 1-5",
                http_url=f"{base}/v2/watchlists/refresh",
                http_method="GET",
                message_template="*Watchlist Alert*\nPrice change >2%: {{$json.data}}",
                tags=["pmi", "prod", "finance"],
            ),
            # --- DevOps Workflows ---
            cls.create_schedule_to_http_to_telegram(
                name="cmd.dev.health_check",
                cron="*/5 * * * *",
                http_url="http://localhost:5000/health",
                http_method="GET",
                message_template="[ALERT] Health check failed: {{$json.error}}",
                tags=["cmd", "dev", "devops"],
            ),
            cls.create_schedule_to_ssh_pipeline(
                name="cmd.dev.db_backup",
                cron="0 2 * * *",
                ssh_command="mongodump --gzip --archive=/backups/mongo_$(date +%Y%m%d).gz",
                tags=["cmd", "dev", "devops"],
            ),
            cls.create_schedule_to_ssh_pipeline(
                name="cmd.dev.ssl_monitor",
                cron="0 6 * * *",
                ssh_command="echo | openssl s_client -connect planmyinvesting.com:443 2>/dev/null | openssl x509 -noout -enddate",
                tags=["cmd", "dev", "devops"],
            ),
            cls.create_schedule_to_ssh_pipeline(
                name="cmd.dev.log_errors",
                cron="0 * * * *",
                ssh_command="tail -n 1000 /var/log/pmi/error.log | grep -c ERROR",
                tags=["cmd", "dev", "devops"],
            ),
        ]

    @classmethod
    def create_all_workflows(cls, client: N8nAPIClient) -> int:
        """Create all 15 planned workflows in n8n. Idempotent (skips existing).

        Args:
            client: Configured N8nAPIClient instance.

        Returns:
            Number of workflows created.
        """
        workflows = cls.get_all_workflow_definitions()

        # Get existing workflow names
        result = client.list_workflows()
        existing = set()
        if result.get("success"):
            existing = {w.get("name", "") for w in result.get("data", [])}

        # Build tag name→id lookup for tagging after creation
        tag_map: Dict[str, str] = {}
        tags_result = client._make_request("/api/v1/tags", method="GET")
        if tags_result.get("success"):
            for t in tags_result.get("data", tags_result.get("tags", [])):
                if isinstance(t, dict) and t.get("id") and t.get("name"):
                    tag_map[t["name"]] = t["id"]

        count = 0
        for wf_def in workflows:
            if wf_def["name"] in existing:
                logger.debug("Workflow '%s' already exists, skipping", wf_def["name"])
                continue

            # Strip read-only / private fields before POST
            tag_names = wf_def.pop("_tags", [])
            payload = {k: v for k, v in wf_def.items()
                       if k not in cls._READ_ONLY_FIELDS}

            create_result = client._make_request(
                "/api/v1/workflows", method="POST", json_data=payload,
            )
            if create_result.get("success"):
                count += 1
                logger.info("Created workflow: %s", wf_def["name"])
                # Tag via PUT with tag IDs (tags are read-only on POST)
                wf_id = create_result.get("id")
                if wf_id and tag_names:
                    tag_ids = [{"id": tag_map[t]} for t in tag_names if t in tag_map]
                    if tag_ids:
                        client._make_request(
                            f"/api/v1/workflows/{wf_id}/tags",
                            method="PUT",
                            json_data=tag_ids,
                        )
            else:
                logger.warning(
                    "Failed to create workflow '%s': %s",
                    wf_def["name"], create_result.get("error"),
                )

        # Refresh cache with all workflows (existing + new)
        try:
            refresh = client.list_workflows()
            if refresh.get("success"):
                N8nDynamicSkillRegistrar.save_cache(refresh["data"], client.BASE_URL)
        except Exception as e:
            logger.warning("Cache refresh failed after workflow creation: %s", e)

        return count


@async_tool_wrapper()
async def setup_n8n_workflows_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    One-time setup: create all standard n8n workflows.

    Creates 15 workflows (11 PMI finance + 4 DevOps) in the configured n8n instance.
    Idempotent: skips workflows that already exist by name.

    Args:
        params: Dictionary containing:
            - api_key (str, optional): n8n API key
            - base_url (str, optional): n8n instance URL

    Returns:
        Dictionary with created count and total available
    """
    status.set_callback(params.pop("_status_callback", None))

    client, error = _get_client(params)
    if error:
        return error

    status.emit("Creating", "Setting up n8n workflows...")
    count = N8nWorkflowFactory.create_all_workflows(client)

    status.emit("Verifying", "Verifying workflows...")
    result = client.list_workflows()
    total = len(result.get("data", [])) if result.get("success") else 0

    return tool_response(
        created=count,
        total_workflows=total,
        message=f"Created {count} new workflows ({total} total)",
    )


__all__ = [
    "N8nAPIClient",
    "N8nWorkflowAnalyzer",
    "WorkflowCapabilityInferrer",
    "N8nDynamicSkillRegistrar",
    "N8nWorkflowFactory",
    "list_n8n_workflows_tool",
    "trigger_n8n_workflow_tool",
    "get_n8n_execution_tool",
    "activate_n8n_workflow_tool",
    "setup_n8n_workflows_tool",
]
