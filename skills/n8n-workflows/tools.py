"""
n8n Workflow Integration Skill

List, trigger, monitor, and manage n8n workflows.
Each workflow is also registered as a derived Jotty skill for planner discovery.
"""

import os
import re
import time
import logging
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
# DYNAMIC SKILL REGISTRAR
# =============================================================================

class N8nDynamicSkillRegistrar:
    """
    Fetches all workflows and registers each as a derived SkillDefinition
    into the unified registry so the planner can discover them by name.
    """

    _registered = False

    @classmethod
    def register_all(cls, registry=None) -> int:
        """
        Register each n8n workflow as a derived skill.

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
        for wf in result.get("data", []):
            skill_name = cls._workflow_to_skill_name(wf)
            if skill_name in getattr(registry, "loaded_skills", {}):
                continue

            summary = N8nWorkflowAnalyzer.summarize_workflow(wf)
            wf_id = wf.get("id", "")
            trigger_type = summary["trigger_type"]

            # Build a pre-bound trigger tool for this specific workflow
            def _make_trigger(wid=wf_id):
                async def _trigger(params: Dict[str, Any]) -> Dict[str, Any]:
                    params["workflow_id"] = wid
                    return await trigger_n8n_workflow_tool(params)
                _trigger.__name__ = f"trigger_{skill_name.replace('-', '_')}"
                _trigger.__doc__ = f"Trigger n8n workflow: {wf.get('name', wid)}"
                _trigger._required_params = []
                return _trigger

            skill = SkillDefinition(
                name=skill_name,
                description=f"n8n workflow: {wf.get('name', '')} ({trigger_type} trigger)",
                tools={f"trigger_{skill_name.replace('-', '_')}": _make_trigger()},
                skill_type=SkillType.DERIVED,
                base_skills=["n8n-workflows"],
                capabilities=["automation"],
                use_when=f"User wants to run the '{wf.get('name', '')}' workflow",
                metadata={
                    "n8n_workflow_id": wf_id,
                    "n8n_trigger_type": trigger_type,
                    "n8n_active": wf.get("active", False),
                },
            )

            if hasattr(registry, "loaded_skills"):
                registry.loaded_skills[skill_name] = skill
                count += 1

        cls._registered = True
        logger.info("Registered %d n8n workflows as derived skills", count)
        return count

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


__all__ = [
    "N8nAPIClient",
    "N8nWorkflowAnalyzer",
    "N8nDynamicSkillRegistrar",
    "list_n8n_workflows_tool",
    "trigger_n8n_workflow_tool",
    "get_n8n_execution_tool",
    "activate_n8n_workflow_tool",
]
