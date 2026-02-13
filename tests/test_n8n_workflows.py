"""
n8n Workflow Integration Tests
==============================

Unit tests with mocked HTTP calls — no real n8n instance needed.
"""

import importlib.util
from pathlib import Path

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

# Load module from hyphenated directory via importlib (not importable as package)
_tools_path = Path(__file__).resolve().parent.parent / "skills" / "n8n-workflows" / "tools.py"
_spec = importlib.util.spec_from_file_location("n8n_workflows_tools", _tools_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

N8nAPIClient = _mod.N8nAPIClient
N8nWorkflowAnalyzer = _mod.N8nWorkflowAnalyzer
N8nDynamicSkillRegistrar = _mod.N8nDynamicSkillRegistrar
list_n8n_workflows_tool = _mod.list_n8n_workflows_tool
trigger_n8n_workflow_tool = _mod.trigger_n8n_workflow_tool
get_n8n_execution_tool = _mod.get_n8n_execution_tool
activate_n8n_workflow_tool = _mod.activate_n8n_workflow_tool
_get_client = _mod._get_client


# =============================================================================
# Fixtures
# =============================================================================

SAMPLE_WORKFLOWS = [
    {
        "id": "wf1",
        "name": "PMI Prod Download Data",
        "active": True,
        "nodes": [
            {"type": "n8n-nodes-base.scheduleTrigger", "parameters": {"rule": {"interval": [{"field": "hours", "hoursInterval": 6}]}}},
            {"type": "n8n-nodes-base.httpRequest", "parameters": {}},
        ],
        "tags": [{"name": "pmi"}],
        "updatedAt": "2026-02-10T10:00:00Z",
    },
    {
        "id": "wf2",
        "name": "Jotty Test Webhook",
        "active": True,
        "nodes": [
            {"type": "n8n-nodes-base.webhook", "parameters": {"path": "jotty-test"}},
            {"type": "n8n-nodes-base.respondToWebhook", "parameters": {}},
        ],
        "tags": [{"name": "jotty"}],
        "updatedAt": "2026-02-12T08:00:00Z",
    },
    {
        "id": "wf3",
        "name": "Manual Report",
        "active": False,
        "nodes": [
            {"type": "n8n-nodes-base.manualTrigger", "parameters": {}},
        ],
        "tags": [],
        "updatedAt": "2026-01-15T12:00:00Z",
    },
]


@pytest.fixture
def n8n_client():
    """N8nAPIClient with test credentials."""
    return N8nAPIClient(api_key="test-key", base_url="https://n8n.test.local")


# =============================================================================
# TestN8nAPIClient
# =============================================================================

@pytest.mark.unit
class TestN8nAPIClient:
    """Test N8nAPIClient header format, base URL, and token loading."""

    def test_headers_use_x_n8n_api_key(self, n8n_client):
        """Headers must include X-N8N-API-KEY, not Authorization."""
        headers = n8n_client._get_headers()
        assert headers["X-N8N-API-KEY"] == "test-key"
        assert "Authorization" not in headers
        assert headers["Content-Type"] == "application/json"

    def test_headers_without_token(self):
        """No X-N8N-API-KEY header when token is missing."""
        client = N8nAPIClient(api_key=None, base_url="https://n8n.test.local")
        client.token = None
        headers = client._get_headers()
        assert "X-N8N-API-KEY" not in headers

    def test_base_url_set(self, n8n_client):
        """BASE_URL is set from constructor."""
        assert n8n_client.BASE_URL == "https://n8n.test.local"

    def test_base_url_strips_trailing_slash(self):
        """Trailing slash is stripped from base URL."""
        client = N8nAPIClient(api_key="k", base_url="https://n8n.test.local/")
        assert client.BASE_URL == "https://n8n.test.local"

    @patch.dict("os.environ", {"N8N_API_KEY": "env-key", "N8N_BASE_URL": "https://env.n8n"})
    def test_loads_from_env(self):
        """Falls back to environment variables."""
        client = N8nAPIClient()
        assert client.token == "env-key"
        assert client.BASE_URL == "https://env.n8n"

    def test_is_configured(self, n8n_client):
        """is_configured returns True when token present."""
        assert n8n_client.is_configured is True

    def test_not_configured_without_key(self):
        """is_configured returns False when token missing."""
        client = N8nAPIClient(api_key=None, base_url="https://x")
        client.token = None
        assert client.is_configured is False


# =============================================================================
# TestN8nWorkflowAnalyzer
# =============================================================================

@pytest.mark.unit
class TestN8nWorkflowAnalyzer:
    """Test trigger classification from workflow nodes."""

    def test_classify_webhook(self):
        nodes = [{"type": "n8n-nodes-base.webhook", "parameters": {"path": "test"}}]
        assert N8nWorkflowAnalyzer.classify_trigger(nodes) == "webhook"

    def test_classify_schedule(self):
        nodes = [{"type": "n8n-nodes-base.scheduleTrigger", "parameters": {}}]
        assert N8nWorkflowAnalyzer.classify_trigger(nodes) == "schedule"

    def test_classify_cron(self):
        nodes = [{"type": "n8n-nodes-base.cron", "parameters": {}}]
        assert N8nWorkflowAnalyzer.classify_trigger(nodes) == "schedule"

    def test_classify_manual(self):
        nodes = [{"type": "n8n-nodes-base.manualTrigger", "parameters": {}}]
        assert N8nWorkflowAnalyzer.classify_trigger(nodes) == "manual"

    def test_classify_execute(self):
        nodes = [{"type": "n8n-nodes-base.executeWorkflowTrigger", "parameters": {}}]
        assert N8nWorkflowAnalyzer.classify_trigger(nodes) == "execute"

    def test_classify_empty_nodes(self):
        assert N8nWorkflowAnalyzer.classify_trigger([]) == "unknown"

    def test_classify_no_trigger_node(self):
        nodes = [{"type": "n8n-nodes-base.httpRequest", "parameters": {}}]
        assert N8nWorkflowAnalyzer.classify_trigger(nodes) == "unknown"

    def test_get_webhook_path(self):
        nodes = [{"type": "n8n-nodes-base.webhook", "parameters": {"path": "jotty-test"}}]
        assert N8nWorkflowAnalyzer.get_webhook_path(nodes) == "jotty-test"

    def test_get_webhook_path_none(self):
        nodes = [{"type": "n8n-nodes-base.scheduleTrigger", "parameters": {}}]
        assert N8nWorkflowAnalyzer.get_webhook_path(nodes) is None

    def test_summarize_workflow(self):
        summary = N8nWorkflowAnalyzer.summarize_workflow(SAMPLE_WORKFLOWS[0])
        assert summary["id"] == "wf1"
        assert summary["name"] == "PMI Prod Download Data"
        assert summary["active"] is True
        assert summary["trigger_type"] == "schedule"
        assert summary["node_count"] == 2
        assert summary["tags"] == ["pmi"]

    def test_summarize_webhook_workflow(self):
        summary = N8nWorkflowAnalyzer.summarize_workflow(SAMPLE_WORKFLOWS[1])
        assert summary["trigger_type"] == "webhook"


# =============================================================================
# TestListWorkflowsTool
# =============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
class TestListWorkflowsTool:
    """Test list_n8n_workflows_tool."""

    @patch.object(N8nAPIClient, "_make_request")
    @patch.dict("os.environ", {"N8N_API_KEY": "k", "N8N_BASE_URL": "https://n8n.test"})
    async def test_list_success(self, mock_req):
        mock_req.return_value = {"success": True, "data": SAMPLE_WORKFLOWS}
        result = await list_n8n_workflows_tool({})
        assert result["success"] is True
        assert result["total"] == 3
        assert len(result["workflows"]) == 3
        assert result["workflows"][0]["id"] == "wf1"

    @patch.object(N8nAPIClient, "_make_request")
    @patch.dict("os.environ", {"N8N_API_KEY": "k", "N8N_BASE_URL": "https://n8n.test"})
    async def test_list_active_only(self, mock_req):
        mock_req.return_value = {"success": True, "data": SAMPLE_WORKFLOWS}
        result = await list_n8n_workflows_tool({"active_only": True})
        assert result["success"] is True
        assert result["total"] == 2  # wf3 is inactive

    @patch.dict("os.environ", {}, clear=True)
    async def test_list_no_api_key(self):
        # Ensure no env vars leak through
        result = await list_n8n_workflows_tool({"base_url": "https://n8n.test"})
        assert result["success"] is False
        assert "API key" in result["error"]

    @patch.dict("os.environ", {"N8N_API_KEY": "k"}, clear=True)
    async def test_list_no_base_url(self):
        result = await list_n8n_workflows_tool({})
        assert result["success"] is False
        assert "base URL" in result["error"]


# =============================================================================
# TestTriggerWorkflowTool
# =============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
class TestTriggerWorkflowTool:
    """Test trigger_n8n_workflow_tool with webhook and direct execution."""

    @patch.object(N8nAPIClient, "_make_request")
    @patch.dict("os.environ", {"N8N_API_KEY": "k", "N8N_BASE_URL": "https://n8n.test"})
    async def test_webhook_trigger(self, mock_req):
        """Webhook workflow should POST to /webhook/{path}."""
        # First call: get_workflow; second call: trigger_via_webhook
        mock_req.side_effect = [
            {"success": True, "nodes": SAMPLE_WORKFLOWS[1]["nodes"]},
            {"success": True, "executionId": "exec-1"},
        ]
        result = await trigger_n8n_workflow_tool({
            "workflow_id": "wf2",
            "data": {"test": True},
        })
        assert result["success"] is True
        assert result["trigger_type"] == "webhook"
        # Verify webhook URL was called
        calls = mock_req.call_args_list
        assert "webhook/jotty-test" in calls[1][0][0]

    @patch.object(N8nAPIClient, "_make_request")
    @patch.dict("os.environ", {"N8N_API_KEY": "k", "N8N_BASE_URL": "https://n8n.test"})
    async def test_direct_execution_no_wait(self, mock_req):
        """Non-webhook workflow with wait=False returns immediately."""
        mock_req.side_effect = [
            {"success": True, "nodes": SAMPLE_WORKFLOWS[2]["nodes"]},  # manual trigger
            {"success": True, "id": "exec-2"},  # create_execution
        ]
        result = await trigger_n8n_workflow_tool({
            "workflow_id": "wf3",
            "wait": False,
        })
        assert result["success"] is True
        assert result["execution_id"] == "exec-2"
        assert result["status"] == "started"

    @patch.object(_mod, "time")
    @patch.object(N8nAPIClient, "_make_request")
    @patch.dict("os.environ", {"N8N_API_KEY": "k", "N8N_BASE_URL": "https://n8n.test"})
    async def test_direct_execution_poll_success(self, mock_req, mock_time):
        """Non-webhook workflow with wait=True polls until success."""
        mock_req.side_effect = [
            {"success": True, "nodes": SAMPLE_WORKFLOWS[2]["nodes"]},
            {"success": True, "id": "exec-3"},
            {"success": True, "status": "running"},
            {"success": True, "status": "success", "data": {"output": "done"}},
        ]
        result = await trigger_n8n_workflow_tool({
            "workflow_id": "wf3",
            "wait": True,
            "timeout": 10,
        })
        assert result["success"] is True
        assert result["finished"] is True
        assert mock_time.sleep.call_count == 1

    @patch.object(N8nAPIClient, "_make_request")
    @patch.dict("os.environ", {"N8N_API_KEY": "k", "N8N_BASE_URL": "https://n8n.test"})
    async def test_trigger_workflow_not_found(self, mock_req):
        """Should return error when workflow doesn't exist."""
        mock_req.return_value = {"success": False, "error": "Not Found"}
        result = await trigger_n8n_workflow_tool({"workflow_id": "bad-id"})
        assert result["success"] is False
        assert "Could not fetch" in result["error"]


# =============================================================================
# TestGetExecutionTool
# =============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
class TestGetExecutionTool:
    """Test get_n8n_execution_tool."""

    @patch.object(N8nAPIClient, "_make_request")
    @patch.dict("os.environ", {"N8N_API_KEY": "k", "N8N_BASE_URL": "https://n8n.test"})
    async def test_get_execution_success(self, mock_req):
        mock_req.return_value = {
            "success": True,
            "status": "success",
            "finished": True,
            "data": {"result": "ok"},
            "startedAt": "2026-02-14T00:00:00Z",
            "stoppedAt": "2026-02-14T00:01:00Z",
        }
        result = await get_n8n_execution_tool({"execution_id": "exec-1"})
        assert result["success"] is True
        assert result["status"] == "success"
        assert result["finished"] is True

    @patch.object(N8nAPIClient, "_make_request")
    @patch.dict("os.environ", {"N8N_API_KEY": "k", "N8N_BASE_URL": "https://n8n.test"})
    async def test_get_execution_error(self, mock_req):
        mock_req.return_value = {"success": False, "error": "Not Found"}
        result = await get_n8n_execution_tool({"execution_id": "bad"})
        assert result["success"] is False

    async def test_get_execution_missing_param(self):
        result = await get_n8n_execution_tool({})
        assert result["success"] is False
        assert "execution_id" in result["error"]


# =============================================================================
# TestActivateWorkflowTool
# =============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
class TestActivateWorkflowTool:
    """Test activate_n8n_workflow_tool."""

    @patch.object(N8nAPIClient, "_make_request")
    @patch.dict("os.environ", {"N8N_API_KEY": "k", "N8N_BASE_URL": "https://n8n.test"})
    async def test_activate(self, mock_req):
        mock_req.return_value = {"success": True, "active": True, "name": "My WF"}
        result = await activate_n8n_workflow_tool({
            "workflow_id": "wf1",
            "active": True,
        })
        assert result["success"] is True
        assert result["active"] is True

    @patch.object(N8nAPIClient, "_make_request")
    @patch.dict("os.environ", {"N8N_API_KEY": "k", "N8N_BASE_URL": "https://n8n.test"})
    async def test_deactivate(self, mock_req):
        mock_req.return_value = {"success": True, "active": False, "name": "My WF"}
        result = await activate_n8n_workflow_tool({
            "workflow_id": "wf1",
            "active": False,
        })
        assert result["success"] is True
        assert result["active"] is False

    @patch.object(N8nAPIClient, "_make_request")
    @patch.dict("os.environ", {"N8N_API_KEY": "k", "N8N_BASE_URL": "https://n8n.test"})
    async def test_activate_failure(self, mock_req):
        mock_req.return_value = {"success": False, "error": "Permission denied"}
        result = await activate_n8n_workflow_tool({"workflow_id": "wf1"})
        assert result["success"] is False


# =============================================================================
# TestDynamicSkillRegistrar
# =============================================================================

@pytest.mark.unit
class TestDynamicSkillRegistrar:
    """Test N8nDynamicSkillRegistrar."""

    def setup_method(self):
        N8nDynamicSkillRegistrar.reset()

    @patch.object(N8nAPIClient, "_make_request")
    @patch.dict("os.environ", {"N8N_API_KEY": "k", "N8N_BASE_URL": "https://n8n.test"})
    def test_register_all(self, mock_req):
        mock_req.return_value = {"success": True, "data": SAMPLE_WORKFLOWS}
        mock_registry = MagicMock()
        mock_registry.loaded_skills = {}

        count = N8nDynamicSkillRegistrar.register_all(registry=mock_registry)
        assert count == 3
        assert "n8n-pmi-prod-download-data" in mock_registry.loaded_skills
        assert "n8n-jotty-test-webhook" in mock_registry.loaded_skills
        assert "n8n-manual-report" in mock_registry.loaded_skills

        # Verify skill properties
        skill = mock_registry.loaded_skills["n8n-pmi-prod-download-data"]
        assert skill.skill_type.value == "derived"
        assert skill.base_skills == ["n8n-workflows"]

    @patch.object(N8nAPIClient, "_make_request")
    @patch.dict("os.environ", {"N8N_API_KEY": "k", "N8N_BASE_URL": "https://n8n.test"})
    def test_register_idempotent(self, mock_req):
        """Second call should return 0 (already registered)."""
        mock_req.return_value = {"success": True, "data": SAMPLE_WORKFLOWS}
        mock_registry = MagicMock()
        mock_registry.loaded_skills = {}

        N8nDynamicSkillRegistrar.register_all(registry=mock_registry)
        count = N8nDynamicSkillRegistrar.register_all(registry=mock_registry)
        assert count == 0

    @patch.dict("os.environ", {}, clear=True)
    def test_register_without_key(self):
        """Should silently skip when no API key."""
        count = N8nDynamicSkillRegistrar.register_all()
        assert count == 0

    def test_workflow_to_skill_name(self):
        assert N8nDynamicSkillRegistrar._workflow_to_skill_name(
            {"name": "PMI Prod Download Data"}
        ) == "n8n-pmi-prod-download-data"

        assert N8nDynamicSkillRegistrar._workflow_to_skill_name(
            {"name": "Jotty Test Webhook"}
        ) == "n8n-jotty-test-webhook"

        assert N8nDynamicSkillRegistrar._workflow_to_skill_name(
            {"name": "  Extra   Spaces!! "}
        ) == "n8n-extra-spaces"
