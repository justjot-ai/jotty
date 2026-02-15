"""
n8n Workflow Integration Tests
==============================

Unit tests with mocked HTTP calls — no real n8n instance needed.
Includes capability inference, cache I/O, and cached registration tests.
"""

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Load module from hyphenated directory via importlib (not importable as package)
_tools_path = Path(__file__).resolve().parent.parent / "skills" / "n8n-workflows" / "tools.py"
_spec = importlib.util.spec_from_file_location("n8n_workflows_tools", _tools_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

N8nAPIClient = _mod.N8nAPIClient
N8nWorkflowAnalyzer = _mod.N8nWorkflowAnalyzer
WorkflowCapabilityInferrer = _mod.WorkflowCapabilityInferrer
N8nDynamicSkillRegistrar = _mod.N8nDynamicSkillRegistrar
list_n8n_workflows_tool = _mod.list_n8n_workflows_tool
trigger_n8n_workflow_tool = _mod.trigger_n8n_workflow_tool
get_n8n_execution_tool = _mod.get_n8n_execution_tool
activate_n8n_workflow_tool = _mod.activate_n8n_workflow_tool
_get_client = _mod._get_client

# Add parent directory for core imports — import skills_registry module directly
# to avoid broken __init__.py in core.registry (ToolValidator NameError)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Pre-register the package hierarchy so relative imports inside skills_registry work
import types

for pkg_name in ["core", "core.registry"]:
    if pkg_name not in sys.modules:
        sys.modules[pkg_name] = types.ModuleType(pkg_name)
        sys.modules[pkg_name].__path__ = [
            str(Path(__file__).resolve().parent.parent / pkg_name.replace(".", "/"))
        ]
        sys.modules[pkg_name].__package__ = pkg_name

_sr_path = Path(__file__).resolve().parent.parent / "core" / "registry" / "skills_registry.py"
_sr_spec = importlib.util.spec_from_file_location(
    "core.registry.skills_registry",
    _sr_path,
    submodule_search_locations=[],
)
_sr_mod = importlib.util.module_from_spec(_sr_spec)
_sr_mod.__package__ = "core.registry"
sys.modules["core.registry.skills_registry"] = _sr_mod
_sr_spec.loader.exec_module(_sr_mod)
SkillsRegistry = _sr_mod.SkillsRegistry
SkillDefinition = _sr_mod.SkillDefinition
SkillType = _sr_mod.SkillType


# =============================================================================
# Fixtures
# =============================================================================

SAMPLE_WORKFLOWS = [
    {
        "id": "wf1",
        "name": "PMI Prod Download Data",
        "active": True,
        "nodes": [
            {
                "type": "n8n-nodes-base.scheduleTrigger",
                "parameters": {"rule": {"interval": [{"field": "hours", "hoursInterval": 6}]}},
            },
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


# -- Workflow fixtures for capability inference tests --


def _make_workflow(name, wf_id="WF001", tags=None, nodes=None, active=True):
    """Factory for workflow dicts that mimic the n8n API response."""
    return {
        "id": wf_id,
        "name": name,
        "active": active,
        "tags": [{"name": t} for t in (tags or [])],
        "nodes": nodes or [{"type": "n8n-nodes-base.manualTrigger"}],
        "updatedAt": "2026-01-01T00:00:00.000Z",
    }


@pytest.fixture
def pmi_workflow():
    """PMI finance workflow with schedule trigger and telegram node."""
    return _make_workflow(
        name="pmi.prod.refresh_watchlists",
        wf_id="RI3QEQQQhfsNlHpc",
        tags=["pmi", "prod", "data"],
        nodes=[
            {"type": "n8n-nodes-base.scheduleTrigger"},
            {"type": "n8n-nodes-base.httpRequest"},
            {"type": "n8n-nodes-base.telegramApi", "parameters": {}},
        ],
        active=False,
    )


@pytest.fixture
def eodreport_workflow():
    return _make_workflow(
        name="pmi.prod.eodreport",
        wf_id="EOD001",
        tags=["pmi", "prod", "report"],
        nodes=[{"type": "n8n-nodes-base.scheduleTrigger"}],
    )


@pytest.fixture
def download_workflow():
    return _make_workflow(
        name="pmi.prod.download_data",
        wf_id="DL001",
        tags=["pmi", "data"],
        nodes=[
            {"type": "n8n-nodes-base.scheduleTrigger"},
            {"type": "n8n-nodes-base.httpRequest"},
        ],
    )


@pytest.fixture
def cmd_workflow():
    return _make_workflow(
        name="cmd.dev.sync_data",
        wf_id="CMD001",
        tags=["cmd", "dev"],
        nodes=[
            {"type": "n8n-nodes-base.manualTrigger"},
            {"type": "n8n-nodes-base.ssh"},
        ],
    )


@pytest.fixture
def bare_workflow():
    """Workflow with no tags and no special nodes."""
    return _make_workflow(
        name="misc_cleanup",
        wf_id="MISC001",
        tags=[],
        nodes=[{"type": "n8n-nodes-base.manualTrigger"}],
    )


@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Override CACHE_DIR to a temp directory for test isolation."""
    original = N8nDynamicSkillRegistrar.CACHE_DIR
    N8nDynamicSkillRegistrar.CACHE_DIR = tmp_path
    yield tmp_path
    N8nDynamicSkillRegistrar.CACHE_DIR = original


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
        result = await trigger_n8n_workflow_tool(
            {
                "workflow_id": "wf2",
                "data": {"test": True},
            }
        )
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
        result = await trigger_n8n_workflow_tool(
            {
                "workflow_id": "wf3",
                "wait": False,
            }
        )
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
        result = await trigger_n8n_workflow_tool(
            {
                "workflow_id": "wf3",
                "wait": True,
                "timeout": 10,
            }
        )
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
        result = await activate_n8n_workflow_tool(
            {
                "workflow_id": "wf1",
                "active": True,
            }
        )
        assert result["success"] is True
        assert result["active"] is True

    @patch.object(N8nAPIClient, "_make_request")
    @patch.dict("os.environ", {"N8N_API_KEY": "k", "N8N_BASE_URL": "https://n8n.test"})
    async def test_deactivate(self, mock_req):
        mock_req.return_value = {"success": True, "active": False, "name": "My WF"}
        result = await activate_n8n_workflow_tool(
            {
                "workflow_id": "wf1",
                "active": False,
            }
        )
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
    def test_register_all(self, mock_req, tmp_cache_dir):
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

        # Verify enriched capabilities (not just generic "automation")
        assert "data-fetch" in skill.capabilities
        assert "automation" in skill.capabilities

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

    @patch.object(N8nAPIClient, "_make_request")
    @patch.dict("os.environ", {"N8N_API_KEY": "k", "N8N_BASE_URL": "https://n8n.test"})
    def test_register_all_enriched_capabilities(self, mock_req, pmi_workflow, tmp_cache_dir):
        """register_all uses inferrer for domain-specific capabilities."""
        mock_req.return_value = {"success": True, "data": [pmi_workflow]}
        mock_registry = MagicMock()
        mock_registry.loaded_skills = {}

        count = N8nDynamicSkillRegistrar.register_all(registry=mock_registry)
        assert count == 1
        skill = mock_registry.loaded_skills["n8n-pmi-prod-refresh-watchlists"]
        assert "finance" in skill.capabilities
        assert "data-fetch" in skill.capabilities
        assert "PlanMyInvesting" in skill.use_when

    @patch.object(N8nAPIClient, "_make_request")
    @patch.dict("os.environ", {"N8N_API_KEY": "k", "N8N_BASE_URL": "https://n8n.test"})
    def test_register_all_saves_cache(self, mock_req, pmi_workflow, tmp_cache_dir):
        """register_all persists cache to disk."""
        mock_req.return_value = {"success": True, "data": [pmi_workflow]}
        mock_registry = MagicMock()
        mock_registry.loaded_skills = {}

        N8nDynamicSkillRegistrar.register_all(registry=mock_registry)
        cache = N8nDynamicSkillRegistrar.load_cache()
        assert cache is not None
        assert len(cache["skills"]) == 1

    def test_workflow_to_skill_name(self):
        assert (
            N8nDynamicSkillRegistrar._workflow_to_skill_name({"name": "PMI Prod Download Data"})
            == "n8n-pmi-prod-download-data"
        )

        assert (
            N8nDynamicSkillRegistrar._workflow_to_skill_name({"name": "Jotty Test Webhook"})
            == "n8n-jotty-test-webhook"
        )

        assert (
            N8nDynamicSkillRegistrar._workflow_to_skill_name({"name": "  Extra   Spaces!! "})
            == "n8n-extra-spaces"
        )


# =============================================================================
# TestWorkflowCapabilityInferrer
# =============================================================================


@pytest.mark.unit
class TestWorkflowCapabilityInferrer:
    """Tests for WorkflowCapabilityInferrer capability/use_when/description inference."""

    def test_pmi_workflow_has_finance(self, pmi_workflow):
        caps = WorkflowCapabilityInferrer.infer_capabilities(pmi_workflow)
        assert "finance" in caps
        assert "data-fetch" in caps

    def test_telegram_node_adds_communicate(self, pmi_workflow):
        caps = WorkflowCapabilityInferrer.infer_capabilities(pmi_workflow)
        assert "communicate" in caps

    def test_always_includes_automation(self, bare_workflow):
        caps = WorkflowCapabilityInferrer.infer_capabilities(bare_workflow)
        assert "automation" in caps

    def test_use_when_with_pmi_project(self, pmi_workflow):
        use_when = WorkflowCapabilityInferrer.infer_use_when(pmi_workflow)
        assert "PlanMyInvesting" in use_when
        assert "refresh" in use_when.lower()

    def test_use_when_fallback(self, bare_workflow):
        use_when = WorkflowCapabilityInferrer.infer_use_when(bare_workflow)
        assert "cleanup" in use_when.lower()

    def test_description_includes_trigger(self, pmi_workflow):
        desc = WorkflowCapabilityInferrer.infer_description(pmi_workflow, "schedule")
        assert "schedule trigger" in desc
        assert "PlanMyInvesting" in desc

    def test_download_segments_add_data_fetch(self, download_workflow):
        caps = WorkflowCapabilityInferrer.infer_capabilities(download_workflow)
        assert "data-fetch" in caps

    def test_eodreport_gets_document_finance(self, eodreport_workflow):
        caps = WorkflowCapabilityInferrer.infer_capabilities(eodreport_workflow)
        assert "document" in caps
        assert "finance" in caps


# =============================================================================
# TestCacheIO
# =============================================================================


@pytest.mark.unit
class TestCacheIO:
    """Tests for N8nDynamicSkillRegistrar disk cache save/load."""

    def test_save_and_load_roundtrip(self, pmi_workflow, tmp_cache_dir):
        N8nDynamicSkillRegistrar.save_cache([pmi_workflow], "http://n8n.local:5678")

        loaded = N8nDynamicSkillRegistrar.load_cache()
        assert loaded is not None
        assert loaded["schema_version"] == "1.0"
        assert loaded["provider"] == "n8n"
        assert len(loaded["skills"]) == 1
        assert loaded["skills"][0]["workflow_id"] == "RI3QEQQQhfsNlHpc"

    def test_load_cache_missing(self, tmp_cache_dir):
        loaded = N8nDynamicSkillRegistrar.load_cache()
        assert loaded is None

    def test_load_cache_bad_schema(self, tmp_cache_dir):
        bad_data = {"schema_version": "99.0", "skills": []}
        cache_path = tmp_cache_dir / N8nDynamicSkillRegistrar.CACHE_FILE
        cache_path.write_text(json.dumps(bad_data))

        loaded = N8nDynamicSkillRegistrar.load_cache()
        assert loaded is None

    def test_cache_contains_enriched_capabilities(self, pmi_workflow, tmp_cache_dir):
        N8nDynamicSkillRegistrar.save_cache([pmi_workflow], "http://n8n.local:5678")

        loaded = N8nDynamicSkillRegistrar.load_cache()
        skill_entry = loaded["skills"][0]
        assert "finance" in skill_entry["capabilities"]
        assert "data-fetch" in skill_entry["capabilities"]
        assert len(skill_entry["capabilities"]) > 1


# =============================================================================
# TestCachedRegistration
# =============================================================================


@pytest.mark.unit
class TestCachedRegistration:
    """Tests for SkillsRegistry._load_cached_dynamic_skills loading from cache."""

    def _write_cache(self, cache_dir, skills):
        """Write a minimal valid cache file."""
        envelope = {
            "schema_version": "1.0",
            "provider": "n8n",
            "base_url": "http://n8n.local:5678",
            "fetched_at": "2026-01-01T00:00:00+00:00",
            "skills": skills,
        }
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "workflow_skills_cache.json").write_text(json.dumps(envelope))

    def test_load_from_cache_creates_skills(self, tmp_path):
        """Mock cache file -> _load_cached_dynamic_skills creates SkillDefinitions."""
        cache_skills = [
            {
                "skill_name": "n8n-pmi-prod-refresh-watchlists",
                "workflow_id": "RI3Q",
                "workflow_name": "pmi.prod.refresh_watchlists",
                "trigger_type": "schedule",
                "active": False,
                "description": "n8n workflow: pmi.prod.refresh_watchlists (PlanMyInvesting, schedule trigger)",
                "capabilities": ["automation", "data-fetch", "finance"],
                "use_when": "User wants to refresh watchlists for PlanMyInvesting",
                "tags": ["pmi", "prod"],
            }
        ]

        registry = SkillsRegistry(skills_dir=str(tmp_path / "skills"))
        intelligence_dir = tmp_path / "intelligence"
        self._write_cache(intelligence_dir, cache_skills)

        with patch.object(SkillsRegistry, "_INTELLIGENCE_DIR", intelligence_dir):
            registry._load_cached_dynamic_skills()

        assert "n8n-pmi-prod-refresh-watchlists" in registry.loaded_skills

    def test_cached_skill_has_correct_type(self, tmp_path):
        cache_skills = [
            {
                "skill_name": "n8n-test-workflow",
                "workflow_id": "T001",
                "workflow_name": "test_workflow",
                "trigger_type": "manual",
                "active": True,
                "description": "test",
                "capabilities": ["automation"],
                "use_when": "test",
                "tags": [],
            }
        ]

        registry = SkillsRegistry(skills_dir=str(tmp_path / "skills"))
        intelligence_dir = tmp_path / "intelligence"
        self._write_cache(intelligence_dir, cache_skills)

        with patch.object(SkillsRegistry, "_INTELLIGENCE_DIR", intelligence_dir):
            registry._load_cached_dynamic_skills()

        skill = registry.loaded_skills["n8n-test-workflow"]
        assert skill.skill_type == SkillType.DERIVED
        assert "n8n-workflows" in skill.base_skills

    def test_cached_skill_has_domain_capabilities(self, tmp_path):
        cache_skills = [
            {
                "skill_name": "n8n-pmi-download",
                "workflow_id": "D001",
                "workflow_name": "pmi.prod.download_data",
                "trigger_type": "schedule",
                "active": True,
                "description": "download data",
                "capabilities": ["automation", "data-fetch", "finance"],
                "use_when": "User wants to download data for PlanMyInvesting",
                "tags": ["pmi"],
            }
        ]

        registry = SkillsRegistry(skills_dir=str(tmp_path / "skills"))
        intelligence_dir = tmp_path / "intelligence"
        self._write_cache(intelligence_dir, cache_skills)

        with patch.object(SkillsRegistry, "_INTELLIGENCE_DIR", intelligence_dir):
            registry._load_cached_dynamic_skills()

        skill = registry.loaded_skills["n8n-pmi-download"]
        assert "finance" in skill.capabilities
        assert "data-fetch" in skill.capabilities

    def test_cached_skill_tool_binds_workflow_id(self, tmp_path):
        """The lazy tool loader pre-binds workflow_id into trigger params."""
        cache_skills = [
            {
                "skill_name": "n8n-bind-test",
                "workflow_id": "BIND001",
                "workflow_name": "bind_test",
                "trigger_type": "webhook",
                "active": True,
                "description": "test binding",
                "capabilities": ["automation"],
                "use_when": "test",
                "tags": [],
            }
        ]

        registry = SkillsRegistry(skills_dir=str(tmp_path / "skills"))
        intelligence_dir = tmp_path / "intelligence"
        self._write_cache(intelligence_dir, cache_skills)

        with patch.object(SkillsRegistry, "_INTELLIGENCE_DIR", intelligence_dir):
            registry._load_cached_dynamic_skills()

        skill = registry.loaded_skills["n8n-bind-test"]
        tools = skill.tools
        assert len(tools) == 1
        tool_fn = list(tools.values())[0]
        assert "bind_test" in (tool_fn.__doc__ or "")

    def test_cached_skill_idempotent(self, tmp_path):
        """Loading cache twice doesn't duplicate skills."""
        cache_skills = [
            {
                "skill_name": "n8n-idempotent-test",
                "workflow_id": "IDEM1",
                "workflow_name": "idempotent_test",
                "trigger_type": "manual",
                "active": True,
                "description": "test",
                "capabilities": ["automation"],
                "use_when": "test",
                "tags": [],
            }
        ]

        registry = SkillsRegistry(skills_dir=str(tmp_path / "skills"))
        intelligence_dir = tmp_path / "intelligence"
        self._write_cache(intelligence_dir, cache_skills)

        with patch.object(SkillsRegistry, "_INTELLIGENCE_DIR", intelligence_dir):
            registry._load_cached_dynamic_skills()
            registry._load_cached_dynamic_skills()

        assert len([n for n in registry.loaded_skills if n == "n8n-idempotent-test"]) == 1


# =============================================================================
# Load N8nWorkflowFactory from updated module
# =============================================================================

N8nWorkflowFactory = _mod.N8nWorkflowFactory
setup_n8n_workflows_tool = _mod.setup_n8n_workflows_tool


# =============================================================================
# TestN8nWorkflowFactory
# =============================================================================


@pytest.mark.unit
class TestN8nWorkflowFactory:
    """Tests for N8nWorkflowFactory workflow creation."""

    def test_schedule_to_http_to_telegram_structure(self):
        """Factory creates valid schedule->http->telegram workflow."""
        wf = N8nWorkflowFactory.create_schedule_to_http_to_telegram(
            name="test.morning_brief",
            cron="45 8 * * 1-5",
            http_url="http://localhost:5000/v2/get_indices",
            http_method="GET",
            message_template="*Morning Brief*\n{{$json.data}}",
            tags=["pmi", "prod"],
        )
        assert wf["name"] == "test.morning_brief"
        assert len(wf["nodes"]) == 3

        # Verify node types
        node_types = [n["type"] for n in wf["nodes"]]
        assert "n8n-nodes-base.scheduleTrigger" in node_types
        assert "n8n-nodes-base.httpRequest" in node_types
        assert "n8n-nodes-base.telegram" in node_types

        # Verify connections
        assert "Schedule Trigger" in wf["connections"]
        assert "HTTP Request" in wf["connections"]

        # Tags stored as private _tags (stripped before POST to n8n)
        assert "pmi" in wf["_tags"]
        assert "active" not in wf  # read-only, not included

    def test_webhook_workflow_structure(self):
        """Factory creates valid webhook->http->telegram workflow."""
        wf = N8nWorkflowFactory.create_webhook_to_http_to_telegram(
            name="test.order_alerts",
            webhook_path="test-orders",
            http_url="http://localhost:5000/v2/orders",
            message_template="*Order Alert*\n{{$json.data}}",
            tags=["pmi", "prod"],
        )
        assert wf["name"] == "test.order_alerts"
        assert len(wf["nodes"]) == 3

        # Verify webhook trigger
        trigger = next(n for n in wf["nodes"] if "webhook" in n["type"])
        assert trigger["parameters"]["path"] == "test-orders"
        assert "Webhook" in wf["connections"]

    def test_ssh_workflow_structure(self):
        """Factory creates valid schedule->ssh pipeline workflow."""
        wf = N8nWorkflowFactory.create_schedule_to_ssh_pipeline(
            name="test.db_backup",
            cron="0 2 * * *",
            ssh_command="mongodump --gzip",
            tags=["cmd", "dev"],
        )
        assert wf["name"] == "test.db_backup"
        assert len(wf["nodes"]) == 3

        # Verify SSH node
        ssh_node = next(n for n in wf["nodes"] if "ssh" in n["type"])
        assert ssh_node["parameters"]["command"] == "mongodump --gzip"

    def test_get_all_workflow_definitions(self):
        """get_all_workflow_definitions returns 15 workflows."""
        workflows = N8nWorkflowFactory.get_all_workflow_definitions()
        assert len(workflows) == 15

        names = [w["name"] for w in workflows]
        assert "pmi.prod.morning_brief" in names
        assert "pmi.prod.closing_brief" in names
        assert "cmd.dev.db_backup" in names
        assert "cmd.dev.ssl_monitor" in names

    @patch.object(N8nAPIClient, "_make_request")
    @patch.dict("os.environ", {"N8N_API_KEY": "k", "N8N_BASE_URL": "https://n8n.test"})
    def test_create_all_idempotent(self, mock_req, tmp_cache_dir):
        """create_all_workflows skips existing workflows by name."""
        mock_req.side_effect = [
            # 1: list_workflows
            {
                "success": True,
                "data": [
                    {"name": "pmi.prod.morning_brief", "id": "existing1"},
                    {"name": "pmi.prod.closing_brief", "id": "existing2"},
                ],
            },
            # 2: list tags
            {"success": True, "data": [{"id": "t1", "name": "pmi"}]},
            # 13 create calls + 13 tag PUT calls (alternating)
            *[
                resp
                for i in range(13)
                for resp in (
                    {"success": True, "id": f"new{i}"},
                    {"success": True},
                )
            ],
            # final list_workflows for cache refresh
            {"success": True, "data": []},
        ]
        client = N8nAPIClient(api_key="k", base_url="https://n8n.test")
        count = N8nWorkflowFactory.create_all_workflows(client)
        assert count == 13  # 15 total minus 2 existing

    @patch.object(N8nAPIClient, "_make_request")
    @patch.dict("os.environ", {"N8N_API_KEY": "k", "N8N_BASE_URL": "https://n8n.test"})
    def test_create_all_handles_failure(self, mock_req, tmp_cache_dir):
        """create_all_workflows handles individual workflow creation failures."""
        mock_req.side_effect = [
            # 1: list_workflows (empty)
            {"success": True, "data": []},
            # 2: list tags
            {"success": True, "data": [{"id": "t1", "name": "pmi"}]},
            # 3-4: workflow #1 create + tag
            {"success": True, "id": "new1"},
            {"success": True},
            # 5: workflow #2 create fails (no tag call)
            {"success": False, "error": "Internal error"},
            # 6-31: workflows #3-#15 create + tag (13 workflows x 2 calls)
            *[
                resp
                for i in range(2, 15)
                for resp in (
                    {"success": True, "id": f"new{i}"},
                    {"success": True},
                )
            ],
            # 32: cache refresh
            {"success": True, "data": []},
        ]
        client = N8nAPIClient(api_key="k", base_url="https://n8n.test")
        count = N8nWorkflowFactory.create_all_workflows(client)
        assert count == 14  # 15 total minus 1 failure
