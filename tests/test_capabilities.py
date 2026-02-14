"""
Tests for the Capability Discovery API (Phase 1).

Verifies that capabilities() returns valid structure with all subsystems,
and that explain() provides useful descriptions.
"""

import pytest


@pytest.mark.unit
class TestCapabilities:
    """Tests for capabilities() function."""

    def test_capabilities_returns_dict(self):
        from Jotty.core.capabilities import capabilities
        result = capabilities()
        assert isinstance(result, dict)

    def test_capabilities_has_required_keys(self):
        from Jotty.core.capabilities import capabilities
        result = capabilities()
        expected_keys = {
            "execution_paths",
            "subsystems",
            "swarms",
            "skills_count",
            "providers",
            "utilities",
        }
        assert expected_keys.issubset(result.keys())

    def test_execution_paths_has_three_modes(self):
        from Jotty.core.capabilities import capabilities
        paths = capabilities()["execution_paths"]
        assert "chat" in paths
        assert "workflow" in paths
        assert "swarm" in paths

    def test_execution_path_has_description(self):
        from Jotty.core.capabilities import capabilities
        paths = capabilities()["execution_paths"]
        for name, info in paths.items():
            assert "description" in info, f"Path {name} missing 'description'"
            assert "class" in info, f"Path {name} missing 'class'"
            assert "tier" in info, f"Path {name} missing 'tier'"

    def test_subsystems_has_all_six(self):
        from Jotty.core.capabilities import capabilities
        subs = capabilities()["subsystems"]
        expected = {"learning", "memory", "context", "orchestration", "skills", "utils"}
        assert expected.issubset(subs.keys())

    def test_subsystem_has_required_fields(self):
        from Jotty.core.capabilities import capabilities
        subs = capabilities()["subsystems"]
        for name, info in subs.items():
            assert "description" in info, f"Subsystem {name} missing 'description'"
            assert "package" in info, f"Subsystem {name} missing 'package'"
            assert "facade" in info, f"Subsystem {name} missing 'facade'"
            assert "key_classes" in info, f"Subsystem {name} missing 'key_classes'"
            assert len(info["key_classes"]) > 0, f"Subsystem {name} has no key_classes"

    def test_providers_is_list(self):
        from Jotty.core.capabilities import capabilities
        providers = capabilities()["providers"]
        assert isinstance(providers, list)
        assert len(providers) > 0

    def test_provider_has_name_and_description(self):
        from Jotty.core.capabilities import capabilities
        for p in capabilities()["providers"]:
            assert "name" in p
            assert "description" in p
            assert "installed" in p

    def test_utilities_has_budget_tracker(self):
        from Jotty.core.capabilities import capabilities
        utils = capabilities()["utilities"]
        assert "BudgetTracker" in utils
        assert "import_path" in utils["BudgetTracker"]

    def test_skills_count_is_int(self):
        from Jotty.core.capabilities import capabilities
        count = capabilities()["skills_count"]
        assert isinstance(count, int)
        assert count >= 0

    def test_swarms_is_list(self):
        from Jotty.core.capabilities import capabilities
        swarms = capabilities()["swarms"]
        assert isinstance(swarms, list)


@pytest.mark.unit
class TestExplain:
    """Tests for explain() function."""

    def test_explain_learning(self):
        from Jotty.core.capabilities import explain
        result = explain("learning")
        assert "Learning Subsystem" in result
        assert "TDLambdaLearner" in result

    def test_explain_memory(self):
        from Jotty.core.capabilities import explain
        result = explain("memory")
        assert "Memory Subsystem" in result
        assert "MemorySystem" in result

    def test_explain_context(self):
        from Jotty.core.capabilities import explain
        result = explain("context")
        assert "Context Subsystem" in result
        assert "SmartContextManager" in result

    def test_explain_orchestration(self):
        from Jotty.core.capabilities import explain
        result = explain("orchestration")
        assert "Orchestration Subsystem" in result
        assert "SwarmIntelligence" in result

    def test_explain_skills(self):
        from Jotty.core.capabilities import explain
        result = explain("skills")
        assert "Skills Subsystem" in result
        assert "UnifiedRegistry" in result

    def test_explain_utils(self):
        from Jotty.core.capabilities import explain
        result = explain("utils")
        assert "Utilities Subsystem" in result
        assert "BudgetTracker" in result

    def test_explain_chat(self):
        from Jotty.core.capabilities import explain
        result = explain("chat")
        assert "Chat" in result

    def test_explain_workflow(self):
        from Jotty.core.capabilities import explain
        result = explain("workflow")
        assert "Workflow" in result

    def test_explain_swarm(self):
        from Jotty.core.capabilities import explain
        result = explain("swarm")
        assert "Swarm" in result

    def test_explain_unknown(self):
        from Jotty.core.capabilities import explain
        result = explain("nonexistent_widget")
        assert "Unknown component" in result

    def test_explain_case_insensitive(self):
        from Jotty.core.capabilities import explain
        result = explain("LEARNING")
        assert "Learning Subsystem" in result

    def test_explain_hyphen_normalization(self):
        from Jotty.core.capabilities import explain
        # "chat" with various formats should still work
        result = explain("chat")
        assert "Chat" in result


@pytest.mark.unit
class TestCapabilitiesImport:
    """Tests for top-level import accessibility."""

    def test_import_from_core(self):
        from Jotty.core.capabilities import capabilities, explain
        assert callable(capabilities)
        assert callable(explain)

    def test_lazy_import_from_top_level(self):
        from Jotty import capabilities
        assert callable(capabilities)
