"""
Tests for advanced learning components.

Tests: UCB in SkillQTable, Reflexion, FewShotCurator, VoyagerSkillLib,
       ValidationResult + validate_output cascade.

All tests are offline — no real LLM calls. DSPy modules are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# UCB in SkillQTable
# =============================================================================


class TestSkillQTableUCB:
    """Test UCB1 exploration in SkillQTable."""

    @pytest.mark.unit
    def test_cold_start_returns_all_skills(self):
        from Jotty.core.intelligence.learning.td_lambda import SkillQTable

        q = SkillQTable()
        result = q.select("research", ["web-search", "calculator", "pdf-gen"])
        assert len(result) == 3
        assert set(result) == {"web-search", "calculator", "pdf-gen"}

    @pytest.mark.unit
    def test_unvisited_skill_explored_first(self):
        from Jotty.core.intelligence.learning.td_lambda import SkillQTable

        q = SkillQTable()
        q.update("research", "web-search", 0.8)
        q.update("research", "web-search", 0.9)
        q.update("research", "calculator", 0.3)

        result = q.select("research", ["web-search", "calculator", "pdf-gen"])
        # pdf-gen has 0 visits → UCB = inf → should be first
        assert result[0] == "pdf-gen"

    @pytest.mark.unit
    def test_ucb_balances_exploitation_and_exploration(self):
        from Jotty.core.intelligence.learning.td_lambda import SkillQTable

        q = SkillQTable(ucb_c=1.41)
        # Give web-search many visits and high Q
        for _ in range(20):
            q.update("research", "web-search", 0.9)
        # Give calculator few visits and moderate Q
        for _ in range(2):
            q.update("research", "calculator", 0.7)

        result = q.select("research", ["web-search", "calculator"])
        # calculator has fewer visits, UCB exploration term is larger
        # With c=1.41, sqrt(ln(22)/2) ≈ 1.24, so calculator UCB ≈ 0.7 + 1.41*1.24 ≈ 2.45
        # web-search UCB ≈ 0.9 + 1.41*sqrt(ln(22)/20) ≈ 0.9 + 0.55 ≈ 1.45
        assert result[0] == "calculator"

    @pytest.mark.unit
    def test_serialization_preserves_ucb_c(self):
        from Jotty.core.intelligence.learning.td_lambda import SkillQTable

        q = SkillQTable(ucb_c=2.0)
        q.update("coding", "python", 0.8)
        data = q.to_dict()
        restored = SkillQTable.from_dict(data)
        assert restored.ucb_c == 2.0
        assert restored.get_q("coding", "python") == q.get_q("coding", "python")


# =============================================================================
# Reflexion
# =============================================================================


class TestReflexion:
    """Test Reflexion failure reflection."""

    @pytest.mark.unit
    def test_reflect_returns_none_when_no_lm(self):
        from Jotty.core.intelligence.learning.advanced_learning import Reflexion

        r = Reflexion()
        r._init_attempts = 10
        result = r.reflect_on_failure("ep1", "agent", "goal", "output")
        assert result is None

    @pytest.mark.unit
    def test_reflect_with_mocked_dspy(self):
        from Jotty.core.intelligence.learning.advanced_learning import Reflexion

        r = Reflexion()
        mock_module = MagicMock()
        mock_module.return_value = MagicMock(
            observation="The API endpoint returned 500",
            analysis="Missing error handling for null input",
            adjustment="Add input validation before API call",
        )
        r._reflect_module = mock_module
        r._lm = MagicMock()

        with (
            patch("dspy.context"),
            patch(
                "Jotty.core.intelligence.learning.learning_store.LearningStore.get_instance"
            ) as mock_get,
        ):
            mock_store = MagicMock()
            mock_get.return_value = mock_store

            result = r.reflect_on_failure(
                "ep1", "CodingSwarm", "Build REST API", "Error 500", "timeout"
            )

        assert result is not None
        assert "validation" in result["adjustment"].lower()
        mock_store.save_reflection.assert_called_once()

    @pytest.mark.unit
    def test_get_relevant_reflections(self):
        from Jotty.core.intelligence.learning.advanced_learning import Reflexion

        r = Reflexion()

        with patch(
            "Jotty.core.intelligence.learning.learning_store.LearningStore.get_instance"
        ) as mock_get:
            mock_store = MagicMock()
            mock_get.return_value = mock_store
            mock_store.get_reflections.return_value = [
                MagicMock(
                    observation="Test failed",
                    adjustment="Add edge case handling",
                ),
            ]

            results = r.get_relevant_reflections("TestAgent", limit=3)

        assert len(results) == 1
        assert "edge case" in results[0]


# =============================================================================
# FewShotCurator
# =============================================================================


class TestFewShotCurator:
    """Test few-shot episode curation."""

    @pytest.mark.unit
    def test_returns_empty_on_no_episodes(self):
        from Jotty.core.intelligence.learning.advanced_learning import FewShotCurator

        curator = FewShotCurator()

        with patch(
            "Jotty.core.intelligence.learning.learning_store.LearningStore.get_instance"
        ) as mock_get:
            mock_store = MagicMock()
            mock_get.return_value = mock_store
            mock_store.query_episodes.return_value = []

            examples = curator.get_examples(domain="coding", n=5)

        assert examples == []

    @pytest.mark.unit
    def test_curates_high_quality_episodes(self):
        from Jotty.core.intelligence.learning.advanced_learning import FewShotCurator

        curator = FewShotCurator()

        mock_ep = MagicMock()
        mock_ep.quality = 0.9
        mock_ep.context = {"task": "Build REST API"}
        mock_ep.action = {"paradigm": "pipeline"}
        mock_ep.outcome = {"content": "Here is the code..."}
        mock_ep.task_type = "coding"

        with patch(
            "Jotty.core.intelligence.learning.learning_store.LearningStore.get_instance"
        ) as mock_get:
            mock_store = MagicMock()
            mock_get.return_value = mock_store
            mock_store.query_episodes.return_value = [mock_ep]

            examples = curator.get_examples(domain="coding", n=5, min_quality=0.7)

        assert len(examples) == 1


# =============================================================================
# VoyagerSkillLib
# =============================================================================


class TestVoyagerSkillLib:
    """Test Voyager-style skill library."""

    @pytest.mark.unit
    def test_skip_low_quality_episodes(self):
        from Jotty.core.intelligence.learning.advanced_learning import VoyagerSkillLib

        lib = VoyagerSkillLib()
        result = lib.extract_skill_pattern(
            "ep1", "coding", "api", "Build API", "pipeline", quality=0.5
        )
        assert result is None

    @pytest.mark.unit
    def test_extract_high_quality_pattern(self):
        from Jotty.core.intelligence.learning.advanced_learning import VoyagerSkillLib

        lib = VoyagerSkillLib()

        with patch(
            "Jotty.core.intelligence.learning.learning_store.LearningStore.get_instance"
        ) as mock_get:
            mock_store = MagicMock()
            mock_get.return_value = mock_store
            mock_store.get_patterns.return_value = []

            result = lib.extract_skill_pattern(
                "ep1", "coding", "api", "Build REST API", "pipeline + test", quality=0.9
            )

        assert result is not None
        assert result.startswith("skill_")
        mock_store.save_pattern.assert_called_once()

    @pytest.mark.unit
    def test_boosts_existing_pattern(self):
        from Jotty.core.intelligence.learning.advanced_learning import VoyagerSkillLib

        lib = VoyagerSkillLib()

        mock_pattern = MagicMock()
        mock_pattern.description = "Build REST API"
        mock_pattern.confidence = 0.8
        mock_pattern.evidence_count = 3
        mock_pattern.pattern_id = "skill_existing"

        with patch(
            "Jotty.core.intelligence.learning.learning_store.LearningStore.get_instance"
        ) as mock_get:
            mock_store = MagicMock()
            mock_get.return_value = mock_store
            mock_store.get_patterns.return_value = [mock_pattern]

            result = lib.extract_skill_pattern(
                "ep2", "coding", "api", "Build REST API", "pipeline", quality=0.85
            )

        assert result == "skill_existing"
        assert abs(mock_pattern.confidence - 0.85) < 1e-9
        assert mock_pattern.evidence_count == 4


# =============================================================================
# Validation Cascade: validate_output + library validators
# =============================================================================


class TestValidationCascade:
    """Test the 4-level verification cascade (library → structural → LLM → human).

    Only L1 (library) and L2 (structural) are tested here — they're free and fast.
    L3 (LLM judge) lives in LearningService and is tested separately.
    """

    @pytest.mark.unit
    def test_validation_result_dataclass(self):
        from Jotty.core.intelligence.learning.advanced_learning import ValidationResult

        r = ValidationResult(valid=True, method="library", confidence=1.0)
        assert r.valid is True
        assert r.issue == ""

        r2 = ValidationResult(
            valid=False,
            method="library",
            confidence=1.0,
            errors=["bad syntax", "missing token"],
        )
        assert r2.valid is False
        assert "bad syntax" in r2.issue
        assert "missing token" in r2.issue

    @pytest.mark.unit
    def test_too_short_output(self):
        from Jotty.core.intelligence.learning.advanced_learning import validate_output

        r = validate_output("hi", "python")
        assert r.valid is False
        assert "too short" in r.issue.lower()

    @pytest.mark.unit
    def test_empty_output(self):
        from Jotty.core.intelligence.learning.advanced_learning import validate_output

        r = validate_output("", "sql")
        assert r.valid is False

    # --- Python (ast.parse) ---

    @pytest.mark.unit
    def test_python_valid(self):
        from Jotty.core.intelligence.learning.advanced_learning import validate_output

        code = "def calculate_total(items):\n    return sum(i.price for i in items)"
        r = validate_output(code, "python")
        assert r.valid is True
        assert r.method == "library"
        assert r.confidence >= 0.9

    @pytest.mark.unit
    def test_python_syntax_error(self):
        from Jotty.core.intelligence.learning.advanced_learning import validate_output

        code = "def calculate_total(items):\n    return sum(i.price for i in items"
        r = validate_output(code, "python")
        assert r.valid is False
        assert r.method == "library"
        assert r.confidence == 1.0
        assert "syntax error" in r.issue.lower()

    # --- SQL (sqlglot) ---

    @pytest.mark.unit
    def test_sql_valid(self):
        from Jotty.core.intelligence.learning.advanced_learning import validate_output

        sql = "SELECT u.name, COUNT(*) FROM users u JOIN orders o ON u.id = o.uid GROUP BY u.name"
        r = validate_output(sql, "sql")
        assert r.valid is True
        assert r.method == "library"

    @pytest.mark.unit
    def test_sql_syntax_error(self):
        from Jotty.core.intelligence.learning.advanced_learning import validate_output

        sql = "SELECTT name FROMM users WHEREE id > 100 AND status = active"
        r = validate_output(sql, "sql")
        assert r.valid is False
        assert r.confidence >= 0.9

    # --- JSON ---

    @pytest.mark.unit
    def test_json_valid(self):
        from Jotty.core.intelligence.learning.advanced_learning import validate_output

        j = '{"users": [{"name": "Alice", "age": 30}], "total": 1}'
        r = validate_output(j, "json")
        assert r.valid is True
        assert r.method == "library"
        assert r.confidence == 1.0

    @pytest.mark.unit
    def test_json_invalid(self):
        from Jotty.core.intelligence.learning.advanced_learning import validate_output

        j = '{"users": [{"name": "Alice", age: 30}], "total": 1}'
        r = validate_output(j, "json")
        assert r.valid is False
        assert r.confidence == 1.0

    # --- YAML ---

    @pytest.mark.unit
    def test_yaml_valid(self):
        from Jotty.core.intelligence.learning.advanced_learning import validate_output

        y = "name: test\nversion: 1.0\nitems:\n  - alpha\n  - beta\n  - gamma"
        r = validate_output(y, "yaml")
        assert r.valid is True
        assert r.method == "library"

    # --- HTML ---

    @pytest.mark.unit
    def test_html_valid(self):
        from Jotty.core.intelligence.learning.advanced_learning import validate_output

        h = "<html><head><title>T</title></head><body><p>Hello</p></body></html>"
        r = validate_output(h, "html")
        assert r.valid is True

    @pytest.mark.unit
    def test_html_unclosed_tag(self):
        from Jotty.core.intelligence.learning.advanced_learning import validate_output

        h = "<html><body><div><p>Unclosed div</p></body></html>"
        r = validate_output(h, "html")
        assert r.valid is False
        assert r.method == "library"

    # --- Mermaid (structural only) ---

    @pytest.mark.unit
    def test_mermaid_valid_flowchart(self):
        from Jotty.core.intelligence.learning.advanced_learning import validate_output

        m = "graph TD\n  A[Start] --> B{Decision}\n  B -->|Yes| C[End]"
        r = validate_output(m, "mermaid")
        assert r.valid is True
        assert r.method == "structural"

    @pytest.mark.unit
    def test_mermaid_missing_type(self):
        from Jotty.core.intelligence.learning.advanced_learning import validate_output

        m = "Some random text about a diagram concept without diagram keywords present"
        r = validate_output(m, "mermaid")
        assert r.valid is False

    # --- PlantUML (structural, server mocked) ---

    @pytest.mark.unit
    def test_plantuml_missing_wrapper(self):
        from Jotty.core.intelligence.learning.advanced_learning import validate_output

        with patch(
            "Jotty.core.intelligence.learning.advanced_learning._validate_plantuml_server",
            return_value=None,
        ):
            text = "class User has name field and login method connecting to database"
            r = validate_output(text, "plantuml")
            assert r.valid is False
            assert "startuml" in r.issue.lower()

    @pytest.mark.unit
    def test_plantuml_valid_with_server(self):
        from Jotty.core.intelligence.learning.advanced_learning import (
            ValidationResult,
            validate_output,
        )

        server_ok = ValidationResult(valid=True, method="library", confidence=0.95)
        with patch(
            "Jotty.core.intelligence.learning.advanced_learning._validate_plantuml_server",
            return_value=server_ok,
        ):
            text = "@startuml\nclass User {\n  +name: String\n}\n@enduml"
            r = validate_output(text, "plantuml")
            assert r.valid is True
            assert r.method == "library"
            assert r.confidence == 0.95

    @pytest.mark.unit
    def test_plantuml_server_rejects(self):
        from Jotty.core.intelligence.learning.advanced_learning import (
            ValidationResult,
            validate_output,
        )

        server_err = ValidationResult(
            valid=False,
            method="library",
            confidence=0.95,
            errors=["PlantUML server rejected diagram (HTTP 400)."],
        )
        with patch(
            "Jotty.core.intelligence.learning.advanced_learning._validate_plantuml_server",
            return_value=server_err,
        ):
            text = "@startuml\nclasss User {{{\n  broken syntax here!!\n}\n@enduml"
            r = validate_output(text, "plantuml")
            assert r.valid is False
            assert r.confidence == 0.95

    # --- Unknown domain (graceful fallback) ---

    @pytest.mark.unit
    def test_unknown_domain_passes_if_long_enough(self):
        from Jotty.core.intelligence.learning.advanced_learning import validate_output

        r = validate_output("This is a long enough text for an unknown domain.", "exotic")
        assert r.valid is True
        assert r.method == "structural"

    # --- Code extraction from markdown fences ---

    @pytest.mark.unit
    def test_code_in_markdown_fence(self):
        from Jotty.core.intelligence.learning.advanced_learning import validate_output

        text = "Here is the code:\n```python\ndef foo():\n    return 42\n```"
        r = validate_output(text, "python")
        assert r.valid is True

    # --- Backward compatibility ---

    @pytest.mark.unit
    def test_check_output_backward_compat(self):
        from Jotty.core.intelligence.learning.advanced_learning import _check_output

        assert _check_output("def foo(): return 42\nprint(foo())", "python") == ""
        assert (
            "syntax error"
            in _check_output(
                "def calculate_total(items):\n    return sum(i.price for i in items",
                "python",
            ).lower()
        )

    # --- Subdomain stripping ---

    @pytest.mark.unit
    def test_subdomain_stripped(self):
        from Jotty.core.intelligence.learning.advanced_learning import validate_output

        r = validate_output(
            "graph TD\n  A --> B\n  B --> C\n  C --> D",
            "mermaid:flowchart",
        )
        assert r.valid is True
