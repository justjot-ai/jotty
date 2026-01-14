"""
Test ParameterResolver Component in Isolation
==============================================

Tests the extracted ParameterResolver component independently
before integrating it into Conductor.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.unit
class TestParameterResolverIsolation:
    """Test ParameterResolver works in isolation."""

    def test_can_import_parameter_resolver(self):
        """Test ParameterResolver can be imported."""
        from core.orchestration.parameter_resolver import ParameterResolver
        assert ParameterResolver is not None

    def test_can_create_parameter_resolver_instance(self):
        """Test ParameterResolver can be instantiated with mocks."""
        from core.orchestration.parameter_resolver import ParameterResolver

        # Create mock dependencies
        mock_io_manager = Mock()
        mock_param_resolver = Mock()
        mock_metadata_fetcher = Mock()
        mock_actors = {}
        mock_actor_signatures = {}
        mock_param_mappings = {}
        mock_data_registry = Mock()
        mock_registration_orchestrator = Mock()
        mock_data_transformer = Mock()
        mock_shared_context = Mock()
        mock_config = Mock()

        # Create instance
        resolver = ParameterResolver(
            io_manager=mock_io_manager,
            param_resolver=mock_param_resolver,
            metadata_fetcher=mock_metadata_fetcher,
            actors=mock_actors,
            actor_signatures=mock_actor_signatures,
            param_mappings=mock_param_mappings,
            data_registry=mock_data_registry,
            registration_orchestrator=mock_registration_orchestrator,
            data_transformer=mock_data_transformer,
            shared_context=mock_shared_context,
            config=mock_config
        )

        assert resolver is not None
        assert resolver.io_manager == mock_io_manager
        assert resolver.config == mock_config

    def test_resolve_param_from_iomanager_returns_none_when_no_manager(self):
        """Test _resolve_param_from_iomanager returns None without io_manager."""
        from core.orchestration.parameter_resolver import ParameterResolver

        # Create resolver with None io_manager
        resolver = ParameterResolver(
            io_manager=None,
            param_resolver=Mock(),
            metadata_fetcher=Mock(),
            actors={},
            actor_signatures={},
            param_mappings={},
            data_registry=Mock(),
            registration_orchestrator=Mock(),
            data_transformer=Mock(),
            shared_context=Mock(),
            config=Mock()
        )

        result = resolver._resolve_param_from_iomanager("test_param")
        assert result is None

    def test_resolve_param_from_iomanager_finds_value(self):
        """Test _resolve_param_from_iomanager finds parameter in outputs."""
        from core.orchestration.parameter_resolver import ParameterResolver

        # Create mock io_manager with outputs
        mock_io_manager = Mock()
        mock_output = Mock()
        mock_output.output_fields = {"test_param": "test_value"}
        mock_io_manager.get_all_outputs.return_value = {
            "actor1": mock_output
        }

        resolver = ParameterResolver(
            io_manager=mock_io_manager,
            param_resolver=Mock(),
            metadata_fetcher=Mock(),
            actors={},
            actor_signatures={},
            param_mappings={},
            data_registry=Mock(),
            registration_orchestrator=Mock(),
            data_transformer=Mock(),
            shared_context=Mock(),
            config=Mock()
        )

        result = resolver._resolve_param_from_iomanager("test_param")
        assert result == "test_value"

    def test_resolve_param_by_type_matches_string_type(self):
        """Test _resolve_param_by_type matches by type."""
        from core.orchestration.parameter_resolver import ParameterResolver

        # Create mock io_manager with typed outputs
        mock_io_manager = Mock()
        mock_output = Mock()
        mock_output.output_fields = {
            "some_field": "string_value",
            "other_field": 123
        }
        mock_io_manager.get_all_outputs.return_value = {
            "actor1": mock_output
        }

        resolver = ParameterResolver(
            io_manager=mock_io_manager,
            param_resolver=Mock(),
            metadata_fetcher=Mock(),
            actors={},
            actor_signatures={},
            param_mappings={},
            data_registry=Mock(),
            registration_orchestrator=Mock(),
            data_transformer=Mock(),
            shared_context=Mock(),
            config=Mock()
        )

        result = resolver._resolve_param_by_type("query", "str")
        assert result == "string_value"

    def test_resolve_param_by_type_returns_none_for_unknown_type(self):
        """Test _resolve_param_by_type returns None for unknown types."""
        from core.orchestration.parameter_resolver import ParameterResolver

        resolver = ParameterResolver(
            io_manager=Mock(),
            param_resolver=Mock(),
            metadata_fetcher=Mock(),
            actors={},
            actor_signatures={},
            param_mappings={},
            data_registry=Mock(),
            registration_orchestrator=Mock(),
            data_transformer=Mock(),
            shared_context=Mock(),
            config=Mock()
        )

        result = resolver._resolve_param_by_type("param", "UnknownType")
        assert result is None


@pytest.mark.unit
class TestParameterResolverMethods:
    """Test individual ParameterResolver methods."""

    def test_has_all_expected_methods(self):
        """Test ParameterResolver has all expected methods."""
        from core.orchestration.parameter_resolver import ParameterResolver

        # Check all methods exist
        assert hasattr(ParameterResolver, '_resolve_param_from_iomanager')
        assert hasattr(ParameterResolver, '_resolve_param_by_type')
        assert hasattr(ParameterResolver, '_build_param_mappings')
        assert hasattr(ParameterResolver, '_find_parameter_producer')
        assert hasattr(ParameterResolver, 'resolve_input')
        assert hasattr(ParameterResolver, '_resolve_parameter')
        assert hasattr(ParameterResolver, '_extract_from_metadata_manager')
        assert hasattr(ParameterResolver, '_semantic_extract')
        assert hasattr(ParameterResolver, '_llm_match_field')
        assert hasattr(ParameterResolver, '_extract_from_output')


def run_parameter_resolver_tests():
    """Run all ParameterResolver tests."""
    print("="*70)
    print("PARAMETER RESOLVER ISOLATION TESTS")
    print("="*70)

    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short"
    ])

    return exit_code


if __name__ == "__main__":
    exit_code = run_parameter_resolver_tests()
    sys.exit(exit_code)
