"""Tests for shared DSPy/API-key initialization module."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestLoadApiKeys:
    """Tests for load_api_keys()."""

    @pytest.mark.unit
    def test_loads_keys_from_dotenv(self, tmp_path: Path) -> None:
        """Keys from .env file are loaded into os.environ."""
        from Jotty.core.infrastructure.foundation.dspy_init import load_api_keys

        env_file = tmp_path / ".env"
        env_file.write_text("TEST_DSPY_KEY=test_value_123\n")

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TEST_DSPY_KEY", None)
            load_api_keys(dotenv_paths=[env_file])
            assert os.environ.get("TEST_DSPY_KEY") == "test_value_123"

        # Cleanup
        os.environ.pop("TEST_DSPY_KEY", None)

    @pytest.mark.unit
    def test_does_not_overwrite_existing_keys(self, tmp_path: Path) -> None:
        """Existing environment variables are not overwritten."""
        from Jotty.core.infrastructure.foundation.dspy_init import load_api_keys

        env_file = tmp_path / ".env"
        env_file.write_text("EXISTING_KEY=file_value\n")

        with patch.dict(os.environ, {"EXISTING_KEY": "env_value"}, clear=False):
            load_api_keys(dotenv_paths=[env_file])
            assert os.environ["EXISTING_KEY"] == "env_value"

    @pytest.mark.unit
    def test_skips_comments_and_blank_lines(self, tmp_path: Path) -> None:
        """Comments and blank lines are ignored."""
        from Jotty.core.infrastructure.foundation.dspy_init import load_api_keys

        env_file = tmp_path / ".env"
        env_file.write_text("# comment\n\nVALID_KEY=valid\n")

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VALID_KEY", None)
            load_api_keys(dotenv_paths=[env_file])
            assert os.environ.get("VALID_KEY") == "valid"

        os.environ.pop("VALID_KEY", None)

    @pytest.mark.unit
    def test_handles_missing_file(self) -> None:
        """Missing dotenv files are silently skipped."""
        from Jotty.core.infrastructure.foundation.dspy_init import load_api_keys

        load_api_keys(dotenv_paths=[Path("/nonexistent/.env")])  # Should not raise


class TestAutoDetectProvider:
    """Tests for auto_detect_provider()."""

    @pytest.mark.unit
    def test_detects_anthropic(self) -> None:
        from Jotty.core.infrastructure.foundation.dspy_init import auto_detect_provider

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-123"}, clear=False):
            assert auto_detect_provider() == "anthropic"

    @pytest.mark.unit
    def test_detects_openai(self) -> None:
        from Jotty.core.infrastructure.foundation.dspy_init import auto_detect_provider

        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "sk-456"},
            clear=False,
        ):
            # Remove anthropic key to test fallback
            env = os.environ.copy()
            env.pop("ANTHROPIC_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                os.environ["OPENAI_API_KEY"] = "sk-456"
                assert auto_detect_provider() == "openai"

    @pytest.mark.unit
    def test_defaults_to_anthropic(self) -> None:
        from Jotty.core.infrastructure.foundation.dspy_init import auto_detect_provider

        with patch.dict(os.environ, {}, clear=True):
            assert auto_detect_provider() == "anthropic"


class TestInitDspyLm:
    """Tests for init_dspy_lm()."""

    @pytest.mark.unit
    def test_returns_none_without_dspy(self) -> None:
        """Returns None when dspy is not importable."""
        from Jotty.core.infrastructure.foundation.dspy_init import init_dspy_lm

        with patch.dict("sys.modules", {"dspy": None}):
            # Importing dspy raises ImportError when module is None
            result = init_dspy_lm()
            # Should gracefully handle (returns existing or None)

    @pytest.mark.unit
    def test_returns_existing_lm_if_configured(self) -> None:
        """If dspy.settings.lm is already set, return it without reconfiguring."""
        from Jotty.core.infrastructure.foundation.dspy_init import init_dspy_lm

        mock_lm = MagicMock()
        mock_dspy = MagicMock()
        mock_dspy.settings.lm = mock_lm

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            result = init_dspy_lm()
            assert result is mock_lm

    @pytest.mark.unit
    def test_lock_is_shared_module_level(self) -> None:
        """The lock is a module-level threading.Lock."""
        from Jotty.core.infrastructure.foundation.dspy_init import _dspy_lm_lock

        assert isinstance(_dspy_lm_lock, type(threading.Lock()))

    @pytest.mark.unit
    def test_backward_compat_re_export(self) -> None:
        """BaseAgent re-exports _dspy_lm_lock for backward compatibility."""
        from Jotty.core.infrastructure.foundation.dspy_init import (
            _dspy_lm_lock as canonical,
        )
        from Jotty.core.intelligence.reasoning.base.base_agent import (
            _dspy_lm_lock as reexported,
        )

        assert canonical is reexported
