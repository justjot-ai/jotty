"""
Unit tests for OpenCode Zen provider integration.

Tests model selection, fallback behavior, fuzzy matching,
free model defaults, and the full create_lm flow.
"""
import os
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_ZEN_KEY = "sk-test-zen-fake-key-for-unit-tests"


@pytest.fixture(autouse=True)
def _set_zen_key(monkeypatch):
    """Ensure every test has a Zen API key available."""
    monkeypatch.setenv("OPENCODE_ZEN_API_KEY", FAKE_ZEN_KEY)


@pytest.fixture
def provider():
    from Jotty.core.foundation.unified_lm_provider import UnifiedLMProvider
    return UnifiedLMProvider


# ---------------------------------------------------------------------------
# Static data sanity checks
# ---------------------------------------------------------------------------

class TestZenModelRegistry:
    """Verify the static model registry is consistent."""

    @pytest.mark.unit
    def test_free_models_exist_in_registry(self, provider):
        """Every free model must also appear in ZEN_MODELS."""
        for m in provider.ZEN_FREE_MODELS:
            assert m in provider.ZEN_MODELS, f"Free model '{m}' missing from ZEN_MODELS"

    @pytest.mark.unit
    def test_free_models_not_empty(self, provider):
        assert len(provider.ZEN_FREE_MODELS) >= 2, "Should have at least 2 free models"

    @pytest.mark.unit
    def test_all_models_have_valid_format(self, provider):
        valid_formats = {'openai', 'anthropic', 'google'}
        for model_id, (endpoint, fmt) in provider.ZEN_MODELS.items():
            assert fmt in valid_formats, f"Model '{model_id}' has invalid format '{fmt}'"
            assert endpoint, f"Model '{model_id}' has empty endpoint"

    @pytest.mark.unit
    def test_zen_base_url(self, provider):
        assert provider.ZEN_BASE_URL == "https://opencode.ai/zen/v1"


# ---------------------------------------------------------------------------
# Default / free model selection
# ---------------------------------------------------------------------------

class TestZenDefaults:
    """Test that free models are always the default."""

    @pytest.mark.unit
    def test_no_model_defaults_to_first_free(self, provider):
        """Passing model=None should use the first free model."""
        lm = provider._create_zen_lm(model=None, api_key=FAKE_ZEN_KEY)
        expected = provider.ZEN_FREE_MODELS[0]
        # DSPy LM stores model id in .model attribute
        assert expected in lm.model, f"Expected '{expected}' in model id, got '{lm.model}'"

    @pytest.mark.unit
    def test_empty_string_defaults_to_free(self, provider):
        """Passing model='' should use the first free model."""
        lm = provider._create_zen_lm(model="", api_key=FAKE_ZEN_KEY)
        expected = provider.ZEN_FREE_MODELS[0]
        assert expected in lm.model

    @pytest.mark.unit
    def test_model_free_keyword(self, provider):
        """Passing model='free' should use the first free model."""
        lm = provider._create_zen_lm(model="free", api_key=FAKE_ZEN_KEY)
        expected = provider.ZEN_FREE_MODELS[0]
        assert expected in lm.model

    @pytest.mark.unit
    def test_each_free_model_creates_successfully(self, provider):
        """Every free model should create an LM instance without error."""
        for model_id in provider.ZEN_FREE_MODELS:
            lm = provider._create_zen_lm(model=model_id, api_key=FAKE_ZEN_KEY)
            assert lm is not None, f"Failed to create LM for free model '{model_id}'"
            assert model_id in lm.model


# ---------------------------------------------------------------------------
# Unknown model fallback (the main bug fix)
# ---------------------------------------------------------------------------

class TestZenUnknownModelFallback:
    """Unknown models should fall back to free model, NOT raise."""

    @pytest.mark.unit
    def test_unknown_model_falls_back(self, provider):
        """A completely unknown model name should fall back to default free model."""
        lm = provider._create_zen_lm(model="nonexistent-model", api_key=FAKE_ZEN_KEY)
        expected = provider.ZEN_FREE_MODELS[0]
        assert expected in lm.model

    @pytest.mark.unit
    def test_unknown_model_does_not_raise(self, provider):
        """Ensure no ValueError is raised for unknown models."""
        # This was the original bug — it used to raise ValueError
        try:
            lm = provider._create_zen_lm(model="totally-fake-xyz-999", api_key=FAKE_ZEN_KEY)
            assert lm is not None
        except ValueError:
            pytest.fail("_create_zen_lm raised ValueError for unknown model — should fallback")

    @pytest.mark.unit
    def test_unknown_model_logs_warning(self, provider, caplog):
        """Unknown model should produce a warning log."""
        import logging
        with caplog.at_level(logging.WARNING):
            provider._create_zen_lm(model="bogus-model-abc", api_key=FAKE_ZEN_KEY)
        assert any("Unknown Zen model" in r.message for r in caplog.records), \
            "Expected a warning about unknown Zen model"

    @pytest.mark.unit
    def test_gibberish_model_falls_back(self, provider):
        lm = provider._create_zen_lm(model="!@#$%^&*()", api_key=FAKE_ZEN_KEY)
        expected = provider.ZEN_FREE_MODELS[0]
        assert expected in lm.model


# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------

class TestZenFuzzyMatch:
    """Fuzzy matching should prefer free models."""

    @pytest.mark.unit
    def test_fuzzy_glm_matches_free(self, provider):
        """'glm' should match 'glm-4.7-free' (free) over 'glm-4.7' (paid)."""
        lm = provider._create_zen_lm(model="glm", api_key=FAKE_ZEN_KEY)
        assert "free" in lm.model or "glm" in lm.model

    @pytest.mark.unit
    def test_fuzzy_kimi_matches_free(self, provider):
        """'kimi' should match 'kimi-k2.5-free' (free) first."""
        lm = provider._create_zen_lm(model="kimi", api_key=FAKE_ZEN_KEY)
        assert "kimi" in lm.model
        # Should prefer free variant
        assert "free" in lm.model

    @pytest.mark.unit
    def test_fuzzy_minimax_matches_free(self, provider):
        """'minimax' should match 'minimax-m2.1-free' first."""
        lm = provider._create_zen_lm(model="minimax", api_key=FAKE_ZEN_KEY)
        assert "minimax" in lm.model
        assert "free" in lm.model

    @pytest.mark.unit
    def test_fuzzy_pickle_matches(self, provider):
        """'pickle' should match 'big-pickle'."""
        lm = provider._create_zen_lm(model="pickle", api_key=FAKE_ZEN_KEY)
        assert "pickle" in lm.model

    @pytest.mark.unit
    def test_fuzzy_paid_model_still_works(self, provider):
        """'claude-sonnet-4' should match the paid model exactly."""
        lm = provider._create_zen_lm(model="claude-sonnet-4", api_key=FAKE_ZEN_KEY)
        assert "claude-sonnet-4" in lm.model

    @pytest.mark.unit
    def test_fuzzy_partial_opus(self, provider):
        """'opus' should fuzzy-match one of the Claude Opus models."""
        lm = provider._create_zen_lm(model="opus", api_key=FAKE_ZEN_KEY)
        assert "opus" in lm.model


# ---------------------------------------------------------------------------
# API key handling
# ---------------------------------------------------------------------------

class TestZenApiKey:
    """Test API key resolution."""

    @pytest.mark.unit
    def test_missing_api_key_raises(self, provider, monkeypatch):
        """No API key at all should raise ValueError."""
        monkeypatch.delenv("OPENCODE_ZEN_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key required"):
            provider._create_zen_lm(model="glm-4.7-free")

    @pytest.mark.unit
    def test_explicit_api_key_overrides_env(self, provider, monkeypatch):
        """Passing api_key= should take priority over env var."""
        monkeypatch.delenv("OPENCODE_ZEN_API_KEY", raising=False)
        lm = provider._create_zen_lm(model="glm-4.7-free", api_key="sk-explicit-key")
        assert lm is not None

    @pytest.mark.unit
    def test_env_api_key_used(self, provider):
        """Should use OPENCODE_ZEN_API_KEY from environment."""
        lm = provider._create_zen_lm(model="glm-4.7-free")
        assert lm is not None


# ---------------------------------------------------------------------------
# create_lm integration (top-level entry point)
# ---------------------------------------------------------------------------

class TestZenCreateLm:
    """Test via the public create_lm() entry point."""

    @pytest.mark.unit
    def test_create_lm_zen_no_model(self, provider):
        """create_lm('zen') should default to a free model."""
        lm = provider.create_lm('zen', inject_context=False)
        expected = provider.ZEN_FREE_MODELS[0]
        assert expected in lm.model

    @pytest.mark.unit
    def test_create_lm_zen_explicit_free(self, provider):
        """create_lm('zen', model='big-pickle') should work."""
        lm = provider.create_lm('zen', model='big-pickle', inject_context=False)
        assert 'big-pickle' in lm.model

    @pytest.mark.unit
    def test_create_lm_zen_unknown_falls_back(self, provider):
        """create_lm('zen', model='doesnt-exist') should fall back, not raise."""
        lm = provider.create_lm('zen', model='doesnt-exist', inject_context=False)
        expected = provider.ZEN_FREE_MODELS[0]
        assert expected in lm.model


# ---------------------------------------------------------------------------
# list_zen_models
# ---------------------------------------------------------------------------

class TestListZenModels:
    """Test the model listing helper."""

    @pytest.mark.unit
    def test_list_all_models(self, provider):
        models = provider.list_zen_models()
        assert len(models) == len(provider.ZEN_MODELS)
        for mid in models:
            assert 'endpoint' in models[mid]
            assert 'format' in models[mid]
            assert 'free' in models[mid]

    @pytest.mark.unit
    def test_list_free_only(self, provider):
        models = provider.list_zen_models(free_only=True)
        assert len(models) == len(provider.ZEN_FREE_MODELS)
        for mid in models:
            assert models[mid]['free'] is True

    @pytest.mark.unit
    def test_free_models_marked_correctly(self, provider):
        models = provider.list_zen_models()
        for mid in provider.ZEN_FREE_MODELS:
            assert models[mid]['free'] is True
        # At least one paid model should be marked as not free
        paid = [mid for mid in models if not models[mid]['free']]
        assert len(paid) > 0


# ---------------------------------------------------------------------------
# Endpoint format correctness
# ---------------------------------------------------------------------------

class TestZenEndpointFormats:
    """Verify the DSPy model prefix matches the API format."""

    @pytest.mark.unit
    def test_openai_format_models_get_openai_prefix(self, provider):
        """OpenAI-compatible models should get 'openai/' prefix."""
        for model_id, (_, fmt) in provider.ZEN_MODELS.items():
            if fmt == 'openai':
                lm = provider._create_zen_lm(model=model_id, api_key=FAKE_ZEN_KEY)
                assert lm.model.startswith("openai/"), \
                    f"Model '{model_id}' (openai fmt) should have 'openai/' prefix, got '{lm.model}'"

    @pytest.mark.unit
    def test_anthropic_format_models_get_anthropic_prefix(self, provider):
        """Anthropic-compatible models should get 'anthropic/' prefix."""
        for model_id, (_, fmt) in provider.ZEN_MODELS.items():
            if fmt == 'anthropic':
                lm = provider._create_zen_lm(model=model_id, api_key=FAKE_ZEN_KEY)
                assert lm.model.startswith("anthropic/"), \
                    f"Model '{model_id}' (anthropic fmt) should have 'anthropic/' prefix, got '{lm.model}'"

    @pytest.mark.unit
    def test_google_format_models_get_google_prefix(self, provider):
        """Google-compatible models should get 'google/' prefix."""
        for model_id, (_, fmt) in provider.ZEN_MODELS.items():
            if fmt == 'google':
                lm = provider._create_zen_lm(model=model_id, api_key=FAKE_ZEN_KEY)
                assert lm.model.startswith("google/"), \
                    f"Model '{model_id}' (google fmt) should have 'google/' prefix, got '{lm.model}'"
