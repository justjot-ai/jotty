"""
Integration test for MetadataOrchestrationManager (Phase 2.6).

Tests:
- Direct metadata fetching (no ReAct overhead)
- Business term enrichment with filters
- Metadata caching
- Statistics tracking
"""
import sys
from pathlib import Path
import time

# Add Jotty to path
jotty_root = Path(__file__).parent
sys.path.insert(0, str(jotty_root))

from core.orchestration.managers.metadata_orchestration_manager import MetadataOrchestrationManager
from core.foundation.data_structures import JottyConfig


class MockMetadataProvider:
    """Mock MetadataProvider for testing."""
    def __init__(self):
        self.call_counts = {}

    def get_all_business_contexts(self):
        """Mock business contexts."""
        self.call_counts['get_all_business_contexts'] = self.call_counts.get('get_all_business_contexts', 0) + 1
        return {
            "customer_segment": {"description": "Customer segmentation"},
            "product_category": {"description": "Product categorization"}
        }

    def get_all_filter_definitions(self):
        """Mock filter definitions."""
        self.call_counts['get_all_filter_definitions'] = self.call_counts.get('get_all_filter_definitions', 0) + 1
        return {
            "customer_segment": {
                "column": "segment",
                "operator": "IN",
                "values": ["premium", "standard"]
            },
            "product_category": {
                "column": "category",
                "operator": "=",
                "value": "electronics"
            }
        }

    def get_all_table_metadata(self):
        """Mock table metadata."""
        self.call_counts['get_all_table_metadata'] = self.call_counts.get('get_all_table_metadata', 0) + 1
        return {
            "customers": {"columns": ["id", "name", "segment"]},
            "products": {"columns": ["id", "name", "category"]}
        }

    def get_all_column_metadata(self):
        """Mock column metadata."""
        self.call_counts['get_all_column_metadata'] = self.call_counts.get('get_all_column_metadata', 0) + 1
        return {
            "customers.id": {"type": "integer", "primary_key": True},
            "customers.name": {"type": "string"}
        }


class MockToolRegistry:
    """Mock MetadataToolRegistry for testing."""
    def __init__(self):
        self.tools = {
            "get_all_business_contexts": {},
            "get_all_filter_definitions": {},
            "get_all_table_metadata": {},
            "get_all_column_metadata": {}
        }


def test_fetch_all_metadata_directly():
    """Test direct metadata fetching (no ReAct overhead)."""
    print("\n" + "="*70)
    print("TEST 1: Direct Metadata Fetching")
    print("="*70)

    config = JottyConfig()
    provider = MockMetadataProvider()
    manager = MetadataOrchestrationManager(config, provider)

    # Fetch all metadata
    metadata = manager.fetch_all_metadata_directly()

    print(f"✅ Fetched {len(metadata)} metadata categories")
    assert 'get_all_business_contexts' in metadata
    assert 'get_all_filter_definitions' in metadata
    assert 'get_all_table_metadata' in metadata
    assert 'get_all_column_metadata' in metadata

    # Verify each category has data
    print(f"   • Business contexts: {len(metadata['get_all_business_contexts'])} items")
    print(f"   • Filter definitions: {len(metadata['get_all_filter_definitions'])} items")
    print(f"   • Table metadata: {len(metadata['get_all_table_metadata'])} items")
    print(f"   • Column metadata: {len(metadata['get_all_column_metadata'])} items")

    assert len(metadata['get_all_business_contexts']) == 2
    assert len(metadata['get_all_filter_definitions']) == 2

    print("✅ TEST PASSED: Direct metadata fetching works")


def test_business_term_enrichment():
    """Test business term enrichment with filter definitions."""
    print("\n" + "="*70)
    print("TEST 2: Business Term Enrichment")
    print("="*70)

    config = JottyConfig()
    provider = MockMetadataProvider()
    manager = MetadataOrchestrationManager(config, provider)

    # Fetch metadata
    metadata = manager.fetch_all_metadata_directly()

    # Enrich business terms with filters
    enriched = manager.enrich_business_terms_with_filters(metadata)

    print(f"✅ Enriched {len(enriched['get_all_business_contexts'])} business terms")

    # Check customer_segment enrichment
    customer_segment = enriched['get_all_business_contexts']['customer_segment']
    print(f"   • customer_segment: {customer_segment}")

    assert 'filter' in customer_segment
    assert customer_segment['filter']['column'] == "segment"
    assert customer_segment['filter']['operator'] == "IN"

    # Check product_category enrichment
    product_category = enriched['get_all_business_contexts']['product_category']
    print(f"   • product_category: {product_category}")

    assert 'filter' in product_category
    assert product_category['filter']['column'] == "category"

    print("✅ TEST PASSED: Business terms enriched with filters")


def test_metadata_caching():
    """Test metadata caching."""
    print("\n" + "="*70)
    print("TEST 3: Metadata Caching")
    print("="*70)

    config = JottyConfig()
    provider = MockMetadataProvider()
    manager = MetadataOrchestrationManager(config, provider)

    # First fetch
    metadata1 = manager.fetch_all_metadata_directly()
    call_count_1 = provider.call_counts.get('get_all_business_contexts', 0)

    # Second fetch (should still call provider, but cache should be updated)
    metadata2 = manager.fetch_all_metadata_directly()
    call_count_2 = provider.call_counts.get('get_all_business_contexts', 0)

    print(f"✅ First fetch: {call_count_1} provider calls")
    print(f"✅ Second fetch: {call_count_2} provider calls")
    assert call_count_2 == call_count_1 + 1  # Called again

    # Check cached metadata
    cached = manager.get_cached_metadata()
    print(f"✅ Cached metadata has {len(cached)} categories")
    assert 'get_all_business_contexts' in cached

    print("✅ TEST PASSED: Metadata caching works")


def test_cache_clearing():
    """Test cache clearing."""
    print("\n" + "="*70)
    print("TEST 4: Cache Clearing")
    print("="*70)

    config = JottyConfig()
    provider = MockMetadataProvider()
    manager = MetadataOrchestrationManager(config, provider)

    # Fetch metadata to populate cache
    manager.fetch_all_metadata_directly()

    # Verify cache has data
    cached_before = manager.get_cached_metadata()
    print(f"✅ Cache before clear: {len(cached_before)} items")
    assert len(cached_before) > 0

    # Clear cache
    manager.clear_cache()
    cached_after = manager.get_cached_metadata()
    print(f"✅ Cache after clear: {len(cached_after)} items")
    assert len(cached_after) == 0

    print("✅ TEST PASSED: Cache clearing works")


def test_statistics_tracking():
    """Test statistics tracking."""
    print("\n" + "="*70)
    print("TEST 5: Statistics Tracking")
    print("="*70)

    config = JottyConfig()
    provider = MockMetadataProvider()
    manager = MetadataOrchestrationManager(config, provider)

    # Initial stats
    stats = manager.get_stats()
    print(f"📊 Initial stats: {stats}")
    assert stats['total_fetches'] == 0

    # Fetch metadata
    manager.fetch_all_metadata_directly()
    time.sleep(0.1)  # Small delay to ensure cache age is measurable

    # Check stats after fetch
    stats = manager.get_stats()
    print(f"📊 Stats after fetch: {stats}")
    assert stats['total_fetches'] == 1
    assert stats['cache_size'] == 4  # 4 metadata categories
    assert stats['cache_age_seconds'] is not None
    assert stats['cache_age_seconds'] >= 0

    # Another fetch
    manager.fetch_all_metadata_directly()
    stats = manager.get_stats()
    print(f"📊 Stats after 2nd fetch: {stats}")
    assert stats['total_fetches'] == 2

    print("✅ TEST PASSED: Statistics tracking works")


def test_no_provider_handling():
    """Test graceful handling when no provider is available."""
    print("\n" + "="*70)
    print("TEST 6: No Provider Handling")
    print("="*70)

    config = JottyConfig()
    manager = MetadataOrchestrationManager(config, metadata_provider=None)

    # Fetch should return empty dict
    metadata = manager.fetch_all_metadata_directly()
    print(f"✅ Metadata without provider: {metadata}")
    assert metadata == {}

    print("✅ TEST PASSED: Gracefully handles missing provider")


def run_all_tests():
    """Run all metadata orchestration tests."""
    print("\n" + "🧪 "*35)
    print("METADATA ORCHESTRATION MANAGER INTEGRATION TESTS (Phase 2.6)")
    print("🧪 "*35)

    try:
        test_fetch_all_metadata_directly()
        test_business_term_enrichment()
        test_metadata_caching()
        test_cache_clearing()
        test_statistics_tracking()
        test_no_provider_handling()

        print("\n" + "✅ "*35)
        print("ALL METADATA ORCHESTRATION MANAGER TESTS PASSED!")
        print("✅ "*35)
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
