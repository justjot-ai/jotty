"""
MetadataOrchestrationManager - Manages metadata fetching and enrichment.

Extracted from conductor.py to improve maintainability.
Handles direct metadata calls and business term enrichment.
"""
import logging
import time
import inspect
from typing import Dict, Any

logger = logging.getLogger(__name__)


class MetadataOrchestrationManager:
    """
    Centralized metadata orchestration management.

    Responsibilities:
    - Fetch all metadata from providers directly (no ReAct overhead)
    - Enrich business terms with filter definitions
    - Merge filter specs into term data
    - Metadata caching and statistics
    """

    def __init__(self, config, metadata_provider=None, metadata_tool_registry=None):
        """
        Initialize metadata orchestration manager.

        Args:
            config: JottyConfig
            metadata_provider: MetadataProvider instance
            metadata_tool_registry: MetadataToolRegistry instance
        """
        self.config = config
        self.metadata_provider = metadata_provider
        self.metadata_tool_registry = metadata_tool_registry
        self.fetch_count = 0
        self.cache = {}

        logger.info("🗄️  MetadataOrchestrationManager initialized")

    def fetch_all_metadata_directly(self) -> Dict[str, Any]:
        """
        Fetch ALL metadata by calling all @jotty_method methods directly.

        No ReAct agent overhead, no guessing, no missing data!

        Returns:
            Dict with ALL metadata using original method names as keys
        """
        if not self.metadata_provider:
            logger.warning("⚠️  No metadata_provider available")
            return {}

        self.fetch_count += 1
        logger.info("🔍 Fetching ALL metadata directly (no ReAct agent, no guessing)...")
        metadata = {}
        start_time = time.time()

        # Known metadata methods (common across implementations)
        known_methods = [
            'get_all_business_contexts',
            'get_all_table_metadata',
            'get_all_filter_definitions',
            'get_all_column_metadata',
            'get_all_term_definitions',
            'get_all_validations'
        ]

        # Call known methods
        for method_name in known_methods:
            if hasattr(self.metadata_provider, method_name):
                try:
                    logger.debug(f"   📞 Calling {method_name}()...")
                    result = getattr(self.metadata_provider, method_name)()
                    metadata[method_name] = result
                    logger.info(f"   ✅ {method_name}: {len(str(result))} chars")
                except Exception as e:
                    logger.warning(f"   ⚠️  {method_name}() failed: {e}")

        # Discover and call additional methods from tool registry
        if self.metadata_tool_registry:
            for tool_name in self.metadata_tool_registry.tools.keys():
                if tool_name not in known_methods:
                    try:
                        if hasattr(self.metadata_provider, tool_name):
                            # Skip methods requiring positional args
                            sig = inspect.signature(getattr(self.metadata_provider, tool_name))
                            required_positional = [
                                p.name for p in sig.parameters.values()
                                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty
                            ]
                            if required_positional:
                                logger.debug(f"   ⏭️  Skipping {tool_name}() (requires args: {required_positional})")
                                continue

                            logger.debug(f"   📞 Calling {tool_name}()...")
                            result = getattr(self.metadata_provider, tool_name)()
                            metadata[tool_name] = result
                            logger.info(f"   ✅ {tool_name}: {len(str(result))} chars")
                    except Exception as e:
                        logger.warning(f"   ⚠️  {tool_name}() failed: {e}")

        elapsed = time.time() - start_time
        logger.info(f"✅ Fetched {len(metadata)} metadata items in {elapsed:.2f}s (direct calls, no LLM!)")

        # Cache result
        self.cache['last_fetch'] = metadata
        self.cache['last_fetch_time'] = time.time()

        return metadata

    def enrich_business_terms_with_filters(self, fetched_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich business_terms with parsed filter conditions.

        Merges filter definitions into business term data for agent context.

        Args:
            fetched_data: Dict from fetch_all_metadata_directly()

        Returns:
            Enriched metadata dict
        """
        # Check if we have both business contexts and filter definitions
        business_contexts = fetched_data.get('get_all_business_contexts', {})
        filter_defs = fetched_data.get('get_all_filter_definitions', {})

        if not business_contexts or not filter_defs:
            logger.debug("No business contexts or filters to enrich")
            return fetched_data

        logger.info("🔧 Enriching business terms with filter definitions...")

        # Enrich each term with its filter info
        enriched_contexts = {}
        for term_name, term_data in business_contexts.items():
            # Create enriched copy
            enriched_term = dict(term_data) if isinstance(term_data, dict) else {'value': term_data}

            # Find matching filter
            if term_name in filter_defs:
                filter_spec = filter_defs[term_name]
                enriched_term['filter'] = filter_spec
                logger.debug(f"   ✅ Enriched '{term_name}' with filter")

            enriched_contexts[term_name] = enriched_term

        # Update fetched_data
        fetched_data['get_all_business_contexts'] = enriched_contexts
        logger.info(f"✅ Enriched {len(enriched_contexts)} business terms")

        return fetched_data

    def merge_filter_into_term(self, term_data: Any, filter_spec: Dict[str, Any], term_name: str) -> Dict[str, Any]:
        """
        Merge filter specification into term data.

        Args:
            term_data: Business term data
            filter_spec: Filter definition
            term_name: Name of the term

        Returns:
            Merged dict with term + filter info
        """
        # Ensure term_data is a dict
        if not isinstance(term_data, dict):
            term_data = {'value': term_data}

        # Add filter spec
        merged = dict(term_data)
        merged['filter'] = filter_spec

        logger.debug(f"Merged filter into '{term_name}'")
        return merged

    def get_cached_metadata(self) -> Dict[str, Any]:
        """
        Get cached metadata from last fetch.

        Returns:
            Cached metadata dict or empty dict
        """
        return self.cache.get('last_fetch', {})

    def get_stats(self) -> Dict[str, Any]:
        """
        Get metadata orchestration statistics.

        Returns:
            Dict with metadata metrics
        """
        cache_age = time.time() - self.cache.get('last_fetch_time', 0) if 'last_fetch_time' in self.cache else None
        return {
            "total_fetches": self.fetch_count,
            "cache_size": len(self.cache.get('last_fetch', {})),
            "cache_age_seconds": cache_age
        }

    def clear_cache(self):
        """Clear metadata cache."""
        self.cache.clear()
        logger.debug("Metadata cache cleared")
