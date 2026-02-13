"""
MetaDataFetcher - Production-Grade Proactive Metadata Fetching

 SOTA DESIGN PRINCIPLES:
- Automatic @jotty_method discovery via introspection
- DSPy ReAct agent with intelligent tool selection
- Caching with TTL for performance
- Comprehensive error handling and retry logic
- Detailed logging for debugging
- Thread-safe operations
- Performance monitoring and metrics
- Fallback strategies for robustness

Architecture:
1. Tool Discovery: Auto-discovers @jotty_method decorated methods
2. Tool Conversion: Wraps methods as dspy.Tool objects
3. ReAct Agent: Uses DSPy ReAct for intelligent metadata fetching
4. Result Caching: Caches results with configurable TTL
5. Error Recovery: Automatic retries with exponential backoff
"""

import dspy
import logging
import time
import json
import hashlib
import threading
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import inspect
import traceback

logger = logging.getLogger(__name__)


@dataclass
class ToolMetadata:
    """Metadata about a discovered tool."""
    name: str
    func: Callable
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    returns: Optional[str] = None
    cache_ttl: int = 300  # 5 minutes default
    when_to_use: str = "Use when relevant"
    call_count: int = 0
    total_time: float = 0.0
    last_called: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0


@dataclass
class CacheEntry:
    """Cached metadata result."""
    data: Any
    timestamp: datetime
    ttl: int
    query_hash: str
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() > self.timestamp + timedelta(seconds=self.ttl)


class MetaDataFetcherSignature(dspy.Signature):
    """
    Intelligent metadata fetching signature for ReAct agent.
    
    The agent analyzes the query and decides which metadata tools to call
    to gather all necessary information for downstream actors.
    """
    query: str = dspy.InputField(
        desc="User's natural language query that needs metadata"
    )
    available_tools: str = dspy.InputField(
        desc="Comprehensive catalog of available metadata tools with descriptions and usage guidelines"
    )
    previous_context: str = dspy.InputField(
        desc="Any previously fetched metadata or context from prior steps"
    )
    
    reasoning: str = dspy.OutputField(
        desc="Step-by-step reasoning about which metadata is needed and which tools to call"
    )
    tool_sequence: str = dspy.OutputField(
        desc="Ordered list of tools to call with rationale for each"
    )
    fetched_metadata: str = dspy.OutputField(
        desc="JSON dictionary of all fetched metadata with semantic keys (e.g., {'business_terms': {...}, 'available_tables': [...]})"
    )


class MetaDataFetcher:
    """
    Production-Grade Metadata Fetcher using DSPy ReAct.
    
    Features:
    - Automatic tool discovery via @jotty_method decorator
    - Intelligent metadata fetching using DSPy ReAct agent
    - Result caching with configurable TTL
    - Comprehensive error handling and retries
    - Performance monitoring and metrics
    - Thread-safe operations
    - Detailed logging for debugging
    
    Usage:
        # In SwarmReVal
        fetcher = MetaDataFetcher(metadata_provider=user_metadata)
        metadata = fetcher.fetch(query="User's natural language query")
        
        # Access metadata (keys depend on what tools returned)
        for key, value in metadata.items():
            logger.info(f"Fetched {key}: {value}")
    """
    
    def __init__(
        self,
        metadata_provider: Any,
        enable_cache: bool = True,
        default_cache_ttl: int = 300,
        max_retries: int = 0,
        retry_delay: float = 0.0,
        react_max_iters: int = 5
    ):
        """
        Initialize MetaDataFetcher with advanced configuration.
        
        Args:
            metadata_provider: Instance with @jotty_method decorated methods
            enable_cache: Enable result caching (default: True)
            default_cache_ttl: Default cache TTL in seconds (default: 300)
            max_retries: Maximum retry attempts on failure (default: 3)
            retry_delay: Initial retry delay in seconds (default: 1.0)
            react_max_iters: Max iterations for ReAct agent (default: 5)
        """
        from Jotty.core.foundation.config_defaults import MAX_RETRIES, RETRY_BACKOFF_SECONDS
        self.metadata_provider = metadata_provider
        self.enable_cache = enable_cache
        self.default_cache_ttl = default_cache_ttl
        self.max_retries = max_retries or MAX_RETRIES
        self.retry_delay = retry_delay or RETRY_BACKOFF_SECONDS
        self.react_max_iters = react_max_iters
        
        # Tool registry
        self.tools: List[dspy.Tool] = []
        self.tool_metadata: Dict[str, ToolMetadata] = {}
        
        # Caching
        self._cache: Dict[str, CacheEntry] = {}
        self._cache_lock = threading.Lock()
        
        # Metrics
        self.total_fetches = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_fetch_time = 0.0
        
        # ReAct agent (lazy init)
        self._react_agent = None
        
        # Initialize
        logger.info(" MetaDataFetcher: Initializing SOTA metadata fetching system")
        self._discover_and_convert_tools()
        logger.info(f" MetaDataFetcher: Ready with {len(self.tools)} tools discovered")
    
    def _discover_and_convert_tools(self):
        """
        Auto-discover @jotty_method decorated methods and convert to dspy.Tool.
        
        Process:
        1. Introspect metadata_provider for all methods
        2. Find methods with _jotty_meta attribute (added by @jotty_method decorator)
        3. Extract rich metadata (desc, params, returns, cache settings, etc.)
        4. Create dspy.Tool wrapper for each method
        5. Store tool metadata for monitoring and optimization
        """
        discovered_count = 0
        
        for attr_name in dir(self.metadata_provider):
            # Skip private/magic methods
            if attr_name.startswith('_'):
                continue
            
            try:
                attr = getattr(self.metadata_provider, attr_name)
                
                # Check if it's a callable with @jotty_method decorator
                if not callable(attr):
                    continue
                
                if not hasattr(attr, '_jotty_meta'):
                    continue
                
                # Extract metadata from decorator
                meta = attr._jotty_meta
                
                # Create tool metadata
                tool_meta = ToolMetadata(
                    name=attr_name,
                    func=attr,
                    description=meta.get('desc', f"Call {attr_name}"),
                    parameters=meta.get('params', {}),
                    returns=meta.get('returns'),
                    cache_ttl=meta.get('cache_ttl', self.default_cache_ttl),
                    when_to_use=meta.get('when', "Use when relevant")
                )
                
                # Create dspy.Tool
                tool = dspy.Tool(
                    func=attr,
                    name=attr_name,
                    desc=f"{tool_meta.description}\nWhen to use: {tool_meta.when_to_use}"
                )
                
                self.tools.append(tool)
                self.tool_metadata[attr_name] = tool_meta
                discovered_count += 1
                
                logger.debug(f" Discovered: {attr_name} - {tool_meta.description}")
                
            except Exception as e:
                logger.warning(f" Error discovering {attr_name}: {e}")
                continue
        
        if discovered_count == 0:
            logger.warning(" No @jotty_method tools found in metadata_provider!")
        else:
            logger.info(f" Tool discovery complete: {discovered_count} tools ready")
    
    def fetch(
        self,
        query: str,
        previous_context: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Proactively fetch metadata relevant to the query.
        
        Uses intelligent caching and ReAct agent to minimize LLM calls
        and maximize metadata relevance.
        
        Args:
            query: User's natural language query
            previous_context: Any previously fetched metadata (for context)
            force_refresh: Bypass cache and fetch fresh data
        
        Returns:
            Dict of fetched metadata with semantic keys
        """
        start_time = time.time()
        self.total_fetches += 1
        
        logger.info(" MetaDataFetcher: Starting metadata fetch")
        logger.info(f"   Query: {query[:100]}...")
        
        # Check cache first (unless force_refresh)
        if not force_refresh and self.enable_cache:
            cached_result = self._get_from_cache(query)
            if cached_result is not None:
                self.cache_hits += 1
                logger.info(" MetaDataFetcher: Cache HIT - returning cached metadata")
                return cached_result
            self.cache_misses += 1
        
        # No cache hit - fetch fresh metadata
        if not self.tools:
            logger.warning(" No tools available, returning empty metadata")
            return {}
        
        # Fetch with retry logic
        result = self._fetch_with_retry(query, previous_context)
        
        # Cache result
        if self.enable_cache and result:
            self._add_to_cache(query, result)
        
        # Update metrics
        elapsed = time.time() - start_time
        self.total_fetch_time += elapsed
        
        logger.info(f" MetaDataFetcher: Completed in {elapsed:.2f}s")
        logger.info(f"   Fetched {len(result)} metadata items: {list(result.keys())}")
        
        return result
    
    def _fetch_with_retry(
        self,
        query: str,
        previous_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Fetch metadata with automatic retry on failure.
        
        Implements exponential backoff for retries.
        """
        last_error = None
        retry_delay = self.retry_delay
        
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f" Fetch attempt {attempt}/{self.max_retries}")
                result = self._execute_fetch(query, previous_context)
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f" Fetch attempt {attempt} failed: {e}")
                
                if attempt < self.max_retries:
                    logger.info(f"   Retrying in {retry_delay:.1f}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f" All {self.max_retries} fetch attempts failed")
                    logger.error(f"   Last error: {last_error}")
                    logger.error(traceback.format_exc())
        
        # All retries failed - return empty dict
        return {}
    
    def _execute_fetch(
        self,
        query: str,
        previous_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute actual metadata fetch using ReAct agent.
        
        This is the core fetching logic that uses DSPy ReAct
        to intelligently select and call metadata tools.
        """
        # Initialize ReAct agent (lazy)
        if self._react_agent is None:
            self._react_agent = dspy.ReAct(
                signature=MetaDataFetcherSignature,
                tools=self.tools,
                max_iters=self.react_max_iters
            )
            logger.debug(" ReAct agent initialized")
        
        # Generate tool catalog for LLM
        tool_catalog = self._generate_tool_catalog()
        
        # Prepare previous context
        context_str = json.dumps(previous_context or {}, indent=2)
        
        # Execute ReAct
        logger.debug(" Executing ReAct agent...")
        result = self._react_agent(
            query=query,
            available_tools=tool_catalog,
            previous_context=context_str
        )
        
        # Extract and parse fetched metadata
        fetched_data = self._parse_react_output(result)
        
        # Log reasoning if available
        if hasattr(result, 'reasoning') and result.reasoning:
            logger.debug(f" Agent reasoning: {result.reasoning[:200]}...")
        
        # A-TEAM CRITICAL FIX: Ensure filter definitions are ALWAYS fetched!
        # If the ReAct agent didn't call get_all_filter_definitions(), do it manually
        if 'filter_conditions' not in fetched_data and 'get_all_filter_definitions' not in fetched_data:
            logger.info(" Filter conditions not fetched by ReAct agent, fetching manually...")
            try:
                # Try to find and call the filter definitions method
                if hasattr(self.metadata_provider, 'get_all_filter_definitions'):
                    filter_text = self.metadata_provider.get_all_filter_definitions()
                    if filter_text:
                        fetched_data['filter_conditions'] = filter_text
                        logger.info(f" Manually fetched filter conditions ({len(filter_text)} chars)")
            except Exception as e:
                logger.warning(f" Failed to manually fetch filter conditions: {e}")
        
        return fetched_data
    
    def _generate_tool_catalog(self) -> str:
        """
        Generate comprehensive tool catalog for LLM.
        
        Includes tool names, descriptions, parameters, and usage guidelines.
        """
        catalog_lines = ["Available Metadata Tools:", ""]
        
        for tool_name, tool_meta in self.tool_metadata.items():
            catalog_lines.append(f"### {tool_name}")
            catalog_lines.append(f"Description: {tool_meta.description}")
            catalog_lines.append(f"When to use: {tool_meta.when_to_use}")
            
            if tool_meta.parameters:
                params_str = ", ".join([
                    f"{k}: {v}" for k, v in tool_meta.parameters.items()
                ])
                catalog_lines.append(f"Parameters: {params_str}")
            
            if tool_meta.returns:
                catalog_lines.append(f"Returns: {tool_meta.returns}")
            
            # Add usage stats if available
            if tool_meta.call_count > 0:
                success_rate = (tool_meta.success_count / tool_meta.call_count) * 100
                avg_time = tool_meta.total_time / tool_meta.call_count
                catalog_lines.append(
                    f"Stats: {tool_meta.call_count} calls, "
                    f"{success_rate:.0f}% success, "
                    f"avg {avg_time:.2f}s"
                )
            
            catalog_lines.append("")  # Blank line between tools
        
        return "\n".join(catalog_lines)
    
    def _parse_react_output(self, result: Any) -> Dict[str, Any]:
        """
        Parse ReAct agent output to extract fetched metadata.
        
         A-TEAM FIX: Extract tool outputs from DSPy ReAct trajectory!
        DSPy ReAct stores trajectory as: {thought_0, tool_name_0, tool_args_0, observation_0, ...}
        """
        fetched_data = {}
        
        # STRATEGY 1: Extract from DSPy ReAct trajectory (CORRECT FORMAT!)
        if hasattr(result, 'trajectory') and result.trajectory:
            trajectory = result.trajectory
            logger.info(f" Parsing DSPy ReAct trajectory with {len(trajectory)} keys")
            
            # Find all observation keys (observation_0, observation_1, ...)
            tool_calls = {}
            for key in trajectory.keys():
                if key.startswith('tool_name_'):
                    idx = key.split('_')[2]  # Extract index
                    tool_name = trajectory.get(f'tool_name_{idx}')
                    observation = trajectory.get(f'observation_{idx}')
                    
                    if tool_name and observation and tool_name != 'finish':
                        tool_calls[idx] = {'tool': tool_name, 'output': observation}
                        logger.info(f" Found tool call {idx}: {tool_name}")
            
            # Map tool outputs to semantic keys
            for idx, call in tool_calls.items():
                tool_name = call['tool']
                observation = call['output']
                
                if tool_name == 'get_business_terms':
                    try:
                        fetched_data['business_terms'] = json.loads(observation) if isinstance(observation, str) else observation
                        logger.info(f" Extracted business_terms from trajectory[{idx}]")
                    except (json.JSONDecodeError, ValueError, TypeError) as e:
                        logger.debug(f"JSON parsing failed for business_terms: {e}")
                        fetched_data['business_terms'] = observation
                elif tool_name == 'get_all_tables':
                    try:
                        fetched_data['available_tables'] = json.loads(observation) if isinstance(observation, str) else observation
                        logger.info(f" Extracted available_tables from trajectory[{idx}]")
                    except (json.JSONDecodeError, ValueError, TypeError) as e:
                        logger.debug(f"JSON parsing failed for available_tables: {e}")
                        fetched_data['available_tables'] = observation
                elif tool_name == 'get_table_schema':
                    try:
                        schema_data = json.loads(observation) if isinstance(observation, str) else observation
                        fetched_data['table_schema'] = schema_data
                        logger.info(f" Extracted table_schema from trajectory[{idx}]")
                    except (json.JSONDecodeError, ValueError, TypeError) as e:
                        logger.debug(f"JSON parsing failed for table_schema: {e}")
                        fetched_data['table_schema'] = observation
                else:
                    # Store with tool name as key
                    fetched_data[tool_name] = observation
                    logger.info(f" Extracted {tool_name} from trajectory[{idx}]")
        
        # STRATEGY 2: Try to parse fetched_metadata field (if LLM followed instructions)
        if hasattr(result, 'fetched_metadata') and result.fetched_metadata:
            try:
                parsed = json.loads(result.fetched_metadata)
                # Merge with trajectory data (trajectory takes precedence)
                for key, value in parsed.items():
                    if key not in fetched_data:
                        fetched_data[key] = value
                logger.info(f" Merged fetched_metadata: {len(parsed)} items")
            except json.JSONDecodeError:
                logger.debug(" fetched_metadata is not valid JSON, relying on trajectory extraction")
        
        # A-TEAM FIX: POST-PROCESS to parse filter conditions if present
        fetched_data = self._post_process_metadata(fetched_data)
        
        logger.info(f" Extracted {len(fetched_data)} metadata items: {list(fetched_data.keys())}")
        
        return fetched_data
    
    def _post_process_metadata(self, fetched_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process fetched metadata to enrich it.
        
        CRITICAL: Parse filter_conditions text into structured format!
        This is GENERIC - works for ANY domain, not just SQL!
        """
        # Check if we have filter conditions (text format)
        filter_text = None
        for key in ['filter_conditions', 'get_all_filter_definitions', 'filters']:
            if key in fetched_data and isinstance(fetched_data[key], str):
                filter_text = fetched_data[key]
                logger.info(f" Found filter conditions in key '{key}' ({len(filter_text)} chars)")
                break
        
        if not filter_text:
            logger.debug("No filter conditions found to parse")
            return fetched_data
        
        # Parse filter conditions using LLM (generic!)
        try:
            parsed_filters = self._parse_filter_conditions_with_llm(filter_text)
            
            if parsed_filters:
                # Store both raw and parsed
                fetched_data['filter_conditions_parsed'] = parsed_filters
                logger.info(f" Parsed {len(parsed_filters)} filter condition sections")
            else:
                logger.warning(" Filter parsing returned empty result")
        
        except Exception as e:
            logger.warning(f" Failed to parse filter conditions: {e}")
            # Don't crash - continue with unparsed data
        
        return fetched_data
    
    def _parse_filter_conditions_with_llm(self, filter_text: str) -> Dict[str, Any]:
        """
        Parse filter conditions text into structured format using LLM.
        
        This is GENERIC - works for ANY filter/condition text, not just SQL!
        No hardcoding of SQL keywords, business terms, or domain-specific knowledge.
        
        Args:
            filter_text: Raw text containing filter conditions (any format)
        
        Returns:
            Dict mapping category names to their filter specifications
        """
        # Use configured LM (no hardcoding of model name!)
        lm = dspy.settings.lm if dspy.settings.lm else None
        if not lm:
            logger.warning(" No LM configured, skipping filter parsing")
            return {}
        
        prompt = f"""You are a metadata parser. Extract filter/condition specifications from text.

INPUT TEXT:
{filter_text[:5000]}  

TASK:
1. Identify ALL sections that define filters, conditions, or constraints
2. For each section, extract:
   - Category/term name (the thing being filtered)
   - Required fields/columns (if mentioned)
   - Filter expression or condition (exact text)
   - Description/purpose (if provided)

OUTPUT FORMAT (JSON):
{{
    "category_name_1": {{
        "fields": ["field1", "field2"],
        "filter_expression": "exact filter text from document",
        "description": "brief description"
    }},
    ...
}}

RULES:
- Extract ONLY what's explicitly in the text
- Do NOT invent or guess data
- Use exact field/column names from text
- Preserve exact filter expressions
- If a section has no clear filters, omit it

Return ONLY the JSON, no other text."""

        try:
            # Call LLM (generic!)
            response = lm(prompt)
            
            # Handle different response formats
            if isinstance(response, list) and len(response) > 0:
                response_text = response[0]
            elif hasattr(response, 'choices') and len(response.choices) > 0:
                response_text = response.choices[0].message.content
            elif isinstance(response, str):
                response_text = response
            else:
                response_text = str(response)
            
            # Extract JSON from response (handle markdown code blocks)
            response_text = response_text.strip()
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            # Parse JSON
            parsed = json.loads(response_text)
            logger.debug(f" LLM parsed {len(parsed)} filter categories")
            return parsed
        
        except Exception as e:
            logger.warning(f" LLM parsing failed: {e}")
            return {}
    
    def _get_from_cache(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached result if available and not expired."""
        query_hash = self._hash_query(query)
        
        with self._cache_lock:
            if query_hash in self._cache:
                entry = self._cache[query_hash]
                
                if not entry.is_expired():
                    logger.debug(f" Cache hit for query (age: {(datetime.now() - entry.timestamp).seconds}s)")
                    return entry.data
                else:
                    logger.debug(" Cache entry expired, removing")
                    del self._cache[query_hash]
        
        return None
    
    def _add_to_cache(self, query: str, data: Dict[str, Any]):
        """Add result to cache."""
        query_hash = self._hash_query(query)
        
        entry = CacheEntry(
            data=data,
            timestamp=datetime.now(),
            ttl=self.default_cache_ttl,
            query_hash=query_hash
        )
        
        with self._cache_lock:
            self._cache[query_hash] = entry
            logger.debug(f" Cached result (TTL: {self.default_cache_ttl}s)")
    
    def _hash_query(self, query: str) -> str:
        """Generate hash for query (for cache keys)."""
        return hashlib.md5(query.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear all cached results."""
        with self._cache_lock:
            cleared_count = len(self._cache)
            self._cache.clear()
            logger.info(f" Cleared {cleared_count} cache entries")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Returns:
            Dict with fetch counts, cache stats, timing info, etc.
        """
        with self._cache_lock:
            active_cache_entries = len(self._cache)
        
        cache_hit_rate = (
            (self.cache_hits / (self.cache_hits + self.cache_misses) * 100)
            if (self.cache_hits + self.cache_misses) > 0
            else 0.0
        )
        
        avg_fetch_time = (
            self.total_fetch_time / self.total_fetches
            if self.total_fetches > 0
            else 0.0
        )
        
        return {
            'total_fetches': self.total_fetches,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'active_cache_entries': active_cache_entries,
            'total_fetch_time': f"{self.total_fetch_time:.2f}s",
            'avg_fetch_time': f"{avg_fetch_time:.2f}s",
            'tools_discovered': len(self.tools),
            'tool_stats': {
                name: {
                    'calls': meta.call_count,
                    'success_rate': f"{(meta.success_count/meta.call_count*100):.0f}%"
                    if meta.call_count > 0 else "N/A",
                    'avg_time': f"{(meta.total_time/meta.call_count):.2f}s"
                    if meta.call_count > 0 else "N/A"
                }
                for name, meta in self.tool_metadata.items()
                if meta.call_count > 0
            }
        }
    
    def __repr__(self) -> str:
        """String representation with key stats."""
        return (
            f"MetaDataFetcher("
            f"tools={len(self.tools)}, "
            f"fetches={self.total_fetches}, "
            f"cache_hit_rate={self.cache_hits/(self.cache_hits+self.cache_misses+1)*100:.0f}%"
            f")"
        )








