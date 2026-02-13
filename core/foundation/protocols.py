"""
Generic Protocols for JOTTY Framework

This module defines the core protocols that make JOTTY domain-agnostic.
Users implement these protocols for their specific use cases (SQL, code-gen, etc.).

JOTTY makes NO assumptions about:
- Domain concepts (tables, functions, documents, etc.)
- Data structures
- Metadata formats

This is the foundation of JOTTY's genericity.
"""

from typing import Protocol, Dict, Any, Optional, List, runtime_checkable


@runtime_checkable
class MetadataProvider(Protocol):
    """Protocol for providing context/metadata to actors in a swarm.
    
    This is the PRIMARY interface between JOTTY and domain-specific code.
    Users implement this for their use case.
    
    Example Implementations:
        - SQLMetadataProvider: Provides business context, table/column metadata
        - CodeGenMetadataProvider: Provides function signatures, class definitions
        - DocSumProvider: Provides document sections, key concepts
        - ImageCaptionProvider: Provides image features, object labels
    
    JOTTY Core Guarantee:
        - Will NEVER assume specific keys in returned dictionaries
        - Will NEVER assume specific data types
        - Will NEVER assume domain concepts
    
    Usage:
        ```python
        # User creates domain-specific provider
        provider = SQLMetadataProvider(
            business_context_path="data/business.md",
            table_metadata_path="data/tables.json",
        )
        
        # Pass to JOTTY (works for ANY provider!)
        swarm = Orchestrator(
            actors=[...],
            metadata_provider=provider,  # Generic interface
            config=config
        )
        ```
    """
    
    def get_context_for_actor(
        self,
        actor_name: str,
        query: str,
        previous_outputs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Return context/metadata needed by a specific actor.
        
        This method is called by JOTTY when an actor is about to execute.
        You can return different context for different actors based on their needs.
        
        Args:
            actor_name: Name of the actor requesting context (e.g., "BusinessTermResolver")
            query: The user's query/task (e.g., "How many events of type X in time period Y?")
            previous_outputs: Outputs from actors that already executed in this swarm.
                            Dict maps actor_name -> output dict.
                            Use this to provide context-aware metadata!
            **kwargs: Additional context:
                     - session_id: User session ID
                     - conversation_history: Previous conversation
                     - current_date: Current date
                     - Any other domain-specific context
        
        Returns:
            Dictionary with ANY keys the actor needs.
            
            JOTTY will:
            1. Introspect the actor's forward() signature
            2. Match dict keys to actor parameters
            3. Pass matched values to the actor
            
            Examples:
                SQL use case:
                    {"business_context": str, "tables": List[str], "columns": Dict}
                
                Code-gen use case:
                    {"functions": List[FunctionDef], "imports": List[str]}
                
                Doc-sum use case:
                    {"document": str, "style_guide": str, "key_concepts": List[str]}
        
        Notes:
            - You can return different keys for different actors
            - You can use previous_outputs to build context incrementally
            - You can return empty dict {} if actor doesn't need metadata
            - Keys should match actor's forward() parameter names
        
        Example Implementation:
            ```python
            def get_context_for_actor(self, actor_name, query, previous_outputs, **kwargs):
                if actor_name == "BusinessTermResolver":
                    return {
                        "business_context": self.business_context_str,
                        "table_metadata": self.table_metadata_str,
                    }
                elif actor_name == "ColumnSelector":
                    # Use output from previous actor!
                    tables = previous_outputs.get("BusinessTermResolver", {}).get("tables", [])
                    return {
                        "column_metadata": self.get_columns_for_tables(tables),
                        "filters": self.get_filters(),
                    }
                else:
                    return {}  # Actor doesn't need metadata
            ```
        """
        ...
    
    def get_swarm_context(self, **kwargs) -> Dict[str, Any]:
        """Return global context for the entire swarm (optional).
        
        This is called ONCE when the swarm is initialized, before any actors run.
        Use for:
            - Global constraints (e.g., "Must use Trino SQL dialect")
            - Shared knowledge (e.g., "Project is called 'MyApp'")
            - Swarm-level metadata (e.g., "Domain: SQL, Database: PostgreSQL")
        
        Args:
            **kwargs: Initialization context (same as get_context_for_actor)
        
        Returns:
            Dictionary with swarm-level context.
            Common keys might include:
                - domain: str (e.g., "sql", "code_gen", "doc_sum")
                - constraints: List[str]
                - global_knowledge: Dict[str, Any]
        
        Notes:
            - This method is OPTIONAL
            - If not implemented or returns empty dict, no swarm context is added
            - Swarm context is available to ALL actors
        
        Example:
            ```python
            def get_swarm_context(self, **kwargs):
                return {
                    "domain": "sql",
                    "database": "trino",
                    "dialect": "presto",
                    "constraints": ["No subqueries in WHERE clause"],
                }
            ```
        """
        ...


@runtime_checkable
class DataProvider(Protocol):
    """Protocol for providing access to raw data sources.
    
    This protocol is for READING/WRITING data (files, DBs, APIs), not metadata.
    
    Example Implementations:
        - FileSystemDataProvider: Read/write local files
        - DatabaseDataProvider: Query databases
        - APIDataProvider: Call external APIs
        - S3DataProvider: Access cloud storage
    
    Difference from MetadataProvider:
        - MetadataProvider: Returns CONTEXT (strings, small dicts) for actors
        - DataProvider: Returns RAW DATA (large files, query results, API responses)
    
    Usage:
        ```python
        data_provider = FileSystemDataProvider(base_path="./data")
        
        # Actors can retrieve data during execution
        data = data_provider.retrieve("customer_data.csv")
        ```
    """
    
    def retrieve(
        self,
        key: str,
        format: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Retrieve data from the provider.
        
        Args:
            key: Data identifier (file path, table name, API endpoint, etc.)
            format: Expected format ("csv", "json", "parquet", etc.) - optional
            **kwargs: Provider-specific options
        
        Returns:
            The data in any format (DataFrame, dict, string, bytes, etc.)
        
        Example:
            ```python
            # File system
            df = provider.retrieve("sales_data.csv", format="dataframe")
            
            # Database
            results = provider.retrieve("SELECT * FROM users", format="dataframe")
            
            # API
            response = provider.retrieve("/api/v1/customers", format="json")
            ```
        """
        ...
    
    def store(
        self,
        key: str,
        value: Any,
        format: Optional[str] = None,
        **kwargs
    ) -> None:
        """Store data to the provider.
        
        Args:
            key: Data identifier for storage
            value: Data to store (any type)
            format: Storage format ("csv", "json", "parquet", etc.) - optional
            **kwargs: Provider-specific options
        
        Example:
            ```python
            # Store DataFrame as CSV
            provider.store("output.csv", df, format="csv")
            
            # Store dict as JSON
            provider.store("results.json", {"status": "success"}, format="json")
            ```
        """
        ...


@runtime_checkable
class ContextExtractor(Protocol):
    """Protocol for extracting relevant context from large documents.
    
    Used when metadata/data is too large to fit in context window.
    Extracts semantically relevant portions based on query/task.
    
    Example Implementations:
        - SemanticExtractor: Uses embeddings to find relevant sections
        - KeywordExtractor: Extracts based on keyword matching
        - LLMExtractor: Uses LLM to identify relevant portions
    """
    
    def extract(
        self,
        content: str,
        query: str,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Extract relevant context from large content.
        
        Args:
            content: Full content (may be very large)
            query: Query/task to extract context for
            max_tokens: Maximum tokens in extracted content
            **kwargs: Extractor-specific options
        
        Returns:
            Extracted content (guaranteed to be <= max_tokens)
        
        Example:
            ```python
            # Content is 100k tokens, but actor only has 5k budget
            extracted = extractor.extract(
                content=full_business_context,
                query="target_concept_for_extraction",
                max_tokens=5000
            )
            # Result: ~5k tokens of relevant context about target concept
            ```
        """
        ...


# Type aliases for convenience
ContextDict = Dict[str, Any]
ActorOutput = Dict[str, Any]

