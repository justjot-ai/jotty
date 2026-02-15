"""
Semantic Layer Skill

Database intelligence and visualization:
- Natural language to SQL/MongoDB
- Schema extraction and analysis
- Data visualization
- LookML generation

All tools are exposed for agent discovery.
"""

# Import core semantic layer components
from .semantic import (
    Column,
    ColumnType,
    DatabaseExtractor,
    DDLExtractor,
    ForeignKey,
    Index,
    MeasureType,
    MongoDBExtractor,
    MongoDBQueryEngine,
    Relationship,
    RelationType,
    Schema,
    SemanticLayer,
    SemanticQueryEngine,
    Table,
    create_semantic_layer,
)

# Import tools for skill discovery
from .tools import (
    analyze_ddl_schema,
    create_dashboard,
    extract_database_schema,
    generate_lookml_from_ddl,
    query_database_natural_language,
    suggest_related_queries,
    visualize_data_from_query,
)

__all__ = [
    # Core components
    "SemanticLayer",
    "create_semantic_layer",
    "Schema",
    "Table",
    "Column",
    # Tools (discovered by registry)
    "query_database_natural_language",
    "suggest_related_queries",
    "analyze_ddl_schema",
    "extract_database_schema",
    "generate_lookml_from_ddl",
    "visualize_data_from_query",
    "create_dashboard",
]
