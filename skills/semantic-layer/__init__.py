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
    SemanticLayer,
    create_semantic_layer,
    Schema,
    Table,
    Column,
    ForeignKey,
    Index,
    Relationship,
    ColumnType,
    RelationType,
    MeasureType,
    DatabaseExtractor,
    DDLExtractor,
    MongoDBExtractor,
    SemanticQueryEngine,
    MongoDBQueryEngine,
)

# Import tools for skill discovery
from .tools import (
    query_database_natural_language,
    suggest_related_queries,
    analyze_ddl_schema,
    extract_database_schema,
    generate_lookml_from_ddl,
    visualize_data_from_query,
    create_dashboard,
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
