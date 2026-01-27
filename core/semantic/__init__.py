"""
Core Semantic Layer

Provides semantic understanding of database schemas for intelligent querying.

Architecture:
    DDL/Database -> Schema -> LookML -> LLM Context -> SQL

Components:
    - models: Core data models (Schema, Table, Column, Relationship)
    - extractors: Extract schema from databases or DDL strings
    - lookml: Generate LookML semantic layer
    - query: Natural language to SQL using semantic context

Usage:
    # From live database
    from core.semantic import SemanticLayer

    layer = SemanticLayer.from_database(
        db_type="postgresql",
        host="localhost",
        database="mydb",
        user="postgres",
        password="secret"
    )

    # Query with natural language
    result = layer.query("Show me total sales by region")
    print(result['generated_sql'])

    # From DDL string
    layer = SemanticLayer.from_ddl(ddl_string, dialect="postgresql")

    # Get LookML
    lookml = layer.to_lookml()
"""
from typing import Dict, Any, Optional, List

from .models import (
    Schema, Table, Column, ForeignKey, Index, Relationship,
    ColumnType, RelationType, MeasureType
)
from .extractors import BaseExtractor, DatabaseExtractor, DDLExtractor, MongoDBExtractor
from .lookml import LookMLGenerator, View, Explore, Dimension, Measure, Join
from .query import SemanticQueryEngine, MongoDBQueryEngine
from .query import (
    ConnectorXLoader,
    DataLoaderFactory,
    OutputFormat,
    SQLDatePreprocessor,
    MongoDBDatePreprocessor,
)


class SemanticLayer:
    """
    Main interface for the semantic layer.

    Provides a unified API for:
    - Extracting schema from databases or DDL
    - Generating LookML semantic models
    - Querying with natural language
    """

    def __init__(
        self,
        schema: Schema,
        connection_params: Dict[str, Any] = None
    ):
        """
        Initialize semantic layer.

        Args:
            schema: Database schema
            connection_params: Optional connection parameters for query execution
        """
        self.schema = schema
        self.connection_params = connection_params or {}

        self._lookml_generator: Optional[LookMLGenerator] = None
        self._lookml_model = None
        self._query_engine: Optional[SemanticQueryEngine] = None
        self._mongodb_query_engine: Optional[MongoDBQueryEngine] = None

    @classmethod
    def from_database(
        cls,
        db_type: str = None,
        host: str = "localhost",
        port: int = None,
        database: str = "",
        user: str = "",
        password: str = "",
        connection_string: str = None,
        schema_name: str = "default",
        **kwargs
    ) -> "SemanticLayer":
        """
        Create semantic layer from live database connection.

        Args:
            db_type: Database type (postgresql, mysql, sqlite, mssql, oracle)
            host: Database host
            port: Database port
            database: Database name
            user: Username
            password: Password
            connection_string: Full connection URL (overrides other params)
            schema_name: Name for the extracted schema
            **kwargs: Additional connection parameters

        Returns:
            SemanticLayer instance
        """
        extractor = DatabaseExtractor(
            db_type=db_type,
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            connection_string=connection_string,
            **kwargs
        )

        schema = extractor.extract(schema_name)

        connection_params = {
            "db_type": db_type or extractor.database_type,
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
            "connection_string": connection_string,
        }
        connection_params.update(kwargs)

        return cls(schema, connection_params)

    @classmethod
    def from_ddl(
        cls,
        ddl: str,
        dialect: str = "postgresql",
        schema_name: str = "default"
    ) -> "SemanticLayer":
        """
        Create semantic layer from DDL string.

        Args:
            ddl: DDL string to parse
            dialect: SQL dialect
            schema_name: Name for the extracted schema

        Returns:
            SemanticLayer instance
        """
        extractor = DDLExtractor(ddl, dialect)
        schema = extractor.extract(schema_name)

        return cls(schema, {"db_type": dialect})

    @classmethod
    def from_ddl_file(
        cls,
        file_path: str,
        dialect: str = "postgresql",
        schema_name: str = "default"
    ) -> "SemanticLayer":
        """
        Create semantic layer from DDL file.

        Args:
            file_path: Path to DDL file
            dialect: SQL dialect
            schema_name: Name for the extracted schema

        Returns:
            SemanticLayer instance
        """
        extractor = DDLExtractor.from_file(file_path, dialect)
        schema = extractor.extract(schema_name)

        return cls(schema, {"db_type": dialect})

    @classmethod
    def from_mongodb(
        cls,
        uri: str = None,
        host: str = "localhost",
        port: int = 27017,
        database: str = "",
        username: str = None,
        password: str = None,
        schema_name: str = None,
        sample_size: int = 100,
        **kwargs
    ) -> "SemanticLayer":
        """
        Create semantic layer from MongoDB connection.

        Args:
            uri: Full MongoDB connection URI (overrides other params)
            host: MongoDB host
            port: MongoDB port
            database: Database name
            username: Username for authentication
            password: Password for authentication
            schema_name: Name for the extracted schema (defaults to database name)
            sample_size: Number of documents to sample per collection
            **kwargs: Additional pymongo connection options

        Returns:
            SemanticLayer instance
        """
        extractor = MongoDBExtractor(
            uri=uri,
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
            sample_size=sample_size,
            **kwargs
        )

        schema = extractor.extract(schema_name or database or "mongodb")

        connection_params = {
            "db_type": "mongodb",
            "uri": uri or extractor.uri,
            "database": database or extractor.database_name,
        }

        return cls(schema, connection_params)

    @property
    def lookml_generator(self) -> LookMLGenerator:
        """Get LookML generator."""
        if self._lookml_generator is None:
            self._lookml_generator = LookMLGenerator(self.schema)
        return self._lookml_generator

    @property
    def lookml_model(self):
        """Get generated LookML model."""
        if self._lookml_model is None:
            self._lookml_model = self.lookml_generator.generate()
        return self._lookml_model

    @property
    def query_engine(self) -> SemanticQueryEngine:
        """Get semantic query engine."""
        if self._query_engine is None:
            self._query_engine = SemanticQueryEngine(
                schema=self.schema,
                lookml_model=self.lookml_model,
                db_type=self.connection_params.get("db_type", self.schema.database_type)
            )
        return self._query_engine

    @property
    def mongodb_query_engine(self) -> Optional[MongoDBQueryEngine]:
        """Get MongoDB query engine (only for MongoDB databases)."""
        if self._mongodb_query_engine is None and self.is_mongodb:
            self._mongodb_query_engine = MongoDBQueryEngine(
                schema=self.schema,
                lookml_model=self.lookml_model,
                uri=self.connection_params.get("uri"),
                database=self.connection_params.get("database")
            )
        return self._mongodb_query_engine

    @property
    def is_mongodb(self) -> bool:
        """Check if this is a MongoDB database."""
        return self.connection_params.get("db_type") == "mongodb"

    def to_lookml(self, pretty: bool = True) -> str:
        """
        Generate LookML string.

        Args:
            pretty: Format output for readability

        Returns:
            LookML string
        """
        return self.lookml_generator.to_lookml_string(self.lookml_model)

    def get_context(self) -> str:
        """
        Get semantic context string for LLM.

        Returns:
            Formatted context with tables, columns, and relationships
        """
        return self.query_engine.get_context()

    def query(
        self,
        question: str,
        execute: bool = True
    ) -> Dict[str, Any]:
        """
        Query using natural language.

        Args:
            question: Natural language question
            execute: Whether to execute the generated query

        Returns:
            Dictionary with generated query and optional results
        """
        # Use MongoDB query engine for MongoDB databases
        if self.is_mongodb:
            return self.mongodb_query_engine.generate_pipeline(
                question=question,
                execute=execute
            )

        # Use SQL query engine for other databases
        return self.query_engine.generate_sql(
            question=question,
            execute=execute,
            connection_params=self.connection_params if execute else None
        )

    def suggest_queries(self, num_suggestions: int = 5) -> List[str]:
        """
        Get suggested queries based on schema.

        Args:
            num_suggestions: Number of suggestions

        Returns:
            List of suggested natural language queries
        """
        if self.is_mongodb:
            return self.mongodb_query_engine.suggest_queries(num_suggestions)
        return self.query_engine.suggest_queries(num_suggestions)

    def validate_sql(self, sql: str) -> Dict[str, Any]:
        """
        Validate SQL syntax.

        Args:
            sql: SQL to validate

        Returns:
            Validation result
        """
        return self.query_engine.validate_sql(sql)

    def get_tables(self) -> List[str]:
        """Get list of table names."""
        return [t.name for t in self.schema.tables]

    def get_table(self, name: str) -> Optional[Table]:
        """Get table by name."""
        return self.schema.get_table(name)

    def get_relationships(self) -> List[Dict[str, Any]]:
        """Get all relationships."""
        return [
            {
                "from_table": r.from_table,
                "from_columns": r.from_columns,
                "to_table": r.to_table,
                "to_columns": r.to_columns,
                "type": r.relation_type.value,
            }
            for r in self.schema.relationships
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Export schema as dictionary."""
        return self.schema.to_dict()


# Convenience functions
def create_semantic_layer(
    db_type: str = None,
    connection_string: str = None,
    ddl: str = None,
    mongodb_uri: str = None,
    **kwargs
) -> SemanticLayer:
    """
    Create semantic layer from various sources.

    Args:
        db_type: Database type for live connection
        connection_string: Full connection URL for SQL databases
        ddl: DDL string to parse
        mongodb_uri: MongoDB connection URI
        **kwargs: Additional parameters

    Returns:
        SemanticLayer instance
    """
    if ddl:
        dialect = kwargs.pop("dialect", db_type or "postgresql")
        return SemanticLayer.from_ddl(ddl, dialect)
    elif mongodb_uri or db_type == "mongodb":
        return SemanticLayer.from_mongodb(
            uri=mongodb_uri,
            **kwargs
        )
    elif connection_string or db_type:
        return SemanticLayer.from_database(
            db_type=db_type,
            connection_string=connection_string,
            **kwargs
        )
    else:
        raise ValueError("Either ddl, connection_string, mongodb_uri, or db_type is required")


__all__ = [
    # Main interface
    "SemanticLayer",
    "create_semantic_layer",

    # Models
    "Schema",
    "Table",
    "Column",
    "ForeignKey",
    "Index",
    "Relationship",
    "ColumnType",
    "RelationType",
    "MeasureType",

    # Extractors
    "BaseExtractor",
    "DatabaseExtractor",
    "DDLExtractor",
    "MongoDBExtractor",

    # LookML
    "LookMLGenerator",
    "View",
    "Explore",
    "Dimension",
    "Measure",
    "Join",

    # Query
    "SemanticQueryEngine",
    "MongoDBQueryEngine",
]
