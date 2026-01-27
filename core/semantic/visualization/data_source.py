"""
Data Source Abstraction

Provides a unified interface for different data sources to feed into LIDA visualization.
Supports: SemanticLayer queries, raw DataFrames, MongoDB results, and more.

DRY Principle: Single abstraction for all data inputs to visualization layer.
"""
from typing import Dict, Any, Optional, Union, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import tempfile
import os

logger = logging.getLogger(__name__)


@dataclass
class DataSourceResult:
    """Result from a data source query."""
    success: bool
    data: Any = None  # DataFrame or dict
    query: str = None  # Original query (NL or SQL/Pipeline)
    generated_query: str = None  # Generated SQL or Pipeline
    error: str = None
    metadata: Dict[str, Any] = None

    def to_dataframe(self):
        """Convert result to pandas DataFrame."""
        import pandas as pd

        if self.data is None:
            return pd.DataFrame()

        if isinstance(self.data, pd.DataFrame):
            return self.data

        # Polars DataFrame
        if hasattr(self.data, 'to_pandas'):
            return self.data.to_pandas()

        # Arrow Table
        if hasattr(self.data, 'to_pandas'):
            return self.data.to_pandas()

        # List of dicts (MongoDB style)
        if isinstance(self.data, list):
            return pd.DataFrame(self.data)

        # Dict with 'rows' key
        if isinstance(self.data, dict) and 'rows' in self.data:
            return pd.DataFrame(self.data['rows'])

        return pd.DataFrame()


class DataSource(ABC):
    """
    Abstract base class for data sources.

    All data sources must implement:
    - query(): Execute a query and return data
    - get_schema(): Return schema information for LIDA context
    """

    @abstractmethod
    def query(self, question: str, **kwargs) -> DataSourceResult:
        """
        Execute a query against the data source.

        Args:
            question: Natural language question or direct query
            **kwargs: Additional query parameters

        Returns:
            DataSourceResult with data and metadata
        """
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Get schema information for the data source.

        Returns:
            Dictionary with table/collection names, columns, types
        """
        pass

    def to_csv_file(self, data: Any, max_rows: int = 10000) -> str:
        """
        Convert data to a temporary CSV file for LIDA.

        Args:
            data: DataFrame or list of dicts
            max_rows: Maximum rows to include

        Returns:
            Path to temporary CSV file
        """
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            df = data
        elif hasattr(data, 'to_pandas'):
            df = data.to_pandas()
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame()

        # Limit rows for LIDA performance
        if len(df) > max_rows:
            df = df.head(max_rows)

        # Create temp file
        fd, path = tempfile.mkstemp(suffix='.csv')
        df.to_csv(path, index=False)
        os.close(fd)

        return path


class SemanticDataSource(DataSource):
    """
    Data source that uses SemanticLayer for NL-to-SQL/MongoDB queries.

    Integrates our full pipeline:
    NL Question → SemanticLayer → SQL/Pipeline → ConnectorX → DataFrame
    """

    def __init__(self, semantic_layer, use_connectorx: bool = True):
        """
        Initialize with a SemanticLayer instance.

        Args:
            semantic_layer: SemanticLayer instance (SQL or MongoDB)
            use_connectorx: Use ConnectorX for fast loading (default: True)
        """
        self.semantic_layer = semantic_layer
        self.use_connectorx = use_connectorx
        self._setup_loader()

    def _setup_loader(self):
        """Setup ConnectorX loader if available."""
        if self.use_connectorx and hasattr(self.semantic_layer, 'query_engine'):
            engine = self.semantic_layer.query_engine
            conn_params = self.semantic_layer.connection_params
            if conn_params:
                engine.set_connection(**conn_params)

    def query(self, question: str, output_format: str = "pandas", **kwargs) -> DataSourceResult:
        """
        Execute NL query through SemanticLayer.

        Args:
            question: Natural language question
            output_format: DataFrame format (pandas, polars, arrow)
            **kwargs: Additional query parameters

        Returns:
            DataSourceResult with DataFrame
        """
        try:
            # Check if MongoDB or SQL
            if self.semantic_layer.is_mongodb:
                result = self.semantic_layer.query(question, execute=True)
                if result.get('success'):
                    rows = result.get('query_result', {}).get('rows', [])
                    import pandas as pd
                    df = pd.DataFrame(rows)
                    return DataSourceResult(
                        success=True,
                        data=df,
                        query=question,
                        generated_query=str(result.get('pipeline', [])),
                        metadata={
                            'collection': result.get('collection'),
                            'date_context': result.get('date_context'),
                        }
                    )
            else:
                # Use ConnectorX if available
                if self.use_connectorx and hasattr(self.semantic_layer.query_engine, 'query_to_dataframe'):
                    result = self.semantic_layer.query_engine.query_to_dataframe(
                        question, output_format=output_format, **kwargs
                    )
                    if result.get('success') and result.get('dataframe') is not None:
                        return DataSourceResult(
                            success=True,
                            data=result['dataframe'],
                            query=question,
                            generated_query=result.get('generated_sql'),
                            metadata={
                                'date_context': result.get('date_context'),
                                'db_type': result.get('db_type'),
                            }
                        )
                else:
                    # Fallback to standard query
                    result = self.semantic_layer.query(question, execute=True)
                    if result.get('success'):
                        rows = result.get('query_result', {}).get('rows', [])
                        import pandas as pd
                        return DataSourceResult(
                            success=True,
                            data=pd.DataFrame(rows),
                            query=question,
                            generated_query=result.get('generated_sql'),
                        )

            # Handle errors
            error = result.get('error') or result.get('query_result', {}).get('error')
            return DataSourceResult(
                success=False,
                query=question,
                error=error or "Query failed"
            )

        except Exception as e:
            logger.error(f"SemanticDataSource query failed: {e}")
            return DataSourceResult(success=False, query=question, error=str(e))

    def get_schema(self) -> Dict[str, Any]:
        """Get schema from SemanticLayer."""
        schema = self.semantic_layer.schema
        return {
            'name': schema.name,
            'database_type': schema.database_type,
            'tables': [
                {
                    'name': t.name,
                    'columns': [
                        {'name': c.name, 'type': c.normalized_type.value}
                        for c in t.columns
                    ]
                }
                for t in schema.tables
            ]
        }


class DataFrameSource(DataSource):
    """
    Data source for raw DataFrames.

    Use when you already have data loaded and want to visualize it.
    """

    def __init__(self, dataframe, name: str = "data"):
        """
        Initialize with a DataFrame.

        Args:
            dataframe: pandas, polars, or arrow DataFrame
            name: Name for the data source
        """
        self.dataframe = dataframe
        self.name = name
        self._df = None

    @property
    def df(self):
        """Get pandas DataFrame."""
        if self._df is None:
            import pandas as pd
            if isinstance(self.dataframe, pd.DataFrame):
                self._df = self.dataframe
            elif hasattr(self.dataframe, 'to_pandas'):
                self._df = self.dataframe.to_pandas()
            else:
                self._df = pd.DataFrame(self.dataframe)
        return self._df

    def query(self, question: str, **kwargs) -> DataSourceResult:
        """
        For DataFrame source, 'query' just returns the data.

        The 'question' is passed to LIDA for goal generation.
        """
        return DataSourceResult(
            success=True,
            data=self.df,
            query=question,
            metadata={'name': self.name, 'rows': len(self.df), 'columns': len(self.df.columns)}
        )

    def get_schema(self) -> Dict[str, Any]:
        """Infer schema from DataFrame."""
        df = self.df
        return {
            'name': self.name,
            'database_type': 'dataframe',
            'tables': [{
                'name': self.name,
                'columns': [
                    {'name': col, 'type': str(df[col].dtype)}
                    for col in df.columns
                ]
            }]
        }


class MongoDBSource(DataSource):
    """
    Data source for MongoDB aggregation results.

    Works with our MongoDBQueryEngine for NL-to-Pipeline queries.
    """

    def __init__(self, uri: str, database: str, semantic_layer=None):
        """
        Initialize MongoDB source.

        Args:
            uri: MongoDB connection URI
            database: Database name
            semantic_layer: Optional SemanticLayer for NL queries
        """
        self.uri = uri
        self.database = database
        self.semantic_layer = semantic_layer
        self._client = None

    @property
    def client(self):
        """Get MongoDB client."""
        if self._client is None:
            from pymongo import MongoClient
            self._client = MongoClient(self.uri)
        return self._client

    def query(self, question: str, collection: str = None, **kwargs) -> DataSourceResult:
        """
        Execute query against MongoDB.

        Args:
            question: NL question or pipeline dict
            collection: Collection name (required if no semantic_layer)
            **kwargs: Additional options

        Returns:
            DataSourceResult with data
        """
        import pandas as pd

        try:
            # If semantic_layer available, use NL-to-Pipeline
            if self.semantic_layer:
                result = self.semantic_layer.query(question, execute=True)
                if result.get('success'):
                    rows = result.get('query_result', {}).get('rows', [])
                    return DataSourceResult(
                        success=True,
                        data=pd.DataFrame(rows),
                        query=question,
                        generated_query=str(result.get('pipeline', [])),
                        metadata={'collection': result.get('collection')}
                    )
                return DataSourceResult(
                    success=False,
                    query=question,
                    error=result.get('error')
                )

            # Direct pipeline execution
            if isinstance(question, list):
                pipeline = question
            else:
                return DataSourceResult(
                    success=False,
                    query=question,
                    error="Provide a pipeline list or use semantic_layer for NL queries"
                )

            if not collection:
                return DataSourceResult(
                    success=False,
                    error="Collection name required for direct pipeline execution"
                )

            db = self.client[self.database]
            results = list(db[collection].aggregate(pipeline))

            return DataSourceResult(
                success=True,
                data=pd.DataFrame(results),
                query=question,
                metadata={'collection': collection}
            )

        except Exception as e:
            logger.error(f"MongoDB query failed: {e}")
            return DataSourceResult(success=False, query=question, error=str(e))

    def get_schema(self) -> Dict[str, Any]:
        """Get schema from MongoDB (sample-based)."""
        if self.semantic_layer:
            return {
                'name': self.database,
                'database_type': 'mongodb',
                'tables': [
                    {'name': t.name, 'columns': [
                        {'name': c.name, 'type': c.normalized_type.value}
                        for c in t.columns
                    ]}
                    for t in self.semantic_layer.schema.tables
                ]
            }

        # Basic schema from collection names
        db = self.client[self.database]
        return {
            'name': self.database,
            'database_type': 'mongodb',
            'tables': [{'name': c, 'columns': []} for c in db.list_collection_names()]
        }


class DataSourceFactory:
    """
    Factory for creating appropriate data source based on input type.
    """

    @classmethod
    def create(cls, source, **kwargs) -> DataSource:
        """
        Create a DataSource from various input types.

        Args:
            source: Can be:
                - SemanticLayer instance
                - pandas/polars/arrow DataFrame
                - Dict with 'uri' and 'database' for MongoDB
                - Path to CSV file

        Returns:
            Appropriate DataSource instance
        """
        # SemanticLayer
        if hasattr(source, 'query') and hasattr(source, 'schema'):
            return SemanticDataSource(source, **kwargs)

        # DataFrame
        import pandas as pd
        if isinstance(source, pd.DataFrame):
            return DataFrameSource(source, **kwargs)

        # Polars DataFrame
        if hasattr(source, 'to_pandas') and hasattr(source, 'columns'):
            return DataFrameSource(source, **kwargs)

        # Arrow Table
        if hasattr(source, 'to_pandas') and hasattr(source, 'schema'):
            return DataFrameSource(source, **kwargs)

        # MongoDB config dict
        if isinstance(source, dict) and 'uri' in source:
            return MongoDBSource(**source, **kwargs)

        # CSV file path
        if isinstance(source, str) and source.endswith('.csv'):
            df = pd.read_csv(source)
            return DataFrameSource(df, name=os.path.basename(source))

        raise ValueError(f"Cannot create DataSource from {type(source)}")


__all__ = [
    'DataSource',
    'DataSourceResult',
    'SemanticDataSource',
    'DataFrameSource',
    'MongoDBSource',
    'DataSourceFactory',
]
