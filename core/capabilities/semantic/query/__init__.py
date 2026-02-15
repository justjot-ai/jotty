"""
Semantic Query Engine

Natural language to SQL/MongoDB using LookML semantic context.

Includes:
- SemanticQueryEngine: NL-to-SQL with LookML context
- MongoDBQueryEngine: NL-to-MongoDB aggregation pipelines
- DatePreprocessor: Natural language date parsing
- DataLoader: High-performance DataFrame loading via ConnectorX
"""
from .engine import SemanticQueryEngine
from .mongodb_engine import MongoDBQueryEngine
from .date_preprocessor import (
    BaseDatePreprocessor,
    SQLDatePreprocessor,
    DatePreprocessor,
    DatePreprocessorFactory,
)
from .data_loader import (
    BaseDataLoader,
    ConnectorXLoader,
    SQLAlchemyLoader,
    DataLoaderFactory,
    OutputFormat,
)

__all__ = [
    # Query Engines
    'SemanticQueryEngine',
    'MongoDBQueryEngine',

    # Date Preprocessing
    'BaseDatePreprocessor',
    'SQLDatePreprocessor',
    'DatePreprocessor',
    'DatePreprocessorFactory',

    # Data Loading (ConnectorX)
    'BaseDataLoader',
    'ConnectorXLoader',
    'SQLAlchemyLoader',
    'DataLoaderFactory',
    'OutputFormat',
]
