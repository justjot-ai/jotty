"""
Schema Extractors

Extract database schema from various sources:
- Live SQL database connections (via SQLAlchemy)
- MongoDB collections (via pymongo)
- DDL strings (via sqlglot/simple-ddl-parser)
- Existing schema files
"""

from .base import BaseExtractor
from .database import DatabaseExtractor
from .ddl import DDLExtractor
from .mongodb import MongoDBExtractor

__all__ = ["BaseExtractor", "DatabaseExtractor", "DDLExtractor", "MongoDBExtractor"]
