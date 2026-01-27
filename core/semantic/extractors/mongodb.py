"""
MongoDB Schema Extractor

Extracts schema from MongoDB collections by sampling documents.
Infers field types, relationships, and structure from actual data.
"""
from typing import List, Optional, Dict, Any, Set
from collections import defaultdict
import logging

from .base import BaseExtractor
from ..models import Column, ForeignKey, Index, ColumnType

logger = logging.getLogger(__name__)


class MongoDBExtractor(BaseExtractor):
    """
    Extract schema from MongoDB by sampling documents.

    MongoDB is schema-less, so we infer schema by:
    1. Sampling documents from each collection
    2. Analyzing field types and frequencies
    3. Detecting relationships from naming conventions (_id references)
    """

    # Map Python/BSON types to normalized column types
    TYPE_MAP = {
        'str': ColumnType.STRING,
        'string': ColumnType.STRING,
        'int': ColumnType.INTEGER,
        'int64': ColumnType.INTEGER,
        'int32': ColumnType.INTEGER,
        'long': ColumnType.INTEGER,
        'float': ColumnType.FLOAT,
        'double': ColumnType.FLOAT,
        'bool': ColumnType.BOOLEAN,
        'datetime': ColumnType.DATETIME,
        'date': ColumnType.DATE,
        'ObjectId': ColumnType.STRING,
        'objectid': ColumnType.STRING,
        'list': ColumnType.JSON,
        'dict': ColumnType.JSON,
        'NoneType': ColumnType.STRING,  # Default to string for nulls
        'Decimal128': ColumnType.DECIMAL,
        'bytes': ColumnType.BINARY,
        'Binary': ColumnType.BINARY,
    }

    def __init__(
        self,
        uri: str = None,
        host: str = "localhost",
        port: int = 27017,
        database: str = "",
        username: str = None,
        password: str = None,
        sample_size: int = 100,
        **kwargs
    ):
        """
        Initialize MongoDB extractor.

        Args:
            uri: Full MongoDB connection URI (overrides other params)
            host: MongoDB host
            port: MongoDB port
            database: Database name
            username: Username for authentication
            password: Password for authentication
            sample_size: Number of documents to sample per collection
            **kwargs: Additional pymongo connection options
        """
        super().__init__(database_type="mongodb")

        self.database_name = database
        self.sample_size = sample_size
        self._client = None
        self._db = None

        # Build connection URI
        if uri:
            self.uri = uri
            # Extract database name from URI if not provided
            if not database and '/' in uri:
                # mongodb://user:pass@host:port/database
                parts = uri.split('/')
                if len(parts) > 3:
                    db_part = parts[3].split('?')[0]
                    if db_part:
                        self.database_name = db_part
        else:
            if username and password:
                self.uri = f"mongodb://{username}:{password}@{host}:{port}/{database}"
            else:
                self.uri = f"mongodb://{host}:{port}/{database}"

        self._collection_schemas: Dict[str, Dict] = {}

    @property
    def client(self):
        """Get or create MongoDB client."""
        if self._client is None:
            from pymongo import MongoClient
            self._client = MongoClient(self.uri)
        return self._client

    @property
    def db(self):
        """Get database instance."""
        if self._db is None:
            self._db = self.client[self.database_name]
        return self._db

    def _extract_tables(self) -> List[str]:
        """Extract list of collection names."""
        try:
            # Get all collection names, excluding system collections
            collections = self.db.list_collection_names()
            return [c for c in collections if not c.startswith('system.')]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    def _extract_columns(self, table_name: str) -> List[Column]:
        """Extract columns by sampling documents from collection."""
        try:
            collection = self.db[table_name]

            # Sample documents
            sample = list(collection.aggregate([
                {"$sample": {"size": self.sample_size}}
            ]))

            if not sample:
                # Try regular find if $sample fails (e.g., small collections)
                sample = list(collection.find().limit(self.sample_size))

            if not sample:
                return []

            # Analyze field types across all sampled documents
            field_types = self._analyze_fields(sample)

            # Convert to Column objects
            columns = []
            for field_name, type_info in field_types.items():
                col = Column(
                    name=field_name,
                    data_type=type_info['most_common_type'],
                    normalized_type=type_info['normalized_type'],
                    nullable=type_info['nullable'],
                    primary_key=(field_name == '_id'),
                )
                columns.append(col)

            # Cache for relationship detection
            self._collection_schemas[table_name] = {
                'fields': field_types,
                'sample_doc': sample[0] if sample else {}
            }

            return columns

        except Exception as e:
            logger.error(f"Failed to extract columns for {table_name}: {e}")
            return []

    def _analyze_fields(self, documents: List[Dict]) -> Dict[str, Dict]:
        """
        Analyze field types across multiple documents.

        Returns dict with field info:
        {
            'field_name': {
                'types': {'str': 50, 'int': 10},
                'most_common_type': 'str',
                'normalized_type': ColumnType.STRING,
                'nullable': True,
                'sample_values': [...]
            }
        }
        """
        field_info = defaultdict(lambda: {
            'types': defaultdict(int),
            'null_count': 0,
            'total_count': 0,
            'sample_values': []
        })

        total_docs = len(documents)

        for doc in documents:
            self._extract_fields_recursive(doc, '', field_info)

        # Process results
        result = {}
        for field_name, info in field_info.items():
            # Find most common type
            types = info['types']
            if types:
                most_common = max(types.keys(), key=lambda t: types[t])
            else:
                most_common = 'str'  # Default to string

            # Determine if nullable (field missing or null in some docs)
            nullable = info['null_count'] > 0 or info['total_count'] < total_docs

            # Get normalized type, try lowercase version too
            normalized = self.TYPE_MAP.get(most_common)
            if normalized is None:
                normalized = self.TYPE_MAP.get(most_common.lower(), ColumnType.STRING)

            result[field_name] = {
                'most_common_type': most_common,
                'normalized_type': normalized,
                'types': dict(types),
                'nullable': nullable,
                'sample_values': info['sample_values'][:5]
            }

        return result

    def _extract_fields_recursive(
        self,
        doc: Dict,
        prefix: str,
        field_info: Dict,
        max_depth: int = 3
    ):
        """Recursively extract fields from nested documents."""
        if max_depth <= 0:
            return

        for key, value in doc.items():
            field_name = f"{prefix}.{key}" if prefix else key

            # Get type name
            type_name = type(value).__name__

            # Handle ObjectId specially
            if type_name == 'ObjectId':
                type_name = 'ObjectId'

            field_info[field_name]['total_count'] += 1

            if value is None:
                field_info[field_name]['null_count'] += 1
                field_info[field_name]['types']['NoneType'] += 1
            else:
                field_info[field_name]['types'][type_name] += 1

                # Store sample values (not for nested objects)
                if type_name not in ['dict', 'list'] and len(field_info[field_name]['sample_values']) < 5:
                    field_info[field_name]['sample_values'].append(str(value)[:100])

                # Recurse into nested documents
                if isinstance(value, dict):
                    self._extract_fields_recursive(value, field_name, field_info, max_depth - 1)

    def _extract_primary_keys(self, table_name: str) -> List[str]:
        """MongoDB always has _id as primary key."""
        return ['_id']

    def _extract_foreign_keys(self, table_name: str) -> List[ForeignKey]:
        """
        Infer foreign keys from field naming conventions.

        Patterns detected:
        - user_id, userId -> users collection
        - author -> authors collection (if exists)
        - *_ids (array of references)
        """
        foreign_keys = []
        schema = self._collection_schemas.get(table_name, {})
        fields = schema.get('fields', {})

        # Get all collection names for matching
        all_collections = set(self._extract_tables())

        for field_name, info in fields.items():
            # Skip _id and nested fields
            if field_name == '_id' or '.' in field_name:
                continue

            # Check for ObjectId type fields (likely references)
            if info['most_common_type'] == 'ObjectId' and field_name != '_id':
                ref_collection = self._infer_collection_name(field_name, all_collections)
                if ref_collection:
                    foreign_keys.append(ForeignKey(
                        columns=[field_name],
                        referenced_table=ref_collection,
                        referenced_columns=['_id']
                    ))

            # Check for _id suffix pattern
            elif field_name.endswith('_id') or field_name.endswith('Id'):
                ref_collection = self._infer_collection_name(field_name, all_collections)
                if ref_collection:
                    foreign_keys.append(ForeignKey(
                        columns=[field_name],
                        referenced_table=ref_collection,
                        referenced_columns=['_id']
                    ))

        return foreign_keys

    def _infer_collection_name(self, field_name: str, all_collections: Set[str]) -> Optional[str]:
        """Infer referenced collection from field name."""
        # Remove common suffixes
        base_name = field_name
        for suffix in ['_id', 'Id', '_ids', 'Ids', 'ID', '_ID']:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break

        # Convert camelCase to snake_case
        import re
        base_name = re.sub('([a-z])([A-Z])', r'\1_\2', base_name).lower()

        # Try various forms
        candidates = [
            base_name,
            base_name + 's',
            base_name + 'es',
            base_name.rstrip('s'),
            base_name.replace('_', ''),
        ]

        for candidate in candidates:
            if candidate in all_collections:
                return candidate
            # Case-insensitive match
            for coll in all_collections:
                if coll.lower() == candidate.lower():
                    return coll

        return None

    def _extract_indexes(self, table_name: str) -> List[Index]:
        """Extract indexes from collection."""
        indexes = []
        try:
            collection = self.db[table_name]
            for idx_info in collection.index_information().values():
                # Skip _id index
                if idx_info.get('key') == [('_id', 1)]:
                    continue

                idx_name = idx_info.get('name', '')
                columns = [k[0] for k in idx_info.get('key', [])]
                unique = idx_info.get('unique', False)

                if columns:
                    indexes.append(Index(
                        name=idx_name,
                        columns=columns,
                        unique=unique
                    ))
        except Exception as e:
            # Suppress warnings for views (not collections)
            if 'is a view' not in str(e):
                logger.warning(f"Failed to get indexes for {table_name}: {e}")

        return indexes

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics for a collection."""
        try:
            stats = self.db.command('collStats', collection_name)
            return {
                'count': stats.get('count', 0),
                'size': stats.get('size', 0),
                'avg_obj_size': stats.get('avgObjSize', 0),
                'storage_size': stats.get('storageSize', 0),
                'indexes': stats.get('nindexes', 0),
            }
        except Exception as e:
            logger.warning(f"Failed to get stats for {collection_name}: {e}")
            return {}

    def test_connection(self) -> Dict[str, Any]:
        """Test MongoDB connection."""
        try:
            # Ping the server
            self.client.admin.command('ping')
            return {
                "success": True,
                "message": "Connected successfully",
                "database": self.database_name,
                "collections": len(self._extract_tables())
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def close(self):
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
