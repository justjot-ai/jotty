"""
MongoDB Query Engine

Generates and executes MongoDB aggregation pipelines from natural language.
"""
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
import json
import re

from ..models import Schema
from ..lookml import LookMLGenerator, LookMLModel
from .date_preprocessor import MongoDBDatePreprocessor, DatePreprocessorFactory

logger = logging.getLogger(__name__)


# Backwards compatibility alias
DatePreprocessor = MongoDBDatePreprocessor


class PipelineValidator:
    """
    Validates and fixes common issues in MongoDB aggregation pipelines.
    """

    # Invalid aggregation stages (db-level commands)
    INVALID_STAGES = {
        '$listCollections', '$listDatabases', '$currentOp',
        '$collStats', '$indexStats', '$planCacheStats'
    }

    # Valid aggregation stages
    VALID_STAGES = {
        '$match', '$group', '$sort', '$limit', '$skip', '$project',
        '$lookup', '$unwind', '$count', '$facet', '$bucket', '$bucketAuto',
        '$sample', '$addFields', '$set', '$unset', '$replaceRoot',
        '$replaceWith', '$merge', '$out', '$unionWith', '$graphLookup',
        '$redact', '$geoNear', '$sortByCount', '$fill', '$densify',
        '$documents', '$setWindowFields'
    }

    @classmethod
    def validate_and_fix(cls, pipeline: List[Dict], schema: 'Schema' = None) -> Tuple[List[Dict], List[str]]:
        """
        Validate and fix common pipeline issues.

        Args:
            pipeline: MongoDB aggregation pipeline
            schema: Optional schema for type information

        Returns:
            Tuple of (fixed_pipeline, list_of_warnings)
        """
        if not pipeline:
            return pipeline, ["Empty pipeline"]

        warnings = []
        fixed_pipeline = []

        for stage in pipeline:
            if not isinstance(stage, dict):
                warnings.append(f"Invalid stage (not a dict): {stage}")
                continue

            # Get stage operator
            stage_keys = list(stage.keys())
            if not stage_keys:
                warnings.append("Empty stage")
                continue

            operator = stage_keys[0]

            # Check for invalid stages
            if operator in cls.INVALID_STAGES:
                warnings.append(f"Removed invalid stage: {operator} (db-level command)")
                continue

            # Check for valid stage
            if operator not in cls.VALID_STAGES and not operator.startswith('$'):
                warnings.append(f"Unknown stage operator: {operator}")

            # Fix common issues in the stage
            fixed_stage = cls._fix_stage(stage, operator, warnings)
            if fixed_stage:
                fixed_pipeline.append(fixed_stage)

        return fixed_pipeline, warnings

    @classmethod
    def _fix_stage(cls, stage: Dict, operator: str, warnings: List[str]) -> Optional[Dict]:
        """Fix common issues in a single stage."""
        stage = stage.copy()

        if operator == '$match':
            stage['$match'] = cls._fix_match(stage['$match'], warnings)
        elif operator == '$group':
            stage['$group'] = cls._fix_group(stage['$group'], warnings)
        elif operator == '$facet':
            stage['$facet'] = cls._fix_facet(stage['$facet'], warnings)
        elif operator == '$lookup':
            stage['$lookup'] = cls._fix_lookup(stage['$lookup'], warnings)

        return stage

    @classmethod
    def _fix_match(cls, match: Dict, warnings: List[str]) -> Dict:
        """Fix common issues in $match stage."""
        if not isinstance(match, dict):
            return match

        match = match.copy()

        # Fix empty $or/$and/$nor arrays
        for op in ['$or', '$and', '$nor']:
            if op in match:
                if not match[op] or (isinstance(match[op], list) and len(match[op]) == 0):
                    warnings.append(f"Removed empty {op} array")
                    del match[op]
                elif isinstance(match[op], list) and len(match[op]) == 1:
                    # Single element $or/$and is redundant
                    if op in ['$or', '$and']:
                        single_condition = match[op][0]
                        del match[op]
                        match.update(single_condition)
                        warnings.append(f"Simplified single-element {op}")

        return match

    @classmethod
    def _fix_group(cls, group: Dict, warnings: List[str]) -> Dict:
        """Fix common issues in $group stage."""
        if not isinstance(group, dict):
            return group

        # Ensure _id field exists
        if '_id' not in group:
            group['_id'] = None
            warnings.append("Added missing _id to $group")

        return group

    @classmethod
    def _fix_facet(cls, facet: Dict, warnings: List[str]) -> Dict:
        """Fix common issues in $facet stage."""
        if not isinstance(facet, dict):
            return facet

        facet = facet.copy()
        for key, subpipeline in list(facet.items()):
            if isinstance(subpipeline, list):
                # Recursively validate sub-pipelines
                fixed_sub, sub_warnings = cls.validate_and_fix(subpipeline)
                facet[key] = fixed_sub
                for w in sub_warnings:
                    warnings.append(f"In $facet.{key}: {w}")

        return facet

    @classmethod
    def _fix_lookup(cls, lookup: Dict, warnings: List[str]) -> Dict:
        """Fix common issues in $lookup stage."""
        if not isinstance(lookup, dict):
            return lookup

        # Check for variable reference in 'from' field
        if 'from' in lookup and lookup['from'].startswith('$'):
            warnings.append(f"$lookup 'from' cannot be a variable: {lookup['from']}")
            # Can't fix this automatically

        return lookup

    @classmethod
    def get_date_fields(cls, schema: 'Schema') -> List[str]:
        """Extract date field names from schema."""
        date_fields = []
        if not schema:
            return date_fields

        for table in schema.tables:
            for col in table.columns:
                if col.normalized_type.value in ['datetime', 'date', 'timestamp']:
                    date_fields.append(col.name)

        return list(set(date_fields))


class MongoDBQueryEngine:
    """
    Query engine for MongoDB using aggregation pipelines.

    Converts natural language to MongoDB aggregation pipelines
    using LLM with LookML semantic context.
    """

    def __init__(
        self,
        schema: Schema = None,
        lookml_model: LookMLModel = None,
        uri: str = None,
        database: str = None
    ):
        """
        Initialize MongoDB query engine.

        Args:
            schema: Database schema
            lookml_model: Pre-generated LookML model
            uri: MongoDB connection URI
            database: Database name
        """
        if lookml_model:
            self.lookml_model = lookml_model
            self.schema = schema
        elif schema:
            self.schema = schema
            generator = LookMLGenerator(schema)
            self.lookml_model = generator.generate()
        else:
            raise ValueError("Either schema or lookml_model is required")

        self.uri = uri
        self.database = database
        self._client = None
        self._db = None
        self._context_cache: Optional[str] = None
        self._date_preprocessor = MongoDBDatePreprocessor()

    @property
    def client(self):
        """Get MongoDB client."""
        if self._client is None and self.uri:
            from pymongo import MongoClient
            self._client = MongoClient(self.uri)
        return self._client

    @property
    def db(self):
        """Get database instance."""
        if self._db is None and self.client and self.database:
            self._db = self.client[self.database]
        return self._db

    def get_context(self) -> str:
        """Get MongoDB-specific context for LLM."""
        if self._context_cache:
            return self._context_cache

        lines = []
        lines.append(f"# MongoDB Database: {self.database}")
        lines.append("")
        lines.append("## Collections and Fields")

        for view in self.lookml_model.views:
            lines.append(f"\n### Collection: {view.name}")

            # Fields (dimensions)
            fields = []
            for d in view.dimensions:
                field_info = f"{d.name} ({d.type.value})"
                if d.primary_key:
                    field_info += " [PK]"
                fields.append(field_info)

            if fields:
                lines.append(f"Fields: {', '.join(fields[:15])}")
                if len(fields) > 15:
                    lines.append(f"  ... and {len(fields) - 15} more fields")

        # Relationships
        if self.lookml_model.explores:
            lines.append("\n## Relationships (for $lookup)")
            for explore in self.lookml_model.explores:
                for join in explore.joins:
                    lines.append(f"- {explore.name} -> {join.name}")

        self._context_cache = "\n".join(lines)
        return self._context_cache

    def generate_pipeline(
        self,
        question: str,
        execute: bool = False,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Generate MongoDB aggregation pipeline from natural language.

        Args:
            question: Natural language question
            execute: Whether to execute the pipeline
            max_retries: Maximum retry attempts on execution failure

        Returns:
            Dictionary with pipeline and optional results
        """
        try:
            from core.llm import generate as llm_generate
        except ImportError:
            return {"success": False, "error": "core.llm module not available"}

        # Preprocess dates in the question using common date preprocessor
        processed_question, date_context = self._date_preprocessor.preprocess(question)

        # Build context with date field awareness
        context = self.get_context()

        # Add date fields information
        date_fields = PipelineValidator.get_date_fields(self.schema)
        if date_fields:
            context += f"\n\n## Date Fields (use proper date comparison)\n"
            context += f"Fields that store dates: {', '.join(date_fields[:10])}\n"
            context += "For date comparisons, dates will be auto-converted to Date objects."

        # Add date context if dates were found
        if date_context:
            context += self._date_preprocessor.get_context_hint(date_context)

        prompt = self._build_prompt(processed_question, context)

        # Generate pipeline using LLM
        response = llm_generate(
            prompt=prompt,
            model="sonnet",
            provider="claude-cli",
            timeout=120
        )

        if not response.success:
            return {"success": False, "error": response.error}

        # Extract pipeline from response
        pipeline_data = self._extract_pipeline(response.text)

        # Validate and fix pipeline
        raw_pipeline = pipeline_data.get("pipeline", [])
        validated_pipeline, validation_warnings = PipelineValidator.validate_and_fix(
            raw_pipeline, self.schema
        )

        # Post-process pipeline to convert ISODate strings to datetime
        pipeline = self._postprocess_pipeline(validated_pipeline)

        result = {
            "success": True,
            "question": question,
            "processed_question": processed_question if date_context else None,
            "date_context": date_context if date_context else None,
            "collection": pipeline_data.get("collection", ""),
            "pipeline": pipeline,
            "validation_warnings": validation_warnings if validation_warnings else None,
            "raw_response": response.text,
        }

        # Execute if requested
        if execute and pipeline and self.db is not None:
            execution_result = self._execute_pipeline(
                pipeline_data["collection"],
                pipeline
            )
            result["executed"] = True
            result["query_result"] = execution_result

            # Retry on execution error
            if not execution_result.get("success") and max_retries > 0:
                retry_result = self._retry_with_error(
                    question=question,
                    original_pipeline=pipeline,
                    error=execution_result.get("error", "Unknown error"),
                    context=context,
                    llm_generate=llm_generate,
                    execute=execute,
                    retries_left=max_retries
                )
                if retry_result:
                    return retry_result

        return result

    def _retry_with_error(
        self,
        question: str,
        original_pipeline: List[Dict],
        error: str,
        context: str,
        llm_generate,
        execute: bool,
        retries_left: int
    ) -> Optional[Dict[str, Any]]:
        """Retry pipeline generation with error feedback."""
        if retries_left <= 0:
            return None

        logger.info(f"Retrying pipeline generation due to error: {error}")

        # Build retry prompt with error feedback
        retry_prompt = f"""You are a MongoDB expert. The previous aggregation pipeline failed with an error.

{context}

PREVIOUS PIPELINE (failed):
{json.dumps(original_pipeline, indent=2, default=str)}

ERROR MESSAGE:
{error}

COMMON FIXES:
- If error mentions "can't subtract string from Date", ensure date fields use proper Date objects
- If error mentions "$or/$and must be nonempty", remove empty arrays or add conditions
- If error mentions invalid stage, use only valid aggregation stages ($match, $group, etc.)
- If error mentions type mismatch, check field types match the operation

Please generate a FIXED pipeline for this question:
{question}

Return ONLY valid JSON with this structure:
{{"collection": "collection_name", "pipeline": [...]}}

JSON:"""

        response = llm_generate(
            prompt=retry_prompt,
            model="sonnet",
            provider="claude-cli",
            timeout=120
        )

        if not response.success:
            return None

        # Extract and process the retry pipeline
        pipeline_data = self._extract_pipeline(response.text)
        validated_pipeline, warnings = PipelineValidator.validate_and_fix(
            pipeline_data.get("pipeline", []), self.schema
        )
        pipeline = self._postprocess_pipeline(validated_pipeline)

        result = {
            "success": True,
            "question": question,
            "collection": pipeline_data.get("collection", ""),
            "pipeline": pipeline,
            "retry_attempt": True,
            "original_error": error,
            "raw_response": response.text,
        }

        # Execute the retry pipeline
        if execute and pipeline and self.db is not None:
            execution_result = self._execute_pipeline(
                pipeline_data["collection"],
                pipeline
            )
            result["executed"] = True
            result["query_result"] = execution_result

            # If still failing, try again
            if not execution_result.get("success") and retries_left > 1:
                return self._retry_with_error(
                    question=question,
                    original_pipeline=pipeline,
                    error=execution_result.get("error", "Unknown error"),
                    context=context,
                    llm_generate=llm_generate,
                    execute=execute,
                    retries_left=retries_left - 1
                )

        return result

    def _build_prompt(self, question: str, context: str) -> str:
        """Build prompt for MongoDB pipeline generation."""
        return f"""You are a MongoDB expert. Generate a MongoDB aggregation pipeline for the question.

{context}

RULES:
1. Return ONLY valid JSON with this structure:
   {{"collection": "collection_name", "pipeline": [...]}}
2. Use proper MongoDB aggregation stages: $match, $group, $sort, $limit, $project, $lookup, $unwind, $facet, $bucket, $sample, $addFields
3. For joins between collections, use $lookup with a LITERAL collection name (not a variable)
4. For date filtering, use ISO date strings: {{"$gte": "2024-01-01T00:00:00Z"}}
5. Always include $limit (default 100) unless counting or using $facet
6. Field names in MongoDB are case-sensitive - use exact names from schema
7. When dates are provided in the question (like "after 2024-12-28"), use those exact dates

AVOID THESE COMMON ERRORS:
- Do NOT use $listCollections, $listDatabases, $currentOp (these are db commands, not aggregation stages)
- Do NOT use empty arrays for $or/$and/$nor - always include at least one condition
- Do NOT use variables in $lookup "from" field - use literal collection names
- For cross-collection queries, use $facet with nested $lookup subpipelines

EXAMPLES:
- Count documents: {{"collection": "ideas", "pipeline": [{{"$count": "total"}}]}}
- Group by field: {{"collection": "ideas", "pipeline": [{{"$group": {{"_id": "$status", "count": {{"$sum": 1}}}}}}, {{"$sort": {{"count": -1}}}}]}}
- Join collections: {{"collection": "ideas", "pipeline": [{{"$lookup": {{"from": "users", "localField": "userId", "foreignField": "_id", "as": "user"}}}}]}}
- Date filter: {{"collection": "orders", "pipeline": [{{"$match": {{"created_at": {{"$gte": "2024-01-01T00:00:00Z"}}}}}}]}}
- Multi-collection stats: {{"collection": "main", "pipeline": [{{"$facet": {{"count_a": [{{"$count": "n"}}], "count_b": [{{"$lookup": {{"from": "other", "pipeline": [{{"$count": "n"}}], "as": "r"}}}}, {{"$unwind": "$r"}}]}}}}]}}

Question: {question}

JSON:"""

    def _extract_pipeline(self, response: str) -> Dict[str, Any]:
        """Extract pipeline JSON from LLM response."""
        # Try to find JSON in response
        response = response.strip()

        # Remove markdown code blocks
        if "```json" in response:
            match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                response = match.group(1)
        elif "```" in response:
            match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                response = match.group(1)

        # Convert MongoDB shell syntax to valid JSON
        response = self._convert_mongo_shell_to_json(response)

        # Try to parse JSON
        try:
            # Find JSON object
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                return {
                    "collection": data.get("collection", ""),
                    "pipeline": data.get("pipeline", [])
                }
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse pipeline JSON: {e}")

        return {"collection": "", "pipeline": []}

    def _convert_mongo_shell_to_json(self, text: str) -> str:
        """
        Convert MongoDB shell syntax to valid JSON.

        Handles:
        - ISODate("...") -> {"$date": "..."}
        - ObjectId("...") -> {"$oid": "..."}
        - NumberLong(...) -> number
        - new Date("...") -> {"$date": "..."}
        """
        # Convert ISODate("...") to {"$date": "..."}
        text = re.sub(
            r'ISODate\s*\(\s*["\']([^"\']+)["\']\s*\)',
            r'{"$date": "\1"}',
            text
        )

        # Convert new Date("...") to {"$date": "..."}
        text = re.sub(
            r'new\s+Date\s*\(\s*["\']([^"\']+)["\']\s*\)',
            r'{"$date": "\1"}',
            text
        )

        # Convert ObjectId("...") to {"$oid": "..."}
        text = re.sub(
            r'ObjectId\s*\(\s*["\']([^"\']+)["\']\s*\)',
            r'{"$oid": "\1"}',
            text
        )

        # Convert NumberLong(...) to just the number
        text = re.sub(
            r'NumberLong\s*\(\s*(\d+)\s*\)',
            r'\1',
            text
        )

        # Convert NumberInt(...) to just the number
        text = re.sub(
            r'NumberInt\s*\(\s*(\d+)\s*\)',
            r'\1',
            text
        )

        return text

    def _postprocess_pipeline(self, pipeline: List[Dict]) -> List[Dict]:
        """
        Post-process pipeline to convert date strings to datetime objects.

        Handles various date formats:
        - ISO date strings: "2024-01-01T00:00:00Z"
        - $date objects: {"$date": "2024-01-01T00:00:00Z"}
        - ISODate function: ISODate("2024-01-01")
        """
        if not pipeline:
            return pipeline

        def convert_dates(obj):
            """Recursively convert date strings to datetime objects."""
            if isinstance(obj, dict):
                # Handle {"$date": "..."} format
                if "$date" in obj and len(obj) == 1:
                    date_val = obj["$date"]
                    if isinstance(date_val, str):
                        return self._parse_date_string(date_val)
                    elif isinstance(date_val, dict):
                        # Nested $date with expressions - try to extract
                        return obj  # Keep as-is for complex expressions

                # Recursively process dict values
                return {k: convert_dates(v) for k, v in obj.items()}

            elif isinstance(obj, list):
                return [convert_dates(item) for item in obj]

            elif isinstance(obj, str):
                # Check if it looks like an ISO date string
                if self._is_iso_date_string(obj):
                    return self._parse_date_string(obj)

            return obj

        return convert_dates(pipeline)

    def _is_iso_date_string(self, s: str) -> bool:
        """Check if string looks like an ISO date."""
        if not isinstance(s, str):
            return False

        # ISO date patterns
        iso_patterns = [
            r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',  # 2024-01-01T00:00:00
            r'^\d{4}-\d{2}-\d{2}$',  # 2024-01-01
        ]

        for pattern in iso_patterns:
            if re.match(pattern, s):
                return True
        return False

    def _parse_date_string(self, date_str: str) -> datetime:
        """Parse various date string formats to datetime."""
        # Remove trailing Z and handle timezone
        date_str = date_str.rstrip('Z')

        formats = [
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d',
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        # If all else fails, try dateutil
        try:
            from dateutil.parser import parse
            return parse(date_str)
        except Exception:
            logger.warning(f"Could not parse date string: {date_str}")
            return date_str  # Return original if can't parse

    def _execute_pipeline(
        self,
        collection_name: str,
        pipeline: List[Dict]
    ) -> Dict[str, Any]:
        """Execute MongoDB aggregation pipeline."""
        try:
            if self.db is None:
                return {"success": False, "error": "No database connection"}

            collection = self.db[collection_name]
            results = list(collection.aggregate(pipeline))

            # Convert ObjectId to string for JSON serialization
            serialized = []
            for doc in results[:100]:
                serialized.append(self._serialize_doc(doc))

            return {
                "success": True,
                "rows": serialized,
                "row_count": len(serialized),
                "truncated": len(results) > 100
            }

        except Exception as e:
            logger.error(f"Pipeline execution error: {e}")
            return {"success": False, "error": str(e)}

    def _serialize_doc(self, doc: Dict) -> Dict:
        """Serialize MongoDB document for JSON output."""
        from bson import ObjectId
        from datetime import datetime

        result = {}
        for key, value in doc.items():
            if isinstance(value, ObjectId):
                result[key] = str(value)
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, dict):
                result[key] = self._serialize_doc(value)
            elif isinstance(value, list):
                result[key] = [
                    self._serialize_doc(v) if isinstance(v, dict)
                    else str(v) if isinstance(v, ObjectId)
                    else v
                    for v in value
                ]
            else:
                result[key] = value
        return result

    def suggest_queries(self, num_suggestions: int = 5) -> List[str]:
        """Suggest queries based on schema."""
        suggestions = []

        for view in self.lookml_model.views[:3]:
            suggestions.append(f"How many documents are in {view.name}?")
            suggestions.append(f"Show {view.name} grouped by status")

        for explore in self.lookml_model.explores:
            if explore.joins:
                j = explore.joins[0]
                suggestions.append(f"List {explore.name} with their {j.name}")
                break

        return suggestions[:num_suggestions]
