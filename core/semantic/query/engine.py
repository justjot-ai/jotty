"""
Semantic Query Engine

Natural language to SQL using LookML semantic layer.
Provides rich context to LLMs for accurate query generation.
"""
from typing import Dict, Any, Optional, List
import logging
import re

from ..models import Schema
from ..lookml import LookMLGenerator, LookMLModel
from .date_preprocessor import SQLDatePreprocessor, DatePreprocessorFactory
from .data_loader import ConnectorXLoader, DataLoaderFactory, OutputFormat

logger = logging.getLogger(__name__)


class SemanticQueryEngine:
    """
    Query engine that uses LookML semantic layer for NL-to-SQL.

    Benefits over raw DDL:
    - Relationships are explicit (joins are pre-defined)
    - Dimensions vs Measures are clear
    - Business-friendly labels
    - Aggregation types are specified
    """

    # SQL dialect syntax differences
    DIALECT_HINTS = {
        "postgresql": "Use PostgreSQL syntax. LIMIT for pagination. || for concatenation.",
        "mysql": "Use MySQL syntax. LIMIT for pagination. CONCAT() for strings.",
        "sqlite": "Use SQLite syntax. LIMIT for pagination. || for concatenation.",
        "mssql": "Use T-SQL syntax. TOP or OFFSET-FETCH for pagination. + for concatenation.",
        "oracle": "Use Oracle syntax. FETCH FIRST for pagination. || for concatenation. No LIMIT.",
        "snowflake": "Use Snowflake SQL. LIMIT for pagination. || for concatenation.",
        "bigquery": "Use BigQuery SQL. LIMIT for pagination. CONCAT() for strings.",
    }

    def __init__(
        self,
        schema: Schema = None,
        lookml_model: LookMLModel = None,
        db_type: str = None
    ):
        """
        Initialize query engine.

        Args:
            schema: Database schema (will generate LookML)
            lookml_model: Pre-generated LookML model
            db_type: Database type for SQL dialect
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

        self.db_type = db_type or (schema.database_type if schema else "unknown")
        self._context_cache: Optional[str] = None
        self._date_preprocessor = SQLDatePreprocessor(dialect=self.db_type)
        self._data_loader: Optional[ConnectorXLoader] = None
        self._connection_params: Dict[str, Any] = {}

    def get_context(self) -> str:
        """
        Get LookML context string for LLM.

        Returns:
            Formatted context string with tables, columns, and relationships
        """
        if self._context_cache:
            return self._context_cache

        generator = LookMLGenerator(self.schema) if self.schema else None
        if generator:
            self._context_cache = generator.to_context_string(self.lookml_model)
        else:
            self._context_cache = self._build_context_from_model()

        return self._context_cache

    def _build_context_from_model(self) -> str:
        """Build context string from LookML model."""
        lines = []
        lines.append(f"# Database Schema")
        lines.append("")

        for view in self.lookml_model.views:
            lines.append(f"## {view.name}")
            if view.sql_table_name:
                lines.append(f"Table: {view.sql_table_name}")

            dims = [f"{d.name} ({d.type.value})" for d in view.dimensions]
            if dims:
                lines.append(f"Dimensions: {', '.join(dims)}")

            measures = [f"{m.name} ({m.type.value})" for m in view.measures]
            if measures:
                lines.append(f"Measures: {', '.join(measures)}")

        return "\n".join(lines)

    def generate_sql(
        self,
        question: str,
        execute: bool = False,
        connection_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate SQL from natural language question.

        Args:
            question: Natural language question
            execute: Whether to execute the generated SQL
            connection_params: Database connection parameters for execution

        Returns:
            Dictionary with generated SQL and optional results
        """
        try:
            from core.llm import generate as llm_generate
        except ImportError:
            return {"success": False, "error": "core.llm module not available"}

        # Preprocess dates in the question using common date preprocessor
        processed_question, date_context = self._date_preprocessor.preprocess(question)

        context = self.get_context()

        # Add date context if dates were found
        if date_context:
            context += self._date_preprocessor.get_context_hint(date_context)

        dialect_hint = self.DIALECT_HINTS.get(self.db_type, "Use standard SQL syntax.")

        prompt = self._build_prompt(processed_question, context, dialect_hint)

        # Generate SQL using LLM
        response = llm_generate(
            prompt=prompt,
            model="sonnet",
            provider="claude-cli",
            timeout=120
        )

        if not response.success:
            return {"success": False, "error": response.error}

        # Extract SQL from response
        sql = self._extract_sql(response.text)

        result = {
            "success": True,
            "question": question,
            "processed_question": processed_question if date_context else None,
            "date_context": date_context if date_context else None,
            "generated_sql": sql,
            "db_type": self.db_type,
            "model": response.model,
            "provider": response.provider,
        }

        # Execute if requested
        if execute and sql and connection_params:
            execution_result = self._execute_sql(sql, connection_params)
            result["executed"] = True
            result["query_result"] = execution_result

        return result

    def _build_prompt(self, question: str, context: str, dialect_hint: str) -> str:
        """Build prompt for LLM."""
        return f"""You are a SQL expert. Convert the natural language question to SQL using the provided schema.

{dialect_hint}

{context}

RULES:
1. Return ONLY the SQL query - no explanations, no markdown code blocks
2. Use the exact table and column names from the schema
3. Use appropriate JOINs based on the relationships shown
4. For aggregations, use the measure types indicated (SUM, COUNT, AVG, etc.)
5. Limit results to 100 rows unless otherwise specified
6. Use proper quoting for identifiers if needed
7. Do NOT use DROP, DELETE, UPDATE, INSERT, or any data-modifying statements

Question: {question}

SQL:"""

    def _extract_sql(self, response: str) -> str:
        """
        Extract SQL from LLM response.

        Handles various response formats:
        - Raw SQL
        - Markdown code blocks
        - SQL with explanations
        """
        # Remove markdown code blocks
        code_block_pattern = r'```(?:sql)?\s*(.*?)```'
        matches = re.findall(code_block_pattern, response, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[0].strip()

        # Try to find SELECT statement
        select_pattern = r'(SELECT\s+.*?)(?:;|\Z)'
        matches = re.findall(select_pattern, response, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[0].strip()

        # Check for other SQL statements
        for keyword in ['WITH', 'INSERT', 'UPDATE', 'DELETE', 'CREATE']:
            pattern = rf'({keyword}\s+.*?)(?:;|\Z)'
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            if matches:
                return matches[0].strip()

        # Return cleaned response
        return response.strip()

    def _execute_sql(self, sql: str, connection_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SQL and return results."""
        try:
            from sqlalchemy import create_engine, text
            from urllib.parse import quote_plus

            # Build connection string
            db_type = connection_params.get('db_type', self.db_type)
            conn_string = connection_params.get('connection_string')

            if not conn_string:
                drivers = {
                    "postgresql": "postgresql+psycopg2",
                    "mysql": "mysql+pymysql",
                    "sqlite": "sqlite",
                    "mssql": "mssql+pymssql",
                    "oracle": "oracle+oracledb",
                }

                driver = drivers.get(db_type, db_type)
                host = connection_params.get('host', 'localhost')
                port = connection_params.get('port', '')
                database = connection_params.get('database', '')
                user = connection_params.get('user', '')
                password = quote_plus(connection_params.get('password', ''))

                if db_type == "sqlite":
                    conn_string = f"sqlite:///{database}"
                else:
                    conn_string = f"{driver}://{user}:{password}@{host}:{port}/{database}"

            engine = create_engine(conn_string)

            with engine.connect() as conn:
                result = conn.execute(text(sql))

                # Fetch results
                if result.returns_rows:
                    columns = list(result.keys())
                    rows = [dict(zip(columns, row)) for row in result.fetchall()]

                    return {
                        "success": True,
                        "columns": columns,
                        "rows": rows[:100],  # Limit to 100 rows
                        "row_count": len(rows),
                        "truncated": len(rows) > 100
                    }
                else:
                    return {
                        "success": True,
                        "affected_rows": result.rowcount,
                        "message": f"{result.rowcount} rows affected"
                    }

        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            return {"success": False, "error": str(e)}

    def suggest_queries(self, num_suggestions: int = 5) -> List[str]:
        """
        Suggest common queries based on schema.

        Returns:
            List of suggested natural language queries
        """
        suggestions = []

        for view in self.lookml_model.views[:3]:  # Top 3 views
            # Count query
            suggestions.append(f"How many records are in {view.name}?")

            # Aggregation query
            measures = [m for m in view.measures if m.type.value != "count"]
            if measures:
                m = measures[0]
                suggestions.append(f"What is the total {m.name.replace('total_', '')}?")

            # Group by query
            dims = [d for d in view.dimensions if not d.hidden and not d.primary_key]
            if dims and measures:
                suggestions.append(
                    f"Show {measures[0].name} by {dims[0].name}"
                )

        # Join query
        for explore in self.lookml_model.explores:
            if explore.joins:
                j = explore.joins[0]
                suggestions.append(
                    f"List {explore.name} with their {j.name} details"
                )
                break

        return suggestions[:num_suggestions]

    def validate_sql(self, sql: str) -> Dict[str, Any]:
        """
        Validate SQL syntax using sqlglot.

        Args:
            sql: SQL query to validate

        Returns:
            Validation result with errors if any
        """
        try:
            import sqlglot

            dialect_map = {
                "postgresql": "postgres",
                "mysql": "mysql",
                "sqlite": "sqlite",
                "mssql": "tsql",
                "oracle": "oracle",
            }

            dialect = dialect_map.get(self.db_type, self.db_type)

            # Parse SQL
            parsed = sqlglot.parse(sql, dialect=dialect)

            if not parsed:
                return {"valid": False, "error": "Failed to parse SQL"}

            # Check for dangerous operations
            dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'UPDATE', 'INSERT']
            sql_upper = sql.upper()

            for keyword in dangerous_keywords:
                if keyword in sql_upper:
                    return {
                        "valid": False,
                        "error": f"Dangerous operation detected: {keyword}"
                    }

            return {"valid": True, "parsed": True}

        except Exception as e:
            return {"valid": False, "error": str(e)}

    def set_connection(self, **connection_params):
        """
        Set database connection parameters for query execution.

        Args:
            **connection_params: Connection parameters (host, port, database, user, password, etc.)
        """
        self._connection_params = connection_params
        self._data_loader = None  # Reset loader to use new params

    @property
    def data_loader(self) -> ConnectorXLoader:
        """
        Get or create ConnectorX data loader.

        Returns:
            ConnectorXLoader instance for fast DataFrame loading
        """
        if self._data_loader is None and self._connection_params:
            # Remove db_type from params to avoid duplicate
            params = {k: v for k, v in self._connection_params.items() if k != 'db_type'}
            self._data_loader = DataLoaderFactory.create(
                db_type=self._connection_params.get('db_type', self.db_type),
                **params
            )
        return self._data_loader

    def load_dataframe(
        self,
        query: str,
        output_format: str = "pandas",
        partition_on: str = None,
        partition_num: int = None,
        **kwargs
    ) -> Any:
        """
        Load query results directly into a DataFrame using ConnectorX.

        This is 10-20x faster than traditional Pandas read_sql for large datasets.

        Args:
            query: SQL query to execute
            output_format: Output format ("pandas", "polars", "arrow")
            partition_on: Column to partition on for parallel loading
            partition_num: Number of partitions for parallel loading
            **kwargs: Additional ConnectorX options

        Returns:
            DataFrame in the requested format

        Example:
            # Fast Pandas DataFrame
            df = engine.load_dataframe("SELECT * FROM large_table")

            # Even faster with Polars
            df = engine.load_dataframe("SELECT * FROM large_table", output_format="polars")

            # Parallel loading for very large tables
            df = engine.load_dataframe(
                "SELECT * FROM huge_table",
                partition_on="id",
                partition_num=4
            )
        """
        if not self.data_loader:
            raise ValueError("Connection parameters not set. Call set_connection() first.")

        return self.data_loader.load(
            query=query,
            output_format=OutputFormat(output_format.lower()),
            partition_on=partition_on,
            partition_num=partition_num,
            **kwargs
        )

    def query_to_dataframe(
        self,
        question: str,
        output_format: str = "pandas",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate SQL from natural language and load results into DataFrame.

        Combines NL-to-SQL generation with fast ConnectorX loading.

        Args:
            question: Natural language question
            output_format: Output format ("pandas", "polars", "arrow")
            **kwargs: Additional options

        Returns:
            Dictionary with generated SQL and DataFrame result
        """
        # Generate SQL
        result = self.generate_sql(question, execute=False)

        if not result.get('success'):
            return result

        sql = result.get('generated_sql')

        # Load into DataFrame using ConnectorX
        try:
            df = self.load_dataframe(sql, output_format=output_format, **kwargs)
            result['dataframe'] = df
            result['row_count'] = len(df) if hasattr(df, '__len__') else None
            result['executed'] = True
        except Exception as e:
            result['dataframe_error'] = str(e)
            result['executed'] = False

        return result

    def _execute_sql_fast(
        self,
        sql: str,
        connection_params: Dict[str, Any],
        output_format: str = "pandas"
    ) -> Dict[str, Any]:
        """
        Execute SQL using ConnectorX for faster DataFrame loading.

        Args:
            sql: SQL query to execute
            connection_params: Database connection parameters
            output_format: Output format

        Returns:
            Dictionary with DataFrame results
        """
        try:
            loader = DataLoaderFactory.create(
                db_type=connection_params.get('db_type', self.db_type),
                **connection_params
            )

            df = loader.load(sql, output_format=OutputFormat(output_format.lower()))

            # Convert DataFrame to list of dicts for consistency
            if output_format.lower() == 'pandas':
                rows = df.to_dict('records')
                columns = list(df.columns)
            elif output_format.lower() == 'polars':
                rows = df.to_dicts()
                columns = df.columns
            else:
                # Arrow table
                rows = df.to_pydict()
                columns = df.column_names

            return {
                "success": True,
                "columns": columns,
                "rows": rows[:100],
                "row_count": len(rows),
                "truncated": len(rows) > 100,
                "dataframe": df,
                "loader": "connectorx"
            }

        except Exception as e:
            logger.warning(f"ConnectorX execution failed, falling back to SQLAlchemy: {e}")
            # Fallback to SQLAlchemy
            return self._execute_sql(sql, connection_params)
