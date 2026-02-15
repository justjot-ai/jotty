"""
Semantic Layer Skill - Database Intelligence & Visualization

Complete semantic layer for database understanding and querying:
- Natural language to SQL/MongoDB query generation
- Schema extraction and analysis from DDL or live databases
- Data visualization from natural language
- LookML semantic model generation
- Support for PostgreSQL, MySQL, SQLite, SQL Server, Oracle, MongoDB

All functionality consolidated into one skill.
"""
import logging
from typing import Dict, Any, Optional, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

logger = logging.getLogger(__name__)
status = SkillStatus("semantic-layer")


# =============================================================================
# SQL QUERY GENERATION
# =============================================================================

@tool_wrapper(required_params=["question"])
def query_database_natural_language(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert natural language question to SQL/MongoDB query using semantic layer.

    Params:
        question: Natural language question (e.g., "Show total sales by region")
        db_type: Database type (postgresql, mysql, sqlite, mssql, oracle, mongodb)
        connection_string: Full connection URL (optional, overrides other params)
        host: Database host (default: localhost)
        port: Database port (uses default if not specified)
        database: Database name
        user: Username
        password: Password
        execute: Execute the query and return results (default: True)
        ddl: DDL string to use instead of live database connection (optional)
        dialect: SQL dialect for DDL mode (default: postgresql)

    Returns:
        {
            "success": True,
            "generated_sql": "SELECT ...",
            "results": [...] (if execute=True),
            "row_count": N (if execute=True),
            "error": None
        }
    """
    status.set_callback(params.pop("_status_callback", None))

    try:
        from .semantic import SemanticLayer

        question = params["question"]
        execute = params.get("execute", True)
        ddl = params.get("ddl")

        status.update("Initializing semantic layer...")

        # Create semantic layer from params
        if ddl:
            layer = SemanticLayer.from_ddl(
                ddl=ddl,
                dialect=params.get("dialect", "postgresql")
            )
        elif params.get("db_type") == "mongodb":
            layer = SemanticLayer.from_mongodb(
                uri=params.get("connection_string"),
                host=params.get("host", "localhost"),
                port=params.get("port", 27017),
                database=params.get("database", ""),
                username=params.get("user"),
                password=params.get("password"),
            )
        else:
            layer = SemanticLayer.from_database(
                db_type=params.get("db_type"),
                connection_string=params.get("connection_string"),
                host=params.get("host", "localhost"),
                port=params.get("port"),
                database=params.get("database", ""),
                user=params.get("user", ""),
                password=params.get("password", ""),
            )

        status.update(f"Generating query for: {question}")

        # Generate query
        result = layer.query(question, execute=execute)

        status.complete("Query generated successfully")

        return tool_response(
            generated_sql=result.get("generated_sql", ""),
            generated_query=result.get("generated_pipeline", ""),  # For MongoDB
            results=result.get("results"),
            row_count=result.get("row_count"),
            explanation=result.get("explanation"),
        )

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return tool_error(f"Missing required library: {e}")
    except Exception as e:
        logger.error(f"Query generation failed: {e}")
        status.error(f"Failed: {e}")
        return tool_error(str(e))


@tool_wrapper(required_params=["question"])
def suggest_related_queries(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate suggested queries based on schema.

    Params:
        question: Initial natural language question
        db_type: Database type
        database: Database name
        host: Database host
        user: Username
        password: Password
        num_suggestions: Number of suggestions to generate (default: 5)
        ddl: DDL string (optional, for DDL mode)

    Returns:
        {
            "success": True,
            "suggestions": ["query1", "query2", ...],
            "count": N
        }
    """
    status.set_callback(params.pop("_status_callback", None))

    try:
        from .semantic import SemanticLayer

        ddl = params.get("ddl")

        if ddl:
            layer = SemanticLayer.from_ddl(ddl=ddl, dialect=params.get("dialect", "postgresql"))
        else:
            layer = SemanticLayer.from_database(
                db_type=params.get("db_type"),
                host=params.get("host", "localhost"),
                database=params.get("database", ""),
                user=params.get("user", ""),
                password=params.get("password", ""),
            )

        num_suggestions = params.get("num_suggestions", 5)
        suggestions = layer.suggest_queries(num_suggestions=num_suggestions)

        status.complete(f"Generated {len(suggestions)} suggestions")

        return tool_response(
            suggestions=suggestions,
            count=len(suggestions)
        )

    except Exception as e:
        logger.error(f"Suggestion generation failed: {e}")
        status.error(f"Failed: {e}")
        return tool_error(str(e))


# =============================================================================
# SCHEMA ANALYSIS
# =============================================================================

@tool_wrapper(required_params=["ddl"])
def analyze_ddl_schema(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse and analyze DDL (Data Definition Language) schema.

    Params:
        ddl: DDL string to parse (CREATE TABLE statements)
        dialect: SQL dialect (postgresql, mysql, sqlite, mssql, oracle) (default: postgresql)
        schema_name: Name for the extracted schema (default: "default")

    Returns:
        {
            "success": True,
            "tables": [...],
            "relationships": [...],
            "table_count": N,
            "total_columns": N
        }
    """
    status.set_callback(params.pop("_status_callback", None))

    try:
        from .extractors import DDLExtractor

        ddl = params["ddl"]
        dialect = params.get("dialect", "postgresql")
        schema_name = params.get("schema_name", "default")

        status.update("Parsing DDL...")

        extractor = DDLExtractor(ddl, dialect)
        schema = extractor.extract(schema_name)

        status.update("Analyzing schema structure...")

        schema.infer_relationships()

        tables_info = []
        total_columns = 0

        for table in schema.tables:
            total_columns += len(table.columns)
            tables_info.append({
                "name": table.name,
                "full_name": table.full_name,
                "column_count": len(table.columns),
                "primary_keys": table.primary_keys,
                "foreign_keys": [
                    {
                        "columns": fk.columns,
                        "referenced_table": fk.referenced_table,
                        "referenced_columns": fk.referenced_columns
                    }
                    for fk in table.foreign_keys
                ],
            })

        relationships_info = [
            {
                "from_table": rel.from_table,
                "to_table": rel.to_table,
                "type": rel.relation_type.value,
            }
            for rel in schema.relationships
        ]

        status.complete(f"Analyzed {len(tables_info)} tables")

        return tool_response(
            tables=tables_info,
            relationships=relationships_info,
            table_count=len(tables_info),
            total_columns=total_columns,
            database_type=schema.database_type
        )

    except Exception as e:
        logger.error(f"DDL analysis failed: {e}")
        status.error(f"Failed: {e}")
        return tool_error(str(e))


@tool_wrapper(required_params=["db_type", "database"])
def extract_database_schema(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and analyze database schema from live connection.

    Params:
        db_type: Database type (postgresql, mysql, sqlite, mssql, oracle, mongodb)
        connection_string: Full connection URL (optional)
        host: Database host (default: localhost)
        port: Database port
        database: Database name
        user: Username
        password: Password
        include_lookml: Generate LookML semantic model (default: False)

    Returns:
        {
            "success": True,
            "schema": {...},
            "lookml": "..." (if include_lookml=True),
            "table_count": N
        }
    """
    status.set_callback(params.pop("_status_callback", None))

    try:
        from .semantic import SemanticLayer

        status.update("Connecting to database...")

        layer = SemanticLayer.from_database(
            db_type=params.get("db_type"),
            connection_string=params.get("connection_string"),
            host=params.get("host", "localhost"),
            port=params.get("port"),
            database=params["database"],
            user=params.get("user", ""),
            password=params.get("password", ""),
        )

        status.update("Extracting schema information...")

        schema_dict = layer.to_dict()
        include_lookml = params.get("include_lookml", False)

        result = {
            "schema": schema_dict,
            "table_count": len(schema_dict.get("tables", [])),
        }

        if include_lookml:
            status.update("Generating LookML model...")
            result["lookml"] = layer.to_lookml()

        status.complete("Schema extracted successfully")

        return tool_response(**result)

    except Exception as e:
        logger.error(f"Schema extraction failed: {e}")
        status.error(f"Failed: {e}")
        return tool_error(str(e))


@tool_wrapper(required_params=["ddl"])
def generate_lookml_from_ddl(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate LookML semantic model from DDL.

    Params:
        ddl: DDL string to parse
        dialect: SQL dialect (default: postgresql)
        model_name: Name for the LookML model (default: "generated_model")

    Returns:
        {
            "success": True,
            "lookml": "...",
            "views": [...],
            "explores": [...]
        }
    """
    status.set_callback(params.pop("_status_callback", None))

    try:
        from .extractors import DDLExtractor
        from .lookml import LookMLGenerator

        ddl = params["ddl"]
        dialect = params.get("dialect", "postgresql")
        model_name = params.get("model_name", "generated_model")

        status.update("Extracting schema from DDL...")

        extractor = DDLExtractor(ddl, dialect)
        schema = extractor.extract(model_name)
        schema.infer_relationships()

        status.update("Generating LookML model...")

        generator = LookMLGenerator(schema)
        lookml_model = generator.generate()
        lookml_string = generator.to_lookml_string(lookml_model)

        status.complete("LookML model generated")

        return tool_response(
            lookml=lookml_string,
            views=[v.name for v in lookml_model.views],
            explores=[e.name for e in lookml_model.explores],
        )

    except Exception as e:
        logger.error(f"LookML generation failed: {e}")
        status.error(f"Failed: {e}")
        return tool_error(str(e))


# =============================================================================
# DATA VISUALIZATION
# =============================================================================

@tool_wrapper(required_params=["question"])
def visualize_data_from_query(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate visualizations from natural language query + data source.

    Params:
        question: Natural language question (e.g., "Show sales trends over time")
        db_type: Database type (postgresql, mysql, mongodb, etc.)
        database: Database name
        host: Database host (default: localhost)
        user: Username
        password: Password
        library: Chart library (matplotlib, seaborn, plotly, altair) (default: matplotlib)
        n_charts: Number of chart variations to generate (default: 1)
        output_format: Output format (base64, html) (default: base64)

    Returns:
        {
            "success": True,
            "charts": [{"base64": "...", "library": "matplotlib"}],
            "chart_count": N
        }
    """
    status.set_callback(params.pop("_status_callback", None))

    try:
        from .semantic import SemanticLayer
        from .visualization import VisualizationLayer

        question = params["question"]
        library = params.get("library", "matplotlib")
        n_charts = params.get("n_charts", 1)
        output_format = params.get("output_format", "base64")

        status.update("Creating semantic layer...")

        semantic_layer = SemanticLayer.from_database(
            db_type=params.get("db_type"),
            host=params.get("host", "localhost"),
            database=params.get("database", ""),
            user=params.get("user", ""),
            password=params.get("password", ""),
        )

        status.update("Initializing visualization layer...")

        viz_layer = VisualizationLayer.from_semantic_layer(semantic_layer)

        status.update(f"Generating {n_charts} chart(s)...")

        charts = viz_layer.visualize(
            question=question,
            library=library,
            n=n_charts
        )

        chart_results = []
        for chart in charts:
            if not chart.success:
                continue

            chart_data = {"library": library}

            if output_format == "base64" and hasattr(chart, 'raster'):
                chart_data["base64"] = chart.raster

            chart_results.append(chart_data)

        status.complete(f"Generated {len(chart_results)} chart(s)")

        return tool_response(
            charts=chart_results,
            chart_count=len(chart_results)
        )

    except ImportError as e:
        return tool_error(f"Missing required library: {e}. Try: pip install lida")
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        status.error(f"Failed: {e}")
        return tool_error(str(e))


@tool_wrapper(required_params=["questions"])
def create_dashboard(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a multi-chart dashboard from multiple questions.

    Params:
        questions: List of natural language questions
        db_type: Database type
        database: Database name
        host: Database host
        user: Username
        password: Password
        library: Chart library (default: matplotlib)
        title: Dashboard title (default: "Dashboard")

    Returns:
        {
            "success": True,
            "dashboard_html": "...",
            "chart_count": N
        }
    """
    status.set_callback(params.pop("_status_callback", None))

    try:
        from .semantic import SemanticLayer
        from .visualization import VisualizationLayer
        from .visualization.renderers import HTMLRenderer

        questions = params["questions"]
        library = params.get("library", "matplotlib")
        title = params.get("title", "Dashboard")

        if not isinstance(questions, list):
            return tool_error("questions must be a list")

        status.update("Creating semantic layer...")

        semantic_layer = SemanticLayer.from_database(
            db_type=params.get("db_type"),
            host=params.get("host", "localhost"),
            database=params.get("database", ""),
            user=params.get("user", ""),
            password=params.get("password", ""),
        )

        viz_layer = VisualizationLayer.from_semantic_layer(semantic_layer)

        status.update(f"Generating dashboard with {len(questions)} charts...")

        charts = viz_layer.dashboard(questions=questions, library=library)

        renderer = HTMLRenderer(title=title)
        render_result = renderer.render_multiple(charts)

        status.complete(f"Dashboard created with {len(charts)} charts")

        return tool_response(
            dashboard_html=render_result.output,
            chart_count=len(charts),
            title=title
        )

    except Exception as e:
        logger.error(f"Dashboard creation failed: {e}")
        status.error(f"Failed: {e}")
        return tool_error(str(e))


__all__ = [
    # SQL/MongoDB Queries
    "query_database_natural_language",
    "suggest_related_queries",

    # Schema Analysis
    "analyze_ddl_schema",
    "extract_database_schema",
    "generate_lookml_from_ddl",

    # Visualization
    "visualize_data_from_query",
    "create_dashboard",
]
