"""
Date Preprocessor - Common Date Parsing for All Database Engines

Provides natural language date parsing that works across all database types:
- MongoDB (ISODate strings)
- PostgreSQL, MySQL, SQLite, SQL Server, Oracle (SQL date literals)
"""

import re
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from dateutil.relativedelta import relativedelta


class DateFormat(Enum):
    """Output date format types for different databases."""

    ISO = "iso"  # 2024-01-01T00:00:00Z (MongoDB, general)
    SQL_STANDARD = "sql"  # '2024-01-01 00:00:00' (PostgreSQL, MySQL, SQLite)
    MSSQL = "mssql"  # '2024-01-01T00:00:00' (SQL Server)
    ORACLE = "oracle"  # TO_DATE('2024-01-01', 'YYYY-MM-DD')


class BaseDatePreprocessor(ABC):
    """
    Base class for date preprocessing across all database engines.

    Converts natural language date expressions to database-specific formats.
    Subclasses implement format_date() for their specific database syntax.
    """

    # Patterns for relative date expressions
    DATE_PATTERNS = [
        # "last N days/weeks/months/years"
        (r"last\s+(\d+)\s+days?", "days"),
        (r"last\s+(\d+)\s+weeks?", "weeks"),
        (r"last\s+(\d+)\s+months?", "months"),
        (r"last\s+(\d+)\s+years?", "years"),
        # "past N days/weeks/months/years"
        (r"past\s+(\d+)\s+days?", "days"),
        (r"past\s+(\d+)\s+weeks?", "weeks"),
        (r"past\s+(\d+)\s+months?", "months"),
        (r"past\s+(\d+)\s+years?", "years"),
        # "in the last N days"
        (r"in\s+the\s+last\s+(\d+)\s+days?", "days"),
        (r"in\s+the\s+last\s+(\d+)\s+weeks?", "weeks"),
        (r"in\s+the\s+last\s+(\d+)\s+months?", "months"),
        (r"in\s+the\s+last\s+(\d+)\s+years?", "years"),
        # "within last N days"
        (r"within\s+(?:the\s+)?last\s+(\d+)\s+days?", "days"),
        (r"within\s+(?:the\s+)?last\s+(\d+)\s+weeks?", "weeks"),
        (r"within\s+(?:the\s+)?last\s+(\d+)\s+months?", "months"),
        (r"within\s+(?:the\s+)?last\s+(\d+)\s+years?", "years"),
        # "N days/weeks/months ago"
        (r"(\d+)\s+days?\s+ago", "days"),
        (r"(\d+)\s+weeks?\s+ago", "weeks"),
        (r"(\d+)\s+months?\s+ago", "months"),
        (r"(\d+)\s+years?\s+ago", "years"),
        # "since N days/weeks ago"
        (r"since\s+(\d+)\s+days?\s+ago", "days"),
        (r"since\s+(\d+)\s+weeks?\s+ago", "weeks"),
        (r"since\s+(\d+)\s+months?\s+ago", "months"),
        (r"since\s+(\d+)\s+years?\s+ago", "years"),
    ]

    def __init__(self) -> None:
        """Initialize with special date expressions."""
        # Define special date expressions as lambdas
        self.special_dates = {
            "today": lambda: (
                datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
                datetime.now(),
            ),
            "yesterday": lambda: (
                datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                - timedelta(days=1),
                datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
            ),
            "this week": lambda: (
                datetime.now() - timedelta(days=datetime.now().weekday()),
                datetime.now(),
            ),
            "last week": lambda: (
                datetime.now() - timedelta(days=datetime.now().weekday() + 7),
                datetime.now() - timedelta(days=datetime.now().weekday()),
            ),
            "this month": lambda: (
                datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0),
                datetime.now(),
            ),
            "last month": lambda: (
                (datetime.now().replace(day=1) - timedelta(days=1)).replace(day=1),
                datetime.now().replace(day=1) - timedelta(seconds=1),
            ),
            "this year": lambda: (
                datetime.now().replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0),
                datetime.now(),
            ),
            "last year": lambda: (
                datetime.now().replace(year=datetime.now().year - 1, month=1, day=1),
                datetime.now().replace(month=1, day=1) - timedelta(seconds=1),
            ),
            "this quarter": lambda: self._get_quarter_dates(0),
            "last quarter": lambda: self._get_quarter_dates(-1),
            "ytd": lambda: (
                datetime.now().replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0),
                datetime.now(),
            ),
            "year to date": lambda: (
                datetime.now().replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0),
                datetime.now(),
            ),
            "mtd": lambda: (
                datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0),
                datetime.now(),
            ),
            "month to date": lambda: (
                datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0),
                datetime.now(),
            ),
        }

    @staticmethod
    def _get_quarter_dates(offset: int = 0) -> Tuple[datetime, datetime]:
        """Get start and end dates for a quarter (0=current, -1=last)."""
        now = datetime.now()
        current_quarter = (now.month - 1) // 3
        target_quarter = current_quarter + offset

        # Handle year wraparound
        year = now.year
        while target_quarter < 0:
            target_quarter += 4
            year -= 1
        while target_quarter > 3:
            target_quarter -= 4
            year += 1

        start_month = target_quarter * 3 + 1
        start = datetime(year, start_month, 1)

        # End of quarter
        end_month = start_month + 2
        if end_month == 12:
            end = datetime(year + 1, 1, 1) - timedelta(seconds=1)
        else:
            end = datetime(year, end_month + 1, 1) - timedelta(seconds=1)

        return start, end

    @abstractmethod
    def format_date(self, dt: datetime) -> str:
        """
        Format datetime for the specific database.

        Args:
            dt: datetime object

        Returns:
            Database-specific date string
        """
        pass

    @abstractmethod
    def get_context_hint(self, date_context: Dict[str, str]) -> str:
        """
        Generate context string for LLM about resolved dates.

        Args:
            date_context: Dictionary of resolved dates

        Returns:
            Context string for the LLM prompt
        """
        pass

    def preprocess(self, query: str) -> Tuple[str, Dict[str, str]]:
        """
        Preprocess a natural language query to replace date expressions
        with actual date strings.

        Args:
            query: Natural language query

        Returns:
            Tuple of (modified_query, date_context)
            date_context contains the resolved dates for the LLM
        """
        date_context = {}
        modified_query = query.lower()

        # Check for special date expressions first
        for expr, date_fn in self.special_dates.items():
            if expr in modified_query:
                start_date, end_date = date_fn()
                date_key = expr.replace(" ", "_")
                date_context[f"{date_key}_start"] = self.format_date(start_date)
                date_context[f"{date_key}_end"] = self.format_date(end_date)

                # Replace in query with explicit dates
                replacement = (
                    f"between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}"
                )
                modified_query = modified_query.replace(expr, replacement)

        # Check for relative date patterns
        for pattern, unit in self.DATE_PATTERNS:
            match = re.search(pattern, modified_query, re.IGNORECASE)
            if match:
                num = int(match.group(1))
                end_date = datetime.now()

                if unit == "days":
                    start_date = end_date - timedelta(days=num)
                elif unit == "weeks":
                    start_date = end_date - timedelta(weeks=num)
                elif unit == "months":
                    start_date = end_date - relativedelta(months=num)
                elif unit == "years":
                    start_date = end_date - relativedelta(years=num)
                else:
                    continue

                date_context["start_date"] = self.format_date(start_date)
                date_context["end_date"] = self.format_date(end_date)

                # Replace the matched expression with explicit dates
                replacement = f"after {start_date.strftime('%Y-%m-%d')}"
                modified_query = re.sub(pattern, replacement, modified_query, flags=re.IGNORECASE)
                break  # Only process first match

        return modified_query, date_context


class DatePreprocessor(BaseDatePreprocessor):
    """Date preprocessor for MongoDB - outputs ISO format strings."""

    def format_date(self, dt: datetime) -> str:
        """Format datetime as ISO string for MongoDB."""
        return dt.isoformat()

    def get_context_hint(self, date_context: Dict[str, str]) -> str:
        """Generate MongoDB-specific context hint."""
        if not date_context:
            return ""

        lines = ["\n## Date Context (use these exact dates)"]
        for key, value in date_context.items():
            lines.append(f"- {key}: {value}")
        lines.append("\nIMPORTANT: Use these exact ISO date strings in $match stages.")
        lines.append('Example: {"created_at": {"$gte": "2024-01-01T00:00:00"}}')

        return "\n".join(lines)


class SQLDatePreprocessor(BaseDatePreprocessor):
    """
    Date preprocessor for SQL databases.

    Supports multiple SQL dialects with appropriate date literal formats.
    """

    # SQL dialect-specific date functions
    DIALECT_FORMATS = {
        "postgresql": "'{date}'::timestamp",
        "mysql": "'{date}'",
        "sqlite": "'{date}'",
        "mssql": "CONVERT(datetime, '{date}', 126)",
        "oracle": "TO_TIMESTAMP('{date}', 'YYYY-MM-DD\"T\"HH24:MI:SS')",
        "snowflake": "'{date}'::timestamp",
        "bigquery": "TIMESTAMP('{date}')",
    }

    def __init__(self, dialect: str = "postgresql") -> None:
        """
        Initialize SQL date preprocessor.

        Args:
            dialect: SQL dialect (postgresql, mysql, sqlite, mssql, oracle, etc.)
        """
        super().__init__()
        self.dialect = dialect.lower()

    def format_date(self, dt: datetime) -> str:
        """Format datetime as SQL-appropriate string."""
        # Use ISO format as base - most databases accept this
        iso_str = dt.strftime("%Y-%m-%dT%H:%M:%S")

        # Get dialect-specific format template
        template = self.DIALECT_FORMATS.get(self.dialect, "'{date}'")
        return template.format(date=iso_str)

    def format_date_simple(self, dt: datetime) -> str:
        """Format datetime as simple string (for LLM context)."""
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def get_context_hint(self, date_context: Dict[str, str]) -> str:
        """Generate SQL-specific context hint."""
        if not date_context:
            return ""

        lines = ["\n## Date Context (use these exact dates in WHERE clauses)"]
        for key, value in date_context.items():
            # Show simplified format for context
            lines.append(f"- {key}: {value}")

        # Add dialect-specific hints
        dialect_hints = {
            "postgresql": "Use PostgreSQL timestamp syntax: column >= '2024-01-01'::timestamp",
            "mysql": "Use MySQL datetime syntax: column >= '2024-01-01 00:00:00'",
            "sqlite": "Use SQLite datetime syntax: column >= '2024-01-01 00:00:00'",
            "mssql": "Use SQL Server datetime syntax: column >= '2024-01-01T00:00:00'",
            "oracle": "Use Oracle TO_TIMESTAMP: column >= TO_TIMESTAMP('2024-01-01', 'YYYY-MM-DD')",
        }

        hint = dialect_hints.get(
            self.dialect, "Use standard SQL datetime format: '2024-01-01 00:00:00'"
        )
        lines.append(f"\nIMPORTANT: {hint}")

        return "\n".join(lines)


class DatePreprocessorFactory:
    """
    Factory for creating appropriate date preprocessor based on database type.
    """

    @staticmethod
    def create(db_type: str) -> BaseDatePreprocessor:
        """
        Create date preprocessor for the given database type.

        Args:
            db_type: Database type (mongodb, postgresql, mysql, etc.)

        Returns:
            Appropriate DatePreprocessor instance
        """
        db_type = db_type.lower() if db_type else "postgresql"

        if db_type == "mongodb":
            return DatePreprocessor()
        else:
            return SQLDatePreprocessor(dialect=db_type)

    @staticmethod
    def get_supported_databases() -> List[str]:
        """Get list of supported database types."""
        return [
            "mongodb",
            "postgresql",
            "mysql",
            "sqlite",
            "mssql",
            "oracle",
            "snowflake",
            "bigquery",
        ]


__all__ = [
    "BaseDatePreprocessor",
    "DatePreprocessor",
    "SQLDatePreprocessor",
    "DatePreprocessorFactory",
    "DateFormat",
]
