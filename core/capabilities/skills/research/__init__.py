"""
Research Skills Module
======================

World-class broker-grade research report generation.
"""

from .report_components import (
    CompanySnapshot,
    FinancialStatements,
    DCFModel,
    PeerComparison,
    FinancialTablesFormatter,
    DCFCalculator,
    PeerComparisonFormatter,
    ChartGenerator,
    ReportTemplate,
)

from .data_fetcher import (
    ResearchDataFetcher,
    FinancialDataConverter,
)

from .enhanced_research import enhanced_stock_research_tool

__all__ = [
    # Data Classes
    "CompanySnapshot",
    "FinancialStatements",
    "DCFModel",
    "PeerComparison",
    # Formatters
    "FinancialTablesFormatter",
    "DCFCalculator",
    "PeerComparisonFormatter",
    "ChartGenerator",
    "ReportTemplate",
    # Data Fetcher
    "ResearchDataFetcher",
    "FinancialDataConverter",
    # Main Tool
    "enhanced_stock_research_tool",
]
