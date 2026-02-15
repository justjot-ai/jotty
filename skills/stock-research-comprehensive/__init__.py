"""
Research Skills Module
======================

World-class broker-grade research report generation.
"""

from .data_fetcher import FinancialDataConverter, ResearchDataFetcher
from .enhanced_research import enhanced_stock_research_tool
from .report_components import (
    ChartGenerator,
    CompanySnapshot,
    DCFCalculator,
    DCFModel,
    FinancialStatements,
    FinancialTablesFormatter,
    PeerComparison,
    PeerComparisonFormatter,
    ReportTemplate,
)

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
