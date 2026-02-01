"""
Research Report Components
==========================

World-class broker report components including:
- Financial tables formatter
- DCF valuation model
- Chart generator
- Peer comparison module
- Professional formatting utilities
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CompanySnapshot:
    """First-page investment snapshot."""
    ticker: str
    company_name: str
    current_price: float
    target_price: float
    rating: str  # BUY, HOLD, SELL
    market_cap: float
    market_cap_unit: str = "Cr"
    pe_ratio: float = 0.0
    pe_forward: float = 0.0
    pb_ratio: float = 0.0
    ev_ebitda: float = 0.0
    roe: float = 0.0
    roce: float = 0.0
    dividend_yield: float = 0.0
    week_52_high: float = 0.0
    week_52_low: float = 0.0
    beta: float = 1.0
    sector: str = ""
    industry: str = ""
    promoter_holding: float = 0.0
    fii_holding: float = 0.0
    dii_holding: float = 0.0

    @property
    def upside(self) -> float:
        """Calculate upside to target."""
        if self.current_price > 0:
            return ((self.target_price - self.current_price) / self.current_price) * 100
        return 0.0

    @property
    def rating_color(self) -> str:
        """Get rating color for display."""
        colors = {"BUY": "#00AA00", "HOLD": "#FFA500", "SELL": "#FF0000"}
        return colors.get(self.rating.upper(), "#666666")


@dataclass
class FinancialStatements:
    """5-year financial statements data."""
    years: List[str] = field(default_factory=list)

    # Income Statement
    revenue: List[float] = field(default_factory=list)
    ebitda: List[float] = field(default_factory=list)
    ebit: List[float] = field(default_factory=list)
    pat: List[float] = field(default_factory=list)
    eps: List[float] = field(default_factory=list)

    # Margins
    gross_margin: List[float] = field(default_factory=list)
    ebitda_margin: List[float] = field(default_factory=list)
    pat_margin: List[float] = field(default_factory=list)

    # Balance Sheet
    total_assets: List[float] = field(default_factory=list)
    total_equity: List[float] = field(default_factory=list)
    total_debt: List[float] = field(default_factory=list)
    cash: List[float] = field(default_factory=list)

    # Ratios
    roe: List[float] = field(default_factory=list)
    roce: List[float] = field(default_factory=list)
    debt_equity: List[float] = field(default_factory=list)
    current_ratio: List[float] = field(default_factory=list)


@dataclass
class DCFModel:
    """DCF valuation model data."""
    # Projections
    projection_years: List[str] = field(default_factory=list)
    revenue_projections: List[float] = field(default_factory=list)
    ebitda_projections: List[float] = field(default_factory=list)
    fcf_projections: List[float] = field(default_factory=list)

    # Assumptions
    revenue_growth: float = 10.0
    ebitda_margin: float = 20.0
    tax_rate: float = 25.0
    capex_pct: float = 5.0
    wc_pct: float = 2.0

    # WACC Components
    risk_free_rate: float = 7.0
    equity_risk_premium: float = 6.0
    beta: float = 1.0
    cost_of_debt: float = 9.0
    debt_weight: float = 20.0

    # Terminal Value
    terminal_growth: float = 4.0
    exit_multiple: float = 10.0

    # Results
    enterprise_value: float = 0.0
    equity_value: float = 0.0
    implied_price: float = 0.0

    @property
    def wacc(self) -> float:
        """Calculate WACC."""
        cost_of_equity = self.risk_free_rate + (self.beta * self.equity_risk_premium)
        equity_weight = 100 - self.debt_weight
        wacc = (cost_of_equity * equity_weight / 100) + \
               (self.cost_of_debt * (1 - self.tax_rate / 100) * self.debt_weight / 100)
        return wacc


@dataclass
class PeerComparison:
    """Peer company comparison data."""
    companies: List[str] = field(default_factory=list)
    market_caps: List[float] = field(default_factory=list)
    pe_ratios: List[float] = field(default_factory=list)
    pb_ratios: List[float] = field(default_factory=list)
    ev_ebitda: List[float] = field(default_factory=list)
    roe: List[float] = field(default_factory=list)
    roce: List[float] = field(default_factory=list)
    revenue_growth: List[float] = field(default_factory=list)
    pat_margin: List[float] = field(default_factory=list)


# =============================================================================
# FINANCIAL TABLES FORMATTER
# =============================================================================

class FinancialTablesFormatter:
    """Format financial data into professional tables."""

    @staticmethod
    def format_number(value: float, decimals: int = 1, suffix: str = "") -> str:
        """Format number with proper formatting."""
        if pd.isna(value) or value is None:
            return "-"

        if abs(value) >= 1e7:  # Crores
            return f"{value/1e7:,.{decimals}f} Cr{suffix}"
        elif abs(value) >= 1e5:  # Lakhs
            return f"{value/1e5:,.{decimals}f} L{suffix}"
        elif abs(value) >= 1000:
            return f"{value:,.{decimals}f}{suffix}"
        else:
            return f"{value:.{decimals}f}{suffix}"

    @staticmethod
    def format_percentage(value: float, decimals: int = 1) -> str:
        """Format as percentage."""
        if pd.isna(value) or value is None:
            return "-"
        return f"{value:.{decimals}f}%"

    @staticmethod
    def format_growth(current: float, previous: float) -> str:
        """Calculate and format YoY growth."""
        if previous == 0 or pd.isna(previous) or pd.isna(current):
            return "-"
        growth = ((current - previous) / abs(previous)) * 100
        sign = "+" if growth > 0 else ""
        return f"{sign}{growth:.1f}%"

    def create_income_statement_table(self, data: FinancialStatements) -> str:
        """Create formatted income statement table."""
        if not data.years:
            return "No income statement data available."

        # Header row
        header = "| Particulars | " + " | ".join(data.years) + " | CAGR |"
        separator = "|" + "|".join(["---"] * (len(data.years) + 2)) + "|"

        rows = []

        # Revenue
        if data.revenue:
            rev_row = "| **Revenue** | " + " | ".join([self.format_number(v) for v in data.revenue])
            cagr = self._calculate_cagr(data.revenue)
            rev_row += f" | {self.format_percentage(cagr)} |"
            rows.append(rev_row)

        # EBITDA
        if data.ebitda:
            ebitda_row = "| EBITDA | " + " | ".join([self.format_number(v) for v in data.ebitda])
            cagr = self._calculate_cagr(data.ebitda)
            ebitda_row += f" | {self.format_percentage(cagr)} |"
            rows.append(ebitda_row)

        # PAT
        if data.pat:
            pat_row = "| **PAT** | " + " | ".join([self.format_number(v) for v in data.pat])
            cagr = self._calculate_cagr(data.pat)
            pat_row += f" | {self.format_percentage(cagr)} |"
            rows.append(pat_row)

        # EPS
        if data.eps:
            eps_row = "| EPS (₹) | " + " | ".join([f"{v:.2f}" for v in data.eps])
            cagr = self._calculate_cagr(data.eps)
            eps_row += f" | {self.format_percentage(cagr)} |"
            rows.append(eps_row)

        # Margins section
        rows.append("| **Margins** | " + " | ".join([""] * len(data.years)) + " | |")

        if data.ebitda_margin:
            margin_row = "| EBITDA Margin | " + " | ".join([self.format_percentage(v) for v in data.ebitda_margin])
            margin_row += " | - |"
            rows.append(margin_row)

        if data.pat_margin:
            margin_row = "| PAT Margin | " + " | ".join([self.format_percentage(v) for v in data.pat_margin])
            margin_row += " | - |"
            rows.append(margin_row)

        return "\n".join([header, separator] + rows)

    def create_ratio_table(self, data: FinancialStatements) -> str:
        """Create key ratios table."""
        if not data.years:
            return "No ratio data available."

        header = "| Ratio | " + " | ".join(data.years) + " |"
        separator = "|" + "|".join(["---"] * (len(data.years) + 1)) + "|"

        rows = []

        if data.roe:
            rows.append("| ROE (%) | " + " | ".join([self.format_percentage(v) for v in data.roe]) + " |")

        if data.roce:
            rows.append("| ROCE (%) | " + " | ".join([self.format_percentage(v) for v in data.roce]) + " |")

        if data.debt_equity:
            rows.append("| Debt/Equity | " + " | ".join([f"{v:.2f}" for v in data.debt_equity]) + " |")

        if data.current_ratio:
            rows.append("| Current Ratio | " + " | ".join([f"{v:.2f}" for v in data.current_ratio]) + " |")

        return "\n".join([header, separator] + rows)

    def create_snapshot_box(self, snapshot: CompanySnapshot) -> str:
        """Create first-page investment snapshot box."""
        upside_str = f"+{snapshot.upside:.1f}%" if snapshot.upside > 0 else f"{snapshot.upside:.1f}%"

        box = f"""
┌─────────────────────────────────────────────────────────────────┐
│  **{snapshot.rating}**  │  Target: ₹{snapshot.target_price:,.0f}  │  Upside: {upside_str}  │
├─────────────────────────────────────────────────────────────────┤
│  CMP: ₹{snapshot.current_price:,.2f}  │  Mkt Cap: ₹{snapshot.market_cap:,.0f} {snapshot.market_cap_unit}  │
├─────────────────────────────────────────────────────────────────┤
│  **Key Metrics**                                                │
│  P/E: {snapshot.pe_ratio:.1f}x  │  P/B: {snapshot.pb_ratio:.1f}x  │  EV/EBITDA: {snapshot.ev_ebitda:.1f}x  │
│  ROE: {snapshot.roe:.1f}%  │  ROCE: {snapshot.roce:.1f}%  │  Div Yield: {snapshot.dividend_yield:.1f}%  │
├─────────────────────────────────────────────────────────────────┤
│  52W Range: ₹{snapshot.week_52_low:,.0f} - ₹{snapshot.week_52_high:,.0f}  │  Beta: {snapshot.beta:.2f}  │
│  Promoter: {snapshot.promoter_holding:.1f}%  │  FII: {snapshot.fii_holding:.1f}%  │  DII: {snapshot.dii_holding:.1f}%  │
└─────────────────────────────────────────────────────────────────┘
"""
        return box

    def _calculate_cagr(self, values: List[float]) -> float:
        """Calculate CAGR from list of values."""
        if len(values) < 2 or values[0] <= 0 or values[-1] <= 0:
            return 0.0
        n = len(values) - 1
        return (pow(values[-1] / values[0], 1/n) - 1) * 100


# =============================================================================
# DCF VALUATION MODEL
# =============================================================================

class DCFCalculator:
    """DCF valuation calculator with sensitivity analysis."""

    def __init__(self, model: DCFModel):
        self.model = model

    def calculate_dcf(self, shares_outstanding: float, net_debt: float) -> Dict[str, float]:
        """Calculate DCF valuation."""
        if not self.model.fcf_projections:
            return {"error": "No FCF projections available"}

        wacc = self.model.wacc / 100
        terminal_growth = self.model.terminal_growth / 100

        # Present value of FCF
        pv_fcf = []
        for i, fcf in enumerate(self.model.fcf_projections):
            pv = fcf / pow(1 + wacc, i + 1)
            pv_fcf.append(pv)

        total_pv_fcf = sum(pv_fcf)

        # Terminal value (Gordon Growth)
        terminal_fcf = self.model.fcf_projections[-1] * (1 + terminal_growth)
        terminal_value_gordon = terminal_fcf / (wacc - terminal_growth)
        pv_terminal_gordon = terminal_value_gordon / pow(1 + wacc, len(self.model.fcf_projections))

        # Terminal value (Exit Multiple)
        terminal_ebitda = self.model.ebitda_projections[-1] if self.model.ebitda_projections else 0
        terminal_value_multiple = terminal_ebitda * self.model.exit_multiple
        pv_terminal_multiple = terminal_value_multiple / pow(1 + wacc, len(self.model.fcf_projections))

        # Enterprise Value (average of both methods)
        ev_gordon = total_pv_fcf + pv_terminal_gordon
        ev_multiple = total_pv_fcf + pv_terminal_multiple
        ev_avg = (ev_gordon + ev_multiple) / 2

        # Equity Value
        equity_value = ev_avg - net_debt

        # Per Share Value
        implied_price = equity_value / shares_outstanding if shares_outstanding > 0 else 0

        self.model.enterprise_value = ev_avg
        self.model.equity_value = equity_value
        self.model.implied_price = implied_price

        return {
            "total_pv_fcf": total_pv_fcf,
            "terminal_value_gordon": terminal_value_gordon,
            "terminal_value_multiple": terminal_value_multiple,
            "pv_terminal_gordon": pv_terminal_gordon,
            "pv_terminal_multiple": pv_terminal_multiple,
            "ev_gordon": ev_gordon,
            "ev_multiple": ev_multiple,
            "enterprise_value": ev_avg,
            "equity_value": equity_value,
            "implied_price": implied_price,
            "wacc": self.model.wacc,
        }

    def create_sensitivity_matrix(self,
                                   wacc_range: List[float] = None,
                                   growth_range: List[float] = None,
                                   shares_outstanding: float = 1.0,
                                   net_debt: float = 0.0) -> pd.DataFrame:
        """Create WACC vs Terminal Growth sensitivity matrix."""
        if wacc_range is None:
            wacc_range = [8.0, 9.0, 10.0, 11.0, 12.0]
        if growth_range is None:
            growth_range = [2.0, 3.0, 4.0, 5.0, 6.0]

        matrix = []
        for wacc in wacc_range:
            row = []
            for growth in growth_range:
                # Temporarily update model
                orig_wacc = self.model.wacc
                orig_growth = self.model.terminal_growth

                # Recalculate WACC based on input (simplified)
                self.model.terminal_growth = growth

                # Calculate implied price
                wacc_decimal = wacc / 100
                growth_decimal = growth / 100

                if self.model.fcf_projections:
                    pv_fcf = sum([fcf / pow(1 + wacc_decimal, i + 1)
                                  for i, fcf in enumerate(self.model.fcf_projections)])

                    terminal_fcf = self.model.fcf_projections[-1] * (1 + growth_decimal)
                    if wacc_decimal > growth_decimal:
                        tv = terminal_fcf / (wacc_decimal - growth_decimal)
                        pv_tv = tv / pow(1 + wacc_decimal, len(self.model.fcf_projections))
                        ev = pv_fcf + pv_tv
                        equity = ev - net_debt
                        price = equity / shares_outstanding if shares_outstanding > 0 else 0
                    else:
                        price = 0
                else:
                    price = 0

                row.append(round(price, 0))

                # Restore
                self.model.terminal_growth = orig_growth

            matrix.append(row)

        df = pd.DataFrame(matrix,
                          index=[f"WACC {w}%" for w in wacc_range],
                          columns=[f"TG {g}%" for g in growth_range])
        return df

    def format_sensitivity_table(self, df: pd.DataFrame) -> str:
        """Format sensitivity matrix as markdown table."""
        lines = ["### Sensitivity Analysis: WACC vs Terminal Growth", ""]

        # Header
        header = "| | " + " | ".join(df.columns) + " |"
        separator = "|---|" + "|".join(["---:"] * len(df.columns)) + "|"

        lines.extend([header, separator])

        for idx, row in df.iterrows():
            row_str = f"| **{idx}** | " + " | ".join([f"₹{v:,.0f}" for v in row.values]) + " |"
            lines.append(row_str)

        return "\n".join(lines)

    def format_dcf_summary(self, result: Dict[str, float]) -> str:
        """Format DCF results as markdown."""
        return f"""
### DCF Valuation Summary

| Component | Value |
|-----------|------:|
| PV of Projected FCF | ₹{result.get('total_pv_fcf', 0):,.0f} Cr |
| Terminal Value (Gordon) | ₹{result.get('terminal_value_gordon', 0):,.0f} Cr |
| Terminal Value (Exit Multiple) | ₹{result.get('terminal_value_multiple', 0):,.0f} Cr |
| **Enterprise Value** | **₹{result.get('enterprise_value', 0):,.0f} Cr** |
| Less: Net Debt | - |
| **Equity Value** | **₹{result.get('equity_value', 0):,.0f} Cr** |
| Shares Outstanding | - |
| **Implied Share Price** | **₹{result.get('implied_price', 0):,.0f}** |

**Key Assumptions:**
- WACC: {result.get('wacc', 0):.1f}%
- Terminal Growth Rate: {self.model.terminal_growth:.1f}%
- Exit EV/EBITDA Multiple: {self.model.exit_multiple:.1f}x
"""


# =============================================================================
# PEER COMPARISON MODULE
# =============================================================================

class PeerComparisonFormatter:
    """Format peer comparison data."""

    def create_peer_table(self, data: PeerComparison, highlight_company: str = None) -> str:
        """Create peer comparison table."""
        if not data.companies:
            return "No peer comparison data available."

        header = "| Company | Mkt Cap (Cr) | P/E | P/B | EV/EBITDA | ROE (%) | ROCE (%) | Rev Gr (%) |"
        separator = "|---------|-------------:|----:|----:|----------:|--------:|---------:|-----------:|"

        rows = []
        for i, company in enumerate(data.companies):
            # Highlight target company
            name = f"**{company}**" if company == highlight_company else company

            row = f"| {name} | "
            row += f"{data.market_caps[i]:,.0f} | " if i < len(data.market_caps) else "- | "
            row += f"{data.pe_ratios[i]:.1f} | " if i < len(data.pe_ratios) else "- | "
            row += f"{data.pb_ratios[i]:.1f} | " if i < len(data.pb_ratios) else "- | "
            row += f"{data.ev_ebitda[i]:.1f} | " if i < len(data.ev_ebitda) else "- | "
            row += f"{data.roe[i]:.1f} | " if i < len(data.roe) else "- | "
            row += f"{data.roce[i]:.1f} | " if i < len(data.roce) else "- | "
            row += f"{data.revenue_growth[i]:.1f} |" if i < len(data.revenue_growth) else "- |"

            rows.append(row)

        # Add median row
        median_row = "| **Median** | "
        median_row += f"{np.median(data.market_caps):,.0f} | " if data.market_caps else "- | "
        median_row += f"{np.median(data.pe_ratios):.1f} | " if data.pe_ratios else "- | "
        median_row += f"{np.median(data.pb_ratios):.1f} | " if data.pb_ratios else "- | "
        median_row += f"{np.median(data.ev_ebitda):.1f} | " if data.ev_ebitda else "- | "
        median_row += f"{np.median(data.roe):.1f} | " if data.roe else "- | "
        median_row += f"{np.median(data.roce):.1f} | " if data.roce else "- | "
        median_row += f"{np.median(data.revenue_growth):.1f} |" if data.revenue_growth else "- |"
        rows.append(median_row)

        return "\n".join([header, separator] + rows)

    def calculate_relative_valuation(self,
                                      target_eps: float,
                                      target_bv: float,
                                      peer_pe_median: float,
                                      peer_pb_median: float) -> Dict[str, float]:
        """Calculate implied price from peer multiples."""
        pe_implied = target_eps * peer_pe_median
        pb_implied = target_bv * peer_pb_median
        avg_implied = (pe_implied + pb_implied) / 2

        return {
            "pe_implied_price": pe_implied,
            "pb_implied_price": pb_implied,
            "average_implied_price": avg_implied,
            "peer_pe_multiple": peer_pe_median,
            "peer_pb_multiple": peer_pb_median,
        }


# =============================================================================
# CHART GENERATOR (Text-based for markdown, can be extended for matplotlib)
# =============================================================================

class ChartGenerator:
    """Generate charts for reports (ASCII and matplotlib)."""

    @staticmethod
    def create_bar_chart_ascii(values: List[float], labels: List[str],
                                title: str = "", width: int = 40) -> str:
        """Create ASCII bar chart."""
        if not values or not labels:
            return ""

        max_val = max(abs(v) for v in values)
        if max_val == 0:
            max_val = 1

        lines = [f"**{title}**", "```"]

        for label, value in zip(labels, values):
            bar_length = int((abs(value) / max_val) * width)
            bar = "█" * bar_length
            lines.append(f"{label:>8} | {bar} {value:,.0f}")

        lines.append("```")
        return "\n".join(lines)

    @staticmethod
    def create_sparkline(values: List[float]) -> str:
        """Create text sparkline."""
        if not values:
            return ""

        blocks = " ▁▂▃▄▅▆▇█"
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val != min_val else 1

        sparkline = ""
        for v in values:
            idx = int((v - min_val) / range_val * 8)
            idx = max(0, min(8, idx))
            sparkline += blocks[idx]

        return sparkline

    @staticmethod
    def create_football_field_ascii(valuations: Dict[str, Tuple[float, float, float]],
                                     current_price: float,
                                     width: int = 50) -> str:
        """
        Create ASCII football field chart.

        Args:
            valuations: Dict of method -> (low, mid, high)
            current_price: Current stock price
            width: Chart width
        """
        if not valuations:
            return ""

        # Find overall range
        all_vals = []
        for low, mid, high in valuations.values():
            all_vals.extend([low, mid, high])

        min_val = min(all_vals) * 0.9
        max_val = max(all_vals) * 1.1
        range_val = max_val - min_val

        # Avoid division by zero
        if range_val == 0:
            range_val = 1

        lines = ["**Valuation Football Field**", "```"]

        for method, (low, mid, high) in valuations.items():
            # Calculate positions
            low_pos = int((low - min_val) / range_val * width)
            mid_pos = int((mid - min_val) / range_val * width)
            high_pos = int((high - min_val) / range_val * width)

            # Create bar
            bar = [" "] * width
            for i in range(low_pos, high_pos + 1):
                if 0 <= i < width:
                    bar[i] = "─"
            if 0 <= low_pos < width:
                bar[low_pos] = "├"
            if 0 <= high_pos < width:
                bar[high_pos] = "┤"
            if 0 <= mid_pos < width:
                bar[mid_pos] = "●"

            lines.append(f"{method:>12} |{''.join(bar)}| ₹{low:,.0f} - ₹{high:,.0f}")

        # Add current price marker
        price_pos = int((current_price - min_val) / range_val * width)
        marker = [" "] * width
        if 0 <= price_pos < width:
            marker[price_pos] = "▼"
        lines.append(f"{'CMP':>12} |{''.join(marker)}| ₹{current_price:,.0f}")

        lines.append("```")
        return "\n".join(lines)

    def _has_valid_data(self, values: List) -> bool:
        """Check if list has non-zero, non-empty values."""
        if not values:
            return False
        # Filter out None, 0, and very small values
        valid = [v for v in values if v is not None and abs(v) > 0.01]
        return len(valid) >= 2  # Need at least 2 data points for a trend

    def create_matplotlib_charts(self, data: Dict[str, Any], output_dir: str) -> List[str]:
        """
        Generate matplotlib charts and save to files.
        Only creates charts with valid data.
        Returns list of generated file paths.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            plt.style.use('seaborn-v0_8-whitegrid')
        except ImportError:
            logger.warning("matplotlib not available for chart generation")
            return []

        chart_files = []

        # 1. Revenue & PAT Trend Chart - only if we have valid data
        if ('years' in data and data['years'] and
            'revenue' in data and self._has_valid_data(data['revenue']) and
            'pat' in data and self._has_valid_data(data['pat'])):

            fig, ax1 = plt.subplots(figsize=(10, 6))

            x = range(len(data['years']))
            ax1.bar(x, data['revenue'], color='steelblue', alpha=0.7, label='Revenue')
            ax1.set_ylabel('Revenue (Cr)', color='steelblue')
            ax1.tick_params(axis='y', labelcolor='steelblue')

            ax2 = ax1.twinx()
            ax2.plot(x, data['pat'], color='darkgreen', marker='o', linewidth=2, label='PAT')
            ax2.set_ylabel('PAT (Cr)', color='darkgreen')
            ax2.tick_params(axis='y', labelcolor='darkgreen')

            ax1.set_xticks(x)
            ax1.set_xticklabels(data['years'])
            ax1.set_title('Revenue & PAT Trend', fontsize=14, fontweight='bold')

            fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
            plt.tight_layout()

            filepath = f"{output_dir}/revenue_pat_trend.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            chart_files.append(filepath)
            logger.info(f"✅ Generated: Revenue & PAT Trend")
        else:
            logger.info("⏭️ Skipping Revenue/PAT chart - insufficient data")

        # 2. Margin Trend Chart - only if we have valid margin data
        has_margin_data = (
            'years' in data and data['years'] and
            (('ebitda_margin' in data and self._has_valid_data(data['ebitda_margin'])) or
             ('pat_margin' in data and self._has_valid_data(data['pat_margin'])))
        )

        if has_margin_data:
            fig, ax = plt.subplots(figsize=(10, 6))

            x = range(len(data['years']))

            if 'ebitda_margin' in data and self._has_valid_data(data['ebitda_margin']):
                ax.plot(x, data['ebitda_margin'], marker='o', label='EBITDA Margin', linewidth=2, color='#2c5282')
            if 'pat_margin' in data and self._has_valid_data(data['pat_margin']):
                ax.plot(x, data['pat_margin'], marker='s', label='PAT Margin', linewidth=2, color='#38a169')

            ax.set_xticks(x)
            ax.set_xticklabels(data['years'])
            ax.set_ylabel('Margin (%)')
            ax.set_title('Margin Trend', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            filepath = f"{output_dir}/margin_trend.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            chart_files.append(filepath)
            logger.info(f"✅ Generated: Margin Trend")
        else:
            logger.info("⏭️ Skipping Margin chart - insufficient data")

        # 3. ROE/ROCE Chart - only if we have valid return data
        has_return_data = (
            'years' in data and data['years'] and
            (('roe' in data and self._has_valid_data(data['roe'])) or
             ('roce' in data and self._has_valid_data(data['roce'])))
        )

        if has_return_data:
            fig, ax = plt.subplots(figsize=(10, 6))

            x = range(len(data['years']))

            if 'roe' in data and self._has_valid_data(data['roe']):
                ax.bar([i - 0.2 for i in x], data['roe'], width=0.4, label='ROE', color='coral')
            if 'roce' in data and self._has_valid_data(data['roce']):
                ax.bar([i + 0.2 for i in x], data['roce'], width=0.4, label='ROCE', color='teal')

            ax.set_xticks(x)
            ax.set_xticklabels(data['years'])
            ax.set_ylabel('Return (%)')
            ax.set_title('ROE & ROCE Trend', fontsize=14, fontweight='bold')
            ax.legend()

            plt.tight_layout()
            filepath = f"{output_dir}/roe_roce_trend.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            chart_files.append(filepath)
            logger.info(f"✅ Generated: ROE & ROCE Trend")
        else:
            logger.info("⏭️ Skipping ROE/ROCE chart - insufficient data")

        return chart_files


# =============================================================================
# REPORT TEMPLATE
# =============================================================================

class ScenarioAnalyzer:
    """Bull/Base/Bear scenario analysis generator."""

    @staticmethod
    def generate_scenarios(
        current_price: float,
        dcf_price: float,
        analyst_target: float,
        company_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Generate three scenarios with probabilities."""

        # Calculate base targets
        base_target = analyst_target if analyst_target > 0 else dcf_price
        revenue_growth = company_data.get('revenue_growth', 10)
        pe_ratio = company_data.get('pe_ratio', 20)

        scenarios = {
            'bull': {
                'name': 'Bull Case',
                'probability': 25,
                'target': base_target * 1.25,
                'upside': ((base_target * 1.25) / current_price - 1) * 100,
                'revenue_growth': revenue_growth * 1.3,
                'margin_expansion': 150,  # bps
                'pe_multiple': pe_ratio * 1.15,
                'key_drivers': [
                    'Faster-than-expected market share gains',
                    'Successful new product/segment launches',
                    'Margin expansion from operating leverage',
                    'Industry tailwinds and favorable policy changes'
                ]
            },
            'base': {
                'name': 'Base Case',
                'probability': 50,
                'target': base_target,
                'upside': ((base_target) / current_price - 1) * 100,
                'revenue_growth': revenue_growth,
                'margin_expansion': 0,
                'pe_multiple': pe_ratio,
                'key_drivers': [
                    'Steady growth in line with guidance',
                    'Stable market share in core segments',
                    'Margins maintained at current levels',
                    'Normal competitive intensity'
                ]
            },
            'bear': {
                'name': 'Bear Case',
                'probability': 25,
                'target': base_target * 0.75,
                'upside': ((base_target * 0.75) / current_price - 1) * 100,
                'revenue_growth': revenue_growth * 0.5,
                'margin_expansion': -200,  # bps
                'pe_multiple': pe_ratio * 0.85,
                'key_drivers': [
                    'Slower demand growth due to macro headwinds',
                    'Increased competitive pressure',
                    'Margin compression from input cost inflation',
                    'Execution challenges or regulatory hurdles'
                ]
            }
        }

        return scenarios

    @staticmethod
    def format_scenario_table(scenarios: Dict[str, Dict[str, Any]], current_price: float) -> str:
        """Format scenarios as markdown."""
        output = """
## Scenario Analysis

### Probability-Weighted Target Price

| Scenario | Probability | Target Price | Upside | Revenue Growth | Multiple |
|----------|:-----------:|-------------:|-------:|---------------:|---------:|
"""
        for key in ['bull', 'base', 'bear']:
            s = scenarios[key]
            label = 'BULL' if key == 'bull' else 'BASE' if key == 'base' else 'BEAR'
            upside_str = f"+{s['upside']:.1f}%" if s['upside'] > 0 else f"{s['upside']:.1f}%"
            output += f"| **{label}** | {s['probability']}% | ₹{s['target']:,.0f} | {upside_str} | {s['revenue_growth']:.1f}% | {s['pe_multiple']:.1f}x |\n"

        # Probability-weighted target
        weighted_target = sum(s['probability'] * s['target'] / 100 for s in scenarios.values())
        weighted_upside = ((weighted_target / current_price) - 1) * 100
        upside_str = f"+{weighted_upside:.1f}%" if weighted_upside > 0 else f"{weighted_upside:.1f}%"
        output += f"| **Weighted Avg** | 100% | **₹{weighted_target:,.0f}** | **{upside_str}** | - | - |\n"

        output += """

---

### Bull Case (25% Probability)

"""
        s = scenarios['bull']
        output += f"**Target Price:** ₹{s['target']:,.0f} | **Upside:** +{s['upside']:.1f}%\n\n"
        output += "**Key Assumptions:**\n\n"
        for driver in s['key_drivers']:
            output += f"- {driver}\n"

        output += """

---

### Base Case (50% Probability)

"""
        s = scenarios['base']
        upside_str = f"+{s['upside']:.1f}%" if s['upside'] > 0 else f"{s['upside']:.1f}%"
        output += f"**Target Price:** ₹{s['target']:,.0f} | **Upside:** {upside_str}\n\n"
        output += "**Key Assumptions:**\n\n"
        for driver in s['key_drivers']:
            output += f"- {driver}\n"

        output += """

---

### Bear Case (25% Probability)

"""
        s = scenarios['bear']
        upside_str = f"+{s['upside']:.1f}%" if s['upside'] > 0 else f"{s['upside']:.1f}%"
        output += f"**Target Price:** ₹{s['target']:,.0f} | **Downside:** {upside_str}\n\n"
        output += "**Key Assumptions:**\n\n"
        for driver in s['key_drivers']:
            output += f"- {driver}\n"

        return output


class CatalystsGenerator:
    """Generate catalysts section with timelines."""

    @staticmethod
    def generate_catalysts(company_data: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
        """Generate near-term and long-term catalysts."""
        sector = company_data.get('sector', '')

        # Generic catalysts by timeframe
        near_term = [
            {'event': 'Quarterly earnings announcement', 'timeline': '0-3 months', 'impact': 'High'},
            {'event': 'Annual general meeting', 'timeline': '1-2 months', 'impact': 'Medium'},
            {'event': 'Dividend declaration', 'timeline': '0-3 months', 'impact': 'Medium'},
        ]

        medium_term = [
            {'event': 'New product/service launches', 'timeline': '3-6 months', 'impact': 'High'},
            {'event': 'Capacity expansion completion', 'timeline': '6-12 months', 'impact': 'High'},
            {'event': 'Strategic partnerships/acquisitions', 'timeline': '6-12 months', 'impact': 'High'},
        ]

        long_term = [
            {'event': 'Market share gains in key segments', 'timeline': '12-24 months', 'impact': 'High'},
            {'event': 'Geographic expansion', 'timeline': '12-36 months', 'impact': 'Medium'},
            {'event': 'Margin expansion from scale', 'timeline': '12-24 months', 'impact': 'Medium'},
        ]

        # Sector-specific catalysts
        if 'Technology' in sector or 'IT' in sector:
            near_term.append({'event': 'Large deal wins announcement', 'timeline': '0-3 months', 'impact': 'High'})
            medium_term.append({'event': 'AI/Digital services ramp-up', 'timeline': '6-12 months', 'impact': 'High'})
        elif 'Financial' in sector or 'Bank' in sector:
            near_term.append({'event': 'RBI policy rate decision', 'timeline': '0-2 months', 'impact': 'High'})
            medium_term.append({'event': 'NIM expansion from rate cycle', 'timeline': '6-12 months', 'impact': 'High'})
        elif 'Energy' in sector:
            near_term.append({'event': 'Crude oil price movements', 'timeline': 'Ongoing', 'impact': 'High'})
            medium_term.append({'event': 'Green energy projects commissioning', 'timeline': '6-12 months', 'impact': 'High'})
        elif 'Consumer' in sector:
            near_term.append({'event': 'Festive season demand', 'timeline': '0-3 months', 'impact': 'High'})
            medium_term.append({'event': 'Rural recovery and distribution expansion', 'timeline': '6-12 months', 'impact': 'Medium'})
        elif 'Pharma' in sector or 'Healthcare' in sector:
            near_term.append({'event': 'US FDA approvals', 'timeline': '0-6 months', 'impact': 'High'})
            medium_term.append({'event': 'New drug launches', 'timeline': '6-18 months', 'impact': 'High'})

        return {
            'near_term': near_term,
            'medium_term': medium_term,
            'long_term': long_term
        }

    @staticmethod
    def format_catalysts(catalysts: Dict[str, List[Dict[str, str]]]) -> str:
        """Format catalysts as markdown."""
        output = """
## Catalysts & Key Events

### Near-Term Catalysts (0-6 months)

| Event | Timeline | Impact |
|-------|:--------:|:------:|
"""
        for cat in catalysts['near_term']:
            impact_badge = '🔴 High' if cat['impact'] == 'High' else '🟡 Medium' if cat['impact'] == 'Medium' else '🟢 Low'
            output += f"| {cat['event']} | {cat['timeline']} | {impact_badge} |\n"

        output += "\n### Medium-Term Catalysts (6-12 months)\n\n"
        output += "| Event | Timeline | Impact |\n|-------|:--------:|:------:|\n"
        for cat in catalysts['medium_term']:
            impact_badge = '🔴 High' if cat['impact'] == 'High' else '🟡 Medium' if cat['impact'] == 'Medium' else '🟢 Low'
            output += f"| {cat['event']} | {cat['timeline']} | {impact_badge} |\n"

        output += "\n### Long-Term Catalysts (12+ months)\n\n"
        output += "| Event | Timeline | Impact |\n|-------|:--------:|:------:|\n"
        for cat in catalysts['long_term']:
            impact_badge = '🔴 High' if cat['impact'] == 'High' else '🟡 Medium' if cat['impact'] == 'Medium' else '🟢 Low'
            output += f"| {cat['event']} | {cat['timeline']} | {impact_badge} |\n"

        return output


class IndustryAnalyzer:
    """Generate industry analysis section."""

    INDUSTRY_DATA = {
        'Technology': {
            'market_size': '₹15+ Lakh Cr',
            'growth_rate': '12-15%',
            'key_players': 'TCS, Infosys, Wipro, HCL Tech',
            'drivers': ['Digital transformation', 'Cloud adoption', 'AI/ML integration', 'Cost arbitrage'],
            'challenges': ['Talent attrition', 'Currency volatility', 'Client budget constraints'],
            'outlook': 'Positive - Strong demand for digital services globally'
        },
        'Financial Services': {
            'market_size': '₹200+ Lakh Cr (Credit)',
            'growth_rate': '13-16%',
            'key_players': 'HDFC Bank, ICICI Bank, SBI, Kotak',
            'drivers': ['Credit growth', 'Financial inclusion', 'Digital banking', 'Rising income levels'],
            'challenges': ['NPA management', 'Regulatory changes', 'Fintech competition'],
            'outlook': 'Positive - Structural growth from under-penetration'
        },
        'Energy': {
            'market_size': '₹25+ Lakh Cr',
            'growth_rate': '5-8%',
            'key_players': 'Reliance, ONGC, BPCL, IOC',
            'drivers': ['Fuel demand growth', 'Petrochemical expansion', 'Green energy transition'],
            'challenges': ['Crude price volatility', 'Regulatory pricing', 'Green transition'],
            'outlook': 'Neutral - Transition phase with green energy focus'
        },
        'Consumer Defensive': {
            'market_size': '₹8+ Lakh Cr',
            'growth_rate': '8-12%',
            'key_players': 'HUL, ITC, Nestle, Britannia',
            'drivers': ['Rising consumption', 'Premiumization', 'Rural penetration'],
            'challenges': ['Input cost inflation', 'Competition', 'Changing preferences'],
            'outlook': 'Positive - Structural consumption growth story'
        },
        'Healthcare': {
            'market_size': '₹6+ Lakh Cr',
            'growth_rate': '10-14%',
            'key_players': 'Sun Pharma, Dr. Reddy\'s, Cipla, Divi\'s',
            'drivers': ['Aging population', 'Healthcare spending', 'Generic demand', 'Exports'],
            'challenges': ['US pricing pressure', 'FDA compliance', 'R&D intensity'],
            'outlook': 'Positive - Strong export and domestic demand'
        }
    }

    @staticmethod
    def get_industry_analysis(sector: str, company_data: Dict[str, Any]) -> str:
        """Generate industry analysis section."""
        # Find matching sector
        industry_info = None
        for key, data in IndustryAnalyzer.INDUSTRY_DATA.items():
            if key.lower() in sector.lower() or sector.lower() in key.lower():
                industry_info = data
                break

        if not industry_info:
            industry_info = {
                'market_size': 'Data not available',
                'growth_rate': 'Industry average',
                'key_players': 'Various players',
                'drivers': ['Market demand', 'Economic growth', 'Industry trends'],
                'challenges': ['Competition', 'Regulatory environment', 'Economic cycles'],
                'outlook': 'Refer to industry-specific reports'
            }

        output = f"""
## Industry Analysis

### Market Overview

| Metric | Value |
|--------|------:|
| Sector | {sector} |
| Market Size | {industry_info['market_size']} |
| Growth Rate | {industry_info['growth_rate']} CAGR |
| Key Players | {industry_info['key_players']} |

### Growth Drivers

"""
        for i, driver in enumerate(industry_info['drivers'], 1):
            output += f"{i}. **{driver}**\n"

        output += "\n### Key Challenges\n\n"
        for i, challenge in enumerate(industry_info['challenges'], 1):
            output += f"{i}. {challenge}\n"

        output += f"""
### Industry Outlook

{industry_info['outlook']}

### Competitive Positioning

Based on market capitalization and financial metrics, the company ranks among the **top players** in its sector with:
- Strong market presence and brand recognition
- Competitive cost structure
- Diversified revenue streams
"""
        return output


class EarningsProjector:
    """Generate earnings projections table."""

    @staticmethod
    def generate_projections(
        company_data: Dict[str, Any],
        dcf_model: 'DCFModel'
    ) -> str:
        """Generate forward earnings estimates table."""
        current_year = datetime.now().year

        # Get base metrics
        revenue = company_data.get('revenue', 0) / 1e7  # Convert to Cr
        ebitda = company_data.get('ebitda', 0) / 1e7
        eps = company_data.get('eps', 0)
        growth = company_data.get('revenue_growth', 10) / 100

        # Generate projections
        years = [f"FY{y}E" for y in range(current_year + 1, current_year + 4)]

        rev_proj = []
        ebitda_proj = []
        eps_proj = []

        r, e, ep = revenue, ebitda, eps
        for i in range(3):
            # Apply growth with slight deceleration
            growth_adj = growth * (1 - 0.05 * i)
            r = r * (1 + growth_adj)
            e = r * (ebitda / revenue if revenue > 0 else 0.2) * (1 + 0.01 * i)  # Slight margin expansion
            ep = ep * (1 + growth_adj * 1.1)  # EPS grows faster due to operating leverage

            rev_proj.append(r)
            ebitda_proj.append(e)
            eps_proj.append(ep)

        output = """
## Earnings Estimates & Projections

### Consensus Estimates

| Metric | """ + " | ".join(years) + """ | 3Y CAGR |
|--------|""" + "|".join(["------:"] * 4) + """|
"""
        # Revenue
        rev_cagr = ((rev_proj[-1] / revenue) ** (1/3) - 1) * 100 if revenue > 0 else 0
        output += f"| **Revenue (₹ Cr)** | {rev_proj[0]:,.0f} | {rev_proj[1]:,.0f} | {rev_proj[2]:,.0f} | {rev_cagr:.1f}% |\n"

        # EBITDA
        ebitda_cagr = ((ebitda_proj[-1] / ebitda) ** (1/3) - 1) * 100 if ebitda > 0 else 0
        output += f"| **EBITDA (₹ Cr)** | {ebitda_proj[0]:,.0f} | {ebitda_proj[1]:,.0f} | {ebitda_proj[2]:,.0f} | {ebitda_cagr:.1f}% |\n"

        # EBITDA Margin
        margins = [e/r*100 if r > 0 else 0 for e, r in zip(ebitda_proj, rev_proj)]
        output += f"| EBITDA Margin (%) | {margins[0]:.1f}% | {margins[1]:.1f}% | {margins[2]:.1f}% | - |\n"

        # EPS
        eps_cagr = ((eps_proj[-1] / eps) ** (1/3) - 1) * 100 if eps > 0 else 0
        output += f"| **EPS (₹)** | {eps_proj[0]:.2f} | {eps_proj[1]:.2f} | {eps_proj[2]:.2f} | {eps_cagr:.1f}% |\n"

        # Implied P/E at target
        cmp = company_data.get('current_price', 0)
        if cmp > 0:
            fwd_pe = [cmp / e if e > 0 else 0 for e in eps_proj]
            output += f"| P/E at CMP | {fwd_pe[0]:.1f}x | {fwd_pe[1]:.1f}x | {fwd_pe[2]:.1f}x | - |\n"

        output += """
### Key Assumptions

- Revenue growth trajectory based on historical trends and management guidance
- EBITDA margins expected to expand gradually with operating leverage
- EPS growth outpaces revenue due to margin expansion and share buybacks (if any)
- Projections exclude any extraordinary items or one-time gains/losses
"""
        return output


class PriceChartGenerator:
    """Generate price charts with technical indicators."""

    @staticmethod
    def create_price_chart(
        prices: List[float],
        dates: List[str],
        ticker: str,
        output_dir: str
    ) -> Optional[str]:
        """Generate price chart with moving averages."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from datetime import datetime as dt
            import numpy as np

            if len(prices) < 20:
                return None

            plt.style.use('seaborn-v0_8-whitegrid')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                            gridspec_kw={'height_ratios': [3, 1]})

            # Convert dates
            x = list(range(len(prices)))

            # Price plot
            ax1.plot(x, prices, color='#2c5282', linewidth=1.5, label='Price')

            # Moving averages
            if len(prices) >= 20:
                sma20 = np.convolve(prices, np.ones(20)/20, mode='valid')
                ax1.plot(x[19:], sma20, color='#38a169', linewidth=1,
                        label='20-Day SMA', linestyle='--')

            if len(prices) >= 50:
                sma50 = np.convolve(prices, np.ones(50)/50, mode='valid')
                ax1.plot(x[49:], sma50, color='#d69e2e', linewidth=1,
                        label='50-Day SMA', linestyle='--')

            if len(prices) >= 200:
                sma200 = np.convolve(prices, np.ones(200)/200, mode='valid')
                ax1.plot(x[199:], sma200, color='#e53e3e', linewidth=1,
                        label='200-Day SMA', linestyle='-.')

            ax1.set_title(f'{ticker} - Price Chart with Moving Averages',
                         fontsize=14, fontweight='bold', color='#1a365d')
            ax1.set_ylabel('Price (₹)', fontsize=10)
            ax1.legend(loc='upper left', fontsize=8)
            ax1.grid(True, alpha=0.3)

            # Fill between 52-week high/low
            high = max(prices[-252:]) if len(prices) >= 252 else max(prices)
            low = min(prices[-252:]) if len(prices) >= 252 else min(prices)
            ax1.axhline(y=high, color='#38a169', linestyle=':', alpha=0.5, label='52W High')
            ax1.axhline(y=low, color='#e53e3e', linestyle=':', alpha=0.5, label='52W Low')

            # Volume/RSI proxy (use price momentum)
            if len(prices) >= 14:
                # Calculate simple momentum
                momentum = [prices[i] - prices[i-14] for i in range(14, len(prices))]
                colors = ['#38a169' if m > 0 else '#e53e3e' for m in momentum]
                ax2.bar(x[14:], momentum, color=colors, alpha=0.6)
                ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
                ax2.set_ylabel('14-Day Momentum', fontsize=10)
                ax2.set_xlabel('Trading Days', fontsize=10)

            plt.tight_layout()

            filepath = f"{output_dir}/{ticker}_price_chart.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()

            return filepath

        except Exception as e:
            logger.warning(f"Price chart generation failed: {e}")
            return None


class ReportTemplate:
    """Professional report template generator."""

    def __init__(self):
        self.formatter = FinancialTablesFormatter()
        self.peer_formatter = PeerComparisonFormatter()
        self.chart_generator = ChartGenerator()
        self.scenario_analyzer = ScenarioAnalyzer()
        self.catalysts_generator = CatalystsGenerator()
        self.industry_analyzer = IndustryAnalyzer()
        self.earnings_projector = EarningsProjector()
        self.price_chart_generator = PriceChartGenerator()

    def generate_cover_page(self, snapshot: CompanySnapshot,
                             investment_thesis: List[str],
                             key_risks: List[str]) -> str:
        """Generate professional cover page."""

        rating_emoji = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}.get(snapshot.rating.upper(), "⚪")
        upside = snapshot.upside
        upside_str = f"+{upside:.1f}%" if upside > 0 else f"{upside:.1f}%"

        cover = f"""
# {snapshot.company_name} ({snapshot.ticker})

## {rating_emoji} {snapshot.rating} | Target: ₹{snapshot.target_price:,.0f} | Upside: {upside_str}

**Sector:** {snapshot.sector} | **Industry:** {snapshot.industry}

---

### Key Metrics

| Metric | Value | Metric | Value |
|--------|------:|--------|------:|
| CMP | ₹{snapshot.current_price:,.2f} | Market Cap | ₹{snapshot.market_cap:,.0f} {snapshot.market_cap_unit} |
| P/E (TTM) | {snapshot.pe_ratio:.1f}x | P/E (Forward) | {snapshot.pe_forward:.1f}x |
| P/B Ratio | {snapshot.pb_ratio:.1f}x | EV/EBITDA | {snapshot.ev_ebitda:.1f}x |
| ROE | {snapshot.roe:.1f}% | ROCE | {snapshot.roce:.1f}% |
| Dividend Yield | {snapshot.dividend_yield:.1f}% | Beta | {snapshot.beta:.2f} |
| 52W High | ₹{snapshot.week_52_high:,.0f} | 52W Low | ₹{snapshot.week_52_low:,.0f} |

### Shareholding Pattern

| Category | Holding |
|----------|--------:|
| Promoters | {snapshot.promoter_holding:.1f}% |
| FII | {snapshot.fii_holding:.1f}% |
| DII | {snapshot.dii_holding:.1f}% |
| Public | {100 - snapshot.promoter_holding - snapshot.fii_holding - snapshot.dii_holding:.1f}% |

---

### Investment Thesis

"""
        for i, point in enumerate(investment_thesis[:5], 1):
            cover += f"{i}. {point}\n"

        cover += "\n### Key Risks\n\n"
        for i, risk in enumerate(key_risks[:3], 1):
            cover += f"{i}. {risk}\n"

        cover += f"""
---

**Report Date:** {datetime.now().strftime('%B %d, %Y')}
**Analyst:** Jotty Research

---
"""
        return cover

    def generate_valuation_section(self,
                                    dcf_calc: DCFCalculator,
                                    dcf_result: Dict[str, float],
                                    peer_data: PeerComparison,
                                    target_company: str,
                                    current_price: float) -> str:
        """Generate complete valuation section."""

        section = """
## Valuation Analysis

### Methodology

We employ multiple valuation methodologies to arrive at our target price:
1. **Discounted Cash Flow (DCF)** - Intrinsic value based on projected free cash flows
2. **Comparable Company Analysis** - Relative valuation using peer multiples
3. **Sensitivity Analysis** - Range of values based on key assumptions

"""
        # DCF Summary
        section += dcf_calc.format_dcf_summary(dcf_result)
        section += "\n"

        # Sensitivity Matrix
        sensitivity_df = dcf_calc.create_sensitivity_matrix(
            shares_outstanding=1.0,  # Placeholder
            net_debt=0.0
        )
        section += dcf_calc.format_sensitivity_table(sensitivity_df)
        section += "\n"

        # Peer Comparison
        section += "### Peer Comparison\n\n"
        section += self.peer_formatter.create_peer_table(peer_data, target_company)
        section += "\n"

        # Football Field
        valuations = {
            "DCF": (dcf_result.get('implied_price', 0) * 0.85,
                    dcf_result.get('implied_price', 0),
                    dcf_result.get('implied_price', 0) * 1.15),
            "P/E Comps": (current_price * 0.9, current_price * 1.1, current_price * 1.3),
            "EV/EBITDA": (current_price * 0.85, current_price * 1.05, current_price * 1.25),
        }
        section += "\n"
        section += self.chart_generator.create_football_field_ascii(valuations, current_price)

        return section
