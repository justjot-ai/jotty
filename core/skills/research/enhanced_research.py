"""
Enhanced Stock Research Tool
============================

World-class broker-grade research report generation.
Integrates:
- Live financial data from Screener.in and Yahoo Finance
- Professional formatting with tables and charts
- DCF valuation with sensitivity analysis
- Peer comparison analysis
- Technical analysis
- Professional PDF output
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


async def enhanced_stock_research_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate world-class broker-grade research report.

    Args:
        params: Dictionary containing:
            - ticker (str, required): Stock ticker symbol
            - company_name (str, optional): Full company name
            - exchange (str, optional): Exchange (NSE, BSE), default: NSE
            - peers (list, optional): List of peer company tickers
            - target_price (float, optional): Analyst target price
            - rating (str, optional): BUY/HOLD/SELL
            - output_dir (str, optional): Output directory
            - report_type (str, optional): 'full', 'quarterly', 'flash'
            - include_charts (bool, optional): Generate matplotlib charts
            - send_telegram (bool, optional): Send to Telegram

    Returns:
        Dictionary with report paths and data
    """
    try:
        # Import components
        from .report_components import (
            CompanySnapshot, FinancialStatements, DCFModel, PeerComparison,
            FinancialTablesFormatter, DCFCalculator, PeerComparisonFormatter,
            ChartGenerator, ReportTemplate, ScenarioAnalyzer, CatalystsGenerator,
            IndustryAnalyzer, EarningsProjector, PriceChartGenerator
        )
        from .data_fetcher import ResearchDataFetcher, FinancialDataConverter

        # Parse parameters
        ticker = params.get('ticker', '').upper().strip()
        if not ticker:
            return {'success': False, 'error': 'ticker parameter is required'}

        company_name = params.get('company_name', ticker)

        # Auto-detect exchange for US stocks
        US_TICKERS = {
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA',
            'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'HD', 'DIS', 'PYPL',
            'BAC', 'ADBE', 'NFLX', 'CRM', 'INTC', 'AMD', 'CSCO', 'PEP', 'KO'
        }
        default_exchange = 'US' if ticker in US_TICKERS else 'NSE'
        exchange = params.get('exchange', default_exchange)
        peers = params.get('peers', [])
        target_price = params.get('target_price')
        rating = params.get('rating')
        output_dir = params.get('output_dir', os.path.expanduser('~/jotty/reports'))
        report_type = params.get('report_type', 'full')
        include_charts = params.get('include_charts', True)
        send_telegram = params.get('send_telegram', True)

        logger.info(f"📊 Starting enhanced research for {ticker}")

        # Initialize components
        fetcher = ResearchDataFetcher()
        formatter = FinancialTablesFormatter()
        peer_formatter = PeerComparisonFormatter()
        chart_gen = ChartGenerator()
        template = ReportTemplate()

        # ================================================================
        # PHASE 1: DATA COLLECTION
        # ================================================================
        logger.info("📥 Phase 1: Fetching live financial data...")

        # Fetch main company data
        company_data = await fetcher.fetch_company_data(ticker, exchange)

        if 'error' in company_data and not company_data.get('current_price'):
            return {
                'success': False,
                'error': f"Failed to fetch data for {ticker}: {company_data.get('error')}"
            }

        # Auto-detect peers if not provided
        if not peers:
            sector = company_data.get('sector', '')
            peers = _get_sector_peers(ticker, sector, exchange)

        # Fetch peer data
        peer_data = {}
        if peers:
            logger.info(f"📊 Fetching peer data for: {', '.join(peers[:5])}")
            peer_data = await fetcher.fetch_peer_data(ticker, peers[:5], exchange)

        # Get analyst ratings
        analyst_data = await fetcher.get_analyst_ratings(ticker, exchange)

        # ================================================================
        # PHASE 2: DATA PROCESSING
        # ================================================================
        logger.info("⚙️ Phase 2: Processing and structuring data...")

        # Convert to data classes
        snapshot = FinancialDataConverter.to_company_snapshot(
            company_data,
            target_price=target_price or analyst_data.get('target_mean'),
            rating=rating
        )

        financials = FinancialDataConverter.to_financial_statements(company_data)
        peer_comparison = FinancialDataConverter.to_peer_comparison(peer_data)

        # Build DCF model
        dcf_model = _build_dcf_model(company_data, financials)
        dcf_calc = DCFCalculator(dcf_model)

        # Calculate DCF
        shares = company_data.get('shares_outstanding', 1e8) / 1e7  # Convert to Cr
        net_debt = (company_data.get('total_debt', 0) - company_data.get('total_cash', 0)) / 1e7
        dcf_result = dcf_calc.calculate_dcf(shares, net_debt)

        # ================================================================
        # PHASE 3: REPORT GENERATION
        # ================================================================
        logger.info("📝 Phase 3: Generating report...")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate investment thesis using available data
        investment_thesis = _generate_investment_thesis(company_data, snapshot, analyst_data)
        key_risks = _generate_key_risks(company_data, snapshot)

        # Build report sections
        report_sections = []

        # 1. Cover Page (Enhanced)
        cover = _generate_enhanced_cover_page(snapshot, investment_thesis, key_risks, analyst_data)
        report_sections.append(cover)

        # 2. Table of Contents (Enhanced)
        toc = _generate_enhanced_toc(report_type)
        report_sections.append(toc)

        # 3. Executive Summary
        exec_summary = _generate_executive_summary(snapshot, company_data, analyst_data, dcf_result)
        report_sections.append(exec_summary)

        # 4. Company Overview
        company_overview = _generate_company_overview(company_data)
        report_sections.append(company_overview)

        # 5. Industry Analysis (NEW)
        industry_section = IndustryAnalyzer.get_industry_analysis(
            company_data.get('sector', ''), company_data
        )
        report_sections.append(industry_section)

        # 6. Financial Analysis
        financial_section = _generate_financial_section(financials, formatter, company_data)
        report_sections.append(financial_section)

        # 7. Earnings Projections (NEW)
        earnings_section = EarningsProjector.generate_projections(company_data, dcf_model)
        report_sections.append(earnings_section)

        # 8. Valuation Section
        valuation_section = template.generate_valuation_section(
            dcf_calc, dcf_result, peer_comparison, ticker, snapshot.current_price
        )
        report_sections.append(valuation_section)

        # 9. Scenario Analysis (NEW)
        scenarios = ScenarioAnalyzer.generate_scenarios(
            snapshot.current_price,
            dcf_result.get('implied_price', 0),
            analyst_data.get('target_mean', 0),
            company_data
        )
        scenario_section = ScenarioAnalyzer.format_scenario_table(scenarios, snapshot.current_price)
        report_sections.append(scenario_section)

        # 10. Catalysts Section (NEW)
        catalysts = CatalystsGenerator.generate_catalysts(company_data)
        catalysts_section = CatalystsGenerator.format_catalysts(catalysts)
        report_sections.append(catalysts_section)

        # 11. Technical Analysis
        if company_data.get('price_history'):
            tech_section = _generate_technical_section(company_data)
            report_sections.append(tech_section)

        # 12. Shareholding Pattern
        shareholding = _generate_shareholding_section(company_data, snapshot)
        report_sections.append(shareholding)

        # 13. Risk Analysis
        risk_section = _generate_risk_section(key_risks, company_data)
        report_sections.append(risk_section)

        # 14. Recommendation
        recommendation = _generate_recommendation(snapshot, dcf_result, analyst_data)
        report_sections.append(recommendation)

        # 15. Disclaimer
        disclaimer = _generate_disclaimer()
        report_sections.append(disclaimer)

        # Combine all sections
        full_report = "\n\n---\n\n".join(report_sections)

        # ================================================================
        # PHASE 4: OUTPUT GENERATION
        # ================================================================
        logger.info("📄 Phase 4: Generating output files...")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_ticker = ticker.replace('/', '-').replace('\\', '-')

        # Save markdown
        md_filename = f"{safe_ticker}_research_{timestamp}.md"
        md_path = output_path / md_filename

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(full_report)

        logger.info(f"✅ Markdown saved: {md_path}")

        # Generate charts if requested
        chart_files = []
        if include_charts:
            chart_data = {
                'years': financials.years,
                'revenue': financials.revenue,
                'pat': financials.pat,
                'ebitda_margin': financials.ebitda_margin,
                'pat_margin': financials.pat_margin,
                'roe': financials.roe,
                'roce': financials.roce,
            }
            chart_files = chart_gen.create_matplotlib_charts(chart_data, str(output_path))

            # Generate price chart with moving averages
            prices = company_data.get('price_history', [])
            if prices and len(prices) >= 20:
                price_chart = PriceChartGenerator.create_price_chart(
                    prices, [], ticker, str(output_path)
                )
                if price_chart:
                    chart_files.append(price_chart)
                    logger.info(f"📈 Generated price chart: {price_chart}")

            if chart_files:
                logger.info(f"📊 Generated {len(chart_files)} charts")

        # Convert to PDF with embedded charts
        pdf_path = None
        try:
            pdf_path = await _convert_to_pdf(
                md_path, output_path, safe_ticker, timestamp,
                chart_files=chart_files,
                template_name=params.get('template', None)
            )
            if pdf_path:
                logger.info(f"✅ PDF saved: {pdf_path}")
        except Exception as e:
            logger.warning(f"⚠️ PDF conversion failed: {e}")

        # Send to Telegram if requested
        telegram_sent = False
        if send_telegram and pdf_path:
            telegram_sent = await _send_to_telegram(pdf_path, ticker, snapshot)

        return {
            'success': True,
            'ticker': ticker,
            'company_name': snapshot.company_name,
            'rating': snapshot.rating,
            'target_price': snapshot.target_price,
            'current_price': snapshot.current_price,
            'upside': snapshot.upside,
            'md_path': str(md_path),
            'pdf_path': str(pdf_path) if pdf_path else None,
            'chart_files': chart_files,
            'telegram_sent': telegram_sent,
            'data_sources': company_data.get('sources', []),
            'dcf_implied_price': dcf_result.get('implied_price', 0),
        }

    except Exception as e:
        logger.error(f"Enhanced research error: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_sector_peers(ticker: str, sector: str, exchange: str) -> List[str]:
    """Get peer companies based on sector and exchange."""

    # US stock peers by sector
    us_sector_peers = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC', 'CRM', 'ORCL'],
        'Consumer Electronics': ['AAPL', 'SONY', 'DELL', 'HPQ', 'LOGI'],
        'Semiconductors': ['NVDA', 'AMD', 'INTC', 'QCOM', 'AVGO', 'TXN', 'MU', 'AMAT'],
        'Financial Services': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC'],
        'Consumer Defensive': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'GIS', 'K'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'OXY'],
        'Healthcare': ['JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'LLY', 'TMO', 'ABT'],
        'Industrials': ['CAT', 'DE', 'BA', 'HON', 'UPS', 'GE', 'MMM', 'LMT'],
        'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'NKE', 'MCD', 'SBUX', 'TGT'],
    }

    # Indian stock peers by sector
    india_sector_peers = {
        'Technology': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTIM'],
        'Financial Services': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK'],
        'Consumer Defensive': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR'],
        'Energy': ['RELIANCE', 'ONGC', 'BPCL', 'IOC', 'GAIL'],
        'Healthcare': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'BIOCON'],
        'Industrials': ['LT', 'SIEMENS', 'ABB', 'BHEL', 'HAVELLS'],
        'Basic Materials': ['TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'VEDL', 'COALINDIA'],
        'Consumer Cyclical': ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'EICHERMOT'],
    }

    # Choose peer set based on exchange
    is_us = exchange.upper() in ('US', 'NYSE', 'NASDAQ', 'AMEX')
    sector_peers = us_sector_peers if is_us else india_sector_peers

    peers = sector_peers.get(sector, [])

    # Remove the target company from peers
    peers = [p for p in peers if p.upper() != ticker.upper()]

    return peers[:5]  # Return max 5 peers


def _build_dcf_model(data: Dict[str, Any], financials: 'FinancialStatements') -> 'DCFModel':
    """Build DCF model from company data."""
    from .report_components import DCFModel

    model = DCFModel()

    # Set projection years
    current_year = datetime.now().year
    model.projection_years = [f"FY{y}" for y in range(current_year + 1, current_year + 6)]

    # Get latest financials
    latest_revenue = financials.revenue[-1] if financials.revenue else data.get('revenue', 0) / 1e7
    latest_ebitda = data.get('ebitda', 0) / 1e7

    # Calculate growth rate
    revenue_growth = data.get('revenue_growth', 10)
    if financials.revenue and len(financials.revenue) >= 2:
        if financials.revenue[-2] > 0:
            revenue_growth = ((financials.revenue[-1] / financials.revenue[-2]) - 1) * 100

    model.revenue_growth = min(max(revenue_growth, 5), 25)  # Cap between 5-25%

    # EBITDA margin
    ebitda_margin = data.get('ebitda_margin', 20)
    model.ebitda_margin = ebitda_margin if ebitda_margin > 0 else 20

    # Project revenue and EBITDA
    model.revenue_projections = []
    model.ebitda_projections = []
    model.fcf_projections = []

    rev = latest_revenue
    for _ in range(5):
        rev = rev * (1 + model.revenue_growth / 100)
        ebitda = rev * model.ebitda_margin / 100
        # FCF = EBITDA * (1 - tax) - capex - working capital change
        fcf = ebitda * (1 - model.tax_rate / 100) * 0.7  # Simplified FCF

        model.revenue_projections.append(rev)
        model.ebitda_projections.append(ebitda)
        model.fcf_projections.append(fcf)

    # Set beta
    model.beta = data.get('beta', 1.0) or 1.0

    return model


def _generate_investment_thesis(data: Dict[str, Any], snapshot: 'CompanySnapshot',
                                  analyst: Dict[str, Any]) -> List[str]:
    """Generate investment thesis points."""
    thesis = []

    # Growth
    growth = data.get('revenue_growth', 0)
    if growth > 15:
        thesis.append(f"Strong revenue growth of {growth:.1f}% indicates robust business momentum")
    elif growth > 10:
        thesis.append(f"Healthy revenue growth of {growth:.1f}% with consistent execution")

    # Profitability
    roe = snapshot.roe
    if roe > 15:
        thesis.append(f"High ROE of {roe:.1f}% reflects efficient capital utilization")

    # Valuation
    pe = snapshot.pe_ratio
    if pe > 0 and pe < 20:
        thesis.append(f"Reasonable valuation at {pe:.1f}x P/E with room for re-rating")
    elif pe > 0 and pe < 30:
        thesis.append(f"Fair valuation at {pe:.1f}x P/E justified by growth profile")

    # Market position
    if data.get('sector'):
        thesis.append(f"Strong market position in {data.get('sector')} sector")

    # Analyst sentiment
    if analyst.get('num_analysts', 0) > 5:
        if analyst.get('upside', 0) > 10:
            thesis.append(f"Positive analyst sentiment with {analyst.get('upside'):.1f}% upside to consensus target")

    # Dividend
    if snapshot.dividend_yield > 1:
        thesis.append(f"Attractive dividend yield of {snapshot.dividend_yield:.1f}% provides downside protection")

    return thesis[:5]  # Return top 5 points


def _generate_key_risks(data: Dict[str, Any], snapshot: 'CompanySnapshot') -> List[str]:
    """Generate key risk points."""
    risks = []

    # Valuation risk
    if snapshot.pe_ratio > 40:
        risks.append("Rich valuation leaves limited margin of safety")

    # Debt risk
    de_ratio = data.get('debt_to_equity', 0)
    if de_ratio > 1:
        risks.append(f"Elevated debt-to-equity ratio of {de_ratio:.2f}x increases financial risk")

    # Growth slowdown
    if data.get('revenue_growth', 0) < 5:
        risks.append("Slowing growth may impact earnings trajectory")

    # Sector risks
    sector = data.get('sector', '')
    if 'Technology' in sector:
        risks.append("Technology disruption and talent retention challenges")
    elif 'Financial' in sector:
        risks.append("Interest rate sensitivity and asset quality concerns")
    elif 'Energy' in sector:
        risks.append("Commodity price volatility and regulatory changes")

    # Generic risks
    risks.append("Macroeconomic headwinds including inflation and currency fluctuations")
    risks.append("Competitive intensity may pressure margins")

    return risks[:5]


def _generate_enhanced_cover_page(snapshot: 'CompanySnapshot', thesis: List[str],
                                    risks: List[str], analyst: Dict[str, Any]) -> str:
    """Generate world-class cover page with large rating badge."""
    rating_emoji = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}.get(snapshot.rating.upper(), "⚪")
    upside = snapshot.upside
    upside_str = f"+{upside:.1f}%" if upside > 0 else f"{upside:.1f}%"

    # Calculate position from 52W range
    range_52w = snapshot.week_52_high - snapshot.week_52_low
    position = ((snapshot.current_price - snapshot.week_52_low) / range_52w * 100) if range_52w > 0 else 50

    # Build investment thesis points
    thesis_points = "\n".join([f"✓ {point}" for point in thesis[:5]]) if thesis else "✓ Strong market position"

    # Build risk points
    risk_points = "\n".join([f"⚠ {risk}" for risk in risks[:3]]) if risks else "⚠ Market volatility"

    return f"""
# {snapshot.company_name}

### {snapshot.ticker} | {rating_emoji} **{snapshot.rating}** | Target: ₹{snapshot.target_price:,.0f} | Upside: {upside_str}

**Sector:** {snapshot.sector} | **Industry:** {snapshot.industry}

---

## Investment Snapshot

| **Valuation Metrics** | Value | **Return Metrics** | Value |
|----------------------|------:|-------------------|------:|
| Current Price | ₹{snapshot.current_price:,.2f} | ROE | {snapshot.roe:.1f}% |
| Market Cap | ₹{snapshot.market_cap:,.0f} Cr | ROCE | {snapshot.roce:.1f}% |
| P/E (TTM) | {snapshot.pe_ratio:.1f}x | Dividend Yield | {snapshot.dividend_yield:.1f}% |
| P/B Ratio | {snapshot.pb_ratio:.1f}x | Beta | {snapshot.beta:.2f} |
| EV/EBITDA | {snapshot.ev_ebitda:.1f}x | P/E (Forward) | {snapshot.pe_forward:.1f}x |

### 52-Week Price Range

| Low | Current | High | Position |
|----:|--------:|-----:|---------:|
| ₹{snapshot.week_52_low:,.0f} | ₹{snapshot.current_price:,.2f} | ₹{snapshot.week_52_high:,.0f} | {position:.0f}% |

### Shareholding Pattern

| Promoters | FII/FPI | DII | Public |
|----------:|--------:|----:|-------:|
| {snapshot.promoter_holding:.1f}% | {snapshot.fii_holding:.1f}% | {snapshot.dii_holding:.1f}% | {100 - snapshot.promoter_holding - snapshot.fii_holding - snapshot.dii_holding:.1f}% |

---

## Investment Thesis

{thesis_points}

## Key Risks

{risk_points}

---

**Analyst Coverage:** {analyst.get('num_analysts', 0)} analysts | **Consensus Target:** ₹{analyst.get('target_mean', 0):,.0f}

**Report Date:** {datetime.now().strftime('%B %d, %Y')} | **Analyst:** Jotty Research

---
"""


def _generate_enhanced_toc(report_type: str) -> str:
    """Generate enhanced table of contents."""
    return """
## Table of Contents

| Section | Page |
|---------|-----:|
| 1. Executive Summary | 2 |
| 2. Company Overview | 3 |
| 3. Industry Analysis | 4 |
| 4. Financial Analysis | 5 |
| 5. Earnings Projections | 6 |
| 6. Valuation Analysis | 7 |
| 7. Scenario Analysis | 9 |
| 8. Catalysts & Events | 10 |
| 9. Technical Analysis | 11 |
| 10. Shareholding Pattern | 12 |
| 11. Risk Analysis | 13 |
| 12. Investment Recommendation | 14 |

---
"""


def _generate_toc(report_type: str) -> str:
    """Generate table of contents (legacy)."""
    return _generate_enhanced_toc(report_type)


def _generate_executive_summary(snapshot: 'CompanySnapshot', data: Dict[str, Any],
                                  analyst: Dict[str, Any], dcf: Dict[str, float]) -> str:
    """Generate executive summary section."""
    upside_str = f"+{snapshot.upside:.1f}%" if snapshot.upside > 0 else f"{snapshot.upside:.1f}%"

    return f"""
## Executive Summary

We initiate coverage on **{snapshot.company_name} ({snapshot.ticker})** with a **{snapshot.rating}** rating
and a 12-month target price of **₹{snapshot.target_price:,.0f}**, implying an upside of **{upside_str}**
from the current market price of ₹{snapshot.current_price:,.2f}.

### Investment Highlights

| Metric | Value | Assessment |
|--------|------:|------------|
| Market Cap | ₹{snapshot.market_cap:,.0f} Cr | {'Large Cap' if snapshot.market_cap > 50000 else 'Mid Cap' if snapshot.market_cap > 10000 else 'Small Cap'} |
| P/E Ratio | {snapshot.pe_ratio:.1f}x | {'Premium' if snapshot.pe_ratio > 30 else 'Fair' if snapshot.pe_ratio > 15 else 'Attractive'} |
| ROE | {snapshot.roe:.1f}% | {'Excellent' if snapshot.roe > 20 else 'Good' if snapshot.roe > 15 else 'Average'} |
| Dividend Yield | {snapshot.dividend_yield:.1f}% | {'High' if snapshot.dividend_yield > 2 else 'Moderate' if snapshot.dividend_yield > 1 else 'Low'} |

### Valuation Summary

- **DCF Implied Price:** ₹{dcf.get('implied_price', 0):,.0f}
- **Analyst Consensus Target:** ₹{analyst.get('target_mean', 0):,.0f}
- **Analyst Coverage:** {analyst.get('num_analysts', 0)} analysts

### Near-Term Catalysts

1. Upcoming quarterly results and earnings growth
2. Sector tailwinds and market share gains
3. New product/service launches
4. Potential dividend announcements
"""


def _generate_company_overview(data: Dict[str, Any]) -> str:
    """Generate company overview section."""
    return f"""
## Company Overview

### Business Description

{data.get('company_name', '')} operates in the **{data.get('sector', 'N/A')}** sector,
specifically within the **{data.get('industry', 'N/A')}** industry.

### Key Business Metrics

| Metric | Value |
|--------|------:|
| Sector | {data.get('sector', 'N/A')} |
| Industry | {data.get('industry', 'N/A')} |
| Market Cap | ₹{data.get('market_cap', 0)/1e7:,.0f} Cr |
| Enterprise Value | ₹{data.get('enterprise_value', 0)/1e7:,.0f} Cr |
| Revenue (TTM) | ₹{data.get('revenue', 0)/1e7:,.0f} Cr |
| EBITDA (TTM) | ₹{data.get('ebitda', 0)/1e7:,.0f} Cr |
| Employees | {data.get('employees', 'N/A')} |
"""


def _generate_financial_section(financials: 'FinancialStatements',
                                  formatter: 'FinancialTablesFormatter',
                                  data: Dict[str, Any]) -> str:
    """Generate financial analysis section."""
    section = """
## Financial Analysis

### Income Statement Summary

"""
    section += formatter.create_income_statement_table(financials)

    section += "\n\n### Key Financial Ratios\n\n"
    section += formatter.create_ratio_table(financials)

    # Add TTM metrics
    section += f"""

### Trailing Twelve Months (TTM) Metrics

| Metric | Value |
|--------|------:|
| Revenue | ₹{data.get('revenue', 0)/1e7:,.0f} Cr |
| EBITDA | ₹{data.get('ebitda', 0)/1e7:,.0f} Cr |
| Net Income | ₹{data.get('net_income', 0)/1e7:,.0f} Cr |
| EPS | ₹{data.get('eps', 0):.2f} |
| EBITDA Margin | {data.get('ebitda_margin', 0):.1f}% |
| Net Margin | {data.get('profit_margin', 0):.1f}% |
| ROE | {data.get('roe', 0):.1f}% |
| Debt/Equity | {data.get('debt_to_equity', 0):.2f}x |
"""

    return section


def _generate_technical_section(data: Dict[str, Any]) -> str:
    """Generate technical analysis section."""
    prices = data.get('price_history', [])
    if not prices:
        return ""

    current = prices[-1] if prices else 0
    high_52w = data.get('week_52_high', 0)
    low_52w = data.get('week_52_low', 0)

    # Simple moving averages
    sma_20 = sum(prices[-20:]) / 20 if len(prices) >= 20 else current
    sma_50 = sum(prices[-50:]) / 50 if len(prices) >= 50 else current
    sma_200 = sum(prices[-200:]) / 200 if len(prices) >= 200 else current

    # Trend
    trend = "Bullish" if current > sma_50 > sma_200 else "Bearish" if current < sma_50 < sma_200 else "Neutral"

    return f"""
## Technical Analysis

### Price Action Summary

| Metric | Value |
|--------|------:|
| Current Price | ₹{current:,.2f} |
| 52-Week High | ₹{high_52w:,.2f} |
| 52-Week Low | ₹{low_52w:,.2f} |
| % from 52W High | {((current - high_52w) / high_52w * 100) if high_52w else 0:.1f}% |
| % from 52W Low | {((current - low_52w) / low_52w * 100) if low_52w else 0:.1f}% |

### Moving Averages

| Moving Average | Value | Signal |
|---------------|------:|--------|
| 20-Day SMA | ₹{sma_20:,.2f} | {'Above' if current > sma_20 else 'Below'} |
| 50-Day SMA | ₹{sma_50:,.2f} | {'Above' if current > sma_50 else 'Below'} |
| 200-Day SMA | ₹{sma_200:,.2f} | {'Above' if current > sma_200 else 'Below'} |

**Overall Trend:** {trend}

### Support & Resistance

- **Immediate Support:** ₹{low_52w + (current - low_52w) * 0.3:,.0f}
- **Strong Support:** ₹{low_52w:,.0f} (52-week low)
- **Immediate Resistance:** ₹{current + (high_52w - current) * 0.3:,.0f}
- **Strong Resistance:** ₹{high_52w:,.0f} (52-week high)
"""


def _generate_shareholding_section(data: Dict[str, Any], snapshot: 'CompanySnapshot') -> str:
    """Generate shareholding pattern section."""
    promoter = snapshot.promoter_holding
    fii = snapshot.fii_holding
    dii = snapshot.dii_holding
    public = 100 - promoter - fii - dii

    return f"""
## Shareholding Pattern

| Category | Holding (%) |
|----------|------------:|
| Promoters | {promoter:.2f}% |
| FII/FPI | {fii:.2f}% |
| DII | {dii:.2f}% |
| Public | {public:.2f}% |

### Analysis

- **Promoter Holding:** {'Strong' if promoter > 50 else 'Moderate' if promoter > 30 else 'Low'} promoter stake indicates {'high' if promoter > 50 else 'moderate'} management confidence
- **FII Interest:** {'High' if fii > 20 else 'Moderate' if fii > 10 else 'Low'} foreign institutional interest
- **DII Support:** {'Strong' if dii > 15 else 'Moderate' if dii > 5 else 'Limited'} domestic institutional support
"""


def _generate_risk_section(risks: List[str], data: Dict[str, Any]) -> str:
    """Generate risk analysis section."""
    section = """
## Risk Analysis

### Key Investment Risks

"""
    for i, risk in enumerate(risks, 1):
        section += f"{i}. **{risk}**\n"

    section += f"""

### Risk Metrics

| Metric | Value | Assessment |
|--------|------:|------------|
| Beta | {data.get('beta', 1.0):.2f} | {'High Vol' if data.get('beta', 1) > 1.2 else 'Low Vol' if data.get('beta', 1) < 0.8 else 'Market Vol'} |
| Debt/Equity | {data.get('debt_to_equity', 0):.2f}x | {'High' if data.get('debt_to_equity', 0) > 1 else 'Moderate' if data.get('debt_to_equity', 0) > 0.5 else 'Low'} |
| Current Ratio | {data.get('current_ratio', 0):.2f}x | {'Strong' if data.get('current_ratio', 0) > 1.5 else 'Adequate' if data.get('current_ratio', 0) > 1 else 'Weak'} |
"""

    return section


def _generate_recommendation(snapshot: 'CompanySnapshot', dcf: Dict[str, float],
                               analyst: Dict[str, Any]) -> str:
    """Generate investment recommendation section."""
    upside_str = f"+{snapshot.upside:.1f}%" if snapshot.upside > 0 else f"{snapshot.upside:.1f}%"

    return f"""
## Investment Recommendation

### Rating: {snapshot.rating}

### Target Price: ₹{snapshot.target_price:,.0f}

| Valuation Method | Implied Price |
|-----------------|-------------:|
| DCF Value | ₹{dcf.get('implied_price', 0):,.0f} |
| Analyst Consensus | ₹{analyst.get('target_mean', 0):,.0f} |
| **Our Target** | **₹{snapshot.target_price:,.0f}** |

### Upside/Downside

- Current Price: ₹{snapshot.current_price:,.2f}
- Target Price: ₹{snapshot.target_price:,.0f}
- **Potential Return: {upside_str}**

### Investment Horizon

- **Short Term (3-6 months):** {'Positive' if snapshot.upside > 5 else 'Neutral' if snapshot.upside > -5 else 'Negative'}
- **Medium Term (6-12 months):** {'Positive' if snapshot.upside > 10 else 'Neutral' if snapshot.upside > 0 else 'Negative'}
- **Long Term (1-3 years):** {'Positive' if snapshot.roe > 15 else 'Neutral'}
"""


def _generate_disclaimer() -> str:
    """Generate disclaimer section."""
    return f"""
---

## Disclaimer

This report is generated by Jotty Research for informational purposes only and does not constitute
investment advice. The information contained herein is obtained from sources believed to be reliable,
but its accuracy and completeness cannot be guaranteed.

Investors should conduct their own due diligence and consult with a qualified financial advisor
before making investment decisions. Past performance is not indicative of future results.

**Report Generated:** {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}

**Analyst:** Jotty AI Research

---

*© {datetime.now().year} Jotty Research. All rights reserved.*
"""


async def _convert_to_pdf(
    md_path: Path,
    output_dir: Path,
    ticker: str,
    timestamp: str,
    chart_files: List[str] = None,
    template_name: str = None
) -> Optional[str]:
    """Convert markdown to professionally styled PDF with embedded charts."""
    try:
        pdf_filename = f"{ticker}_research_{timestamp}.pdf"
        pdf_path = output_dir / pdf_filename

        # Try our professional PDF template first
        try:
            from .pdf_template import convert_md_to_pdf
            result_path = await convert_md_to_pdf(
                str(md_path),
                str(pdf_path),
                template_name=template_name,
                chart_files=chart_files
            )
            if result_path:
                logger.info(f"✅ Professional PDF generated: {result_path}")
                return result_path
        except ImportError as e:
            logger.info(f"Professional PDF template needs dependencies: {e}")
        except Exception as e:
            logger.warning(f"Professional PDF generation failed: {e}")

        # Fallback to document-converter skill
        try:
            from Jotty.core.registry.skills_registry import get_skills_registry
            registry = get_skills_registry()
            registry.init()

            converter = registry.get_skill('document-converter')
            if converter:
                convert_tool = converter.tools.get('convert_to_pdf_tool')
                if convert_tool:
                    import inspect
                    if inspect.iscoroutinefunction(convert_tool):
                        result = await convert_tool({
                            'input_file': str(md_path),
                            'output_file': str(pdf_path),
                            'page_size': 'a4',
                        })
                    else:
                        result = convert_tool({
                            'input_file': str(md_path),
                            'output_file': str(pdf_path),
                            'page_size': 'a4',
                        })

                    if result.get('success'):
                        return result.get('output_path', str(pdf_path))
        except Exception as e:
            logger.warning(f"Document converter failed: {e}")

        return None

    except Exception as e:
        logger.warning(f"PDF conversion failed: {e}")
        return None


async def _send_to_telegram(pdf_path: str, ticker: str, snapshot: 'CompanySnapshot') -> bool:
    """Send report to Telegram."""
    try:
        from Jotty.core.registry.skills_registry import get_skills_registry
        registry = get_skills_registry()
        registry.init()

        telegram = registry.get_skill('telegram-sender')
        if telegram:
            send_tool = telegram.tools.get('send_telegram_file_tool')
            if send_tool:
                caption = (
                    f"📊 {snapshot.company_name} ({ticker}) Research Report\n\n"
                    f"Rating: {snapshot.rating} | Target: ₹{snapshot.target_price:,.0f}\n"
                    f"Upside: {'+' if snapshot.upside > 0 else ''}{snapshot.upside:.1f}%"
                )

                import inspect
                if inspect.iscoroutinefunction(send_tool):
                    result = await send_tool({
                        'file_path': pdf_path,
                        'caption': caption,
                    })
                else:
                    result = send_tool({
                        'file_path': pdf_path,
                        'caption': caption,
                    })

                return result.get('success', False)

        return False

    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")
        return False
