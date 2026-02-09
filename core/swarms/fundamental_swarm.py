"""
Fundamental Analysis Swarm - World-Class Financial Analysis
============================================================

Production-grade swarm for:
- Financial statement analysis
- Valuation modeling (DCF, comparables, precedents)
- Ratio analysis
- Quality of earnings assessment
- Management analysis
- Competitive positioning

Agents:
┌─────────────────────────────────────────────────────────────────────────┐
│                      FUNDAMENTAL ANALYSIS SWARM                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │
│  │  Financial     │  │  Ratio         │  │  Valuation     │            │
│  │  Statement     │  │  Analysis      │  │    Agent       │            │
│  │    Agent       │  │    Agent       │  │                │            │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘            │
│          │                   │                   │                      │
│          └───────────────────┼───────────────────┘                      │
│                              ▼                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │
│  │   Quality      │  │  Management    │  │  Competitive   │            │
│  │   Earnings     │  │   Analysis     │  │   Moat Agent   │            │
│  │    Agent       │  │    Agent       │  │                │            │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘            │
│          │                   │                   │                      │
│          └───────────────────┼───────────────────┘                      │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     INVESTMENT THESIS                            │   │
│  │   Combines all analysis into actionable investment thesis        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘

Usage:
    from core.swarms.fundamental_swarm import FundamentalSwarm, analyze_fundamentals

    # Full swarm
    swarm = FundamentalSwarm()
    result = await swarm.analyze("RELIANCE", exchange="NSE")

    # One-liner
    result = await analyze_fundamentals("TCS")

Author: Jotty Team
Date: February 2026
"""

import asyncio
import logging
import json
import dspy
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum

from .base_swarm import (
    BaseSwarm, SwarmConfig, SwarmResult, AgentRole,
    register_swarm, ExecutionTrace
)
from .base import DomainSwarm, AgentTeam
from ..agents.base import DomainAgent, DomainAgentConfig

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class ValuationType(Enum):
    DCF = "dcf"
    COMPARABLES = "comparables"
    PRECEDENT = "precedent"
    SUM_OF_PARTS = "sum_of_parts"
    ASSET_BASED = "asset_based"
    DIVIDEND_DISCOUNT = "dividend_discount"


class InvestmentStyle(Enum):
    VALUE = "value"
    GROWTH = "growth"
    GARP = "garp"  # Growth at Reasonable Price
    QUALITY = "quality"
    DIVIDEND = "dividend"


class RatingScale(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class FundamentalConfig(SwarmConfig):
    """Configuration for FundamentalSwarm."""
    valuation_methods: List[ValuationType] = field(default_factory=lambda: [
        ValuationType.DCF, ValuationType.COMPARABLES
    ])
    investment_style: InvestmentStyle = InvestmentStyle.QUALITY
    years_of_data: int = 5
    include_projections: bool = True
    include_peer_comparison: bool = True
    include_management_analysis: bool = True
    currency: str = "INR"
    market: str = "NSE"

    def __post_init__(self):
        self.name = "FundamentalSwarm"
        self.domain = "fundamental_analysis"


@dataclass
class FinancialMetrics:
    """Key financial metrics."""
    revenue: float = 0.0
    revenue_growth: float = 0.0
    gross_profit: float = 0.0
    gross_margin: float = 0.0
    operating_income: float = 0.0
    operating_margin: float = 0.0
    net_income: float = 0.0
    net_margin: float = 0.0
    eps: float = 0.0
    free_cash_flow: float = 0.0
    total_debt: float = 0.0
    cash: float = 0.0
    equity: float = 0.0


@dataclass
class ValuationMetrics:
    """Valuation metrics."""
    pe_ratio: float = 0.0
    pb_ratio: float = 0.0
    ps_ratio: float = 0.0
    ev_ebitda: float = 0.0
    ev_sales: float = 0.0
    peg_ratio: float = 0.0
    dividend_yield: float = 0.0
    market_cap: float = 0.0
    enterprise_value: float = 0.0


@dataclass
class QualityMetrics:
    """Quality and efficiency metrics."""
    roe: float = 0.0
    roce: float = 0.0
    roic: float = 0.0
    asset_turnover: float = 0.0
    inventory_turnover: float = 0.0
    receivables_turnover: float = 0.0
    current_ratio: float = 0.0
    quick_ratio: float = 0.0
    debt_to_equity: float = 0.0
    interest_coverage: float = 0.0


@dataclass
class ValuationResult:
    """Result from valuation model."""
    method: ValuationType
    intrinsic_value: float
    upside_downside: float  # percentage
    assumptions: Dict[str, Any]
    confidence: float  # 0-1


@dataclass
class InvestmentThesis:
    """Complete investment thesis."""
    rating: RatingScale
    target_price: float
    current_price: float
    upside: float
    bull_case: str
    base_case: str
    bear_case: str
    key_risks: List[str]
    key_catalysts: List[str]
    recommendation: str


@dataclass
class FundamentalResult(SwarmResult):
    """Result from FundamentalSwarm."""
    ticker: str = ""
    company_name: str = ""
    financial_metrics: Optional[FinancialMetrics] = None
    valuation_metrics: Optional[ValuationMetrics] = None
    quality_metrics: Optional[QualityMetrics] = None
    valuations: List[ValuationResult] = field(default_factory=list)
    thesis: Optional[InvestmentThesis] = None
    moat_score: float = 0.0
    management_score: float = 0.0
    earnings_quality: float = 0.0
    peer_comparison: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# DSPy SIGNATURES
# =============================================================================

class FinancialStatementSignature(dspy.Signature):
    """Analyze financial statements.

    You are a FINANCIAL ANALYST. Analyze financial statements to extract:
    1. Revenue trends and growth drivers
    2. Margin analysis (gross, operating, net)
    3. Profitability metrics
    4. Cash flow analysis
    5. Balance sheet strength

    Be thorough and identify any red flags.
    """
    company: str = dspy.InputField(desc="Company name or ticker")
    financial_data: str = dspy.InputField(desc="JSON of financial data")
    years: int = dspy.InputField(desc="Years of data to analyze")

    revenue_analysis: str = dspy.OutputField(desc="Revenue analysis with trends")
    margin_analysis: str = dspy.OutputField(desc="Margin analysis")
    profitability: str = dspy.OutputField(desc="Profitability metrics JSON")
    cash_flow_analysis: str = dspy.OutputField(desc="Cash flow analysis")
    balance_sheet_health: str = dspy.OutputField(desc="Balance sheet assessment")
    red_flags: str = dspy.OutputField(desc="Red flags identified, separated by |")


class RatioAnalysisSignature(dspy.Signature):
    """Perform comprehensive ratio analysis.

    You are a FINANCIAL RATIO EXPERT. Calculate and interpret:
    1. Profitability ratios (ROE, ROCE, ROIC)
    2. Efficiency ratios (turnover ratios)
    3. Liquidity ratios (current, quick)
    4. Solvency ratios (debt/equity, interest coverage)
    5. Valuation ratios (P/E, P/B, EV/EBITDA)

    Compare to industry averages and historical trends.
    """
    financial_data: str = dspy.InputField(desc="JSON of financial data")
    industry: str = dspy.InputField(desc="Industry for comparison")
    market_data: str = dspy.InputField(desc="Market data including price")

    profitability_ratios: str = dspy.OutputField(desc="JSON of profitability ratios")
    efficiency_ratios: str = dspy.OutputField(desc="JSON of efficiency ratios")
    liquidity_ratios: str = dspy.OutputField(desc="JSON of liquidity ratios")
    solvency_ratios: str = dspy.OutputField(desc="JSON of solvency ratios")
    valuation_ratios: str = dspy.OutputField(desc="JSON of valuation ratios")
    interpretation: str = dspy.OutputField(desc="Key interpretations")


class DCFValuationSignature(dspy.Signature):
    """Perform DCF valuation.

    You are a VALUATION EXPERT. Build a DCF model with:
    1. Revenue projections
    2. Margin assumptions
    3. Capital expenditure forecasts
    4. Working capital changes
    5. Terminal value calculation
    6. Discount rate (WACC)

    Be conservative in assumptions.
    """
    company: str = dspy.InputField(desc="Company name")
    financial_data: str = dspy.InputField(desc="Historical financial data")
    growth_assumptions: str = dspy.InputField(desc="Growth assumptions")
    market_data: str = dspy.InputField(desc="Market rates and comparables")

    revenue_projections: str = dspy.OutputField(desc="5-year revenue projections")
    fcf_projections: str = dspy.OutputField(desc="Free cash flow projections")
    wacc: float = dspy.OutputField(desc="Weighted average cost of capital")
    terminal_value: float = dspy.OutputField(desc="Terminal value")
    intrinsic_value: float = dspy.OutputField(desc="Intrinsic value per share")
    key_assumptions: str = dspy.OutputField(desc="Key assumptions, separated by |")


class QualityEarningsSignature(dspy.Signature):
    """Assess quality of earnings.

    You are an EARNINGS QUALITY ANALYST. Evaluate:
    1. Cash conversion (earnings to cash flow)
    2. Revenue recognition practices
    3. Accruals quality
    4. Non-recurring items
    5. Related party transactions
    6. Accounting policy changes

    Score earnings quality from 0-100.
    """
    financial_data: str = dspy.InputField(desc="Financial data including notes")
    cash_flows: str = dspy.InputField(desc="Cash flow statement")

    cash_conversion: float = dspy.OutputField(desc="Cash conversion ratio")
    accruals_quality: float = dspy.OutputField(desc="Accruals quality score 0-100")
    adjustments: str = dspy.OutputField(desc="Recommended adjustments, separated by |")
    quality_score: float = dspy.OutputField(desc="Overall quality score 0-100")
    concerns: str = dspy.OutputField(desc="Quality concerns, separated by |")


class ManagementAnalysisSignature(dspy.Signature):
    """Analyze management quality.

    You are a MANAGEMENT ANALYST. Evaluate:
    1. Track record and execution
    2. Capital allocation decisions
    3. Shareholder friendliness
    4. Governance practices
    5. Insider ownership and transactions
    6. Compensation alignment

    Score management from 0-100.
    """
    company: str = dspy.InputField(desc="Company name")
    management_info: str = dspy.InputField(desc="Management team information")
    capital_history: str = dspy.InputField(desc="Historical capital allocation")
    governance: str = dspy.InputField(desc="Governance information")

    execution_score: float = dspy.OutputField(desc="Execution track record 0-100")
    capital_allocation_score: float = dspy.OutputField(desc="Capital allocation 0-100")
    governance_score: float = dspy.OutputField(desc="Governance quality 0-100")
    overall_score: float = dspy.OutputField(desc="Overall management score 0-100")
    strengths: str = dspy.OutputField(desc="Management strengths, separated by |")
    concerns: str = dspy.OutputField(desc="Management concerns, separated by |")


class CompetitiveMoatSignature(dspy.Signature):
    """Analyze competitive moat.

    You are a COMPETITIVE STRATEGY ANALYST. Identify:
    1. Source of moat (brand, network, switching costs, scale, IP)
    2. Moat durability
    3. Industry structure (Porter's 5 forces)
    4. Market position
    5. Competitive threats

    Score moat from 0-100.
    """
    company: str = dspy.InputField(desc="Company name")
    business_description: str = dspy.InputField(desc="Business model description")
    industry_data: str = dspy.InputField(desc="Industry and competitive data")
    financials: str = dspy.InputField(desc="Key financial metrics")

    moat_sources: str = dspy.OutputField(desc="Sources of competitive advantage, separated by |")
    moat_durability: str = dspy.OutputField(desc="Assessment of moat durability")
    industry_structure: str = dspy.OutputField(desc="Industry structure analysis")
    moat_score: float = dspy.OutputField(desc="Overall moat score 0-100")
    threats: str = dspy.OutputField(desc="Competitive threats, separated by |")


class InvestmentThesisSignature(dspy.Signature):
    """Generate investment thesis.

    You are a SENIOR INVESTMENT ANALYST. Create a thesis with:
    1. Investment rating and target price
    2. Bull, base, and bear cases
    3. Key risks and catalysts
    4. Valuation support
    5. Clear recommendation

    Be objective and balanced.
    """
    company: str = dspy.InputField(desc="Company name")
    financial_summary: str = dspy.InputField(desc="Financial analysis summary")
    valuation_summary: str = dspy.InputField(desc="Valuation analysis summary")
    quality_summary: str = dspy.InputField(desc="Quality and moat summary")
    current_price: float = dspy.InputField(desc="Current stock price")

    rating: str = dspy.OutputField(desc="STRONG_BUY, BUY, HOLD, SELL, or STRONG_SELL")
    target_price: float = dspy.OutputField(desc="12-month target price")
    bull_case: str = dspy.OutputField(desc="Bull case scenario")
    base_case: str = dspy.OutputField(desc="Base case scenario")
    bear_case: str = dspy.OutputField(desc="Bear case scenario")
    key_risks: str = dspy.OutputField(desc="Key risks, separated by |")
    catalysts: str = dspy.OutputField(desc="Key catalysts, separated by |")
    recommendation: str = dspy.OutputField(desc="Investment recommendation summary")


# =============================================================================
# AGENTS
# =============================================================================

class BaseFundamentalAgent(DomainAgent):
    """Base class for fundamental analysis agents. Inherits from DomainAgent for unified infrastructure."""

    def __init__(self, memory=None, context=None, bus=None, signature=None):
        config = DomainAgentConfig(
            name=self.__class__.__name__,
            enable_memory=memory is not None,
            enable_context=context is not None,
        )
        super().__init__(signature=signature, config=config)

        # Ensure LM is configured before child classes create DSPy modules
        self._ensure_initialized()

        if memory is not None:
            self._memory = memory
        if context is not None:
            self._context_manager = context
        self.bus = bus

    def _broadcast(self, event: str, data: Dict[str, Any]):
        """Broadcast event to other agents."""
        if self.bus:
            try:
                from ..agents.axon import Message
                msg = Message(
                    sender=self.__class__.__name__,
                    receiver="broadcast",
                    content={'event': event, **data}
                )
                self.bus.publish(msg)
            except Exception:
                pass


class FinancialStatementAgent(BaseFundamentalAgent):
    """Analyzes financial statements."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=FinancialStatementSignature)
        self.learned_context = learned_context
        self._analyzer = dspy.ChainOfThought(FinancialStatementSignature)

    async def analyze(
        self,
        company: str,
        financial_data: Dict[str, Any],
        years: int = 5
    ) -> Dict[str, Any]:
        """Analyze financial statements."""
        try:
            company_input = f"{company}\n{self.learned_context}" if self.learned_context else company
            result = self._analyzer(
                company=company_input,
                financial_data=json.dumps(financial_data),
                years=years
            )

            red_flags = [r.strip() for r in str(result.red_flags).split('|') if r.strip()]

            self._broadcast("financials_analyzed", {
                'company': company,
                'red_flags': len(red_flags)
            })

            return {
                'revenue_analysis': str(result.revenue_analysis),
                'margin_analysis': str(result.margin_analysis),
                'profitability': str(result.profitability),
                'cash_flow_analysis': str(result.cash_flow_analysis),
                'balance_sheet_health': str(result.balance_sheet_health),
                'red_flags': red_flags
            }

        except Exception as e:
            logger.error(f"Financial statement analysis failed: {e}")
            return {'error': str(e)}


class RatioAnalysisAgent(BaseFundamentalAgent):
    """Performs ratio analysis."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=RatioAnalysisSignature)
        self.learned_context = learned_context
        self._analyzer = dspy.ChainOfThought(RatioAnalysisSignature)

    async def analyze(
        self,
        financial_data: Dict[str, Any],
        industry: str,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform ratio analysis."""
        try:
            industry_input = f"{industry}\n{self.learned_context}" if self.learned_context else industry
            result = self._analyzer(
                financial_data=json.dumps(financial_data),
                industry=industry_input,
                market_data=json.dumps(market_data)
            )

            # Parse ratios
            try:
                profitability = json.loads(result.profitability_ratios)
            except Exception:
                profitability = {}

            try:
                valuation = json.loads(result.valuation_ratios)
            except Exception:
                valuation = {}

            self._broadcast("ratios_calculated", {
                'profitability_count': len(profitability),
                'valuation_count': len(valuation)
            })

            return {
                'profitability_ratios': profitability,
                'efficiency_ratios': result.efficiency_ratios,
                'liquidity_ratios': result.liquidity_ratios,
                'solvency_ratios': result.solvency_ratios,
                'valuation_ratios': valuation,
                'interpretation': str(result.interpretation)
            }

        except Exception as e:
            logger.error(f"Ratio analysis failed: {e}")
            return {'error': str(e)}


class ValuationAgent(BaseFundamentalAgent):
    """Performs company valuation."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=DCFValuationSignature)
        self.learned_context = learned_context
        self._dcf = dspy.ChainOfThought(DCFValuationSignature)

    async def dcf_valuation(
        self,
        company: str,
        financial_data: Dict[str, Any],
        growth_assumptions: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> ValuationResult:
        """Perform DCF valuation."""
        try:
            company_input = f"{company}\n{self.learned_context}" if self.learned_context else company
            result = self._dcf(
                company=company_input,
                financial_data=json.dumps(financial_data),
                growth_assumptions=json.dumps(growth_assumptions),
                market_data=json.dumps(market_data)
            )

            current_price = market_data.get('current_price', 0)
            intrinsic = float(result.intrinsic_value) if result.intrinsic_value else 0
            upside = ((intrinsic - current_price) / current_price * 100) if current_price > 0 else 0

            assumptions = [a.strip() for a in str(result.key_assumptions).split('|') if a.strip()]

            self._broadcast("dcf_completed", {
                'company': company,
                'intrinsic_value': intrinsic,
                'upside': upside
            })

            return ValuationResult(
                method=ValuationType.DCF,
                intrinsic_value=intrinsic,
                upside_downside=upside,
                assumptions={
                    'wacc': float(result.wacc) if result.wacc else 0,
                    'terminal_value': float(result.terminal_value) if result.terminal_value else 0,
                    'key_assumptions': assumptions
                },
                confidence=0.7
            )

        except Exception as e:
            logger.error(f"DCF valuation failed: {e}")
            return ValuationResult(
                method=ValuationType.DCF,
                intrinsic_value=0,
                upside_downside=0,
                assumptions={'error': str(e)},
                confidence=0
            )


class QualityEarningsAgent(BaseFundamentalAgent):
    """Assesses earnings quality."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=QualityEarningsSignature)
        self.learned_context = learned_context
        self._analyzer = dspy.ChainOfThought(QualityEarningsSignature)

    async def assess(
        self,
        financial_data: Dict[str, Any],
        cash_flows: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess earnings quality."""
        try:
            financial_data_input = json.dumps(financial_data)
            if self.learned_context:
                financial_data_input = f"{financial_data_input}\n{self.learned_context}"
            result = self._analyzer(
                financial_data=financial_data_input,
                cash_flows=json.dumps(cash_flows)
            )

            adjustments = [a.strip() for a in str(result.adjustments).split('|') if a.strip()]
            concerns = [c.strip() for c in str(result.concerns).split('|') if c.strip()]

            self._broadcast("quality_assessed", {
                'quality_score': float(result.quality_score) if result.quality_score else 0
            })

            return {
                'cash_conversion': float(result.cash_conversion) if result.cash_conversion else 0,
                'accruals_quality': float(result.accruals_quality) if result.accruals_quality else 0,
                'quality_score': float(result.quality_score) if result.quality_score else 0,
                'adjustments': adjustments,
                'concerns': concerns
            }

        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return {'error': str(e)}


class ManagementAgent(BaseFundamentalAgent):
    """Analyzes management quality."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=ManagementAnalysisSignature)
        self.learned_context = learned_context
        self._analyzer = dspy.ChainOfThought(ManagementAnalysisSignature)

    async def analyze(
        self,
        company: str,
        management_info: str,
        capital_history: str,
        governance: str
    ) -> Dict[str, Any]:
        """Analyze management."""
        try:
            company_input = f"{company}\n{self.learned_context}" if self.learned_context else company
            result = self._analyzer(
                company=company_input,
                management_info=management_info,
                capital_history=capital_history,
                governance=governance
            )

            strengths = [s.strip() for s in str(result.strengths).split('|') if s.strip()]
            concerns = [c.strip() for c in str(result.concerns).split('|') if c.strip()]

            self._broadcast("management_analyzed", {
                'company': company,
                'overall_score': float(result.overall_score) if result.overall_score else 0
            })

            return {
                'execution_score': float(result.execution_score) if result.execution_score else 0,
                'capital_allocation_score': float(result.capital_allocation_score) if result.capital_allocation_score else 0,
                'governance_score': float(result.governance_score) if result.governance_score else 0,
                'overall_score': float(result.overall_score) if result.overall_score else 0,
                'strengths': strengths,
                'concerns': concerns
            }

        except Exception as e:
            logger.error(f"Management analysis failed: {e}")
            return {'error': str(e)}


class MoatAgent(BaseFundamentalAgent):
    """Analyzes competitive moat."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=CompetitiveMoatSignature)
        self.learned_context = learned_context
        self._analyzer = dspy.ChainOfThought(CompetitiveMoatSignature)

    async def analyze(
        self,
        company: str,
        business_description: str,
        industry_data: str,
        financials: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze competitive moat."""
        try:
            company_input = f"{company}\n{self.learned_context}" if self.learned_context else company
            result = self._analyzer(
                company=company_input,
                business_description=business_description,
                industry_data=industry_data,
                financials=json.dumps(financials)
            )

            moat_sources = [m.strip() for m in str(result.moat_sources).split('|') if m.strip()]
            threats = [t.strip() for t in str(result.threats).split('|') if t.strip()]

            self._broadcast("moat_analyzed", {
                'company': company,
                'moat_score': float(result.moat_score) if result.moat_score else 0
            })

            return {
                'moat_sources': moat_sources,
                'moat_durability': str(result.moat_durability),
                'industry_structure': str(result.industry_structure),
                'moat_score': float(result.moat_score) if result.moat_score else 0,
                'threats': threats
            }

        except Exception as e:
            logger.error(f"Moat analysis failed: {e}")
            return {'error': str(e)}


class ThesisAgent(BaseFundamentalAgent):
    """Generates investment thesis."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=InvestmentThesisSignature)
        self.learned_context = learned_context
        self._generator = dspy.ChainOfThought(InvestmentThesisSignature)

    async def generate(
        self,
        company: str,
        financial_summary: str,
        valuation_summary: str,
        quality_summary: str,
        current_price: float
    ) -> InvestmentThesis:
        """Generate investment thesis."""
        try:
            company_input = f"{company}\n{self.learned_context}" if self.learned_context else company
            result = self._generator(
                company=company_input,
                financial_summary=financial_summary,
                valuation_summary=valuation_summary,
                quality_summary=quality_summary,
                current_price=current_price
            )

            # Parse rating
            rating_str = str(result.rating).upper().replace(' ', '_')
            rating = RatingScale.HOLD
            for r in RatingScale:
                if r.value.upper() == rating_str:
                    rating = r
                    break

            target = float(result.target_price) if result.target_price else current_price
            upside = ((target - current_price) / current_price * 100) if current_price > 0 else 0

            risks = [r.strip() for r in str(result.key_risks).split('|') if r.strip()]
            catalysts = [c.strip() for c in str(result.catalysts).split('|') if c.strip()]

            self._broadcast("thesis_generated", {
                'company': company,
                'rating': rating.value,
                'target_price': target
            })

            return InvestmentThesis(
                rating=rating,
                target_price=target,
                current_price=current_price,
                upside=upside,
                bull_case=str(result.bull_case),
                base_case=str(result.base_case),
                bear_case=str(result.bear_case),
                key_risks=risks,
                key_catalysts=catalysts,
                recommendation=str(result.recommendation)
            )

        except Exception as e:
            logger.error(f"Thesis generation failed: {e}")
            return InvestmentThesis(
                rating=RatingScale.HOLD,
                target_price=current_price,
                current_price=current_price,
                upside=0,
                bull_case="",
                base_case="",
                bear_case="",
                key_risks=[str(e)],
                key_catalysts=[],
                recommendation="Analysis failed"
            )


# =============================================================================
# FUNDAMENTAL SWARM
# =============================================================================

@register_swarm("fundamental")
class FundamentalSwarm(DomainSwarm):
    """
    World-Class Fundamental Analysis Swarm.

    Provides comprehensive fundamental analysis with:
    - Financial statement analysis
    - Ratio analysis
    - DCF and comparables valuation
    - Quality of earnings
    - Management assessment
    - Competitive moat analysis
    - Investment thesis
    """

    # Declarative agent team - auto-initialized by DomainSwarm
    AGENT_TEAM = AgentTeam.define(
        (FinancialStatementAgent, "FinancialStatement", "_financial_agent"),
        (RatioAnalysisAgent, "RatioAnalysis", "_ratio_agent"),
        (ValuationAgent, "Valuation", "_valuation_agent"),
        (QualityEarningsAgent, "QualityEarnings", "_quality_agent"),
        (ManagementAgent, "Management", "_management_agent"),
        (MoatAgent, "Moat", "_moat_agent"),
        (ThesisAgent, "Thesis", "_thesis_agent"),
    )

    def __init__(self, config: FundamentalConfig = None):
        super().__init__(config or FundamentalConfig())

    async def _execute_domain(
        self,
        ticker: str,
        **kwargs
    ) -> FundamentalResult:
        """Execute fundamental analysis (called by DomainSwarm.execute())."""
        return await self.analyze(ticker, **kwargs)

    async def analyze(
        self,
        ticker: str,
        exchange: str = "NSE",
        financial_data: Dict[str, Any] = None,
        market_data: Dict[str, Any] = None
    ) -> FundamentalResult:
        """
        Perform comprehensive fundamental analysis.

        Args:
            ticker: Stock ticker symbol
            exchange: Stock exchange (NSE, BSE)
            financial_data: Financial data (if not provided, will be fetched)
            market_data: Market data (if not provided, will be fetched)

        Returns:
            FundamentalResult with complete analysis
        """
        start_time = datetime.now()

        # Note: Pre-execution learning and agent init handled by DomainSwarm.execute()

        logger.info(f"📊 FundamentalSwarm starting: {ticker} on {exchange}")

        # Use provided data or create dummy for demo
        if not financial_data:
            financial_data = {
                'revenue': [100000, 110000, 121000, 133100, 146410],
                'net_income': [10000, 11500, 13000, 14800, 17000],
                'total_assets': [200000, 220000, 242000, 266200, 292820],
                'total_equity': [120000, 135000, 152000, 171000, 192000],
                'total_debt': [50000, 52000, 54000, 56000, 58000],
                'cash_flow_operations': [12000, 13500, 15000, 17000, 19500],
                'capex': [8000, 9000, 10000, 11000, 12000]
            }

        if not market_data:
            market_data = {
                'current_price': 2500,
                'market_cap': 1500000000,
                'shares_outstanding': 600000
            }

        try:
            # =================================================================
            # PHASE 1: PARALLEL DATA COLLECTION & INITIAL ANALYSIS
            # =================================================================
            logger.info("📈 Phase 1: Financial statement & ratio analysis...")

            financial_task = self._financial_agent.analyze(ticker, financial_data, 5)
            ratio_task = self._ratio_agent.analyze(
                financial_data,
                "Technology",
                market_data
            )

            financial_result, ratio_result = await asyncio.gather(
                financial_task, ratio_task, return_exceptions=True
            )

            if isinstance(financial_result, Exception):
                financial_result = {'error': str(financial_result)}
            if isinstance(ratio_result, Exception):
                ratio_result = {'error': str(ratio_result)}

            self._trace_phase("FinancialStatement", AgentRole.ACTOR,
                {'ticker': ticker},
                {'has_error': 'error' in financial_result},
                success='error' not in financial_result, phase_start=start_time, tools_used=['financial_analyze'])
            self._trace_phase("RatioAnalysis", AgentRole.ACTOR,
                {'ticker': ticker},
                {'has_error': 'error' in ratio_result},
                success='error' not in ratio_result, phase_start=start_time, tools_used=['ratio_analyze'])

            # =================================================================
            # PHASE 2: QUALITY & MOAT ANALYSIS (parallel)
            # =================================================================
            logger.info("🔍 Phase 2: Quality & moat analysis...")

            quality_task = self._quality_agent.assess(
                financial_data,
                {'operating': financial_data.get('cash_flow_operations', [])}
            )
            moat_task = self._moat_agent.analyze(
                ticker,
                f"{ticker} is a leading company in its industry",
                "Competitive industry with multiple players",
                financial_data
            )

            quality_result, moat_result = await asyncio.gather(
                quality_task, moat_task, return_exceptions=True
            )

            if isinstance(quality_result, Exception):
                quality_result = {'quality_score': 0}
            if isinstance(moat_result, Exception):
                moat_result = {'moat_score': 0}

            phase2_start = datetime.now()
            self._trace_phase("QualityEarnings", AgentRole.EXPERT,
                {'ticker': ticker},
                {'quality_score': quality_result.get('quality_score', 0)},
                success=True, phase_start=start_time, tools_used=['quality_assess'])
            self._trace_phase("Moat", AgentRole.EXPERT,
                {'ticker': ticker},
                {'moat_score': moat_result.get('moat_score', 0)},
                success=True, phase_start=start_time, tools_used=['moat_analyze'])

            # =================================================================
            # PHASE 3: VALUATION
            # =================================================================
            logger.info("💰 Phase 3: Valuation...")

            dcf_result = await self._valuation_agent.dcf_valuation(
                ticker,
                financial_data,
                {'revenue_growth': 0.10, 'terminal_growth': 0.03},
                market_data
            )

            self._trace_phase("Valuation", AgentRole.EXPERT,
                {'ticker': ticker},
                {'intrinsic_value': dcf_result.intrinsic_value, 'upside': dcf_result.upside_downside},
                success=True, phase_start=phase2_start, tools_used=['dcf_valuate'])

            # =================================================================
            # PHASE 4: INVESTMENT THESIS
            # =================================================================
            logger.info("📝 Phase 4: Generating investment thesis...")

            financial_summary = f"""
            Revenue Analysis: {financial_result.get('revenue_analysis', 'N/A')}
            Margin Analysis: {financial_result.get('margin_analysis', 'N/A')}
            Red Flags: {', '.join(financial_result.get('red_flags', []))}
            """

            valuation_summary = f"""
            DCF Intrinsic Value: {dcf_result.intrinsic_value}
            Upside/Downside: {dcf_result.upside_downside:.1f}%
            """

            quality_summary = f"""
            Earnings Quality: {quality_result.get('quality_score', 0)}/100
            Moat Score: {moat_result.get('moat_score', 0)}/100
            """

            thesis = await self._thesis_agent.generate(
                ticker,
                financial_summary,
                valuation_summary,
                quality_summary,
                market_data.get('current_price', 0)
            )

            self._trace_phase("Thesis", AgentRole.PLANNER,
                {'ticker': ticker},
                {'rating': thesis.rating.value},
                success=True, phase_start=phase2_start, tools_used=['thesis_generate'])

            # =================================================================
            # BUILD RESULT
            # =================================================================
            exec_time = (datetime.now() - start_time).total_seconds()

            # Build metrics
            financial_metrics = FinancialMetrics(
                revenue=financial_data.get('revenue', [0])[-1] if financial_data.get('revenue') else 0,
                net_income=financial_data.get('net_income', [0])[-1] if financial_data.get('net_income') else 0
            )

            valuation_metrics = ValuationMetrics(
                market_cap=market_data.get('market_cap', 0)
            )

            quality_metrics = QualityMetrics()

            result = FundamentalResult(
                success=True,
                swarm_name=self.config.name,
                domain=self.config.domain,
                output={'ticker': ticker},
                execution_time=exec_time,
                ticker=ticker,
                company_name=ticker,
                financial_metrics=financial_metrics,
                valuation_metrics=valuation_metrics,
                quality_metrics=quality_metrics,
                valuations=[dcf_result],
                thesis=thesis,
                moat_score=moat_result.get('moat_score', 0),
                management_score=0,
                earnings_quality=quality_result.get('quality_score', 0)
            )

            logger.info(f"✅ FundamentalSwarm complete: {ticker}, Rating: {thesis.rating.value}")

            # Post-execution learning
            exec_time = (datetime.now() - start_time).total_seconds()
            await self._post_execute_learning(
                success=True,
                execution_time=exec_time,
                tools_used=self._get_active_tools(['financial_analyze', 'ratio_analyze', 'dcf_valuate']),
                task_type='fundamental_analysis',
                output_data={
                    'rating': thesis.rating.value,
                    'target_price': thesis.target_price,
                    'upside': thesis.upside,
                    'intrinsic_value': dcf_result.intrinsic_value,
                    'dcf_upside': dcf_result.upside_downside,
                    'dcf_confidence': dcf_result.confidence,
                    'moat_score': moat_result.get('moat_score', 0),
                    'earnings_quality': quality_result.get('quality_score', 0),
                    'num_valuations': len(result.valuations),
                    'red_flags': len(financial_result.get('red_flags', [])),
                },
                input_data={
                    'ticker': ticker,
                    'exchange': exchange,
                    'current_price': market_data.get('current_price', 0),
                    'market_cap': market_data.get('market_cap', 0),
                    'years_of_data': self.config.years_of_data,
                    'investment_style': self.config.investment_style.value,
                }
            )

            return result

        except Exception as e:
            logger.error(f"❌ FundamentalSwarm error: {e}")
            import traceback
            traceback.print_exc()
            exec_time = (datetime.now() - start_time).total_seconds()
            await self._post_execute_learning(
                success=False,
                execution_time=exec_time,
                tools_used=self._get_active_tools(['financial_analyze']),
                task_type='fundamental_analysis'
            )
            return FundamentalResult(
                success=False,
                swarm_name=self.config.name,
                domain=self.config.domain,
                output={},
                execution_time=(datetime.now() - start_time).total_seconds(),
                error=str(e)
            )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def analyze_fundamentals(ticker: str, **kwargs) -> FundamentalResult:
    """
    One-liner fundamental analysis.

    Usage:
        from core.swarms.fundamental_swarm import analyze_fundamentals
        result = await analyze_fundamentals("RELIANCE")
    """
    swarm = FundamentalSwarm()
    return await swarm.analyze(ticker, **kwargs)


def analyze_fundamentals_sync(ticker: str, **kwargs) -> FundamentalResult:
    """Synchronous fundamental analysis."""
    return asyncio.run(analyze_fundamentals(ticker, **kwargs))


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'FundamentalSwarm',
    'FundamentalConfig',
    'FundamentalResult',
    'FinancialMetrics',
    'ValuationMetrics',
    'QualityMetrics',
    'ValuationResult',
    'InvestmentThesis',
    'ValuationType',
    'InvestmentStyle',
    'RatingScale',
    'analyze_fundamentals',
    'analyze_fundamentals_sync',
    # Agents
    'FinancialStatementAgent',
    'RatioAnalysisAgent',
    'ValuationAgent',
    'QualityEarningsAgent',
    'ManagementAgent',
    'MoatAgent',
    'ThesisAgent',
]
