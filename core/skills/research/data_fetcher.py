"""
Research Data Fetcher
=====================

Fetches live financial data from multiple sources:
- Screener.in - Indian company financials
- Yahoo Finance - Global stock data
- NSE/BSE - Indian market data
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)


class ResearchDataFetcher:
    """Fetch research data from multiple sources including web search."""

    def __init__(self) -> None:
        self._screener_available = False
        self._yfinance_available = False
        self._web_search_available = False
        self._web_search_tool = None
        self._init_sources()

    def _init_sources(self) -> Any:
        """Initialize data sources including web search."""
        try:
            # Check if screener skill is available
            from Jotty.skills import get_skill
            self._screener_skill = get_skill('screener-financials')
            self._screener_available = self._screener_skill is not None
        except Exception:
            self._screener_available = False

        try:
            import yfinance as yf
            self._yfinance_available = True
        except ImportError:
            self._yfinance_available = False
            logger.info("yfinance not available - install with: pip install yfinance")

        # Initialize web search for real news/data
        try:
            from Jotty.core.registry.skills_registry import get_skills_registry
            registry = get_skills_registry()
            registry.init()
            web_skill = registry.get_skill('web-search')
            if web_skill:
                self._web_search_tool = web_skill.tools.get('search_web_tool')
                self._web_search_available = self._web_search_tool is not None
                logger.info(" Web search enabled for real-time news")
        except Exception as e:
            logger.debug(f"Web search init: {e}")
            self._web_search_available = False

    async def fetch_company_data(self, ticker: str, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Fetch comprehensive company data from all available sources.

        Args:
            ticker: Stock ticker symbol
            exchange: Exchange (NSE, BSE, etc.)

        Returns:
            Dictionary with all fetched data
        """
        data = {
            "ticker": ticker,
            "exchange": exchange,
            "fetch_time": datetime.now().isoformat(),
            "sources": [],
        }

        # Fetch from multiple sources in parallel
        tasks = []

        if self._yfinance_available:
            tasks.append(self._fetch_yahoo_data(ticker, exchange))

        if self._screener_available:
            tasks.append(self._fetch_screener_data(ticker))

        # Always fetch web search for real news (critical for accurate data)
        if self._web_search_available:
            tasks.append(self._fetch_web_search_data(ticker, exchange))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results
        api_data_found = False
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Data fetch error: {result}")
                continue
            if isinstance(result, dict):
                source = result.pop("_source", "unknown")
                data["sources"].append(source)
                # Track if we got real API data
                if source in ['yahoo_finance', 'screener'] and result.get('current_price'):
                    api_data_found = True
                data.update(result)

        # If no API data found, web search news becomes primary source
        if not api_data_found and data.get('web_search_news'):
            logger.info(" Using web search as primary data source (API data unavailable)")
            data['data_source'] = 'web_search'

        return data

    async def _fetch_yahoo_data(self, ticker: str, exchange: str = "NSE") -> Dict[str, Any]:
        """Fetch data from Yahoo Finance."""
        import yfinance as yf

        # Auto-detect exchange from ticker pattern
        # Common US stock tickers (no suffix needed)
        US_TICKERS = {
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA',
            'BRK.A', 'BRK.B', 'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH',
            'HD', 'DIS', 'PYPL', 'BAC', 'ADBE', 'NFLX', 'CRM', 'INTC', 'AMD',
            'CSCO', 'PEP', 'KO', 'TMO', 'ABT', 'COST', 'AVGO', 'NKE', 'MRK',
            'ORCL', 'ACN', 'MCD', 'LLY', 'DHR', 'TXN', 'QCOM', 'UPS', 'NEE',
            'IBM', 'GE', 'CAT', 'BA', 'RTX', 'GS', 'MS', 'BLK', 'SCHW', 'AXP'
        }

        ticker_upper = ticker.upper().strip()

        # Common Indian stock tickers (to avoid misclassifying as US)
        INDIAN_TICKERS = {
            'PAYTM', 'RELIANCE', 'TCS', 'INFY', 'HDFC', 'HDFCBANK', 'ICICIBANK',
            'SBIN', 'BAJFINANCE', 'BHARTIARTL', 'ITC', 'KOTAKBANK', 'LT', 'AXISBANK',
            'WIPRO', 'HCLTECH', 'MARUTI', 'TITAN', 'ULTRACEMCO', 'TATASTEEL',
            'TECHM', 'SUNPHARMA', 'ONGC', 'NTPC', 'POWERGRID', 'ASIANPAINT',
            'ZOMATO', 'DMART', 'ADANIENT', 'JSWSTEEL', 'TATAMOTORS', 'HINDALCO',
            'NYKAA', 'POLICYBAZAAR', 'PHONEPE', 'RAZORPAY', 'ZERODHA', 'CRED'
        }

        # Check if it's a known US ticker FIRST (takes priority over default exchange)
        is_us_ticker = (
            ticker_upper in US_TICKERS or
            exchange.upper() in ('US', 'NYSE', 'NASDAQ', 'AMEX')
        )
        # Only treat as Indian if explicitly Indian AND not a known US ticker
        is_indian_ticker = (
            not is_us_ticker and (
                ticker_upper in INDIAN_TICKERS or
                exchange.upper() in ('NSE', 'BSE', 'INDIA')
            )
        )

        # Format ticker for Yahoo Finance
        if is_us_ticker:
            yf_ticker = ticker_upper  # No suffix for US stocks
        elif exchange.upper() == "NSE":
            yf_ticker = f"{ticker}.NS"
        elif exchange.upper() == "BSE":
            yf_ticker = f"{ticker}.BO"
        else:
            yf_ticker = ticker

        try:
            stock = yf.Ticker(yf_ticker)
            info = stock.info

            # Get price data
            hist = stock.history(period="1y")

            data = {
                "_source": "yahoo_finance",
                "company_name": info.get("longName", ticker),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),

                # Price data
                "current_price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
                "previous_close": info.get("previousClose", 0),
                "open": info.get("open", 0),
                "day_high": info.get("dayHigh", 0),
                "day_low": info.get("dayLow", 0),
                "week_52_high": info.get("fiftyTwoWeekHigh", 0),
                "week_52_low": info.get("fiftyTwoWeekLow", 0),
                "volume": info.get("volume", 0),
                "avg_volume": info.get("averageVolume", 0),

                # Valuation
                "market_cap": info.get("marketCap", 0),
                "enterprise_value": info.get("enterpriseValue", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "forward_pe": info.get("forwardPE", 0),
                "pb_ratio": info.get("priceToBook", 0),
                "ps_ratio": info.get("priceToSalesTrailing12Months", 0),
                "ev_ebitda": info.get("enterpriseToEbitda", 0),
                "ev_revenue": info.get("enterpriseToRevenue", 0),

                # Fundamentals
                "revenue": info.get("totalRevenue", 0),
                "revenue_growth": info.get("revenueGrowth", 0) * 100 if info.get("revenueGrowth") else 0,
                "gross_margin": info.get("grossMargins", 0) * 100 if info.get("grossMargins") else 0,
                "ebitda": info.get("ebitda", 0),
                "ebitda_margin": info.get("ebitdaMargins", 0) * 100 if info.get("ebitdaMargins") else 0,
                "operating_margin": info.get("operatingMargins", 0) * 100 if info.get("operatingMargins") else 0,
                "profit_margin": info.get("profitMargins", 0) * 100 if info.get("profitMargins") else 0,
                "net_income": info.get("netIncomeToCommon", 0),

                # Per share
                "eps": info.get("trailingEps", 0),
                "forward_eps": info.get("forwardEps", 0),
                "book_value": info.get("bookValue", 0),

                # Returns
                "roe": info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else 0,
                "roa": info.get("returnOnAssets", 0) * 100 if info.get("returnOnAssets") else 0,

                # Dividend
                "dividend_rate": info.get("dividendRate", 0),
                "dividend_yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0,
                "payout_ratio": info.get("payoutRatio", 0) * 100 if info.get("payoutRatio") else 0,

                # Balance Sheet
                "total_cash": info.get("totalCash", 0),
                "total_debt": info.get("totalDebt", 0),
                "debt_to_equity": info.get("debtToEquity", 0),
                "current_ratio": info.get("currentRatio", 0),
                "quick_ratio": info.get("quickRatio", 0),

                # Other
                "beta": info.get("beta", 1.0),
                "shares_outstanding": info.get("sharesOutstanding", 0),
                "float_shares": info.get("floatShares", 0),

                # Analyst data
                "target_mean_price": info.get("targetMeanPrice", 0),
                "target_high_price": info.get("targetHighPrice", 0),
                "target_low_price": info.get("targetLowPrice", 0),
                "recommendation": info.get("recommendationKey", ""),
                "num_analysts": info.get("numberOfAnalystOpinions", 0),

                # Historical prices (last 252 trading days)
                "price_history": hist["Close"].tolist() if not hist.empty else [],
                "volume_history": hist["Volume"].tolist() if not hist.empty else [],
                "dates": [d.strftime("%Y-%m-%d") for d in hist.index] if not hist.empty else [],
            }

            # Calculate additional metrics
            if data["current_price"] and data["week_52_low"]:
                data["price_vs_52w_low"] = ((data["current_price"] - data["week_52_low"]) / data["week_52_low"]) * 100
            if data["current_price"] and data["week_52_high"]:
                data["price_vs_52w_high"] = ((data["current_price"] - data["week_52_high"]) / data["week_52_high"]) * 100

            return data

        except Exception as e:
            logger.error(f"Yahoo Finance error for {ticker}: {e}")
            return {"_source": "yahoo_finance", "error": str(e)}

    async def _fetch_web_search_data(self, ticker: str, exchange: str = "NSE") -> Dict[str, Any]:
        """Fetch real-time news and data via web search (DuckDuckGo)."""
        import inspect

        if not self._web_search_tool:
            return {"_source": "web_search", "error": "Web search not available"}

        try:
            logger.info(f" Searching web for {ticker} news and data...")

            # Run multiple search queries for comprehensive coverage
            search_queries = [
                f"{ticker} stock latest news 2024 2025",
                f"{ticker} quarterly results earnings profit",
                f"{ticker} stock price target analyst rating",
                f"{ticker} {exchange} financial performance revenue",
            ]

            all_results = []
            for query in search_queries:
                try:
                    if inspect.iscoroutinefunction(self._web_search_tool):
                        result = await self._web_search_tool({'query': query, 'max_results': 8})
                    else:
                        result = self._web_search_tool({'query': query, 'max_results': 8})

                    if result.get('success') and result.get('results'):
                        all_results.extend(result['results'])
                except Exception as e:
                    logger.debug(f"Search query failed: {e}")

            if not all_results:
                return {"_source": "web_search", "error": "No results found"}

            # Deduplicate by URL
            seen_urls = set()
            unique_results = []
            for r in all_results:
                url = r.get('url', '')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_results.append(r)

            # Extract key data from search results
            news_items = []
            for r in unique_results[:20]:  # Top 20 results
                news_items.append({
                    'title': r.get('title', ''),
                    'snippet': r.get('snippet', '')[:400],
                    'url': r.get('url', ''),
                    'source': r.get('source', '')
                })

            # Compile comprehensive news summary
            news_text = "\n".join([
                f"• {n['title']}: {n['snippet']}"
                for n in news_items[:15]
            ])

            logger.info(f" Found {len(unique_results)} web search results for {ticker}")

            return {
                "_source": "web_search",
                "web_search_news": news_text,
                "web_search_results": news_items,
                "web_search_count": len(unique_results),
            }

        except Exception as e:
            logger.error(f"Web search error for {ticker}: {e}")
            return {"_source": "web_search", "error": str(e)}

    async def _fetch_screener_data(self, ticker: str) -> Dict[str, Any]:
        """Fetch data from Screener.in."""
        try:
            # Get financials from screener skill
            financials_tool = self._screener_skill.tools.get('get_company_financials_tool')
            if not financials_tool:
                return {"_source": "screener", "error": "Tool not found"}

            result = financials_tool({
                'company_name': ticker,
                'data_type': 'all',
                'period': 'annual',
                'format': 'json'
            })

            if not result.get('success'):
                return {"_source": "screener", "error": result.get('error', 'Unknown error')}

            # Parse screener data
            data = {
                "_source": "screener",
                "screener_company_name": result.get('company_name', ticker),
                "screener_code": result.get('company_code', ticker),
            }

            # Extract P&L data
            pl_data = result.get('data', {}).get('profit_loss', {})
            if pl_data.get('headers') and pl_data.get('rows'):
                data['pl_headers'] = pl_data['headers']
                data['pl_rows'] = pl_data['rows']

                # Parse into structured format
                years = pl_data['headers'][1:]  # Skip first column (metric name)
                data['financial_years'] = years

                for row in pl_data['rows']:
                    if len(row) > 1:
                        metric = row[0].lower().strip()
                        values = self._parse_numeric_values(row[1:])

                        if 'sales' in metric or 'revenue' in metric:
                            data['screener_revenue'] = values
                        elif 'operating profit' in metric:
                            data['screener_operating_profit'] = values
                        elif 'net profit' in metric or 'profit after tax' in metric:
                            data['screener_pat'] = values
                        elif 'eps' in metric:
                            data['screener_eps'] = values

            # Extract ratios
            ratios = result.get('data', {}).get('ratios', {})
            data['screener_ratios'] = ratios

            # Parse key ratios
            for key, value in ratios.items():
                key_lower = key.lower()
                parsed_value = self._parse_ratio_value(value)

                if 'roe' in key_lower:
                    data['screener_roe'] = parsed_value
                elif 'roce' in key_lower:
                    data['screener_roce'] = parsed_value
                elif 'debt' in key_lower and 'equity' in key_lower:
                    data['screener_debt_equity'] = parsed_value
                elif 'current ratio' in key_lower:
                    data['screener_current_ratio'] = parsed_value
                elif 'promoter' in key_lower and 'hold' in key_lower:
                    data['promoter_holding'] = parsed_value
                elif 'fii' in key_lower or 'foreign' in key_lower:
                    data['fii_holding'] = parsed_value
                elif 'dii' in key_lower or 'domestic' in key_lower:
                    data['dii_holding'] = parsed_value

            return data

        except Exception as e:
            logger.error(f"Screener.in error for {ticker}: {e}")
            return {"_source": "screener", "error": str(e)}

    def _parse_numeric_values(self, values: List[str]) -> List[float]:
        """Parse list of string values to floats."""
        result = []
        for v in values:
            try:
                # Remove commas and handle Cr/L suffixes
                v_clean = str(v).replace(',', '').strip()
                if 'cr' in v_clean.lower():
                    v_clean = v_clean.lower().replace('cr', '').strip()
                    result.append(float(v_clean) * 1e7)
                elif 'l' in v_clean.lower() or 'lakh' in v_clean.lower():
                    v_clean = re.sub(r'[lL](?:akh)?', '', v_clean).strip()
                    result.append(float(v_clean) * 1e5)
                else:
                    result.append(float(v_clean) if v_clean and v_clean != '-' else 0.0)
            except (ValueError, TypeError):
                result.append(0.0)
        return result

    def _parse_ratio_value(self, value: str) -> float:
        """Parse ratio string to float."""
        try:
            v_clean = str(value).replace(',', '').replace('%', '').strip()
            return float(v_clean) if v_clean and v_clean != '-' else 0.0
        except (ValueError, TypeError):
            return 0.0

    async def fetch_peer_data(self, ticker: str, peers: List[str], exchange: str = "NSE") -> Dict[str, List[Any]]:
        """Fetch data for peer comparison."""
        all_companies = [ticker] + peers
        peer_data = {
            "companies": [],
            "market_caps": [],
            "pe_ratios": [],
            "pb_ratios": [],
            "ev_ebitda": [],
            "roe": [],
            "roce": [],
            "revenue_growth": [],
            "pat_margin": [],
        }

        for company in all_companies:
            try:
                data = await self.fetch_company_data(company, exchange)

                peer_data["companies"].append(company)
                peer_data["market_caps"].append(data.get("market_cap", 0) / 1e7)  # Convert to Cr
                peer_data["pe_ratios"].append(data.get("pe_ratio", 0))
                peer_data["pb_ratios"].append(data.get("pb_ratio", 0))
                peer_data["ev_ebitda"].append(data.get("ev_ebitda", 0))
                peer_data["roe"].append(data.get("roe", 0))
                peer_data["roce"].append(data.get("screener_roce", data.get("roe", 0)))
                peer_data["revenue_growth"].append(data.get("revenue_growth", 0))
                peer_data["pat_margin"].append(data.get("profit_margin", 0))

            except Exception as e:
                logger.warning(f"Failed to fetch peer data for {company}: {e}")

        return peer_data

    async def get_analyst_ratings(self, ticker: str, exchange: str = "NSE") -> Dict[str, Any]:
        """Get analyst ratings and target prices."""
        data = await self.fetch_company_data(ticker, exchange)

        return {
            "current_price": data.get("current_price", 0),
            "target_mean": data.get("target_mean_price", 0),
            "target_high": data.get("target_high_price", 0),
            "target_low": data.get("target_low_price", 0),
            "recommendation": data.get("recommendation", ""),
            "num_analysts": data.get("num_analysts", 0),
            "upside": ((data.get("target_mean_price", 0) - data.get("current_price", 1)) /
                       data.get("current_price", 1) * 100) if data.get("current_price") else 0,
        }


class FinancialDataConverter:
    """Convert raw data to report data classes."""

    @staticmethod
    def to_company_snapshot(data: Dict[str, Any], target_price: float = None,
                             rating: str = None) -> 'CompanySnapshot':
        """Convert fetched data to CompanySnapshot."""
        from .report_components import CompanySnapshot

        current_price = data.get("current_price", 0)
        if target_price is None:
            target_price = data.get("target_mean_price", current_price * 1.15)

        if rating is None:
            upside = ((target_price - current_price) / current_price * 100) if current_price else 0
            if upside > 15:
                rating = "BUY"
            elif upside < -10:
                rating = "SELL"
            else:
                rating = "HOLD"

        market_cap = data.get("market_cap", 0)
        market_cap_cr = market_cap / 1e7 if market_cap > 1e7 else market_cap

        return CompanySnapshot(
            ticker=data.get("ticker", ""),
            company_name=data.get("company_name", data.get("ticker", "")),
            current_price=current_price,
            target_price=target_price,
            rating=rating,
            market_cap=market_cap_cr,
            market_cap_unit="Cr",
            pe_ratio=data.get("pe_ratio", 0) or 0,
            pe_forward=data.get("forward_pe", 0) or 0,
            pb_ratio=data.get("pb_ratio", 0) or 0,
            ev_ebitda=data.get("ev_ebitda", 0) or 0,
            roe=data.get("roe", 0) or 0,
            roce=data.get("screener_roce", data.get("roe", 0)) or 0,
            dividend_yield=data.get("dividend_yield", 0) or 0,
            week_52_high=data.get("week_52_high", 0) or 0,
            week_52_low=data.get("week_52_low", 0) or 0,
            beta=data.get("beta", 1.0) or 1.0,
            sector=data.get("sector", ""),
            industry=data.get("industry", ""),
            promoter_holding=data.get("promoter_holding", 0) or 0,
            fii_holding=data.get("fii_holding", 0) or 0,
            dii_holding=data.get("dii_holding", 0) or 0,
        )

    @staticmethod
    def to_financial_statements(data: Dict[str, Any]) -> 'FinancialStatements':
        """Convert fetched data to FinancialStatements."""
        from .report_components import FinancialStatements

        fs = FinancialStatements()

        # Use screener data if available
        if 'financial_years' in data:
            fs.years = data['financial_years']

        if 'screener_revenue' in data:
            fs.revenue = data['screener_revenue']

        if 'screener_pat' in data:
            fs.pat = data['screener_pat']

        if 'screener_eps' in data:
            fs.eps = data['screener_eps']

        # Calculate margins if we have revenue and profit
        if fs.revenue and fs.pat:
            fs.pat_margin = [
                (p / r * 100) if r > 0 else 0
                for p, r in zip(fs.pat, fs.revenue)
            ]

        return fs

    @staticmethod
    def to_peer_comparison(peer_data: Dict[str, List[Any]]) -> 'PeerComparison':
        """Convert peer data to PeerComparison."""
        from .report_components import PeerComparison

        return PeerComparison(
            companies=peer_data.get("companies", []),
            market_caps=peer_data.get("market_caps", []),
            pe_ratios=peer_data.get("pe_ratios", []),
            pb_ratios=peer_data.get("pb_ratios", []),
            ev_ebitda=peer_data.get("ev_ebitda", []),
            roe=peer_data.get("roe", []),
            roce=peer_data.get("roce", []),
            revenue_growth=peer_data.get("revenue_growth", []),
            pat_margin=peer_data.get("pat_margin", []),
        )
