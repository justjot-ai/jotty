"""
PMI Skills Tests
================

Unit tests for all PlanMyInvesting skill packs and financial-analysis.
All HTTP calls are mocked — no real API needed.
"""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, Any

import pytest
from unittest.mock import patch, MagicMock

from Jotty.core.utils.api_client import BaseAPIClient


# =============================================================================
# Module Loading Helpers (hyphenated dirs can't be imported normally)
# =============================================================================

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


def _load_skill_module(skill_dir: str, module_name: str = "tools"):
    """Load a tools.py module from a hyphenated skill directory."""
    path = SKILLS_DIR / skill_dir / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"{skill_dir}_{module_name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Load PlanMyInvestingClient (for client-specific tests)
_client_mod = _load_skill_module("pmi-market-data", "pmi_client")
PlanMyInvestingClient = _client_mod.PlanMyInvestingClient

# Load all skill modules
_market_data = _load_skill_module("pmi-market-data")
_portfolio = _load_skill_module("pmi-portfolio")
_watchlist = _load_skill_module("pmi-watchlist")
_trading = _load_skill_module("pmi-trading")
_strategies = _load_skill_module("pmi-strategies")
_alerts = _load_skill_module("pmi-alerts")
_broker = _load_skill_module("pmi-broker")
_financial = _load_skill_module("financial-analysis")

# Decorator shorthand: patch _make_request on BaseAPIClient (always the canonical class)
_mock_api = patch.object(BaseAPIClient, "_make_request")
_mock_env = patch.dict("os.environ", {"PMI_API_TOKEN": "t", "PMI_API_URL": "http://test"})


# =============================================================================
# TestPlanMyInvestingClient
# =============================================================================

@pytest.mark.unit
class TestPlanMyInvestingClient:
    """Test PlanMyInvestingClient configuration and headers."""

    def test_headers_use_bearer(self):
        """Headers must include Authorization: Bearer."""
        client = PlanMyInvestingClient(token="test-token")
        headers = client._get_headers()
        assert headers["Authorization"] == "Bearer test-token"
        assert headers["Content-Type"] == "application/json"

    @patch.dict("os.environ", {"PMI_API_URL": "https://pmi.example.com", "PMI_API_TOKEN": "env-tok"})
    def test_base_url_from_env(self):
        """BASE_URL loaded from PMI_API_URL env var."""
        client = PlanMyInvestingClient()
        assert client.BASE_URL == "https://pmi.example.com"
        assert client.token == "env-tok"

    def test_base_url_default_localhost(self):
        """Default BASE_URL is localhost:5000."""
        with patch.dict("os.environ", {}, clear=True):
            client = PlanMyInvestingClient(token="t")
            assert "localhost:5000" in client.BASE_URL

    def test_is_configured(self):
        """is_configured True when token present."""
        client = PlanMyInvestingClient(token="test")
        assert client.is_configured is True

    def test_not_configured_without_token(self):
        """is_configured False when no token."""
        with patch.dict("os.environ", {}, clear=True):
            client = PlanMyInvestingClient()
            client.token = None
            assert client.is_configured is False

    def test_require_token_returns_error(self):
        """require_token returns error dict when no token."""
        with patch.dict("os.environ", {}, clear=True):
            client = PlanMyInvestingClient()
            client.token = None
            err = client.require_token()
            assert err is not None
            assert err["success"] is False
            assert "PMI_API_TOKEN" in err["error"]

    def test_convenience_get(self):
        """get() calls _make_request with GET."""
        client = PlanMyInvestingClient(token="t", base_url="http://test")
        with patch.object(client, "_make_request", return_value={"success": True}) as mock:
            result = client.get("/v2/test", params={"a": 1})
            mock.assert_called_once_with("/v2/test", method="GET", params={"a": 1}, timeout=None)
            assert result["success"] is True

    def test_convenience_post(self):
        """post() calls _make_request with POST."""
        client = PlanMyInvestingClient(token="t", base_url="http://test")
        with patch.object(client, "_make_request", return_value={"success": True}) as mock:
            client.post("/v2/test", data={"b": 2})
            mock.assert_called_once_with("/v2/test", method="POST", json_data={"b": 2}, timeout=None)


# =============================================================================
# TestPMIMarketData
# =============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
class TestPMIMarketData:
    """Test pmi-market-data tools with mocked API."""

    @_mock_api
    @_mock_env
    async def test_get_quote_success(self, mock_req):
        mock_req.return_value = {
            "success": True, "ltp": 2450.50, "change": 12.5,
            "change_percent": 0.51, "volume": 1000000,
        }
        result = await _market_data.get_quote_tool({"symbol": "RELIANCE"})
        assert result["success"] is True
        assert result["symbol"] == "RELIANCE"
        assert result["ltp"] == 2450.50

    @_mock_api
    @_mock_env
    async def test_get_quotes_multiple(self, mock_req):
        mock_req.return_value = {
            "success": True,
            "quotes": [{"symbol": "TCS", "ltp": 3500}, {"symbol": "INFY", "ltp": 1600}],
        }
        result = await _market_data.get_quotes_tool({"symbols": ["TCS", "INFY"]})
        assert result["success"] is True
        assert result["count"] == 2

    @_mock_api
    @_mock_env
    async def test_get_quotes_csv_string(self, mock_req):
        mock_req.return_value = {"success": True, "quotes": [{"symbol": "A"}]}
        result = await _market_data.get_quotes_tool({"symbols": "TCS, INFY"})
        assert result["success"] is True

    @_mock_api
    @_mock_env
    async def test_search_symbols(self, mock_req):
        mock_req.return_value = {
            "success": True, "symbols": [{"symbol": "RELIANCE", "name": "Reliance Industries"}],
        }
        result = await _market_data.search_symbols_tool({"query": "reliance"})
        assert result["success"] is True
        assert result["query"] == "reliance"

    @_mock_api
    @_mock_env
    async def test_get_indices(self, mock_req):
        mock_req.return_value = {
            "success": True, "indices": [{"name": "NIFTY 50", "value": 22500}],
        }
        result = await _market_data.get_indices_tool({})
        assert result["success"] is True

    @_mock_api
    @_mock_env
    async def test_get_chart_data(self, mock_req):
        mock_req.return_value = {
            "success": True,
            "candles": [{"o": 100, "h": 110, "l": 95, "c": 105, "v": 50000}],
        }
        result = await _market_data.get_chart_data_tool({"symbol": "TCS", "interval": "1d"})
        assert result["success"] is True
        assert result["symbol"] == "TCS"
        assert result["count"] == 1

    @_mock_api
    @_mock_env
    async def test_get_market_breadth(self, mock_req):
        mock_req.return_value = {
            "success": True, "advances": 800, "declines": 500, "unchanged": 50,
        }
        result = await _market_data.get_market_breadth_tool({})
        assert result["success"] is True
        assert result["advances"] == 800

    @_mock_api
    @_mock_env
    async def test_get_sector_analysis(self, mock_req):
        mock_req.return_value = {
            "success": True, "sectors": [{"name": "IT", "change": 1.5}],
        }
        result = await _market_data.get_sector_analysis_tool({"period": "1d"})
        assert result["success"] is True
        assert result["period"] == "1d"

    async def test_get_quote_missing_symbol(self):
        """Required param validation."""
        result = await _market_data.get_quote_tool({})
        assert result["success"] is False
        assert "symbol" in result["error"].lower()


# =============================================================================
# TestPMIPortfolio
# =============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
class TestPMIPortfolio:
    """Test pmi-portfolio tools."""

    @_mock_api
    @_mock_env
    async def test_get_portfolio(self, mock_req):
        mock_req.return_value = {
            "success": True,
            "holdings": [{"symbol": "TCS", "qty": 10}],
            "total_value": 35000,
            "total_pnl": 500,
        }
        result = await _portfolio.get_portfolio_tool({})
        assert result["success"] is True
        assert result["count"] == 1
        assert result["total_value"] == 35000

    @_mock_api
    @_mock_env
    async def test_get_pnl_summary(self, mock_req):
        mock_req.return_value = {
            "success": True, "realized_pnl": 1000, "unrealized_pnl": 500,
            "total_pnl": 1500, "day_pnl": 200,
        }
        result = await _portfolio.get_pnl_summary_tool({})
        assert result["success"] is True
        assert result["total_pnl"] == 1500

    @_mock_api
    @_mock_env
    async def test_get_available_cash(self, mock_req):
        mock_req.return_value = {"success": True, "total": 50000}
        result = await _portfolio.get_available_cash_tool({})
        assert result["success"] is True
        assert result["total"] == 50000

    @_mock_api
    @_mock_env
    async def test_get_account_limits(self, mock_req):
        mock_req.return_value = {
            "success": True, "margin_available": 100000,
            "margin_used": 50000, "collateral": 75000,
        }
        result = await _portfolio.get_account_limits_tool({})
        assert result["success"] is True
        assert result["margin_available"] == 100000


# =============================================================================
# TestPMIWatchlist
# =============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
class TestPMIWatchlist:
    """Test pmi-watchlist tools."""

    @_mock_api
    @_mock_env
    async def test_list_watchlists(self, mock_req):
        mock_req.return_value = {
            "success": True, "watchlists": [{"id": "w1", "name": "Tech"}],
        }
        result = await _watchlist.list_watchlists_tool({})
        assert result["success"] is True
        assert result["count"] == 1

    @_mock_api
    @_mock_env
    async def test_create_watchlist(self, mock_req):
        mock_req.return_value = {"success": True, "id": "w2"}
        result = await _watchlist.create_watchlist_tool({"name": "Banks"})
        assert result["success"] is True
        assert result["name"] == "Banks"

    @_mock_api
    @_mock_env
    async def test_add_to_watchlist(self, mock_req):
        mock_req.return_value = {"success": True}
        result = await _watchlist.add_to_watchlist_tool({
            "watchlist_id": "w1", "symbol": "HDFC",
        })
        assert result["success"] is True
        assert result["symbol"] == "HDFC"

    @_mock_api
    @_mock_env
    async def test_remove_from_watchlist(self, mock_req):
        mock_req.return_value = {"success": True}
        result = await _watchlist.remove_from_watchlist_tool({
            "watchlist_id": "w1", "symbol": "HDFC",
        })
        assert result["success"] is True
        assert result["removed"] is True

    @_mock_api
    @_mock_env
    async def test_refresh_watchlist(self, mock_req):
        mock_req.return_value = {
            "success": True, "symbols": [{"symbol": "TCS", "ltp": 3500}],
        }
        result = await _watchlist.refresh_watchlist_tool({"watchlist_id": "w1"})
        assert result["success"] is True
        assert result["count"] == 1


# =============================================================================
# TestPMITrading
# =============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
class TestPMITrading:
    """Test pmi-trading tools."""

    @_mock_api
    @_mock_env
    async def test_place_order(self, mock_req):
        mock_req.return_value = {"success": True, "order_id": "ORD001", "status": "placed"}
        result = await _trading.place_order_tool({
            "symbol": "RELIANCE", "quantity": 10,
            "order_type": "MARKET", "transaction_type": "BUY",
        })
        assert result["success"] is True
        assert result["order_id"] == "ORD001"

    @_mock_api
    @_mock_env
    async def test_place_smart_order(self, mock_req):
        mock_req.return_value = {
            "success": True, "order_id": "ORD002",
            "entry_price": 2450, "target": 2499, "stoploss": 2425.5,
        }
        result = await _trading.place_smart_order_tool({
            "symbol": "RELIANCE", "quantity": 5,
        })
        assert result["success"] is True
        assert result["entry_price"] == 2450

    @_mock_api
    @_mock_env
    async def test_exit_position(self, mock_req):
        mock_req.return_value = {
            "success": True, "order_id": "ORD003",
            "quantity_exited": 10, "exit_price": 2500,
        }
        result = await _trading.exit_position_tool({"symbol": "RELIANCE"})
        assert result["success"] is True
        assert result["quantity_exited"] == 10

    @_mock_api
    @_mock_env
    async def test_cancel_order(self, mock_req):
        mock_req.return_value = {"success": True}
        result = await _trading.cancel_order_tool({"order_id": "ORD001"})
        assert result["success"] is True
        assert result["cancelled"] is True

    @_mock_api
    @_mock_env
    async def test_get_orders(self, mock_req):
        mock_req.return_value = {
            "success": True, "orders": [{"order_id": "ORD001", "status": "completed"}],
        }
        result = await _trading.get_orders_tool({})
        assert result["success"] is True
        assert result["count"] == 1


# =============================================================================
# TestPMIStrategies
# =============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
class TestPMIStrategies:
    """Test pmi-strategies tools."""

    @_mock_api
    @_mock_env
    async def test_list_strategies(self, mock_req):
        mock_req.return_value = {
            "success": True,
            "strategies": [{"id": "s1", "name": "Momentum"}],
        }
        result = await _strategies.list_strategies_tool({})
        assert result["success"] is True
        assert result["count"] == 1

    @_mock_api
    @_mock_env
    async def test_run_strategy(self, mock_req):
        mock_req.return_value = {
            "success": True, "execution_id": "ex1",
            "signals": [{"symbol": "TCS", "action": "BUY"}],
            "orders_placed": 0,
        }
        result = await _strategies.run_strategy_tool({
            "strategy_id": "s1", "dry_run": True,
        })
        assert result["success"] is True
        assert result["dry_run"] is True

    @_mock_api
    @_mock_env
    async def test_get_strategy_status(self, mock_req):
        mock_req.return_value = {
            "success": True, "name": "Momentum", "active": True,
            "win_rate": 0.65, "total_trades": 42,
        }
        result = await _strategies.get_strategy_status_tool({"strategy_id": "s1"})
        assert result["success"] is True
        assert result["win_rate"] == 0.65

    @_mock_api
    @_mock_env
    async def test_generate_signals(self, mock_req):
        mock_req.return_value = {
            "success": True, "signals": [{"symbol": "INFY", "action": "SELL"}],
        }
        result = await _strategies.generate_signals_tool({})
        assert result["success"] is True
        assert result["count"] == 1


# =============================================================================
# TestPMIAlerts
# =============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
class TestPMIAlerts:
    """Test pmi-alerts tools."""

    @_mock_api
    @_mock_env
    async def test_list_alerts(self, mock_req):
        mock_req.return_value = {
            "success": True, "alerts": [{"id": "a1", "symbol": "TCS"}],
        }
        result = await _alerts.list_alerts_tool({})
        assert result["success"] is True
        assert result["count"] == 1

    @_mock_api
    @_mock_env
    async def test_create_alert(self, mock_req):
        mock_req.return_value = {"success": True, "alert_id": "a2"}
        result = await _alerts.create_alert_tool({
            "symbol": "RELIANCE", "condition": "above", "value": 2500,
        })
        assert result["success"] is True
        assert result["alert_id"] == "a2"

    @_mock_api
    @_mock_env
    async def test_delete_alert(self, mock_req):
        mock_req.return_value = {"success": True}
        result = await _alerts.delete_alert_tool({"alert_id": "a1"})
        assert result["success"] is True
        assert result["deleted"] is True

    @_mock_api
    @_mock_env
    async def test_get_alert_stats(self, mock_req):
        mock_req.return_value = {
            "success": True, "total": 10, "active": 7, "triggered": 3,
        }
        result = await _alerts.get_alert_stats_tool({})
        assert result["success"] is True
        assert result["active"] == 7


# =============================================================================
# TestPMIBroker
# =============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
class TestPMIBroker:
    """Test pmi-broker tools."""

    @_mock_api
    @_mock_env
    async def test_list_brokers(self, mock_req):
        mock_req.return_value = {
            "success": True, "brokers": [{"name": "zerodha", "connected": True}],
        }
        result = await _broker.list_brokers_tool({})
        assert result["success"] is True
        assert result["count"] == 1

    @_mock_api
    @_mock_env
    async def test_get_broker_status(self, mock_req):
        mock_req.return_value = {
            "success": True, "connected": True, "token_valid": True,
            "last_login": "2026-02-14T09:00:00Z",
        }
        result = await _broker.get_broker_status_tool({"broker": "zerodha"})
        assert result["success"] is True
        assert result["token_valid"] is True

    @_mock_api
    @_mock_env
    async def test_refresh_tokens(self, mock_req):
        mock_req.return_value = {"success": True, "token_valid": True}
        result = await _broker.refresh_tokens_tool({"broker": "zerodha"})
        assert result["success"] is True
        assert result["refreshed"] is True


# =============================================================================
# TestFinancialAnalysis
# =============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
class TestFinancialAnalysis:
    """Test financial-analysis LLM tools with mocked API and LLM."""

    @_mock_api
    @_mock_env
    async def test_sentiment_analysis(self, mock_req):
        mock_req.side_effect = [
            {"success": True, "ltp": 2450, "change": 12, "change_percent": 0.5, "volume": 1000000},
            {"success": True, "candles": [{"o": 2440, "h": 2460, "l": 2430, "c": 2450}]},
        ]
        with patch.object(_financial, "_call_lm", return_value=json.dumps({
            "sentiment": "BULLISH", "confidence": "MEDIUM",
            "signals": ["Price above SMA", "Volume increasing"],
            "narrative": "Stock shows bullish momentum.",
        })):
            result = await _financial.sentiment_analysis_tool({"symbol": "RELIANCE"})
        assert result["success"] is True
        assert result["sentiment"] == "BULLISH"
        assert result["confidence"] == "MEDIUM"
        assert len(result["signals"]) == 2

    @_mock_api
    @_mock_env
    async def test_earnings_analysis(self, mock_req):
        mock_req.side_effect = [
            {"success": True, "data": {"revenue": [100, 110, 120], "profit": [10, 12, 15]}},
            {"success": True, "ltp": 3500, "change_percent": 1.2},
        ]
        with patch.object(_financial, "_call_lm", return_value=json.dumps({
            "revenue_trend": "GROWING", "profit_trend": "GROWING",
            "highlights": ["Consistent revenue growth"],
            "quality": "High quality earnings",
            "outlook": "Strong forward outlook.",
        })):
            result = await _financial.earnings_analysis_tool({"symbol": "TCS"})
        assert result["success"] is True
        assert result["revenue_trend"] == "GROWING"

    @_mock_api
    @_mock_env
    async def test_stock_comparison(self, mock_req):
        mock_req.side_effect = [
            {"success": True, "ltp": 3500, "change": 50},
            {"success": True, "ltp": 1600, "change": -10},
        ]
        with patch.object(_financial, "_call_lm", return_value=json.dumps({
            "ranking": [
                {"symbol": "TCS", "rank": 1, "reason": "Better growth"},
                {"symbol": "INFY", "rank": 2, "reason": "Lower valuation"},
            ],
            "differentiators": ["TCS has higher revenue growth"],
            "narrative": "TCS leads in growth metrics.",
            "risk_comparison": "Both have moderate risk profiles.",
        })):
            result = await _financial.stock_comparison_tool({
                "symbols": ["TCS", "INFY"],
            })
        assert result["success"] is True
        assert len(result["ranking"]) == 2
        assert result["symbols"] == ["TCS", "INFY"]

    async def test_stock_comparison_needs_two_symbols(self):
        """Should error with fewer than 2 symbols."""
        with patch.dict("os.environ", {"PMI_API_TOKEN": "t", "PMI_API_URL": "http://test"}):
            result = await _financial.stock_comparison_tool({"symbols": ["TCS"]})
        assert result["success"] is False
        assert "2 symbols" in result["error"]
