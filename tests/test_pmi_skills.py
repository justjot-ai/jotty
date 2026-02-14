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


# =============================================================================
# TestToolSchemaOutputs — verify output field parsing from docstrings
# =============================================================================

from Jotty.core.agents._execution_types import ToolSchema, ToolParam


@pytest.mark.unit
class TestToolSchemaOutputs:
    """Verify ToolSchema correctly extracts output fields from Returns: docstrings."""

    def test_inline_returns_parsing(self):
        """Parse 'Dictionary with field_a, field_b' inline format."""
        doc = """Get data.\n\nReturns:\n    Dictionary with holdings, total_value, total_pnl"""
        result = ToolSchema._parse_docstring_returns(doc)
        assert "holdings" in result
        assert "total_value" in result
        assert "total_pnl" in result

    def test_structured_returns_parsing(self):
        """Parse structured '- field (type): description' format."""
        doc = """Get data.\n\nReturns:\n    Dictionary with:\n        - holdings (list): Portfolio holdings\n        - total_value (float): Total value\n        - count (int): Number of items"""
        result = ToolSchema._parse_docstring_returns(doc)
        assert result["holdings"]["type"] == "list"
        assert result["total_value"]["type"] == "float"
        assert result["count"]["type"] == "int"
        assert result["holdings"]["description"] == "Portfolio holdings"

    def test_returns_stops_at_next_section(self):
        """Returns parsing should stop at Raises: or other sections."""
        doc = """Get data.\n\nReturns:\n    Dictionary with:\n        - value (float): A value\n\nRaises:\n    ValueError: If bad"""
        result = ToolSchema._parse_docstring_returns(doc)
        assert "value" in result
        assert len(result) == 1

    def test_empty_docstring_returns_empty(self):
        """No Returns section should return empty dict."""
        doc = """Get data.\n\nArgs:\n    params: Dictionary containing:\n        - x (str): Something"""
        result = ToolSchema._parse_docstring_returns(doc)
        assert result == {}

    def test_from_tool_function_captures_outputs(self):
        """ToolSchema.from_tool_function should populate outputs from Returns docstring."""
        schema = ToolSchema.from_tool_function(_portfolio.get_portfolio_tool, "get_portfolio_tool")
        output_names = [o.name for o in schema.outputs]
        assert "holdings" in output_names
        assert "total_value" in output_names
        assert "total_pnl" in output_names
        assert "count" in output_names

    def test_to_dict_includes_returns(self):
        """ToolSchema.to_dict() should include 'returns' field when outputs declared."""
        schema = ToolSchema.from_tool_function(_portfolio.get_portfolio_tool, "get_portfolio_tool")
        d = schema.to_dict()
        assert "returns" in d
        return_names = [r["name"] for r in d["returns"]]
        assert "holdings" in return_names
        assert "total_value" in return_names

    def test_to_dict_no_returns_when_empty(self):
        """ToolSchema.to_dict() should omit 'returns' when no outputs declared."""
        schema = ToolSchema(name="test_tool", description="Test")
        d = schema.to_dict()
        assert "returns" not in d

    def test_pnl_summary_outputs(self):
        """PnL summary tool should declare realized_pnl, unrealized_pnl, total_pnl, day_pnl outputs."""
        schema = ToolSchema.from_tool_function(_portfolio.get_pnl_summary_tool, "get_pnl_summary_tool")
        output_names = [o.name for o in schema.outputs]
        assert "realized_pnl" in output_names
        assert "unrealized_pnl" in output_names
        assert "total_pnl" in output_names
        assert "day_pnl" in output_names

    def test_market_data_quote_outputs(self):
        """Quote tool should declare ltp, change, change_percent, volume outputs."""
        schema = ToolSchema.from_tool_function(_market_data.get_quote_tool, "get_quote_tool")
        output_names = [o.name for o in schema.outputs]
        assert "ltp" in output_names
        assert "change" in output_names
        assert "volume" in output_names

    def test_trading_place_order_outputs(self):
        """Place order tool should declare order_id, status outputs."""
        schema = ToolSchema.from_tool_function(_trading.place_order_tool, "place_order_tool")
        output_names = [o.name for o in schema.outputs]
        assert "order_id" in output_names
        assert "status" in output_names

    def test_output_types_preserved(self):
        """Output field types from structured Returns should be preserved."""
        schema = ToolSchema.from_tool_function(_portfolio.get_portfolio_tool, "get_portfolio_tool")
        holdings_out = next(o for o in schema.outputs if o.name == "holdings")
        total_value_out = next(o for o in schema.outputs if o.name == "total_value")
        count_out = next(o for o in schema.outputs if o.name == "count")
        assert holdings_out.type_hint == "list"
        assert total_value_out.type_hint == "float"
        assert count_out.type_hint == "int"

    def test_output_field_names_property(self):
        """output_field_names property should return list of output field names."""
        schema = ToolSchema.from_tool_function(_market_data.get_indices_tool, "get_indices_tool")
        assert "indices" in schema.output_field_names

    def test_repr_includes_outputs(self):
        """__repr__ should include output fields."""
        schema = ToolSchema.from_tool_function(_broker.list_brokers_tool, "list_brokers_tool")
        r = repr(schema)
        assert "outputs=" in r
        assert "brokers" in r

    def test_all_pmi_tools_have_outputs(self):
        """Every PMI tool should have at least one output field declared."""
        modules = {
            "pmi-market-data": _market_data,
            "pmi-portfolio": _portfolio,
            "pmi-watchlist": _watchlist,
            "pmi-trading": _trading,
            "pmi-strategies": _strategies,
            "pmi-alerts": _alerts,
            "pmi-broker": _broker,
        }
        missing = []
        for skill_name, mod in modules.items():
            for tool_name in getattr(mod, "__all__", []):
                func = getattr(mod, tool_name)
                schema = ToolSchema.from_tool_function(func, tool_name)
                if not schema.outputs:
                    missing.append(f"{skill_name}/{tool_name}")
        assert not missing, f"Tools missing output declarations: {missing}"

    def test_from_metadata_with_returns(self):
        """ToolSchema.from_metadata should parse returns list."""
        metadata = {
            "description": "Test tool",
            "parameters": {"properties": {}, "required": []},
            "returns": [
                {"name": "result", "type": "str", "description": "The result"},
                {"name": "count", "type": "int", "description": "Count"},
            ],
        }
        schema = ToolSchema.from_metadata("test_tool", metadata)
        assert len(schema.outputs) == 2
        assert schema.outputs[0].name == "result"
        assert schema.outputs[1].type_hint == "int"


# =============================================================================
# I/O Contract Enrichment Tests
# =============================================================================


class TestIOContractEnrichment:
    """Tests for _enrich_io_contracts post-processing in PlanUtilsMixin."""

    def _make_step(self, skill_name="web-search", tool_name="search_web_tool",
                   params=None, output_key="step_0", depends_on=None,
                   inputs_needed=None, outputs_produced=None):
        from Jotty.core.agents._execution_types import ExecutionStep
        return ExecutionStep(
            skill_name=skill_name,
            tool_name=tool_name,
            params=params or {},
            description="Test step",
            depends_on=depends_on or [],
            output_key=output_key,
            inputs_needed=inputs_needed or {},
            outputs_produced=outputs_produced or [],
        )

    def _make_mixin(self):
        from Jotty.core.agents._plan_utils_mixin import PlanUtilsMixin
        return PlanUtilsMixin()

    # -- _match_output_field tests --

    def test_match_exact_name(self):
        """Exact param name match to output field."""
        mixin = self._make_mixin()
        assert mixin._match_output_field("holdings", ["holdings", "total_pnl"]) == "holdings"

    def test_match_content_param_to_results(self):
        """Content-like param should match results/data/text fields."""
        mixin = self._make_mixin()
        assert mixin._match_output_field("content", ["success", "results", "count"]) == "results"

    def test_match_text_param_to_text_field(self):
        """text param should match text output field."""
        mixin = self._make_mixin()
        assert mixin._match_output_field("text", ["text", "error"]) == "text"

    def test_match_message_param_skips_meta(self):
        """message param should pick first non-meta field."""
        mixin = self._make_mixin()
        result = mixin._match_output_field("message", ["success", "indices", "count"])
        assert result == "indices"

    def test_match_single_non_meta_field(self):
        """Single non-meta field should match any param."""
        mixin = self._make_mixin()
        assert mixin._match_output_field("whatever", ["success", "holdings", "count"]) == "holdings"

    def test_no_match_ambiguous(self):
        """Multiple non-meta fields with no name match returns None."""
        mixin = self._make_mixin()
        result = mixin._match_output_field("xyz", ["holdings", "total_pnl", "day_pnl"])
        assert result is None

    def test_no_match_empty(self):
        """Empty output fields returns None."""
        mixin = self._make_mixin()
        assert mixin._match_output_field("anything", []) is None

    # -- _enrich_io_contracts tests --

    def test_empty_steps(self):
        """Empty step list passes through unchanged."""
        mixin = self._make_mixin()
        assert mixin._enrich_io_contracts([]) == []

    def test_preserves_existing_outputs_produced(self):
        """Already-set outputs_produced should not be overwritten."""
        mixin = self._make_mixin()
        step = self._make_step(outputs_produced=["custom_field"])
        result = mixin._enrich_io_contracts([step])
        assert result[0].outputs_produced == ["custom_field"]

    def test_preserves_existing_inputs_needed(self):
        """Already-set inputs_needed should not be overwritten."""
        mixin = self._make_mixin()
        step = self._make_step(
            params={"content": "${step_0}"},
            inputs_needed={"content": "step_0.custom"},
        )
        result = mixin._enrich_io_contracts([step])
        assert result[0].inputs_needed["content"] == "step_0.custom"

    def test_field_level_ref_already_present(self):
        """Params with ${key.field} refs should be left alone."""
        mixin = self._make_mixin()
        step = self._make_step(
            params={"content": "${step_0.holdings}"},
        )
        result = mixin._enrich_io_contracts([step])
        assert result[0].params["content"] == "${step_0.holdings}"

    @pytest.mark.unit
    def test_upgrade_bare_ref_with_registry(self):
        """Bare ${key} should be upgraded to ${key.field} when tool has returns schema.

        This test mocks the registry to provide a tool with declared outputs.
        """
        from Jotty.core.agents._execution_types import ToolSchema, ToolParam

        mixin = self._make_mixin()

        # Step 0: produces data
        step0 = self._make_step(
            skill_name="pmi-portfolio",
            tool_name="get_portfolio_tool",
            output_key="portfolio",
        )
        # Step 1: consumes with bare ref
        step1 = self._make_step(
            skill_name="claude-cli-llm",
            tool_name="summarize_text_tool",
            params={"content": "${portfolio}"},
            output_key="summary",
            depends_on=[0],
        )

        # Mock registry to return a schema with outputs for step0
        mock_schema = ToolSchema(name="get_portfolio_tool")
        mock_schema.outputs = [
            ToolParam(name="holdings", type_hint="list", required=False, description=""),
            ToolParam(name="total_value", type_hint="float", required=False, description=""),
            ToolParam(name="total_pnl", type_hint="float", required=False, description=""),
            ToolParam(name="count", type_hint="int", required=False, description=""),
        ]

        mock_func = MagicMock()
        mock_func._tool_schema = mock_schema

        mock_skill = MagicMock()
        mock_skill.tools = {"get_portfolio_tool": mock_func}

        mock_registry = MagicMock()
        mock_registry.get_skill.return_value = mock_skill

        with patch("Jotty.core.registry.skills_registry.get_skills_registry", return_value=mock_registry):
            result = mixin._enrich_io_contracts([step0, step1])

        # Step 0 should have outputs_produced auto-populated
        assert "holdings" in result[0].outputs_produced
        assert "total_value" in result[0].outputs_produced

        # Step 1 content param should be upgraded from ${portfolio} to ${portfolio.holdings}
        assert "${portfolio.holdings}" in result[1].params["content"]

    @pytest.mark.unit
    def test_upgrade_data_param_to_data_field(self):
        """data param should match data output field via exact match."""
        from Jotty.core.agents._execution_types import ToolSchema, ToolParam

        mixin = self._make_mixin()
        step0 = self._make_step(output_key="fetch")
        step1 = self._make_step(params={"data": "${fetch}"}, depends_on=[0])

        mock_schema = ToolSchema(name="fetch_tool")
        mock_schema.outputs = [
            ToolParam(name="data", type_hint="dict", required=False, description=""),
            ToolParam(name="status", type_hint="str", required=False, description=""),
        ]
        mock_func = MagicMock()
        mock_func._tool_schema = mock_schema
        mock_skill = MagicMock()
        mock_skill.tools = {step0.tool_name: mock_func}
        mock_registry = MagicMock()
        mock_registry.get_skill.return_value = mock_skill

        with patch("Jotty.core.registry.skills_registry.get_skills_registry", return_value=mock_registry):
            result = mixin._enrich_io_contracts([step0, step1])

        assert result[1].params["data"] == "${fetch.data}"

    @pytest.mark.unit
    def test_no_upgrade_when_no_match(self):
        """Bare ref with no matching field should stay as-is."""
        from Jotty.core.agents._execution_types import ToolSchema, ToolParam

        mixin = self._make_mixin()
        step0 = self._make_step(output_key="src")
        step1 = self._make_step(params={"xyz": "${src}"}, depends_on=[0])

        mock_schema = ToolSchema(name="tool")
        mock_schema.outputs = [
            ToolParam(name="alpha", type_hint="str", required=False, description=""),
            ToolParam(name="beta", type_hint="str", required=False, description=""),
            ToolParam(name="gamma", type_hint="str", required=False, description=""),
        ]
        mock_func = MagicMock()
        mock_func._tool_schema = mock_schema
        mock_skill = MagicMock()
        mock_skill.tools = {step0.tool_name: mock_func}
        mock_registry = MagicMock()
        mock_registry.get_skill.return_value = mock_skill

        with patch("Jotty.core.registry.skills_registry.get_skills_registry", return_value=mock_registry):
            result = mixin._enrich_io_contracts([step0, step1])

        # Should NOT be upgraded (ambiguous)
        assert result[1].params["xyz"] == "${src}"

    @pytest.mark.unit
    def test_infers_inputs_needed_from_template(self):
        """inputs_needed should be auto-populated from ${key.field} refs in params."""
        from Jotty.core.agents._execution_types import ToolSchema, ToolParam

        mixin = self._make_mixin()
        step0 = self._make_step(output_key="search")
        step1 = self._make_step(
            params={"content": "${search.results}", "query": "test"},
            depends_on=[0],
        )

        # No registry needed — step1 already has field-level ref
        with patch("Jotty.core.registry.skills_registry.get_skills_registry", return_value=None):
            result = mixin._enrich_io_contracts([step0, step1])

        assert result[1].inputs_needed.get("content") == "search.results"

    @pytest.mark.unit
    def test_multiple_refs_in_single_param(self):
        """Multiple template refs in one param value should all be processed."""
        from Jotty.core.agents._execution_types import ToolSchema, ToolParam

        mixin = self._make_mixin()
        step0 = self._make_step(output_key="a")
        step1 = self._make_step(output_key="b")
        step2 = self._make_step(
            params={"prompt": "Compare ${a} with ${b}"},
            depends_on=[0, 1],
        )

        # Mock schemas for both producer steps
        for s, key in [(step0, "a"), (step1, "b")]:
            mock_schema = ToolSchema(name=s.tool_name)
            mock_schema.outputs = [
                ToolParam(name="results", type_hint="list", required=False, description=""),
                ToolParam(name="count", type_hint="int", required=False, description=""),
            ]
            mock_func = MagicMock()
            mock_func._tool_schema = mock_schema
            mock_skill = MagicMock()
            mock_skill.tools = {s.tool_name: mock_func}

        # Single registry mock that handles both skills
        mock_registry = MagicMock()
        mock_registry.get_skill.return_value = mock_skill

        with patch("Jotty.core.registry.skills_registry.get_skills_registry", return_value=mock_registry):
            result = mixin._enrich_io_contracts([step0, step1, step2])

        # prompt param: 'prompt' is a content-like param, so ${a} -> ${a.results}
        assert "${a.results}" in result[2].params["prompt"] or "${a}" in result[2].params["prompt"]
