"""Crypto Price Tracker Skill — fetch prices from CoinGecko."""
import requests
from typing import Dict, Any

from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

status = SkillStatus("crypto-price-tracker")

SYMBOL_MAP = {
    "btc": "bitcoin", "eth": "ethereum", "bnb": "binancecoin",
    "xrp": "ripple", "ada": "cardano", "sol": "solana",
    "dot": "polkadot", "doge": "dogecoin", "avax": "avalanche-2",
    "matic": "matic-network", "ltc": "litecoin", "link": "chainlink",
    "uni": "uniswap", "atom": "cosmos", "xlm": "stellar",
}


@tool_wrapper(required_params=["coin"])
def crypto_price_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get current cryptocurrency price from CoinGecko."""
    status.set_callback(params.pop("_status_callback", None))
    coin = params["coin"].strip().lower()
    currency = params.get("currency", "usd").strip().lower()

    coin_id = SYMBOL_MAP.get(coin, coin)

    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={
                "ids": coin_id,
                "vs_currencies": currency,
                "include_24hr_change": "true",
                "include_market_cap": "true",
                "include_24hr_vol": "true",
            },
            timeout=10,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()

        if coin_id not in data:
            return tool_error(f"Coin not found: {coin}. Use full name (e.g. bitcoin) or symbol (e.g. btc)")

        coin_data = data[coin_id]
        return tool_response(
            coin=coin_id,
            symbol=coin,
            currency=currency,
            price=coin_data.get(currency),
            change_24h=coin_data.get(f"{currency}_24h_change"),
            market_cap=coin_data.get(f"{currency}_market_cap"),
            volume_24h=coin_data.get(f"{currency}_24h_vol"),
        )
    except requests.RequestException as e:
        return tool_error(f"Failed to fetch crypto price: {e}")


@tool_wrapper()
def crypto_top_coins_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get top cryptocurrencies by market cap."""
    status.set_callback(params.pop("_status_callback", None))
    currency = params.get("currency", "usd").lower()
    limit = min(int(params.get("limit", 10)), 50)

    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={"vs_currency": currency, "order": "market_cap_desc",
                    "per_page": limit, "page": 1},
            timeout=10,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        coins = []
        for c in resp.json():
            coins.append({
                "name": c.get("name"), "symbol": c.get("symbol"),
                "price": c.get("current_price"),
                "market_cap": c.get("market_cap"),
                "change_24h": c.get("price_change_percentage_24h"),
            })
        return tool_response(coins=coins, currency=currency, count=len(coins))
    except requests.RequestException as e:
        return tool_error(f"Failed to fetch top coins: {e}")


__all__ = ["crypto_price_tool", "crypto_top_coins_tool"]
