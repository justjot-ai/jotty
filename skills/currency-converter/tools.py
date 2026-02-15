"""Currency Converter Skill — convert currencies using frankfurter.app."""

from typing import Any, Dict

import requests

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("currency-converter")


@tool_wrapper(required_params=["amount", "from_currency", "to_currency"])
def convert_currency_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert between currencies using live exchange rates."""
    status.set_callback(params.pop("_status_callback", None))
    try:
        amount = float(params["amount"])
    except (ValueError, TypeError):
        return tool_error("amount must be a number")

    from_cur = params["from_currency"].upper().strip()
    to_cur = params["to_currency"].upper().strip()

    if from_cur == to_cur:
        return tool_response(
            converted=amount, rate=1.0, from_currency=from_cur, to_currency=to_cur, amount=amount
        )

    try:
        resp = requests.get(
            f"https://api.frankfurter.app/latest",
            params={"amount": amount, "from": from_cur, "to": to_cur},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if "rates" not in data or to_cur not in data["rates"]:
            return tool_error(f"Could not get rate for {from_cur} to {to_cur}")

        converted = data["rates"][to_cur]
        rate = converted / amount if amount != 0 else 0

        return tool_response(
            converted=round(converted, 4),
            rate=round(rate, 6),
            from_currency=from_cur,
            to_currency=to_cur,
            amount=amount,
            date=data.get("date", ""),
        )
    except requests.RequestException as e:
        return tool_error(f"Currency conversion failed: {e}")


@tool_wrapper(required_params=["base"])
def list_rates_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """List all exchange rates for a base currency."""
    status.set_callback(params.pop("_status_callback", None))
    base = params["base"].upper().strip()
    try:
        resp = requests.get(
            f"https://api.frankfurter.app/latest",
            params={"from": base},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return tool_response(base=base, rates=data.get("rates", {}), date=data.get("date", ""))
    except requests.RequestException as e:
        return tool_error(f"Failed to fetch rates: {e}")


__all__ = ["convert_currency_tool", "list_rates_tool"]
