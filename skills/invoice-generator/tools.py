"""Invoice Generator Skill - generate structured invoice data."""
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("invoice-generator")

CURRENCY_SYMBOLS = {
    "USD": "$", "EUR": "\u20ac", "GBP": "\u00a3", "JPY": "\u00a5",
    "CAD": "CA$", "AUD": "A$", "CHF": "CHF", "INR": "\u20b9",
}


@tool_wrapper(required_params=["client_name", "items"])
def generate_invoice_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a structured invoice with line items and totals."""
    status.set_callback(params.pop("_status_callback", None))
    client = params["client_name"]
    items = params["items"]
    tax_rate = float(params.get("tax_rate", 0))
    currency = params.get("currency", "USD").upper()
    due_days = int(params.get("due_days", 30))
    notes = params.get("notes", "")

    if not items or not isinstance(items, list):
        return tool_error("items must be a non-empty list of {description, quantity, rate}")

    inv_number = params.get("invoice_number", f"INV-{uuid.uuid4().hex[:8].upper()}")
    now = datetime.now(timezone.utc)
    due_date = now + timedelta(days=due_days)
    symbol = CURRENCY_SYMBOLS.get(currency, currency + " ")

    line_items = []
    subtotal = 0.0
    for i, item in enumerate(items):
        desc = item.get("description", f"Item {i + 1}")
        qty = float(item.get("quantity", 1))
        rate = float(item.get("rate", 0))
        amount = round(qty * rate, 2)
        subtotal += amount
        line_items.append({
            "line": i + 1,
            "description": desc,
            "quantity": qty,
            "rate": rate,
            "amount": amount,
            "formatted_amount": f"{symbol}{amount:,.2f}",
        })

    subtotal = round(subtotal, 2)
    tax_amount = round(subtotal * tax_rate / 100, 2) if tax_rate else 0.0
    total = round(subtotal + tax_amount, 2)

    invoice = {
        "invoice_number": inv_number,
        "date": now.strftime("%Y-%m-%d"),
        "due_date": due_date.strftime("%Y-%m-%d"),
        "client_name": client,
        "currency": currency,
        "line_items": line_items,
        "subtotal": subtotal,
        "tax_rate": tax_rate,
        "tax_amount": tax_amount,
        "total": total,
        "formatted_subtotal": f"{symbol}{subtotal:,.2f}",
        "formatted_tax": f"{symbol}{tax_amount:,.2f}",
        "formatted_total": f"{symbol}{total:,.2f}",
        "notes": notes,
        "status": "pending",
    }

    return tool_response(invoice=invoice)


__all__ = ["generate_invoice_tool"]
