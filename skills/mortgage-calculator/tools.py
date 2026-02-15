"""Mortgage calculator — payment, total interest, amortization."""
from typing import Dict, Any
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus
status = SkillStatus("mortgage-calculator")


@tool_wrapper(required_params=["principal", "annual_rate", "term_years"])
def mortgage_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate monthly mortgage payment and amortization summary."""
    status.set_callback(params.pop("_status_callback", None))
    try:
        P = float(params["principal"])
        rate = float(params["annual_rate"]) / 100
        years = int(params["term_years"])
    except (ValueError, TypeError):
        return tool_error("principal, annual_rate, term_years must be numeric")
    if P <= 0 or rate < 0 or years <= 0:
        return tool_error("Values must be positive (rate can be 0)")
    n = years * 12
    if rate == 0:
        monthly = round(P / n, 2)
        total_interest = 0.0
    else:
        r = rate / 12
        monthly = round(P * r * (1 + r) ** n / ((1 + r) ** n - 1), 2)
        total_interest = round(monthly * n - P, 2)
    total_paid = round(monthly * n, 2)
    # Yearly amortization summary (first 5 and last year)
    balance = P
    r = rate / 12
    yearly = []
    yr_principal = yr_interest = 0.0
    for m in range(1, n + 1):
        mi = round(balance * r, 2) if rate > 0 else 0.0
        mp = round(monthly - mi, 2)
        balance = round(balance - mp, 2)
        yr_interest += mi
        yr_principal += mp
        if m % 12 == 0:
            yr_num = m // 12
            yearly.append({"year": yr_num, "principal_paid": round(yr_principal, 2),
                           "interest_paid": round(yr_interest, 2),
                           "remaining_balance": max(round(balance, 2), 0)})
            yr_principal = yr_interest = 0.0
    summary = yearly[:5] + (yearly[-1:] if len(yearly) > 5 else [])
    return tool_response(
        monthly_payment=monthly, total_paid=total_paid,
        total_interest=total_interest, term_months=n,
        amortization_summary=summary,
    )


__all__ = ["mortgage_tool"]
