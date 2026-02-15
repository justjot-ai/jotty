"""Loan Amortization Calculator Skill."""
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("loan-amortization-calculator")


@tool_wrapper(required_params=["principal", "annual_rate", "years"])
def amortization_schedule_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a loan amortization schedule."""
    status.set_callback(params.pop("_status_callback", None))
    principal = float(params["principal"])
    annual_rate = float(params["annual_rate"])
    years = int(params["years"])
    extra = float(params.get("extra_payment", 0))

    if principal <= 0:
        return tool_error("Principal must be positive")
    if annual_rate < 0:
        return tool_error("Interest rate cannot be negative")
    if years <= 0:
        return tool_error("Term must be positive")

    monthly_rate = annual_rate / 100 / 12
    num_payments = years * 12

    if monthly_rate == 0:
        monthly_payment = principal / num_payments
    else:
        monthly_payment = principal * (monthly_rate * (1 + monthly_rate) ** num_payments) / \
                          ((1 + monthly_rate) ** num_payments - 1)

    monthly_payment = round(monthly_payment, 2)

    # Generate schedule
    balance = principal
    total_interest = 0.0
    total_paid = 0.0
    schedule = []

    for month in range(1, num_payments + 1):
        interest_payment = round(balance * monthly_rate, 2)
        principal_payment = round(monthly_payment - interest_payment + extra, 2)

        if principal_payment > balance:
            principal_payment = balance
            interest_payment = round(balance * monthly_rate, 2)

        balance = round(balance - principal_payment, 2)
        total_interest += interest_payment
        total_paid += interest_payment + principal_payment

        entry = {
            "month": month,
            "payment": round(interest_payment + principal_payment, 2),
            "principal": principal_payment,
            "interest": interest_payment,
            "balance": max(0, balance),
        }
        schedule.append(entry)

        if balance <= 0:
            break

    # Return first 12 months + last month for brevity
    summary_schedule = schedule[:12]
    if len(schedule) > 12:
        summary_schedule.append(schedule[-1])

    return tool_response(
        monthly_payment=monthly_payment,
        total_interest=round(total_interest, 2),
        total_paid=round(total_paid, 2),
        principal=principal,
        annual_rate=annual_rate,
        term_years=years,
        extra_payment=extra,
        payoff_months=len(schedule),
        schedule=summary_schedule,
        interest_savings=round((monthly_payment * num_payments - total_paid), 2) if extra else 0,
    )


__all__ = ["amortization_schedule_tool"]
