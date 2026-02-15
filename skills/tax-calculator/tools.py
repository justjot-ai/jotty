"""Tax Calculator Skill - US federal income tax brackets."""
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("tax-calculator")

# 2024 US Federal Tax Brackets
BRACKETS_2024 = {
    "single": [
        (11600, 0.10), (47150, 0.12), (100525, 0.22), (191950, 0.24),
        (243725, 0.32), (609350, 0.35), (float("inf"), 0.37),
    ],
    "married_joint": [
        (23200, 0.10), (94300, 0.12), (201050, 0.22), (383900, 0.24),
        (487450, 0.32), (731200, 0.35), (float("inf"), 0.37),
    ],
    "married_separate": [
        (11600, 0.10), (47150, 0.12), (100525, 0.22), (191950, 0.24),
        (243725, 0.32), (365600, 0.35), (float("inf"), 0.37),
    ],
    "head_of_household": [
        (16550, 0.10), (63100, 0.12), (100500, 0.22), (191950, 0.24),
        (243700, 0.32), (609350, 0.35), (float("inf"), 0.37),
    ],
}

STANDARD_DEDUCTIONS_2024 = {
    "single": 14600, "married_joint": 29200,
    "married_separate": 14600, "head_of_household": 21900,
}


def _calc_tax(taxable_income: float, brackets: list) -> tuple:
    tax = 0.0
    prev_limit = 0
    breakdown = []
    marginal_rate = 0.0

    for limit, rate in brackets:
        if taxable_income <= 0:
            break
        bracket_income = min(taxable_income, limit) - prev_limit
        if bracket_income <= 0:
            prev_limit = limit
            continue
        bracket_tax = round(bracket_income * rate, 2)
        tax += bracket_tax
        marginal_rate = rate
        breakdown.append({
            "bracket": f"{int(rate * 100)}%",
            "income_in_bracket": round(bracket_income, 2),
            "tax": bracket_tax,
            "range": f"${prev_limit:,.0f} - ${limit:,.0f}" if limit != float("inf") else f"${prev_limit:,.0f}+",
        })
        prev_limit = limit
        if taxable_income <= limit:
            break

    return round(tax, 2), marginal_rate, breakdown


@tool_wrapper(required_params=["income"])
def calculate_federal_tax_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate US federal income tax."""
    status.set_callback(params.pop("_status_callback", None))
    income = float(params["income"])
    filing = params.get("filing_status", "single").lower().replace(" ", "_")
    deductions = float(params.get("deductions", 0))

    if income < 0:
        return tool_error("Income cannot be negative")
    if filing not in BRACKETS_2024:
        return tool_error(f"Invalid filing status. Use: {list(BRACKETS_2024.keys())}")

    std_deduction = STANDARD_DEDUCTIONS_2024[filing]
    actual_deduction = max(deductions, std_deduction)
    deduction_type = "itemized" if deductions > std_deduction else "standard"
    taxable_income = max(0, income - actual_deduction)

    tax, marginal_rate, breakdown = _calc_tax(taxable_income, BRACKETS_2024[filing])
    effective_rate = round((tax / income) * 100, 2) if income > 0 else 0.0

    return tool_response(
        gross_income=income,
        deduction=actual_deduction,
        deduction_type=deduction_type,
        taxable_income=taxable_income,
        tax_owed=tax,
        effective_rate=effective_rate,
        marginal_rate=round(marginal_rate * 100, 1),
        filing_status=filing,
        breakdown=breakdown,
        formatted_tax=f"${tax:,.2f}",
    )


__all__ = ["calculate_federal_tax_tool"]
