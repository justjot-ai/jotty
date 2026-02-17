"""
Math Toolkit — Unified math, finance, and conversion skill.

Consolidates: calculator, unit-converter, bmi-calculator, tip-calculator,
mortgage-calculator, loan-amortization-calculator, statistics-calculator,
currency-converter, number-base-converter, binary-converter, roman-numeral-converter.
"""

import math
import re
from collections import Counter
from typing import Any, Dict, List

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("math-toolkit")

# =============================================================================
# SHARED CONSTANTS
# =============================================================================

SAFE_MATH = {
    "__builtins__": {},
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": float("inf"),
    "nan": float("nan"),
}

_LENGTH = {
    "m": 1,
    "ft": 0.3048,
    "in": 0.0254,
    "km": 1000,
    "mi": 1609.344,
    "cm": 0.01,
    "mm": 0.001,
    "yd": 0.9144,
    "miles": 1609.344,
    "yards": 0.9144,
    "feet": 0.3048,
    "inches": 0.0254,
}
_WEIGHT = {
    "kg": 1,
    "lb": 0.453592,
    "lbs": 0.453592,
    "oz": 0.0283495,
    "g": 0.001,
    "mg": 1e-6,
    "ton": 907.185,
}
_VOLUME = {
    "l": 1,
    "gal": 3.78541,
    "ml": 0.001,
    "cup": 0.236588,
    "pt": 0.473176,
    "qt": 0.946353,
    "fl_oz": 0.0295735,
}
_SPEED = {"km/h": 1, "mph": 1.60934, "m/s": 3.6, "kn": 1.852, "ft/s": 1.09728}

_ROMAN_VALS = [
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
]
_ROMAN_MAP = {
    "I": 1,
    "V": 5,
    "X": 10,
    "L": 50,
    "C": 100,
    "D": 500,
    "M": 1000,
}


# =============================================================================
# INTERNAL HELPERS
# =============================================================================


def _extract_math_expression(text: str) -> str:
    """Extract a math expression from natural language."""
    cleaned = text.strip()
    if re.match(r"^[\d\s+\-*/().,%e^]+$", cleaned):
        return cleaned
    cleaned = re.sub(r"\$([0-9,.]+)", r"\1", cleaned)
    cleaned = cleaned.replace(",", "")
    math_match = re.search(r"([\d.]+\s*[+\-*/^%]\s*[\d.]+(?:\s*[+\-*/^%]\s*[\d.]+)*)", cleaned)
    if math_match:
        return math_match.group(1).replace("^", "**")
    numbers = [float(x) for x in re.findall(r"[\d]+\.?\d*", cleaned)]
    lower = cleaned.lower()
    if len(numbers) >= 2:
        a, b = numbers[0], numbers[1]
        if any(
            kw in lower
            for kw in (
                "percentage gain",
                "percent gain",
                "% gain",
                "percentage change",
                "percent change",
                "percentage increase",
            )
        ):
            return f"({b}-{a})/{a}*100"
        if any(
            kw in lower for kw in ("percentage decrease", "percent decrease", "percentage loss")
        ):
            return f"({a}-{b})/{a}*100"
        if any(
            kw in lower
            for kw in ("convert", "at rate", "multiply", "times", "rate of", "p/e", "pe ratio")
        ):
            return f"{a}*{b}"
        if any(kw in lower for kw in ("divide", "ratio of", "divided by")):
            return f"{a}/{b}"
        if any(kw in lower for kw in ("difference", "subtract", "minus", "less")):
            return f"{a}-{b}"
        if any(kw in lower for kw in ("sum", "add", "plus", "total", "combined")):
            return f"{a}+{b}"
    return text


def _convert_table(val: float, f: str, t: str, tbl: dict) -> float | None:
    if f in tbl and t in tbl:
        return val * tbl[f] / tbl[t]
    return None


def _temp_convert(val: float, f: str, t: str) -> float | None:
    aliases = {"c": "c", "celsius": "c", "f": "f", "fahrenheit": "f", "k": "k", "kelvin": "k"}
    fc, tc = aliases.get(f), aliases.get(t)
    if not fc or not tc:
        return None
    if fc == tc:
        return val
    to_c = {"c": lambda v: v, "f": lambda v: (v - 32) * 5 / 9, "k": lambda v: v - 273.15}
    from_c = {"c": lambda v: v, "f": lambda v: v * 9 / 5 + 32, "k": lambda v: v + 273.15}
    return from_c[tc](to_c[fc](val))


def _int_to_roman(num: int) -> str:
    result = ""
    for value, numeral in _ROMAN_VALS:
        while num >= value:
            result += numeral
            num -= value
    return result


def _roman_to_int(s: str) -> int:
    total = 0
    prev = 0
    for ch in reversed(s.upper()):
        val = _ROMAN_MAP.get(ch, 0)
        if val < prev:
            total -= val
        else:
            total += val
        prev = val
    return total


# =============================================================================
# TOOLS
# =============================================================================


@tool_wrapper(required_params=["expression"])
def calculate_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate mathematical expressions with natural language support.

    Supports: +, -, *, /, **, %, sqrt, sin, cos, tan, log, exp, abs, round,
    floor, ceil, pi, e. Also understands 'percentage gain from 500 to 850'.
    """
    status.set_callback(params.pop("_status_callback", None))
    raw = params["expression"]
    expression = _extract_math_expression(raw)
    try:
        result = eval(expression, SAFE_MATH)
        if not isinstance(result, (int, float)):
            return tool_error(f"Expression did not evaluate to a number: {result}")
        response = tool_response(result=float(result), expression=expression)
        if expression != raw:
            response["original_input"] = raw
            response["parsed_expression"] = expression
        return response
    except ZeroDivisionError:
        return tool_error(f"Division by zero. Expression: {expression}")
    except NameError as e:
        return tool_error(
            f"Unknown function/variable: {e}. Available: sqrt, sin, cos, tan, log, exp, abs, round."
        )
    except SyntaxError as e:
        return tool_error(f"Invalid syntax: {e}. Expression: {expression}")


@tool_wrapper(required_params=["value", "from_unit", "to_unit"])
def convert_units_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert between length, weight, temperature, volume, and speed units."""
    status.set_callback(params.pop("_status_callback", None))
    try:
        val = float(params["value"])
    except (ValueError, TypeError):
        return tool_error("value must be a number")
    f = params["from_unit"].lower().strip()
    t = params["to_unit"].lower().strip()
    for tbl in [_LENGTH, _WEIGHT, _VOLUME, _SPEED]:
        r = _convert_table(val, f, t, tbl)
        if r is not None:
            return tool_response(result=round(r, 6), from_unit=f, to_unit=t, value=val)
    r = _temp_convert(val, f, t)
    if r is not None:
        return tool_response(result=round(r, 6), from_unit=f, to_unit=t, value=val)
    return tool_error(
        f"Unsupported conversion: {f} -> {t}. Supported: length, weight, volume, speed, temperature."
    )


@tool_wrapper(required_params=["weight_kg", "height_m"])
def bmi_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate BMI, category, and healthy weight range."""
    status.set_callback(params.pop("_status_callback", None))
    try:
        w = float(params["weight_kg"])
        h = float(params["height_m"])
    except (ValueError, TypeError):
        return tool_error("weight_kg and height_m must be numbers")
    if h <= 0 or w <= 0:
        return tool_error("height and weight must be positive")
    bmi = round(w / (h * h), 2)
    if bmi < 18.5:
        cat = "Underweight"
    elif bmi < 25:
        cat = "Normal weight"
    elif bmi < 30:
        cat = "Overweight"
    else:
        cat = "Obese"
    return tool_response(
        bmi=bmi,
        category=cat,
        healthy_weight_range_kg={"min": round(18.5 * h * h, 1), "max": round(24.9 * h * h, 1)},
        weight_kg=w,
        height_m=h,
    )


@tool_wrapper(required_params=["bill_amount"])
def tip_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate tip amount, total, and per-person split."""
    status.set_callback(params.pop("_status_callback", None))
    try:
        bill = float(params["bill_amount"])
    except (ValueError, TypeError):
        return tool_error("bill_amount must be a number")
    if bill < 0:
        return tool_error("bill_amount cannot be negative")
    pct = float(params.get("tip_percent", 18))
    people = int(params.get("num_people", 1))
    if people < 1:
        return tool_error("num_people must be at least 1")
    tip = round(bill * pct / 100, 2)
    total = round(bill + tip, 2)
    per_person = round(total / people, 2)
    suggestions = {}
    for p in [15, 18, 20, 25]:
        t = round(bill * p / 100, 2)
        suggestions[f"{p}%"] = {
            "tip": t,
            "total": round(bill + t, 2),
            "per_person": round((bill + t) / people, 2),
        }
    return tool_response(
        bill_amount=bill,
        tip_percent=pct,
        tip_amount=tip,
        total=total,
        num_people=people,
        per_person=per_person,
        suggestions=suggestions,
    )


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
    balance = P
    r = rate / 12
    yearly: List[Dict[str, Any]] = []
    yr_principal = yr_interest = 0.0
    for m in range(1, n + 1):
        mi = round(balance * r, 2) if rate > 0 else 0.0
        mp = round(monthly - mi, 2)
        balance = round(balance - mp, 2)
        yr_interest += mi
        yr_principal += mp
        if m % 12 == 0:
            yearly.append(
                {
                    "year": m // 12,
                    "principal_paid": round(yr_principal, 2),
                    "interest_paid": round(yr_interest, 2),
                    "remaining_balance": max(round(balance, 2), 0),
                }
            )
            yr_principal = yr_interest = 0.0
    summary = yearly[:5] + (yearly[-1:] if len(yearly) > 5 else [])
    return tool_response(
        monthly_payment=monthly,
        total_paid=total_paid,
        total_interest=total_interest,
        term_months=n,
        amortization_summary=summary,
    )


@tool_wrapper(required_params=["principal", "annual_rate", "years"])
def amortization_schedule_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate detailed loan amortization schedule with optional extra payments."""
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
        monthly_payment = (
            principal
            * (monthly_rate * (1 + monthly_rate) ** num_payments)
            / ((1 + monthly_rate) ** num_payments - 1)
        )
    monthly_payment = round(monthly_payment, 2)
    balance = principal
    total_interest = total_paid = 0.0
    schedule = []
    for month in range(1, num_payments + 1):
        interest_pmt = round(balance * monthly_rate, 2)
        principal_pmt = round(monthly_payment - interest_pmt + extra, 2)
        if principal_pmt > balance:
            principal_pmt = balance
            interest_pmt = round(balance * monthly_rate, 2)
        balance = round(balance - principal_pmt, 2)
        total_interest += interest_pmt
        total_paid += interest_pmt + principal_pmt
        schedule.append(
            {
                "month": month,
                "payment": round(interest_pmt + principal_pmt, 2),
                "principal": principal_pmt,
                "interest": interest_pmt,
                "balance": max(0, balance),
            }
        )
        if balance <= 0:
            break
    summary_schedule = schedule[:12] + (schedule[-1:] if len(schedule) > 12 else [])
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
        interest_savings=round(monthly_payment * num_payments - total_paid, 2) if extra else 0,
    )


@tool_wrapper(required_params=["data"])
def statistics_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Compute descriptive statistics on a numeric dataset."""
    status.set_callback(params.pop("_status_callback", None))
    data = params["data"]
    if not isinstance(data, list) or not data:
        return tool_error("data must be a non-empty list of numbers")
    try:
        nums = sorted(float(x) for x in data)
    except (ValueError, TypeError):
        return tool_error("All items in data must be numbers")
    n = len(nums)
    mean_val = sum(nums) / n
    mid = n // 2
    median_val = nums[mid] if n % 2 else (nums[mid - 1] + nums[mid]) / 2
    counter = Counter(nums)
    mode_val = counter.most_common(1)[0][0]
    variance = sum((x - mean_val) ** 2 for x in nums) / n
    std_dev = math.sqrt(variance)
    result = tool_response(
        count=n,
        mean=round(mean_val, 6),
        median=round(median_val, 6),
        mode=mode_val,
        std_dev=round(std_dev, 6),
        variance=round(variance, 6),
        min=nums[0],
        max=nums[-1],
        range=round(nums[-1] - nums[0], 6),
        sum=round(sum(nums), 6),
    )
    if params.get("include_percentiles"):

        def _pct(p: float) -> float:
            k = (n - 1) * p / 100
            f_idx = int(k)
            c_idx = min(f_idx + 1, n - 1)
            return round(nums[f_idx] + (k - f_idx) * (nums[c_idx] - nums[f_idx]), 6)

        result["percentiles"] = {f"p{p}": _pct(p) for p in [10, 25, 50, 75, 90]}
    return result


@tool_wrapper(required_params=["value"])
def convert_number_base_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numbers between bases (binary, octal, decimal, hex, any base 2-36)."""
    status.set_callback(params.pop("_status_callback", None))
    value = str(params["value"]).strip()
    from_base = int(params.get("from_base", 10))
    to_base = int(params.get("to_base", 2))
    if not (2 <= from_base <= 36) or not (2 <= to_base <= 36):
        return tool_error("Bases must be between 2 and 36")
    try:
        decimal_val = int(value, from_base)
    except ValueError:
        return tool_error(f"Invalid number '{value}' for base {from_base}")
    if to_base == 10:
        converted = str(decimal_val)
    elif to_base == 2:
        converted = bin(decimal_val)[2:]
    elif to_base == 8:
        converted = oct(decimal_val)[2:]
    elif to_base == 16:
        converted = hex(decimal_val)[2:].upper()
    else:
        digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if decimal_val == 0:
            converted = "0"
        else:
            result_digits = []
            n = abs(decimal_val)
            while n:
                result_digits.append(digits[n % to_base])
                n //= to_base
            if decimal_val < 0:
                result_digits.append("-")
            converted = "".join(reversed(result_digits))
    return tool_response(
        result=converted,
        decimal=decimal_val,
        from_base=from_base,
        to_base=to_base,
        all_bases={
            "binary": bin(decimal_val)[2:],
            "octal": oct(decimal_val)[2:],
            "decimal": str(decimal_val),
            "hex": hex(decimal_val)[2:].upper(),
        },
    )


@tool_wrapper(required_params=["value"])
def roman_numeral_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert between Roman numerals and integers."""
    status.set_callback(params.pop("_status_callback", None))
    value = str(params["value"]).strip()
    if value.isdigit():
        num = int(value)
        if num < 1 or num > 3999:
            return tool_error("Integer must be between 1 and 3999 for Roman numeral conversion")
        return tool_response(roman=_int_to_roman(num), integer=num)
    elif all(c in _ROMAN_MAP for c in value.upper()):
        num = _roman_to_int(value)
        return tool_response(roman=value.upper(), integer=num)
    else:
        return tool_error(
            f"Invalid input: '{value}'. Provide an integer (1-3999) or a Roman numeral."
        )


__all__ = [
    "calculate_tool",
    "convert_units_tool",
    "bmi_tool",
    "tip_tool",
    "mortgage_tool",
    "amortization_schedule_tool",
    "statistics_tool",
    "convert_number_base_tool",
    "roman_numeral_tool",
]
