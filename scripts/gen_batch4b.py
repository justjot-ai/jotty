"""Batch 4b — 10 utility skills with real implementations."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.generate_skills import create_skill

# ---------------------------------------------------------------------------
# 1. text-statistics
# ---------------------------------------------------------------------------
create_skill(
    name="text-statistics",
    frontmatter_name="text-statistics",
    description="Compute word count, character count, sentence count, reading time, and Flesch-Kincaid grade level from text.",
    category="text-analysis",
    capabilities=["Word count", "Character count", "Sentence count", "Reading time estimate", "Flesch-Kincaid grade level"],
    triggers=["analyze text statistics", "word count", "reading time", "readability score"],
    eval_tool="analyze_text_tool",
    eval_input={"text": "Hello world. This is a test."},
    tool_docs="### analyze_text_tool\nCompute statistics for a body of text.",
    tools_code='''"""Text statistics — word count, reading time, Flesch-Kincaid."""
import re, math
from typing import Dict, Any
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus
status = SkillStatus("text-statistics")


def _count_syllables(word: str) -> int:
    word = word.lower().strip()
    if not word:
        return 0
    if len(word) <= 2:
        return 1
    word = re.sub(r"(?:es|ed|e)$", "", word) or word
    vowels = re.findall(r"[aeiouy]+", word)
    return max(1, len(vowels))


@tool_wrapper(required_params=["text"])
def analyze_text_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return word count, char count, sentence count, reading time, FK grade."""
    status.set_callback(params.pop("_status_callback", None))
    text = params["text"]
    words = re.findall(r"\\b[a-zA-Z0-9\\']+\\b", text)
    word_count = len(words)
    char_count = len(text)
    char_no_spaces = len(text.replace(" ", ""))
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sentence_count = max(len(sentences), 1)
    reading_time_min = round(word_count / 238, 2)
    total_syllables = sum(_count_syllables(w) for w in words)
    if word_count > 0:
        fk = (0.39 * (word_count / sentence_count)
              + 11.8 * (total_syllables / word_count)
              - 15.59)
        fk = round(fk, 2)
    else:
        fk = 0.0
    return tool_response(
        word_count=word_count, char_count=char_count,
        char_count_no_spaces=char_no_spaces,
        sentence_count=len(sentences), reading_time_minutes=reading_time_min,
        flesch_kincaid_grade=fk, syllable_count=total_syllables,
    )


__all__ = ["analyze_text_tool"]
''',
)

# ---------------------------------------------------------------------------
# 2. ascii-art-generator
# ---------------------------------------------------------------------------
create_skill(
    name="ascii-art-generator",
    frontmatter_name="ascii-art-generator",
    description="Convert text to ASCII block-letter art using a built-in font map. Pure Python.",
    category="text-fun",
    capabilities=["Text to ASCII art", "Block letter rendering"],
    triggers=["ascii art", "text to art", "block letters"],
    eval_tool="ascii_art_tool",
    eval_input={"text": "HI"},
    tool_docs="### ascii_art_tool\nConvert text to ASCII block art.",
    tools_code='''"""ASCII art generator — block-letter font, pure Python."""
from typing import Dict, Any
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus
status = SkillStatus("ascii-art-generator")

_F = {
    "A": ["  #  ", " # # ", "#####", "#   #", "#   #"],
    "B": ["#### ", "#   #", "#### ", "#   #", "#### "],
    "C": [" ####", "#    ", "#    ", "#    ", " ####"],
    "D": ["#### ", "#   #", "#   #", "#   #", "#### "],
    "E": ["#####", "#    ", "###  ", "#    ", "#####"],
    "F": ["#####", "#    ", "###  ", "#    ", "#    "],
    "G": [" ####", "#    ", "# ###", "#   #", " ####"],
    "H": ["#   #", "#   #", "#####", "#   #", "#   #"],
    "I": ["#####", "  #  ", "  #  ", "  #  ", "#####"],
    "J": ["#####", "    #", "    #", "#   #", " ### "],
    "K": ["#   #", "#  # ", "###  ", "#  # ", "#   #"],
    "L": ["#    ", "#    ", "#    ", "#    ", "#####"],
    "M": ["#   #", "## ##", "# # #", "#   #", "#   #"],
    "N": ["#   #", "##  #", "# # #", "#  ##", "#   #"],
    "O": [" ### ", "#   #", "#   #", "#   #", " ### "],
    "P": ["#### ", "#   #", "#### ", "#    ", "#    "],
    "Q": [" ### ", "#   #", "# # #", "#  ##", " ####"],
    "R": ["#### ", "#   #", "#### ", "#  # ", "#   #"],
    "S": [" ####", "#    ", " ### ", "    #", "#### "],
    "T": ["#####", "  #  ", "  #  ", "  #  ", "  #  "],
    "U": ["#   #", "#   #", "#   #", "#   #", " ### "],
    "V": ["#   #", "#   #", " # # ", " # # ", "  #  "],
    "W": ["#   #", "#   #", "# # #", "## ##", "#   #"],
    "X": ["#   #", " # # ", "  #  ", " # # ", "#   #"],
    "Y": ["#   #", " # # ", "  #  ", "  #  ", "  #  "],
    "Z": ["#####", "   # ", "  #  ", " #   ", "#####"],
    "0": [" ### ", "#   #", "#   #", "#   #", " ### "],
    "1": ["  #  ", " ##  ", "  #  ", "  #  ", "#####"],
    "2": [" ### ", "#   #", "  ## ", " #   ", "#####"],
    "3": ["#### ", "    #", " ### ", "    #", "#### "],
    "4": ["#   #", "#   #", "#####", "    #", "    #"],
    "5": ["#####", "#    ", "#### ", "    #", "#### "],
    "6": [" ####", "#    ", "#### ", "#   #", " ### "],
    "7": ["#####", "   # ", "  #  ", " #   ", "#    "],
    "8": [" ### ", "#   #", " ### ", "#   #", " ### "],
    "9": [" ### ", "#   #", " ####", "    #", "#### "],
    " ": ["     ", "     ", "     ", "     ", "     "],
}


@tool_wrapper(required_params=["text"])
def ascii_art_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert text to ASCII block-letter art."""
    status.set_callback(params.pop("_status_callback", None))
    text = params["text"].upper()[:40]
    char = params.get("char", "#")
    lines = ["", "", "", "", ""]
    for ch in text:
        glyph = _F.get(ch, ["?????"] * 5)
        for i in range(5):
            row = glyph[i] if char == "#" else glyph[i].replace("#", char)
            lines[i] += row + "  "
    art = "\\n".join(l.rstrip() for l in lines)
    return tool_response(art=art, text=params["text"])


__all__ = ["ascii_art_tool"]
''',
)

# ---------------------------------------------------------------------------
# 3. unit-converter
# ---------------------------------------------------------------------------
create_skill(
    name="unit-converter",
    frontmatter_name="unit-converter",
    description="Convert between length, weight, temperature, volume, and speed units.",
    category="utility",
    capabilities=["Length conversion", "Weight conversion", "Temperature conversion", "Volume conversion", "Speed conversion"],
    triggers=["convert units", "meters to feet", "kg to lbs", "celsius to fahrenheit"],
    eval_tool="convert_unit_tool",
    eval_input={"value": 100, "from_unit": "km", "to_unit": "mi"},
    tool_docs="### convert_unit_tool\nConvert a value between supported units.",
    tools_code='''"""Unit converter — length, weight, temperature, volume, speed."""
from typing import Dict, Any
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus
status = SkillStatus("unit-converter")

_LENGTH = {"m": 1, "ft": 0.3048, "in": 0.0254, "km": 1000, "mi": 1609.344, "cm": 0.01, "mm": 0.001, "yd": 0.9144}
_WEIGHT = {"kg": 1, "lb": 0.453592, "oz": 0.0283495, "g": 0.001, "mg": 1e-6, "ton": 907.185}
_VOLUME = {"l": 1, "gal": 3.78541, "ml": 0.001, "cup": 0.236588, "pt": 0.473176, "qt": 0.946353, "fl_oz": 0.0295735}
_SPEED = {"km/h": 1, "mph": 1.60934, "m/s": 3.6, "kn": 1.852, "ft/s": 1.09728}


def _convert_table(val: float, f: str, t: str, tbl: dict) -> float | None:
    if f in tbl and t in tbl:
        return val * tbl[f] / tbl[t]
    return None


def _temp(val: float, f: str, t: str) -> float | None:
    aliases = {"c": "c", "celsius": "c", "f": "f", "fahrenheit": "f", "k": "k", "kelvin": "k"}
    fc, tc = aliases.get(f), aliases.get(t)
    if not fc or not tc:
        return None
    if fc == tc:
        return val
    to_c = {"c": lambda v: v, "f": lambda v: (v - 32) * 5 / 9, "k": lambda v: v - 273.15}
    from_c = {"c": lambda v: v, "f": lambda v: v * 9 / 5 + 32, "k": lambda v: v + 273.15}
    return from_c[tc](to_c[fc](val))


@tool_wrapper(required_params=["value", "from_unit", "to_unit"])
def convert_unit_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert value between units."""
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
    r = _temp(val, f, t)
    if r is not None:
        return tool_response(result=round(r, 6), from_unit=f, to_unit=t, value=val)
    return tool_error(f"Unsupported conversion: {f} -> {t}")


__all__ = ["convert_unit_tool"]
''',
)

# ---------------------------------------------------------------------------
# 4. bmi-calculator
# ---------------------------------------------------------------------------
create_skill(
    name="bmi-calculator",
    frontmatter_name="bmi-calculator",
    description="Calculate BMI from height and weight, return category and healthy weight range.",
    category="health",
    capabilities=["BMI calculation", "BMI category classification", "Healthy weight range"],
    triggers=["calculate bmi", "body mass index", "am I overweight"],
    eval_tool="bmi_tool",
    eval_input={"weight_kg": 70, "height_m": 1.75},
    tool_docs="### bmi_tool\nCalculate BMI, category, and healthy weight range.",
    tools_code='''"""BMI calculator — value, category, healthy range."""
from typing import Dict, Any
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus
status = SkillStatus("bmi-calculator")


@tool_wrapper(required_params=["weight_kg", "height_m"])
def bmi_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate BMI from weight (kg) and height (m)."""
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
    low = round(18.5 * h * h, 1)
    high = round(24.9 * h * h, 1)
    return tool_response(
        bmi=bmi, category=cat,
        healthy_weight_range_kg={"min": low, "max": high},
        weight_kg=w, height_m=h,
    )


__all__ = ["bmi_tool"]
''',
)

# ---------------------------------------------------------------------------
# 5. mortgage-calculator
# ---------------------------------------------------------------------------
create_skill(
    name="mortgage-calculator",
    frontmatter_name="mortgage-calculator",
    description="Calculate monthly payment, total interest, and amortization summary for a mortgage.",
    category="finance",
    capabilities=["Monthly payment calculation", "Total interest", "Amortization summary"],
    triggers=["mortgage calculator", "monthly payment", "home loan calculation"],
    eval_tool="mortgage_tool",
    eval_input={"principal": 300000, "annual_rate": 6.5, "term_years": 30},
    tool_docs="### mortgage_tool\nCalculate mortgage payment and amortization.",
    tools_code='''"""Mortgage calculator — payment, total interest, amortization."""
from typing import Dict, Any
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus
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
''',
)

# ---------------------------------------------------------------------------
# 6. tip-calculator
# ---------------------------------------------------------------------------
create_skill(
    name="tip-calculator",
    frontmatter_name="tip-calculator",
    description="Calculate tip amount, total bill, and per-person split.",
    category="finance",
    capabilities=["Tip calculation", "Bill splitting", "Multiple tip percentages"],
    triggers=["calculate tip", "tip calculator", "split the bill"],
    eval_tool="tip_tool",
    eval_input={"bill_amount": 85.50, "tip_percent": 18, "num_people": 3},
    tool_docs="### tip_tool\nCalculate tip, total, and per-person split.",
    tools_code='''"""Tip calculator — amount, total, per-person split."""
from typing import Dict, Any
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus
status = SkillStatus("tip-calculator")


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
        suggestions[f"{p}%"] = {"tip": t, "total": round(bill + t, 2),
                                "per_person": round((bill + t) / people, 2)}
    return tool_response(
        bill_amount=bill, tip_percent=pct, tip_amount=tip,
        total=total, num_people=people, per_person=per_person,
        suggestions=suggestions,
    )


__all__ = ["tip_tool"]
''',
)

# ---------------------------------------------------------------------------
# 7. timezone-converter
# ---------------------------------------------------------------------------
create_skill(
    name="timezone-converter",
    frontmatter_name="timezone-converter",
    description="Convert datetime between timezones using UTC offsets and common timezone names.",
    category="utility",
    capabilities=["Timezone conversion", "Common timezone support", "UTC offset handling"],
    triggers=["convert timezone", "what time is it in", "EST to PST"],
    eval_tool="timezone_convert_tool",
    eval_input={"datetime_str": "2025-01-15 14:30", "from_tz": "EST", "to_tz": "PST"},
    tool_docs="### timezone_convert_tool\nConvert datetime between timezones.",
    tools_code='''"""Timezone converter — common TZ names to UTC offsets."""
from datetime import datetime, timedelta
from typing import Dict, Any
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus
status = SkillStatus("timezone-converter")

_TZ = {
    "UTC": 0, "GMT": 0, "EST": -5, "EDT": -4, "CST": -6, "CDT": -5,
    "MST": -7, "MDT": -6, "PST": -8, "PDT": -7, "AKST": -9, "AKDT": -8,
    "HST": -10, "IST": 5.5, "JST": 9, "KST": 9, "CST_CN": 8, "SGT": 8,
    "HKT": 8, "AEST": 10, "AEDT": 11, "NZST": 12, "NZDT": 13,
    "CET": 1, "CEST": 2, "EET": 2, "EEST": 3, "WET": 0, "WEST": 1,
    "BRT": -3, "ART": -3, "GST": 4, "PKT": 5, "NPT": 5.75,
    "ICT": 7, "WIB": 7, "WITA": 8, "WIT": 9,
}

_FORMATS = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M", "%m/%d/%Y %H:%M", "%d/%m/%Y %H:%M",
            "%Y-%m-%d", "%H:%M"]


def _parse_dt(s: str) -> datetime | None:
    for fmt in _FORMATS:
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None


def _get_offset(tz: str) -> float | None:
    up = tz.upper().strip()
    if up in _TZ:
        return _TZ[up]
    try:
        return float(tz)
    except ValueError:
        return None


@tool_wrapper(required_params=["datetime_str", "from_tz", "to_tz"])
def timezone_convert_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert datetime between timezones."""
    status.set_callback(params.pop("_status_callback", None))
    dt = _parse_dt(params["datetime_str"])
    if dt is None:
        return tool_error(f"Cannot parse datetime: {params[\'datetime_str\']}. Use YYYY-MM-DD HH:MM format.")
    fo = _get_offset(params["from_tz"])
    to = _get_offset(params["to_tz"])
    if fo is None:
        return tool_error(f"Unknown timezone: {params[\'from_tz\']}. Supported: {', '.join(sorted(_TZ))}")
    if to is None:
        return tool_error(f"Unknown timezone: {params[\'to_tz\']}. Supported: {', '.join(sorted(_TZ))}")
    utc_dt = dt - timedelta(hours=fo)
    result_dt = utc_dt + timedelta(hours=to)
    return tool_response(
        original=dt.strftime("%Y-%m-%d %H:%M:%S"),
        converted=result_dt.strftime("%Y-%m-%d %H:%M:%S"),
        from_tz=params["from_tz"].upper(), to_tz=params["to_tz"].upper(),
        utc=utc_dt.strftime("%Y-%m-%d %H:%M:%S"),
        offset_diff=to - fo,
    )


__all__ = ["timezone_convert_tool"]
''',
)

# ---------------------------------------------------------------------------
# 8. countdown-timer
# ---------------------------------------------------------------------------
create_skill(
    name="countdown-timer",
    frontmatter_name="countdown-timer",
    description="Calculate days, hours, and minutes until a target date or event.",
    category="utility",
    capabilities=["Countdown to date", "Holiday countdowns", "Deadline tracking"],
    triggers=["countdown to", "days until", "how long until"],
    eval_tool="countdown_tool",
    eval_input={"target_date": "2025-12-25"},
    tool_docs="### countdown_tool\nCalculate time remaining until a target date.",
    tools_code='''"""Countdown timer — days/hours/minutes to a target date."""
from datetime import datetime, timedelta
from typing import Dict, Any
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus
status = SkillStatus("countdown-timer")

_HOLIDAYS = {
    "new year": "01-01", "valentine": "02-14", "st patrick": "03-17",
    "easter": "04-20", "mother": "05-11", "father": "06-15",
    "independence day": "07-04", "halloween": "10-31",
    "thanksgiving": "11-27", "christmas": "12-25", "new year eve": "12-31",
}
_FORMATS = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
            "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S"]


def _parse_target(s: str) -> datetime | None:
    low = s.lower().strip()
    for name, md in _HOLIDAYS.items():
        if name in low:
            now = datetime.now()
            dt = datetime.strptime(f"{now.year}-{md}", "%Y-%m-%d")
            if dt < now:
                dt = dt.replace(year=now.year + 1)
            return dt
    for fmt in _FORMATS:
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None


@tool_wrapper(required_params=["target_date"])
def countdown_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate time remaining until target date."""
    status.set_callback(params.pop("_status_callback", None))
    target = _parse_target(params["target_date"])
    if target is None:
        return tool_error(f"Cannot parse date: {params[\'target_date\']}. Use YYYY-MM-DD or a holiday name.")
    now = datetime.now()
    delta = target - now
    total_sec = int(delta.total_seconds())
    is_past = total_sec < 0
    total_sec = abs(total_sec)
    days = total_sec // 86400
    hours = (total_sec % 86400) // 3600
    minutes = (total_sec % 3600) // 60
    seconds = total_sec % 60
    weeks = days // 7
    return tool_response(
        target=target.strftime("%Y-%m-%d %H:%M:%S"),
        is_past=is_past,
        total_days=days, weeks=weeks, hours=hours, minutes=minutes, seconds=seconds,
        human=f"{'Passed ' if is_past else ''}{days}d {hours}h {minutes}m {seconds}s{'ago' if is_past else ''}",
    )


__all__ = ["countdown_tool"]
''',
)

# ---------------------------------------------------------------------------
# 9. word-frequency-analyzer
# ---------------------------------------------------------------------------
create_skill(
    name="word-frequency-analyzer",
    frontmatter_name="word-frequency-analyzer",
    description="Count word frequencies, find most common words, generate word cloud data. Excludes stop words.",
    category="text-analysis",
    capabilities=["Word frequency counting", "Stop word filtering", "Word cloud data generation"],
    triggers=["word frequency", "most common words", "word cloud", "text frequency analysis"],
    eval_tool="word_freq_tool",
    eval_input={"text": "the cat sat on the mat the cat"},
    tool_docs="### word_freq_tool\nAnalyze word frequencies in text.",
    tools_code='''"""Word frequency analyzer — count, rank, cloud data."""
import re
from typing import Dict, Any
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus
status = SkillStatus("word-frequency-analyzer")

_STOP = frozenset(
    "a an the and or but in on at to for of is it its this that was were be "
    "been being have has had do does did will would shall should may might can "
    "could am are not no nor so if then than too very just about above after "
    "again all also any because before between both by during each few from "
    "further get got he her here hers herself him himself his how i into me "
    "more most my myself off once only other our ours ourselves out over own "
    "same she some such them themselves there these they those through under "
    "until up us we what when where which while who whom why with you your "
    "yours yourself yourselves".split()
)


@tool_wrapper(required_params=["text"])
def word_freq_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Count word frequencies excluding stop words."""
    status.set_callback(params.pop("_status_callback", None))
    text = params["text"]
    top_n = int(params.get("top_n", 20))
    include_stop = params.get("include_stop_words", False)
    words = re.findall(r"[a-zA-Z\\']+", text.lower())
    total = len(words)
    freq: dict = {}
    for w in words:
        if not include_stop and w in _STOP:
            continue
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    top = ranked[:top_n]
    max_count = top[0][1] if top else 1
    cloud = [{"word": w, "count": c, "weight": round(c / max_count, 3)} for w, c in top]
    return tool_response(
        total_words=total, unique_words=len(freq),
        top_words=[{"word": w, "count": c} for w, c in top],
        word_cloud_data=cloud,
        stop_words_excluded=not include_stop,
    )


__all__ = ["word_freq_tool"]
''',
)

# ---------------------------------------------------------------------------
# 10. palindrome-checker
# ---------------------------------------------------------------------------
create_skill(
    name="palindrome-checker",
    frontmatter_name="palindrome-checker",
    description="Check if text is a palindrome, find palindromic substrings, and generate palindromes.",
    category="text-fun",
    capabilities=["Palindrome check", "Find palindromic substrings", "Palindrome generation"],
    triggers=["is this a palindrome", "palindrome check", "find palindromes"],
    eval_tool="palindrome_tool",
    eval_input={"text": "racecar"},
    tool_docs="### palindrome_tool\nCheck palindromes and find palindromic substrings.",
    tools_code='''"""Palindrome checker — check, find substrings, generate."""
import re
from typing import Dict, Any, List
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus
status = SkillStatus("palindrome-checker")

_KNOWN = ["racecar", "level", "deified", "civic", "rotor", "kayak", "madam",
           "refer", "noon", "radar", "repaper", "rotator", "reviver", "sagas",
           "solos", "stats", "tenet", "wow", "deed", "peep"]


def _clean(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _is_pal(s: str) -> bool:
    return s == s[::-1]


def _find_palindromes(text: str, min_len: int = 2) -> List[str]:
    clean = _clean(text)
    n = len(clean)
    found: set = set()
    for i in range(n):
        # odd-length
        l, r = i, i
        while l >= 0 and r < n and clean[l] == clean[r]:
            if r - l + 1 >= min_len:
                found.add(clean[l:r + 1])
            l -= 1
            r += 1
        # even-length
        l, r = i, i + 1
        while l >= 0 and r < n and clean[l] == clean[r]:
            if r - l + 1 >= min_len:
                found.add(clean[l:r + 1])
            l -= 1
            r += 1
    return sorted(found, key=len, reverse=True)


@tool_wrapper(required_params=["text"])
def palindrome_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Check palindrome, find palindromic substrings."""
    status.set_callback(params.pop("_status_callback", None))
    text = params["text"]
    min_len = int(params.get("min_length", 2))
    clean = _clean(text)
    is_palindrome = _is_pal(clean) if clean else False
    substrings = _find_palindromes(text, min_len)[:20]
    longest = substrings[0] if substrings else ""
    return tool_response(
        text=text, cleaned=clean, is_palindrome=is_palindrome,
        palindromic_substrings=substrings, longest_palindrome=longest,
        substring_count=len(substrings),
        examples=_KNOWN[:5],
    )


__all__ = ["palindrome_tool"]
''',
)

print(f"\nBatch 4b complete — 10 skills generated.")
