"""Password Generator Skill — secure passwords and passphrases."""
import secrets
import string
import math
from typing import Dict, Any, List

from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

status = SkillStatus("password-generator")

WORD_LIST = [
    "abandon", "ability", "able", "about", "above", "absent", "absorb", "abstract",
    "absurd", "abuse", "access", "accident", "account", "accuse", "achieve", "acid",
    "acquire", "across", "action", "actor", "actress", "actual", "adapt", "address",
    "adjust", "admit", "adult", "advance", "advice", "afford", "again", "agent",
    "agree", "ahead", "alarm", "album", "alert", "alien", "alley", "allow", "almost",
    "alone", "alpha", "already", "also", "alter", "always", "amateur", "amazing",
    "among", "amount", "anchor", "ancient", "anger", "angle", "animal", "ankle",
    "annual", "answer", "antenna", "apple", "armor", "army", "arrive", "arrow",
    "basket", "battle", "beach", "beauty", "become", "before", "begin", "behind",
    "believe", "below", "bench", "benefit", "beyond", "bicycle", "blanket", "blast",
    "blossom", "board", "bonus", "border", "bottle", "bounce", "brave", "breeze",
    "bridge", "bright", "broken", "bronze", "bubble", "budget", "buffalo", "burden",
    "cabin", "cable", "camera", "canal", "canyon", "carbon", "cargo", "carpet",
    "castle", "catalog", "cattle", "caught", "cause", "ceiling", "cement", "census",
    "certain", "chair", "change", "chapter", "charge", "cherry", "chicken", "choice",
    "circle", "citizen", "civil", "claim", "clap", "clarify", "claw", "click",
    "climb", "clinic", "clock", "close", "cluster", "coach", "coconut", "coffee",
    "collect", "column", "combine", "comfort", "common", "company", "concert",
    "connect", "consider", "control", "convince", "copper", "coral", "core",
    "correct", "cotton", "country", "couple", "course", "cousin", "cover", "craft",
    "cream", "credit", "cricket", "cross", "crowd", "cruel", "cruise", "crystal",
    "custom", "cycle", "damage", "dance", "danger", "daring", "dawn", "debate",
    "decade", "decline", "define", "demand", "depart", "depend", "deposit", "depth",
    "derive", "desert", "design", "detail", "detect", "develop", "device", "devote",
    "diamond", "diary", "diesel", "differ", "digital", "dinner", "dinosaur",
    "direct", "dismiss", "disorder", "display", "distance", "divide", "dolphin",
    "domain", "donkey", "donor", "dragon", "drama", "dream", "drift", "drink",
    "driver", "during", "dutch", "dwarf", "dynamic", "eager", "eagle", "early",
    "earth", "easily", "ecology", "economy", "educate", "effort", "eight", "either",
    "elbow", "elder", "electric", "elegant", "element", "elephant", "elevator",
    "elite", "embark", "embody", "embrace", "emerge", "emotion", "employ", "empower",
    "enable", "endorse", "energy", "enforce", "engage", "engine", "enhance", "enjoy",
    "enrich", "ensure", "enter", "entire", "entry", "envelope", "episode", "equal",
    "equip", "erode", "escape", "essence", "estate", "eternal", "evidence", "evolve",
    "exact", "example", "excess", "exchange", "excite", "exclude", "excuse",
    "execute", "exercise", "exhaust", "exhibit", "exile", "expand", "expect",
    "expire", "explain", "expose", "extend", "extra", "fabric", "faculty", "faint",
    "falcon", "family", "famous", "fancy", "fantasy", "fashion", "father", "fault",
    "favorite", "feature", "federal", "fence", "festival", "fiction", "field",
    "figure", "filter", "final", "finger", "finish", "fitness", "flame", "flash",
    "flavor", "flight", "float", "flock", "floor", "flower", "fluid", "focus",
    "follow", "force", "forest", "forget", "formal", "fortune", "forum", "forward",
    "fossil", "foster", "found", "fragile", "frame", "freedom", "frequent", "fresh",
    "friend", "fringe", "frost", "frozen", "fruit", "fuel", "furnace", "future",
    "gadget", "galaxy", "gallery", "garden", "garlic", "gather", "gauge", "general",
    "gentle", "genius", "genre", "gesture", "giant", "ginger", "giraffe", "glacier",
    "glance", "glimpse", "globe", "gloom", "glory", "glove", "glow", "goddess",
    "golden", "gospel", "govern", "grace", "grain", "grant", "gravity", "grocery",
    "group", "growth", "guard", "guitar", "habit", "hammer", "hamster", "harbor",
    "harvest", "hawk", "hazard", "health", "heart", "heaven", "heavy", "hedgehog",
    "height", "helmet", "hidden", "highway", "history", "hobby", "hockey", "holiday",
    "hollow", "honey", "horizon", "horror", "hospital", "host", "hotel", "hover",
    "humble", "humor", "hundred", "hungry", "hurdle", "hybrid",
]


def _estimate_strength(length: int, charset_size: int) -> str:
    bits = length * math.log2(charset_size) if charset_size > 0 else 0
    if bits >= 128:
        return "very_strong"
    elif bits >= 80:
        return "strong"
    elif bits >= 60:
        return "moderate"
    else:
        return "weak"


@tool_wrapper()
def generate_password_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate secure random passwords."""
    status.set_callback(params.pop("_status_callback", None))
    length = min(max(int(params.get("length", 16)), 4), 128)
    count = min(max(int(params.get("count", 1)), 1), 20)
    inc_upper = params.get("uppercase", True)
    inc_lower = params.get("lowercase", True)
    inc_digits = params.get("digits", True)
    inc_symbols = params.get("symbols", True)

    charset = ""
    if inc_lower:
        charset += string.ascii_lowercase
    if inc_upper:
        charset += string.ascii_uppercase
    if inc_digits:
        charset += string.digits
    if inc_symbols:
        charset += "!@#$%^&*()-_=+[]{}|;:,.<>?"

    if not charset:
        return tool_error("At least one character class must be enabled")

    passwords = ["".join(secrets.choice(charset) for _ in range(length)) for _ in range(count)]
    strength = _estimate_strength(length, len(charset))

    return tool_response(passwords=passwords, strength=strength, length=length,
                         charset_size=len(charset))


@tool_wrapper()
def generate_passphrase_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a passphrase from random words."""
    status.set_callback(params.pop("_status_callback", None))
    word_count = min(max(int(params.get("words", 5)), 3), 12)
    separator = params.get("separator", "-")
    capitalize = params.get("capitalize", True)

    words = [secrets.choice(WORD_LIST) for _ in range(word_count)]
    if capitalize:
        words = [w.capitalize() for w in words]

    passphrase = separator.join(words)
    bits = word_count * math.log2(len(WORD_LIST))

    return tool_response(passphrase=passphrase, word_count=word_count,
                         entropy_bits=round(bits, 1),
                         strength=_estimate_strength(word_count, len(WORD_LIST)))


__all__ = ["generate_password_tool", "generate_passphrase_tool"]
