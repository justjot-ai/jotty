"""Generate random test data without external dependencies."""
import random
import string
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("random-data-generator")

_FIRST = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank",
          "Ivy", "Jack", "Kate", "Leo", "Mia", "Noah", "Olivia", "Paul",
          "Quinn", "Rose", "Sam", "Tina", "Uma", "Vince", "Wendy", "Xander", "Yara", "Zane"]
_LAST = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
         "Davis", "Rodriguez", "Martinez", "Wilson", "Anderson", "Thomas", "Lee", "Harris"]
_DOMAINS = ["example.com", "test.org", "demo.net", "sample.io", "mock.dev"]
_STREETS = ["Main St", "Oak Ave", "Pine Rd", "Elm Dr", "Cedar Ln", "Maple Blvd", "Park Way"]
_CITIES = ["Springfield", "Riverside", "Georgetown", "Fairview", "Madison", "Clinton", "Franklin"]
_STATES = ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]


def _person() -> Dict[str, str]:
    first, last = random.choice(_FIRST), random.choice(_LAST)
    email = f"{first.lower()}.{last.lower()}@{random.choice(_DOMAINS)}"
    phone = f"+1-{random.randint(200,999)}-{random.randint(200,999)}-{random.randint(1000,9999)}"
    return {"first_name": first, "last_name": last, "email": email, "phone": phone}


def _email() -> str:
    user = "".join(random.choices(string.ascii_lowercase, k=random.randint(5, 10)))
    return f"{user}@{random.choice(_DOMAINS)}"


def _phone() -> str:
    return f"+1-{random.randint(200,999)}-{random.randint(200,999)}-{random.randint(1000,9999)}"


def _address() -> Dict[str, str]:
    return {"street": f"{random.randint(1,9999)} {random.choice(_STREETS)}",
            "city": random.choice(_CITIES), "state": random.choice(_STATES),
            "zip": f"{random.randint(10000,99999)}"}


@tool_wrapper(required_params=["type"])
def generate_random_data(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate random test data of a given type."""
    status.set_callback(params.pop("_status_callback", None))
    dtype = params["type"].lower()
    count = min(int(params.get("count", 5)), 1000)
    seed = params.get("seed")
    if seed is not None:
        random.seed(int(seed))
    generators = {"person": _person, "email": _email, "phone": _phone, "address": _address}
    gen = generators.get(dtype)
    if not gen:
        return tool_error(f"Unknown type: {dtype}. Use: {', '.join(generators)}")
    items: List = [gen() for _ in range(count)]
    return tool_response(items=items, count=count, type=dtype)


__all__ = ["generate_random_data"]
