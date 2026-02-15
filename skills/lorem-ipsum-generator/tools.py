"""Lorem Ipsum Generator Skill."""

import random
from typing import Any, Dict

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_wrapper

status = SkillStatus("lorem-ipsum-generator")

WORDS = [
    "lorem",
    "ipsum",
    "dolor",
    "sit",
    "amet",
    "consectetur",
    "adipiscing",
    "elit",
    "sed",
    "do",
    "eiusmod",
    "tempor",
    "incididunt",
    "ut",
    "labore",
    "et",
    "dolore",
    "magna",
    "aliqua",
    "enim",
    "ad",
    "minim",
    "veniam",
    "quis",
    "nostrud",
    "exercitation",
    "ullamco",
    "laboris",
    "nisi",
    "aliquip",
    "ex",
    "ea",
    "commodo",
    "consequat",
    "duis",
    "aute",
    "irure",
    "in",
    "reprehenderit",
    "voluptate",
    "velit",
    "esse",
    "cillum",
    "fugiat",
    "nulla",
    "pariatur",
    "excepteur",
    "sint",
    "occaecat",
    "cupidatat",
    "non",
    "proident",
    "sunt",
    "culpa",
    "qui",
    "officia",
    "deserunt",
    "mollit",
    "anim",
    "id",
    "est",
    "laborum",
    "perspiciatis",
    "unde",
    "omnis",
    "iste",
    "natus",
    "error",
    "voluptatem",
    "accusantium",
    "doloremque",
    "laudantium",
    "totam",
    "rem",
    "aperiam",
    "eaque",
    "ipsa",
    "quae",
    "ab",
    "illo",
    "inventore",
    "veritatis",
    "quasi",
    "architecto",
    "beatae",
    "vitae",
    "dicta",
    "explicabo",
    "nemo",
    "ipsam",
    "quia",
    "voluptas",
    "aspernatur",
    "aut",
    "odit",
    "fugit",
    "consequuntur",
    "magni",
    "dolores",
    "eos",
    "ratione",
]


def _sentence(min_words: int = 5, max_words: int = 15) -> str:
    n = random.randint(min_words, max_words)
    s = " ".join(random.choice(WORDS) for _ in range(n))
    return s[0].upper() + s[1:] + "."


def _paragraph(min_sentences: int = 3, max_sentences: int = 7) -> str:
    n = random.randint(min_sentences, max_sentences)
    return " ".join(_sentence() for _ in range(n))


@tool_wrapper()
def lorem_ipsum_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate Lorem Ipsum placeholder text."""
    status.set_callback(params.pop("_status_callback", None))
    word_count = params.get("words")
    sent_count = params.get("sentences")
    para_count = params.get("paragraphs", 1)

    if word_count:
        words = [random.choice(WORDS) for _ in range(int(word_count))]
        words[0] = words[0].capitalize()
        text = " ".join(words) + "."
    elif sent_count:
        text = " ".join(_sentence() for _ in range(int(sent_count)))
    else:
        text = "\n\n".join(_paragraph() for _ in range(int(para_count)))

    return tool_response(text=text, word_count=len(text.split()))


__all__ = ["lorem_ipsum_tool"]
