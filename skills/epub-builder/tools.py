"""EPUB Builder Skill - build EPUB ebooks from text/markdown."""

import re as _re
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("epub-builder")

_XHTML_TPL = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    "<!DOCTYPE html>\n"
    '<html xmlns="http://www.w3.org/1999/xhtml">\n'
    "<head><title>{title}</title>\n"
    "<style>body{{font-family:serif;line-height:1.6;margin:1em;}}h1{{color:#333;}}</style>\n"
    "</head>\n"
    "<body><h1>{title}</h1>{body}</body></html>"
)

_OPF_TPL = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">\n'
    '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
    '<dc:identifier id="bookid">urn:uuid:{book_id}</dc:identifier>\n'
    "<dc:title>{title}</dc:title>\n"
    "<dc:creator>{author}</dc:creator>\n"
    "<dc:language>{language}</dc:language>\n"
    "<dc:description>{description}</dc:description>\n"
    '<meta property="dcterms:modified">{now}</meta>\n'
    "</metadata>\n"
    "<manifest>\n"
    '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>\n'
    "{manifest}\n"
    "</manifest>\n"
    "<spine>{spine}</spine>\n"
    "</package>"
)

_NAV_TPL = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    "<!DOCTYPE html>\n"
    '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">\n'
    "<head><title>Table of Contents</title></head>\n"
    '<body><nav epub:type="toc"><h1>Table of Contents</h1>\n'
    "<ol>{toc}</ol></nav></body></html>"
)


def _make_xhtml(title: str, body: str) -> str:
    return _XHTML_TPL.format(title=title, body=body)


def _text_to_html(text: str) -> str:
    """Convert plain text to basic HTML paragraphs."""
    paragraphs = text.strip().split("\n\n")
    html_parts = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        m = _re.match(r"^(#{1,6})\s+(.+)$", p)
        if m:
            level = len(m.group(1))
            html_parts.append("<h{0}>{1}</h{0}>".format(level, m.group(2)))
        else:
            lines = p.replace("\n", "<br/>")
            html_parts.append("<p>{}</p>".format(lines))
    return "\n".join(html_parts)


@tool_wrapper(required_params=["title", "author", "chapters"])
def build_epub_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Build an EPUB e-book from chapters."""
    status.set_callback(params.pop("_status_callback", None))
    title = params["title"]
    author = params["author"]
    chapters = params["chapters"]
    language = params.get("language", "en")
    description = params.get("description", "")
    book_id = str(uuid.uuid4())

    if not chapters or not isinstance(chapters, list):
        return tool_error("chapters must be a non-empty list of {title, content} dicts")

    slug = title.lower().replace(" ", "_")[:50]
    output_path = params.get("output_path", slug + ".epub")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as epub:
            epub.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)

            epub.writestr(
                "META-INF/container.xml",
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                '<rootfiles><rootfile full-path="OEBPS/content.opf" '
                'media-type="application/oebps-package+xml"/></rootfiles></container>',
            )

            manifest_items = []
            spine_items = []
            toc_items = []

            for i, ch in enumerate(chapters):
                ch_title = ch.get("title", "Chapter {}".format(i + 1))
                ch_content = ch.get("content", "")
                ch_html = _text_to_html(ch_content)
                ch_id = "chapter{}".format(i + 1)
                ch_file = ch_id + ".xhtml"

                epub.writestr("OEBPS/" + ch_file, _make_xhtml(ch_title, ch_html))
                manifest_items.append(
                    '<item id="{}" href="{}" media-type="application/xhtml+xml"/>'.format(
                        ch_id, ch_file
                    )
                )
                spine_items.append('<itemref idref="{}"/>'.format(ch_id))
                toc_items.append('<li><a href="{}">{}</a></li>'.format(ch_file, ch_title))

            opf = _OPF_TPL.format(
                book_id=book_id,
                title=title,
                author=author,
                language=language,
                description=description,
                now=now,
                manifest="\n".join(manifest_items),
                spine="\n".join(spine_items),
            )
            epub.writestr("OEBPS/content.opf", opf)

            nav = _NAV_TPL.format(toc="\n".join(toc_items))
            epub.writestr("OEBPS/nav.xhtml", nav)

        out = Path(output_path)
        return tool_response(
            output_path=str(out.resolve()),
            chapter_count=len(chapters),
            size_bytes=out.stat().st_size,
            book_id=book_id,
        )
    except Exception as e:
        return tool_error("EPUB creation failed: {}".format(e))


__all__ = ["build_epub_tool"]
