import base64
import html
import re
import urllib.parse
from typing import Any, Dict

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

# Status emitter for progress updates
status = SkillStatus("text-utils")


@tool_wrapper()
def encode_text_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Encode text using various encodings.

    Args:
        params: Dictionary containing:
            - text (str, required): Text to encode
            - encoding (str, required): Encoding type - 'base64', 'url', 'html', 'hex'

    Returns:
        Dictionary with:
            - success (bool): Whether encoding succeeded
            - encoded (str): Encoded text
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop("_status_callback", None))

    try:
        text = params.get("text")
        encoding = params.get("encoding", "").lower()

        if text is None:
            return {"success": False, "error": "text parameter is required"}

        if not encoding:
            return {
                "success": False,
                "error": "encoding parameter is required. Options: base64, url, html, hex",
            }

        text = str(text)

        if encoding == "base64":
            encoded_bytes = text.encode("utf-8")
            encoded = base64.b64encode(encoded_bytes).decode("ascii")
        elif encoding == "url":
            encoded = urllib.parse.quote(text, safe="")
        elif encoding == "html":
            encoded = html.escape(text)
        elif encoding == "hex":
            encoded = text.encode("utf-8").hex()
        else:
            return {
                "success": False,
                "error": f"Unsupported encoding: {encoding}. Supported: base64, url, html, hex",
            }

        return {
            "success": True,
            "encoded": encoded,
            "encoding": encoding,
            "original_length": len(text),
        }
    except Exception as e:
        return {"success": False, "error": f"Error encoding text: {str(e)}"}


@tool_wrapper()
def decode_text_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decode text from various encodings.

    Args:
        params: Dictionary containing:
            - text (str, required): Encoded text to decode
            - encoding (str, required): Encoding type - 'base64', 'url', 'html', 'hex'

    Returns:
        Dictionary with:
            - success (bool): Whether decoding succeeded
            - decoded (str): Decoded text
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop("_status_callback", None))

    try:
        text = params.get("text")
        encoding = params.get("encoding", "").lower()

        if text is None:
            return {"success": False, "error": "text parameter is required"}

        if not encoding:
            return {
                "success": False,
                "error": "encoding parameter is required. Options: base64, url, html, hex",
            }

        text = str(text)

        if encoding == "base64":
            decoded_bytes = base64.b64decode(text)
            decoded = decoded_bytes.decode("utf-8")
        elif encoding == "url":
            decoded = urllib.parse.unquote(text)
        elif encoding == "html":
            decoded = html.unescape(text)
        elif encoding == "hex":
            decoded = bytes.fromhex(text).decode("utf-8")
        else:
            return {
                "success": False,
                "error": f"Unsupported encoding: {encoding}. Supported: base64, url, html, hex",
            }

        return {
            "success": True,
            "decoded": decoded,
            "encoding": encoding,
            "decoded_length": len(decoded),
        }
    except Exception as e:
        return {"success": False, "error": f"Error decoding text: {str(e)}"}


@tool_wrapper()
def format_text_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format text in various ways.

    Args:
        params: Dictionary containing:
            - text (str, required): Text to format
            - format (str, required): Format type - 'upper', 'lower', 'title', 'capitalize', 'swapcase', 'strip', 'reverse'

    Returns:
        Dictionary with:
            - success (bool): Whether formatting succeeded
            - formatted (str): Formatted text
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop("_status_callback", None))

    try:
        text = params.get("text")
        format_type = params.get("format", "").lower()

        if text is None:
            return {"success": False, "error": "text parameter is required"}

        if not format_type:
            return {
                "success": False,
                "error": "format parameter is required. Options: upper, lower, title, capitalize, swapcase, strip, reverse",
            }

        text = str(text)

        if format_type == "upper":
            formatted = text.upper()
        elif format_type == "lower":
            formatted = text.lower()
        elif format_type == "title":
            formatted = text.title()
        elif format_type == "capitalize":
            formatted = text.capitalize()
        elif format_type == "swapcase":
            formatted = text.swapcase()
        elif format_type == "strip":
            formatted = text.strip()
        elif format_type == "reverse":
            formatted = text[::-1]
        else:
            return {
                "success": False,
                "error": f"Unsupported format: {format_type}. Supported: upper, lower, title, capitalize, swapcase, strip, reverse",
            }

        return {
            "success": True,
            "formatted": formatted,
            "format": format_type,
            "original_length": len(text),
            "formatted_length": len(formatted),
        }
    except Exception as e:
        return {"success": False, "error": f"Error formatting text: {str(e)}"}


@tool_wrapper()
def extract_text_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract text from various formats.

    Args:
        params: Dictionary containing:
            - content (str, required): Content to extract text from
            - format (str, required): Format type - 'json', 'html', 'markdown', 'plain'

    Returns:
        Dictionary with:
            - success (bool): Whether extraction succeeded
            - extracted (str): Extracted text
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop("_status_callback", None))

    try:
        content = params.get("content")
        format_type = params.get("format", "").lower()

        if content is None:
            return {"success": False, "error": "content parameter is required"}

        if not format_type:
            return {
                "success": False,
                "error": "format parameter is required. Options: json, html, markdown, plain",
            }

        content = str(content)

        if format_type == "json":
            import json

            try:
                data = json.loads(content)

                # Extract all string values recursively
                def extract_strings(obj):
                    if isinstance(obj, str):
                        return obj
                    elif isinstance(obj, dict):
                        return " ".join(str(v) for v in obj.values() if isinstance(v, str))
                    elif isinstance(obj, list):
                        return " ".join(str(item) for item in obj if isinstance(item, str))
                    return str(obj)

                extracted = extract_strings(data)
            except json.JSONDecodeError:
                return {"success": False, "error": "Invalid JSON format"}
        elif format_type == "html":
            # Remove HTML tags
            extracted = re.sub(r"<[^>]+>", " ", content)
            # Clean up whitespace
            extracted = re.sub(r"\s+", " ", extracted).strip()
            # Decode HTML entities
            extracted = html.unescape(extracted)
        elif format_type == "markdown":
            # Remove markdown syntax (basic)
            extracted = re.sub(r"#{1,6}\s+", "", content)  # Headers
            extracted = re.sub(r"\*\*([^*]+)\*\*", r"\1", extracted)  # Bold
            extracted = re.sub(r"\*([^*]+)\*", r"\1", extracted)  # Italic
            extracted = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", extracted)  # Links
            extracted = re.sub(r"`([^`]+)`", r"\1", extracted)  # Code
            extracted = re.sub(r"\s+", " ", extracted).strip()
        elif format_type == "plain":
            extracted = content
        else:
            return {
                "success": False,
                "error": f"Unsupported format: {format_type}. Supported: json, html, markdown, plain",
            }

        return {
            "success": True,
            "extracted": extracted,
            "format": format_type,
            "length": len(extracted),
        }
    except Exception as e:
        return {"success": False, "error": f"Error extracting text: {str(e)}"}


@tool_wrapper()
def count_text_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Count words, characters, lines, etc. in text.

    Args:
        params: Dictionary containing:
            - text (str, required): Text to analyze
            - count_type (str, optional): What to count - 'words', 'chars', 'lines', 'sentences', 'paragraphs' (default: 'words')

    Returns:
        Dictionary with:
            - success (bool): Whether counting succeeded
            - count (int): Count result
            - count_type (str): Type of count performed
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop("_status_callback", None))

    try:
        text = params.get("text")
        count_type = params.get("count_type", "words").lower()

        if text is None:
            return {"success": False, "error": "text parameter is required"}

        text = str(text)

        if count_type == "words":
            words = text.split()
            count = len([w for w in words if w.strip()])
        elif count_type == "chars":
            count = len(text)
        elif count_type == "chars_no_spaces":
            count = len(text.replace(" ", "").replace("\n", "").replace("\t", ""))
        elif count_type == "lines":
            count = len(text.splitlines())
        elif count_type == "sentences":
            # Simple sentence detection
            sentences = re.split(r"[.!?]+", text)
            count = len([s for s in sentences if s.strip()])
        elif count_type == "paragraphs":
            paragraphs = text.split("\n\n")
            count = len([p for p in paragraphs if p.strip()])
        else:
            return {
                "success": False,
                "error": f"Unsupported count_type: {count_type}. Supported: words, chars, chars_no_spaces, lines, sentences, paragraphs",
            }

        return {"success": True, "count": count, "count_type": count_type, "text_length": len(text)}
    except Exception as e:
        return {"success": False, "error": f"Error counting text: {str(e)}"}


@tool_wrapper()
def replace_text_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find and replace text patterns.

    Args:
        params: Dictionary containing:
            - text (str, required): Text to process
            - find (str, required): Text to find
            - replace (str, required): Replacement text
            - case_sensitive (bool, optional): Case-sensitive matching (default: True)
            - regex (bool, optional): Use regex pattern (default: False)

    Returns:
        Dictionary with:
            - success (bool): Whether replacement succeeded
            - result (str): Text with replacements
            - replacements (int): Number of replacements made
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop("_status_callback", None))

    try:
        text = params.get("text")
        find = params.get("find")
        replace = params.get("replace", "")
        case_sensitive = params.get("case_sensitive", True)
        use_regex = params.get("regex", False)

        if text is None:
            return {"success": False, "error": "text parameter is required"}

        if find is None:
            return {"success": False, "error": "find parameter is required"}

        text = str(text)
        find = str(find)
        replace = str(replace)

        if use_regex:
            try:
                flags = 0 if case_sensitive else re.IGNORECASE
                pattern = re.compile(find, flags)
                result, count = pattern.subn(replace, text)
            except re.error as e:
                return {"success": False, "error": f"Invalid regex pattern: {str(e)}"}
        else:
            if case_sensitive:
                result = text.replace(find, replace)
                count = text.count(find)
            else:
                # Case-insensitive replacement
                import re as re_module

                pattern = re_module.compile(re_module.escape(find), re_module.IGNORECASE)
                result = pattern.sub(replace, text)
                # Count occurrences
                count = len(re_module.findall(pattern, text))

        return {
            "success": True,
            "result": result,
            "replacements": count,
            "original_length": len(text),
            "result_length": len(result),
        }
    except Exception as e:
        return {"success": False, "error": f"Error replacing text: {str(e)}"}
