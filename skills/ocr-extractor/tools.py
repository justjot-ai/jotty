"""OCR Extractor Skill - extract text from images.

NOTE: Requires external dependencies:
  System: tesseract-ocr (apt install tesseract-ocr / brew install tesseract)
  Python: pytesseract, Pillow (pip install pytesseract Pillow)
"""
from pathlib import Path
from typing import Dict, Any
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

status = SkillStatus("ocr-extractor")


@tool_wrapper(required_params=["image_path"])
def ocr_extract_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Extract text from an image using OCR (Tesseract)."""
    status.set_callback(params.pop("_status_callback", None))

    image_path = Path(params["image_path"])
    if not image_path.exists():
        return tool_error(f"File not found: {image_path}")

    language = params.get("language", "eng")
    psm = int(params.get("psm", 3))

    try:
        from PIL import Image
    except ImportError:
        return tool_error("Pillow is required. Install with: pip install Pillow")

    try:
        import pytesseract
    except ImportError:
        return tool_error(
            "pytesseract is required. Install with: pip install pytesseract\n"
            "Also requires Tesseract OCR engine: sudo apt install tesseract-ocr"
        )

    try:
        img = Image.open(image_path)
        custom_config = f"--psm {psm}"
        text = pytesseract.image_to_string(img, lang=language, config=custom_config)

        # Get confidence data
        try:
            data = pytesseract.image_to_data(img, lang=language, output_type=pytesseract.Output.DICT,
                                              config=custom_config)
            confidences = [int(c) for c in data.get("conf", []) if str(c).isdigit() and int(c) > 0]
            avg_confidence = round(sum(confidences) / len(confidences), 1) if confidences else 0.0
        except Exception:
            avg_confidence = 0.0

        cleaned = text.strip()
        word_count = len(cleaned.split()) if cleaned else 0

        return tool_response(
            text=cleaned, confidence=avg_confidence,
            word_count=word_count, language=language,
            image_path=str(image_path),
        )
    except Exception as e:
        return tool_error(f"OCR failed: {e}")


__all__ = ["ocr_extract_tool"]
