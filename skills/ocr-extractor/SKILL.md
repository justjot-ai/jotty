---
name: extracting-ocr-text
description: "Extract text from images using OCR. Requires pytesseract and Tesseract engine. Use when the user wants to OCR, extract text from image, read image text."
---

# Ocr Extractor Skill

Extract text from images using OCR. Requires pytesseract and Tesseract engine. Use when the user wants to OCR, extract text from image, read image text.

## Type
base

## Capabilities
- analyze

## Reference
For detailed tool documentation, see [REFERENCE.md](REFERENCE.md).

## Workflow

```
Task Progress:
- [ ] Step 1: Parse input parameters
- [ ] Step 2: Execute operation
- [ ] Step 3: Return results
```

## Triggers
- "ocr"
- "extract text"
- "image to text"
- "read image"
- "tesseract"

## Category
data-analysis

## Tools

### ocr_extract_tool
Extract text from an image using OCR.

**Parameters:**
- `image_path` (str, required): Path to image file
- `language` (str, optional): OCR language code (default: eng)
- `psm` (int, optional): Page segmentation mode 0-13 (default: 3)

**Returns:**
- `success` (bool)
- `text` (str): Extracted text
- `confidence` (float): Average confidence score
- `word_count` (int): Number of words extracted

**Note:** Requires Tesseract OCR engine installed on the system.
Install: `sudo apt install tesseract-ocr` or `brew install tesseract`
Python: `pip install pytesseract Pillow`

## Dependencies
pytesseract, Pillow, Tesseract-OCR engine (system package)
