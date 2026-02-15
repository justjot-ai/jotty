"""
Visual Inspector Skill
======================

VLM-powered visual state extraction for screenshots, files, presentations, PDFs, and code.
Uses litellm for unified VLM access (Claude Sonnet, GPT-4V, etc.).

"""

import base64
import logging
import os
import subprocess
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from Jotty.core.infrastructure.utils.env_loader import get_env, load_jotty_env
from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

load_jotty_env()

logger = logging.getLogger(__name__)
status = SkillStatus("visual-inspector")

# Storage for screenshots and temp files
SCREENSHOT_PATH = os.path.expanduser("~/jotty/screenshots")


class VLMClient:
    """Vision Language Model client using litellm."""

    def __init__(self):
        self._model = get_env("VLM_MODEL") or "claude-sonnet-4-5-20250929"
        self._api_key = get_env("LITELLM_API_KEY") or get_env("OPENAI_API_KEY") or ""
        self._api_base = get_env("LITELLM_BASE_URL") or ""

    @staticmethod
    def _encode_image(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _get_mime_type(path: str) -> str:
        ext = Path(path).suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        return mime_map.get(ext, "image/png")

    def analyze_image(
        self, image_path: str, prompt: str, detail: str = "high", max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """Call VLM with an image and prompt."""
        try:
            import litellm
        except ImportError:
            return {"status": "error", "error": "litellm not installed. Run: pip install litellm"}

        if not os.path.exists(image_path):
            return {"status": "error", "error": f"Image not found: {image_path}"}

        try:
            b64 = self._encode_image(image_path)
            mime = self._get_mime_type(image_path)

            kwargs = {
                "model": self._model,
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{b64}", "detail": detail},
                            },
                        ],
                    }
                ],
            }
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._api_base:
                kwargs["api_base"] = self._api_base

            response = litellm.completion(**kwargs)
            return {
                "status": "success",
                "analysis": response.choices[0].message.content,
                "model": self._model,
            }
        except Exception as e:
            logger.error(f"VLM call failed: {e}")
            return {"status": "error", "error": str(e), "model": self._model}

    def analyze_text(self, prompt: str, max_tokens: int = 2500) -> Dict[str, Any]:
        """Call VLM with text-only prompt (for code inspection)."""
        try:
            import litellm
        except ImportError:
            return {"status": "error", "error": "litellm not installed"}

        try:
            kwargs = {
                "model": self._model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._api_base:
                kwargs["api_base"] = self._api_base

            response = litellm.completion(**kwargs)
            return {
                "status": "success",
                "analysis": response.choices[0].message.content,
                "model": self._model,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def analyze_batch(
        self, image_paths: List[str], prompt: str, detail: str = "high", max_tokens: int = 2000
    ) -> List[Dict[str, Any]]:
        """Analyze multiple images in parallel."""
        with ThreadPoolExecutor(max_workers=min(len(image_paths), 5)) as executor:
            futures = [
                executor.submit(self.analyze_image, img, prompt, detail, max_tokens)
                for img in image_paths
            ]
            return [f.result() for f in futures]


# Singleton client
_vlm_client = None


def _get_vlm() -> VLMClient:
    global _vlm_client
    if _vlm_client is None:
        _vlm_client = VLMClient()
    return _vlm_client


def _ensure_dirs():
    Path(SCREENSHOT_PATH).mkdir(parents=True, exist_ok=True)


class FileConverter:
    """Converts various file types to inspectable images."""

    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
    TEXT_EXTS = {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".html",
        ".css",
        ".scss",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".xml",
        ".md",
        ".txt",
        ".sh",
        ".bash",
        ".zsh",
        ".rs",
        ".go",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".sql",
        ".r",
        ".lua",
        ".vim",
        ".conf",
        ".ini",
        ".env",
        ".dockerfile",
        ".gitignore",
        ".csv",
        ".log",
    }

    @staticmethod
    def convert(file_path: str, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Convert file to inspectable format."""
        _ensure_dirs()
        if not os.path.exists(file_path):
            return {"status": "error", "error": f"File not found: {file_path}"}

        ext = Path(file_path).suffix.lower()
        if save_path is None:
            save_path = os.path.join(SCREENSHOT_PATH, f"file_{uuid.uuid4().hex[:8]}.png")

        if ext in FileConverter.IMAGE_EXTS:
            return {"status": "success", "path": file_path, "file_type": "image"}

        if ext in FileConverter.TEXT_EXTS or ext == "":
            return FileConverter._render_text(file_path, save_path)

        if ext == ".svg":
            return FileConverter._convert_svg(file_path, save_path)

        if ext == ".pdf":
            return FileConverter._convert_pdf_first_page(file_path, save_path)

        if ext == ".pptx":
            return {"status": "error", "error": "Use inspect_pptx_slides_tool for PPTX files."}

        return {"status": "error", "error": f"Unsupported file type: {ext}"}

    @staticmethod
    def _render_text(file_path: str, save_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, "r", errors="replace") as f:
                lines = f.readlines()[:150]
            numbered = [f"{i:4d} | {line.rstrip(chr(10))}" for i, line in enumerate(lines, 1)]
            content = "\n".join(numbered)
            if len(lines) == 150:
                content += "\n... (truncated at 150 lines)"

            # Try pygmentize for image rendering
            try:
                ext = Path(file_path).suffix.lstrip(".")
                result = subprocess.run(
                    [
                        "pygmentize",
                        "-f",
                        "png",
                        "-l",
                        ext or "text",
                        "-O",
                        "font_size=14,line_numbers=True,style=monokai",
                        "-o",
                        save_path,
                        file_path,
                    ],
                    capture_output=True,
                    timeout=10,
                )
                if result.returncode == 0 and os.path.exists(save_path):
                    return {"status": "success", "path": save_path, "file_type": "code_rendered"}
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

            # Fallback: save as text with line numbers
            text_path = save_path.replace(".png", ".txt")
            header = f"=== FILE: {file_path} ===\n=== Lines: {len(lines)} ===\n\n"
            with open(text_path, "w") as f:
                f.write(header + content)
            return {"status": "success", "path": text_path, "file_type": "text_with_line_numbers"}
        except Exception as e:
            return {"status": "error", "error": f"Failed to render file: {e}"}

    @staticmethod
    def _convert_svg(svg_path: str, save_path: str) -> Dict[str, Any]:
        try:
            result = subprocess.run(
                ["convert", svg_path, save_path], capture_output=True, timeout=10
            )
            if result.returncode == 0 and os.path.exists(save_path):
                return {"status": "success", "path": save_path, "file_type": "svg_converted"}
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return {"status": "error", "error": f"SVG conversion failed. Install ImageMagick."}

    @staticmethod
    def pdf_pages_to_images(pdf_path: str, output_dir: str) -> List[str]:
        """Convert PDF pages to PNG images using pdftoppm or ImageMagick."""
        for cmd, pattern in [
            (
                ["pdftoppm", "-png", "-r", "200", pdf_path, os.path.join(output_dir, "page")],
                "page-*.png",
            ),
            (
                ["convert", "-density", "200", pdf_path, os.path.join(output_dir, "page_%03d.png")],
                "page_*.png",
            ),
        ]:
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=60)
                if result.returncode == 0:
                    images = sorted(Path(output_dir).glob(pattern))
                    if images:
                        return [str(img) for img in images]
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        return []

    @staticmethod
    def _convert_pdf_first_page(pdf_path: str, save_path: str) -> Dict[str, Any]:
        import shutil

        output_dir = os.path.dirname(save_path) or tempfile.gettempdir()
        images = FileConverter.pdf_pages_to_images(pdf_path, output_dir)
        if images:
            shutil.move(images[0], save_path)
            return {
                "status": "success",
                "path": save_path,
                "file_type": "pdf_page",
                "total_pages": len(images) + 1,
            }
        return {
            "status": "error",
            "error": "PDF conversion failed. Install poppler or ImageMagick.",
        }

    @staticmethod
    def pptx_to_images(pptx_path: str) -> List[str]:
        """Convert PPTX slides to images. Tries LibreOffice, then PDF pipeline."""
        _ensure_dirs()
        output_dir = os.path.join(SCREENSHOT_PATH, f"pptx_{uuid.uuid4().hex[:8]}")
        os.makedirs(output_dir, exist_ok=True)

        # Strategy 1: LibreOffice -> PNG
        try:
            result = subprocess.run(
                ["soffice", "--headless", "--convert-to", "png", "--outdir", output_dir, pptx_path],
                capture_output=True,
                timeout=120,
            )
            if result.returncode == 0:
                images = sorted(Path(output_dir).glob("*.png"))
                if images:
                    return [str(img) for img in images]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Strategy 2: LibreOffice -> PDF -> images
        try:
            result = subprocess.run(
                ["soffice", "--headless", "--convert-to", "pdf", "--outdir", output_dir, pptx_path],
                capture_output=True,
                timeout=120,
            )
            if result.returncode == 0:
                pdfs = list(Path(output_dir).glob("*.pdf"))
                if pdfs:
                    images = FileConverter.pdf_pages_to_images(str(pdfs[0]), output_dir)
                    if images:
                        return images
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Strategy 3: Extract embedded media (last resort)
        import zipfile

        images = []
        try:
            with zipfile.ZipFile(pptx_path, "r") as z:
                media_dir = os.path.join(output_dir, "media")
                os.makedirs(media_dir, exist_ok=True)
                for name in z.namelist():
                    if name.startswith("ppt/media/") and any(
                        name.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif"]
                    ):
                        extracted = z.extract(name, media_dir)
                        images.append(extracted)
        except Exception:
            pass
        return images


# =============================================================================
# TOOL FUNCTIONS (Jotty pattern)
# =============================================================================


@tool_wrapper(required_params=["image_path"])
def visual_inspect_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze an image using a Vision Language Model to extract visual state.

    Args:
        params: Dictionary containing:
            - image_path (str, required): Path to image file
            - question (str, optional): Analysis question
            - task_context (str, optional): Current task
            - goal_context (str, optional): Overall goal

    Returns:
        Dictionary with visual_state, model, image_path
    """
    status.set_callback(params.pop("_status_callback", None))
    image_path = params["image_path"]
    question = params.get("question", "Describe the current visual state in detail.")
    task_context = params.get("task_context", "")
    goal_context = params.get("goal_context", "")

    status.emit("Inspecting", f"Analyzing image: {Path(image_path).name}")

    prompt_parts = [
        "You are a Visual State Inspector. Analyze this image and extract detailed state information.",
        f"\nQUESTION: {question}",
    ]
    if task_context:
        prompt_parts.append(f"\nCURRENT TASK: {task_context}")
    if goal_context:
        prompt_parts.append(f"\nOVERALL GOAL: {goal_context}")
    prompt_parts.extend(
        [
            "\nProvide a comprehensive analysis covering:",
            "1. CURRENT STATE: What is visible? What state is the UI/file/content in?",
            "2. ELEMENTS: List key elements (buttons, forms, text, errors, warnings)",
            "3. VISUAL QUALITY: Layout, colors, spacing, alignment, consistency",
            "4. ISSUES: Any errors, misalignment, broken elements, or anomalies",
            "5. ACTIONABLE INSIGHTS: What actions could be taken based on what you see?",
        ]
    )

    result = _get_vlm().analyze_image(
        image_path, "\n".join(prompt_parts), detail="high", max_tokens=2500
    )
    if result["status"] == "success":
        return tool_response(
            visual_state=result["analysis"], model=result.get("model", ""), image_path=image_path
        )
    return tool_error(result.get("error", "VLM analysis failed"))


@tool_wrapper(required_params=["file_path"])
def inspect_file_visually_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Open any file and visually inspect it using VLM.

    Args:
        params: Dictionary containing:
            - file_path (str, required): Path to file
            - question (str, optional): Analysis question
            - task_context (str, optional): Current task context

    Returns:
        Dictionary with visual_state, file_path, file_type
    """
    status.set_callback(params.pop("_status_callback", None))
    file_path = params["file_path"]
    question = params.get(
        "question", "Analyze this file content, formatting, and identify any issues."
    )
    task_context = params.get("task_context", "")

    status.emit("Inspecting", f"Analyzing file: {Path(file_path).name}")

    capture = FileConverter.convert(file_path)
    if capture["status"] != "success":
        return tool_error(capture.get("error", "Failed to convert file"))

    file_type = capture.get("file_type", "unknown")

    # For text files, use text-mode VLM analysis
    if file_type == "text_with_line_numbers":
        with open(capture["path"], "r", errors="replace") as f:
            text_content = f.read()
        enhanced_q = (
            f"{question}\n\nFILE CONTENT:\n```\n{text_content[:8000]}\n```\n\n"
            "Focus on: indentation errors, formatting issues, syntax problems."
        )
        result = _get_vlm().analyze_text(enhanced_q)
        if result["status"] == "success":
            return tool_response(
                visual_state=result["analysis"],
                file_path=file_path,
                file_type=file_type,
                model=result.get("model", ""),
            )
        return tool_error(result.get("error", "VLM analysis failed"))

    # For images, use standard visual inspect
    result = _get_vlm().analyze_image(capture["path"], question, detail="high", max_tokens=2500)
    if result["status"] == "success":
        return tool_response(
            visual_state=result["analysis"],
            file_path=file_path,
            file_type=file_type,
            model=result.get("model", ""),
        )
    return tool_error(result.get("error", "VLM analysis failed"))


@tool_wrapper(required_params=["pptx_path"])
def inspect_pptx_slides_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert PPTX slides to images and analyze ALL slides in parallel with VLM.

    Args:
        params: Dictionary containing:
            - pptx_path (str, required): Path to .pptx file
            - question (str, optional): Analysis question per slide
            - slides (list, optional): Specific slide indices (0-based). None = all.

    Returns:
        Dictionary with summary, slide_analyses, total_slides
    """
    status.set_callback(params.pop("_status_callback", None))
    pptx_path = params["pptx_path"]
    question = params.get("question", "Analyze each slide design, content, and visual quality.")
    slides = params.get("slides")

    if not os.path.exists(pptx_path):
        return tool_error(f"PPTX file not found: {pptx_path}")

    status.emit("Converting", "Converting PPTX slides to images...")
    slide_images = FileConverter.pptx_to_images(pptx_path)

    if not slide_images:
        return tool_error("Could not convert PPTX to images. Install LibreOffice and/or poppler.")

    if slides is not None:
        slide_images = [slide_images[i] for i in slides if i < len(slide_images)]

    status.emit("Analyzing", f"Analyzing {len(slide_images)} slides with VLM...")

    slide_prompt = (
        "You are analyzing slide {num} of a PowerPoint presentation.\n"
        f"{question}\n\n"
        "Report:\n1. CONTENT: Text, data, charts, images\n"
        "2. DESIGN: Colors, fonts, layout, whitespace\n"
        "3. QUALITY: Readability, contrast, consistency\n"
        "4. ISSUES: Problems (text overflow, low contrast, cluttered layout)\n"
        "5. SUGGESTIONS: Specific improvements"
    )

    analyses = []
    with ThreadPoolExecutor(max_workers=min(len(slide_images), 5)) as executor:
        futures = [
            executor.submit(
                _get_vlm().analyze_image,
                img,
                slide_prompt.replace("{num}", str(i + 1)),
                "high",
                1500,
            )
            for i, img in enumerate(slide_images)
        ]
        for i, future in enumerate(futures):
            r = future.result()
            analyses.append(
                {
                    "slide": i + 1,
                    "analysis": r.get("analysis", r.get("error", "Failed")),
                    "image_path": slide_images[i],
                    "status": r["status"],
                }
            )

    successful = [a for a in analyses if a["status"] == "success"]
    summary_parts = [
        f"Analyzed {len(successful)}/{len(slide_images)} slides from {Path(pptx_path).name}:"
    ]
    for a in successful:
        summary_parts.append(f"\n--- Slide {a['slide']} ---\n{a['analysis'][:500]}")

    return tool_response(
        summary="\n".join(summary_parts),
        slide_analyses=analyses,
        total_slides=len(slide_images),
        pptx_path=pptx_path,
    )


@tool_wrapper(required_params=["pdf_path"])
def inspect_pdf_pages_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert PDF pages to images and analyze in parallel with VLM.

    Args:
        params: Dictionary containing:
            - pdf_path (str, required): Path to .pdf file
            - question (str, optional): Analysis question per page
            - pages (list, optional): Specific page indices (0-based). None = all.

    Returns:
        Dictionary with summary, page_analyses, total_pages
    """
    status.set_callback(params.pop("_status_callback", None))
    pdf_path = params["pdf_path"]
    question = params.get(
        "question", "Analyze the content, layout, and visual quality of this page."
    )
    pages = params.get("pages")

    if not os.path.exists(pdf_path):
        return tool_error(f"PDF file not found: {pdf_path}")

    _ensure_dirs()
    output_dir = os.path.join(SCREENSHOT_PATH, f"pdf_{uuid.uuid4().hex[:8]}")
    os.makedirs(output_dir, exist_ok=True)

    status.emit("Converting", "Converting PDF pages to images...")
    page_images = FileConverter.pdf_pages_to_images(pdf_path, output_dir)

    if not page_images:
        return tool_error("PDF conversion failed. Install poppler (pdftoppm) or ImageMagick.")

    if pages is not None:
        page_images = [page_images[i] for i in pages if i < len(page_images)]

    status.emit("Analyzing", f"Analyzing {len(page_images)} pages with VLM...")

    page_prompt = (
        "You are analyzing page {num} of a PDF document.\n"
        f"{question}\n\nReport:\n"
        "1. CONTENT: Text, data, charts, images, headings\n"
        "2. LAYOUT: Margins, columns, alignment, whitespace\n"
        "3. QUALITY: Readability, contrast, visual consistency\n"
        "4. ISSUES: Problems (truncated text, missing content, low contrast)\n"
    )

    analyses = []
    with ThreadPoolExecutor(max_workers=min(len(page_images), 5)) as executor:
        futures = [
            executor.submit(
                _get_vlm().analyze_image,
                img,
                page_prompt.replace("{num}", str(i + 1)),
                "high",
                1500,
            )
            for i, img in enumerate(page_images)
        ]
        for i, future in enumerate(futures):
            r = future.result()
            analyses.append(
                {
                    "page": i + 1,
                    "analysis": r.get("analysis", r.get("error", "Failed")),
                    "image_path": page_images[i],
                    "status": r["status"],
                }
            )

    successful = [a for a in analyses if a["status"] == "success"]
    summary_parts = [
        f"Analyzed {len(successful)}/{len(page_images)} pages from {Path(pdf_path).name}:"
    ]
    for a in successful:
        summary_parts.append(f"\n--- Page {a['page']} ---\n{a['analysis'][:500]}")

    return tool_response(
        summary="\n".join(summary_parts),
        page_analyses=analyses,
        total_pages=len(page_images),
        pdf_path=pdf_path,
    )


@tool_wrapper(required_params=["file_path"])
def inspect_code_for_errors_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inspect code file for indentation, syntax, formatting, and logic errors.

    Args:
        params: Dictionary containing:
            - file_path (str, required): Path to code file
            - error_context (str, optional): Known error messages

    Returns:
        Dictionary with issues, file_path, lines_analyzed, total_lines
    """
    status.set_callback(params.pop("_status_callback", None))
    file_path = params["file_path"]
    error_context = params.get("error_context", "")

    if not os.path.exists(file_path):
        return tool_error(f"File not found: {file_path}")

    status.emit("Inspecting", f"Checking code: {Path(file_path).name}")

    try:
        with open(file_path, "r", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        return tool_error(f"Failed to read file: {e}")

    numbered = ""
    for i, line in enumerate(lines[:300], 1):
        visible = line.rstrip("\n").replace("\t", "    ")
        numbered += f"{i:4d} | {visible}\n"
    if len(lines) > 300:
        numbered += f"\n... (truncated, {len(lines)} total lines)"

    prompt = (
        f"You are a code quality inspector. Analyze this code file for errors.\n\n"
        f"FILE: {file_path}\nLANGUAGE: {Path(file_path).suffix.lstrip('.') or 'unknown'}\n"
    )
    if error_context:
        prompt += f"KNOWN ERROR: {error_context}\n"
    prompt += (
        f"\nCODE (with line numbers):\n```\n{numbered}\n```\n\n"
        "Report ALL issues:\n"
        "1. INDENTATION ERRORS: Inconsistent indentation, mixed tabs/spaces\n"
        "2. SYNTAX ISSUES: Missing brackets, unclosed strings\n"
        "3. FORMATTING: Line length, spacing, organization\n"
        "4. LOGIC: Potential bugs, undefined variables\n"
        "5. STYLE: Naming conventions, unused imports\n\n"
        "For each issue provide: Line number(s), Description, Suggested fix."
    )

    result = _get_vlm().analyze_text(prompt, max_tokens=3000)
    if result["status"] == "success":
        return tool_response(
            issues=result["analysis"],
            file_path=file_path,
            lines_analyzed=min(len(lines), 300),
            total_lines=len(lines),
            model=result.get("model", ""),
        )
    return tool_error(result.get("error", "Code inspection failed"))


@tool_wrapper()
def inspect_browser_state_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Capture browser screenshot and analyze with VLM.

    Args:
        params: Dictionary containing:
            - question (str, optional): What to analyze
            - task_context (str, optional): Current task
            - goal_context (str, optional): Overall goal

    Returns:
        Dictionary with visual_state, screenshot_path
    """
    status.set_callback(params.pop("_status_callback", None))
    question = params.get("question", "What is the current state of the browser page?")
    task_context = params.get("task_context", "")
    goal_context = params.get("goal_context", "")

    _ensure_dirs()
    save_path = os.path.join(SCREENSHOT_PATH, f"browser_{uuid.uuid4().hex[:8]}.png")

    status.emit("Capturing", "Taking browser screenshot...")

    # Try to capture screenshot from browser-automation skill
    screenshot_captured = False
    try:
        from Jotty.skills import get_skill_tools

        browser_tools = get_skill_tools("browser-automation")
        if browser_tools and hasattr(browser_tools, "browser_screenshot_tool"):
            result = browser_tools.browser_screenshot_tool(
                {"output_path": save_path, "full_page": False}
            )
            if result.get("success") and os.path.exists(save_path):
                screenshot_captured = True
    except Exception:
        pass

    if not screenshot_captured:
        return tool_error("No browser available for screenshot. Initialize browser first.")

    # Analyze the screenshot
    inspect_result = visual_inspect_tool(
        {
            "image_path": save_path,
            "question": question,
            "task_context": task_context,
            "goal_context": goal_context,
        }
    )

    if inspect_result.get("success"):
        inspect_result["screenshot_path"] = save_path
    return inspect_result


# =============================================================================
# VISUAL VERIFICATION PROTOCOL
# =============================================================================

VISUAL_VERIFICATION_PROTOCOL = """
WHEN TO VISUALLY VERIFY:
- After state-changing actions (UI changes, file generation, form submissions)
- Before irreversible or high-stakes actions (deleting data, sending messages)
- When stuck or encountering unexpected behavior
- After producing visual output (files, presentations, PDFs, charts)
- When CSS selectors or element locators fail repeatedly

PRINCIPLE: Observe before acting when uncertain. Verify after acting when the action has visual consequences.

HOW TO VERIFY:
1. Take a screenshot of the current state
2. Use visual_inspect_tool or inspect_browser_state_tool to analyze
3. Compare observed state against expected state
4. If mismatch: investigate before retrying
"""


def get_visual_verification_guidance() -> str:
    """Return Visual Verification Protocol guidance for agent system prompts.

    Append this to agent instructions when the agent has access to
    visual inspection tools (browser-automation, visual-inspector).
    """
    return VISUAL_VERIFICATION_PROTOCOL


__all__ = [
    "visual_inspect_tool",
    "inspect_file_visually_tool",
    "inspect_pptx_slides_tool",
    "inspect_pdf_pages_tool",
    "inspect_code_for_errors_tool",
    "inspect_browser_state_tool",
    "get_visual_verification_guidance",
    "VISUAL_VERIFICATION_PROTOCOL",
]
