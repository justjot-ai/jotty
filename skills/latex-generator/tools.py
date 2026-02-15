"""
LaTeX Document Generator - StatQuest Style

Generates professional LaTeX documents with proper mathematical notation,
styled in the StatQuest format.
"""

import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

# Status emitter for progress updates
status = SkillStatus("latex-generator")


logger = logging.getLogger(__name__)


class LatexGenerator:
    """LaTeX Document Generator in StatQuest Style"""

    def __init__(self, config: Dict[str, Any]):
        self.config = {
            "document_class": "article",
            "paper_size": "a4paper",
            "font_size": "11pt",
            "include_toc": True,
            "date": self._get_current_date(),
            "color_theme": {
                "primary": "46,117,182",
                "secondary": "213,232,240",
            },
            **config,
        }
        self.sections: List[Dict[str, Any]] = []

    def _get_current_date(self) -> str:
        months = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        now = datetime.now()
        return f"{months[now.month - 1]} {now.year}"

    def add_section(self, section: Dict[str, Any]):
        self.sections.append(section)

    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters"""
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "^": r"\textasciicircum{}",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
        }
        result = text
        for char, replacement in replacements.items():
            result = result.replace(char, replacement)
        return result

    def _generate_preamble(self) -> str:
        """Generate LaTeX preamble"""
        config = self.config
        color_theme = config.get("color_theme", {})

        escaped_title = self._escape_latex(self.config.get("title", "Document"))

        preamble = f"""\\documentclass[{config['font_size']},{config['paper_size']}]{{{config['document_class']}}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{amsmath,amssymb,amsthm}}
\\usepackage{{geometry}}
\\usepackage{{graphicx}}
\\usepackage{{xcolor}}
\\usepackage{{hyperref}}
\\usepackage{{enumitem}}
\\usepackage{{tcolorbox}}
\\usepackage{{fancyhdr}}
\\usepackage{{titlesec}}
\\usepackage{{listings}}

\\geometry{{margin=1in}}

% Define colors
\\definecolor{{statquestblue}}{{RGB}}{{{color_theme.get('primary', '46,117,182')}}}
\\definecolor{{lightblue}}{{RGB}}{{{color_theme.get('secondary', '213,232,240')}}}
"""

        if color_theme.get("accent"):
            preamble += f"\\definecolor{{accentcolor}}{{RGB}}{{{color_theme['accent']}}}\n"

        escaped_title = self._escape_latex(self.config.get("title", "Document"))

        # Build preamble with proper escaping - use string replacement instead of format
        preamble_template = """
% Title formatting
\\titleformat{\\section}
  {\\color{statquestblue}\\Large\\bfseries}
  {\\thesection}{1em}{}

\\titleformat{\\subsection}
  {\\color{statquestblue}\\large\\bfseries}
  {\\thesubsection}{1em}{}

\\titleformat{\\subsubsection}
  {\\color{statquestblue}\\normalsize\\bfseries}
  {\\thesubsubsection}{1em}{}

% Custom box for key concepts
\\newtcolorbox{keybox}[1]{{
  colback=lightblue,
  colframe=statquestblue,
  fonttitle=\\bfseries,
  title=#1
}}

% Custom box for examples
\\newtcolorbox{examplebox}[1]{{
  colback=white,
  colframe=statquestblue,
  fonttitle=\\bfseries,
  title=#1
}}

% Header and footer
\\pagestyle{fancy}
\\fancyhf{}
\\rfoot{Page \\thepage}
\\lfoot{ESCAPED_TITLE_PLACEHOLDER}
\\renewcommand{\\headrulewidth}{0pt}

% Code listing style
\\lstset{{
  basicstyle=\\ttfamily\\small,
  breaklines=true,
  frame=single,
  numbers=left,
  numberstyle=\\tiny,
  backgroundcolor=\\color{lightblue!30}
}}
"""
        preamble += preamble_template.replace("ESCAPED_TITLE_PLACEHOLDER", escaped_title)

        return preamble

    def _generate_title(self) -> str:
        """Generate title section"""
        title = self.config.get("title", "Document")
        subtitle = self.config.get("subtitle")
        author = self.config.get("author", "")
        date = self.config.get("date", self._get_current_date())

        escaped_title = self._escape_latex(title)
        # Use string concatenation to avoid f-string brace issues
        title_content = "\\title{\\textcolor{statquestblue}{\\Huge\\textbf{" + escaped_title + "}}}"

        if subtitle:
            escaped_subtitle = self._escape_latex(subtitle)
            title_content += "\\\\[0.5cm]\n\\Large\\textit{" + escaped_subtitle + "}"

        if author:
            escaped_author = self._escape_latex(author)
            title_content += "\n\\author{" + escaped_author + "}"
        else:
            title_content += "\n\\author{}"

        title_content += "\n\\date{" + date + "}"

        return title_content

    def _render_content_block(self, block: Dict[str, Any]) -> str:
        """Render a content block"""
        block_type = block.get("type")

        if block_type == "text":
            return self._render_text(block)
        elif block_type == "equation":
            return self._render_equation(block)
        elif block_type == "list":
            return self._render_list(block)
        elif block_type == "keybox":
            return self._render_keybox(block)
        elif block_type == "example":
            return self._render_example(block)
        elif block_type == "table":
            return self._render_table(block)
        elif block_type == "code":
            return self._render_code(block)
        else:
            return ""

    def _render_text(self, block: Dict[str, Any]) -> str:
        """Render text block"""
        content = self._escape_latex(block.get("content", ""))

        if block.get("bold"):
            content = f"\\textbf{{{content}}}"

        if block.get("italic"):
            content = f"\\textit{{{content}}}"

        spacing = block.get("spacing", {})
        spacing_str = ""
        if spacing:
            before = spacing.get("before", 0)
            after = spacing.get("after", 0)
            if before or after:
                spacing_str = f"[before={before}, after={after}]"

        return f"\\paragraph{{{spacing_str}}} {content}\n\n"

    def _render_equation(self, block: Dict[str, Any]) -> str:
        """Render equation block"""
        latex = block.get("latex", "")
        numbered = block.get("numbered", False)
        label = block.get("label")
        boxed = block.get("boxed", False)

        if boxed:
            latex = f"\\boxed{{{latex}}}"

        if numbered:
            label_str = f"\\label{{{label}}}" if label else ""
            return f"\\begin{{equation}}{label_str}\n{latex}\n\\end{{equation}}\n\n"
        else:
            return f"\\[\n{latex}\n\\]\n\n"

    def _render_list(self, block: Dict[str, Any]) -> str:
        """Render list block"""
        items = block.get("items", [])
        ordered = block.get("ordered", False)
        env = "enumerate" if ordered else "itemize"

        result = f"\\begin{{{env}}}\n"
        for item in items:
            result += f"\\item {self._escape_latex(item)}\n"
        result += f"\\end{{{env}}}\n\n"

        return result

    def _render_keybox(self, block: Dict[str, Any]) -> str:
        """Render key box"""
        title = self._escape_latex(block.get("title", ""))
        content = self._escape_latex(block.get("content", ""))
        return f"\\begin{{keybox}}{{{title}}}\n{content}\n\\end{{keybox}}\n\n"

    def _render_example(self, block: Dict[str, Any]) -> str:
        """Render example box"""
        title = self._escape_latex(block.get("title", ""))
        content = self._escape_latex(block.get("content", ""))
        return f"\\begin{{examplebox}}{{{title}}}\n{content}\n\\end{{examplebox}}\n\n"

    def _render_table(self, block: Dict[str, Any]) -> str:
        """Render table"""
        headers = block.get("headers", [])
        rows = block.get("rows", [])
        caption = block.get("caption")

        num_cols = len(headers)
        col_spec = "l" * num_cols

        result = "\\begin{table}[h]\n\\centering\n"
        result += f"\\begin{{tabular}}{{|{col_spec}|}}\n\\hline\n"

        # Headers
        escaped_headers = [self._escape_latex(h) for h in headers]
        result += " & ".join(escaped_headers) + " \\\\\n\\hline\n"

        # Rows
        for row in rows:
            escaped_row = [self._escape_latex(str(cell)) for cell in row]
            result += " & ".join(escaped_row) + " \\\\\n"

        result += "\\hline\n\\end{tabular}\n"

        if caption:
            result += f"\\caption{{{self._escape_latex(caption)}}}\n"

        result += "\\end{table}\n\n"

        return result

    def _render_code(self, block: Dict[str, Any]) -> str:
        """Render code block"""
        language = block.get("language", "text")
        code = block.get("code", "")
        return f"\\begin{{lstlisting}}[language={language}]\n{code}\n\\end{{lstlisting}}\n\n"

    def _render_section(self, section: Dict[str, Any]) -> str:
        """Render a section"""
        title = self._escape_latex(section.get("title", ""))
        level = section.get("level", 1)
        content_blocks = section.get("content", [])

        section_commands = ["section", "subsection", "subsubsection"]
        command = section_commands[min(level - 1, 2)]

        result = f"\\{command}{{{title}}}\n\n"

        for block in content_blocks:
            result += self._render_content_block(block)

        return result

    def generate(self) -> str:
        """Generate complete LaTeX document"""
        latex = self._generate_preamble()
        latex += "\n\n" + self._generate_title()
        latex += "\n\n\\begin{document}\n\n"
        latex += "\\maketitle\n"
        latex += "\\thispagestyle{empty}\n\n"

        if self.config.get("include_toc"):
            latex += "\\newpage\n\\tableofcontents\n\\newpage\n\n"

        for section in self.sections:
            latex += self._render_section(section)

        latex += "\\end{document}\n"

        return latex

    def compile_to_pdf(self, tex_file: str, output_dir: Optional[str] = None) -> str:
        """Compile LaTeX to PDF"""
        tex_path = Path(tex_file)
        output_dir = Path(output_dir) if output_dir else tex_path.parent
        basename = tex_path.stem
        pdf_file = output_dir / f"{basename}.pdf"

        logger.info(f"Compiling LaTeX to PDF: {tex_file}")

        try:
            # First pass
            subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    f"-output-directory={output_dir}",
                    str(tex_path),
                ],
                check=True,
                capture_output=True,
            )

            # Second pass for references and TOC
            subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    f"-output-directory={output_dir}",
                    str(tex_path),
                ],
                check=True,
                capture_output=True,
            )

            # Clean up auxiliary files
            aux_extensions = [".aux", ".log", ".out", ".toc"]
            for ext in aux_extensions:
                aux_file = output_dir / f"{basename}{ext}"
                if aux_file.exists():
                    aux_file.unlink()

            logger.info(f"PDF generated: {pdf_file}")
            return str(pdf_file)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error compiling LaTeX: {e}")
            raise
        except FileNotFoundError:
            logger.error("pdflatex not found. Please install LaTeX distribution (e.g., texlive)")
            raise


@tool_wrapper()
def generate_latex_document_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a LaTeX document from structured content.

    Args:
        params: Dictionary containing:
            - title (str, required): Document title
            - subtitle (str, optional): Document subtitle
            - author (str, optional): Document author
            - sections (list, required): List of section dictionaries
            - output_file (str, optional): Output .tex filename
            - output_dir (str, optional): Output directory (default: current directory)
            - compile_pdf (bool, optional): Whether to compile to PDF (default: True)
            - include_toc (bool, optional): Include table of contents (default: True)
            - color_theme (dict, optional): Custom color theme

    Returns:
        Dictionary with:
            - success (bool): Whether generation succeeded
            - tex_file (str): Path to generated .tex file
            - pdf_file (str, optional): Path to generated PDF if compiled
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop("_status_callback", None))

    try:
        title = params.get("title")
        if not title:
            return {"success": False, "error": "title parameter is required"}

        sections = params.get("sections", [])
        if not sections:
            return {"success": False, "error": "sections parameter is required"}

        # Create generator
        config = {
            "title": title,
            "subtitle": params.get("subtitle"),
            "author": params.get("author"),
            "include_toc": params.get("include_toc", True),
            "color_theme": params.get("color_theme", {}),
        }

        generator = LatexGenerator(config)

        # Add sections
        for section in sections:
            generator.add_section(section)

        # Generate LaTeX
        latex_content = generator.generate()

        # Determine output file
        output_dir = Path(params.get("output_dir", "."))
        output_dir.mkdir(parents=True, exist_ok=True)

        if params.get("output_file"):
            tex_file = output_dir / params["output_file"]
        else:
            # Auto-generate filename from title
            safe_title = "".join(c if c.isalnum() or c in (" ", "-", "_") else "" for c in title)
            safe_title = safe_title.replace(" ", "_").lower()[:50]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tex_file = output_dir / f"{safe_title}_{timestamp}.tex"

        # Save .tex file
        tex_file.write_text(latex_content, encoding="utf-8")
        logger.info(f"LaTeX file saved: {tex_file}")

        result = {"success": True, "tex_file": str(tex_file)}

        # Compile to PDF if requested
        compile_pdf = params.get("compile_pdf", True)
        if compile_pdf:
            try:
                pdf_file = generator.compile_to_pdf(str(tex_file), str(output_dir))
                result["pdf_file"] = pdf_file
            except Exception as e:
                logger.warning(f"PDF compilation failed: {e}")
                result["pdf_error"] = str(e)
                result["pdf_file"] = None

        return result

    except Exception as e:
        logger.error(f"Error generating LaTeX document: {e}", exc_info=True)
        return {"success": False, "error": f"Failed to generate LaTeX document: {str(e)}"}


__all__ = ["generate_latex_document_tool", "LatexGenerator"]
