"""
UnifiedToolGenerator - Auto-Generate Claude Tools from Multiple Sources
========================================================================

Generates Claude-native tool definitions from:
1. Input tools (web_search, file_read, fetch_url)
2. Output tools (save_docx, save_pdf, save_slides, telegram)
3. Visualization tools (from section schemas - 70+ types)
4. Skills tools (from SkillsRegistry - dynamic)

This enables a single LLM call where Claude decides which tools to use,
eliminating the need for DSPy signatures or hardcoded decision logic.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Get output directory from environment or use sensible defaults
# Priority: JOTTY_OUTPUT_DIR > /data/outputs (Docker) > ~/jotty/outputs (local)
def _get_output_dir() -> Path:
    """Get the output directory for generated files."""
    # Check environment variable first
    if os.environ.get('JOTTY_OUTPUT_DIR'):
        output_dir = Path(os.environ['JOTTY_OUTPUT_DIR'])
    # Check if /data exists (Docker container)
    elif Path('/data').exists():
        output_dir = Path('/data/outputs')
    # Fall back to home directory
    else:
        output_dir = Path.home() / "jotty" / "outputs"

    # Ensure directory exists
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not create output dir {output_dir}: {e}")
        # Fallback to /tmp if all else fails
        output_dir = Path('/tmp/jotty_outputs')
        output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir

OUTPUT_DIR = _get_output_dir()
logger.info(f"UnifiedToolGenerator output directory: {OUTPUT_DIR}")


@dataclass
class ToolDefinition:
    """Tool definition with metadata."""
    name: str
    description: str
    parameters: Dict[str, Any]
    category: str  # 'input', 'output', 'visualization', 'skill'
    executor: Optional[Callable] = None


class UnifiedToolGenerator:
    """
    Auto-generate Claude tool definitions from multiple sources.

    Combines input, output, visualization, and skills tools into
    a unified set that Claude can choose from with tool_choice="auto".
    """

    def __init__(
        self,
        include_input_tools: bool = True,
        include_output_tools: bool = True,
        include_visualization_tools: bool = True,
        include_skills_tools: bool = True,
        skills_filter: Optional[List[str]] = None
    ):
        """
        Initialize tool generator.

        Args:
            include_input_tools: Include web_search, file_read, etc.
            include_output_tools: Include save_docx, save_pdf, etc.
            include_visualization_tools: Include A2UI section tools
            include_skills_tools: Include tools from SkillsRegistry
            skills_filter: Only include these skill names (None = all)
        """
        self.include_input_tools = include_input_tools
        self.include_output_tools = include_output_tools
        self.include_visualization_tools = include_visualization_tools
        self.include_skills_tools = include_skills_tools
        self.skills_filter = skills_filter

        # Cache for generated tools
        self._tools_cache: Optional[List[Dict]] = None
        self._executors: Dict[str, Callable] = {}

        # Lazy-load registries
        self._schema_registry = None
        self._skills_registry = None

    @property
    def schema_registry(self):
        """Lazy-load schema registry."""
        if self._schema_registry is None:
            try:
                from Jotty.core.ui.schema_validator import schema_registry
                self._schema_registry = schema_registry
            except ImportError:
                logger.warning("Schema registry not available")
        return self._schema_registry

    @property
    def skills_registry(self):
        """Lazy-load skills registry."""
        if self._skills_registry is None:
            try:
                from Jotty.core.registry.skills_registry import get_skills_registry
                self._skills_registry = get_skills_registry()
                self._skills_registry.init()
            except ImportError:
                logger.warning("Skills registry not available")
        return self._skills_registry

    def generate_all_tools(self, force_refresh: bool = False) -> List[Dict]:
        """
        Generate all available tools in Claude format.

        Args:
            force_refresh: Bypass cache and regenerate

        Returns:
            List of tool definitions in Anthropic format
        """
        if self._tools_cache is not None and not force_refresh:
            return self._tools_cache

        tools = []

        # 1. Input tools (hardcoded, well-defined)
        if self.include_input_tools:
            tools.extend(self._generate_input_tools())

        # 2. Output tools (hardcoded, well-defined)
        if self.include_output_tools:
            tools.extend(self._generate_output_tools())

        # 3. Visualization tools (auto-generated from schemas)
        if self.include_visualization_tools:
            tools.extend(self._generate_visualization_tools())

        # 4. Skills tools (dynamic from SkillsRegistry)
        if self.include_skills_tools:
            tools.extend(self._generate_skills_tools())

        self._tools_cache = tools
        logger.info(f"Generated {len(tools)} unified tools")
        return tools

    def _generate_input_tools(self) -> List[Dict]:
        """Generate input tool definitions."""
        tools = []

        # Web Search Tool
        web_search = {
            "name": "web_search",
            "description": "Search the web for current information. Use when user asks about news, prices, weather, recent events, or anything requiring up-to-date data.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find relevant information"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 10)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        }
        tools.append(web_search)
        self._executors["web_search"] = self._execute_web_search

        # File Read Tool
        file_read = {
            "name": "file_read",
            "description": "Read content from a local file. Use when user wants to analyze, summarize, or work with file content.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file"
                    }
                },
                "required": ["file_path"]
            }
        }
        tools.append(file_read)
        self._executors["file_read"] = self._execute_file_read

        # Fetch URL Tool
        fetch_url = {
            "name": "fetch_url",
            "description": "Fetch and parse a web page. Use when user provides a specific URL to analyze.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to fetch and parse"
                    },
                    "extract_text": {
                        "type": "boolean",
                        "description": "Extract text content only (default: true)",
                        "default": True
                    }
                },
                "required": ["url"]
            }
        }
        tools.append(fetch_url)
        self._executors["fetch_url"] = self._execute_fetch_url

        return tools

    def _generate_output_tools(self) -> List[Dict]:
        """Generate output tool definitions."""
        tools = []

        # Save DOCX Tool
        save_docx = {
            "name": "save_docx",
            "description": "Save content as a Word document (.docx). Use when user explicitly asks to save as Word or document.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Content to save (markdown or plain text)"
                    },
                    "filename": {
                        "type": "string",
                        "description": "Output filename (without extension)"
                    },
                    "title": {
                        "type": "string",
                        "description": "Document title"
                    }
                },
                "required": ["content"]
            }
        }
        tools.append(save_docx)
        self._executors["save_docx"] = self._execute_save_docx

        # Save PDF Tool
        save_pdf = {
            "name": "save_pdf",
            "description": "Save content as a PDF document. Use when user explicitly asks to save as PDF.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Content to save (markdown or plain text)"
                    },
                    "filename": {
                        "type": "string",
                        "description": "Output filename (without extension)"
                    },
                    "title": {
                        "type": "string",
                        "description": "Document title"
                    }
                },
                "required": ["content"]
            }
        }
        tools.append(save_pdf)
        self._executors["save_pdf"] = self._execute_save_pdf

        # Save Slides Tool
        save_slides = {
            "name": "save_slides",
            "description": "Create a PowerPoint presentation. Use when user asks for a presentation or slides.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Presentation title"
                    },
                    "subtitle": {
                        "type": "string",
                        "description": "Presentation subtitle"
                    },
                    "slides": {
                        "type": "array",
                        "description": "Array of slide objects with title and bullets",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "bullets": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            }
                        }
                    },
                    "export_pdf": {
                        "type": "boolean",
                        "description": "Also export as PDF (default: false)",
                        "default": False
                    }
                },
                "required": ["title", "slides"]
            }
        }
        tools.append(save_slides)
        self._executors["save_slides"] = self._execute_save_slides

        # Send Telegram Tool
        send_telegram = {
            "name": "send_telegram",
            "description": "Send content via Telegram. Use when user explicitly asks to send via Telegram.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message content to send"
                    }
                },
                "required": ["message"]
            }
        }
        tools.append(send_telegram)
        self._executors["send_telegram"] = self._execute_send_telegram

        # Save to JustJot Tool
        save_justjot = {
            "name": "save_to_justjot",
            "description": "Save as an idea to JustJot.ai. Use when user asks to save to JustJot.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Idea title"
                    },
                    "content": {
                        "type": "string",
                        "description": "Idea content/description"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorization"
                    }
                },
                "required": ["title", "content"]
            }
        }
        tools.append(save_justjot)
        self._executors["save_to_justjot"] = self._execute_save_justjot

        return tools

    def _generate_visualization_tools(self) -> List[Dict]:
        """Generate visualization tools from section schemas."""
        tools = []

        if not self.schema_registry:
            # Fallback to hardcoded core visualization tools
            return self._generate_fallback_visualization_tools()

        try:
            for section_type in self.schema_registry.list_sections():
                schema = self.schema_registry.get_schema(section_type)
                if not schema:
                    continue

                # Convert section type to tool name (kebab-case to snake_case)
                tool_name = f"return_{section_type.replace('-', '_')}"

                # Sanitize the content schema for use as a property definition
                # Remove meta-keywords that aren't valid in nested schemas
                content_schema = self._sanitize_schema_for_property(
                    schema.get('schema', {"type": "object"})
                )

                # Build tool definition
                tool = {
                    "name": tool_name,
                    "description": schema.get('description', f'Return {section_type} visualization'),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "content": content_schema,
                            "title": {
                                "type": "string",
                                "description": "Title for the section"
                            }
                        },
                        "required": ["content"]
                    }
                }
                tools.append(tool)

                # Create executor for this visualization type
                self._executors[tool_name] = self._create_visualization_executor(section_type)

            logger.debug(f"Generated {len(tools)} visualization tools from schemas")

        except Exception as e:
            logger.warning(f"Failed to generate visualization tools from schemas: {e}")
            return self._generate_fallback_visualization_tools()

        return tools

    def _sanitize_schema_for_property(self, schema: Dict[str, Any], is_properties_dict: bool = False) -> Dict[str, Any]:
        """
        Sanitize a JSON schema for use as a property definition.

        Removes meta-keywords ($schema, $id, etc.) that aren't valid
        when the schema is used as a nested property definition.
        Also converts custom schema extensions to valid JSON Schema 2020-12:
        - "type": "enum" with "values" -> "type": "string" with "enum"
        - removes "optional": true (not valid in JSON Schema)

        This ensures compatibility with Claude's tool calling which
        requires JSON Schema draft 2020-12 compliance.

        Args:
            schema: The schema dict to sanitize
            is_properties_dict: True if this dict is a "properties" mapping (not a schema itself)
        """
        if not isinstance(schema, dict):
            return {"type": "object"}

        # Keywords to remove (not valid in nested property definitions or JSON Schema)
        invalid_keywords = {
            '$schema', '$id', '$ref', '$defs', '$vocabulary',
            '$comment', '$anchor', '$dynamicRef', '$dynamicAnchor',
            'optional',  # Not valid JSON Schema - optionality is via "required"
            'values',    # Custom extension - handled separately for enum conversion
            'contentType',  # Custom extension
            'transforms',   # Custom extension
            'llmHint',      # Custom extension
        }

        # Handle custom "type": "enum" conversion to proper JSON Schema
        # Only if this is a schema object (not a properties dict)
        if not is_properties_dict and schema.get('type') == 'enum' and 'values' in schema:
            return {
                "type": "string",
                "enum": schema['values']
            }

        # Create sanitized copy
        sanitized = {}
        for key, value in schema.items():
            if key in invalid_keywords:
                continue

            # Recursively sanitize nested schemas
            if isinstance(value, dict):
                # The "properties" key contains a dict of property names -> schemas
                # Each value in "properties" is a schema, but the dict itself is not
                if key == 'properties':
                    sanitized[key] = self._sanitize_schema_for_property(value, is_properties_dict=True)
                else:
                    sanitized[key] = self._sanitize_schema_for_property(value, is_properties_dict=False)
            elif isinstance(value, list):
                sanitized[key] = [
                    self._sanitize_schema_for_property(item, is_properties_dict=False) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                sanitized[key] = value

        # Only add "type": "object" if this is an actual schema (not a properties dict)
        # and it's missing a type specifier
        if not is_properties_dict:
            if 'type' not in sanitized and 'anyOf' not in sanitized and 'oneOf' not in sanitized:
                sanitized['type'] = 'object'

        return sanitized

    def _generate_fallback_visualization_tools(self) -> List[Dict]:
        """Generate core visualization tools as fallback."""
        tools = []

        # Return Text (default)
        return_text = {
            "name": "return_text",
            "description": "Return plain text or markdown content. DEFAULT for most responses.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Text or markdown content"
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional title"
                    }
                },
                "required": ["content"]
            }
        }
        tools.append(return_text)
        self._executors["return_text"] = self._create_visualization_executor("text")

        # Return Kanban
        return_kanban = {
            "name": "return_kanban",
            "description": "Return kanban board for task tracking. Use for tasks, todos, project boards.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "object",
                        "description": "Kanban data with columns and items",
                        "properties": {
                            "columns": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "title": {"type": "string"},
                                        "items": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "id": {"type": "string"},
                                                    "title": {"type": "string"},
                                                    "description": {"type": "string"},
                                                    "priority": {"type": "string"}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "title": {"type": "string"}
                },
                "required": ["content"]
            }
        }
        tools.append(return_kanban)
        self._executors["return_kanban"] = self._create_visualization_executor("kanban-board")

        # Return Chart
        return_chart = {
            "name": "return_chart",
            "description": "Return chart visualization. Use for data, metrics, comparisons.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["bar", "line", "pie", "doughnut", "radar", "scatter"],
                                "description": "Chart type"
                            },
                            "data": {
                                "type": "object",
                                "description": "Chart data with labels and datasets"
                            }
                        }
                    },
                    "title": {"type": "string"}
                },
                "required": ["content"]
            }
        }
        tools.append(return_chart)
        self._executors["return_chart"] = self._create_visualization_executor("chart")

        # Return Mermaid
        return_mermaid = {
            "name": "return_mermaid",
            "description": "Return Mermaid diagram. Use for flowcharts, sequences, architectures.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Mermaid diagram syntax"
                    },
                    "title": {"type": "string"}
                },
                "required": ["content"]
            }
        }
        tools.append(return_mermaid)
        self._executors["return_mermaid"] = self._create_visualization_executor("mermaid")

        # Return Data Table
        return_data_table = {
            "name": "return_data_table",
            "description": "Return data table. Use for structured data, CSVs, spreadsheet-like data.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "CSV-formatted data"
                    },
                    "title": {"type": "string"}
                },
                "required": ["content"]
            }
        }
        tools.append(return_data_table)
        self._executors["return_data_table"] = self._create_visualization_executor("data-table")

        # Return File Download - for displaying downloadable files inline
        return_file_download = {
            "name": "return_file_download",
            "description": "Display a downloadable file inline in chat. Use when you have a file to share with the user (PDF, DOCX, PPTX, etc.).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Path or URL to the file"
                    },
                    "filename": {
                        "type": "string",
                        "description": "Display name for the file"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["pdf", "docx", "pptx", "xlsx", "csv", "txt", "md", "html", "zip"],
                        "description": "File format"
                    },
                    "size": {
                        "type": "string",
                        "description": "Human-readable file size (e.g., '2.5 MB')"
                    },
                    "preview": {
                        "type": "boolean",
                        "description": "Show inline preview (default: true for PDF)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Brief description of the file"
                    },
                    "title": {"type": "string"}
                },
                "required": ["url", "filename", "format"]
            }
        }
        tools.append(return_file_download)
        self._executors["return_file_download"] = self._execute_return_file_download

        # Return Image - for displaying images inline
        return_image = {
            "name": "return_image",
            "description": "Display an image inline in chat. Use for generated images, charts, diagrams.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Path or URL to the image"
                    },
                    "alt": {
                        "type": "string",
                        "description": "Alt text for accessibility"
                    },
                    "caption": {
                        "type": "string",
                        "description": "Caption displayed below the image"
                    },
                    "width": {
                        "type": "string",
                        "description": "CSS width (e.g., '100%', '500px')"
                    },
                    "title": {"type": "string"}
                },
                "required": ["url"]
            }
        }
        tools.append(return_image)
        self._executors["return_image"] = self._create_visualization_executor("image")

        return tools

    def _generate_skills_tools(self) -> List[Dict]:
        """Generate tools from SkillsRegistry."""
        tools = []

        if not self.skills_registry:
            logger.debug("Skills registry not available")
            return tools

        try:
            for skill in self.skills_registry.loaded_skills.values():
                # Apply filter if set
                if self.skills_filter and skill.name not in self.skills_filter:
                    continue

                # Create tool for each skill
                tool_name = f"skill_{skill.name.replace('-', '_')}"

                tool = {
                    "name": tool_name,
                    "description": skill.description or f"Execute {skill.name} skill",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "params": {
                                "type": "object",
                                "description": "Parameters for the skill"
                            }
                        }
                    }
                }
                tools.append(tool)

                # Create executor for this skill
                self._executors[tool_name] = self._create_skill_executor(skill.name)

            logger.debug(f"Generated {len(tools)} skill tools")

        except Exception as e:
            logger.warning(f"Failed to generate skill tools: {e}")

        return tools

    def get_executor(self, tool_name: str) -> Optional[Callable]:
        """
        Get executor function for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Callable executor or None
        """
        # Ensure tools are generated
        if self._tools_cache is None:
            self.generate_all_tools()

        return self._executors.get(tool_name)

    def get_tools_by_category(self, category: str) -> List[Dict]:
        """Get tools filtered by category."""
        tools = self.generate_all_tools()

        category_prefixes = {
            'input': ['web_search', 'file_read', 'fetch_url'],
            'output': ['save_', 'send_'],
            'visualization': ['return_'],
            'skill': ['skill_']
        }

        prefixes = category_prefixes.get(category, [])
        return [t for t in tools if any(t['name'].startswith(p) for p in prefixes)]

    # ==========================================================================
    # Tool Executors
    # ==========================================================================

    async def _execute_web_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute web search."""
        if not self.skills_registry:
            return {"success": False, "error": "Skills registry not available"}

        skill = self.skills_registry.get_skill('web-search')
        if not skill:
            return {"success": False, "error": "web-search skill not found"}

        search_tool = skill.tools.get('search_web_tool')
        if not search_tool:
            return {"success": False, "error": "search_web_tool not found"}

        try:
            result = search_tool({
                'query': params.get('query', ''),
                'max_results': params.get('max_results', 10)
            })
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_file_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute file read."""
        file_path = params.get('file_path')
        if not file_path:
            return {"success": False, "error": "file_path required"}

        try:
            with open(file_path, 'r') as f:
                content = f.read()
            return {"success": True, "content": content}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_fetch_url(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute URL fetch."""
        url = params.get('url')
        if not url:
            return {"success": False, "error": "url required"}

        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()

                if params.get('extract_text', True):
                    # Basic text extraction
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(response.text, 'html.parser')
                    text = soup.get_text(separator='\n', strip=True)
                    return {"success": True, "content": text[:50000]}
                else:
                    return {"success": True, "content": response.text[:50000]}
        except ImportError:
            return {"success": False, "error": "httpx or bs4 not installed"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_save_docx(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute save as DOCX and return file-download section for UI display."""
        if not self.skills_registry:
            return {"success": False, "error": "Skills registry not available"}

        skill = self.skills_registry.get_skill('docx-tools')
        if not skill:
            return {"success": False, "error": "docx-tools skill not found"}

        create_tool = skill.tools.get('create_docx_tool')
        if not create_tool:
            return {"success": False, "error": "create_docx_tool not found"}

        try:
            from pathlib import Path
            from datetime import datetime

            output_dir = OUTPUT_DIR

            filename = params.get('filename', f"document_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            display_name = params.get('title', filename)
            output_path = output_dir / f"{filename}.docx"

            result = create_tool({
                'content': params.get('content', ''),
                'output_path': str(output_path),
                'title': display_name
            })

            if result.get('success'):
                file_path = result.get('file_path', str(output_path))
                # Get file size
                try:
                    size_bytes = Path(file_path).stat().st_size
                    size_str = self._format_file_size(size_bytes)
                except Exception:
                    size_str = None

                # Return file-download section for UI display
                return self._create_file_download_section(
                    file_path=file_path,
                    filename=f"{display_name}.docx",
                    format="docx",
                    size=size_str,
                    description=f"Word document created from your content",
                    preview=True  # Enable preview via Google Docs Viewer
                )
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_save_pdf(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute save as PDF and return file-download section for UI display."""
        if not self.skills_registry:
            return {"success": False, "error": "Skills registry not available"}

        skill = self.skills_registry.get_skill('document-converter')
        if not skill:
            return {"success": False, "error": "document-converter skill not found"}

        convert_tool = skill.tools.get('convert_to_pdf_tool')
        if not convert_tool:
            return {"success": False, "error": "convert_to_pdf_tool not found"}

        try:
            from pathlib import Path
            from datetime import datetime

            output_dir = OUTPUT_DIR

            filename = params.get('filename', f"document_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            display_name = params.get('title', filename)

            # First save as markdown
            md_path = output_dir / f"{filename}.md"
            md_path.write_text(params.get('content', ''))

            pdf_path = output_dir / f"{filename}.pdf"
            result = convert_tool({
                'input_file': str(md_path),
                'output_file': str(pdf_path)
            })

            if result.get('success'):
                file_path = result.get('output_path', str(pdf_path))
                # Get file size
                try:
                    size_bytes = Path(file_path).stat().st_size
                    size_str = self._format_file_size(size_bytes)
                except Exception:
                    size_str = None

                # Return file-download section for UI display
                return self._create_file_download_section(
                    file_path=file_path,
                    filename=f"{display_name}.pdf",
                    format="pdf",
                    size=size_str,
                    description=f"PDF document ready for download",
                    preview=True  # Enable PDF preview
                )
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_save_slides(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute save as slides and return file-download section for UI display."""
        if not self.skills_registry:
            return {"success": False, "error": "Skills registry not available"}

        skill = self.skills_registry.get_skill('slide-generator')
        if not skill:
            return {"success": False, "error": "slide-generator skill not found"}

        slides_tool = skill.tools.get('generate_slides_tool')
        if not slides_tool:
            return {"success": False, "error": "generate_slides_tool not found"}

        try:
            from pathlib import Path

            output_dir = OUTPUT_DIR

            display_name = params.get('title', 'Presentation')

            result = slides_tool({
                'title': display_name,
                'subtitle': params.get('subtitle', ''),
                'slides': params.get('slides', []),
                'template': 'dark',
                'output_path': str(output_dir)
            })

            if result.get('success'):
                file_path = result.get('file_path')
                if file_path:
                    # Get file size
                    try:
                        size_bytes = Path(file_path).stat().st_size
                        size_str = self._format_file_size(size_bytes)
                    except Exception:
                        size_str = None

                    # Determine format from file extension
                    ext = Path(file_path).suffix.lower().replace('.', '')
                    fmt = ext if ext in ['pptx', 'pdf'] else 'pptx'

                    # Return file-download section for UI display
                    return self._create_file_download_section(
                        file_path=file_path,
                        filename=f"{display_name}.{fmt}",
                        format=fmt,
                        size=size_str,
                        description=f"PowerPoint presentation with {len(params.get('slides', []))} slides",
                        preview=True  # Enable preview via Google Docs Viewer
                    )
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_send_telegram(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute send via Telegram."""
        if not self.skills_registry:
            return {"success": False, "error": "Skills registry not available"}

        skill = self.skills_registry.get_skill('telegram-sender')
        if not skill:
            return {"success": False, "error": "telegram-sender skill not found"}

        send_tool = skill.tools.get('send_message_tool')
        if not send_tool:
            return {"success": False, "error": "send_message_tool not found"}

        try:
            result = send_tool({
                'message': params.get('message', '')[:4000]
            })
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_save_justjot(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute save to JustJot."""
        if not self.skills_registry:
            return {"success": False, "error": "Skills registry not available"}

        skill = self.skills_registry.get_skill('mcp-justjot')
        if not skill:
            return {"success": False, "error": "JustJot skill not found"}

        create_tool = skill.tools.get('create_idea_tool') or skill.tools.get('create_idea')
        if not create_tool:
            return {"success": False, "error": "create_idea tool not found"}

        try:
            import asyncio
            tool_params = {
                'title': params.get('title', 'Untitled'),
                'description': params.get('content', ''),
                'tags': params.get('tags', ['jotty'])
            }

            if asyncio.iscoroutinefunction(create_tool):
                result = await create_tool(tool_params)
            else:
                result = create_tool(tool_params)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_return_file_download(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute return_file_download tool - displays a file inline in chat."""
        try:
            from Jotty.core.ui import return_file_download
            return return_file_download(
                url=params.get('url', ''),
                filename=params.get('filename', 'file'),
                format=params.get('format', 'pdf'),
                size=params.get('size'),
                preview=params.get('preview', True),
                description=params.get('description'),
                title=params.get('title')
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

    def _create_file_download_section(
        self,
        file_path: str,
        filename: str,
        format: str,
        size: str = None,
        description: str = None,
        preview: bool = False
    ) -> Dict[str, Any]:
        """
        Create a file-download section for displaying downloadable files in UI.

        This is the proper way to return generated documents so they display
        inline in the chat with download buttons.
        """
        try:
            from Jotty.core.ui import return_file_download
            return return_file_download(
                url=file_path,
                filename=filename,
                format=format,
                size=size,
                preview=preview,
                description=description,
                title=f"📄 {filename}"
            )
        except ImportError:
            # Fallback if helper not available
            return {
                "success": True,
                "file_path": file_path,
                "filename": filename,
                "format": format,
                "size": size,
                "section": {
                    "type": "file-download",
                    "content": {
                        "url": file_path,
                        "filename": filename,
                        "format": format,
                        "size": size,
                        "preview": preview,
                        "description": description
                    },
                    "title": f"📄 {filename}"
                }
            }

    def _create_visualization_executor(self, section_type: str) -> Callable:
        """Create executor for visualization tool."""
        async def executor(params: Dict[str, Any]) -> Dict[str, Any]:
            try:
                from Jotty.core.ui import return_section
                return return_section(
                    section_type=section_type,
                    content=params.get('content'),
                    title=params.get('title')
                )
            except Exception as e:
                return {"success": False, "error": str(e)}
        return executor

    def _create_skill_executor(self, skill_name: str) -> Callable:
        """Create executor for skill tool."""
        async def executor(params: Dict[str, Any]) -> Dict[str, Any]:
            if not self.skills_registry:
                return {"success": False, "error": "Skills registry not available"}

            skill = self.skills_registry.get_skill(skill_name)
            if not skill:
                return {"success": False, "error": f"Skill {skill_name} not found"}

            # Get first tool from skill
            tools = skill.tools
            if not tools:
                return {"success": False, "error": f"No tools in skill {skill_name}"}

            tool_name = list(tools.keys())[0]
            tool = tools[tool_name]

            try:
                import asyncio
                tool_params = params.get('params', {})

                if asyncio.iscoroutinefunction(tool):
                    result = await tool(tool_params)
                else:
                    result = tool(tool_params)
                return result
            except Exception as e:
                return {"success": False, "error": str(e)}
        return executor
