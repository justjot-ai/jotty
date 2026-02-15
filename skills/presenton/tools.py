"""
Presenton AI Presentation Generator Skill

Integrates with Presenton (https://github.com/presenton/presenton) to generate
AI-powered presentations via Docker container and REST API.
"""

import logging
import os
import subprocess
import time
from typing import Any, Dict, Optional

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import async_tool_wrapper, tool_wrapper

logger = logging.getLogger(__name__)

# Default configuration

# Status emitter for progress updates
status = SkillStatus("presenton")

DEFAULT_PRESENTON_URL = "http://localhost:5000"
DEFAULT_CONTAINER_NAME = "presenton"
DEFAULT_DOCKER_IMAGE = "ghcr.io/presenton/presenton:latest"


class PresentonClient:
    """HTTP client for Presenton API."""

    def __init__(self, base_url: str = DEFAULT_PRESENTON_URL):
        self.base_url = base_url.rstrip("/")

    def _make_request(
        self, method: str, endpoint: str, json_data: Optional[Dict] = None, timeout: int = 300
    ) -> Dict[str, Any]:
        """Make HTTP request to Presenton API."""
        try:
            import requests
        except ImportError:
            return {
                "success": False,
                "error": "requests library not installed. Install with: pip install requests",
            }

        url = f"{self.base_url}{endpoint}"

        try:
            if method.upper() == "GET":
                response = requests.get(url, timeout=timeout)
            elif method.upper() == "POST":
                response = requests.post(url, json=json_data, timeout=timeout)
            else:
                return {"success": False, "error": f"Unsupported HTTP method: {method}"}

            response.raise_for_status()
            return {"success": True, "data": response.json()}

        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": f"Cannot connect to Presenton at {url}. Is the container running?",
            }
        except requests.exceptions.Timeout:
            return {"success": False, "error": f"Request timed out after {timeout} seconds"}
        except requests.exceptions.HTTPError as e:
            return {"success": False, "error": f"HTTP error: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": f"Request failed: {str(e)}"}

    def health_check(self) -> bool:
        """Check if Presenton API is accessible."""
        result = self._make_request("GET", "/api/v1/health", timeout=5)
        if result.get("success"):
            return True
        # Try alternative endpoint
        result = self._make_request("GET", "/", timeout=5)
        return result.get("success", False)

    def generate_presentation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a presentation via API."""
        return self._make_request(
            "POST",
            "/api/v1/ppt/presentation/generate",
            json_data=params,
            timeout=params.get("timeout", 300),
        )

    def get_templates(self) -> Dict[str, Any]:
        """Get available templates."""
        return self._make_request("GET", "/api/v1/ppt/templates")

    def get_presentation(self, presentation_id: str) -> Dict[str, Any]:
        """Get presentation details."""
        return self._make_request("GET", f"/api/v1/ppt/presentation/{presentation_id}")


@tool_wrapper()
def check_presenton_status_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if Presenton Docker container is running and API is accessible.

    Args:
        params: Dictionary containing:
            - container_name (str, optional): Docker container name (default: 'presenton')
            - base_url (str, optional): Presenton API URL (default: 'http://localhost:5000')

    Returns:
        Dictionary with:
            - success (bool): Whether check succeeded
            - container_running (bool): Whether Docker container is running
            - api_accessible (bool): Whether API is responding
            - container_name (str): Container name checked
            - base_url (str): API URL checked
            - error (str, optional): Error message if check failed
    """
    status.set_callback(params.pop("_status_callback", None))

    container_name = params.get("container_name", DEFAULT_CONTAINER_NAME)
    base_url = params.get("base_url", DEFAULT_PRESENTON_URL)

    result = {
        "success": True,
        "container_name": container_name,
        "base_url": base_url,
        "container_running": False,
        "api_accessible": False,
    }

    # Check Docker container status
    try:
        check_cmd = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if check_cmd.returncode == 0 and check_cmd.stdout.strip() == "true":
            result["container_running"] = True
            logger.info(f"Container '{container_name}' is running")
        else:
            logger.info(f"Container '{container_name}' is not running")

    except subprocess.TimeoutExpired:
        result["error"] = "Docker command timed out"
        return result
    except FileNotFoundError:
        result["error"] = "Docker is not installed or not in PATH"
        return result
    except Exception as e:
        result["error"] = f"Failed to check container: {str(e)}"
        return result

    # Check API accessibility
    if result["container_running"]:
        client = PresentonClient(base_url)
        result["api_accessible"] = client.health_check()

        if result["api_accessible"]:
            logger.info(f"Presenton API accessible at {base_url}")
        else:
            logger.warning(f"Container running but API not accessible at {base_url}")

    return result


@tool_wrapper()
def start_presenton_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Start the Presenton Docker container.

    Args:
        params: Dictionary containing:
            - container_name (str, optional): Docker container name (default: 'presenton')
            - port (int, optional): Port to expose (default: 5000)
            - data_dir (str, optional): Directory for app data (default: './presenton_data')
            - image (str, optional): Docker image (default: 'ghcr.io/presenton/presenton:latest')
            - llm (str, optional): LLM provider - 'openai', 'google', 'anthropic', 'ollama' (default: 'anthropic')
            - openai_api_key (str, optional): OpenAI API key
            - google_api_key (str, optional): Google API key
            - anthropic_api_key (str, optional): Anthropic API key
            - image_provider (str, optional): Image provider - 'pexels', 'pixabay', 'dalle3', etc.
            - force_recreate (bool, optional): Remove and recreate container if exists (default: False)

    Returns:
        Dictionary with:
            - success (bool): Whether container started successfully
            - container_name (str): Container name
            - url (str): URL to access Presenton
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop("_status_callback", None))

    container_name = params.get("container_name", DEFAULT_CONTAINER_NAME)
    port = params.get("port", 5000)
    data_dir = params.get("data_dir", os.path.expanduser("~/presenton_data"))
    image = params.get("image", DEFAULT_DOCKER_IMAGE)
    force_recreate = params.get("force_recreate", False)

    # LLM configuration
    llm = params.get("llm", "anthropic")
    openai_api_key = params.get("openai_api_key", os.environ.get("OPENAI_API_KEY", ""))
    google_api_key = params.get("google_api_key", os.environ.get("GOOGLE_API_KEY", ""))
    anthropic_api_key = params.get("anthropic_api_key", os.environ.get("ANTHROPIC_API_KEY", ""))
    image_provider = params.get("image_provider", "pexels")

    base_url = f"http://localhost:{port}"

    try:
        # Check if container already exists
        check_cmd = subprocess.run(
            ["docker", "ps", "-a", "-q", "-f", f"name={container_name}"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        container_exists = bool(check_cmd.stdout.strip())

        if container_exists:
            if force_recreate:
                logger.info(f"Removing existing container '{container_name}'")
                subprocess.run(
                    ["docker", "rm", "-f", container_name], capture_output=True, timeout=30
                )
            else:
                # Try to start existing container
                logger.info(f"Starting existing container '{container_name}'")
                start_cmd = subprocess.run(
                    ["docker", "start", container_name], capture_output=True, text=True, timeout=30
                )

                if start_cmd.returncode == 0:
                    # Wait for API to be ready
                    client = PresentonClient(base_url)
                    for _ in range(30):
                        if client.health_check():
                            return {
                                "success": True,
                                "container_name": container_name,
                                "url": base_url,
                                "message": "Existing container started successfully",
                            }
                        time.sleep(1)

                    return {
                        "success": True,
                        "container_name": container_name,
                        "url": base_url,
                        "warning": "Container started but API may not be ready yet",
                    }
                else:
                    logger.warning(f"Failed to start existing container: {start_cmd.stderr}")
                    # Fall through to create new container
                    subprocess.run(
                        ["docker", "rm", "-f", container_name], capture_output=True, timeout=30
                    )

        # Create data directory
        os.makedirs(data_dir, exist_ok=True)

        # Build docker run command
        docker_cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
            "-p",
            f"{port}:80",
            "-v",
            f"{data_dir}:/app_data",
        ]

        # Add environment variables
        env_vars = {"LLM": llm, "IMAGE_PROVIDER": image_provider, "CAN_CHANGE_KEYS": "false"}

        if openai_api_key:
            env_vars["OPENAI_API_KEY"] = openai_api_key
        if google_api_key:
            env_vars["GOOGLE_API_KEY"] = google_api_key
        if anthropic_api_key:
            env_vars["ANTHROPIC_API_KEY"] = anthropic_api_key

        for key, value in env_vars.items():
            docker_cmd.extend(["-e", f"{key}={value}"])

        docker_cmd.append(image)

        logger.info(f"Starting Presenton container: {container_name}")

        run_cmd = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=120)

        if run_cmd.returncode != 0:
            return {"success": False, "error": f"Failed to start container: {run_cmd.stderr}"}

        # Wait for API to be ready
        logger.info("Waiting for Presenton API to be ready...")
        client = PresentonClient(base_url)

        for i in range(60):  # Wait up to 60 seconds
            if client.health_check():
                logger.info("Presenton API is ready")
                return {
                    "success": True,
                    "container_name": container_name,
                    "url": base_url,
                    "message": "Container started and API is ready",
                }
            time.sleep(1)

        return {
            "success": True,
            "container_name": container_name,
            "url": base_url,
            "warning": "Container started but API may take longer to initialize",
        }

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Docker command timed out"}
    except FileNotFoundError:
        return {"success": False, "error": "Docker is not installed or not in PATH"}
    except Exception as e:
        logger.error(f"Failed to start Presenton: {e}", exc_info=True)
        return {"success": False, "error": f"Failed to start container: {str(e)}"}


@tool_wrapper()
def stop_presenton_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stop the Presenton Docker container.

    Args:
        params: Dictionary containing:
            - container_name (str, optional): Docker container name (default: 'presenton')
            - remove (bool, optional): Remove container after stopping (default: False)

    Returns:
        Dictionary with:
            - success (bool): Whether container stopped successfully
            - container_name (str): Container name
            - removed (bool): Whether container was removed
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop("_status_callback", None))

    container_name = params.get("container_name", DEFAULT_CONTAINER_NAME)
    remove = params.get("remove", False)

    try:
        # Stop container
        stop_cmd = subprocess.run(
            ["docker", "stop", container_name], capture_output=True, text=True, timeout=30
        )

        if stop_cmd.returncode != 0:
            return {"success": False, "error": f"Failed to stop container: {stop_cmd.stderr}"}

        removed = False
        if remove:
            rm_cmd = subprocess.run(
                ["docker", "rm", container_name], capture_output=True, text=True, timeout=30
            )
            removed = rm_cmd.returncode == 0

        return {"success": True, "container_name": container_name, "removed": removed}

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Docker command timed out"}
    except Exception as e:
        return {"success": False, "error": f"Failed to stop container: {str(e)}"}


@tool_wrapper()
def generate_presentation_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate an AI-powered presentation using Presenton.

    Args:
        params: Dictionary containing:
            - content (str, required): Topic or content for the presentation
            - n_slides (int, optional): Number of slides (default: 8)
            - language (str, optional): Target language (default: 'English')
            - template (str, optional): Design template name
            - export_as (str, optional): Export format - 'pptx' or 'pdf' (default: 'pptx')
            - tone (str, optional): Content tone - 'professional', 'casual', 'funny', 'formal', 'inspirational'
            - web_search (bool, optional): Enable web grounding for research (default: False)
            - base_url (str, optional): Presenton API URL (default: 'http://localhost:5000')
            - auto_start (bool, optional): Auto-start container if not running (default: True)
            - timeout (int, optional): Request timeout in seconds (default: 300)

    Returns:
        Dictionary with:
            - success (bool): Whether generation succeeded
            - presentation_id (str): Unique presentation ID
            - file_path (str): Path to generated presentation file
            - edit_url (str): URL to edit presentation in browser
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop("_status_callback", None))

    content = params.get("content")
    if not content:
        return {
            "success": False,
            "error": "content parameter is required - provide the presentation topic",
        }

    base_url = params.get("base_url", DEFAULT_PRESENTON_URL)
    auto_start = params.get("auto_start", True)

    # Check if Presenton is running
    preso_status = check_presenton_status_tool({"base_url": base_url})

    if not preso_status.get("api_accessible"):
        if auto_start:
            logger.info("Presenton not running, attempting to start...")
            start_result = start_presenton_tool(
                {
                    "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
                    "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
                }
            )

            if not start_result.get("success"):
                return {
                    "success": False,
                    "error": f"Failed to start Presenton: {start_result.get('error')}",
                }

            base_url = start_result.get("url", base_url)
        else:
            return {
                "success": False,
                "error": "Presenton is not running. Start it first or set auto_start=True",
            }

    # Build API request
    api_params = {
        "content": content,
        "n_slides": params.get("n_slides", 8),
        "language": params.get("language", "English"),
        "export_as": params.get("export_as", "pptx"),
    }

    if params.get("template"):
        api_params["template"] = params["template"]
    if params.get("tone"):
        api_params["tone"] = params["tone"]
    if params.get("web_search"):
        api_params["web_search"] = params["web_search"]

    # Generate presentation
    logger.info(f"Generating presentation: {content[:50]}...")

    client = PresentonClient(base_url)
    api_params["timeout"] = params.get("timeout", 300)
    result = client.generate_presentation(api_params)

    if not result.get("success"):
        return {"success": False, "error": result.get("error", "Unknown error during generation")}

    data = result.get("data", {})
    presentation_id = data.get("presentation_id", "")
    file_path = data.get("path", "")
    edit_path = data.get("edit_path", "")

    return {
        "success": True,
        "presentation_id": presentation_id,
        "file_path": file_path,
        "edit_url": f"{base_url}{edit_path}" if edit_path else "",
        "download_url": (
            f"{base_url}/api/v1/ppt/presentation/{presentation_id}/download"
            if presentation_id
            else ""
        ),
        "content": content,
        "n_slides": api_params["n_slides"],
        "format": api_params["export_as"],
    }


@tool_wrapper()
def list_templates_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List available presentation templates in Presenton.

    Args:
        params: Dictionary containing:
            - base_url (str, optional): Presenton API URL (default: 'http://localhost:5000')

    Returns:
        Dictionary with:
            - success (bool): Whether request succeeded
            - templates (list): List of available templates
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop("_status_callback", None))

    base_url = params.get("base_url", DEFAULT_PRESENTON_URL)

    client = PresentonClient(base_url)
    result = client.get_templates()

    if not result.get("success"):
        return {"success": False, "error": result.get("error", "Failed to fetch templates")}

    return {"success": True, "templates": result.get("data", {}).get("templates", [])}


@async_tool_wrapper()
async def generate_presentation_from_research_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a presentation from web research on a topic.

    This tool combines web search research with presentation generation
    to create informative, data-backed presentations.

    Args:
        params: Dictionary containing:
            - topic (str, required): Research topic for the presentation
            - n_slides (int, optional): Number of slides (default: 10)
            - language (str, optional): Target language (default: 'English')
            - template (str, optional): Design template name
            - export_as (str, optional): Export format - 'pptx' or 'pdf' (default: 'pptx')
            - tone (str, optional): Content tone (default: 'professional')
            - max_search_results (int, optional): Max search results to use (default: 10)
            - base_url (str, optional): Presenton API URL

    Returns:
        Dictionary with:
            - success (bool): Whether generation succeeded
            - presentation_id (str): Unique presentation ID
            - file_path (str): Path to generated presentation
            - edit_url (str): URL to edit presentation
            - research_summary (str): Summary of research used
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop("_status_callback", None))

    topic = params.get("topic")
    if not topic:
        return {"success": False, "error": "topic parameter is required"}

    try:
        try:
            from Jotty.core.capabilities.registry.skills_registry import get_skills_registry
        except ImportError:
            from Jotty.core.capabilities.registry.skills_registry import get_skills_registry

        registry = get_skills_registry()
        registry.init()

        # Get web search skill
        web_search_skill = registry.get_skill("web-search")
        if not web_search_skill:
            # Fall back to regular generation with web_search enabled
            logger.warning("web-search skill not available, using Presenton's built-in web search")
            return generate_presentation_tool({**params, "content": topic, "web_search": True})

        # Perform research
        search_tool = web_search_skill.tools.get("search_web_tool")
        max_results = params.get("max_search_results", 10)

        logger.info(f"Researching topic: {topic}")

        search_result = search_tool(
            {"query": f"{topic} latest trends statistics facts", "max_results": max_results}
        )

        # Build enhanced content from research
        research_content = f"Create a presentation about: {topic}\n\n"
        research_content += "Key Research Findings:\n"

        if search_result.get("success") and search_result.get("results"):
            for i, result in enumerate(search_result["results"][:max_results], 1):
                research_content += f"\n{i}. {result.get('title', '')}\n"
                research_content += f"   {result.get('snippet', '')}\n"

        research_content += f"\n\nUse these research findings to create an informative, accurate presentation about {topic}."

        # Generate presentation with research
        presentation_result = generate_presentation_tool(
            {
                "content": research_content,
                "n_slides": params.get("n_slides", 10),
                "language": params.get("language", "English"),
                "template": params.get("template"),
                "export_as": params.get("export_as", "pptx"),
                "tone": params.get("tone", "professional"),
                "base_url": params.get("base_url", DEFAULT_PRESENTON_URL),
                "auto_start": params.get("auto_start", True),
            }
        )

        if presentation_result.get("success"):
            presentation_result["research_summary"] = (
                f"Researched {len(search_result.get('results', []))} sources"
            )
            presentation_result["topic"] = topic

        return presentation_result

    except Exception as e:
        logger.error(f"Research presentation generation failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": f"Failed to generate presentation from research: {str(e)}",
        }


@tool_wrapper()
def download_presentation_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Download a generated presentation file.

    Args:
        params: Dictionary containing:
            - presentation_id (str, required): Presentation ID to download
            - output_path (str, optional): Path to save file (default: ~/presentations/)
            - format (str, optional): Download format - 'pptx' or 'pdf' (default: 'pptx')
            - base_url (str, optional): Presenton API URL

    Returns:
        Dictionary with:
            - success (bool): Whether download succeeded
            - file_path (str): Path to downloaded file
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop("_status_callback", None))

    try:
        import requests
    except ImportError:
        return {
            "success": False,
            "error": "requests library not installed. Install with: pip install requests",
        }

    presentation_id = params.get("presentation_id")
    if not presentation_id:
        return {"success": False, "error": "presentation_id parameter is required"}

    base_url = params.get("base_url", DEFAULT_PRESENTON_URL).rstrip("/")
    format_type = params.get("format", "pptx")
    output_dir = params.get("output_path", os.path.expanduser("~/presentations"))

    os.makedirs(output_dir, exist_ok=True)

    try:
        download_url = f"{base_url}/api/v1/ppt/presentation/{presentation_id}/download"

        response = requests.get(download_url, stream=True, timeout=60)
        response.raise_for_status()

        # Determine filename
        filename = f"presentation_{presentation_id}.{format_type}"
        if "content-disposition" in response.headers:
            import re

            match = re.search(r'filename="?([^"]+)"?', response.headers["content-disposition"])
            if match:
                filename = match.group(1)

        file_path = os.path.join(output_dir, filename)

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Presentation downloaded to: {file_path}")

        return {"success": True, "file_path": file_path, "presentation_id": presentation_id}

    except requests.exceptions.HTTPError as e:
        return {"success": False, "error": f"Download failed: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Download failed: {str(e)}"}
