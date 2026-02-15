"""
Content & Branding Pipeline Composite Skill

Complete content creation workflow:
1. Brainstorm domain names (domain-name-brainstormer)
2. Create HTML artifacts (artifacts-builder)
3. Apply brand guidelines (brand-guidelines)
4. Apply theme (theme-factory)

Source → Processor → Processor → Sink pattern.
"""

import asyncio
import inspect
import logging
from pathlib import Path
from typing import Any, Dict

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import (
    async_tool_wrapper,
    tool_error,
    tool_response,
)

# Status emitter for progress updates
status = SkillStatus("content-branding-pipeline")


logger = logging.getLogger(__name__)


@async_tool_wrapper()
async def content_branding_pipeline_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Complete content and branding workflow: domain → artifacts → brand → theme.

    Args:
        params:
            - project_description (str, required): Project description
            - project_name (str, optional): Project name (will brainstorm if not provided)
            - artifact_type (str, optional): 'html', 'presentation', 'document' (default: 'html')
            - include_domain_brainstorm (bool, optional): Brainstorm domains (default: True)
            - create_artifact (bool, optional): Create artifact (default: True)
            - apply_brand (bool, optional): Apply brand guidelines (default: True)
            - apply_theme (bool, optional): Apply theme (default: True)
            - brand_style (str, optional): Brand style (default: 'anthropic')
            - theme_name (str, optional): Theme name (default: 'ocean_depths')
            - max_domain_suggestions (int, optional): Max domain suggestions (default: 10)

    Returns:
        Dictionary with domains, artifact path, brand applied, theme applied
    """
    status.set_callback(params.pop("_status_callback", None))

    try:
        try:
            from Jotty.core.capabilities.registry.skills_registry import get_skills_registry
        except ImportError:
            from Jotty.core.capabilities.registry.skills_registry import get_skills_registry

        registry = get_skills_registry()
        registry.init()

        project_description = params.get("project_description")
        if not project_description:
            return {"success": False, "error": "project_description is required"}

        logger.info(f"🎨 Content & Branding Pipeline: {project_description}")

        domains = []
        artifact_path = None
        brand_applied = False
        theme_applied = False

        # Step 1: Brainstorm domain names (Source)
        if params.get("include_domain_brainstorm", True):
            logger.info("💡 Step 1: Brainstorming domain names...")
            domain_skill = registry.get_skill("domain-name-brainstormer")
            if domain_skill:
                domain_tool = domain_skill.tools.get("brainstorm_domains_tool")
                if domain_tool:
                    max_suggestions = params.get("max_domain_suggestions", 10)
                    domain_params = {
                        "project_description": project_description,
                        "max_suggestions": max_suggestions,
                    }

                    if inspect.iscoroutinefunction(domain_tool):
                        domain_result = await domain_tool(domain_params)
                    else:
                        domain_result = domain_tool(domain_params)

                    if domain_result.get("success"):
                        domains = domain_result.get("suggestions", [])
                        logger.info(f"✅ Generated {len(domains)} domain suggestions")
                    else:
                        logger.warning(
                            f"⚠️  Domain brainstorming failed: {domain_result.get('error')}"
                        )

        project_name = params.get("project_name") or domains[0] if domains else "project"

        # Step 2: Create HTML artifact (Processor)
        if params.get("create_artifact", True):
            logger.info("📄 Step 2: Creating artifact...")
            artifact_skill = registry.get_skill("artifacts-builder")
            if artifact_skill:
                init_tool = artifact_skill.tools.get("init_artifact_project_tool")
                if init_tool:
                    artifact_type = params.get("artifact_type", "html")
                    artifact_params = {
                        "project_name": project_name.lower().replace(" ", "-"),
                        "include_shadcn": artifact_type == "html",
                        "include_tailwind": artifact_type == "html",
                    }

                    if inspect.iscoroutinefunction(init_tool):
                        artifact_result = await init_tool(artifact_params)
                    else:
                        artifact_result = init_tool(artifact_params)

                    if artifact_result.get("success"):
                        artifact_path = artifact_result.get("project_path")
                        logger.info(f"✅ Artifact created: {artifact_path}")
                    else:
                        logger.warning(
                            f"⚠️  Artifact creation failed: {artifact_result.get('error')}"
                        )

        # Step 3: Apply brand guidelines (Processor)
        if params.get("apply_brand", True) and artifact_path:
            logger.info("🎨 Step 3: Applying brand guidelines...")
            brand_skill = registry.get_skill("brand-guidelines")
            if brand_skill:
                brand_tool = brand_skill.tools.get("apply_brand_styling_tool")
                if brand_tool:
                    # Find the actual file and let brand-guidelines auto-detect type
                    artifact_dir = Path(artifact_path) if artifact_path else None
                    brand_file = None

                    if artifact_dir and artifact_dir.exists():
                        # Look for actual files by extension (prefer actual file over param)
                        for ext in ["*.html", "*.htm", "*.docx", "*.pptx"]:
                            files = list(artifact_dir.glob(ext))
                            if files:
                                brand_file = str(files[0])
                                break
                        if not brand_file:
                            brand_file = str(artifact_dir / "index.html")
                    else:
                        brand_file = artifact_path

                    # Don't pass file_type - let brand-guidelines auto-detect from extension
                    brand_params = {
                        "input_file": brand_file,
                        "brand_style": params.get("brand_style", "anthropic"),
                    }

                    if inspect.iscoroutinefunction(brand_tool):
                        brand_result = await brand_tool(brand_params)
                    else:
                        brand_result = brand_tool(brand_params)

                    if brand_result.get("success"):
                        brand_applied = True
                        logger.info("✅ Brand guidelines applied")
                    else:
                        logger.warning(f"⚠️  Brand application failed: {brand_result.get('error')}")

        # Step 4: Apply theme (Sink)
        if params.get("apply_theme", True) and artifact_path:
            logger.info("🌈 Step 4: Applying theme...")
            theme_skill = registry.get_skill("theme-factory")
            if theme_skill:
                theme_tool = theme_skill.tools.get("apply_theme_tool")
                if theme_tool:
                    # Find the actual file and let theme-factory auto-detect type
                    artifact_dir = Path(artifact_path) if artifact_path else None
                    theme_file = None

                    if artifact_dir and artifact_dir.exists():
                        # Look for actual files by extension
                        for ext in ["*.html", "*.htm", "*.docx", "*.pptx", "*.css"]:
                            files = list(artifact_dir.glob(ext))
                            if files:
                                theme_file = str(files[0])
                                break
                        if not theme_file:
                            theme_file = str(artifact_dir / "index.html")
                    else:
                        theme_file = artifact_path

                    # Don't pass artifact_type - let theme-factory auto-detect from extension
                    theme_params = {
                        "theme_name": params.get("theme_name", "ocean_depths"),
                        "artifact_path": theme_file,
                    }

                    if inspect.iscoroutinefunction(theme_tool):
                        theme_result = await theme_tool(theme_params)
                    else:
                        theme_result = theme_tool(theme_params)

                    if theme_result.get("success"):
                        theme_applied = True
                        logger.info("✅ Theme applied")
                    else:
                        logger.warning(f"⚠️  Theme application failed: {theme_result.get('error')}")

        return {
            "success": True,
            "project_description": project_description,
            "project_name": project_name,
            "domains": domains,
            "artifact_path": artifact_path,
            "brand_applied": brand_applied,
            "theme_applied": theme_applied,
        }

    except Exception as e:
        logger.error(f"Content & Branding Pipeline error: {e}", exc_info=True)
        return {"success": False, "error": f"Workflow failed: {str(e)}"}


__all__ = ["content_branding_pipeline_tool"]
