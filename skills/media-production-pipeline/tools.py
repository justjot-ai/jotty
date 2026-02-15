"""
Media Production Pipeline Composite Skill

Complete media production workflow:
1. Enhance image (image-enhancer)
2. Create design (canvas-design)
3. Create GIF (slack-gif-creator)

Source → Processor → Sink pattern.
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
status = SkillStatus("media-production-pipeline")


logger = logging.getLogger(__name__)


@async_tool_wrapper()
async def media_production_pipeline_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Complete media production workflow: enhance → design → GIF.

    Args:
        params:
            - workflow_type (str, required): 'enhance', 'design', 'gif', or 'full'
            - image_path (str, optional): Path to image for enhancement
            - target_resolution (str, optional): Target resolution (default: '2x')
            - design_brief (str, optional): Design brief for canvas design
            - design_dimensions (tuple, optional): Design dimensions (default: (800, 600))
            - design_output_format (str, optional): 'png' or 'pdf' (default: 'png')
            - gif_description (str, optional): Description for GIF
            - gif_type (str, optional): 'emoji' or 'animation' (default: 'emoji')
            - animation_type (str, optional): Animation type (default: 'bounce')
            - enhance_image (bool, optional): Enhance image (default: True for 'full')
            - create_design (bool, optional): Create design (default: True for 'full')
            - create_gif (bool, optional): Create GIF (default: True for 'full')

    Returns:
        Dictionary with enhanced image, design, and GIF paths
    """
    status.set_callback(params.pop("_status_callback", None))

    try:
        try:
            from Jotty.core.capabilities.registry.skills_registry import get_skills_registry
        except ImportError:
            from Jotty.core.capabilities.registry.skills_registry import get_skills_registry

        registry = get_skills_registry()
        registry.init()

        workflow_type = params.get("workflow_type", "full")

        logger.info(f"🎨 Media Production Pipeline: {workflow_type}")

        enhanced_image_path = None
        design_path = None
        gif_path = None

        # Step 1: Enhance image (Source)
        if workflow_type in ["enhance", "full"] and params.get(
            "enhance_image", workflow_type == "full"
        ):
            logger.info("✨ Step 1: Enhancing image...")
            enhance_skill = registry.get_skill("image-enhancer")
            if enhance_skill:
                enhance_tool = enhance_skill.tools.get("enhance_image_tool")
                if enhance_tool:
                    image_path = params.get("image_path")
                    if not image_path:
                        logger.warning("⚠️  No image_path provided, skipping enhancement")
                    else:
                        enhance_params = {
                            "image_path": image_path,
                            "target_resolution": params.get("target_resolution", "2x"),
                        }

                        if inspect.iscoroutinefunction(enhance_tool):
                            enhance_result = await enhance_tool(enhance_params)
                        else:
                            enhance_result = enhance_tool(enhance_params)

                        if enhance_result.get("success"):
                            enhanced_image_path = enhance_result.get("output_path")
                            logger.info(f"✅ Image enhanced: {enhanced_image_path}")
                        else:
                            logger.warning(
                                f"⚠️  Image enhancement failed: {enhance_result.get('error')}"
                            )

        # Step 2: Create design (Processor)
        if workflow_type in ["design", "full"] and params.get(
            "create_design", workflow_type == "full"
        ):
            logger.info("🎨 Step 2: Creating design...")
            design_skill = registry.get_skill("canvas-design")
            if design_skill:
                design_tool = design_skill.tools.get("create_design_artwork_tool")
                if design_tool:
                    design_brief = params.get("design_brief", "A minimalist design")
                    design_params = {
                        "design_brief": design_brief,
                        "output_format": params.get("design_output_format", "png"),
                        "dimensions": params.get("design_dimensions", (800, 600)),
                    }

                    if inspect.iscoroutinefunction(design_tool):
                        design_result = await design_tool(design_params)
                    else:
                        design_result = design_tool(design_params)

                    if design_result.get("success"):
                        design_path = design_result.get("output_path")
                        logger.info(f"✅ Design created: {design_path}")
                    else:
                        logger.warning(f"⚠️  Design creation failed: {design_result.get('error')}")

        # Step 3: Create GIF (Sink)
        if workflow_type in ["gif", "full"] and params.get("create_gif", workflow_type == "full"):
            logger.info("🎬 Step 3: Creating GIF...")
            gif_skill = registry.get_skill("slack-gif-creator")
            if gif_skill:
                gif_tool = gif_skill.tools.get("create_slack_gif_tool")
                if gif_tool:
                    gif_description = params.get("gif_description", "A bouncing ball")
                    gif_params = {
                        "description": gif_description,
                        "gif_type": params.get("gif_type", "emoji"),
                        "animation_type": params.get("animation_type", "bounce"),
                    }

                    if inspect.iscoroutinefunction(gif_tool):
                        gif_result = await gif_tool(gif_params)
                    else:
                        gif_result = gif_tool(gif_params)

                    if gif_result.get("success"):
                        gif_path = gif_result.get("gif_path")
                        logger.info(f"✅ GIF created: {gif_path}")
                    else:
                        logger.warning(f"⚠️  GIF creation failed: {gif_result.get('error')}")

        return {
            "success": True,
            "workflow_type": workflow_type,
            "enhanced_image_path": enhanced_image_path,
            "design_path": design_path,
            "gif_path": gif_path,
        }

    except Exception as e:
        logger.error(f"Media Production Pipeline error: {e}", exc_info=True)
        return {"success": False, "error": f"Workflow failed: {str(e)}"}


__all__ = ["media_production_pipeline_tool"]
