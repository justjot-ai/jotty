"""
LIDA to JustJot Skill

Transforms data and natural language questions into AI-generated visualizations
saved as JustJot ideas with multiple section types.

Multi-agent workflow:
1. Load data (DataFrame/CSV)
2. Use LIDA VisualizationLayer for chart generation
3. Transform to JustJot sections using registry-driven SectionTransformer
4. Create idea on JustJot.ai via MCP client
"""
import asyncio
import inspect
import logging
import json
import os
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

import pandas as pd

from Jotty.core.utils.skill_status import SkillStatus

# Status emitter for progress updates
status = SkillStatus("lida-to-justjot")


logger = logging.getLogger(__name__)


class LidaToJustJotSkill:
    """
    LIDA to JustJot skill for creating visualization ideas.

    Integrates with:
    - LIDA VisualizationLayer for AI-generated charts
    - JustJot section registry for all 47+ section types
    - JustJot MCP client for idea creation
    """

    def __init__(self):
        self._viz_layer = None
        self._idea_builder = None
        self._registry = None

    @property
    def registry(self):
        """Lazy load skills registry."""
        if self._registry is None:
            try:
                from Jotty.core.registry.skills_registry import get_skills_registry
            except ImportError:
                from Jotty.core.registry.skills_registry import get_skills_registry
            self._registry = get_skills_registry()
            self._registry.init()
        return self._registry

    def _get_viz_layer(self, df: pd.DataFrame):
        """Create VisualizationLayer from DataFrame."""
        from Jotty.core.semantic.visualization import VisualizationLayer
        return VisualizationLayer.from_dataframe(df)

    def _get_idea_builder(self, viz_layer, config: Dict = None):
        """Create JustJotIdeaBuilder with config."""
        from Jotty.core.semantic.visualization.justjot import (
            JustJotIdeaBuilder,
            VisualizationIdeaConfig
        )

        builder_config = VisualizationIdeaConfig(
            include_data=config.get('include_data', True) if config else True,
            include_chart=config.get('include_chart', True) if config else True,
            include_code=config.get('include_code', True) if config else True,
            include_insights=config.get('include_insights', True) if config else True,
            interactive=config.get('interactive', True) if config else True,
            max_data_rows=config.get('max_data_rows', 100) if config else 100,
        )

        return JustJotIdeaBuilder(viz_layer, builder_config)

    async def visualize_to_idea(
        self,
        df: pd.DataFrame,
        question: str,
        title: str = None,
        description: str = None,
        tags: List[str] = None,
        user_id: str = None,
        author: str = None,
        include_data: bool = True,
        include_chart: bool = True,
        include_code: bool = True,
        include_insights: bool = True,
        interactive: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate visualization and create JustJot idea.

        Args:
            df: Source DataFrame
            question: Natural language question (e.g., "Show sales by region as bar chart")
            title: Idea title (auto-generated if not provided)
            description: Idea description
            tags: Tags for the idea
            user_id: Clerk user ID for idea assignment
            author: Author name
            include_data: Include data-table section
            include_chart: Include chart section
            include_code: Include visualization code section
            include_insights: Include AI-generated insights section
            interactive: Use interactive charts

        Returns:
            Dict with success status, idea_id, and details
        """
        try:
            logger.info(f"Starting LIDA-to-JustJot workflow for: {question}")

            # Step 1: Create visualization layer
            logger.info("Step 1: Creating VisualizationLayer...")
            viz_layer = self._get_viz_layer(df)

            # Step 2: Create idea builder with config
            logger.info("Step 2: Creating IdeaBuilder with config...")
            config = {
                'include_data': include_data,
                'include_chart': include_chart,
                'include_code': include_code,
                'include_insights': include_insights,
                'interactive': interactive,
            }
            builder = self._get_idea_builder(viz_layer, config)

            # Step 3: Generate visualization idea
            logger.info("Step 3: Generating visualization idea...")
            idea = builder.create_visualization_idea(
                question=question,
                title=title,
                description=description,
                tags=tags or ['lida', 'visualization', 'ai-generated'],
            )

            # Step 4: Convert to MCP format
            logger.info("Step 4: Converting to MCP format...")
            mcp_data = builder.to_justjot_mcp_format(idea)

            # Add user assignment
            if user_id:
                mcp_data['userId'] = user_id
            if author:
                mcp_data['author'] = author

            # Step 5: Create idea (MongoDB direct -> MCP fallback -> HTTP fallback)
            logger.info("Step 5: Creating idea in JustJot...")
            result = await self._create_idea_mcp(mcp_data)

            if result.get('success'):
                idea_id = result.get('id')
                logger.info(f"Successfully created idea: {idea_id}")
                return {
                    'success': True,
                    'idea_id': idea_id,
                    'title': idea.title,
                    'description': idea.description,
                    'sections': len(idea.sections),
                    'section_types': [s.type for s in idea.sections],
                    'tags': idea.tags,
                    'method': result.get('method', 'mongodb'),
                    'message': f'Visualization idea "{idea.title}" created with {len(idea.sections)} sections'
                }
            else:
                return {
                    'success': False,
                    'error': result.get('error', 'Failed to create idea')
                }

        except Exception as e:
            logger.error(f"LIDA-to-JustJot error: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    async def create_dashboard_idea(
        self,
        df: pd.DataFrame,
        user_request: str,
        num_charts: int = 4,
        title: str = None,
        tags: List[str] = None,
        user_id: str = None,
        author: str = None,
    ) -> Dict[str, Any]:
        """
        Create multi-chart dashboard idea.

        Args:
            df: Source DataFrame
            user_request: High-level analysis request
            num_charts: Number of charts to generate
            title: Dashboard title
            tags: Tags for the idea
            user_id: Clerk user ID
            author: Author name

        Returns:
            Dict with success status and details
        """
        try:
            logger.info(f"Starting dashboard creation for: {user_request}")

            viz_layer = self._get_viz_layer(df)
            builder = self._get_idea_builder(viz_layer, {
                'include_data': True,
                'include_insights': True,
                'interactive': True,
            })

            idea = builder.create_dashboard_idea(
                user_request=user_request,
                num_charts=num_charts,
                title=title,
                tags=tags or ['dashboard', 'lida', 'ai-generated'],
            )

            mcp_data = builder.to_justjot_mcp_format(idea)
            if user_id:
                mcp_data['userId'] = user_id
            if author:
                mcp_data['author'] = author

            result = await self._create_idea_mcp(mcp_data)

            if result.get('success'):
                return {
                    'success': True,
                    'idea_id': result.get('id'),
                    'title': idea.title,
                    'sections': len(idea.sections),
                    'charts': num_charts,
                    'tags': idea.tags,
                    'message': f'Dashboard "{idea.title}" created with {len(idea.sections)} sections'
                }
            else:
                return {
                    'success': False,
                    'error': result.get('error', 'Failed to create dashboard')
                }

        except Exception as e:
            logger.error(f"Dashboard creation error: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    async def create_custom_idea(
        self,
        sections: List[Dict[str, Any]],
        title: str,
        description: str = None,
        tags: List[str] = None,
        user_id: str = None,
        author: str = None,
    ) -> Dict[str, Any]:
        """
        Create idea with custom sections using any section type from registry.

        Args:
            sections: List of section dicts with 'type', 'title', 'content'
            title: Idea title
            description: Idea description
            tags: Tags
            user_id: Clerk user ID
            author: Author name

        Returns:
            Dict with success status and details

        Example:
            await skill.create_custom_idea(
                title="My Analysis",
                sections=[
                    {'type': 'text', 'title': 'Overview', 'content': '# Analysis'},
                    {'type': 'chart', 'title': 'Sales', 'content': chart_json},
                    {'type': 'kanban-board', 'title': 'Tasks', 'content': kanban_json},
                    {'type': 'swot', 'title': 'SWOT Analysis', 'content': swot_json},
                ]
            )
        """
        try:
            from Jotty.core.semantic.visualization.justjot import (
                JustJotIdea,
                JustJotSection,
                SectionTransformer,
            )

            logger.info(f"Creating custom idea: {title} with {len(sections)} sections")

            # Validate section types against registry
            transformer = SectionTransformer()
            validated_sections = []

            for section_data in sections:
                section_type = section_data.get('type', 'text')
                section_title = section_data.get('title', 'Section')
                content = section_data.get('content', '')

                # Use transformer for proper content serialization
                section = transformer.transform(section_type, content, section_title)
                validated_sections.append(section)

            # Create idea
            idea = JustJotIdea(
                title=title,
                description=description or f"Custom idea with {len(validated_sections)} sections",
                sections=validated_sections,
                tags=tags or ['custom', 'ai-generated'],
                template_name="Blank",
                status="Draft"
            )

            # Convert to MCP format
            mcp_data = {
                'title': idea.title,
                'description': idea.description,
                'tags': idea.tags,
                'templateName': idea.template_name,
                'status': idea.status,
                'sections': [s.to_dict() for s in idea.sections]
            }

            if user_id:
                mcp_data['userId'] = user_id
            if author:
                mcp_data['author'] = author

            result = await self._create_idea_mcp(mcp_data)

            if result.get('success'):
                return {
                    'success': True,
                    'idea_id': result.get('id'),
                    'title': idea.title,
                    'sections': len(idea.sections),
                    'section_types': [s.type for s in idea.sections],
                    'message': f'Custom idea "{idea.title}" created'
                }
            else:
                return {
                    'success': False,
                    'error': result.get('error', 'Failed to create idea')
                }

        except Exception as e:
            logger.error(f"Custom idea creation error: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    async def _create_idea_mcp(self, mcp_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create idea via JustJot (direct MongoDB preferred)."""
        # Try direct MongoDB first (most reliable)
        result = await self._create_idea_mongodb(mcp_data)
        if result.get('success') and result.get('id'):
            return result

        # Fallback to mcp-justjot HTTP API skill
        logger.info("Trying mcp-justjot HTTP API...")
        return await self._create_idea_http(mcp_data)

    async def _create_idea_mongodb(self, mcp_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create idea directly in MongoDB (bypasses MCP/HTTP for reliability)."""
        try:
            from pymongo import MongoClient
            from datetime import datetime
            from bson import ObjectId

            # Try multiple MongoDB URIs in order of preference
            mongodb_uri = os.environ.get('MONGODB_URI')

            if not mongodb_uri:
                # Try to load from JustJot.ai .env.local
                justjot_env = Path('/var/www/sites/personal/stock_market/JustJot.ai/.env.local')
                if justjot_env.exists():
                    logger.info(f"Loading MongoDB URI from {justjot_env}")
                    with open(justjot_env) as f:
                        for line in f:
                            if line.startswith('MONGODB_URI='):
                                mongodb_uri = line.split('=', 1)[1].strip().strip('"').strip("'")
                                logger.info(f"Found MONGODB_URI in .env.local")
                                break

            # Always use JustJot production MongoDB for reliability
            if not mongodb_uri or 'localhost' in mongodb_uri or '127.0.0.1' in mongodb_uri or 'planmyinvesting' in mongodb_uri:
                # Use JustJot production MongoDB (same as MCP server uses)
                mongodb_uri = 'mongodb://justjot:ksG07jjmU9lO5zNd61W3Su9J@150.230.143.6:27017/justjot?authSource=admin'
                logger.info("Using JustJot production MongoDB")

            logger.info(f"Connecting to MongoDB at {mongodb_uri.split('@')[-1].split('/')[0]}...")
            client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)

            # Get database name from URI or default
            db_name = 'justjot'  # Default to JustJot production DB
            if '/' in mongodb_uri:
                db_part = mongodb_uri.split('/')[-1].split('?')[0]
                if db_part and db_part not in ['', 'admin']:
                    db_name = db_part
            logger.info(f"Using database: {db_name}")

            db = client[db_name]
            ideas_collection = db['ideas']

            # Prepare idea document
            now = datetime.utcnow()
            idea_doc = {
                'title': mcp_data.get('title', 'Untitled'),
                'description': mcp_data.get('description', ''),
                'status': mcp_data.get('status', 'Draft'),
                'templateName': mcp_data.get('templateName', 'Blank'),
                'tags': mcp_data.get('tags', []),
                'sections': [],
                'createdAt': now,
                'updatedAt': now,
            }

            # Add userId if provided
            if mcp_data.get('userId'):
                idea_doc['userId'] = mcp_data['userId']

            # Add author if provided
            if mcp_data.get('author'):
                idea_doc['author'] = mcp_data['author']

            # Process sections
            for idx, section in enumerate(mcp_data.get('sections', [])):
                idea_doc['sections'].append({
                    'index': idx,
                    'title': section.get('title', f'Section {idx + 1}'),
                    'type': section.get('type', 'text'),
                    'content': section.get('content', ''),
                    'notes': '',
                    'isBookmarked': False,
                })

            # Insert into MongoDB
            result = ideas_collection.insert_one(idea_doc)
            idea_id = str(result.inserted_id)

            client.close()
            logger.info(f"Created idea directly in MongoDB: {idea_id}")

            return {
                'success': True,
                'id': idea_id,
                'title': idea_doc['title'],
                'sections': len(idea_doc['sections']),
                'method': 'mongodb-direct',
            }

        except Exception as e:
            logger.warning(f"Direct MongoDB failed: {e}")
            return {'success': False, 'error': str(e)}

    async def _create_idea_http(self, mcp_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create idea via mcp-justjot HTTP API skill."""
        try:
            skill = self.registry.get_skill('mcp-justjot')
            if not skill:
                return {'success': False, 'error': 'mcp-justjot skill not found'}

            create_tool = skill.tools.get('create_idea_tool')
            if not create_tool:
                return {'success': False, 'error': 'create_idea_tool not found'}

            logger.info("Using mcp-justjot skill...")
            if inspect.iscoroutinefunction(create_tool):
                result = await create_tool(mcp_data)
            else:
                result = create_tool(mcp_data)

            if result.get('success'):
                result['method'] = 'http-api'
            return result

        except Exception as e:
            logger.error(f"HTTP API error: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def get_available_section_types(self) -> List[Dict[str, str]]:
        """Get all available section types from JustJot registry."""
        from Jotty.core.semantic.visualization.justjot import get_all_section_types

        types = get_all_section_types()
        return [
            {
                'id': type_id,
                'label': info.label if hasattr(info, 'label') else type_id,
                'category': info.category if hasattr(info, 'category') else 'Other',
                'description': info.description if hasattr(info, 'description') else '',
            }
            for type_id, info in types.items()
        ]

    def get_section_types_context(self) -> str:
        """Get LLM context for section type selection."""
        from Jotty.core.semantic.visualization.justjot import get_section_types_context
        return get_section_types_context()


# ============================================
# Tool Functions (Skill Interface)
# ============================================

_skill_instance = None

def _get_skill() -> LidaToJustJotSkill:
    """Get or create skill instance."""
    global _skill_instance
    if _skill_instance is None:
        _skill_instance = LidaToJustJotSkill()
    return _skill_instance


async def visualize_to_idea_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate LIDA visualization and create JustJot idea.

    Args:
        params: Dictionary containing:
            - data (required): DataFrame, CSV string, CSV path, or list of dicts
            - question (str, required): Natural language visualization question
            - title (str, optional): Idea title
            - description (str, optional): Idea description
            - tags (list, optional): Tags for the idea
            - userId (str, optional): Clerk user ID
            - author (str, optional): Author name
            - include_data (bool, optional): Include data table section (default: True)
            - include_chart (bool, optional): Include chart section (default: True)
            - include_code (bool, optional): Include code section (default: True)
            - include_insights (bool, optional): Include insights section (default: True)
            - interactive (bool, optional): Use interactive charts (default: True)

    Returns:
        Dictionary with success status, idea_id, and details
    """
    status.set_callback(params.pop('_status_callback', None))

    # Extract parameters
    data = params.get('data')
    question = params.get('question')

    if data is None:
        return {'success': False, 'error': 'data parameter is required'}
    if not question:
        return {'success': False, 'error': 'question parameter is required'}

    # Convert data to DataFrame
    df = _to_dataframe(data)
    if df is None:
        return {'success': False, 'error': 'Failed to convert data to DataFrame'}

    skill = _get_skill()
    return await skill.visualize_to_idea(
        df=df,
        question=question,
        title=params.get('title'),
        description=params.get('description'),
        tags=params.get('tags'),
        user_id=params.get('userId'),
        author=params.get('author'),
        include_data=params.get('include_data', True),
        include_chart=params.get('include_chart', True),
        include_code=params.get('include_code', True),
        include_insights=params.get('include_insights', True),
        interactive=params.get('interactive', True),
    )


async def create_dashboard_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create multi-chart dashboard idea.

    Args:
        params: Dictionary containing:
            - data (required): DataFrame, CSV string, CSV path, or list of dicts
            - request (str, required): High-level analysis request
            - num_charts (int, optional): Number of charts (default: 4)
            - title (str, optional): Dashboard title
            - tags (list, optional): Tags
            - userId (str, optional): Clerk user ID
            - author (str, optional): Author name

    Returns:
        Dictionary with success status and details
    """
    status.set_callback(params.pop('_status_callback', None))

    data = params.get('data')
    request = params.get('request')

    if data is None:
        return {'success': False, 'error': 'data parameter is required'}
    if not request:
        return {'success': False, 'error': 'request parameter is required'}

    df = _to_dataframe(data)
    if df is None:
        return {'success': False, 'error': 'Failed to convert data to DataFrame'}

    skill = _get_skill()
    return await skill.create_dashboard_idea(
        df=df,
        user_request=request,
        num_charts=params.get('num_charts', 4),
        title=params.get('title'),
        tags=params.get('tags'),
        user_id=params.get('userId'),
        author=params.get('author'),
    )


async def create_custom_idea_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create idea with custom sections using any section type.

    Args:
        params: Dictionary containing:
            - title (str, required): Idea title
            - sections (list, required): List of section dicts with type, title, content
            - description (str, optional): Idea description
            - tags (list, optional): Tags
            - userId (str, optional): Clerk user ID
            - author (str, optional): Author name

    Returns:
        Dictionary with success status and details

    Example:
        await create_custom_idea_tool({
            'title': 'Business Analysis',
            'sections': [
                {'type': 'text', 'title': 'Overview', 'content': '# Analysis'},
                {'type': 'chart', 'title': 'Revenue', 'content': '{"type":"bar",...}'},
                {'type': 'swot', 'title': 'SWOT', 'content': '{"strengths":[...]}'},
            ]
        })
    """
    status.set_callback(params.pop('_status_callback', None))
    title = params.get('title')
    sections = params.get('sections')

    if not title:
        return {'success': False, 'error': 'title parameter is required'}
    if not sections or not isinstance(sections, list):
        return {'success': False, 'error': 'sections parameter is required (list of dicts)'}

    skill = _get_skill()
    return await skill.create_custom_idea(
        sections=sections,
        title=title,
        description=params.get('description'),
        tags=params.get('tags'),
        user_id=params.get('userId'),
        author=params.get('author'),
    )


def get_section_types_tool(params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get all available JustJot section types.

    Returns:
        Dictionary with available section types and their info
    """
    status.set_callback(params.pop('_status_callback', None))

    skill = _get_skill()
    types = skill.get_available_section_types()

    # Group by category
    by_category = {}
    for t in types:
        cat = t.get('category', 'Other')
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(t)

    return {
        'success': True,
        'count': len(types),
        'types': types,
        'by_category': by_category,
    }


def get_section_context_tool(params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get LLM context for section type selection.

    Returns:
        Dictionary with context string for LLM prompts
    """
    status.set_callback(params.pop('_status_callback', None))

    skill = _get_skill()
    context = skill.get_section_types_context()

    return {
        'success': True,
        'context': context,
        'usage': 'Include this context in LLM prompts to help select appropriate section types'
    }


def _to_dataframe(data: Any) -> Optional[pd.DataFrame]:
    """Convert various data formats to DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data

    if isinstance(data, str):
        # Check if it's a file path
        if os.path.exists(data):
            if data.endswith('.csv'):
                return pd.read_csv(data)
            elif data.endswith(('.xlsx', '.xls')):
                return pd.read_excel(data)
            elif data.endswith('.json'):
                return pd.read_json(data)
        else:
            # Try parsing as CSV string
            try:
                from io import StringIO
                return pd.read_csv(StringIO(data))
            except:
                pass
            # Try parsing as JSON
            try:
                return pd.read_json(data)
            except:
                pass

    if isinstance(data, list):
        try:
            return pd.DataFrame(data)
        except:
            pass

    if isinstance(data, dict):
        try:
            return pd.DataFrame(data)
        except:
            try:
                return pd.DataFrame([data])
            except:
                pass

    return None


__all__ = [
    'LidaToJustJotSkill',
    'visualize_to_idea_tool',
    'create_dashboard_tool',
    'create_custom_idea_tool',
    'get_section_types_tool',
    'get_section_context_tool',
]
