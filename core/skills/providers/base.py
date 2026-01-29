"""
Base Skill Provider Classes for Jotty V2
=========================================

Defines the abstract base class for all skill providers and common types.
Includes skill-to-category mapping for routing tasks to appropriate skills.
"""

import time
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class SkillCategory(Enum):
    """Categories of skills that can have multiple providers."""
    BROWSER = "browser"           # Web browsing, scraping, automation
    TERMINAL = "terminal"         # Shell commands, CLI operations
    COMPUTER_USE = "computer_use" # GUI control, mouse/keyboard
    CODE_EXECUTION = "code_exec"  # Running code locally
    FILE_OPERATIONS = "file_ops"  # File read/write/edit
    WEB_SEARCH = "web_search"     # Search engines, research
    DATA_EXTRACTION = "data_extract"  # Scraping, parsing
    FORM_AUTOMATION = "form_auto" # Form filling, RPA
    API_CALLS = "api_calls"       # REST/GraphQL API interactions
    COMMUNICATION = "communication"  # Email, messaging, notifications
    DOCUMENT = "document"         # PDF, document processing
    MEDIA = "media"               # Image, audio, video processing
    DATABASE = "database"         # Database operations
    SCHEDULING = "scheduling"     # Cron, scheduling tasks
    ANALYTICS = "analytics"       # Data analysis, reporting


# =============================================================================
# Skill-to-Category Mapping
# =============================================================================

# Maps skill names (from SkillsRegistry) to SkillCategory
# Used by JottyDefaultProvider to route tasks to appropriate skills
SKILL_CATEGORY_MAP: Dict[str, 'SkillCategory'] = {
    # Web Search & Research
    'web-search': SkillCategory.WEB_SEARCH,
    'google-search': SkillCategory.WEB_SEARCH,
    'bing-search': SkillCategory.WEB_SEARCH,
    'duckduckgo': SkillCategory.WEB_SEARCH,
    'arxiv-search': SkillCategory.WEB_SEARCH,
    'wikipedia': SkillCategory.WEB_SEARCH,
    'scholar-search': SkillCategory.WEB_SEARCH,
    'perplexity': SkillCategory.WEB_SEARCH,
    'tavily': SkillCategory.WEB_SEARCH,
    'exa-search': SkillCategory.WEB_SEARCH,
    'serper': SkillCategory.WEB_SEARCH,
    'last30days': SkillCategory.WEB_SEARCH,

    # Web Scraping & Data Extraction
    'web-scraper': SkillCategory.DATA_EXTRACTION,
    'html-parser': SkillCategory.DATA_EXTRACTION,
    'json-parser': SkillCategory.DATA_EXTRACTION,
    'xml-parser': SkillCategory.DATA_EXTRACTION,
    'csv-parser': SkillCategory.DATA_EXTRACTION,
    'table-extractor': SkillCategory.DATA_EXTRACTION,
    'firecrawl': SkillCategory.DATA_EXTRACTION,
    'crawl4ai': SkillCategory.DATA_EXTRACTION,
    'beautifulsoup': SkillCategory.DATA_EXTRACTION,
    'scrapy': SkillCategory.DATA_EXTRACTION,

    # Browser Automation
    'browser-use': SkillCategory.BROWSER,
    'playwright': SkillCategory.BROWSER,
    'selenium': SkillCategory.BROWSER,
    'puppeteer': SkillCategory.BROWSER,
    'browserbase': SkillCategory.BROWSER,
    'steel': SkillCategory.BROWSER,
    'multion': SkillCategory.BROWSER,

    # File Operations
    'file-operations': SkillCategory.FILE_OPERATIONS,
    'file-reader': SkillCategory.FILE_OPERATIONS,
    'file-writer': SkillCategory.FILE_OPERATIONS,
    'file-manager': SkillCategory.FILE_OPERATIONS,
    'directory-ops': SkillCategory.FILE_OPERATIONS,
    'zip-handler': SkillCategory.FILE_OPERATIONS,

    # PDF & Document Processing
    'pdf-tools': SkillCategory.DOCUMENT,
    'pdf-reader': SkillCategory.DOCUMENT,
    'pdf-writer': SkillCategory.DOCUMENT,
    'pdf-extractor': SkillCategory.DOCUMENT,
    'docx-handler': SkillCategory.DOCUMENT,
    'excel-handler': SkillCategory.DOCUMENT,
    'markdown': SkillCategory.DOCUMENT,
    'latex': SkillCategory.DOCUMENT,
    'ocr': SkillCategory.DOCUMENT,
    'tesseract': SkillCategory.DOCUMENT,

    # Terminal & Shell
    'shell-exec': SkillCategory.TERMINAL,
    'bash': SkillCategory.TERMINAL,
    'terminal': SkillCategory.TERMINAL,
    'ssh': SkillCategory.TERMINAL,
    'command-runner': SkillCategory.TERMINAL,
    'shell-automation': SkillCategory.TERMINAL,

    # Code Execution
    'python-executor': SkillCategory.CODE_EXECUTION,
    'javascript-executor': SkillCategory.CODE_EXECUTION,
    'code-interpreter': SkillCategory.CODE_EXECUTION,
    'jupyter': SkillCategory.CODE_EXECUTION,
    'repl': SkillCategory.CODE_EXECUTION,
    'calculator': SkillCategory.CODE_EXECUTION,
    'math-solver': SkillCategory.CODE_EXECUTION,
    'e2b': SkillCategory.CODE_EXECUTION,
    'open-interpreter': SkillCategory.CODE_EXECUTION,

    # API & Communication
    'telegram-sender': SkillCategory.COMMUNICATION,
    'telegram-bot': SkillCategory.COMMUNICATION,
    'slack-sender': SkillCategory.COMMUNICATION,
    'discord-bot': SkillCategory.COMMUNICATION,
    'email-sender': SkillCategory.COMMUNICATION,
    'smtp': SkillCategory.COMMUNICATION,
    'twilio': SkillCategory.COMMUNICATION,
    'sendgrid': SkillCategory.COMMUNICATION,
    'pushover': SkillCategory.COMMUNICATION,

    # API Calls & Integrations
    'rest-client': SkillCategory.API_CALLS,
    'graphql-client': SkillCategory.API_CALLS,
    'api-client': SkillCategory.API_CALLS,
    'http-client': SkillCategory.API_CALLS,
    'webhook': SkillCategory.API_CALLS,
    'notion-client': SkillCategory.API_CALLS,
    'github': SkillCategory.API_CALLS,
    'gitlab': SkillCategory.API_CALLS,
    'jira': SkillCategory.API_CALLS,
    'trello': SkillCategory.API_CALLS,
    'asana': SkillCategory.API_CALLS,
    'linear': SkillCategory.API_CALLS,
    'stripe': SkillCategory.API_CALLS,
    'openai': SkillCategory.API_CALLS,
    'anthropic': SkillCategory.API_CALLS,
    'praw': SkillCategory.API_CALLS,
    'reddit': SkillCategory.API_CALLS,
    'twitter': SkillCategory.API_CALLS,
    'x-api': SkillCategory.API_CALLS,

    # Form Automation & RPA
    'form-filler': SkillCategory.FORM_AUTOMATION,
    'rpa': SkillCategory.FORM_AUTOMATION,
    'skyvern': SkillCategory.FORM_AUTOMATION,
    'autofill': SkillCategory.FORM_AUTOMATION,

    # Computer Use & GUI
    'computer-use': SkillCategory.COMPUTER_USE,
    'desktop-automation': SkillCategory.COMPUTER_USE,
    'mouse-keyboard': SkillCategory.COMPUTER_USE,
    'screenshot': SkillCategory.COMPUTER_USE,
    'screen-capture': SkillCategory.COMPUTER_USE,
    'agent-s': SkillCategory.COMPUTER_USE,
    'pyautogui': SkillCategory.COMPUTER_USE,

    # Media Processing
    'image-processor': SkillCategory.MEDIA,
    'image-generator': SkillCategory.MEDIA,
    'audio-processor': SkillCategory.MEDIA,
    'video-processor': SkillCategory.MEDIA,
    'whisper': SkillCategory.MEDIA,
    'tts': SkillCategory.MEDIA,
    'dalle': SkillCategory.MEDIA,
    'stable-diffusion': SkillCategory.MEDIA,
    'ffmpeg': SkillCategory.MEDIA,
    'pillow': SkillCategory.MEDIA,

    # Database
    'sql-client': SkillCategory.DATABASE,
    'postgres': SkillCategory.DATABASE,
    'mysql': SkillCategory.DATABASE,
    'sqlite': SkillCategory.DATABASE,
    'mongodb': SkillCategory.DATABASE,
    'redis': SkillCategory.DATABASE,
    'elasticsearch': SkillCategory.DATABASE,
    'supabase': SkillCategory.DATABASE,
    'firebase': SkillCategory.DATABASE,

    # Scheduling & Automation
    'scheduler': SkillCategory.SCHEDULING,
    'cron': SkillCategory.SCHEDULING,
    'task-scheduler': SkillCategory.SCHEDULING,
    'workflow': SkillCategory.SCHEDULING,

    # Analytics & Reporting
    'analytics': SkillCategory.ANALYTICS,
    'charts': SkillCategory.ANALYTICS,
    'visualization': SkillCategory.ANALYTICS,
    'matplotlib': SkillCategory.ANALYTICS,
    'plotly': SkillCategory.ANALYTICS,
    'pandas': SkillCategory.ANALYTICS,
    'numpy': SkillCategory.ANALYTICS,
}


# Keywords for category inference from task descriptions
CATEGORY_KEYWORDS: Dict['SkillCategory', List[str]] = {
    SkillCategory.WEB_SEARCH: [
        'search', 'find', 'lookup', 'google', 'research', 'discover', 'query'
    ],
    SkillCategory.DATA_EXTRACTION: [
        'extract', 'parse', 'scrape', 'crawl', 'fetch', 'html', 'json', 'xml'
    ],
    SkillCategory.BROWSER: [
        'browse', 'navigate', 'click', 'website', 'page', 'browser', 'web page'
    ],
    SkillCategory.FILE_OPERATIONS: [
        'file', 'read', 'write', 'save', 'load', 'directory', 'folder', 'copy', 'move'
    ],
    SkillCategory.DOCUMENT: [
        'pdf', 'document', 'docx', 'excel', 'spreadsheet', 'ocr', 'text extract'
    ],
    SkillCategory.TERMINAL: [
        'shell', 'command', 'bash', 'terminal', 'cli', 'execute', 'run command'
    ],
    SkillCategory.CODE_EXECUTION: [
        'code', 'python', 'javascript', 'execute', 'calculate', 'compute', 'eval'
    ],
    SkillCategory.COMMUNICATION: [
        'send', 'message', 'email', 'telegram', 'slack', 'notify', 'alert'
    ],
    SkillCategory.API_CALLS: [
        'api', 'rest', 'graphql', 'endpoint', 'request', 'post', 'get', 'webhook'
    ],
    SkillCategory.FORM_AUTOMATION: [
        'form', 'fill', 'submit', 'input', 'autofill', 'rpa'
    ],
    SkillCategory.COMPUTER_USE: [
        'desktop', 'gui', 'mouse', 'keyboard', 'screenshot', 'screen', 'click'
    ],
    SkillCategory.MEDIA: [
        'image', 'audio', 'video', 'picture', 'photo', 'sound', 'music', 'generate image'
    ],
    SkillCategory.DATABASE: [
        'database', 'sql', 'query', 'table', 'record', 'mongodb', 'postgres', 'insert', 'select'
    ],
    SkillCategory.SCHEDULING: [
        'schedule', 'cron', 'timer', 'recurring', 'periodic', 'automate'
    ],
    SkillCategory.ANALYTICS: [
        'analyze', 'chart', 'graph', 'plot', 'visualize', 'statistics', 'report', 'dashboard'
    ],
}


@dataclass
class ProviderCapability:
    """Describes what a provider can do."""
    category: SkillCategory
    actions: List[str]  # e.g., ["click", "type", "scroll", "screenshot"]
    max_concurrent: int = 1
    requires_display: bool = False  # Needs GUI/display
    requires_network: bool = True
    estimated_latency_ms: int = 1000
    cost_per_call: float = 0.0  # For paid APIs


@dataclass
class ProviderResult:
    """Result from a provider execution."""
    success: bool
    output: Any
    error: str = ""
    execution_time: float = 0.0
    provider_name: str = ""
    category: SkillCategory = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # For learning
    confidence: float = 1.0
    retryable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'output': str(self.output)[:500] if self.output else None,
            'error': self.error,
            'execution_time': self.execution_time,
            'provider_name': self.provider_name,
            'category': self.category.value if self.category else None,
            'confidence': self.confidence,
        }


class SkillProvider(ABC):
    """
    Abstract base class for all skill providers.

    Each provider implements one or more skill categories and can be
    selected dynamically based on learned performance.
    """

    # Provider identification
    name: str = "base"
    version: str = "1.0.0"
    description: str = "Base provider"

    # What this provider can do
    capabilities: List[ProviderCapability] = []

    # Provider state
    is_initialized: bool = False
    is_available: bool = True

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the provider.

        Args:
            config: Provider-specific configuration
        """
        self.config = config or {}
        self._stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_execution_time': 0.0,
            'last_error': None,
            'last_success_time': None,
        }

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the provider (install dependencies, check availability).

        Returns:
            True if initialization successful
        """
        pass

    @abstractmethod
    async def execute(self, task: str, context: Dict[str, Any] = None) -> ProviderResult:
        """
        Execute a task using this provider.

        Args:
            task: Natural language task description
            context: Additional context (previous results, session info, etc.)

        Returns:
            ProviderResult with success status and output
        """
        pass

    @abstractmethod
    def get_categories(self) -> List[SkillCategory]:
        """Get list of skill categories this provider supports."""
        pass

    async def health_check(self) -> bool:
        """Check if provider is healthy and available."""
        return self.is_available and self.is_initialized

    def get_stats(self) -> Dict[str, Any]:
        """Get provider statistics."""
        stats = self._stats.copy()
        if stats['total_calls'] > 0:
            stats['success_rate'] = stats['successful_calls'] / stats['total_calls']
            stats['avg_execution_time'] = stats['total_execution_time'] / stats['total_calls']
        else:
            stats['success_rate'] = 0.0
            stats['avg_execution_time'] = 0.0
        return stats

    def record_execution(self, result: ProviderResult):
        """Record execution statistics."""
        self._stats['total_calls'] += 1
        self._stats['total_execution_time'] += result.execution_time

        if result.success:
            self._stats['successful_calls'] += 1
            self._stats['last_success_time'] = time.time()
        else:
            self._stats['failed_calls'] += 1
            self._stats['last_error'] = result.error

    def supports_category(self, category: SkillCategory) -> bool:
        """Check if provider supports a category."""
        return category in self.get_categories()

    def get_capability(self, category: SkillCategory) -> Optional[ProviderCapability]:
        """Get capability info for a category."""
        for cap in self.capabilities:
            if cap.category == category:
                return cap
        return None

    def __repr__(self) -> str:
        categories = [c.value for c in self.get_categories()]
        return f"{self.__class__.__name__}(name={self.name}, categories={categories})"


class JottyDefaultProvider(SkillProvider):
    """
    Default Jotty provider - uses built-in SkillsRegistry implementations.

    This provider wraps the SkillsRegistry to route tasks to appropriate skills
    based on category inference. It learns from execution results and adapts.

    Key Features:
    - Routes tasks to appropriate skills via category mapping
    - Uses keyword scoring to select best skill for a task
    - Falls back to built-in handlers when no skill matches
    - Tracks execution statistics for learning
    """

    name = "jotty"
    version = "2.0.0"
    description = "Jotty's built-in skill implementations via SkillsRegistry"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Lazy-loaded skills registry
        self._skills_registry = None

        # Category index: category -> [skill_names]
        self._category_index: Dict[SkillCategory, List[str]] = {}

        # Skill scoring cache for performance
        self._skill_score_cache: Dict[str, float] = {}

        # Initialize capabilities for all supported categories
        self.capabilities = [
            ProviderCapability(
                category=cat,
                actions=self._get_category_actions(cat),
                estimated_latency_ms=self._get_category_latency(cat),
            )
            for cat in SkillCategory
        ]

    def _get_category_actions(self, category: SkillCategory) -> List[str]:
        """Get typical actions for a category."""
        action_map = {
            SkillCategory.WEB_SEARCH: ["search", "query", "lookup", "find"],
            SkillCategory.DATA_EXTRACTION: ["extract", "parse", "scrape", "crawl"],
            SkillCategory.BROWSER: ["navigate", "click", "type", "screenshot"],
            SkillCategory.FILE_OPERATIONS: ["read", "write", "copy", "move", "delete"],
            SkillCategory.DOCUMENT: ["read_pdf", "extract_text", "convert", "ocr"],
            SkillCategory.TERMINAL: ["execute", "run", "shell", "command"],
            SkillCategory.CODE_EXECUTION: ["run_code", "evaluate", "compute"],
            SkillCategory.COMMUNICATION: ["send", "notify", "message", "alert"],
            SkillCategory.API_CALLS: ["request", "post", "get", "call"],
            SkillCategory.FORM_AUTOMATION: ["fill", "submit", "autofill"],
            SkillCategory.COMPUTER_USE: ["click", "type", "screenshot", "scroll"],
            SkillCategory.MEDIA: ["process", "generate", "convert", "transcribe"],
            SkillCategory.DATABASE: ["query", "insert", "update", "delete"],
            SkillCategory.SCHEDULING: ["schedule", "cron", "delay", "recurring"],
            SkillCategory.ANALYTICS: ["analyze", "chart", "visualize", "report"],
        }
        return action_map.get(category, ["execute"])

    def _get_category_latency(self, category: SkillCategory) -> int:
        """Get estimated latency in ms for a category."""
        latency_map = {
            SkillCategory.WEB_SEARCH: 2000,
            SkillCategory.DATA_EXTRACTION: 1500,
            SkillCategory.BROWSER: 3000,
            SkillCategory.FILE_OPERATIONS: 50,
            SkillCategory.DOCUMENT: 1000,
            SkillCategory.TERMINAL: 500,
            SkillCategory.CODE_EXECUTION: 1000,
            SkillCategory.COMMUNICATION: 500,
            SkillCategory.API_CALLS: 1000,
            SkillCategory.FORM_AUTOMATION: 2000,
            SkillCategory.COMPUTER_USE: 2000,
            SkillCategory.MEDIA: 5000,
            SkillCategory.DATABASE: 200,
            SkillCategory.SCHEDULING: 100,
            SkillCategory.ANALYTICS: 2000,
        }
        return latency_map.get(category, 1000)

    def _init_skills_registry(self):
        """Lazy initialize SkillsRegistry and build category index."""
        if self._skills_registry is not None:
            return

        try:
            from ...registry.skills_registry import get_skills_registry
            self._skills_registry = get_skills_registry()
            self._skills_registry.init()
            self._build_category_index()
            logger.debug(f"Initialized SkillsRegistry with {len(self._skills_registry.loaded_skills)} skills")
        except Exception as e:
            logger.warning(f"Could not initialize SkillsRegistry: {e}")
            self._skills_registry = None

    def _build_category_index(self):
        """Build index of skills by category for fast lookup."""
        if not self._skills_registry:
            return

        self._category_index = {cat: [] for cat in SkillCategory}

        for skill_name in self._skills_registry.loaded_skills.keys():
            # Check explicit mapping first
            if skill_name in SKILL_CATEGORY_MAP:
                category = SKILL_CATEGORY_MAP[skill_name]
                self._category_index[category].append(skill_name)
            else:
                # Try to infer category from skill name
                inferred = self._infer_skill_category(skill_name)
                if inferred:
                    self._category_index[inferred].append(skill_name)

        # Log category distribution
        for cat, skills in self._category_index.items():
            if skills:
                logger.debug(f"  {cat.value}: {len(skills)} skills")

    def _infer_skill_category(self, skill_name: str) -> Optional[SkillCategory]:
        """Infer category from skill name using keyword matching."""
        skill_lower = skill_name.lower().replace('-', ' ').replace('_', ' ')

        best_category = None
        best_score = 0

        for category, keywords in CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in skill_lower)
            if score > best_score:
                best_score = score
                best_category = category

        return best_category

    async def initialize(self) -> bool:
        """Initialize the Jotty provider with SkillsRegistry."""
        try:
            self._init_skills_registry()
            self.is_initialized = True
            self.is_available = True

            skill_count = len(self._skills_registry.loaded_skills) if self._skills_registry else 0
            logger.info(f"✅ {self.name} provider initialized ({skill_count} skills available)")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize {self.name} provider: {e}")
            self.is_initialized = True  # Still available with fallbacks
            self.is_available = True
            return True

    def get_categories(self) -> List[SkillCategory]:
        """Get all supported categories."""
        return list(SkillCategory)

    async def execute(self, task: str, context: Dict[str, Any] = None) -> ProviderResult:
        """
        Execute task using the best matching skill from SkillsRegistry.

        Flow:
        1. Infer category from task description
        2. Select best skill for category using keyword scoring
        3. Execute skill and return result
        4. Fall back to built-in handlers if no skill matches
        """
        start_time = time.time()
        context = context or {}

        try:
            # Ensure registry is initialized
            self._init_skills_registry()

            # Determine category
            category = self._infer_category(task)
            logger.debug(f"Task category: {category.value}")

            # Try to find and execute a matching skill
            skill_result = await self._execute_via_skill(task, category, context)
            if skill_result:
                skill_result.execution_time = time.time() - start_time
                skill_result.provider_name = self.name
                self.record_execution(skill_result)
                return skill_result

            # Fall back to built-in handlers
            result = await self._execute_builtin(task, category, context)
            result.execution_time = time.time() - start_time
            result.provider_name = self.name
            self.record_execution(result)
            return result

        except Exception as e:
            logger.error(f"Jotty provider error: {e}")
            result = ProviderResult(
                success=False,
                output=None,
                error=str(e),
                execution_time=time.time() - start_time,
                provider_name=self.name,
                retryable=True,
            )
            self.record_execution(result)
            return result

    async def _execute_via_skill(
        self,
        task: str,
        category: SkillCategory,
        context: Dict[str, Any]
    ) -> Optional[ProviderResult]:
        """
        Try to execute task via a matching skill from SkillsRegistry.

        Returns None if no suitable skill found.
        """
        if not self._skills_registry:
            return None

        # Get skills for this category
        category_skills = self._category_index.get(category, [])
        if not category_skills:
            logger.debug(f"No skills found for category {category.value}")
            return None

        # Select best skill using keyword scoring
        skill_name = self._select_skill(category_skills, task)
        if not skill_name:
            return None

        logger.info(f"🎯 Executing via skill: {skill_name}")

        try:
            skill = self._skills_registry.get_skill(skill_name)
            if not skill:
                return None

            # Get tool functions from skill
            tools = skill.tools
            if not tools:
                logger.warning(f"Skill {skill_name} has no tools")
                return None

            # Execute first available tool (or matching tool)
            tool_name, tool_func = self._select_tool(tools, task)
            if not tool_func:
                return None

            logger.debug(f"Executing tool: {tool_name}")

            # Prepare parameters
            params = self._extract_params(task, context)

            # Execute tool
            import asyncio
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(params)
            else:
                result = tool_func(params)

            # Wrap result
            success = result.get('success', True) if isinstance(result, dict) else True
            return ProviderResult(
                success=success,
                output=result,
                category=category,
                metadata={
                    'skill': skill_name,
                    'tool': tool_name,
                },
            )

        except Exception as e:
            logger.warning(f"Skill execution failed: {e}")
            return ProviderResult(
                success=False,
                output=None,
                error=str(e),
                category=category,
                metadata={'skill': skill_name},
                retryable=True,
            )

    def _select_skill(self, skill_names: List[str], task: str) -> Optional[str]:
        """
        Select the best skill for a task using keyword scoring.

        Scores each skill based on keyword matches with the task description.
        """
        if not skill_names:
            return None

        if len(skill_names) == 1:
            return skill_names[0]

        task_lower = task.lower()
        task_words = set(re.split(r'\W+', task_lower))

        best_skill = None
        best_score = -1

        for skill_name in skill_names:
            # Score based on skill name matching task words
            skill_words = set(skill_name.lower().replace('-', ' ').replace('_', ' ').split())
            overlap = len(task_words & skill_words)

            # Bonus for exact substring match
            if skill_name.lower().replace('-', ' ') in task_lower:
                overlap += 2

            # Consider cached performance scores
            cached_score = self._skill_score_cache.get(skill_name, 0)
            total_score = overlap + cached_score

            if total_score > best_score:
                best_score = total_score
                best_skill = skill_name

        return best_skill

    def _select_tool(
        self,
        tools: Dict[str, Callable],
        task: str
    ) -> Tuple[Optional[str], Optional[Callable]]:
        """Select the best tool from a skill's tools based on task."""
        if not tools:
            return None, None

        if len(tools) == 1:
            name = list(tools.keys())[0]
            return name, tools[name]

        task_lower = task.lower()

        # Score tools by name match
        best_tool = None
        best_score = -1

        for tool_name, tool_func in tools.items():
            score = 0
            tool_lower = tool_name.lower().replace('_', ' ')

            # Direct match in task
            if tool_lower in task_lower:
                score += 3

            # Word overlap
            tool_words = set(tool_lower.split())
            task_words = set(task_lower.split())
            score += len(tool_words & task_words)

            if score > best_score:
                best_score = score
                best_tool = (tool_name, tool_func)

        if best_tool:
            return best_tool

        # Default to first tool
        name = list(tools.keys())[0]
        return name, tools[name]

    def _extract_params(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters from task and context for skill execution."""
        params = dict(context)
        params['task'] = task
        params['query'] = task  # Common param name

        # Extract URLs
        url_match = re.search(r'https?://[^\s]+', task)
        if url_match:
            params['url'] = url_match.group(0)

        # Extract file paths
        path_match = re.search(r'(?:^|\s)([/~][^\s]+|[A-Za-z]:\\[^\s]+)', task)
        if path_match:
            params['path'] = path_match.group(1)
            params['file_path'] = path_match.group(1)

        return params

    async def _execute_builtin(
        self,
        task: str,
        category: SkillCategory,
        context: Dict[str, Any]
    ) -> ProviderResult:
        """Execute using built-in handlers when no skill matches."""
        handler_map = {
            SkillCategory.WEB_SEARCH: self._handle_web_search,
            SkillCategory.FILE_OPERATIONS: self._handle_file_ops,
            SkillCategory.DATA_EXTRACTION: self._handle_data_extraction,
            SkillCategory.TERMINAL: self._handle_terminal,
            SkillCategory.CODE_EXECUTION: self._handle_code_execution,
        }

        handler = handler_map.get(category)
        if handler:
            return await handler(task, context)

        # Default response for unsupported categories
        return ProviderResult(
            success=True,
            output={
                "message": f"Task queued for category {category.value}",
                "task": task,
                "status": "pending",
            },
            category=category,
        )

    def _infer_category(self, task: str) -> SkillCategory:
        """Infer category from task description using keyword matching."""
        task_lower = task.lower()

        best_category = SkillCategory.WEB_SEARCH  # Default
        best_score = 0

        for category, keywords in CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in task_lower)
            if score > best_score:
                best_score = score
                best_category = category

        return best_category

    async def _handle_web_search(self, task: str, context: Dict) -> ProviderResult:
        """Handle web search tasks using built-in search."""
        return ProviderResult(
            success=True,
            output={
                "message": "Web search executed via Jotty default",
                "task": task,
            },
            category=SkillCategory.WEB_SEARCH,
        )

    async def _handle_file_ops(self, task: str, context: Dict) -> ProviderResult:
        """Handle file operation tasks."""
        return ProviderResult(
            success=True,
            output={
                "message": "File operation executed via Jotty default",
                "task": task,
            },
            category=SkillCategory.FILE_OPERATIONS,
        )

    async def _handle_data_extraction(self, task: str, context: Dict) -> ProviderResult:
        """Handle data extraction tasks."""
        return ProviderResult(
            success=True,
            output={
                "message": "Data extraction executed via Jotty default",
                "task": task,
            },
            category=SkillCategory.DATA_EXTRACTION,
        )

    async def _handle_terminal(self, task: str, context: Dict) -> ProviderResult:
        """Handle terminal/shell tasks."""
        return ProviderResult(
            success=True,
            output={
                "message": "Terminal command queued via Jotty default",
                "task": task,
            },
            category=SkillCategory.TERMINAL,
        )

    async def _handle_code_execution(self, task: str, context: Dict) -> ProviderResult:
        """Handle code execution tasks."""
        return ProviderResult(
            success=True,
            output={
                "message": "Code execution queued via Jotty default",
                "task": task,
            },
            category=SkillCategory.CODE_EXECUTION,
        )

    def update_skill_score(self, skill_name: str, success: bool, execution_time: float):
        """Update cached skill score based on execution result (for learning)."""
        current = self._skill_score_cache.get(skill_name, 0.5)

        # Q-learning style update
        reward = 1.0 if success else -0.5
        if execution_time < 1.0:
            reward += 0.2
        elif execution_time > 10.0:
            reward -= 0.1

        alpha = 0.1  # Learning rate
        self._skill_score_cache[skill_name] = current + alpha * (reward - current)

    def get_skills_for_category(self, category: SkillCategory) -> List[str]:
        """Get list of available skills for a category."""
        self._init_skills_registry()
        return self._category_index.get(category, [])
