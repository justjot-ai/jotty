"""
Browser-Use Provider for Jotty V2
==================================

Integrates the browser-use library for web automation.
https://github.com/browser-use/browser-use

Capabilities:
- Web browsing and navigation
- Form filling and submission
- Data extraction and scraping
- Screenshot capture
- Multi-tab support
"""

import time
import logging
import asyncio
from typing import Any, Dict, List, Optional

from .base import SkillProvider, SkillCategory, ProviderCapability, ProviderResult

logger = logging.getLogger(__name__)

# Try to import browser-use
try:
    from browser_use import Agent as BrowserAgent
    from browser_use import Browser, BrowserConfig
    BROWSER_USE_AVAILABLE = True
except ImportError:
    BROWSER_USE_AVAILABLE = False
    BrowserAgent = None
    Browser = None
    BrowserConfig = None


class BrowserUseProvider(SkillProvider):
    """
    Provider using browser-use library for web automation.

    Features:
    - Natural language browser control
    - Automatic element detection
    - Multi-step task execution
    - Session persistence
    """

    name = "browser-use"
    version = "0.11.0"
    description = "Web automation via browser-use library"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        self.capabilities = [
            ProviderCapability(
                category=SkillCategory.BROWSER,
                actions=["navigate", "click", "type", "scroll", "screenshot", "extract"],
                max_concurrent=3,
                requires_display=False,  # Can run headless
                requires_network=True,
                estimated_latency_ms=2000,
            ),
            ProviderCapability(
                category=SkillCategory.WEB_SEARCH,
                actions=["search", "browse_results", "extract_info"],
                estimated_latency_ms=3000,
            ),
            ProviderCapability(
                category=SkillCategory.DATA_EXTRACTION,
                actions=["scrape", "extract_text", "extract_links", "extract_tables"],
                estimated_latency_ms=2000,
            ),
            ProviderCapability(
                category=SkillCategory.FORM_AUTOMATION,
                actions=["fill_form", "submit", "login", "checkout"],
                estimated_latency_ms=5000,
            ),
        ]

        # Browser instance
        self._browser = None
        self._llm = None

        # Configuration
        self.headless = config.get('headless', True) if config else True
        self.timeout = config.get('timeout', 60) if config else 60

    async def initialize(self) -> bool:
        """Initialize browser-use."""
        if not BROWSER_USE_AVAILABLE:
            logger.warning("browser-use not installed. Run: pip install browser-use")
            self.is_available = False
            return False

        try:
            # Try to set up LLM
            await self._setup_llm()

            # Initialize browser config
            if BrowserConfig:
                browser_config = BrowserConfig(
                    headless=self.headless,
                )
                self._browser = Browser(config=browser_config)

            self.is_initialized = True
            self.is_available = True
            logger.info(f"✅ {self.name} provider initialized (headless={self.headless})")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize browser-use: {e}")
            self.is_available = False
            return False

    async def _setup_llm(self):
        """Set up LLM for browser-use agent."""
        try:
            # Try to use Claude CLI via our adapter
            from ...integration.direct_claude_cli_lm import DirectClaudeCLI
            import dspy
            self._llm = DirectClaudeCLI(model="sonnet")
            dspy.configure(lm=self._llm)
            logger.info("Using Claude CLI for browser-use")
        except Exception as e:
            logger.warning(f"Could not set up Claude CLI: {e}")
            # browser-use will use its own LLM configuration

    def get_categories(self) -> List[SkillCategory]:
        return [
            SkillCategory.BROWSER,
            SkillCategory.WEB_SEARCH,
            SkillCategory.DATA_EXTRACTION,
            SkillCategory.FORM_AUTOMATION,
        ]

    async def execute(self, task: str, context: Dict[str, Any] = None) -> ProviderResult:
        """Execute browser automation task."""
        if not BROWSER_USE_AVAILABLE:
            return ProviderResult(
                success=False,
                output=None,
                error="browser-use not installed",
                provider_name=self.name,
            )

        start_time = time.time()
        context = context or {}

        try:
            # Create agent for this task
            agent = BrowserAgent(
                task=task,
                llm=self._llm,
                browser=self._browser,
            )

            # Run the agent
            logger.info(f"🌐 browser-use executing: {task[:50]}...")
            result = await asyncio.wait_for(
                agent.run(),
                timeout=self.timeout
            )

            execution_time = time.time() - start_time

            # Extract output
            output = {
                'task': task,
                'result': result,
                'execution_time': execution_time,
            }

            provider_result = ProviderResult(
                success=True,
                output=output,
                execution_time=execution_time,
                provider_name=self.name,
                category=SkillCategory.BROWSER,
                confidence=0.9,
            )

            self.record_execution(provider_result)
            logger.info(f"✅ browser-use completed in {execution_time:.2f}s")
            return provider_result

        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            result = ProviderResult(
                success=False,
                output=None,
                error=f"Task timed out after {self.timeout}s",
                execution_time=execution_time,
                provider_name=self.name,
                retryable=True,
            )
            self.record_execution(result)
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"browser-use error: {e}")
            result = ProviderResult(
                success=False,
                output=None,
                error=str(e),
                execution_time=execution_time,
                provider_name=self.name,
                retryable=True,
            )
            self.record_execution(result)
            return result

    async def navigate(self, url: str) -> ProviderResult:
        """Navigate to a URL."""
        return await self.execute(f"Navigate to {url}")

    async def search(self, query: str, engine: str = "google") -> ProviderResult:
        """Perform a web search."""
        return await self.execute(f"Search for '{query}' on {engine} and return the results")

    async def extract_content(self, url: str, selectors: List[str] = None) -> ProviderResult:
        """Extract content from a webpage."""
        task = f"Go to {url} and extract the main content"
        if selectors:
            task += f" focusing on elements: {', '.join(selectors)}"
        return await self.execute(task)

    async def fill_form(self, url: str, form_data: Dict[str, str]) -> ProviderResult:
        """Fill and submit a form."""
        task = f"Go to {url} and fill the form with: "
        task += ", ".join([f"{k}='{v}'" for k, v in form_data.items()])
        task += ", then submit the form"
        return await self.execute(task)

    async def screenshot(self, url: str = None) -> ProviderResult:
        """Take a screenshot."""
        if url:
            return await self.execute(f"Navigate to {url} and take a screenshot")
        return await self.execute("Take a screenshot of the current page")

    async def close(self):
        """Close the browser."""
        if self._browser:
            try:
                await self._browser.close()
            except Exception as e:
                logger.warning(f"Error closing browser: {e}")
            self._browser = None


class BrowserUseCompositeProvider(SkillProvider):
    """
    Composite provider that combines browser-use with other tools.

    For complex tasks that need:
    - Browser automation + Data processing
    - Multi-step web workflows
    - Browser + Terminal operations
    """

    name = "browser-use-composite"
    version = "1.0.0"
    description = "Composite browser automation with data processing"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Sub-providers
        self.browser_provider = BrowserUseProvider(config)
        self.jotty_provider = None  # Lazy init

        self.capabilities = [
            ProviderCapability(
                category=SkillCategory.BROWSER,
                actions=["browse", "scrape", "process", "analyze"],
                max_concurrent=2,
                estimated_latency_ms=5000,
            ),
            ProviderCapability(
                category=SkillCategory.DATA_EXTRACTION,
                actions=["scrape", "transform", "analyze", "export"],
                estimated_latency_ms=5000,
            ),
        ]

    async def initialize(self) -> bool:
        """Initialize composite provider."""
        browser_ok = await self.browser_provider.initialize()

        from .base import JottyDefaultProvider
        self.jotty_provider = JottyDefaultProvider()
        await self.jotty_provider.initialize()

        self.is_initialized = browser_ok
        self.is_available = browser_ok
        return browser_ok

    def get_categories(self) -> List[SkillCategory]:
        return [SkillCategory.BROWSER, SkillCategory.DATA_EXTRACTION]

    async def execute(self, task: str, context: Dict[str, Any] = None) -> ProviderResult:
        """Execute composite task."""
        start_time = time.time()
        context = context or {}

        try:
            # Step 1: Browser task
            browser_result = await self.browser_provider.execute(task, context)

            if not browser_result.success:
                return browser_result

            # Step 2: Process the data with Jotty
            # (Could add data transformation, analysis, etc.)

            execution_time = time.time() - start_time

            return ProviderResult(
                success=True,
                output={
                    'browser_output': browser_result.output,
                    'processed': True,
                },
                execution_time=execution_time,
                provider_name=self.name,
                category=SkillCategory.BROWSER,
            )

        except Exception as e:
            return ProviderResult(
                success=False,
                output=None,
                error=str(e),
                execution_time=time.time() - start_time,
                provider_name=self.name,
            )
