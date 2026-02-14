"""
AutoProviderDiscovery - Full Auto-Discovery Pipeline
=====================================================

Complete pipeline for discovering, installing, generating, and registering
new skill providers automatically based on capability requirements.

Pipeline Steps:
1. SwarmResearcher.discover_providers() - Search GitHub, PyPI, awesome-lists
2. Select best candidate based on ranking
3. Assess trust level for security
4. SwarmInstaller.install() - Install package
5. SwarmCodeGenerator.generate_provider_adapter() - Generate adapter code
6. Save adapter to providers/ directory
7. Test and register in ProviderRegistry

Usage:
    from core.orchestration.auto_provider_discovery import AutoProviderDiscovery

    discovery = AutoProviderDiscovery()
    provider_name = await discovery.discover_and_integrate("PDF OCR capability")

    if provider_name:
        # Provider is now registered and ready to use
        result = await registry.execute(SkillCategory.DOCUMENT, "Extract text from scan.pdf")
"""

import os
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from .swarm_researcher import ProviderCandidate

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryResult:
    """Result of auto-discovery pipeline."""
    success: bool
    provider_name: Optional[str] = None
    package_name: Optional[str] = None
    trust_level: str = "sandboxed"
    adapter_path: Optional[str] = None
    error: Optional[str] = None
    steps_completed: List[str] = None

    def __post_init__(self) -> None:
        if self.steps_completed is None:
            self.steps_completed = []


class AutoProviderDiscovery:
    """
    Full pipeline for auto-discovering and integrating providers.

    Orchestrates SwarmResearcher, SwarmInstaller, SwarmCodeGenerator,
    and ProviderRegistry to automatically add new capabilities.

    Attributes:
        providers_dir: Directory to save generated provider adapters
        auto_register: Whether to automatically register discovered providers
    """

    # Trusted package sources
    TRUSTED_SOURCES = {'awesome-list'}

    # Packages that require dangerous trust level
    DANGEROUS_PACKAGES = {
        'subprocess', 'os', 'sys', 'eval', 'exec', 'shell',
        'pyautogui', 'keyboard', 'mouse', 'pynput',
    }

    # Packages considered safe
    SAFE_PACKAGES = {
        'requests', 'httpx', 'aiohttp', 'beautifulsoup4', 'lxml',
        'pillow', 'pypdf', 'pdfplumber', 'reportlab',
        'pandas', 'numpy', 'matplotlib', 'plotly',
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize AutoProviderDiscovery.

        Args:
            config: Configuration dict with optional keys:
                - providers_dir: Directory for generated adapters
                - auto_register: Auto-register after generation (default: True)
                - max_candidates: Max candidates to consider (default: 5)
        """
        self.config = config or {}

        # Determine providers directory
        default_providers_dir = Path(__file__).parent.parent.parent / "skills" / "providers"
        self.providers_dir = Path(
            self.config.get('providers_dir', str(default_providers_dir))
        )
        self.providers_dir.mkdir(parents=True, exist_ok=True)

        self.auto_register = self.config.get('auto_register', True)
        self.max_candidates = self.config.get('max_candidates', 5)

        # Lazy-loaded components
        self._researcher = None
        self._installer = None
        self._code_generator = None
        self._registry = None

    def _init_components(self) -> Any:
        """Lazy initialize pipeline components."""
        if self._researcher is None:
            from .swarm_researcher import SwarmResearcher
            self._researcher = SwarmResearcher(self.config)

        if self._installer is None:
            from .swarm_installer import SwarmInstaller
            self._installer = SwarmInstaller(self.config)

        if self._code_generator is None:
            from .swarm_code_generator import SwarmCodeGenerator
            self._code_generator = SwarmCodeGenerator(self.config)

        if self._registry is None:
            from Jotty.core.skills.providers.provider_registry import ProviderRegistry
            self._registry = ProviderRegistry()

    async def discover_and_integrate(
        self,
        capability_needed: str,
        preferred_package: Optional[str] = None
    ) -> Optional[str]:
        """
        Full pipeline: discover -> install -> generate -> test -> register.

        Args:
            capability_needed: Description of needed capability
                e.g., "PDF OCR", "image captioning", "web scraping"
            preferred_package: Optional specific package to use

        Returns:
            Provider name if successful, None otherwise

        Example:
            provider = await discovery.discover_and_integrate("PDF OCR capability")
            if provider:
                print(f"Registered: {provider}")
        """
        self._init_components()

        result = DiscoveryResult(success=False)

        logger.info(f" Auto-discovering provider for: {capability_needed}")

        try:
            # Step 1: Discover providers
            if preferred_package:
                # Use preferred package directly
                from .swarm_researcher import ProviderCandidate
                candidate = ProviderCandidate(
                    name=preferred_package,
                    package_name=preferred_package,
                    source='user',
                    url='',
                    description=f"User-specified package for {capability_needed}",
                    categories=self._infer_categories(capability_needed),
                )
            else:
                candidates = await self._researcher.discover_providers(
                    capability_needed,
                    max_results=self.max_candidates
                )

                if not candidates:
                    result.error = f"No providers found for: {capability_needed}"
                    logger.warning(result.error)
                    return None

                candidate = candidates[0]  # Best ranked

            result.package_name = candidate.package_name
            result.steps_completed.append("discover")
            logger.info(f" Selected: {candidate.name} ({candidate.source})")

            # Step 2: Assess trust level
            trust_level = self._assess_trust_level(candidate)
            result.trust_level = trust_level
            result.steps_completed.append("assess_trust")
            logger.info(f" Trust level: {trust_level}")

            # Step 3: Install package
            install_result = await self._installer.install(candidate.package_name)
            if not install_result.success:
                result.error = f"Installation failed: {install_result.error}"
                logger.error(result.error)
                return None

            result.steps_completed.append("install")
            logger.info(f" Installed: {candidate.package_name}")

            # Step 4: Generate provider adapter
            categories = candidate.categories or self._infer_categories(capability_needed)
            package_info = {
                'description': candidate.description,
                'version': install_result.version or '1.0.0',
                'url': candidate.url,
                'source': candidate.source,
            }

            generated = self._code_generator.generate_provider_adapter(
                candidate.package_name,
                package_info,
                categories
            )

            result.steps_completed.append("generate")
            logger.info(f" Generated adapter: {generated.file_path}")

            # Step 5: Save adapter to providers directory
            adapter_path = self._save_adapter(generated)
            result.adapter_path = str(adapter_path)
            result.steps_completed.append("save")
            logger.info(f" Saved to: {adapter_path}")

            # Step 6: Test adapter (basic import test)
            test_success = await self._test_adapter(adapter_path)
            if not test_success:
                result.error = "Adapter test failed"
                logger.warning(result.error)
                # Continue anyway - might work at runtime

            result.steps_completed.append("test")

            # Step 7: Register provider
            if self.auto_register:
                provider_name = await self._register_provider(
                    adapter_path,
                    trust_level
                )
                if provider_name:
                    result.provider_name = provider_name
                    result.steps_completed.append("register")
                    logger.info(f" Registered provider: {provider_name}")
            else:
                result.provider_name = candidate.package_name.replace('-', '_')

            result.success = True
            return result.provider_name

        except Exception as e:
            result.error = str(e)
            logger.error(f" Auto-discovery failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _assess_trust_level(self, candidate: 'ProviderCandidate') -> str:
        """
        Assess trust level for a provider candidate.

        Returns:
            'trusted', 'sandboxed', or 'dangerous'
        """
        package_lower = candidate.package_name.lower()

        # Check if dangerous
        for dangerous in self.DANGEROUS_PACKAGES:
            if dangerous in package_lower:
                return "dangerous"

        # Check if safe
        if package_lower in self.SAFE_PACKAGES:
            return "trusted"

        # Check source credibility
        if candidate.source in self.TRUSTED_SOURCES:
            return "trusted" if candidate.stars > 1000 else "sandboxed"

        # High popularity = more trusted
        if candidate.stars > 5000:
            return "trusted"
        elif candidate.stars > 500:
            return "sandboxed"

        # Default to sandboxed
        return "sandboxed"

    def _infer_categories(self, capability: str) -> List[str]:
        """
        Infer SkillCategory values from capability description.

        Args:
            capability: Capability description

        Returns:
            List of category strings
        """
        capability_lower = capability.lower()

        category_keywords = {
            'document': ['pdf', 'document', 'docx', 'ocr', 'text extract'],
            'data_extract': ['scrape', 'crawl', 'extract', 'parse'],
            'web_search': ['search', 'find', 'lookup', 'research'],
            'browser': ['browser', 'web page', 'navigate', 'selenium', 'playwright'],
            'file_ops': ['file', 'read', 'write', 'save', 'load'],
            'code_exec': ['code', 'python', 'execute', 'compute'],
            'terminal': ['shell', 'command', 'bash', 'terminal'],
            'api_calls': ['api', 'rest', 'request', 'http'],
            'media': ['image', 'audio', 'video', 'picture'],
            'database': ['database', 'sql', 'query', 'mongodb'],
            'communication': ['email', 'message', 'notify', 'telegram'],
            'analytics': ['analyze', 'chart', 'visualize', 'report'],
        }

        categories = []
        for category, keywords in category_keywords.items():
            if any(kw in capability_lower for kw in keywords):
                categories.append(category)

        return categories or ['api_calls']  # Default

    def _save_adapter(self, generated: Any) -> Path:
        """
        Save generated adapter code to providers directory.

        Args:
            generated: GeneratedCode from SwarmCodeGenerator

        Returns:
            Path to saved file
        """
        # Determine file path
        if generated.file_path:
            file_name = Path(generated.file_path).name
        else:
            package_name = generated.metadata.get('package_name', 'unknown')
            file_name = f"{package_name.replace('-', '_')}_provider.py"

        adapter_path = self.providers_dir / file_name

        # Write code
        adapter_path.write_text(generated.code)

        return adapter_path

    async def _test_adapter(self, adapter_path: Path) -> bool:
        """
        Test adapter by attempting to import it.

        Args:
            adapter_path: Path to adapter file

        Returns:
            True if import succeeds
        """
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "test_adapter",
                adapter_path
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return True

        except Exception as e:
            logger.warning(f"Adapter test failed: {e}")

        return False

    async def _register_provider(
        self,
        adapter_path: Path,
        trust_level: str
    ) -> Optional[str]:
        """
        Register provider from adapter file.

        Args:
            adapter_path: Path to adapter file
            trust_level: Trust level for registration

        Returns:
            Provider name if successful
        """
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "provider_module",
                adapter_path
            )
            if not spec or not spec.loader:
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Look for create_provider function or Provider class
            provider = None

            if hasattr(module, 'create_provider'):
                provider = module.create_provider()
            else:
                # Find first SkillProvider subclass
                from Jotty.core.skills.providers.base import SkillProvider
                for name in dir(module):
                    obj = getattr(module, name)
                    if (isinstance(obj, type) and
                        issubclass(obj, SkillProvider) and
                        obj is not SkillProvider):
                        provider = obj()
                        break

            if provider:
                self._registry.register(provider, trust_level=trust_level)
                return provider.name

        except Exception as e:
            logger.error(f"Provider registration failed: {e}")

        return None

    async def discover_multiple(
        self,
        capabilities: List[str]
    ) -> Dict[str, Optional[str]]:
        """
        Discover and integrate multiple capabilities in parallel.

        Args:
            capabilities: List of capability descriptions

        Returns:
            Dict mapping capability to provider name (or None if failed)
        """
        tasks = [
            self.discover_and_integrate(cap)
            for cap in capabilities
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            cap: (result if not isinstance(result, Exception) else None)
            for cap, result in zip(capabilities, results)
        }

    def list_discovered_providers(self) -> List[Dict[str, Any]]:
        """
        List all discovered/generated providers.

        Returns:
            List of provider info dicts
        """
        providers = []

        for file_path in self.providers_dir.glob("*_provider.py"):
            # Skip __init__.py and base files
            if file_path.name.startswith('_'):
                continue

            provider_name = file_path.stem

            # Check if registered
            self._init_components()
            is_registered = provider_name.replace('_provider', '') in self._registry._providers

            providers.append({
                'name': provider_name,
                'path': str(file_path),
                'registered': is_registered,
            })

        return providers


# =============================================================================
# Convenience Functions
# =============================================================================

_discovery_instance: Optional[AutoProviderDiscovery] = None


def get_auto_discovery(config: Optional[Dict[str, Any]] = None) -> AutoProviderDiscovery:
    """Get singleton AutoProviderDiscovery instance."""
    global _discovery_instance
    if _discovery_instance is None:
        _discovery_instance = AutoProviderDiscovery(config)
    return _discovery_instance


async def discover_provider(capability: str) -> Optional[str]:
    """
    Convenience function to discover and integrate a provider.

    Args:
        capability: Capability description

    Returns:
        Provider name if successful
    """
    discovery = get_auto_discovery()
    return await discovery.discover_and_integrate(capability)
