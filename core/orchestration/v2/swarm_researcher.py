"""
SwarmResearcher - Autonomous Research Capability

Researches solutions, APIs, tools, and best practices.
Includes provider discovery for auto-integration.
Follows DRY: Reuses existing skills and tools where possible.
"""
import logging
import re
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ResearchResult:
    """Result of research query."""
    query: str
    findings: List[Dict[str, Any]]
    tools_found: List[str]
    apis_found: List[str]
    documentation_urls: List[str]
    confidence: float


@dataclass
class ProviderCandidate:
    """A candidate provider discovered from search."""
    name: str
    package_name: str
    source: str  # "github", "pypi", "npm", "awesome-list"
    url: str
    description: str
    stars: int = 0
    downloads: int = 0
    last_updated: str = ""
    categories: List[str] = field(default_factory=list)
    install_command: str = ""
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SwarmResearcher:
    """
    Autonomous researcher for discovering solutions, APIs, and tools.
    
    DRY Principle: Reuses existing skills (web-search, etc.) when available.
    Falls back to LLM-based research when skills unavailable.
    """
    
    def __init__(self, config=None):
        """
        Initialize SwarmResearcher.
        
        Args:
            config: Optional JottyConfig
        """
        self.config = config
        self._skills_registry = None
        self._tools_registry = None
        self._planner = None
    
    def _init_dependencies(self):
        """Lazy load dependencies (DRY: avoid circular imports)."""
        if self._skills_registry is None:
            from ...registry.skills_registry import get_skills_registry
            self._skills_registry = get_skills_registry()
            self._skills_registry.init()
        
        if self._tools_registry is None:
            from ...registry.tools_registry import get_tools_registry
            self._tools_registry = get_tools_registry()
        
        if self._planner is None:
            from ...agents.agentic_planner import AgenticPlanner
            self._planner = AgenticPlanner()
    
    async def research(
        self,
        query: str,
        research_type: str = "general"
    ) -> ResearchResult:
        """
        Research a topic autonomously.
        
        Discovers solutions, APIs, tools, and best practices.
        
        Args:
            query: Research query (e.g., "Reddit API", "Python web scraping")
            research_type: Type of research ("api", "tool", "best_practice", "general")
            
        Returns:
            ResearchResult with findings, tools_found, apis_found, documentation_urls
            
        Example:
            result = await researcher.research("Reddit API")
            print(result.tools_found)  # ['praw', 'reddit-api']
            print(result.apis_found)  # ['Reddit API']
        """
        """
        Research a topic autonomously.
        
        Args:
            query: Research query (e.g., "Reddit API", "Python web scraping")
            research_type: Type of research ("api", "tool", "best_practice", "general")
            
        Returns:
            ResearchResult with findings
        """
        self._init_dependencies()
        
        logger.info(f"🔍 SwarmResearcher: Researching '{query}' (type: {research_type})")
        
        findings = []
        tools_found = []
        apis_found = []
        documentation_urls = []
        
        # STEP 1: Check skills registry first (DRY: reuse existing skills)
        registry_tools = self._search_registries(query, research_type)
        if registry_tools:
            logger.info(f"✅ Found {len(registry_tools)} tools in registries")
            tools_found.extend(registry_tools)
            # Create findings from registry results
            for tool in registry_tools:
                findings.append({
                    'type': 'tool',
                    'name': tool,
                    'text': f"Found in registry: {tool}",
                    'source': 'registry'
                })
        
        # STEP 2: Check web-search skill if available
        web_search_skill = self._skills_registry.get_skill("web-search")
        if web_search_skill:
            try:
                search_result = await self._search_with_skill(web_search_skill, query)
                search_findings = search_result.get("results", [])
                if search_findings:
                    findings.extend(search_findings)
                    logger.info(f"✅ Found {len(search_findings)} results from web-search skill")
            except Exception as e:
                logger.debug(f"Web-search skill failed: {e}")
        
        # STEP 3: Only use LLM research if registries didn't find anything
        if not tools_found and not findings:
            logger.info("ℹ️  No tools found in registries, using LLM research as fallback")
            llm_findings = await self._research_with_llm(query, research_type)
            if llm_findings:
                findings.extend(llm_findings)
                # Validate LLM suggestions against registries
                llm_tools = self._extract_tools(llm_findings)
                validated_tools = self._validate_tools_against_registry(llm_tools)
                tools_found.extend(validated_tools)
        else:
            # Extract additional tools from findings if any
            additional_tools = self._extract_tools(findings)
            validated_additional = self._validate_tools_against_registry(additional_tools)
            tools_found.extend([t for t in validated_additional if t not in tools_found])
        
        # Extract APIs and documentation from findings
        apis_found = self._extract_apis(findings)
        documentation_urls = self._extract_documentation(findings)
        
        confidence = min(0.9, len(findings) * 0.1)  # Simple confidence metric
        
        return ResearchResult(
            query=query,
            findings=findings,
            tools_found=tools_found,
            apis_found=apis_found,
            documentation_urls=documentation_urls,
            confidence=confidence
        )
    
    async def _search_with_skill(self, skill, query: str) -> Dict[str, Any]:
        """Use existing web-search skill (DRY: reuse)."""
        try:
            # Execute skill's search tool
            if hasattr(skill, 'execute'):
                result = await skill.execute({"query": query})
                return result
            
            # Get tools - SkillDefinition.tools is Dict[str, Callable]
            tools = getattr(skill, 'tools', {})
            if not tools:
                # Try to get tools from skill definition
                if hasattr(skill, 'definition') and hasattr(skill.definition, 'tools'):
                    tools = skill.definition.tools
            
            # Handle dictionary of tools: {tool_name: callable}
            if isinstance(tools, dict):
                for tool_name, tool_func in tools.items():
                    # Check if this is a search/web tool
                    if 'search' in tool_name.lower() or 'web' in tool_name.lower():
                        # Execute tool function
                        try:
                            if asyncio.iscoroutinefunction(tool_func):
                                result = await tool_func(query=query)
                            else:
                                result = tool_func(query=query)
                            
                            # Ensure result is a dict
                            if isinstance(result, dict):
                                return result
                            else:
                                return {"results": [result] if result else []}
                        except Exception as e:
                            logger.debug(f"Tool {tool_name} execution failed: {e}")
                            continue
            elif isinstance(tools, list):
                # Handle list of tools (legacy format)
                for tool in tools:
                    tool_name = tool.name if hasattr(tool, 'name') else str(tool)
                    if 'search' in tool_name.lower() or 'web' in tool_name.lower():
                        if callable(tool):
                            if asyncio.iscoroutinefunction(tool):
                                return await tool(query=query)
                            else:
                                return tool(query=query)
        except Exception as e:
            logger.debug(f"Skill execution failed: {e}")
            import traceback
            logger.debug(f"   Traceback: {traceback.format_exc()}")
        
        return {"results": []}
    
    async def _research_with_llm(self, query: str, research_type: str) -> List[Dict[str, Any]]:
        """Fallback LLM-based research."""
        self._init_dependencies()
        
        # Use planner for LLM-based research (DRY: reuse AgenticPlanner)
        research_prompt = f"""
Research the following: {query}
Research type: {research_type}

Provide a structured response with:
1. Relevant tools/libraries (list package names)
2. APIs available (list API names)
3. Documentation links (list URLs)
4. Best practices (brief summary)

Format your response clearly with sections for tools, APIs, and documentation.
"""
        
        # Use planner's LLM capability (DRY: reuse existing LLM access)
        try:
            import dspy
            if hasattr(dspy, 'settings') and dspy.settings.lm:
                lm = dspy.settings.lm
                # Use shorter timeout for research
                try:
                    if hasattr(lm, '__call__'):
                        response = lm(research_prompt, timeout=30)
                    else:
                        response = lm(research_prompt)
                    
                    # Handle list responses
                    if isinstance(response, list):
                        response = ' '.join(str(r) for r in response)
                    elif not isinstance(response, str):
                        response = str(response)
                    
                    # Parse response into structured findings
                    findings = self._parse_llm_response(response)
                    
                    # If no findings, try keyword-based extraction
                    if not findings:
                        findings = self._extract_findings_from_keywords(response, query, research_type)
                    
                    return findings
                except TimeoutError:
                    logger.warning("LLM research timed out")
                    return []
        except Exception as e:
            logger.warning(f"LLM research failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        return []
    
    def _parse_llm_response(self, response) -> List[Dict[str, Any]]:
        """Parse LLM response into structured findings."""
        findings = []
        
        # Handle both string and list responses (DSPy can return lists)
        if isinstance(response, list):
            # DSPy returns list of strings, join them
            response = ' '.join(str(r) for r in response)
        elif not isinstance(response, str):
            response = str(response)
        
        # Enhanced parsing with multiple strategies
        lines = response.split('\n')
        current_section = None
        current_finding = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_finding:
                    findings.append(current_finding)
                    current_finding = {}
                continue
            
            # Detect sections
            line_lower = line.lower()
            if 'tool' in line_lower and ('list' in line_lower or ':' in line):
                current_section = 'tools'
                continue
            elif 'api' in line_lower and ('list' in line_lower or ':' in line):
                current_section = 'apis'
                continue
            elif 'documentation' in line_lower or 'docs' in line_lower:
                current_section = 'docs'
                continue
            
            # Parse bullet points
            if line.startswith('-') or line.startswith('*') or line.startswith('•'):
                content = re.sub(r'^[-*•]\s*', '', line).strip()
                
                # Extract tool/library names
                if current_section == 'tools' or 'tool' in line_lower or 'library' in line_lower:
                    # Extract package name (often in format "package-name" or "package_name")
                    package_match = re.search(r'([a-zA-Z0-9_-]+)', content)
                    if package_match:
                        findings.append({
                            'type': 'tool',
                            'name': package_match.group(1),
                            'text': content
                        })
                
                # Extract API names
                elif current_section == 'apis' or 'api' in line_lower:
                    api_match = re.search(r'([A-Z][a-zA-Z\s]+)\s+API', content, re.IGNORECASE)
                    if api_match:
                        findings.append({
                            'type': 'api',
                            'name': api_match.group(1).strip(),
                            'text': content
                        })
                    else:
                        # Try to extract API name directly
                        api_name = content.split()[0] if content.split() else content
                        findings.append({
                            'type': 'api',
                            'name': api_name,
                            'text': content
                        })
                
                # Extract URLs
                elif current_section == 'docs' or 'http' in line_lower:
                    url_match = re.search(r'https?://[^\s\)]+', content)
                    if url_match:
                        findings.append({
                            'type': 'documentation',
                            'url': url_match.group(0),
                            'text': content
                        })
            
            # Look for inline mentions
            else:
                # Check for package names (pip install style)
                pip_match = re.search(r'pip install\s+([a-zA-Z0-9_-]+)', line_lower)
                if pip_match:
                    findings.append({
                        'type': 'tool',
                        'name': pip_match.group(1),
                        'text': line
                    })
                
                # Check for API mentions
                api_match = re.search(r'([A-Z][a-zA-Z\s]+)\s+API', line, re.IGNORECASE)
                if api_match:
                    findings.append({
                        'type': 'api',
                        'name': api_match.group(1).strip(),
                        'text': line
                    })
                
                # Check for URLs
                url_match = re.search(r'https?://[^\s\)]+', line)
                if url_match:
                    findings.append({
                        'type': 'documentation',
                        'url': url_match.group(0),
                        'text': line
                    })
        
        if current_finding:
            findings.append(current_finding)
        
        return findings
    
    def _extract_findings_from_keywords(self, response: str, query: str, research_type: str) -> List[Dict[str, Any]]:
        """Extract findings using keyword matching as fallback."""
        findings = []
        
        # Common tool/library patterns
        tool_patterns = [
            r'\b([a-z]+[-_][a-z]+)\b',  # kebab-case or snake_case
            r'pip install\s+([a-zA-Z0-9_-]+)',
            r'install\s+([a-zA-Z0-9_-]+)',
        ]
        
        # Common API patterns
        api_patterns = [
            r'([A-Z][a-zA-Z\s]+)\s+API',
            r'API:\s*([A-Z][a-zA-Z]+)',
        ]
        
        # Extract tools
        for pattern in tool_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                findings.append({
                    'type': 'tool',
                    'name': match.lower().replace(' ', '-'),
                    'text': f"Found tool: {match}"
                })
        
        # Extract APIs
        for pattern in api_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                findings.append({
                    'type': 'api',
                    'name': match.strip(),
                    'text': f"Found API: {match}"
                })
        
        return findings
    
    def _is_valid_tool_name(self, name: str) -> bool:
        """Check if tool name is valid (not a generic word or built-in module)."""
        if not name or len(name) < 2:
            return False
        
        # Filter out generic words
        invalid_names = {
            'multiple', 'various', 'several', 'many', 'some', 'few', 'all',
            'each', 'every', 'both', 'either', 'neither', 'other', 'another',
            'this', 'that', 'these', 'those', 'such', 'same', 'different',
            'new', 'old', 'first', 'last', 'next', 'previous', 'current',
            'common', 'standard', 'basic', 'advanced', 'simple', 'complex',
            'good', 'best', 'better', 'great', 'excellent', 'perfect',
            'important', 'useful', 'helpful', 'effective', 'efficient',
            'popular', 'famous', 'well-known', 'widely', 'commonly',
            'available', 'possible', 'suitable', 'appropriate', 'relevant',
            'example', 'instance', 'case', 'situation', 'scenario',
            'method', 'way', 'approach', 'technique', 'strategy',
            'tool', 'library', 'package', 'module', 'framework',
            'solution', 'option', 'alternative', 'choice', 'selection'
        }
        
        if name.lower() in invalid_names:
            return False
        
        # Check if it's a built-in Python module
        import sys
        builtin_modules = {
            'multiprocessing', 'threading', 'asyncio', 'json', 'os', 'sys',
            'datetime', 'time', 'random', 'math', 'collections', 'itertools'
        }
        if name.lower() in builtin_modules:
            return False
        
        # Must contain at least one letter
        if not re.search(r'[a-zA-Z]', name):
            return False
        
        return True
    
    def _extract_tools(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Extract tool names from findings."""
        tools = []
        for finding in findings:
            if finding.get('type') == 'tool' and 'name' in finding:
                tool_name = finding['name']
                if self._is_valid_tool_name(tool_name):
                    tools.append(tool_name)
            elif 'tool' in finding.get('text', '').lower():
                # Try to extract tool name from text
                text = finding.get('text', '')
                # Simple extraction (can be enhanced)
                if 'python' in text.lower() or 'pip install' in text.lower():
                    # Extract package name
                    matches = re.findall(r'pip install (\w+)', text)
                    for match in matches:
                        if self._is_valid_tool_name(match):
                            tools.append(match)
        
        return list(set(tools))  # Remove duplicates
    
    def _extract_apis(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Extract API names from findings."""
        apis = []
        for finding in findings:
            if finding.get('type') == 'api' and 'name' in finding:
                apis.append(finding['name'])
            elif 'api' in finding.get('text', '').lower():
                text = finding.get('text', '')
                # Extract API names (simple pattern matching)
                matches = re.findall(r'(\w+)\s+API', text, re.IGNORECASE)
                apis.extend(matches)
        
        return list(set(apis))
    
    def _extract_documentation(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Extract documentation URLs from findings."""
        urls = []
        for finding in findings:
            if 'url' in finding:
                urls.append(finding['url'])
            elif 'http' in finding.get('text', ''):
                url_matches = re.findall(r'https?://[^\s]+', finding.get('text', ''))
                urls.extend(url_matches)
        
        return list(set(urls))
    
    def _search_registries(self, query: str, research_type: str) -> List[str]:
        """
        Search skills and tools registries for matching tools.
        
        This is the PRIMARY method - checks actual registries before guessing.
        """
        tools_found = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Search skills registry
        try:
            # Get all skills (returns List[Dict[str, Any]])
            if hasattr(self._skills_registry, 'list_skills'):
                all_skills_data = self._skills_registry.list_skills()
                # Extract skill names from dicts
                skill_names = [s.get('name', s) if isinstance(s, dict) else s for s in all_skills_data]
            else:
                # Fallback: try to get all registered skills
                skill_names = []
                if hasattr(self._skills_registry, '_skills'):
                    skill_names = list(self._skills_registry._skills.keys())
            
            # Search by name and description
            for skill_name in skill_names:
                if isinstance(skill_name, dict):
                    skill_name = skill_name.get('name', '')
                
                skill = self._skills_registry.get_skill(skill_name)
                if skill:
                    # Check if query matches skill name
                    skill_name_lower = skill_name.lower()
                    if query_lower in skill_name_lower or skill_name_lower in query_lower:
                        tools_found.append(skill_name)
                        continue
                    
                    # Check skill description if available
                    if hasattr(skill, 'description'):
                        desc = str(skill.description).lower()
                        if query_lower in desc or any(word in desc for word in query_words):
                            tools_found.append(skill_name)
                            continue
                    
                    # Check skill tags/keywords if available
                    if hasattr(skill, 'tags') and skill.tags:
                        tags = [str(t).lower() for t in skill.tags]
                        if any(query_lower in tag or tag in query_lower or any(w in tag for w in query_words) for tag in tags):
                            tools_found.append(skill_name)
        except Exception as e:
            logger.debug(f"Skills registry search failed: {e}")
        
        # Search tools registry
        try:
            # Get all tools (returns List[ToolSchema])
            all_tools = self._tools_registry.get_all() if hasattr(self._tools_registry, 'get_all') else []
            for tool in all_tools:
                tool_name = tool.name if hasattr(tool, 'name') else str(tool)
                tool_name_lower = tool_name.lower()
                
                # Check name match
                if query_lower in tool_name_lower or tool_name_lower in query_lower:
                    tools_found.append(tool_name)
                    continue
                
                # Check description match
                if hasattr(tool, 'description'):
                    desc = str(tool.description).lower()
                    if query_lower in desc or any(word in desc for word in query_words):
                        tools_found.append(tool_name)
                        continue
                
                # Check category match
                if hasattr(tool, 'category'):
                    category = str(tool.category).lower()
                    if query_lower in category or category in query_lower:
                        tools_found.append(tool_name)
        except Exception as e:
            logger.debug(f"Tools registry search failed: {e}")
        
        return list(set(tools_found))  # Remove duplicates
    
    def _validate_tools_against_registry(self, tools: List[str]) -> List[str]:
        """
        Validate LLM-suggested tools against actual registries.
        
        Only returns tools that exist in registries or are valid installable packages.
        """
        validated = []
        
        for tool in tools:
            if not self._is_valid_tool_name(tool):
                continue
            
            # Check if tool exists in registries
            try:
                # Check skills registry
                skill = self._skills_registry.get_skill(tool)
                if skill:
                    validated.append(tool)
                    continue
                
                # Check tools registry (uses get() method)
                if hasattr(self._tools_registry, 'get'):
                    tool_obj = self._tools_registry.get(tool)
                    if tool_obj:
                        validated.append(tool)
                        continue
            except Exception:
                pass
            
            # If not in registry, it might still be a valid installable package
            # (e.g., pip/npm packages), so we keep it but log
            logger.debug(f"Tool '{tool}' not found in registries, but may be installable")
            validated.append(tool)
        
        return validated
    
    async def find_solutions(self, requirement: str) -> ResearchResult:
        """
        Find solutions for a requirement (alias for research with "tool" type).

        Convenience method for tool discovery.

        Args:
            requirement: Requirement description (e.g., "web scraping", "API integration")

        Returns:
            ResearchResult with tools and solutions

        Example:
            result = await researcher.find_solutions("web scraping")
            print(result.tools_found)  # ['scrapy', 'beautifulsoup4', 'selenium']
        """
        return await self.research(requirement, research_type="tool")

    # =========================================================================
    # Provider Discovery Methods
    # =========================================================================

    async def discover_providers(
        self,
        capability: str,
        max_results: int = 10
    ) -> List[ProviderCandidate]:
        """
        Discover providers for a capability by searching GitHub, PyPI, and awesome-lists.

        Args:
            capability: Capability description (e.g., "PDF OCR", "web scraping")
            max_results: Maximum number of results to return

        Returns:
            List of ProviderCandidate sorted by relevance

        Example:
            providers = await researcher.discover_providers("PDF OCR")
            for p in providers:
                print(f"{p.name}: {p.description} (stars: {p.stars})")
        """
        logger.info(f"🔍 Discovering providers for: {capability}")

        # Search multiple sources in parallel
        tasks = [
            self._search_github_providers(capability),
            self._search_pypi_providers(capability),
            self._search_awesome_lists(capability),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect all candidates
        all_candidates: List[ProviderCandidate] = []

        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Provider search failed: {result}")
                continue
            if isinstance(result, list):
                all_candidates.extend(result)

        # Rank and deduplicate
        ranked = self._rank_providers(all_candidates, capability)

        # Return top results
        return ranked[:max_results]

    async def _search_github_providers(self, capability: str) -> List[ProviderCandidate]:
        """
        Search GitHub for relevant repositories.

        Uses GitHub search API to find Python packages related to the capability.
        """
        candidates = []

        try:
            import aiohttp

            # Build search query
            query = f"{capability} python library language:python"
            url = f"https://api.github.com/search/repositories"
            params = {
                'q': query,
                'sort': 'stars',
                'order': 'desc',
                'per_page': 20,
            }

            headers = {
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'Jotty-SwarmResearcher/2.0',
            }

            # Add GitHub token if available
            import os
            github_token = os.getenv('GITHUB_TOKEN')
            if github_token:
                headers['Authorization'] = f'token {github_token}'

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers, timeout=30) as response:
                    if response.status != 200:
                        logger.warning(f"GitHub search failed: {response.status}")
                        return candidates

                    data = await response.json()

                    for item in data.get('items', []):
                        # Extract package name (often repo name)
                        package_name = item.get('name', '').lower().replace('-', '_')

                        candidate = ProviderCandidate(
                            name=item.get('name', ''),
                            package_name=package_name,
                            source='github',
                            url=item.get('html_url', ''),
                            description=item.get('description', '') or '',
                            stars=item.get('stargazers_count', 0),
                            last_updated=item.get('updated_at', ''),
                            install_command=f"pip install {package_name}",
                            metadata={
                                'full_name': item.get('full_name', ''),
                                'language': item.get('language', ''),
                                'topics': item.get('topics', []),
                                'forks': item.get('forks_count', 0),
                                'open_issues': item.get('open_issues_count', 0),
                            }
                        )
                        candidates.append(candidate)

            logger.debug(f"GitHub search found {len(candidates)} candidates")

        except ImportError:
            logger.debug("aiohttp not available for GitHub search")
        except Exception as e:
            logger.warning(f"GitHub search error: {e}")

        return candidates

    async def _search_pypi_providers(self, capability: str) -> List[ProviderCandidate]:
        """
        Search PyPI for relevant packages.

        Uses PyPI JSON API to find Python packages.
        """
        candidates = []

        try:
            import aiohttp

            # PyPI doesn't have a search API, use the warehouse API
            # We'll search for common package names based on capability
            search_terms = capability.lower().replace(' ', '-').split('-')

            # Try direct package lookup for common patterns
            package_patterns = [
                capability.lower().replace(' ', '-'),
                capability.lower().replace(' ', '_'),
                f"python-{capability.lower().replace(' ', '-')}",
                f"py{capability.lower().replace(' ', '')}",
            ]

            async with aiohttp.ClientSession() as session:
                for package_name in package_patterns:
                    try:
                        url = f"https://pypi.org/pypi/{package_name}/json"
                        async with session.get(url, timeout=10) as response:
                            if response.status != 200:
                                continue

                            data = await response.json()
                            info = data.get('info', {})

                            # Calculate downloads (approximate from releases)
                            downloads = 0
                            releases = data.get('releases', {})
                            if releases:
                                latest = list(releases.values())[-1]
                                if latest:
                                    downloads = sum(r.get('downloads', 0) for r in latest)

                            candidate = ProviderCandidate(
                                name=info.get('name', package_name),
                                package_name=info.get('name', package_name),
                                source='pypi',
                                url=info.get('project_url', f"https://pypi.org/project/{package_name}/"),
                                description=info.get('summary', '') or '',
                                downloads=downloads,
                                last_updated=info.get('version', ''),
                                install_command=f"pip install {info.get('name', package_name)}",
                                metadata={
                                    'version': info.get('version', ''),
                                    'author': info.get('author', ''),
                                    'license': info.get('license', ''),
                                    'requires_python': info.get('requires_python', ''),
                                    'keywords': info.get('keywords', ''),
                                }
                            )
                            candidates.append(candidate)

                    except Exception as e:
                        logger.debug(f"PyPI lookup for {package_name} failed: {e}")
                        continue

            logger.debug(f"PyPI search found {len(candidates)} candidates")

        except ImportError:
            logger.debug("aiohttp not available for PyPI search")
        except Exception as e:
            logger.warning(f"PyPI search error: {e}")

        return candidates

    async def _search_awesome_lists(self, capability: str) -> List[ProviderCandidate]:
        """
        Search awesome-lists for curated tools.

        Parses awesome-python and similar lists for relevant packages.
        """
        candidates = []

        # Map capabilities to awesome-list sections
        capability_lower = capability.lower()
        awesome_mappings = {
            'pdf': 'awesome-python/README.md#pdf',
            'ocr': 'awesome-python/README.md#ocr',
            'web scraping': 'awesome-python/README.md#web-crawling',
            'scraping': 'awesome-python/README.md#web-crawling',
            'browser': 'awesome-python/README.md#browser-automation',
            'automation': 'awesome-python/README.md#browser-automation',
            'api': 'awesome-python/README.md#restful-api',
            'database': 'awesome-python/README.md#database',
            'image': 'awesome-python/README.md#image-processing',
            'video': 'awesome-python/README.md#video',
            'audio': 'awesome-python/README.md#audio',
            'machine learning': 'awesome-python/README.md#machine-learning',
            'ml': 'awesome-python/README.md#machine-learning',
            'nlp': 'awesome-python/README.md#natural-language-processing',
            'text': 'awesome-python/README.md#text-processing',
        }

        # Find matching awesome-list section
        matched_section = None
        for key, section in awesome_mappings.items():
            if key in capability_lower:
                matched_section = section
                break

        if not matched_section:
            logger.debug(f"No awesome-list mapping for: {capability}")
            return candidates

        try:
            import aiohttp

            # Fetch the awesome-list
            url = f"https://raw.githubusercontent.com/vinta/awesome-python/master/README.md"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    if response.status != 200:
                        logger.warning(f"Awesome-list fetch failed: {response.status}")
                        return candidates

                    content = await response.text()

                    # Parse markdown to find packages in relevant section
                    section_name = matched_section.split('#')[-1].replace('-', ' ')
                    candidates = self._parse_awesome_list_section(content, section_name)

            logger.debug(f"Awesome-list search found {len(candidates)} candidates")

        except ImportError:
            logger.debug("aiohttp not available for awesome-list search")
        except Exception as e:
            logger.warning(f"Awesome-list search error: {e}")

        return candidates

    def _parse_awesome_list_section(
        self,
        content: str,
        section_name: str
    ) -> List[ProviderCandidate]:
        """Parse awesome-list markdown to extract packages from a section."""
        candidates = []

        # Find section
        section_pattern = rf'## {re.escape(section_name)}.*?\n(.*?)(?=\n## |\Z)'
        section_match = re.search(section_pattern, content, re.IGNORECASE | re.DOTALL)

        if not section_match:
            return candidates

        section_content = section_match.group(1)

        # Extract package entries (format: * [name](url) - description)
        entry_pattern = r'\*\s*\[([^\]]+)\]\(([^)]+)\)\s*-?\s*(.*)'

        for match in re.finditer(entry_pattern, section_content):
            name = match.group(1)
            url = match.group(2)
            description = match.group(3).strip()

            # Infer package name from URL
            package_name = name.lower().replace('-', '_').replace(' ', '_')
            if 'github.com' in url:
                # Extract from GitHub URL
                parts = url.rstrip('/').split('/')
                if len(parts) >= 2:
                    package_name = parts[-1].lower().replace('-', '_')

            candidate = ProviderCandidate(
                name=name,
                package_name=package_name,
                source='awesome-list',
                url=url,
                description=description,
                install_command=f"pip install {package_name}",
                metadata={'section': section_name}
            )
            candidates.append(candidate)

        return candidates

    def _rank_providers(
        self,
        candidates: List[ProviderCandidate],
        capability: str
    ) -> List[ProviderCandidate]:
        """
        Rank provider candidates by relevance to capability.

        Scoring factors:
        - Keyword match with capability
        - GitHub stars / PyPI downloads
        - Recency of updates
        - Source credibility (awesome-list > github > pypi)
        """
        capability_lower = capability.lower()
        capability_words = set(capability_lower.split())

        # Deduplicate by package name
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c.package_name not in seen:
                seen.add(c.package_name)
                unique_candidates.append(c)

        for candidate in unique_candidates:
            score = 0.0

            # Keyword match in name
            name_lower = candidate.name.lower()
            name_words = set(name_lower.replace('-', ' ').replace('_', ' ').split())
            name_overlap = len(capability_words & name_words)
            score += name_overlap * 10

            # Keyword match in description
            if candidate.description:
                desc_lower = candidate.description.lower()
                if capability_lower in desc_lower:
                    score += 15
                else:
                    desc_words = set(desc_lower.split())
                    desc_overlap = len(capability_words & desc_words)
                    score += desc_overlap * 3

            # Source credibility
            source_scores = {
                'awesome-list': 20,  # Curated = trustworthy
                'github': 10,
                'pypi': 5,
            }
            score += source_scores.get(candidate.source, 0)

            # Popularity (normalized)
            if candidate.stars > 0:
                import math
                score += math.log10(candidate.stars + 1) * 2

            if candidate.downloads > 0:
                import math
                score += math.log10(candidate.downloads + 1)

            candidate.relevance_score = score

        # Sort by relevance score descending
        unique_candidates.sort(key=lambda c: c.relevance_score, reverse=True)

        return unique_candidates

    async def get_best_provider(self, capability: str) -> Optional[ProviderCandidate]:
        """
        Get the best provider for a capability.

        Convenience method that returns top-ranked provider.

        Args:
            capability: Capability description

        Returns:
            Best ProviderCandidate or None
        """
        providers = await self.discover_providers(capability, max_results=1)
        return providers[0] if providers else None
