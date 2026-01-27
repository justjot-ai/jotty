"""
SwarmResearcher - Autonomous Research Capability

Researches solutions, APIs, tools, and best practices.
Follows DRY: Reuses existing skills and tools where possible.
"""
import logging
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

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
        self._planner = None
    
    def _init_dependencies(self):
        """Lazy load dependencies (DRY: avoid circular imports)."""
        if self._skills_registry is None:
            from ...registry.skills_registry import get_skills_registry
            self._skills_registry = get_skills_registry()
            self._skills_registry.init()
        
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
        
        # Try to use web-search skill if available (DRY: reuse existing skills)
        findings = []
        tools_found = []
        apis_found = []
        documentation_urls = []
        
        # Check if web-search skill exists
        web_search_skill = self._skills_registry.get_skill("web-search")
        
        if web_search_skill:
            try:
                # Use existing web-search skill (DRY)
                search_result = await self._search_with_skill(web_search_skill, query)
                findings.extend(search_result.get("results", []))
            except Exception as e:
                logger.warning(f"⚠️  Web-search skill failed: {e}, falling back to LLM research")
        
        # Fallback to LLM-based research if skill unavailable
        if not findings:
            findings = await self._research_with_llm(query, research_type)
        
        # Extract tools and APIs from findings
        tools_found = self._extract_tools(findings)
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
                    import re
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
        import re
        
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
    
    def _extract_tools(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Extract tool names from findings."""
        tools = []
        for finding in findings:
            if finding.get('type') == 'tool' and 'name' in finding:
                tools.append(finding['name'])
            elif 'tool' in finding.get('text', '').lower():
                # Try to extract tool name from text
                text = finding.get('text', '')
                # Simple extraction (can be enhanced)
                if 'python' in text.lower() or 'pip install' in text.lower():
                    # Extract package name
                    import re
                    matches = re.findall(r'pip install (\w+)', text)
                    tools.extend(matches)
        
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
                import re
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
                import re
                url_matches = re.findall(r'https?://[^\s]+', finding.get('text', ''))
                urls.extend(url_matches)
        
        return list(set(urls))
    
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
