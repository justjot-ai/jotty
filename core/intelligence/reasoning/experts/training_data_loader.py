"""
Training Data Loader

Loads training examples from various sources (GitHub, files, etc.)
and converts them to gold_standards format for expert training.
"""

import logging
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import urllib.request
import urllib.parse

logger = logging.getLogger(__name__)


class TrainingDataLoader:
    """Loads training examples from various sources."""
    
    def __init__(self, domain: str, validator: Optional[Any] = None) -> None:
        """
        Initialize training data loader.
        
        Args:
            domain: Domain name (e.g., "plantuml", "mermaid")
            validator: Optional domain validator for validation
        """
        self.domain = domain
        self.validator = validator
    
    def load_from_github_repo(
        self,
        repo_url: str,
        path: str = "",
        file_pattern: str = "*.puml",
        max_files: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Load training examples from GitHub repository.
        
        Args:
            repo_url: GitHub repository URL (e.g., "https://github.com/user/repo")
            path: Path within repository (e.g., "examples/")
            file_pattern: File pattern to match (e.g., "*.puml", "*.md")
            max_files: Maximum number of files to load
        
        Returns:
            List of training examples in format: {"code", "description", "type", "source"}
        """
        examples = []
        
        try:
            # Extract repo info from URL
            # Format: https://github.com/user/repo or https://github.com/user/repo.git
            repo_match = re.search(r'github\.com/([^/]+)/([^/]+)', repo_url)
            if not repo_match:
                logger.error(f"Invalid GitHub URL: {repo_url}")
                return examples
            
            owner = repo_match.group(1)
            repo = repo_match.group(2).replace('.git', '')
            
            # Use GitHub API to list files
            api_path = path.rstrip('/') if path else ""
            api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{api_path}"
            
            logger.info(f"Loading examples from GitHub: {repo_url}/{path}")
            logger.debug(f"API URL: {api_url}")
            
            # Fetch file list
            req = urllib.request.Request(api_url)
            req.add_header('Accept', 'application/vnd.github.v3+json')
            req.add_header('User-Agent', 'Jotty-TrainingDataLoader/1.0')
            
            with urllib.request.urlopen(req, timeout=10) as response:
                files_data = json.loads(response.read().decode('utf-8'))
            
            logger.debug(f"Received {len(files_data)} items from GitHub API")
            
            # Recursively search for files
            matching_files = []
            
            def search_directory(dir_path: str, current_depth: int = 0, max_depth: int = 2) -> None:
                """Recursively search directories for matching files."""
                if current_depth > max_depth or len(matching_files) >= max_files:
                    return
                
                try:
                    dir_api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{dir_path}" if dir_path else api_url
                    dir_req = urllib.request.Request(dir_api_url)
                    dir_req.add_header('Accept', 'application/vnd.github.v3+json')
                    dir_req.add_header('User-Agent', 'Jotty-TrainingDataLoader/1.0')
                    
                    with urllib.request.urlopen(dir_req, timeout=10) as dir_response:
                        dir_items = json.loads(dir_response.read().decode('utf-8'))
                        
                        for item in dir_items:
                            if not isinstance(item, dict):
                                continue
                            
                            item_type = item.get('type', '')
                            item_name = item.get('name', '')
                            
                            # Skip hidden/system directories
                            if item_name.startswith('.'):
                                continue
                            
                            if item_type == 'file' and self._matches_pattern(item_name, file_pattern):
                                matching_files.append(item)
                                if len(matching_files) >= max_files:
                                    return
                            
                            elif item_type == 'dir' and current_depth < max_depth:
                                # Recurse into subdirectories (skip common non-example dirs)
                                if item_name.lower() not in ['.git', 'node_modules', 'bin', '__pycache__']:
                                    subdir_path = f"{dir_path}/{item_name}" if dir_path else item_name
                                    search_directory(subdir_path, current_depth + 1, max_depth)
                                    
                                    if len(matching_files) >= max_files:
                                        return
                
                except urllib.error.HTTPError as e:
                    if e.code == 403:
                        logger.warning("GitHub API rate limit exceeded - consider using GitHub token")
                        raise  # Re-raise to stop searching
                    logger.debug(f"Failed to search directory {dir_path}: {e.code} {e.reason}")
                except Exception as e:
                    logger.debug(f"Failed to search directory {dir_path}: {e}")
            
            # Start recursive search
            try:
                search_directory(path if path else "", max_depth=2)
            except urllib.error.HTTPError as e:
                if e.code == 403:
                    logger.error("GitHub API rate limit exceeded. Please wait or use GitHub token.")
                    return examples  # Return empty list if rate limited
            
            matching_files = matching_files[:max_files]
            logger.info(f"Found {len(matching_files)} matching files")
            
            # Load each file
            for file_info in matching_files:
                try:
                    file_url = file_info.get('download_url')
                    if not file_url:
                        logger.debug(f"No download_url for {file_info.get('name')}")
                        continue
                    
                    # Download file content
                    logger.debug(f"Downloading {file_info.get('name')} from {file_url}")
                    with urllib.request.urlopen(file_url, timeout=10) as file_response:
                        code = file_response.read().decode('utf-8')
                    
                    if not code or len(code.strip()) == 0:
                        logger.debug(f"Empty file: {file_info.get('name')}")
                        continue
                    
                    # Extract description from filename or first comment
                    description = self._extract_description(file_info.get('name', ''), code)
                    
                    # Detect type
                    diagram_type = self._detect_type(code)
                    
                    examples.append({
                        "code": code,
                        "description": description,
                        "type": diagram_type,
                        "source": f"github:{owner}/{repo}",
                        "file_path": file_info.get('path', '')
                    })
                    
                    logger.debug(f"Loaded example: {file_info.get('name')} (type: {diagram_type})")
                    
                except Exception as e:
                    logger.warning(f"Failed to load file {file_info.get('name')}: {e}")
                    continue
            
            logger.info(f"Loaded {len(examples)} examples from GitHub")
            
        except urllib.error.HTTPError as e:
            logger.error(f"HTTP error loading from GitHub: {e.code} {e.reason}")
            if e.code == 404:
                logger.error(f"Repository or path not found: {api_url}")
            elif e.code == 403:
                logger.error("GitHub API rate limit exceeded or access denied")
        except Exception as e:
            logger.error(f"Failed to load from GitHub: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        return examples
    
    def validate_examples(
        self,
        examples: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Validate examples using domain validator.
        
        Args:
            examples: List of examples to validate
        
        Returns:
            Tuple of (valid_examples, invalid_examples)
        """
        valid = []
        invalid = []
        
        for example in examples:
            code = example.get('code', '')
            diagram_type = example.get('type', 'unknown')
            
            if self.validator:
                try:
                    is_valid, error, metadata = self.validator.validate(
                        output=code,
                        expected_type=diagram_type,
                        context={"task": example.get('description', '')}
                    )
                    
                    if is_valid:
                        valid.append(example)
                    else:
                        invalid.append({**example, "validation_error": error})
                except Exception as e:
                    logger.warning(f"Validation error: {e}")
                    # Include anyway if validation fails
                    valid.append(example)
            else:
                # No validator - accept all
                valid.append(example)
        
        return valid, invalid
    
    def convert_to_gold_standards(
        self,
        examples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert examples format to gold_standards format.
        
        Args:
            examples: List of examples with {"code", "description", "type"}
        
        Returns:
            List of gold_standards with {"task", "context", "gold_standard"}
        """
        gold_standards = []
        
        for example in examples:
            code = example.get('code', '')
            description = example.get('description', '')
            diagram_type = example.get('type', 'unknown')
            
            # Create task from description
            task = f"Generate {diagram_type} diagram: {description}"
            
            gold_standards.append({
                "task": task,
                "context": {
                    "description": description,
                    "diagram_type": diagram_type,
                    "source": example.get('source', 'unknown')
                },
                "gold_standard": code
            })
        
        return gold_standards
    
    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Check if filename matches pattern."""
        import fnmatch
        
        # Normalize pattern - handle common PlantUML extensions
        normalized_pattern = pattern.lower()
        if normalized_pattern == "*.puml":
            # Also match .plantuml and .pu extensions
            return (filename.lower().endswith('.puml') or 
                   filename.lower().endswith('.plantuml') or
                   filename.lower().endswith('.pu'))
        
        return fnmatch.fnmatch(filename.lower(), normalized_pattern)
    
    def _extract_description(self, filename: str, code: str) -> str:
        """Extract description from filename or code."""
        # Try filename first (remove extension)
        description = Path(filename).stem.replace('_', ' ').replace('-', ' ')
        
        # Try to extract from code comments
        if self.domain == "plantuml":
            # Look for @startuml title or comments
            title_match = re.search(r'title\s+(.+)', code, re.IGNORECASE)
            if title_match:
                description = title_match.group(1).strip()
        
        return description or filename
    
    def _detect_type(self, code: str) -> str:
        """Detect diagram type from code."""
        code_lower = code.lower()
        
        if self.domain == "plantuml":
            if 'sequence' in code_lower or 'participant' in code_lower:
                return "sequence"
            elif 'class' in code_lower and 'diagram' in code_lower:
                return "class"
            elif 'state' in code_lower or 'statechart' in code_lower:
                return "state"
            elif 'activity' in code_lower:
                return "activity"
            elif 'component' in code_lower:
                return "component"
            else:
                return "unknown"
        
        elif self.domain == "mermaid":
            first_line = code.split('\n')[0].lower()
            if 'gitgraph' in first_line:
                return "gitGraph"
            elif 'sequence' in first_line:
                return "sequenceDiagram"
            elif 'state' in first_line:
                return "stateDiagram-v2"
            elif 'gantt' in first_line:
                return "gantt"
            elif 'er' in first_line or 'entity' in first_line:
                return "erDiagram"
            elif 'journey' in first_line:
                return "journey"
            elif 'class' in first_line:
                return "classDiagram"
            elif 'graph' in first_line or 'flowchart' in first_line:
                return "flowchart"
            else:
                return "unknown"
        
        return "unknown"
