import requests
import re
import json
import time
import random
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import xml.etree.ElementTree as ET

from Jotty.core.utils.skill_status import SkillStatus

logger = logging.getLogger(__name__)

# Status emitter for progress updates
status = SkillStatus("arxiv-downloader")

# ---------------------------------------------------------------------------
# Anti-rate-limit: user-agent rotation
# ---------------------------------------------------------------------------
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36",
]


def _get_random_headers() -> Dict[str, str]:
    """Get random browser-like headers to reduce rate-limiting."""
    return {
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept": "application/xml, text/xml, */*",
        "Accept-Language": random.choice(["en-US,en;q=0.9", "en-GB,en;q=0.8", "en;q=0.7"]),
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
        "DNT": "1",
    }


# ---------------------------------------------------------------------------
# Caching layer
# ---------------------------------------------------------------------------
_CACHE_DIR = Path("/tmp/arxiv_cache")


def _get_cache_path(key: str) -> Path:
    """Get cache file path for a given key."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_key = key.replace("/", "_").replace(".", "_")
    return _CACHE_DIR / f"{safe_key}.json"


def _load_cache(key: str) -> Optional[Dict[str, Any]]:
    """Load cached result if available."""
    path = _get_cache_path(key)
    if path.exists():
        try:
            data = json.loads(path.read_text())
            logger.debug(f"Cache hit: {key}")
            return data
        except Exception:
            pass
    return None


def _save_cache(key: str, data: Dict[str, Any]) -> None:
    """Save result to cache."""
    path = _get_cache_path(key)
    try:
        path.write_text(json.dumps(data))
    except Exception as e:
        logger.debug(f"Cache save failed: {e}")


# ---------------------------------------------------------------------------
# Request helper with retry + backoff
# ---------------------------------------------------------------------------
def _request_with_retry(url: str, params: Optional[Dict] = None, max_retries: int = 4, timeout: int = 30) -> Optional[requests.Response]:
    """HTTP GET with exponential backoff and random headers."""
    import os
    proxy_url = os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY') or os.environ.get('ARXIV_PROXY')
    proxies = {'http': proxy_url, 'https': proxy_url} if proxy_url else None

    for attempt in range(max_retries):
        if attempt > 0:
            delay = (5 * (2 ** attempt)) + random.uniform(1, 5)
            logger.info(f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s delay...")
            time.sleep(delay)

        try:
            resp = requests.get(
                url,
                params=params,
                headers=_get_random_headers(),
                proxies=proxies,
                timeout=timeout,
            )
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429:
                logger.warning("Rate limited (429), will retry...")
                continue
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.debug(f"Request attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                return None
    return None



def download_arxiv_paper_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Download an arXiv paper and extract its content.
    
    Args:
        params: Dictionary containing:
            - arxiv_id (str, required): arXiv ID (e.g., '2010.11929') or URL
            - extract_mode (str, optional): 'text' (fast) or 'pdf' (slow), default: 'text'
            - output_dir (str, optional): Output directory, default: './output/arxiv'
            - clean_latex (bool, optional): Remove LaTeX commands, default: True
            - include_bibliography (bool, optional): Include bibliography, default: True
    
    Returns:
        Dictionary with:
            - success (bool): Whether download succeeded
            - arxiv_id (str): Extracted arXiv ID
            - title (str): Paper title
            - authors (list): List of authors
            - content (str): Extracted text content
            - output_path (str): Path to downloaded files
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        arxiv_input = params.get('arxiv_id')
        if not arxiv_input:
            return {
                'success': False,
                'error': 'arxiv_id parameter is required'
            }
        
        # Extract arXiv ID from various formats
        arxiv_id = _extract_arxiv_id(arxiv_input)
        if not arxiv_id:
            return {
                'success': False,
                'error': f'Invalid arXiv ID or URL: {arxiv_input}'
            }
        
        extract_mode = params.get('extract_mode', 'text')
        use_cache = params.get('use_cache', True)
        output_dir = Path(params.get('output_dir', './output/arxiv'))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check full-result cache
        cache_key = f"paper_{arxiv_id}_{extract_mode}"
        if use_cache:
            cached = _load_cache(cache_key)
            if cached:
                return cached
        
        # Fetch paper metadata
        metadata = _fetch_arxiv_metadata(arxiv_id, use_cache=use_cache)
        if not metadata:
            return {
                'success': False,
                'error': f'Failed to fetch metadata for arXiv:{arxiv_id}'
            }
        
        # Download source
        if extract_mode == 'text':
            # Download LaTeX source and extract text
            content = _download_latex_source(arxiv_id, output_dir)
            if not content:
                return {
                    'success': False,
                    'error': 'Failed to download LaTeX source'
                }
            
            # Clean LaTeX if requested
            if params.get('clean_latex', True):
                content = _clean_latex(content)
        else:
            # Download PDF (simpler approach - just return PDF URL)
            pdf_url = f'https://arxiv.org/pdf/{arxiv_id}.pdf'
            content = f'PDF available at: {pdf_url}'
        
        result = {
            'success': True,
            'arxiv_id': arxiv_id,
            'title': metadata.get('title', ''),
            'authors': metadata.get('authors', []),
            'abstract': metadata.get('abstract', ''),
            'categories': metadata.get('categories', []),
            'published': metadata.get('published', ''),
            'content': content[:50000],  # Limit content size
            'content_length': len(content),
            'output_path': str(output_dir),
            'pdf_url': metadata.get('pdf_url', f'https://arxiv.org/pdf/{arxiv_id}.pdf'),
            'arxiv_url': metadata.get('arxiv_url', f'https://arxiv.org/abs/{arxiv_id}'),
        }

        if use_cache:
            _save_cache(cache_key, result)

        return result
    except Exception as e:
        return {
            'success': False,
            'error': f'Error downloading arXiv paper: {str(e)}'
        }


def search_arxiv_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search arXiv for papers matching a query.
    
    Args:
        params: Dictionary containing:
            - query (str, required): Search query
            - max_results (int, optional): Maximum results, default: 10
            - sort_by (str, optional): Sort order, default: 'relevance'
    
    Returns:
        Dictionary with:
            - success (bool): Whether search succeeded
            - results (list): List of papers
            - count (int): Number of results
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        query = params.get('query')
        if not query:
            return {
                'success': False,
                'error': 'query parameter is required'
            }
        
        max_results = params.get('max_results', 10)
        sort_by = params.get('sort_by', 'relevance')
        
        # ArXiv API search with retry
        url = 'http://export.arxiv.org/api/query'
        params_data = {
            'search_query': query,
            'start': 0,
            'max_results': min(max_results, 100),
            'sortBy': sort_by,
        }
        
        response = _request_with_retry(url, params=params_data)
        if response is None:
            return {'success': False, 'error': 'ArXiv API unreachable after retries'}
        
        # Parse XML response
        root = ET.fromstring(response.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        ns_arxiv = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
        results = []
        for entry in root.findall('atom:entry', ns):
            arxiv_id = entry.find('atom:id', ns).text.split('/')[-1]
            title_elem = entry.find('atom:title', ns)
            title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else ''
            
            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns)
                if name is not None:
                    authors.append(name.text)
            
            abstract_elem = entry.find('atom:summary', ns)
            abstract = abstract_elem.text.strip() if abstract_elem is not None else ''
            
            published_elem = entry.find('atom:published', ns)
            published = published_elem.text[:10] if published_elem is not None else ''

            # Extract categories
            categories = []
            for cat in entry.findall('atom:category', ns_arxiv):
                if cat.get('term') and cat.get('term') not in categories:
                    categories.append(cat.get('term'))
            
            results.append({
                'id': arxiv_id,
                'title': title,
                'authors': authors,
                'abstract': abstract[:500],  # Limit abstract length
                'categories': categories,
                'published': published,
                'url': f'https://arxiv.org/abs/{arxiv_id}',
                'pdf_url': f'https://arxiv.org/pdf/{arxiv_id}.pdf',
            })
        
        return {
            'success': True,
            'results': results,
            'count': len(results),
            'query': query
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error searching arXiv: {str(e)}'
        }


def _extract_arxiv_id(input_str: str) -> Optional[str]:
    """Extract arXiv ID from various formats"""
    # Remove whitespace
    input_str = input_str.strip()
    
    # Handle URLs
    if 'arxiv.org' in input_str:
        # Extract ID from URL
        match = re.search(r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})', input_str)
        if match:
            return match.group(1)
        match = re.search(r'arxiv\.org/(?:abs|pdf)/([^/]+)', input_str)
        if match:
            return match.group(1).replace('.pdf', '')
    
    # Handle arxiv: prefix
    if input_str.startswith('arxiv:'):
        return input_str[6:]
    
    # Check if it's already a valid ID format
    if re.match(r'^\d{4}\.\d{4,5}$', input_str):
        return input_str
    
    # Try to extract from any string
    match = re.search(r'(\d{4}\.\d{4,5})', input_str)
    if match:
        return match.group(1)
    
    return None


def _fetch_arxiv_metadata(arxiv_id: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
    """Fetch paper metadata from arXiv API with caching and retry."""
    cache_key = f"meta_{arxiv_id}"
    if use_cache:
        cached = _load_cache(cache_key)
        if cached:
            return cached

    try:
        url = 'http://export.arxiv.org/api/query'
        params = {
            'id_list': arxiv_id,
            'max_results': 1,
        }

        response = _request_with_retry(url, params=params)
        if response is None:
            return None

        root = ET.fromstring(response.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}

        entry = root.find('atom:entry', ns)
        if entry is None:
            return None

        title = entry.find('atom:title', ns).text.strip().replace('\n', ' ') if entry.find('atom:title', ns) is not None else ''

        authors = []
        for author in entry.findall('atom:author', ns):
            name = author.find('atom:name', ns)
            if name is not None:
                authors.append(name.text)

        abstract_elem = entry.find('atom:summary', ns)
        abstract = abstract_elem.text.strip() if abstract_elem is not None else ''

        # Categories
        categories = []
        for cat in entry.findall('arxiv:primary_category', ns):
            if cat.get('term'):
                categories.append(cat.get('term'))
        for cat in entry.findall('atom:category', ns):
            if cat.get('term') and cat.get('term') not in categories:
                categories.append(cat.get('term'))

        # Published date
        published_elem = entry.find('atom:published', ns)
        published = published_elem.text[:10] if published_elem is not None else ''

        result = {
            'title': title,
            'authors': authors,
            'abstract': abstract,
            'categories': categories,
            'published': published,
            'pdf_url': f'https://arxiv.org/pdf/{arxiv_id}.pdf',
            'arxiv_url': f'https://arxiv.org/abs/{arxiv_id}',
        }

        if use_cache:
            _save_cache(cache_key, result)
        return result
    except Exception as e:
        logger.debug(f"Metadata fetch failed: {e}")
        return None


def _download_latex_source(arxiv_id: str, output_dir: Path) -> Optional[str]:
    """Download LaTeX source and extract text"""
    try:
        import tarfile
        import tempfile
        
        # Try to download source tar.gz
        source_url = f'https://arxiv.org/src/{arxiv_id}'
        
        response = _request_with_retry(source_url, timeout=60)
        if response.status_code == 200:
            # Save to file
            tar_path = output_dir / f'{arxiv_id}.tar.gz'
            tar_path.write_bytes(response.content)
            
            # Extract tar.gz and find main .tex file
            with tempfile.TemporaryDirectory() as temp_dir:
                with tarfile.open(tar_path, 'r:gz') as tar:
                    tar.extractall(temp_dir)
                
                # Find main .tex file (usually root.tex or paper.tex or arxiv_id.tex)
                tex_files = list(Path(temp_dir).rglob('*.tex'))
                main_tex = None
                
                # Prefer root.tex, paper.tex, or arxiv_id.tex
                for name in ['root.tex', 'paper.tex', f'{arxiv_id}.tex', 'main.tex']:
                    for tex_file in tex_files:
                        if tex_file.name.lower() == name.lower():
                            main_tex = tex_file
                            break
                    if main_tex:
                        break
                
                # If no preferred name found, use largest .tex file
                if not main_tex and tex_files:
                    main_tex = max(tex_files, key=lambda f: f.stat().st_size)
                
                if main_tex:
                    # Read and extract text from LaTeX (basic extraction)
                    tex_content = main_tex.read_text(encoding='utf-8', errors='ignore')
                    # Simple LaTeX to text conversion
                    text = _extract_text_from_latex(tex_content)
                    return text[:50000]  # Limit size
            
            return f'LaTeX source downloaded to {tar_path}. Content extraction may be limited.'
        else:
            # Fallback: return abstract from metadata
            metadata = _fetch_arxiv_metadata(arxiv_id)
            if metadata:
                return metadata.get('abstract', '')
            return None
    except Exception as e:
        # Fallback to abstract
        metadata = _fetch_arxiv_metadata(arxiv_id)
        if metadata:
            return metadata.get('abstract', '')
        return None


def _extract_text_from_latex(latex_content: str) -> str:
    """Extract readable text from LaTeX source"""
    text = latex_content
    
    # Remove comments
    text = re.sub(r'%.*?$', '', text, flags=re.MULTILINE)
    
    # Remove LaTeX commands (but keep content in braces)
    text = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})*', '', text)
    
    # Remove math environments
    text = re.sub(r'\$[^$]+\$', '[MATH]', text)
    text = re.sub(r'\\\[.*?\\\]', '[MATH]', text, flags=re.DOTALL)
    text = re.sub(r'\\\(.*?\\\)', '[MATH]', text, flags=re.DOTALL)
    
    # Extract text from braces (content)
    def extract_braces(match):
        content = match.group(1)
        # Remove nested commands
        content = re.sub(r'\\[a-zA-Z]+\*?', '', content)
        return content + ' '
    
    text = re.sub(r'\{([^{}]*)\}', extract_braces, text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()


def _clean_latex(text: str) -> str:
    """Remove LaTeX commands from text (basic cleaning)"""
    return _extract_text_from_latex(text)
