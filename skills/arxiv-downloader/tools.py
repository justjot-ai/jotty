import requests
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import xml.etree.ElementTree as ET


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
        output_dir = Path(params.get('output_dir', './output/arxiv'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Fetch paper metadata
        metadata = _fetch_arxiv_metadata(arxiv_id)
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
        
        return {
            'success': True,
            'arxiv_id': arxiv_id,
            'title': metadata.get('title', ''),
            'authors': metadata.get('authors', []),
            'abstract': metadata.get('abstract', ''),
            'content': content[:50000],  # Limit content size
            'content_length': len(content),
            'output_path': str(output_dir),
            'pdf_url': f'https://arxiv.org/pdf/{arxiv_id}.pdf'
        }
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
    try:
        query = params.get('query')
        if not query:
            return {
                'success': False,
                'error': 'query parameter is required'
            }
        
        max_results = params.get('max_results', 10)
        sort_by = params.get('sort_by', 'relevance')
        
        # ArXiv API search
        url = 'http://export.arxiv.org/api/query'
        params_data = {
            'search_query': query,
            'start': 0,
            'max_results': min(max_results, 100),
            'sortBy': sort_by
        }
        
        response = requests.get(url, params=params_data, timeout=30)
        response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(response.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        results = []
        for entry in root.findall('atom:entry', ns):
            arxiv_id = entry.find('atom:id', ns).text.split('/')[-1]
            title = entry.find('atom:title', ns).text.strip()
            
            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns).text
                authors.append(name)
            
            abstract_elem = entry.find('atom:summary', ns)
            abstract = abstract_elem.text.strip() if abstract_elem is not None else ''
            
            published = entry.find('atom:published', ns).text if entry.find('atom:published', ns) is not None else ''
            
            results.append({
                'id': arxiv_id,
                'title': title,
                'authors': authors,
                'abstract': abstract[:500],  # Limit abstract length
                'published': published,
                'url': f'https://arxiv.org/abs/{arxiv_id}',
                'pdf_url': f'https://arxiv.org/pdf/{arxiv_id}.pdf'
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


def _fetch_arxiv_metadata(arxiv_id: str) -> Optional[Dict[str, Any]]:
    """Fetch paper metadata from arXiv API"""
    try:
        url = f'http://export.arxiv.org/api/query'
        params = {
            'id_list': arxiv_id,
            'max_results': 1
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        entry = root.find('atom:entry', ns)
        if entry is None:
            return None
        
        title = entry.find('atom:title', ns).text.strip() if entry.find('atom:title', ns) is not None else ''
        
        authors = []
        for author in entry.findall('atom:author', ns):
            name = author.find('atom:name', ns).text
            authors.append(name)
        
        abstract_elem = entry.find('atom:summary', ns)
        abstract = abstract_elem.text.strip() if abstract_elem is not None else ''
        
        return {
            'title': title,
            'authors': authors,
            'abstract': abstract
        }
    except Exception:
        return None


def _download_latex_source(arxiv_id: str, output_dir: Path) -> Optional[str]:
    """Download LaTeX source and extract text"""
    try:
        import tarfile
        import tempfile
        
        # Try to download source tar.gz
        source_url = f'https://arxiv.org/src/{arxiv_id}'
        
        response = requests.get(source_url, timeout=60, allow_redirects=True)
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
