from typing import Dict, Any
import re


def generate_mindmap_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a Mermaid mindmap diagram from text content.
    
    Uses simple text analysis to create a hierarchical mindmap structure.
    For advanced LLM-based generation, use Jotty's LLM integration.
    
    Args:
        params: Dictionary containing:
            - content (str, required): Text content to analyze
            - title (str, optional): Mindmap title, default: 'Mindmap'
            - style (str, optional): Style - 'hierarchical' or 'radial', default: 'hierarchical'
            - max_branches (int, optional): Maximum main branches (3-10), default: 7
            - max_depth (int, optional): Maximum depth, default: 3
    
    Returns:
        Dictionary with:
            - success (bool): Whether generation succeeded
            - mindmap_code (str): Mermaid mindmap code
            - title (str): Mindmap title
            - output_format (str): Output format
            - error (str, optional): Error message if failed
    """
    try:
        content = params.get('content')
        if not content:
            return {
                'success': False,
                'error': 'content parameter is required'
            }
        
        title = params.get('title', 'Mindmap')
        style = params.get('style', 'hierarchical')
        max_branches = min(max(params.get('max_branches', 7), 3), 10)
        max_depth = params.get('max_depth', 3)
        
        # Simple text analysis to extract key concepts
        # Split into sentences and extract important words/phrases
        sentences = re.split(r'[.!?]+', content)
        
        # Extract key phrases (simple approach)
        key_phrases = []
        for sentence in sentences[:50]:  # Limit analysis
            sentence = sentence.strip()
            if len(sentence) > 10:  # Skip very short sentences
                # Extract capitalized words or important phrases
                words = sentence.split()
                if len(words) >= 3:
                    # Take first few words as key phrase
                    phrase = ' '.join(words[:5])
                    if phrase not in key_phrases:
                        key_phrases.append(phrase)
                        if len(key_phrases) >= max_branches:
                            break
        
        # Generate Mermaid mindmap syntax
        mindmap_lines = ['mindmap']
        mindmap_lines.append(f'  root(({title}))')
        
        # Add branches
        for i, phrase in enumerate(key_phrases[:max_branches]):
            # Clean phrase for mindmap syntax
            clean_phrase = phrase.replace('"', "'").replace('\n', ' ')[:50]
            mindmap_lines.append(f'    {clean_phrase}')
        
        mindmap_code = '\n'.join(mindmap_lines)
        
        return {
            'success': True,
            'mindmap_code': mindmap_code,
            'title': title,
            'output_format': 'mermaid',
            'branches': len(key_phrases[:max_branches])
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error generating mindmap: {str(e)}'
        }
