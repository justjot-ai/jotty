from typing import Dict, Any, List, Optional
import re


def chunk_text_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Split text into semantic chunks for RAG systems.
    
    Uses recursive character splitting with markdown-aware separators.
    
    Args:
        params: Dictionary containing:
            - text (str, required): Text to chunk
            - chunk_size (int, optional): Maximum chunk size in characters, default: 500
            - chunk_overlap (int, optional): Overlap between chunks, default: 100
            - separators (list, optional): Custom separators (default: markdown-aware)
            - preserve_headers (bool, optional): Include headers in context, default: True
            - token_limit (int, optional): Token limit instead of character limit
    
    Returns:
        Dictionary with:
            - success (bool): Whether chunking succeeded
            - chunks (list): List of chunk dicts
            - count (int): Number of chunks
            - error (str, optional): Error message if failed
    """
    try:
        text = params.get('text')
        if not text:
            return {
                'success': False,
                'error': 'text parameter is required'
            }
        
        chunk_size = params.get('chunk_size', 500)
        chunk_overlap = params.get('chunk_overlap', 100)
        separators = params.get('separators', [
            '\n\n\n',  # Triple newline
            '\n\n',    # Double newline
            '\n',      # Single newline
            '. ',      # Sentence end
            '! ', '? ', '; ', ': ', ', ', ' ', ''  # Other separators
        ])
        preserve_headers = params.get('preserve_headers', True)
        
        # Simple chunking implementation (can be enhanced with langchain-text-splitters)
        chunks = []
        current_chunk = ''
        current_size = 0
        chunk_index = 0
        
        # Split by paragraphs first (preserve structure)
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if adding this paragraph would exceed chunk size
            if current_size + len(para) + 2 > chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'index': chunk_size,
                    'text': current_chunk.strip(),
                    'char_count': len(current_chunk.strip())
                })
                chunk_index += 1
                
                # Start new chunk with overlap
                if chunk_overlap > 0 and current_chunk:
                    # Take last part of previous chunk for overlap
                    overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
                    current_chunk = overlap_text + '\n\n' + para
                    current_size = len(current_chunk)
                else:
                    current_chunk = para
                    current_size = len(para)
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += '\n\n' + para
                    current_size += len(para) + 2
                else:
                    current_chunk = para
                    current_size = len(para)
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'index': chunk_index,
                'text': current_chunk.strip(),
                'char_count': len(current_chunk.strip())
            })
        
        return {
            'success': True,
            'chunks': chunks,
            'count': len(chunks),
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error chunking text: {str(e)}'
        }
