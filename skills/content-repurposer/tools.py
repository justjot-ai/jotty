from typing import Dict, Any, List
import re

from Jotty.core.utils.skill_status import SkillStatus

# Status emitter for progress updates
status = SkillStatus("content-repurposer")



def repurpose_content_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Repurpose content for different platforms.
    
    Simple text-based repurposing. For advanced LLM-based repurposing,
    use Jotty's LLM integration.
    
    Args:
        params: Dictionary containing:
            - content (str, required): Source content
            - outputs (list, required): List of output formats
            - title (str, optional): Content title
            - custom_settings (dict, optional): Platform-specific settings
    
    Returns:
        Dictionary with:
            - success (bool): Whether repurposing succeeded
            - outputs (dict): Repurposed content for each format
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        content = params.get('content')
        if not content:
            return {
                'success': False,
                'error': 'content parameter is required'
            }
        
        outputs = params.get('outputs', [])
        if not outputs:
            return {
                'success': False,
                'error': 'outputs parameter is required (list of formats)'
            }
        
        title = params.get('title', 'Content')
        custom_settings = params.get('custom_settings', {})
        
        # Extract key sentences
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        repurposed = {}
        
        for output_format in outputs:
            if output_format == 'twitter_thread':
                # Create Twitter thread (8-10 tweets, ~280 chars each)
                tweets = []
                chunk_size = 250  # Leave room for numbering
                current_chunk = ''
                tweet_num = 1
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 20 < chunk_size:
                        current_chunk += sentence + '. '
                    else:
                        if current_chunk:
                            tweets.append(f"{tweet_num}/{len(sentences)//3 + 1} {current_chunk.strip()}")
                            tweet_num += 1
                            current_chunk = sentence + '. '
                        if len(tweets) >= 10:
                            break
                
                if current_chunk and len(tweets) < 10:
                    tweets.append(f"{tweet_num}/{len(tweets) + 1} {current_chunk.strip()}")
                
                repurposed['twitter_thread'] = '\n\n'.join(tweets[:10])
            
            elif output_format == 'linkedin_post':
                # LinkedIn post (~1500 chars)
                post = f"{title}\n\n"
                post += '\n\n'.join(sentences[:10])
                post = post[:1500]
                repurposed['linkedin_post'] = post
            
            elif output_format == 'linkedin_carousel':
                # Carousel outline (8 slides)
                slides = [f"Slide {i+1}: {sentences[i][:100] if i < len(sentences) else 'Content'}" 
                         for i in range(8)]
                repurposed['linkedin_carousel'] = '\n'.join(slides)
            
            elif output_format == 'blog_excerpt':
                # Blog excerpt with hook
                hook = sentences[0] if sentences else content[:200]
                excerpt = f"{hook}\n\n"
                excerpt += '\n'.join(sentences[1:5])
                repurposed['blog_excerpt'] = excerpt[:500]
            
            elif output_format == 'email_newsletter':
                # Newsletter format
                newsletter = f"# {title}\n\n"
                newsletter += "## Summary\n\n"
                newsletter += sentences[0] + '\n\n'
                newsletter += "## Key Points\n\n"
                for i, sentence in enumerate(sentences[1:6], 1):
                    newsletter += f"{i}. {sentence}\n"
                repurposed['email_newsletter'] = newsletter
        
        return {
            'success': True,
            'outputs': repurposed,
            'formats_generated': list(repurposed.keys())
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error repurposing content: {str(e)}'
        }
