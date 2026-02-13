"""
Claude CLI LLM Skill

Text generation, summarization, and prompt ensembling using DSPy LM.
Uses the configured DSPy language model for all operations.

Prompt Ensembling Strategies (based on latest research):
- self_consistency: Same prompt, multiple samples, majority vote
- multi_perspective: Different personas/lenses analyze same question
- gsa: Generative Self-Aggregation (diverse generation + synthesis)
- debate: Multi-round debate between perspectives
"""
import logging
import json
import asyncio
import time as _time
from typing import Dict, Any, List, Optional
from collections import Counter

from Jotty.core.utils.skill_status import SkillStatus

logger = logging.getLogger(__name__)


# =============================================================================
# RATE LIMIT AWARE LLM CALLING
# =============================================================================

# Tracks RPM budget across calls within this module.
# Reset automatically when a minute boundary is crossed.
_rpm_state = {
    'call_count': 0,
    'window_start': 0.0,
    'detected_rpm_limit': None,  # Set when we get a 429
    'last_429_at': 0.0,
}


def _call_lm_with_retry(lm, *, prompt: str = None, messages=None,
                         max_retries: int = 4, base_delay: float = 8.0,
                         **kwargs):
    """
    Call LM with exponential backoff on rate-limit (429) errors.

    Tracks call cadence and auto-inserts pre-call delays when we know
    the provider has a tight RPM limit (e.g. 8 RPM on OpenRouter free).
    """
    import time

    # Auto-pace: if we've seen a 429 recently, pre-delay to stay under RPM
    now = time.time()
    rpm_limit = _rpm_state.get('detected_rpm_limit')
    if rpm_limit and rpm_limit <= 20:
        # Calculate safe interval: 60s / rpm_limit with 20% headroom
        safe_interval = 60.0 / rpm_limit * 1.2
        elapsed_since_last = now - _rpm_state.get('last_call_at', 0)
        if elapsed_since_last < safe_interval:
            wait = safe_interval - elapsed_since_last
            logger.debug(f"Rate-limit pacing: waiting {wait:.1f}s (RPM limit: {rpm_limit})")
            time.sleep(wait)

    for attempt in range(max_retries + 1):
        try:
            _rpm_state['last_call_at'] = time.time()
            if prompt:
                result = lm(prompt=prompt, **kwargs)
            elif messages:
                result = lm(messages=messages, **kwargs)
            else:
                result = lm(prompt="", **kwargs)
            _rpm_state['call_count'] += 1
            return result
        except Exception as e:
            err_str = str(e)
            is_rate_limit = ('429' in err_str or 'RateLimit' in err_str or
                             'rate limit' in err_str.lower() or
                             'Too Many Requests' in err_str)
            if is_rate_limit and attempt < max_retries:
                # Extract RPM limit from headers if present
                import re
                rpm_match = re.search(r'X-RateLimit-Limit.*?(\d+)', err_str)
                if rpm_match:
                    _rpm_state['detected_rpm_limit'] = int(rpm_match.group(1))
                else:
                    # Conservative fallback: assume 8 RPM (OpenRouter free tier)
                    _rpm_state['detected_rpm_limit'] = _rpm_state.get('detected_rpm_limit') or 8

                _rpm_state['last_429_at'] = time.time()
                delay = base_delay * (2 ** attempt)  # 8, 16, 32, 64
                logger.info(f"Rate limited (attempt {attempt+1}/{max_retries}), "
                            f"retrying in {delay:.0f}s...")
                logger.warning("      [rate limited, retry in %.0fs]", delay)
                time.sleep(delay)
            else:
                raise


# =============================================================================
# PROMPT ENSEMBLING - Multi-Perspective Synthesis
# =============================================================================

# Default perspectives for multi-perspective ensembling

# Status emitter for progress updates
status = SkillStatus("claude-cli-llm")

DEFAULT_PERSPECTIVES = [
    {
        "name": "analytical",
        "system": "You are an analytical expert. Focus on data, logic, evidence, and quantitative reasoning. Be precise and fact-based.",
    },
    {
        "name": "creative",
        "system": "You are a creative thinker. Consider unconventional angles, innovative solutions, and possibilities others might miss.",
    },
    {
        "name": "critical",
        "system": "You are a devil's advocate. Challenge assumptions, identify weaknesses, risks, and potential problems.",
    },
    {
        "name": "practical",
        "system": "You are a pragmatic implementer. Focus on feasibility, actionable steps, and real-world constraints.",
    },
]


def ensemble_prompt_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prompt Ensembling: Ask same question from multiple perspectives and synthesize.

    Based on latest research (2025-2026):
    - Multi-Perspective Simulation (virtual expert panel)
    - Generative Self-Aggregation (GSA)
    - Mixture-of-Prompts (MoP)

    Args:
        params: Dictionary containing:
            - prompt (str, required): The question/task to analyze
            - strategy (str): Ensembling strategy:
                - 'self_consistency': Same prompt, N samples, majority vote
                - 'multi_perspective': Different personas analyze question (default)
                - 'gsa': Generative Self-Aggregation
                - 'debate': Multi-round debate
            - num_samples (int): Number of samples for self_consistency (default: 5)
            - perspectives (list): Custom perspectives for multi_perspective
            - debate_rounds (int): Number of debate rounds (default: 2)
            - synthesis_style (str): How to synthesize - 'detailed', 'concise', 'structured'

    Returns:
        Dictionary with:
            - success (bool): Whether ensembling succeeded
            - response (str): Synthesized final response
            - perspectives_used (list): Perspectives/samples used
            - individual_responses (dict): Each perspective's response
            - confidence (float): Confidence in synthesized answer (0-1)
            - strategy (str): Strategy used
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        import dspy

        prompt = params.get('prompt')
        if not prompt:
            return {'success': False, 'error': 'prompt parameter is required'}

        strategy = params.get('strategy', 'multi_perspective')

        # Use DSPy's configured LM
        lm = dspy.settings.lm
        if not lm:
            return {'success': False, 'error': 'No LLM configured in DSPy'}

        # Execute based on strategy
        if strategy == 'self_consistency':
            return _self_consistency_ensemble(lm, prompt, params)
        elif strategy == 'multi_perspective':
            return _multi_perspective_ensemble(lm, prompt, params)
        elif strategy == 'gsa':
            return _gsa_ensemble(lm, prompt, params)
        elif strategy == 'debate':
            return _debate_ensemble(lm, prompt, params)
        else:
            return {'success': False, 'error': f'Unknown strategy: {strategy}'}

    except Exception as e:
        logger.error(f"Ensemble prompt error: {e}", exc_info=True)
        return {'success': False, 'error': f'Ensembling failed: {str(e)}'}


def _self_consistency_ensemble(lm, prompt: str, params: Dict) -> Dict[str, Any]:
    """
    Self-Consistency: Same prompt, multiple samples, majority vote or synthesis.

    Research shows this accounts for most gains in multi-agent approaches.
    """
    num_samples = params.get('num_samples', 5)

    responses = []
    for i in range(num_samples):
        try:
            # Add temperature variation for diversity (with retry on rate limits)
            response = _call_lm_with_retry(lm, prompt=prompt)
            text = response[0] if isinstance(response, list) else str(response)
            responses.append(text)
        except Exception as e:
            logger.warning(f"Sample {i+1} failed: {e}")

    if not responses:
        return {'success': False, 'error': 'All samples failed'}

    # Synthesize responses
    synthesis_prompt = f"""You have {len(responses)} different responses to the same question.
Analyze them and synthesize the best answer, combining insights from all responses.
Look for consensus points while also noting where responses diverge.

Question: {prompt}

Responses:
{chr(10).join(f'Response {i+1}: {r[:500]}...' if len(r) > 500 else f'Response {i+1}: {r}' for i, r in enumerate(responses))}

Synthesize the best answer, combining the strongest points from all responses:"""

    synthesis = _call_lm_with_retry(lm, prompt=synthesis_prompt)
    final_response = synthesis[0] if isinstance(synthesis, list) else str(synthesis)

    # Calculate agreement score as confidence
    confidence = _calculate_agreement_score(responses)

    return {
        'success': True,
        'response': final_response,
        'perspectives_used': [f'sample_{i+1}' for i in range(len(responses))],
        'individual_responses': {f'sample_{i+1}': r for i, r in enumerate(responses)},
        'confidence': confidence,
        'strategy': 'self_consistency',
        'num_samples': len(responses)
    }


def _multi_perspective_ensemble(lm, prompt: str, params: Dict) -> Dict[str, Any]:
    """
    Multi-Perspective: Different personas/lenses analyze same question.

    Based on Multi-Perspective Simulation - runs a virtual expert panel.
    Research shows this identifies critical considerations overlooked ~70% of time.

    Enhanced with:
    - Domain-aware perspective selection
    - Quality scoring for each perspective
    - Relevance validation
    """
    synthesis_style = params.get('synthesis_style', 'detailed')
    verbose = params.get('verbose', False)

    # Optima-inspired adaptive ensemble sizing (Chen et al., 2024):
    # Simple tasks get fewer perspectives (less token waste).
    # max_perspectives param lets callers control cost vs. quality tradeoff.
    max_perspectives = params.get('max_perspectives', 4)

    # Detect domain and select appropriate perspectives using LLM
    perspectives = params.get('perspectives')
    if not perspectives:
        perspectives = _select_domain_perspectives(prompt, lm=lm, max_perspectives=max_perspectives)
    else:
        perspectives = perspectives[:max_perspectives]

    individual_responses = {}
    quality_scores = {}

    total = len(perspectives)

    def _generate_one(idx, perspective):
        """Generate a single perspective with retry on rate limits."""
        if isinstance(perspective, dict):
            name = perspective.get('name', 'expert')
            system = perspective.get('system', '')
        else:
            name = str(perspective) if perspective else f'expert_{idx+1}'
            system = f"You are a {name}. Provide your expert analysis."

        perspective_prompt = f"""{system}

Analyze the following from your unique perspective. Be specific and provide concrete insights, not generic observations.

Question/Task: {prompt}

Requirements:
- Provide specific, actionable insights (not generic advice)
- Include concrete examples or data points where relevant
- Identify something that might be overlooked from other perspectives
- Be concise but substantive (200-400 words)

Your analysis:"""

        try:
            response = _call_lm_with_retry(lm, prompt=perspective_prompt)
            text = response[0] if isinstance(response, list) else str(response)
            score = _score_perspective_quality(text, prompt)
            logger.info("    [%d/%d] %s... done (quality: %.0f%%)", idx+1, total, name, score * 100)
            return name, text, score
        except Exception as e:
            logger.warning(f"Perspective '{name}' failed: {e}")
            logger.warning("    [%d/%d] %s... failed: %s", idx+1, total, name, e)
            return name, f"[Failed: {e}]", 0.0

    # Rate-limit aware execution strategy:
    # If we've detected a tight RPM limit (<=20), run SEQUENTIALLY with pacing.
    # Otherwise, run in parallel for speed.
    rpm_limit = _rpm_state.get('detected_rpm_limit')
    if rpm_limit and rpm_limit <= 20:
        logger.info("Ensemble: generating %d perspectives sequentially (RPM limit: %s)...", total, rpm_limit)
        for idx, p in enumerate(perspectives):
            name, text, score = _generate_one(idx, p)
            individual_responses[name] = text
            quality_scores[name] = score
    else:
        logger.info("Ensemble: generating %d perspectives in parallel...", total)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=min(total, 4)) as executor:
            futures = {
                executor.submit(_generate_one, idx, p): idx
                for idx, p in enumerate(perspectives)
            }
            for future in as_completed(futures):
                name, text, score = future.result()
                individual_responses[name] = text
                quality_scores[name] = score

    if not individual_responses:
        return {'success': False, 'error': 'All perspectives failed'}

    logger.info("Ensemble: synthesizing %d perspectives...", len(individual_responses))

    # Filter out low-quality perspectives (score < 0.3)
    valid_responses = {k: v for k, v in individual_responses.items()
                      if quality_scores.get(k, 0) >= 0.3 and '[Failed' not in v}

    if not valid_responses:
        valid_responses = individual_responses  # Fallback to all

    # Synthesize based on style
    if synthesis_style == 'concise':
        synthesis_instruction = "Provide a concise synthesis (2-3 paragraphs) of the key insights."
    elif synthesis_style == 'structured':
        synthesis_instruction = """Provide a structured synthesis with:
1. **Consensus Points**: Where perspectives agree
2. **Key Tensions**: Where perspectives diverge
3. **Unique Insights**: What each perspective uniquely contributed
4. **Synthesized Recommendation**: The balanced conclusion"""
    else:  # detailed
        synthesis_instruction = "Provide a comprehensive synthesis that integrates all perspectives into a balanced, nuanced answer."

    synthesis_prompt = f"""You have received analyses from {len(valid_responses)} expert perspectives.

Question: {prompt}

Expert Perspectives (with quality scores):
{chr(10).join(f'**{name.upper()}** (quality: {quality_scores.get(name, 0):.1%}):{chr(10)}{resp[:800]}' for name, resp in valid_responses.items())}

{synthesis_instruction}

Weight higher-quality perspectives more heavily in your synthesis.

Synthesized Answer:"""

    synthesis = _call_lm_with_retry(lm, prompt=synthesis_prompt)
    final_response = synthesis[0] if isinstance(synthesis, list) else str(synthesis)

    # Calculate weighted confidence based on quality scores
    avg_quality = sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0
    success_rate = len([r for r in individual_responses.values() if '[Failed' not in r]) / len(perspectives)
    confidence = (avg_quality + success_rate) / 2

    result = {
        'success': True,
        'response': final_response,
        'perspectives_used': list(valid_responses.keys()),
        'individual_responses': individual_responses,
        'quality_scores': quality_scores,
        'confidence': confidence,
        'strategy': 'multi_perspective',
        'synthesis_style': synthesis_style
    }

    # Add perspective summaries for verbose mode
    if verbose:
        result['perspective_summaries'] = {
            name: resp[:200] + '...' if len(resp) > 200 else resp
            for name, resp in individual_responses.items()
        }

    return result


def _select_domain_perspectives(prompt: str, lm=None, max_perspectives: int = 4) -> List[Dict[str, str]]:
    """
    Use LLM to generate domain-appropriate perspectives dynamically.

    Optima-inspired adaptive sizing (Chen et al., 2024):
    - For 2 perspectives: skip LLM call, use fast heuristic (saves ~10s)
    - For 3-4 perspectives: full LLM-driven perspective selection
    """
    import dspy

    # Use provided LM or get from settings
    if lm is None:
        lm = dspy.settings.lm if hasattr(dspy.settings, 'lm') else None

    if not lm:
        # Fallback to defaults if no LLM
        logger.info("Ensemble: using default perspectives (no LLM)")
        return DEFAULT_PERSPECTIVES[:max_perspectives]

    # Adaptive: For 2 perspectives, use fast heuristic (skip LLM call for perspective selection)
    # This saves ~10s by avoiding the "generate perspectives" LLM call.
    if max_perspectives <= 2:
        fast_perspectives = [
            {"name": "domain_expert", "system": f"You are a senior domain expert. Provide deep, practical analysis."},
            {"name": "critical_reviewer", "system": f"You are a critical reviewer. Identify risks, gaps, and overlooked considerations."},
        ]
        logger.info("Ensemble: fast mode (%d perspectives, no LLM selection)", max_perspectives)
        return fast_perspectives[:max_perspectives]

    try:
        logger.info("Ensemble: LLM generating domain-specific perspectives...")
        # Ask LLM to generate appropriate perspectives (with retry for rate limits)
        perspective_prompt = f"""Generate {max_perspectives} DOMAIN-EXPERT perspectives for this specific task.

Task: {prompt}

CRITICAL RULES:
1. Perspectives MUST be directly relevant to the task's domain
2. Use ACTUAL job titles/roles that exist in that industry
3. NO generic perspectives like "analytical", "creative", "practical", "ux_designer"
4. Each expert should have DEEP domain knowledge of the subject matter

DOMAIN DETECTION:
- If task mentions regulations/compliance/BaFin/SEC → use regulatory experts, compliance officers, legal counsel
- If task mentions finance/stocks/trading → use financial analysts, portfolio managers, risk officers
- If task mentions code/software/engineering → use senior engineers, architects, security experts
- If task mentions healthcare/medical → use doctors, clinical experts, regulatory affairs
- If task mentions marketing/business → use CMO, market researcher, growth strategist

Return as JSON array:
[
  {{"name": "role_name_with_underscores", "system": "You are a [specific job title with years of experience]. Focus on [domain-specific aspects]..."}},
  ...
]

EXAMPLE for "BaFin KGAB compliance checklist":
[
  {{"name": "bafin_compliance_officer", "system": "You are a senior BaFin compliance officer with 15 years experience. Focus on regulatory requirements, reporting obligations, and common compliance gaps."}},
  {{"name": "fund_administrator", "system": "You are a fund administration expert specializing in SPV oversight. Focus on operational controls, NAV calculations, and investor reporting."}},
  {{"name": "risk_management_director", "system": "You are a risk management director for a KGAB-regulated firm. Focus on risk identification, mitigation strategies, and control frameworks."}},
  {{"name": "external_auditor", "system": "You are an external auditor specializing in investment funds. Focus on audit trails, documentation requirements, and common deficiencies."}}
]

JSON array for the given task:"""

        response = _call_lm_with_retry(lm, prompt=perspective_prompt)
        text = response[0] if isinstance(response, list) else str(response)

        # Parse JSON from response
        import json
        import re

        # Extract JSON array from response
        json_match = re.search(r'\[[\s\S]*\]', text)
        if json_match:
            perspectives = json.loads(json_match.group())
            if isinstance(perspectives, list) and len(perspectives) >= 2:
                # Validate structure
                valid = all(
                    isinstance(p, dict) and 'name' in p and 'system' in p
                    for p in perspectives
                )
                if valid:
                    perspectives = perspectives[:max_perspectives]
                    names = [p['name'] for p in perspectives]
                    logger.info("Perspective generation done")
                    logger.info("Perspectives: %s", ', '.join(names))
                    logger.info(f"LLM generated {len(perspectives)} perspectives: {names}")
                    return perspectives

        logger.warning("Perspective generation failed (invalid JSON)")

    except Exception as e:
        logger.warning("Perspective generation failed (%s)", e)
        logger.warning(f"LLM perspective generation failed, using defaults: {e}")

    # Fallback to defaults
    logger.info("Ensemble: using default perspectives (fallback)")
    return DEFAULT_PERSPECTIVES


def _score_perspective_quality(response: str, original_prompt: str) -> float:
    """
    Score perspective quality based on heuristics.

    Returns 0.0-1.0 score based on:
    - Length (too short = low quality)
    - Specificity (contains specific terms from prompt)
    - Structure (has formatting/organization)
    - Not generic (avoids filler phrases)
    """
    if not response or '[Failed' in response:
        return 0.0

    score = 0.0

    # Length check (200-800 words is ideal)
    word_count = len(response.split())
    if word_count < 50:
        score += 0.1
    elif word_count < 150:
        score += 0.2
    elif word_count <= 600:
        score += 0.3
    else:
        score += 0.25  # Too long might be rambling

    # Specificity: does it reference terms from the prompt?
    prompt_words = set(original_prompt.lower().split())
    response_lower = response.lower()
    matches = sum(1 for w in prompt_words if len(w) > 4 and w in response_lower)
    specificity = min(matches / 5, 1.0) * 0.3
    score += specificity

    # Structure: has headers, bullets, or numbers
    structure_indicators = ['**', '##', '- ', '1.', '2.', '•']
    has_structure = any(ind in response for ind in structure_indicators)
    if has_structure:
        score += 0.2

    # Penalize generic filler phrases
    filler_phrases = [
        'it depends', 'there are many factors', 'in general',
        'it is important to note', 'as mentioned', 'in conclusion'
    ]
    filler_count = sum(1 for phrase in filler_phrases if phrase in response_lower)
    score -= filler_count * 0.05

    return max(0.0, min(1.0, score))


def _gsa_ensemble(lm, prompt: str, params: Dict) -> Dict[str, Any]:
    """
    Generative Self-Aggregation (GSA): Diverse generation + context-enriched synthesis.

    Unlike voting, this enables the model to synthesize an improved solution
    by learning from diverse attempts.
    """
    num_drafts = params.get('num_samples', 3)

    # Step 1: Generate diverse drafts with different approaches
    approaches = [
        "Approach this step-by-step with careful reasoning.",
        "Consider this from first principles. What are the fundamentals?",
        "Think about edge cases and potential issues first, then provide solution.",
    ]

    drafts = {}
    for i, approach in enumerate(approaches[:num_drafts]):
        draft_prompt = f"""{approach}

Question: {prompt}

Your response:"""

        try:
            response = _call_lm_with_retry(lm, prompt=draft_prompt)
            text = response[0] if isinstance(response, list) else str(response)
            drafts[f'draft_{i+1}'] = text
        except Exception as e:
            logger.warning(f"Draft {i+1} failed: {e}")

    if not drafts:
        return {'success': False, 'error': 'All drafts failed'}

    # Step 2: Context-enriched synthesis (the key GSA innovation)
    synthesis_prompt = f"""You have generated {len(drafts)} draft responses using different approaches.
Now synthesize the BEST possible answer by:
1. Identifying the strongest reasoning from each draft
2. Combining complementary insights
3. Resolving any contradictions with careful analysis
4. Producing a response that's BETTER than any individual draft

Question: {prompt}

Draft Responses:
{chr(10).join(f'**DRAFT {i+1}:**{chr(10)}{d}' for i, d in enumerate(drafts.values()))}

Your improved, synthesized answer (should be better than any individual draft):"""

    synthesis = _call_lm_with_retry(lm, prompt=synthesis_prompt)
    final_response = synthesis[0] if isinstance(synthesis, list) else str(synthesis)

    return {
        'success': True,
        'response': final_response,
        'perspectives_used': list(drafts.keys()),
        'individual_responses': drafts,
        'confidence': 0.85,  # GSA typically high confidence
        'strategy': 'gsa'
    }


def _debate_ensemble(lm, prompt: str, params: Dict) -> Dict[str, Any]:
    """
    Multi-Agent Debate: Perspectives debate over multiple rounds.

    Research shows debate improves factual validity and reduces hallucinations.
    """
    num_rounds = params.get('debate_rounds', 2)

    # Define debaters
    debaters = [
        {"name": "advocate", "stance": "Present the strongest case FOR"},
        {"name": "critic", "stance": "Present the strongest case AGAINST or identify problems with"},
        {"name": "synthesizer", "stance": "Find the balanced middle ground on"},
    ]

    debate_history = []
    current_responses = {}

    # Initial round
    for debater in debaters:
        initial_prompt = f"""{debater['stance']} the following:

{prompt}

Your position:"""

        try:
            response = _call_lm_with_retry(lm, prompt=initial_prompt)
            text = response[0] if isinstance(response, list) else str(response)
            current_responses[debater['name']] = text
            debate_history.append({'round': 0, 'debater': debater['name'], 'response': text})
        except Exception as e:
            logger.warning(f"Debater '{debater['name']}' failed: {e}")

    # Debate rounds
    for round_num in range(1, num_rounds + 1):
        new_responses = {}
        for debater in debaters:
            # Each debater responds to others' arguments
            other_args = {k: v for k, v in current_responses.items() if k != debater['name']}

            debate_prompt = f"""Previous positions in the debate:
{chr(10).join(f'{k.upper()}: {v[:400]}' for k, v in other_args.items())}

You are the {debater['name']}. Respond to the other positions and strengthen your argument.
Question: {prompt}

Your refined position (Round {round_num}):"""

            try:
                response = _call_lm_with_retry(lm, prompt=debate_prompt)
                text = response[0] if isinstance(response, list) else str(response)
                new_responses[debater['name']] = text
                debate_history.append({'round': round_num, 'debater': debater['name'], 'response': text})
            except Exception as e:
                logger.warning(f"Round {round_num}, debater '{debater['name']}' failed: {e}")
                new_responses[debater['name']] = current_responses.get(debater['name'], '')

        current_responses = new_responses

    # Final synthesis after debate
    synthesis_prompt = f"""A debate has concluded on the following question:

{prompt}

Final positions after {num_rounds} rounds of debate:
{chr(10).join(f'**{k.upper()}:** {v}' for k, v in current_responses.items())}

As an impartial judge, synthesize the debate into a final answer that:
1. Acknowledges valid points from all sides
2. Resolves contradictions with reasoning
3. Provides the most accurate and complete answer

Final judgment:"""

    synthesis = _call_lm_with_retry(lm, prompt=synthesis_prompt)
    final_response = synthesis[0] if isinstance(synthesis, list) else str(synthesis)

    return {
        'success': True,
        'response': final_response,
        'perspectives_used': [d['name'] for d in debaters],
        'individual_responses': current_responses,
        'debate_history': debate_history,
        'confidence': 0.9,  # Debate typically high confidence
        'strategy': 'debate',
        'num_rounds': num_rounds
    }


def _calculate_agreement_score(responses: List[str]) -> float:
    """Calculate agreement score between responses (0-1)."""
    if len(responses) < 2:
        return 1.0

    # Simple heuristic: check for common key phrases
    # More sophisticated: use embedding similarity
    try:
        # Extract key words from each response
        import re
        word_sets = []
        for r in responses:
            words = set(re.findall(r'\b\w{4,}\b', r.lower()))
            word_sets.append(words)

        # Calculate Jaccard similarity between all pairs
        similarities = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                intersection = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                if union > 0:
                    similarities.append(intersection / union)

        return sum(similarities) / len(similarities) if similarities else 0.5
    except Exception:
        return 0.5  # Default moderate confidence


def summarize_text_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize text using Claude.

    Args:
        params: Dictionary containing:
            - content (str, required): Text content to summarize
            - prompt (str, optional): Custom prompt (default: "Summarize the following:")
            - model (str, optional): Claude model - 'sonnet', 'opus', 'haiku' (default: 'sonnet')

    Returns:
        Dictionary with:
            - success (bool): Whether summarization succeeded
            - summary (str): Summarized text
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        import dspy

        content = params.get('content')
        if not content:
            return {'success': False, 'error': 'content parameter is required'}

        status.emit("Summarizing", "🧠 Summarizing text...")

        # Build prompt
        custom_prompt = params.get('prompt', 'Summarize the following content in a clear and concise way:')
        full_prompt = f"{custom_prompt}\n\n{content}"

        # Use DSPy's configured LM
        lm = dspy.settings.lm
        if not lm:
            return {'success': False, 'error': 'No LLM configured in DSPy'}

        # Direct LLM call (with rate-limit retry)
        response = _call_lm_with_retry(lm, prompt=full_prompt)

        # Extract text from response
        if isinstance(response, list):
            text = response[0] if response else ""
        else:
            text = str(response)

        return {
            'success': True,
            'summary': text,
            'model': getattr(lm, 'model', 'unknown'),
            'provider': 'dspy'
        }

    except Exception as e:
        logger.error(f"Summarize text error: {e}", exc_info=True)
        return {'success': False, 'error': f'Summarization failed: {str(e)}'}


def generate_text_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate text using Claude.

    Args:
        params: Dictionary containing:
            - prompt (str, required): Text generation prompt
            - model (str, optional): Claude model (default: 'sonnet')
            - max_tokens (int, optional): Maximum tokens (default: 4096)

    Returns:
        Dictionary with:
            - success (bool): Whether generation succeeded
            - text (str): Generated text
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        import dspy

        prompt = params.get('prompt')
        if not prompt:
            return {'success': False, 'error': 'prompt parameter is required'}

        status.emit("Generating", "🧠 Generating text...")

        # Use DSPy's configured LM
        lm = dspy.settings.lm
        if not lm:
            return {'success': False, 'error': 'No LLM configured in DSPy'}

        # Direct LLM call (with rate-limit retry)
        response = _call_lm_with_retry(lm, prompt=prompt)

        # Extract text from response
        if isinstance(response, list):
            text = response[0] if response else ""
        else:
            text = str(response)

        return {
            'success': True,
            'text': text,
            'model': getattr(lm, 'model', 'unknown'),
            'provider': 'dspy'
        }

    except Exception as e:
        logger.error(f"Generate text error: {e}", exc_info=True)
        return {'success': False, 'error': f'Text generation failed: {str(e)}'}


def claude_cli_llm_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unified LLM tool for text generation with optional ensembling.

    Args:
        params: Dictionary containing:
            - prompt (str, required): The prompt/question
            - max_tokens (int, optional): Max tokens for response
            - ensemble (bool, optional): Enable prompt ensembling (default: False)
            - ensemble_strategy (str, optional): Strategy if ensemble=True
                - 'self_consistency', 'multi_perspective', 'gsa', 'debate'

    Returns:
        Dictionary with:
            - success (bool): Whether generation succeeded
            - response (str): Generated/synthesized response
            - ensemble_used (bool): Whether ensembling was used
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        import dspy

        prompt = params.get('prompt')
        if not prompt:
            return {'success': False, 'error': 'prompt parameter is required'}

        # Check if ensembling requested
        ensemble = params.get('ensemble', False)
        if ensemble:
            strategy = params.get('ensemble_strategy', 'multi_perspective')
            return ensemble_prompt_tool({
                'prompt': prompt,
                'strategy': strategy,
                'num_samples': params.get('num_samples', 5),
                'synthesis_style': params.get('synthesis_style', 'detailed')
            })

        # Standard single LLM call (with rate-limit retry)
        lm = dspy.settings.lm
        if not lm:
            return {'success': False, 'error': 'No LLM configured in DSPy'}

        response = _call_lm_with_retry(lm, prompt=prompt)
        text = response[0] if isinstance(response, list) else str(response)

        return {
            'success': True,
            'response': text,
            'ensemble_used': False,
            'model': getattr(lm, 'model', 'unknown')
        }

    except Exception as e:
        logger.error(f"Claude CLI LLM error: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


__all__ = [
    'summarize_text_tool',
    'generate_text_tool',
    'ensemble_prompt_tool',
    'claude_cli_llm_tool'
]
