"""
Review Mixin - Collaborative design, team planning, and review pipeline.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional

import dspy

from .types import ResearchContext
from .signatures import (
    CollaborativeArchitectSignature,
    ResearchResponseSignature,
    TeamPlanningSignature,
    TeamPlanningConsolidationSignature,
    TeamReviewSignature,
)
from .teams import TeamPersona
from .agents import ArbitratorAgent
from .utils import (
    _extract_components_from_text,
    _progress,
    _stream_call,
    _strip_code_fences,
)

logger = logging.getLogger(__name__)


class ReviewMixin:
    def _build_review_criteria(self) -> str:
        """Collect guiding principles from all team reviewers.

        Shifts review expectations left: the developer sees what reviewers
        will check BEFORE writing code, reducing first-round rejections.
        """
        if not self._team_config:
            return ""
        seen = set()
        criteria_lines = []
        for phase in ("functional", "quality"):
            for persona in self._team_config.get_reviewers(phase):
                if persona.name in seen:
                    continue
                seen.add(persona.name)
                for principle in persona.guiding_principles:
                    criteria_lines.append(f"- [{persona.name}] {principle}")
        return "\n".join(criteria_lines)

    # -----------------------------------------------------------------
    # COLLABORATIVE DESIGN LOOP (Architect + Researcher)
    # -----------------------------------------------------------------

    async def _collaborative_design_loop(
        self,
        requirements: str,
        language: str,
        style: str,
        constraints: str,
        max_iterations: int = 3,
    ) -> tuple:
        """Run collaborative design loop between Architect and Researcher.

        The Architect drafts/refines architecture while the Researcher searches
        for best practices, library docs, and pitfalls to inform each iteration.

        Args:
            requirements: Software requirements
            language: Target programming language
            style: Coding style preference
            constraints: Technical constraints
            max_iterations: Number of collaborative iterations (default: 3)

        Returns:
            (arch_result_dict, research_context)
        """
        _progress("Phase 1", "CollaborativeDesign", f"Starting {max_iterations}-iteration design loop...")

        # Initialize modules
        if not hasattr(self, '_collab_architect_module') or self._collab_architect_module is None:
            self._collab_architect_module = dspy.ChainOfThought(CollaborativeArchitectSignature)
        if not hasattr(self, '_research_response_module') or self._research_response_module is None:
            self._research_response_module = dspy.ChainOfThought(ResearchResponseSignature)

        # Get web search function
        web_search_fn = None
        try:
            from Jotty.core.capabilities.skills import get_skill_registry
            registry = get_skill_registry()
            web_search_fn = registry.get('web-search', {}).get('search_web_tool')
        except Exception:
            pass

        if web_search_fn is None:
            try:
                from Jotty.core.capabilities.skills import search_web_tool as web_search_fn
            except Exception:
                pass

        if web_search_fn is None:
            try:
                import importlib
                mod = importlib.import_module('skills.web-search.tools')
                web_search_fn = getattr(mod, 'search_web_tool', None)
            except Exception:
                pass

        # State for iterative refinement
        current_architecture = ""
        current_components = []
        current_file_structure = ""
        current_interfaces = ""
        accumulated_research = ResearchContext()
        research_findings_str = ""

        config = self.config
        frameworks_json = json.dumps(getattr(config, 'frameworks', []))

        for iteration in range(1, max_iterations + 1):
            _progress("Phase 1", "CollaborativeDesign", f"=== Iteration {iteration}/{max_iterations} ===")

            # --- Architect Phase ---
            _progress("Phase 1", "Architect", f"Designing architecture (iteration {iteration})...")
            try:
                arch_result = await _stream_call(
                    self._collab_architect_module, "Phase 1", "Architect",
                    requirements=requirements,
                    language=language,
                    style=style,
                    constraints=constraints,
                    iteration=iteration,
                    previous_architecture=current_architecture,
                    research_findings=research_findings_str,
                )

                current_architecture = str(arch_result.architecture)
                current_file_structure = str(arch_result.file_structure)
                current_interfaces = str(arch_result.interfaces)
                research_requests = str(arch_result.research_requests)

                # Parse components - try JSON first, fallback to text extraction
                try:
                    current_components = json.loads(str(arch_result.components))
                    if not isinstance(current_components, list):
                        current_components = []
                except (json.JSONDecodeError, TypeError):
                    current_components = []

                # Fallback: extract components from file structure/architecture text
                if not current_components:
                    current_components = _extract_components_from_text(
                        current_file_structure,
                        current_architecture,
                        current_interfaces
                    )

                n_components = len(current_components)
                _progress("Phase 1", "Architect", f"Designed {n_components} component(s)")

                # Show file structure preview
                if current_file_structure:
                    for line in current_file_structure.strip().split('\n')[:5]:
                        _progress("Phase 1", "Architect", f"  {line.strip()}")

                # Show research requests
                if research_requests:
                    _progress("Phase 1", "Architect", f"Requesting research on: {research_requests[:100]}...")

            except Exception as e:
                logger.error(f"Architect iteration {iteration} failed: {e}")
                _progress("Phase 1", "Architect", f"Error: {e}")
                break

            # --- Researcher Phase (skip on last iteration) ---
            if iteration < max_iterations and getattr(config, 'enable_research', True):
                _progress("Phase 1.5", "Researcher", f"Researching Architect's requests (iteration {iteration})...")
                try:
                    # Generate targeted queries based on architect's requests
                    research_result = await _stream_call(
                        self._research_response_module, "Phase 1.5", "Researcher",
                        requirements=requirements,
                        architecture=current_architecture,
                        research_requests=research_requests,
                        language=language,
                        frameworks=frameworks_json,
                    )

                    # Parse search queries
                    try:
                        search_queries = json.loads(str(research_result.search_queries))
                        if not isinstance(search_queries, list):
                            search_queries = []
                    except (json.JSONDecodeError, TypeError):
                        search_queries = []

                    # Execute web searches
                    if web_search_fn and search_queries:
                        for qi, query in enumerate(search_queries[:5]):
                            _progress("Phase 1.5", "Researcher", f"Searching ({qi+1}/{min(len(search_queries),5)}): {str(query)[:50]}")
                            try:
                                result = web_search_fn({'query': str(query), 'max_results': 3})
                                if result.get('success') and result.get('results'):
                                    for r in result['results']:
                                        snippet = r.get('snippet', r.get('description', ''))
                                        if snippet:
                                            title = r.get('title', '')
                                            # Categorize findings
                                            query_lower = str(query).lower()
                                            if any(kw in query_lower for kw in ['best practice', 'pattern', 'convention', 'standard']):
                                                accumulated_research.best_practices.append(f"{title}: {snippet}")
                                            elif any(kw in query_lower for kw in ['doc', 'api', 'reference', 'example']):
                                                accumulated_research.api_references.append(f"{title}: {snippet}")
                                            elif any(kw in query_lower for kw in ['pitfall', 'warning', 'avoid', 'mistake', 'security']):
                                                accumulated_research.warnings.append(f"{title}: {snippet}")
                                            else:
                                                accumulated_research.library_docs.append(f"{title}: {snippet}")
                            except Exception as search_err:
                                logger.debug(f"Web search failed: {search_err}")

                    # Accumulate research analysis
                    analysis = str(research_result.analysis)
                    best_practices = str(research_result.best_practices)
                    warnings = str(research_result.warnings)
                    recommendations = str(research_result.recommendations)

                    # Build research findings string for next iteration
                    research_parts = []
                    if analysis:
                        research_parts.append(f"ANALYSIS:\n{analysis}")
                    if best_practices:
                        research_parts.append(f"BEST PRACTICES:\n{best_practices}")
                        for bp in best_practices.split('\n')[:3]:
                            if bp.strip():
                                accumulated_research.best_practices.append(bp.strip())
                    if warnings:
                        research_parts.append(f"WARNINGS:\n{warnings}")
                        for w in warnings.split('\n')[:3]:
                            if w.strip():
                                accumulated_research.warnings.append(w.strip())
                    if recommendations:
                        research_parts.append(f"RECOMMENDATIONS:\n{recommendations}")

                    research_findings_str = "\n\n".join(research_parts)

                    total_findings = (len(accumulated_research.best_practices) +
                                    len(accumulated_research.library_docs) +
                                    len(accumulated_research.api_references) +
                                    len(accumulated_research.warnings))
                    _progress("Phase 1.5", "Researcher", f"Done -- {total_findings} total finding(s) accumulated")

                except Exception as e:
                    logger.error(f"Researcher iteration {iteration} failed (non-blocking): {e}")
                    _progress("Phase 1.5", "Researcher", f"Research skipped: {str(e)[:50]}")

        # Build final result
        arch_result_dict = {
            'architecture': current_architecture,
            'components': current_components,
            'file_structure': current_file_structure,
            'interfaces': current_interfaces,
        }

        total_research = (len(accumulated_research.best_practices) +
                         len(accumulated_research.library_docs) +
                         len(accumulated_research.api_references) +
                         len(accumulated_research.warnings))
        _progress("Phase 1", "CollaborativeDesign", f"Done -- {len(current_components)} component(s), {total_research} research finding(s)")

        return arch_result_dict, accumulated_research

    # -----------------------------------------------------------------
    # TEAM PLANNING METHODS
    # -----------------------------------------------------------------

    async def _run_persona_planning(
        self, requirements: str, architecture: str, research_findings: str, persona: TeamPersona
    ) -> Dict[str, Any]:
        """Run a single persona's planning input. Non-blocking on failure."""
        try:
            _progress("Phase 2", persona.name, "Providing planning input...")
            if not hasattr(self, '_planning_module') or self._planning_module is None:
                self._planning_module = dspy.ChainOfThought(TeamPlanningSignature)
            result = await _stream_call(self._planning_module, "Phase 2", persona.name,
                requirements=requirements,
                architecture=architecture,
                research_findings=research_findings,
                persona_context=persona.to_prompt(),
            )
            try:
                concerns = json.loads(str(result.concerns))
                if not isinstance(concerns, list):
                    concerns = []
            except (json.JSONDecodeError, TypeError):
                concerns = []
            _progress("Phase 2", persona.name, f"Done -- {len(concerns)} concern(s)")
            return {
                "persona": persona.name,
                "concerns": concerns,
                "recommendations": str(result.recommendations),
                "implementation_notes": str(result.implementation_notes),
            }
        except Exception as e:
            logger.error(f"Persona planning failed for {persona.name} (non-blocking): {e}")
            _progress("Phase 2", persona.name, "Failed (skipped)")
            return {"persona": persona.name, "concerns": [], "recommendations": "", "implementation_notes": ""}

    async def _team_planning(
        self, requirements: str, architecture: str, research_findings: str
    ) -> Dict[str, Any]:
        """Orchestrate Phase 2: team planning discussion to refine architecture.

        All team personas provide input on the architecture based on their expertise
        and research findings. Their feedback is consolidated into a refined plan.

        Returns:
            Dict with refined_architecture, implementation_plan, and team_feedback
        """
        planning_result = {
            "refined_architecture": architecture,  # Default to original if planning fails
            "implementation_plan": "",
            "risk_mitigations": "",
            "team_agreements": "",
            "team_feedback": [],
        }

        if not self._team_config:
            return planning_result

        # Gather input from all personas (both functional and quality reviewers)
        all_personas = []
        seen = set()
        for phase in ("functional", "quality"):
            for persona in self._team_config.get_reviewers(phase):
                if persona.name not in seen:
                    seen.add(persona.name)
                    all_personas.append(persona)

        if not all_personas:
            return planning_result

        persona_names = ", ".join(p.name for p in all_personas)
        _progress("Phase 2", "TeamPlanning", f"Gathering input from: {persona_names}")

        # Run all persona planning in parallel
        tasks = [
            self._run_persona_planning(requirements, architecture, research_findings, persona)
            for persona in all_personas
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect feedback
        feedback_parts = []
        all_concerns = []
        for r in results:
            if isinstance(r, Exception):
                continue
            planning_result["team_feedback"].append(r)
            if r.get("recommendations"):
                feedback_parts.append(f"[{r['persona']}] RECOMMENDATIONS:\n{r['recommendations']}")
            if r.get("implementation_notes"):
                feedback_parts.append(f"[{r['persona']}] IMPLEMENTATION NOTES:\n{r['implementation_notes']}")
            for concern in r.get("concerns", []):
                if isinstance(concern, dict):
                    all_concerns.append(f"[{r['persona']}] [{concern.get('severity', '?')}] {concern.get('description', '')}")

        if all_concerns:
            _progress("Phase 2", "TeamPlanning", f"Team raised {len(all_concerns)} concern(s)")
            for concern in all_concerns[:5]:
                _progress("Phase 2", "TeamPlanning", f"  {concern[:80]}")

        # Consolidate feedback into refined architecture
        if feedback_parts:
            _progress("Phase 2", "TeamPlanning", "Consolidating team feedback...")
            try:
                if not hasattr(self, '_consolidation_module') or self._consolidation_module is None:
                    self._consolidation_module = dspy.ChainOfThought(TeamPlanningConsolidationSignature)

                team_feedback_str = "\n\n".join(feedback_parts)
                if all_concerns:
                    team_feedback_str += "\n\nCONCERNS:\n" + "\n".join(all_concerns)

                consolidation = await _stream_call(
                    self._consolidation_module, "Phase 2", "TeamPlanning",
                    original_architecture=architecture,
                    team_feedback=team_feedback_str,
                    research_findings=research_findings,
                    requirements=requirements,
                )

                planning_result["refined_architecture"] = str(consolidation.refined_architecture)
                planning_result["implementation_plan"] = str(consolidation.implementation_plan)
                planning_result["risk_mitigations"] = str(consolidation.risk_mitigations)
                planning_result["team_agreements"] = str(consolidation.team_agreements)

                _progress("Phase 2", "TeamPlanning", "Done -- architecture refined with team input")

            except Exception as e:
                logger.error(f"Planning consolidation failed (non-blocking): {e}")
                _progress("Phase 2", "TeamPlanning", "Consolidation failed, using original architecture")
        else:
            _progress("Phase 2", "TeamPlanning", "No feedback to consolidate, using original architecture")

        return planning_result

    # -----------------------------------------------------------------
    # TEAM REVIEW METHODS
    # -----------------------------------------------------------------

    async def _run_persona_review(
        self, code: str, requirements: str, phase: str, persona: TeamPersona,
        team_agreements: str = ""
    ) -> Dict[str, Any]:
        """Run a single persona review. Non-blocking on failure."""
        try:
            _progress("Phase 6", persona.name, f"Reviewing ({phase})...")
            if self._review_module is None:
                self._review_module = dspy.ChainOfThought(TeamReviewSignature)

            # Build team agreements context
            agreements_context = team_agreements or "No specific team agreements recorded."

            result = await _stream_call(self._review_module, "Phase 6", persona.name,
                code=code,
                requirements=requirements,
                review_phase=phase,
                persona_context=persona.to_prompt(),
                team_agreements=agreements_context,
            )
            verdict = str(result.verdict).strip().upper()
            try:
                issues = json.loads(str(result.issues))
                if not isinstance(issues, list):
                    issues = []
            except (json.JSONDecodeError, TypeError):
                issues = []
            _progress("Phase 6", persona.name, f"Done -- {verdict}")
            return {
                "persona": persona.name,
                "verdict": verdict,
                "issues": issues,
                "feedback": str(result.feedback),
                "evidence": str(getattr(result, 'evidence', '')),
            }
        except Exception as e:
            logger.error(f"Persona review failed for {persona.name} (non-blocking): {e}")
            _progress("Phase 6", persona.name, "Failed (auto-approved)")
            return {"persona": persona.name, "verdict": "APPROVED", "issues": [], "feedback": "", "evidence": ""}

    async def _run_review_phase(
        self, reviewers: List[TeamPersona], code: str, requirements: str, phase: str,
        team_agreements: str = ""
    ) -> List[Dict[str, Any]]:
        """Run all reviewers for a phase in parallel."""
        reviewer_names = ", ".join(p.name for p in reviewers)
        _progress("Phase 6", "TeamReview", f"{phase.capitalize()} review: {reviewer_names}")
        tasks = [
            self._run_persona_review(code, requirements, phase, persona, team_agreements)
            for persona in reviewers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        parsed = []
        for r in results:
            if isinstance(r, Exception):
                parsed.append({"persona": "unknown", "verdict": "APPROVED", "issues": [], "feedback": ""})
            else:
                parsed.append(r)
                verdict = r.get('verdict', '?')
                feedback_preview = r.get('feedback', '')[:100]
                n_issues = len(r.get('issues', []))
                detail = f" ({n_issues} issue(s))" if n_issues else ""
                _progress("Phase 6", r.get("persona", "Reviewer"), f"{verdict}{detail}")
                if feedback_preview and verdict == "REJECTED":
                    _progress("Phase 6", r.get("persona", "Reviewer"), f"  {feedback_preview}")
        return parsed

    def _collect_feedback(self, results: List[Dict[str, Any]]) -> str:
        """Aggregate feedback strings from review results."""
        parts = []
        for r in results:
            if r.get("feedback"):
                parts.append(f"[{r.get('persona', 'Reviewer')}] {r['feedback']}")
        return "\n".join(parts)

    async def _team_review(
        self, all_code: str, requirements: str, architecture: str,
        files: Dict[str, str], main_file: Optional[str],
        planning_result: Dict[str, Any] = None,
        simplicity_result: Dict[str, Any] = None
    ) -> tuple:
        """Orchestrate Phase 6: two-phase team review with auto-fix loop.

        Args:
            all_code: Combined code string from all generated files
            requirements: Original requirements
            architecture: Architecture design
            files: Dict of filename -> code content
            main_file: Name of the main/entry file
            planning_result: Result from team planning phase, contains team_agreements
            simplicity_result: Result from SimplicityJudge (Phase 5.5), to avoid contradictions

        Returns:
            (review_result_dict, possibly_updated_files)
        """
        review_result = {
            "approved": True,
            "team": self._team_config.name if self._team_config else "unknown",
            "phase_a_results": [],
            "phase_b_results": [],
            "feedback": "",
            "rework_attempts": 0,
        }

        if not self._team_config:
            return review_result, files

        current_code = all_code

        # Extract team agreements from planning phase to avoid contradictory rejections
        team_agreements = ""
        if planning_result and planning_result.get('team_agreements'):
            team_agreements = str(planning_result['team_agreements'])

        # Include SimplicityJudge verdict to prevent contradictory "too simple" rejections
        if simplicity_result:
            verdict = simplicity_result.get('verdict', 'ACCEPT')
            score = simplicity_result.get('simplicity_score', 1.0)
            if verdict == 'SIMPLIFY' or score < 0.5:
                team_agreements += f"\n\n**SIMPLICITY OVERRIDE (Phase 5.5)**: Code was intentionally simplified by SimplicityJudge (score: {score:.2f}). Do NOT reject for being 'too simple' or 'missing architecture components'. The simplified code is the correct implementation."
            else:
                team_agreements += f"\n\n**Simplicity Check**: Code passed SimplicityJudge (score: {score:.2f}, verdict: {verdict})."

        # --- Phase 6a: Functional Review ---
        func_reviewers = self._team_config.get_reviewers("functional")
        if func_reviewers:
            phase_a = await self._run_review_phase(func_reviewers, current_code, requirements, "functional", team_agreements)
            review_result["phase_a_results"] = phase_a

            rejected = [r for r in phase_a if r.get("verdict") == "REJECTED"]

            # Arbitrator: validate rejections before rework
            if rejected and getattr(self.config, 'enable_arbitrator', True):
                if not hasattr(self, '_arbitrator') or self._arbitrator is None:
                    self._arbitrator = ArbitratorAgent(
                        self._memory, self._context, self._bus, self._agent_context("Verifier"))
                validated = []
                for r in rejected:
                    arb_result = await self._arbitrator.evaluate(
                        current_code, r.get('feedback', ''), r.get('evidence', ''), r['persona']
                    )
                    if arb_result.get('valid', True):
                        validated.append(r)
                        _progress("Phase 6", "Arbitrator", f"CONFIRMED: {r['persona']}")
                    else:
                        r['verdict'] = 'APPROVED'
                        r['arbitrator_overruled'] = True
                        _progress("Phase 6", "Arbitrator", f"OVERRULED: {r['persona']}")
                rejected = validated

            if rejected:
                feedback_text = self._collect_feedback(rejected)
                _progress("Phase 6", "Optimizer", f"Reworking code ({len(rejected)} rejection(s) in functional review)...")
                review_result["rework_attempts"] += 1

                opt_result = await self._optimizer.optimize(
                    code=current_code,
                    focus="review_feedback",
                    requirements=requirements,
                    constraints=feedback_text,
                )
                if "optimized_code" in opt_result:
                    current_code = _strip_code_fences(opt_result["optimized_code"])
                    if main_file and main_file in files:
                        files[main_file] = current_code
                    _progress("Phase 6", "Optimizer", "Rework applied, re-reviewing...")

                # Re-review once
                phase_a = await self._run_review_phase(func_reviewers, current_code, requirements, "functional", team_agreements)
                review_result["phase_a_results"] = phase_a

        # --- Phase 6b: Code Quality Review ---
        quality_reviewers = self._team_config.get_reviewers("quality")
        if quality_reviewers:
            phase_b = await self._run_review_phase(quality_reviewers, current_code, requirements, "quality", team_agreements)
            review_result["phase_b_results"] = phase_b

            rejected = [r for r in phase_b if r.get("verdict") == "REJECTED"]

            # Arbitrator: validate rejections before rework
            if rejected and getattr(self.config, 'enable_arbitrator', True):
                if not hasattr(self, '_arbitrator') or self._arbitrator is None:
                    self._arbitrator = ArbitratorAgent(
                        self._memory, self._context, self._bus, self._agent_context("Verifier"))
                validated = []
                for r in rejected:
                    arb_result = await self._arbitrator.evaluate(
                        current_code, r.get('feedback', ''), r.get('evidence', ''), r['persona']
                    )
                    if arb_result.get('valid', True):
                        validated.append(r)
                        _progress("Phase 6", "Arbitrator", f"CONFIRMED: {r['persona']}")
                    else:
                        r['verdict'] = 'APPROVED'
                        r['arbitrator_overruled'] = True
                        _progress("Phase 6", "Arbitrator", f"OVERRULED: {r['persona']}")
                rejected = validated

            if rejected:
                feedback_text = self._collect_feedback(rejected)
                _progress("Phase 6", "Optimizer", f"Reworking code ({len(rejected)} rejection(s) in quality review)...")
                review_result["rework_attempts"] += 1

                opt_result = await self._optimizer.optimize(
                    code=current_code,
                    focus="review_feedback",
                    requirements=requirements,
                    constraints=feedback_text,
                )
                if "optimized_code" in opt_result:
                    current_code = _strip_code_fences(opt_result["optimized_code"])
                    if main_file and main_file in files:
                        files[main_file] = current_code
                    _progress("Phase 6", "Optimizer", "Rework applied, re-reviewing...")

                # Re-review once
                phase_b = await self._run_review_phase(quality_reviewers, current_code, requirements, "quality", team_agreements)
                review_result["phase_b_results"] = phase_b

        # Final verdict
        all_results = review_result["phase_a_results"] + review_result["phase_b_results"]
        all_approved = all(r.get("verdict") == "APPROVED" for r in all_results)
        review_result["approved"] = all_approved
        review_result["feedback"] = self._collect_feedback(
            [r for r in all_results if r.get("verdict") == "REJECTED"]
        )

        verdict_str = "APPROVED" if all_approved else "REJECTED"
        _progress("Phase 6", "TeamReview", f"Final verdict: {verdict_str} (rework attempts: {review_result['rework_attempts']})")

        return review_result, files

