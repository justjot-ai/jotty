"""
Review Swarm - World-Class Code & Documentation Review
======================================================

Production-grade swarm for:
- Code review with best practices
- Security vulnerability detection
- Performance analysis
- Documentation quality check
- API contract validation
- Architecture assessment

Agents:
┌─────────────────────────────────────────────────────────────────────────┐
│                          REVIEW SWARM                                    │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │
│  │   Code         │  │   Security     │  │   Performance  │            │
│  │   Reviewer     │  │   Scanner      │  │   Analyzer     │            │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘            │
│          │                   │                   │                      │
│          └───────────────────┼───────────────────┘                      │
│                              ▼                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │
│  │   Architecture │  │   Style        │  │   Doc          │            │
│  │   Reviewer     │  │   Checker      │  │   Reviewer     │            │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘            │
│          │                   │                   │                      │
│          └───────────────────┼───────────────────┘                      │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     REVIEW SYNTHESIZER                           │   │
│  │   Combines all reviews into actionable report with priorities    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘

Usage:
    from core.swarms.review_swarm import ReviewSwarm, review_code

    # Full swarm
    swarm = ReviewSwarm()
    result = await swarm.review(code, language="python")

    # One-liner
    result = await review_code(code, context="Pull request for user auth")

Author: Jotty Team
Date: February 2026
"""

import asyncio
import logging
import json
import dspy
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

from .base_swarm import (
    BaseSwarm, SwarmBaseConfig, SwarmResult, AgentRole,
    register_swarm, ExecutionTrace
)
from .base import DomainSwarm, AgentTeam, _split_field
from .swarm_signatures import ReviewSwarmSignature
from ..agents.base import DomainAgent, DomainAgentConfig, BaseSwarmAgent

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class ReviewType(Enum):
    CODE = "code"
    SECURITY = "security"
    PERFORMANCE = "performance"
    ARCHITECTURE = "architecture"
    STYLE = "style"
    DOCUMENTATION = "documentation"


class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ReviewStatus(Enum):
    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    NEEDS_DISCUSSION = "needs_discussion"
    BLOCKED = "blocked"


@dataclass
class ReviewConfig(SwarmBaseConfig):
    """Configuration for ReviewSwarm."""
    review_types: List[ReviewType] = field(default_factory=lambda: [
        ReviewType.CODE, ReviewType.SECURITY, ReviewType.PERFORMANCE
    ])
    include_suggestions: bool = True
    include_examples: bool = True
    strictness: str = "moderate"  # lenient, moderate, strict
    language: str = "python"
    style_guide: str = "pep8"

    def __post_init__(self):
        self.name = "ReviewSwarm"
        self.domain = "code_review"


@dataclass
class ReviewComment:
    """A review comment."""
    line: int
    severity: Severity
    category: str
    message: str
    suggestion: str = ""
    example: str = ""


@dataclass
class SecurityFinding:
    """A security vulnerability finding."""
    vulnerability_type: str
    severity: Severity
    location: str
    description: str
    cwe: str = ""
    fix_recommendation: str = ""


@dataclass
class PerformanceFinding:
    """A performance issue."""
    issue_type: str
    severity: Severity
    location: str
    description: str
    impact: str
    optimization: str = ""


@dataclass
class ArchitectureFinding:
    """An architecture concern."""
    concern_type: str
    severity: Severity
    description: str
    affected_components: List[str]
    recommendation: str


@dataclass
class ReviewResult(SwarmResult):
    """Result from ReviewSwarm."""
    status: ReviewStatus = ReviewStatus.NEEDS_DISCUSSION
    comments: List[ReviewComment] = field(default_factory=list)
    security_findings: List[SecurityFinding] = field(default_factory=list)
    performance_findings: List[PerformanceFinding] = field(default_factory=list)
    architecture_findings: List[ArchitectureFinding] = field(default_factory=list)
    summary: str = ""
    score: float = 0.0  # 0-100
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    approved: bool = False


# =============================================================================
# DSPy SIGNATURES
# =============================================================================

class CodeReviewSignature(dspy.Signature):
    """Review code for quality.

    You are a SENIOR CODE REVIEWER. Review for:
    1. Logic errors and bugs
    2. Code clarity and readability
    3. Error handling
    4. Edge cases
    5. Best practices

    Be constructive and specific.
    """
    code: str = dspy.InputField(desc="Code to review")
    language: str = dspy.InputField(desc="Programming language")
    context: str = dspy.InputField(desc="Context about the code/PR")
    strictness: str = dspy.InputField(desc="Review strictness level")

    issues: str = dspy.OutputField(desc="JSON list of issues with line, severity, message")
    positives: str = dspy.OutputField(desc="Positive aspects of the code, separated by |")
    suggestions: str = dspy.OutputField(desc="Improvement suggestions, separated by |")
    overall_assessment: str = dspy.OutputField(desc="Overall code quality assessment")


class SecurityReviewSignature(dspy.Signature):
    """Scan code for security vulnerabilities.

    You are a SECURITY EXPERT. Scan for:
    1. Injection vulnerabilities (SQL, XSS, Command)
    2. Authentication/Authorization flaws
    3. Sensitive data exposure
    4. Insecure configurations
    5. OWASP Top 10 issues

    Report all potential security risks.
    """
    code: str = dspy.InputField(desc="Code to scan")
    language: str = dspy.InputField(desc="Programming language")
    context: str = dspy.InputField(desc="Application context")

    vulnerabilities: str = dspy.OutputField(desc="JSON list of vulnerabilities with type, severity, location, description")
    security_score: float = dspy.OutputField(desc="Security score 0-100")
    recommendations: str = dspy.OutputField(desc="Security recommendations, separated by |")


class PerformanceReviewSignature(dspy.Signature):
    """Analyze code for performance issues.

    You are a PERFORMANCE ENGINEER. Identify:
    1. Algorithmic inefficiencies
    2. Memory issues
    3. I/O bottlenecks
    4. Resource leaks
    5. Scalability concerns

    Focus on impactful improvements.
    """
    code: str = dspy.InputField(desc="Code to analyze")
    language: str = dspy.InputField(desc="Programming language")
    scale: str = dspy.InputField(desc="Expected scale of usage")

    issues: str = dspy.OutputField(desc="JSON list of performance issues with type, severity, location, impact")
    complexity_analysis: str = dspy.OutputField(desc="Time/space complexity analysis")
    optimizations: str = dspy.OutputField(desc="Optimization suggestions, separated by |")


class ArchitectureReviewSignature(dspy.Signature):
    """Review code architecture.

    You are a SOFTWARE ARCHITECT. Evaluate:
    1. Design patterns usage
    2. SOLID principles adherence
    3. Separation of concerns
    4. Coupling and cohesion
    5. Testability

    Provide architectural guidance.
    """
    code: str = dspy.InputField(desc="Code to review")
    context: str = dspy.InputField(desc="System context")
    patterns: str = dspy.InputField(desc="Expected patterns/architecture")

    concerns: str = dspy.OutputField(desc="JSON list of architectural concerns")
    patterns_found: str = dspy.OutputField(desc="Design patterns identified, separated by |")
    recommendations: str = dspy.OutputField(desc="Architectural recommendations, separated by |")
    score: float = dspy.OutputField(desc="Architecture quality score 0-100")


class StyleReviewSignature(dspy.Signature):
    """Check code style and formatting.

    You are a STYLE GUIDE ENFORCER. Check for:
    1. Naming conventions
    2. Formatting consistency
    3. Documentation standards
    4. Code organization
    5. Idiomatic usage

    Ensure code style consistency.
    """
    code: str = dspy.InputField(desc="Code to check")
    language: str = dspy.InputField(desc="Programming language")
    style_guide: str = dspy.InputField(desc="Style guide to follow")

    violations: str = dspy.OutputField(desc="JSON list of style violations")
    formatting_issues: str = dspy.OutputField(desc="Formatting issues, separated by |")
    naming_issues: str = dspy.OutputField(desc="Naming convention issues, separated by |")
    compliance_score: float = dspy.OutputField(desc="Style compliance score 0-100")


class ReviewSynthesisSignature(dspy.Signature):
    """Synthesize all reviews into final report.

    You are a REVIEW SYNTHESIZER. Create report that:
    1. Prioritizes critical issues
    2. Groups related findings
    3. Provides clear next steps
    4. Balances thoroughness with actionability
    5. Makes clear approval decision

    Be decisive but fair.
    """
    code_review: str = dspy.InputField(desc="Code review findings")
    security_review: str = dspy.InputField(desc="Security findings")
    performance_review: str = dspy.InputField(desc="Performance findings")
    architecture_review: str = dspy.InputField(desc="Architecture findings")
    context: str = dspy.InputField(desc="PR/code context")

    status: str = dspy.OutputField(desc="APPROVED, CHANGES_REQUESTED, NEEDS_DISCUSSION, or BLOCKED")
    summary: str = dspy.OutputField(desc="Executive summary of the review")
    priority_actions: str = dspy.OutputField(desc="Top priority actions, separated by |")
    overall_score: float = dspy.OutputField(desc="Overall code quality score 0-100")


# =============================================================================
# AGENTS
# =============================================================================



class CodeReviewer(BaseSwarmAgent):
    """Reviews code quality."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=CodeReviewSignature)
        self._reviewer = dspy.ChainOfThought(CodeReviewSignature)
        self.learned_context = learned_context

    async def review(
        self,
        code: str,
        language: str,
        context: str,
        strictness: str = "moderate"
    ) -> Dict[str, Any]:
        """Review code."""
        try:
            result = self._reviewer(
                code=code,
                language=language,
                context=context + ("\n" + self.learned_context if self.learned_context else ""),
                strictness=strictness
            )

            try:
                issues = json.loads(result.issues)
            except Exception:
                issues = []

            positives = _split_field(result.positives)
            suggestions = _split_field(result.suggestions)

            self._broadcast("code_reviewed", {
                'issues_found': len(issues),
                'language': language
            })

            return {
                'issues': issues,
                'positives': positives,
                'suggestions': suggestions,
                'overall_assessment': str(result.overall_assessment)
            }

        except Exception as e:
            logger.error(f"Code review failed: {e}")
            return {'error': str(e)}


class SecurityScanner(BaseSwarmAgent):
    """Scans for security vulnerabilities."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=SecurityReviewSignature)
        self._scanner = dspy.ChainOfThought(SecurityReviewSignature)
        self.learned_context = learned_context

    async def scan(
        self,
        code: str,
        language: str,
        context: str
    ) -> Dict[str, Any]:
        """Scan for vulnerabilities."""
        try:
            result = self._scanner(
                code=code,
                language=language,
                context=context + ("\n" + self.learned_context if self.learned_context else "")
            )

            try:
                vulnerabilities = json.loads(result.vulnerabilities)
            except Exception:
                vulnerabilities = []

            recommendations = _split_field(result.recommendations)

            self._broadcast("security_scanned", {
                'vulnerabilities_found': len(vulnerabilities)
            })

            return {
                'vulnerabilities': vulnerabilities,
                'security_score': float(result.security_score) if result.security_score else 80.0,
                'recommendations': recommendations
            }

        except Exception as e:
            logger.error(f"Security scan failed: {e}")
            return {'error': str(e)}


class PerformanceAnalyzer(BaseSwarmAgent):
    """Analyzes performance issues."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=PerformanceReviewSignature)
        self._analyzer = dspy.ChainOfThought(PerformanceReviewSignature)
        self.learned_context = learned_context

    async def analyze(
        self,
        code: str,
        language: str,
        scale: str = "medium"
    ) -> Dict[str, Any]:
        """Analyze performance."""
        try:
            result = self._analyzer(
                code=code,
                language=language,
                scale=scale + ("\n" + self.learned_context if self.learned_context else "")
            )

            try:
                issues = json.loads(result.issues)
            except Exception:
                issues = []

            optimizations = _split_field(result.optimizations)

            self._broadcast("performance_analyzed", {
                'issues_found': len(issues)
            })

            return {
                'issues': issues,
                'complexity_analysis': str(result.complexity_analysis),
                'optimizations': optimizations
            }

        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return {'error': str(e)}


class ArchitectureReviewer(BaseSwarmAgent):
    """Reviews architecture."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=ArchitectureReviewSignature)
        self._reviewer = dspy.ChainOfThought(ArchitectureReviewSignature)
        self.learned_context = learned_context

    async def review(
        self,
        code: str,
        context: str,
        expected_patterns: List[str] = None
    ) -> Dict[str, Any]:
        """Review architecture."""
        try:
            result = self._reviewer(
                code=code,
                context=context + ("\n" + self.learned_context if self.learned_context else ""),
                patterns=",".join(expected_patterns) if expected_patterns else "general"
            )

            try:
                concerns = json.loads(result.concerns)
            except Exception:
                concerns = []

            patterns = _split_field(result.patterns_found)
            recommendations = _split_field(result.recommendations)

            self._broadcast("architecture_reviewed", {
                'concerns_found': len(concerns)
            })

            return {
                'concerns': concerns,
                'patterns_found': patterns,
                'recommendations': recommendations,
                'score': float(result.score) if result.score else 75.0
            }

        except Exception as e:
            logger.error(f"Architecture review failed: {e}")
            return {'error': str(e)}


class StyleChecker(BaseSwarmAgent):
    """Checks code style."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=StyleReviewSignature)
        self._checker = dspy.ChainOfThought(StyleReviewSignature)
        self.learned_context = learned_context

    async def check(
        self,
        code: str,
        language: str,
        style_guide: str = "pep8"
    ) -> Dict[str, Any]:
        """Check style."""
        try:
            result = self._checker(
                code=code,
                language=language,
                style_guide=style_guide + ("\n" + self.learned_context if self.learned_context else "")
            )

            try:
                violations = json.loads(result.violations)
            except Exception:
                violations = []

            formatting_issues = _split_field(result.formatting_issues)
            naming_issues = _split_field(result.naming_issues)

            self._broadcast("style_checked", {
                'violations_found': len(violations)
            })

            return {
                'violations': violations,
                'formatting_issues': formatting_issues,
                'naming_issues': naming_issues,
                'compliance_score': float(result.compliance_score) if result.compliance_score else 80.0
            }

        except Exception as e:
            logger.error(f"Style check failed: {e}")
            return {'error': str(e)}


class ReviewSynthesizer(BaseSwarmAgent):
    """Synthesizes all reviews."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=ReviewSynthesisSignature)
        self.learned_context = learned_context
        self._synthesizer = dspy.ChainOfThought(ReviewSynthesisSignature)

    async def synthesize(
        self,
        code_review: Dict[str, Any],
        security_review: Dict[str, Any],
        performance_review: Dict[str, Any],
        architecture_review: Dict[str, Any],
        context: str
    ) -> Dict[str, Any]:
        """Synthesize reviews."""
        try:
            result = self._synthesizer(
                code_review=json.dumps(code_review),
                security_review=json.dumps(security_review),
                performance_review=json.dumps(performance_review),
                architecture_review=json.dumps(architecture_review),
                context=context + ("\n" + self.learned_context if self.learned_context else "")
            )

            # Parse status
            status_str = str(result.status).upper().replace(' ', '_')
            status = ReviewStatus.NEEDS_DISCUSSION
            for s in ReviewStatus:
                if s.value.upper() == status_str:
                    status = s
                    break

            priority_actions = _split_field(result.priority_actions)

            return {
                'status': status,
                'summary': str(result.summary),
                'priority_actions': priority_actions,
                'overall_score': float(result.overall_score) if result.overall_score else 70.0
            }

        except Exception as e:
            logger.error(f"Review synthesis failed: {e}")
            return {
                'status': ReviewStatus.NEEDS_DISCUSSION,
                'summary': str(e),
                'priority_actions': [],
                'overall_score': 0
            }


# =============================================================================
# REVIEW SWARM
# =============================================================================

@register_swarm("review")
class ReviewSwarm(DomainSwarm):
    """
    World-Class Review Swarm.

    Provides comprehensive code review with:
    - Code quality analysis
    - Security scanning
    - Performance analysis
    - Architecture review
    - Style checking
    """

    AGENT_TEAM = AgentTeam.define(
        (CodeReviewer, "CodeReviewer", "_code_reviewer"),
        (SecurityScanner, "SecurityScanner", "_security_scanner"),
        (PerformanceAnalyzer, "PerformanceAnalyzer", "_performance_analyzer"),
        (ArchitectureReviewer, "ArchitectureReviewer", "_architecture_reviewer"),
        (StyleChecker, "StyleChecker", "_style_checker"),
        (ReviewSynthesizer, "ReviewSynthesizer", "_synthesizer"),
    )
    SWARM_SIGNATURE = ReviewSwarmSignature

    def __init__(self, config: ReviewConfig = None):
        super().__init__(config or ReviewConfig())

    async def review(
        self,
        code: str,
        context: str = "",
        language: str = None,
        **kwargs
    ) -> ReviewResult:
        """
        Perform comprehensive code review.

        Convenience method that delegates to execute().

        Args:
            code: Code to review
            context: Context about the code/PR
            language: Programming language

        Returns:
            ReviewResult with all findings
        """
        return await self.execute(code, context=context, language=language, **kwargs)

    async def _execute_domain(
        self,
        code: str,
        context: str = "",
        language: str = None,
        **kwargs
    ) -> ReviewResult:
        """
        Perform comprehensive code review.

        Delegates to _safe_execute_domain which handles try/except,
        timing, and post-execution learning automatically via PhaseExecutor.

        Args:
            code: Code to review
            context: Context about the code/PR
            language: Programming language

        Returns:
            ReviewResult with all findings
        """
        config = self.config
        lang = language or config.language

        logger.info(f"ReviewSwarm starting: {lang}")

        return await self._safe_execute_domain(
            task_type='code_review',
            default_tools=['code_review', 'security_review', 'performance_review'],
            result_class=ReviewResult,
            execute_fn=lambda executor: self._execute_phases(
                executor, code, context, lang, config
            ),
            output_data_fn=lambda result: self._build_output_data(result),
            input_data_fn=lambda: {
                'language': lang,
                'context': context,
                'code_length': len(code),
                'strictness': config.strictness,
                'style_guide': config.style_guide,
                'review_types': [rt.value for rt in config.review_types],
            },
        )

    async def _execute_phases(self, executor, code, context, lang, config):
        """Execute all review phases using PhaseExecutor.

        Args:
            executor: PhaseExecutor instance (provided by _safe_execute_domain)
            code: Code to review
            context: Context about the code/PR
            lang: Programming language
            config: ReviewConfig instance

        Returns:
            ReviewResult with all findings
        """
        # =================================================================
        # PHASE 1: PARALLEL REVIEWS
        # =================================================================
        parallel_results = await executor.run_parallel(1, "Parallel Reviews", [
            (
                "CodeReviewer", AgentRole.REVIEWER,
                self._code_reviewer.review(code, lang, context or "Code review", config.strictness),
                ['code_review'],
            ),
            (
                "SecurityScanner", AgentRole.EXPERT,
                self._security_scanner.scan(code, lang, context),
                ['security_scan'],
            ),
            (
                "PerformanceAnalyzer", AgentRole.EXPERT,
                self._performance_analyzer.analyze(code, lang),
                ['performance_analyze'],
            ),
            (
                "ArchitectureReviewer", AgentRole.EXPERT,
                self._architecture_reviewer.review(code, context),
                ['arch_review'],
            ),
        ])

        code_result, security_result, performance_result, architecture_result = parallel_results

        # =================================================================
        # PHASE 2: STYLE CHECK
        # =================================================================
        style_result = await executor.run_phase(
            2, "Style Check", "StyleChecker", AgentRole.REVIEWER,
            self._style_checker.check(code, lang, config.style_guide),
            input_data={'language': lang},
            tools_used=['style_check'],
        )

        # =================================================================
        # PHASE 3: SYNTHESIS
        # =================================================================
        synthesis = await executor.run_phase(
            3, "Review Synthesis", "ReviewSynthesizer", AgentRole.ORCHESTRATOR,
            self._synthesizer.synthesize(
                code_result, security_result, performance_result,
                architecture_result, context
            ),
            input_data={'reviews_count': 4},
            tools_used=['review_synthesize'],
        )

        # =================================================================
        # BUILD RESULT
        # =================================================================
        return self._build_review_result(
            executor, code_result, security_result,
            performance_result, synthesis, lang
        )

    def _build_review_result(
        self, executor, code_result, security_result,
        performance_result, synthesis, lang
    ):
        """Build the final ReviewResult from all phase outputs.

        Args:
            executor: PhaseExecutor (used for elapsed time)
            code_result: Output from CodeReviewer
            security_result: Output from SecurityScanner
            performance_result: Output from PerformanceAnalyzer
            synthesis: Output from ReviewSynthesizer
            lang: Programming language

        Returns:
            ReviewResult with all findings aggregated
        """
        # Convert issues to ReviewComment objects
        comments = []
        for issue in code_result.get('issues', []):
            severity = self._parse_severity(issue.get('severity', 'medium'))
            comments.append(ReviewComment(
                line=issue.get('line', 0),
                severity=severity,
                category="code",
                message=issue.get('message', ''),
                suggestion=issue.get('suggestion', '')
            ))

        # Convert vulnerabilities to SecurityFinding objects
        security_findings = []
        for vuln in security_result.get('vulnerabilities', []):
            severity = self._parse_severity(vuln.get('severity', 'medium'))
            security_findings.append(SecurityFinding(
                vulnerability_type=vuln.get('type', 'unknown'),
                severity=severity,
                location=vuln.get('location', ''),
                description=vuln.get('description', ''),
                fix_recommendation=vuln.get('fix', '')
            ))

        # Convert performance issues
        performance_findings = []
        for issue in performance_result.get('issues', []):
            severity = self._parse_severity(issue.get('severity', 'medium'))
            performance_findings.append(PerformanceFinding(
                issue_type=issue.get('type', 'unknown'),
                severity=severity,
                location=issue.get('location', ''),
                description=issue.get('description', ''),
                impact=issue.get('impact', ''),
                optimization=issue.get('optimization', '')
            ))

        # Count by severity
        all_findings = comments + security_findings + performance_findings
        critical_count = sum(1 for f in all_findings if hasattr(f, 'severity') and f.severity == Severity.CRITICAL)
        high_count = sum(1 for f in all_findings if hasattr(f, 'severity') and f.severity == Severity.HIGH)
        medium_count = sum(1 for f in all_findings if hasattr(f, 'severity') and f.severity == Severity.MEDIUM)
        low_count = sum(1 for f in all_findings if hasattr(f, 'severity') and f.severity == Severity.LOW)

        status_str = synthesis['status'].value if hasattr(synthesis['status'], 'value') else str(synthesis['status'])
        logger.info(f"ReviewSwarm complete: {status_str}, Score: {synthesis['overall_score']:.1f}")

        return ReviewResult(
            success=True,
            swarm_name=self.config.name,
            domain=self.config.domain,
            output={
                'summary': synthesis['summary'],
                'score': synthesis['overall_score'],
                'approved': synthesis['status'] == ReviewStatus.APPROVED,
                'status': status_str,
                'findings_count': len(all_findings),
                'language': lang,
            },
            execution_time=executor.elapsed(),
            status=synthesis['status'],
            comments=comments,
            security_findings=security_findings,
            performance_findings=performance_findings,
            architecture_findings=[],
            summary=synthesis['summary'],
            score=synthesis['overall_score'],
            critical_count=critical_count,
            high_count=high_count,
            medium_count=medium_count,
            low_count=low_count,
            approved=synthesis['status'] == ReviewStatus.APPROVED
        )

    @staticmethod
    def _parse_severity(severity_str: str) -> Severity:
        """Parse a severity string into a Severity enum value.

        Args:
            severity_str: Severity string (e.g. 'medium', 'HIGH')

        Returns:
            Severity enum value, defaulting to MEDIUM on parse failure
        """
        try:
            return Severity(severity_str.lower())
        except (ValueError, AttributeError):
            return Severity.MEDIUM

    @staticmethod
    def _build_output_data(result: 'ReviewResult') -> Dict[str, Any]:
        """Build output data dict for post-execution learning.

        Args:
            result: The completed ReviewResult

        Returns:
            Dict with key metrics for learning
        """
        status_str = result.status.value if hasattr(result.status, 'value') else str(result.status)
        return {
            'status': status_str,
            'overall_score': result.score,
            'comments_count': len(result.comments),
            'security_findings_count': len(result.security_findings),
            'performance_findings_count': len(result.performance_findings),
            'critical_count': result.critical_count,
            'high_count': result.high_count,
            'medium_count': result.medium_count,
            'low_count': result.low_count,
            'approved': result.approved,
            'execution_time': result.execution_time,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def review_code(code: str, **kwargs) -> ReviewResult:
    """
    One-liner code review.

    Usage:
        from core.swarms.review_swarm import review_code
        result = await review_code(my_code, language="python")
    """
    swarm = ReviewSwarm()
    return await swarm.review(code, **kwargs)


def review_code_sync(code: str, **kwargs) -> ReviewResult:
    """Synchronous code review."""
    return asyncio.run(review_code(code, **kwargs))


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'ReviewSwarm',
    'ReviewConfig',
    'ReviewResult',
    'ReviewComment',
    'SecurityFinding',
    'PerformanceFinding',
    'ArchitectureFinding',
    'ReviewType',
    'Severity',
    'ReviewStatus',
    'review_code',
    'review_code_sync',
    # Agents
    'CodeReviewer',
    'SecurityScanner',
    'PerformanceAnalyzer',
    'ArchitectureReviewer',
    'StyleChecker',
    'ReviewSynthesizer',
]
