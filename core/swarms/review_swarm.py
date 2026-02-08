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
from datetime import datetime
from pathlib import Path
from enum import Enum

from .base_swarm import (
    BaseSwarm, SwarmConfig, SwarmResult, AgentRole,
    register_swarm, ExecutionTrace
)
from .base import DomainSwarm, AgentTeam
from ..agents.base import DomainAgent, DomainAgentConfig

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
class ReviewConfig(SwarmConfig):
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

class BaseReviewAgent(DomainAgent):
    """Base class for review agents. Inherits from DomainAgent for unified infrastructure."""

    def __init__(self, memory=None, context=None, bus=None, signature=None):
        config = DomainAgentConfig(
            name=self.__class__.__name__,
            enable_memory=memory is not None,
            enable_context=context is not None,
        )
        super().__init__(signature=signature, config=config)

        # Ensure LM is configured before child classes create DSPy modules
        self._ensure_initialized()

        if memory is not None:
            self._memory = memory
        if context is not None:
            self._context_manager = context
        self.bus = bus

    def _broadcast(self, event: str, data: Dict[str, Any]):
        """Broadcast event to other agents."""
        if self.bus:
            try:
                from ..agents.axon import Message
                msg = Message(
                    sender=self.__class__.__name__,
                    receiver="broadcast",
                    content={'event': event, **data}
                )
                self.bus.publish(msg)
            except Exception:
                pass


class CodeReviewer(BaseReviewAgent):
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
            except:
                issues = []

            positives = [p.strip() for p in str(result.positives).split('|') if p.strip()]
            suggestions = [s.strip() for s in str(result.suggestions).split('|') if s.strip()]

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


class SecurityScanner(BaseReviewAgent):
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
            except:
                vulnerabilities = []

            recommendations = [r.strip() for r in str(result.recommendations).split('|') if r.strip()]

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


class PerformanceAnalyzer(BaseReviewAgent):
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
            except:
                issues = []

            optimizations = [o.strip() for o in str(result.optimizations).split('|') if o.strip()]

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


class ArchitectureReviewer(BaseReviewAgent):
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
            except:
                concerns = []

            patterns = [p.strip() for p in str(result.patterns_found).split('|') if p.strip()]
            recommendations = [r.strip() for r in str(result.recommendations).split('|') if r.strip()]

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


class StyleChecker(BaseReviewAgent):
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
            except:
                violations = []

            formatting_issues = [f.strip() for f in str(result.formatting_issues).split('|') if f.strip()]
            naming_issues = [n.strip() for n in str(result.naming_issues).split('|') if n.strip()]

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


class ReviewSynthesizer(BaseReviewAgent):
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

            priority_actions = [p.strip() for p in str(result.priority_actions).split('|') if p.strip()]

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

        Args:
            code: Code to review
            context: Context about the code/PR
            language: Programming language

        Returns:
            ReviewResult with all findings
        """
        start_time = datetime.now()

        config = self.config
        lang = language or config.language

        logger.info(f"🔍 ReviewSwarm starting: {lang}")

        try:
            # =================================================================
            # PHASE 1: PARALLEL REVIEWS
            # =================================================================
            logger.info("📋 Phase 1: Running parallel reviews...")

            code_task = self._code_reviewer.review(
                code, lang, context or "Code review", config.strictness
            )
            security_task = self._security_scanner.scan(code, lang, context)
            performance_task = self._performance_analyzer.analyze(code, lang)
            architecture_task = self._architecture_reviewer.review(code, context)

            results = await asyncio.gather(
                code_task, security_task, performance_task, architecture_task,
                return_exceptions=True
            )

            code_result = results[0] if not isinstance(results[0], Exception) else {'error': str(results[0])}
            security_result = results[1] if not isinstance(results[1], Exception) else {'error': str(results[1])}
            performance_result = results[2] if not isinstance(results[2], Exception) else {'error': str(results[2])}
            architecture_result = results[3] if not isinstance(results[3], Exception) else {'error': str(results[3])}

            self._trace_phase("CodeReviewer", AgentRole.REVIEWER,
                {'language': lang},
                {'issues': len(code_result.get('issues', []))},
                success='error' not in code_result, phase_start=start_time, tools_used=['code_review'])
            self._trace_phase("SecurityScanner", AgentRole.EXPERT,
                {'language': lang},
                {'vulnerabilities': len(security_result.get('vulnerabilities', []))},
                success='error' not in security_result, phase_start=start_time, tools_used=['security_scan'])
            self._trace_phase("PerformanceAnalyzer", AgentRole.EXPERT,
                {'language': lang},
                {'issues': len(performance_result.get('issues', []))},
                success='error' not in performance_result, phase_start=start_time, tools_used=['performance_analyze'])
            self._trace_phase("ArchitectureReviewer", AgentRole.EXPERT,
                {'has_context': bool(context)},
                {'has_result': 'error' not in architecture_result},
                success='error' not in architecture_result, phase_start=start_time, tools_used=['arch_review'])

            # =================================================================
            # PHASE 2: STYLE CHECK
            # =================================================================
            logger.info("🎨 Phase 2: Style checking...")

            style_result = await self._style_checker.check(code, lang, config.style_guide)

            phase2_start = datetime.now()
            self._trace_phase("StyleChecker", AgentRole.REVIEWER,
                {'language': lang},
                {'has_result': bool(style_result)},
                success=True, phase_start=start_time, tools_used=['style_check'])

            # =================================================================
            # PHASE 3: SYNTHESIS
            # =================================================================
            logger.info("📊 Phase 3: Synthesizing reviews...")

            synthesis = await self._synthesizer.synthesize(
                code_result, security_result, performance_result,
                architecture_result, context
            )

            self._trace_phase("ReviewSynthesizer", AgentRole.ORCHESTRATOR,
                {'reviews_count': 4},
                {'status': str(synthesis.get('status', '')), 'score': synthesis.get('overall_score', 0)},
                success=True, phase_start=phase2_start, tools_used=['review_synthesize'])

            # =================================================================
            # BUILD RESULT
            # =================================================================
            exec_time = (datetime.now() - start_time).total_seconds()

            # Convert issues to ReviewComment objects
            comments = []
            for issue in code_result.get('issues', []):
                severity_str = issue.get('severity', 'medium').lower()
                try:
                    severity = Severity(severity_str)
                except:
                    severity = Severity.MEDIUM

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
                severity_str = vuln.get('severity', 'medium').lower()
                try:
                    severity = Severity(severity_str)
                except:
                    severity = Severity.MEDIUM

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
                severity_str = issue.get('severity', 'medium').lower()
                try:
                    severity = Severity(severity_str)
                except:
                    severity = Severity.MEDIUM

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

            result = ReviewResult(
                success=True,
                swarm_name=self.config.name,
                domain=self.config.domain,
                output={'language': lang},
                execution_time=exec_time,
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

            status_str = synthesis['status'].value if hasattr(synthesis['status'], 'value') else str(synthesis['status'])
            logger.info(f"✅ ReviewSwarm complete: {status_str}, Score: {synthesis['overall_score']:.1f}")

            # Post-execution learning
            exec_time = (datetime.now() - start_time).total_seconds()
            await self._post_execute_learning(
                success=True,
                execution_time=exec_time,
                tools_used=self._get_active_tools(['code_review', 'security_review', 'performance_review']),
                task_type='code_review',
                output_data={
                    'status': status_str,
                    'overall_score': synthesis['overall_score'],
                    'comments_count': len(comments),
                    'security_findings_count': len(security_findings),
                    'performance_findings_count': len(performance_findings),
                    'critical_count': critical_count,
                    'high_count': high_count,
                    'medium_count': medium_count,
                    'low_count': low_count,
                    'approved': synthesis['status'] == ReviewStatus.APPROVED,
                    'execution_time': exec_time,
                },
                input_data={
                    'language': lang,
                    'context': context,
                    'code_length': len(code),
                    'strictness': config.strictness,
                    'style_guide': config.style_guide,
                    'review_types': [rt.value for rt in config.review_types],
                }
            )

            return result

        except Exception as e:
            logger.error(f"❌ ReviewSwarm error: {e}")
            import traceback
            traceback.print_exc()
            exec_time = (datetime.now() - start_time).total_seconds()
            await self._post_execute_learning(
                success=False,
                execution_time=exec_time,
                tools_used=self._get_active_tools(['code_review']),
                task_type='code_review'
            )
            return ReviewResult(
                success=False,
                swarm_name=self.config.name,
                domain=self.config.domain,
                output={},
                execution_time=(datetime.now() - start_time).total_seconds(),
                error=str(e)
            )


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
