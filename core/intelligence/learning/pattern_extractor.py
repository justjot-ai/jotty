"""
Pattern Extractor - Behavioral pattern extraction from episode data.

Extracts success strategies, quality drivers, speed patterns, causal patterns,
failure avoidance patterns, and cross-domain transfer patterns from accumulated
learning episodes.

Extracted from learning_service.py for modularity.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

from .learning_store import EpisodeRecord, LearningStore, PatternRecord

logger = logging.getLogger(__name__)


class PatternExtractor:
    """
    Extracts behavioral patterns from accumulated episode data.

    Takes a LearningStore instance and min_episodes threshold.
    Maintains a tautology cache for filtering obvious patterns.
    """

    def __init__(self, store: LearningStore, min_episodes: int = 2) -> None:
        self._store = store
        self._min_episodes = min_episodes
        self._tautology_cache: Dict[str, bool] = {}

    def is_tautological_pattern(self, domain: str, recommendation: str) -> bool:
        """Check single pattern against cache, falling back to heuristic."""
        cache_key = f"{domain}:{recommendation[:100]}"
        if cache_key in self._tautology_cache:
            return self._tautology_cache[cache_key]
        is_taut = self.heuristic_tautology_check(domain, recommendation)
        self._tautology_cache[cache_key] = is_taut
        return is_taut

    def batch_tautology_filter(self, candidates: List[Tuple[str, str, str]]) -> Dict[str, bool]:
        """
        Classify multiple patterns in one LLM call.

        Args:
            candidates: List of (pattern_id, domain, recommendation)

        Returns:
            Dict mapping pattern_id -> is_tautological
        """
        uncached = []
        results: Dict[str, bool] = {}
        for pid, domain, rec in candidates:
            cache_key = f"{domain}:{rec[:100]}"
            if cache_key in self._tautology_cache:
                results[pid] = self._tautology_cache[cache_key]
            else:
                uncached.append((pid, domain, rec))

        if not uncached:
            return results

        # Build a single batched prompt
        lines = []
        for i, (pid, domain, rec) in enumerate(uncached, 1):
            lines.append(f"{i}. [{domain}] {rec}")
        prompt = (
            "For each pattern below, answer YES if it is TAUTOLOGICAL "
            "(obvious domain convention any competent AI already knows, e.g. "
            "'for coding tasks: include code examples' or "
            "'for research tasks: cite sources'), NO if it teaches something non-obvious.\n\n"
            + "\n".join(lines)
            + "\n\nRespond with ONLY the numbers and yes/no, one per line. Example:\n1. yes\n2. no"
        )

        try:
            import anthropic

            client = anthropic.Anthropic()
            resp = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=len(uncached) * 10,
                messages=[{"role": "user", "content": prompt}],
            )
            answer_text = getattr(resp.content[0], "text", "").strip().lower()

            for i, (pid, domain, rec) in enumerate(uncached, 1):
                is_taut = f"{i}. yes" in answer_text or f"{i}.yes" in answer_text
                results[pid] = is_taut
                self._tautology_cache[f"{domain}:{rec[:100]}"] = is_taut
                if is_taut:
                    logger.debug(f"Tautological: {rec[:80]}")
        except Exception as e:
            logger.debug(f"Batch tautology LLM failed, using heuristic: {e}")
            for pid, domain, rec in uncached:
                is_taut = self.heuristic_tautology_check(domain, rec)
                results[pid] = is_taut
                self._tautology_cache[f"{domain}:{rec[:100]}"] = is_taut

        return results

    def purge_stale_tautological_patterns(self) -> int:
        """
        Scan all existing patterns in DB and delete any that are tautological.
        Returns count of purged patterns.
        """
        all_patterns = self._store.get_patterns(limit=500)
        if not all_patterns:
            return 0

        candidates = [
            (p.pattern_id, p.source_domain, p.recommendation)
            for p in all_patterns
            if p.pattern_type in ("quality_driver", "causal")
        ]
        if not candidates:
            return 0

        verdicts = self.batch_tautology_filter(candidates)
        to_delete = [pid for pid, is_taut in verdicts.items() if is_taut]
        if to_delete:
            deleted = self._store.delete_patterns(to_delete)
            logger.info(f"Purged {deleted} tautological patterns from DB")
            return deleted
        return 0

    @staticmethod
    def heuristic_tautology_check(domain: str, recommendation: str) -> bool:
        """Fast fallback when LLM is unavailable."""
        rec_lower = recommendation.lower()
        obvious = {
            "coding": [
                "include code",
                "code examples",
                "use headings",
                "section heading",
                "mathematical formul",
                "include math",
            ],
            "research": ["cite sources", "cite papers", "use headings", "references"],
            "system_design": ["use headings", "include code", "use diagrams", "code examples"],
            "economics": ["cite sources", "include math", "mathematical formul"],
            "data_science": [
                "include code",
                "code examples",
                "include math",
                "mathematical formul",
            ],
        }
        for phrase in obvious.get(domain, []):
            if phrase in rec_lower:
                return True
        return False

    def extract_patterns(self, domain: str) -> None:
        """
        Auto-extract behavioral patterns from accumulated episode data.

        Extracts:
        1. Success strategies (what actions/approaches lead to high quality)
        2. Quality drivers (structural features that correlate with quality)
        3. Speed patterns (what's fast vs slow)
        4. Domain-specific insights (coding needs code, research needs citations)
        5. Failure avoidance patterns

        Also purges stale tautological patterns from prior runs.
        """
        episodes = self._store.query_episodes(domain=domain, limit=100)
        if len(episodes) < self._min_episodes:
            return

        self.purge_stale_tautological_patterns()

        successes = [e for e in episodes if e.success and e.quality >= 0.7]
        failures = [e for e in episodes if not e.success]
        high_quality = [e for e in episodes if e.quality >= 0.70]
        low_quality = [e for e in episodes if e.success and e.quality < 0.55]

        # 1. Success strategy patterns -- extract WHAT made responses successful.
        #    Two sources: (a) structural analysis keys if present, (b) tool/action
        #    patterns from raw data that any caller can provide.
        if len(successes) >= 2:
            self._extract_success_strategies(domain, successes, failures)

        # 2. Quality driver patterns -- only from LLM-judged episodes to avoid
        #    Goodhart's Law (heuristic rewards has_code -> pattern says "use code" -> circular).
        #    Tautological patterns are filtered by is_tautological_pattern() which
        #    uses LLM classification instead of hardcoded skip lists.
        llm_judged = [e for e in high_quality if (e.outcome or {}).get("llm_judged")]
        quality_source = llm_judged if len(llm_judged) >= 2 else high_quality
        if len(quality_source) >= 2:
            self._extract_quality_drivers(domain, quality_source)

        # 3. Speed patterns -- what's efficient
        if len(successes) >= 3:
            self._extract_speed_patterns(domain, successes)

        # 4. Quality contrast -- what distinguishes high from low quality
        if len(high_quality) >= 2 and len(low_quality) >= 2:
            self._extract_quality_contrast(domain, high_quality, low_quality)

        # 5. Failure avoidance patterns (original, kept)
        if len(failures) >= 3:
            self._extract_failure_patterns(domain, failures)

        # 6. CAUSAL patterns -- A/B comparison of feature presence vs absence
        if len(episodes) >= 3:
            self.extract_causal_patterns(domain, episodes)

        # 7. Cross-domain transfer patterns
        self.extract_transfer_patterns(domain, episodes)

        logger.debug(
            f"Pattern extraction complete for domain={domain}: "
            f"{len(successes)} successes, {len(high_quality)} high-quality, "
            f"{len(failures)} failures"
        )

    def _extract_success_strategies(
        self,
        domain: str,
        successes: List[EpisodeRecord],
        failures: List[EpisodeRecord],
    ) -> None:
        """Extract success strategy patterns from successful episodes."""
        # Skip infrastructure keys -- these don't help the model produce better output
        _INFRA_KEYS = {
            "model",
            "provider",
            "mode",
            "domain",
            "task_type",
            "exploration",
            "exploration_reason",
            "streamed",
            "retries",
            "strategy",
            "temperature",
            "paradigm",
            "key",
            "action",
        }

        # Extract structural success patterns from outcomes (if present)
        struct_counts: Dict[str, int] = defaultdict(int)
        total_words = 0
        total_code_blocks = 0
        total_assertions = 0
        for e in successes:
            out = e.outcome or {}
            if out.get("has_code"):
                struct_counts["include code implementations"] += 1
                total_code_blocks += out.get("code_block_count", 1)
            if out.get("has_class"):
                struct_counts["define proper classes with methods"] += 1
            if out.get("has_tests"):
                struct_counts["include test functions (def test_*)"] += 1
            if out.get("assertion_count", 0) >= 3:
                struct_counts["include 3+ assertions for verification"] += 1
                total_assertions += out.get("assertion_count", 0)
            if out.get("has_headings"):
                struct_counts["use section headings to organize"] += 1
            if out.get("has_math") and domain not in (
                "coding",
                "algorithms",
                "compiler_design",
            ):
                # Skip has_math for code-heavy domains -- O(n) notation triggers it
                # but "include mathematical formulations" is not actionable advice
                # for programmers.  Keep it for research/economics/data_science.
                struct_counts["include mathematical formulations"] += 1
            if out.get("has_citations"):
                struct_counts["cite sources and references"] += 1
            if out.get("has_table"):
                struct_counts["use tables for comparisons"] += 1
            if out.get("has_conclusion"):
                struct_counts["include summary/conclusion"] += 1
            if out.get("reasoning_density", 0) >= 1.0:
                struct_counts["explain reasoning (because/therefore/since)"] += 1
            if out.get("example_count", 0) >= 2:
                struct_counts["provide concrete examples"] += 1
            wc = out.get("word_count", 0)
            if wc > 0:
                total_words += wc
            gc = out.get("goal_coverage", 0)
            if gc > 0.7:
                struct_counts["address all key aspects of the task"] += 1

        # Extract tool usage patterns (works with raw action data)
        tool_success: Dict[str, int] = defaultdict(int)
        for e in successes:
            tools = e.action.get("tools", [])
            if isinstance(tools, list):
                for tool in tools:
                    tool_success[str(tool)] += 1
                # Tool combination patterns (pairs)
                if len(tools) >= 2:
                    combo = "+".join(sorted(str(t) for t in tools[:3]))
                    struct_counts[f"use tool combination [{combo}]"] += 1

        # Find tools that appear in successes but rarely in failures
        tool_failure: Dict[str, int] = defaultdict(int)
        for e in failures:
            tools = e.action.get("tools", [])
            if isinstance(tools, list):
                for tool in tools:
                    tool_failure[str(tool)] += 1

        for tool, succ_count in tool_success.items():
            fail_count = tool_failure.get(tool, 0)
            total = succ_count + fail_count
            if total >= 3 and succ_count / total >= 0.7:
                struct_counts[f"use {tool} tool (succeeds {succ_count}/{total} times)"] += 1

        # Extract approach patterns from non-infra action keys
        for e in successes:
            for key, val in e.action.items():
                if key in _INFRA_KEYS:
                    continue
                if isinstance(val, (str, int, float, bool)):
                    struct_counts[f"use {key}={val} approach"] += 1

        for strategy, count in struct_counts.items():
            if count >= 2:
                confidence = count / len(successes)
                pattern_id = hashlib.md5(f"success_{domain}_{strategy}".encode()).hexdigest()[:12]

                self._store.save_pattern(
                    PatternRecord(
                        pattern_id=pattern_id,
                        source_domain=domain,
                        pattern_type="success_strategy",
                        description=(
                            f"Successful {domain} responses: {strategy} "
                            f"({count}/{len(successes)} episodes)"
                        ),
                        conditions={"domain": domain},
                        recommendation=f"For {domain} tasks: {strategy}",
                        confidence=confidence,
                        evidence_count=count,
                        applicable_domains=[domain],
                    )
                )

        # Add aggregate depth pattern
        if total_words > 0 and len(successes) >= 2:
            avg_words = total_words // len(successes)
            avg_code = total_code_blocks // max(len(successes), 1)
            avg_asserts = total_assertions // max(len(successes), 1)
            depth_id = hashlib.md5(f"depth_{domain}".encode()).hexdigest()[:12]
            depth_desc = f"aim for ~{avg_words} words"
            if avg_code > 0:
                depth_desc += f" with {avg_code}+ code blocks"
            if avg_asserts > 0:
                depth_desc += f" and {avg_asserts}+ assertions"
            self._store.save_pattern(
                PatternRecord(
                    pattern_id=depth_id,
                    source_domain=domain,
                    pattern_type="success_strategy",
                    description=(
                        f"Successful {domain} responses average {avg_words} words"
                        + (f" and {avg_code} code blocks" if avg_code > 0 else "")
                    ),
                    conditions={"domain": domain},
                    recommendation=f"For {domain} tasks: {depth_desc}",
                    confidence=0.7,
                    evidence_count=len(successes),
                    applicable_domains=[domain],
                )
            )

    def _extract_quality_drivers(self, domain: str, quality_source: List[EpisodeRecord]) -> None:
        """Extract quality driver patterns from high-quality episodes."""
        quality_signals: Dict[str, int] = defaultdict(int)
        for e in quality_source:
            out = e.outcome or {}
            if out.get("has_code"):
                quality_signals["include_code"] += 1
            if out.get("has_citations"):
                quality_signals["cite_sources"] += 1
            if out.get("has_math"):
                quality_signals["include_math"] += 1
            if out.get("has_table"):
                quality_signals["use_tables"] += 1
            gc = out.get("goal_coverage", 0)
            if gc > 0.7:
                quality_signals["high_goal_coverage"] += 1

        readable_names = {
            "include_code": "include code examples with implementations",
            "cite_sources": "cite sources, papers, and references",
            "include_math": "include mathematical formulations where relevant",
            "use_tables": "use tables for data comparisons",
            "high_goal_coverage": "address all key aspects of the task",
        }

        for signal, count in quality_signals.items():
            if count >= 2:
                desc = readable_names.get(signal, signal)
                recommendation = f"For {domain} tasks: {desc}"
                if self.is_tautological_pattern(domain, recommendation):
                    continue
                confidence = count / len(quality_source)
                pattern_id = hashlib.md5(f"quality_{domain}_{signal}".encode()).hexdigest()[:12]

                self._store.save_pattern(
                    PatternRecord(
                        pattern_id=pattern_id,
                        source_domain=domain,
                        pattern_type="quality_driver",
                        description=(
                            f"LLM-judged high-quality {domain} responses tend to {desc} "
                            f"({count}/{len(quality_source)} top episodes)"
                        ),
                        conditions={"domain": domain},
                        recommendation=recommendation,
                        confidence=confidence,
                        evidence_count=count,
                        applicable_domains=[domain, "general"],
                    )
                )

    def _extract_speed_patterns(self, domain: str, successes: List[EpisodeRecord]) -> None:
        """Extract speed optimization patterns."""
        times = [e.execution_time for e in successes if e.execution_time > 0]
        if not times:
            return
        avg_time = sum(times) / len(times)
        fast = [e for e in successes if 0 < e.execution_time < avg_time * 0.7]
        if len(fast) < 2:
            return
        fast_actions: Dict[str, int] = defaultdict(int)
        for e in fast:
            for k, v in e.action.items():
                if isinstance(v, (str, int, float, bool)):
                    fast_actions[f"{k}={v}"] += 1
        for action_str, count in fast_actions.items():
            if count >= 2:
                pattern_id = hashlib.md5(f"speed_{domain}_{action_str}".encode()).hexdigest()[:12]
                self._store.save_pattern(
                    PatternRecord(
                        pattern_id=pattern_id,
                        source_domain=domain,
                        pattern_type="speed_optimization",
                        description=(
                            f"In {domain}, {action_str} tends to be faster "
                            f"(avg {avg_time:.0f}s, fast episodes <{avg_time*0.7:.0f}s)"
                        ),
                        conditions={"domain": domain},
                        recommendation=(f"For faster {domain} execution, prefer {action_str}"),
                        confidence=count / len(fast),
                        evidence_count=count,
                        applicable_domains=[domain],
                    )
                )

    def _extract_quality_contrast(
        self,
        domain: str,
        high_quality: List[EpisodeRecord],
        low_quality: List[EpisodeRecord],
    ) -> None:
        """Extract patterns that distinguish high from low quality."""
        high_features: Dict[str, float] = defaultdict(float)
        low_features: Dict[str, float] = defaultdict(float)
        for e in high_quality:
            out = e.outcome or {}
            for k in ["word_count", "structure_score", "goal_coverage", "code_block_count"]:
                high_features[k] += float(out.get(k, 0))
        for e in low_quality:
            out = e.outcome or {}
            for k in ["word_count", "structure_score", "goal_coverage", "code_block_count"]:
                low_features[k] += float(out.get(k, 0))

        for k in high_features:
            h_avg = high_features[k] / len(high_quality)
            l_avg = low_features[k] / len(low_quality)
            if h_avg > l_avg * 1.5 and h_avg > 0:
                pattern_id = hashlib.md5(f"contrast_{domain}_{k}".encode()).hexdigest()[:12]
                self._store.save_pattern(
                    PatternRecord(
                        pattern_id=pattern_id,
                        source_domain=domain,
                        pattern_type="quality_contrast",
                        description=(
                            f"In {domain}, high-quality responses have {h_avg:.0f} avg {k} "
                            f"vs {l_avg:.0f} in low-quality -- {h_avg/max(l_avg,1):.1f}x difference"
                        ),
                        conditions={"domain": domain},
                        recommendation=(
                            f"For {domain} tasks, aim for higher {k} " f"(target: {h_avg:.0f}+)"
                        ),
                        confidence=min(0.9, 0.5 + (h_avg - l_avg) / max(h_avg, 1) * 0.4),
                        evidence_count=len(high_quality) + len(low_quality),
                        applicable_domains=[domain],
                    )
                )

    def _extract_failure_patterns(self, domain: str, failures: List[EpisodeRecord]) -> None:
        """Extract failure avoidance patterns."""
        error_counts: Dict[str, int] = defaultdict(int)
        for e in failures:
            if e.error_type:
                error_counts[e.error_type] += 1

        for error_type, count in error_counts.items():
            if count >= 2:
                pattern_id = hashlib.md5(f"failure_{domain}_{error_type}".encode()).hexdigest()[:12]

                self._store.save_pattern(
                    PatternRecord(
                        pattern_id=pattern_id,
                        source_domain=domain,
                        pattern_type="failure_avoidance",
                        description=(
                            f"In {domain}, {error_type} errors occur frequently ({count} times)"
                        ),
                        conditions={"domain": domain, "error_type": error_type},
                        recommendation=f"Add error handling for {error_type} in {domain} tasks",
                        confidence=min(0.9, count / len(failures)),
                        evidence_count=count,
                        applicable_domains=[domain],
                    )
                )

    def extract_causal_patterns(self, domain: str, episodes: List[EpisodeRecord]) -> None:
        """
        Causal analysis: compare episodes WITH a feature vs WITHOUT.
        Only records patterns where the causal effect is statistically meaningful.
        Skips tautological features (code in coding, citations in research).
        """
        features_to_test = [
            ("has_code", "including code examples"),
            ("has_headings", "using section headings"),
            ("has_citations", "citing sources"),
            ("has_table", "using tables"),
            ("has_numbered_list", "using numbered lists"),
            ("has_math", "including math formulations"),
        ]

        successful = [e for e in episodes if e.success and e.quality > 0]
        if len(successful) < 3:
            return

        for feature_key, feature_desc in features_to_test:
            with_feature = [e for e in successful if (e.outcome or {}).get(feature_key)]
            without_feature = [e for e in successful if not (e.outcome or {}).get(feature_key)]

            if len(with_feature) < 1 or len(without_feature) < 1:
                continue

            avg_with = sum(e.quality for e in with_feature) / len(with_feature)
            avg_without = sum(e.quality for e in without_feature) / len(without_feature)
            delta = avg_with - avg_without

            min_evidence = len(with_feature) + len(without_feature)
            # Require at least 3 per group and 8 total for statistical meaning
            if len(with_feature) < 3 or len(without_feature) < 3 or min_evidence < 8:
                continue
            if delta < 0.05:
                # Only save positive "use X" patterns -- "avoid X" patterns are
                # almost always confounded by output length (shorter outputs lack
                # features AND sometimes score differently, creating a spurious
                # negative correlation).
                continue

            recommendation = (
                f"For {domain} tasks: {feature_desc} "
                f"(quality boost: +{delta*100:.1f}% across {min_evidence} episodes)"
            )
            if self.is_tautological_pattern(domain, recommendation):
                continue

            pattern_id = hashlib.md5(
                f"causal_{domain}_{feature_key}_improves".encode()
            ).hexdigest()[:12]

            self._store.save_pattern(
                PatternRecord(
                    pattern_id=pattern_id,
                    source_domain=domain,
                    pattern_type="causal",
                    description=(
                        f"In {domain}, {feature_desc} improves quality by "
                        f"{delta*100:.1f}% "
                        f"(with={avg_with:.3f} [{len(with_feature)} eps] "
                        f"vs without={avg_without:.3f} [{len(without_feature)} eps])"
                    ),
                    conditions={
                        "domain": domain,
                        "feature": feature_key,
                        "direction": "improves",
                        "delta": round(delta, 4),
                    },
                    recommendation=recommendation,
                    confidence=min(0.95, 0.4 + min(len(with_feature), len(without_feature)) * 0.1),
                    evidence_count=len(with_feature) + len(without_feature),
                    applicable_domains=[domain],
                )
            )

    def extract_transfer_patterns(self, domain: str, domain_episodes: List[EpisodeRecord]) -> None:
        """
        Cross-domain transfer: if a feature helps in domain A, suggest it for domain B.
        'Structured headings improved quality 40% in coding -> apply to economics too.'
        """
        # Get causal patterns from THIS domain
        patterns = self._store.get_patterns(domain=domain)
        causal = [p for p in patterns if p.pattern_type == "causal"]

        if not causal:
            return

        # Get all OTHER domains
        conn = self._store._get_conn()
        rows = conn.execute(
            "SELECT DISTINCT domain FROM episodes WHERE domain != ? AND domain != ''",
            (domain,),
        ).fetchall()
        other_domains = [r["domain"] for r in rows]

        for pattern in causal:
            delta = pattern.conditions.get("delta", 0)
            feature = pattern.conditions.get("feature", "")
            direction = pattern.conditions.get("direction", "")

            if abs(delta) < 0.05 or direction != "improves":
                continue

            for other_domain in other_domains:
                # Check if this feature is already tested in the other domain
                other_patterns = self._store.get_patterns(domain=other_domain)
                already_tested = any(
                    p.pattern_type == "causal" and p.conditions.get("feature") == feature
                    for p in other_patterns
                )
                if already_tested:
                    continue

                pattern_id = hashlib.md5(
                    f"transfer_{domain}_{other_domain}_{feature}".encode()
                ).hexdigest()[:12]

                feature_desc = {
                    "has_code": "including code examples",
                    "has_headings": "using section headings",
                    "has_citations": "citing sources",
                    "has_table": "using tables",
                    "has_numbered_list": "using numbered lists",
                    "has_math": "including math formulations",
                }.get(feature, feature)

                self._store.save_pattern(
                    PatternRecord(
                        pattern_id=pattern_id,
                        source_domain=domain,
                        pattern_type="cross_domain_transfer",
                        description=(
                            f"In {domain}, {feature_desc} improved quality by "
                            f"{delta*100:.1f}%. Transfer hypothesis: apply to {other_domain}."
                        ),
                        conditions={
                            "source_domain": domain,
                            "target_domain": other_domain,
                            "feature": feature,
                            "source_delta": round(delta, 4),
                        },
                        recommendation=(
                            f"Try {feature_desc} in {other_domain} tasks -- "
                            f"it improved {domain} quality by {delta*100:.1f}%"
                        ),
                        confidence=min(0.6, 0.3 + abs(delta)),
                        evidence_count=pattern.evidence_count,
                        applicable_domains=[other_domain, domain],
                    )
                )
